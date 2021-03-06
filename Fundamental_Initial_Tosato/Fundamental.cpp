// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>


using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
                //cout<<"matches size"<<matches.size()<<endl;
            }
        }
    }
}

vector<Match> getS (vector<Match>& matches, int sampleNb){
    //cout<<"sampleNb: "<<sampleNb<<endl;
    const int matchSize = matches.size();
    if(sampleNb>matchSize){
        cerr << "Not enough matches to generate a sample" << endl;
    }
    int samplesInd[sampleNb]; //array to store the random numbers in
    vector<Match> sMatches;
    //generate random numbers:
    for (int i=0;i<sampleNb;i++)
    {
        bool check;
        int n;
        do
        {
        n=rand()%matchSize;
        //check if number is already used:
        check=true;
        for (int j=0;j<i;j++)
            if (n == samplesInd[j]) //number is already used
            {
                check=false;
                break;
            }
        } while (!check); //loop until new, unique number is found
        samplesInd[i]=n;
        sMatches.push_back(matches[n]);
    }
    return sMatches;
}

int getIterNb(int currentIterNb, int inlierNb, int sampleNb, int matchNb){
    int nextIterNb;
    //cout<<"nextIterNb: "<<nextIterNb<<endl;
    nextIterNb = log(BETA)/log(1- pow( (float)inlierNb/(float)matchNb, sampleNb));
    if(nextIterNb > currentIterNb){
        nextIterNb = currentIterNb;
    }
    return nextIterNb;
}
// Receives 8 matches and computes F on those samples
FMatrix<float,3,3> FiterComp (vector<Match>& sMatches){
    if(sMatches.size()<8){
        cerr << "Not enough matches for F calculation !" << endl;
    }
    // Create normalization matrix
    FMatrix<float,3,3> N(0.f);
    N(0,0) = 0.001;
    N(1,1) = 0.001;
    N(2,2) = 1;

    // Create linear system matrix
    FMatrix<float,9,9> A;
    for(int i=0; i<8; i++){
        Match curMatch = sMatches[i];
        // Get points
        DoublePoint3 point1;
        DoublePoint3 point2;
        point1[0] = curMatch.x1; point1[1] = curMatch.y1; point1[2] = 1;
        point2[0] = curMatch.x2; point2[1] = curMatch.y2; point2[2] = 1;

        // Normalize
        point1 = N*point1;
        point2 = N*point2;

        // Populate linear system
        float x1 = point1[0]; float y1 = point1[1]; float x2 = point2[0]; float y2 = point2[1];
        A(i,0) = x1*x2; A(i,1) = x1*y2; A(i,2) = x1;
        A(i,3) = y1*x2; A(i,4) = y1*y2; A(i,5) = y1;
        A(i,6) = x2; A(i,7) = y2; A(i,8) = 1;
    }
    for(int j=0; j<9; j++){
        A(8,j) = 0;
    }

    // Solve linear system using svd
    FVector<float,9> S;
    FMatrix<float,9,9> U, Vt;
    svd(A,U,S,Vt);
    FMatrix<float,3,3> computedF;
    for (int q=0; q<3; q++){
        for(int l=0; l<3; l++){
            computedF(q,l)= Vt.getRow(8)[3*q+l];
        }
    }
    // Add rank(F)=2 constraint
    FVector<float,3> S2;
    FMatrix<float,3,3> U2, Vt2;
    svd(computedF,U2,S2,Vt2);
    S2[2] = 0;
    computedF = U2 * Diagonal(S2) * Vt2;

    // Normalization of computedF
    computedF = N*computedF*N;
    return computedF;
}

// Gets indexes of current inliers
vector<int> Inlierscomp (vector<Match>& matches, FMatrix<float,3,3>& tempF, float dist){
    vector<int> inliers;
    for(int i=0; i<matches.size(); i++){
        Match curMatch = matches[i];
        DoublePoint3 point1;
        DoublePoint3 point2;
        point1[0] = curMatch.x1; point1[1] = curMatch.y1; point1[2] = 1;
        point2[0] = curMatch.x2; point2[1] = curMatch.y2; point2[2] = 1;
        FVector<float, 3> line;
        line = tempF*point2;
        float norm = sqrt(pow(line[0],2.0) + pow(line[1],2.0));
        line /= norm;
        if(abs(point1*line) < dist){
            inliers.push_back(i);
        }
    }
    return inliers;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    //cout<<"Inizio: "<<endl;
    const float Maxd = 1.5f; // Pixel error for inlier/outlier discrimination
    double Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;

    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    FMatrix<float,3,3> tempF;
    vector<int> inliers;
    vector<Match> sMatches;
    int samplingNb=0;

    cout<<"Initial samplingNb: "<<samplingNb<<endl;

    int const sampleSize(8);
    //cout<<"sampleSize: "<<sampleSize<<endl;
    //cout << "IbestInliers.size(): " << bestInliers.size() << endl;
    while(samplingNb < Niter){
        //cout<<"samplingNb: "<<samplingNb<<endl;
        //cout<<"Niter: "<<Niter<<endl;
        samplingNb++;
        sMatches = getS(matches, sampleSize);

        //cout<<"sMatches: "<<sMatches<<endl;
        tempF = FiterComp(sMatches);
        //cout<<"tempF.size(): "<<tempF.size()<<endl;
        inliers = Inlierscomp(matches, tempF, Maxd);
        //cout << "inliers.size() : " << inliers.size() << endl;
        //cout << "bestInliers.size(): " << bestInliers.size() << endl;
        if(inliers.size() > bestInliers.size()){
            //cout << "Iteration number: " << samplingNb << endl;
            //cout << "Better model found with " << inliers.size() << " inliers ! " << endl;
            bestInliers = inliers;
            bestF = tempF;
            if(bestInliers.size()>60){ // avoids int overflow
                Niter = getIterNb(Niter, bestInliers.size(), sampleSize, matches.size());
            }
            //cout <<"AAA: "<< Niter << " iterations wanted." << endl;
        }
    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    cout<<"Total numer of iteration: "<< samplingNb<<endl;
    cout<<"bestInliers: "<<bestInliers.size()<<endl;
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2, const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        DoublePoint3 point2;
        DoublePoint3 point1;
        int w = I1.width();
        if(x>w){
            point2[0] = x - w; point2[1] = y; point2[2] = 1;
            FVector<float, 3> line;
            line = F*point2;
            drawLine(0,(-1)*line[2]/line[1],w,(-1)*(line[2]+line[0]*w)/line[1], RED,3);
        }
        if(x<=w){
            point1[0] = x; point1[1] = y; point1[2] = 1;
            FVector<float, 3> line;
            line = transpose(F)*point1;
            drawLine(w, (-1)*(line[2])/line[1], 2*w, (-1)*(line[2]+line[0]*w)/line[1], BLUE,3);
        }
    }
}

int main(int argc, char* argv[])
{
    cout<<"Start"<<endl;
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,30,std::to_string(n)+ " matches",RED);
    click();

    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;
    cout<<"Right botton to close"<<endl;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);

    //too see the matching point in  the two images
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);

    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
