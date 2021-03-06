// Imagine++ project
// Project:  GraphCuts
// Author:   Renaud Marlet
//Tosato Lucrezia
#include <Imagine/Images.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "maxflow/graph.h"

using namespace std;
using namespace Imagine;

typedef Image<byte> byteImage;
typedef Image<double> doubleImage;

// neighbors
static const int dx[]={+1,  0};
static const int dy[]={ 0, +1};

static const int nN = (sizeof(dx)/sizeof(*dx));

// Return image of mean intensity value over (2n+1)x(2n+1) patch
doubleImage meanImage(const doubleImage&I, int n)
{
    // Create image for mean values
    int w = I.width(), h = I.height();
    doubleImage IM(w,h);
    // Compute patch area
    double area = (2*n+1)*(2*n+1);
    // For each pixel
    for (int i=0;i<w;i++)
        for (int j=0;j<h;j++) {
            // If pixel is close to border (less than n pixels)
            if (j<n || j>=h-n || i<n ||i>=w-n) {
                // Mean is meaningless
                IM(i,j)=0;
                continue;
            }
            // Otherwise, initialize sum
            double sum = 0;
            // For each pixel displacement in patch
            for(int x = i-n; x <= i+n; x++)
                for (int y = j-n; y <= j+n; y++)
                    // Sum pixel intensity
                    sum += I(x,y);
            // Remember average intensity
            IM(i,j) = sum / area;
    }
    return IM;
}

// Compute correlation between two pixels in images 1 and 2
double correl(
    const doubleImage& I1,  // Image 1
    const doubleImage& I1M, // Image of mean intensity value over patch
    const doubleImage& I2,  // Image2
    const doubleImage& I2M, // Image of mean intensity value over patch
    int u1, int v1,         // Pixel of interest in image 1
    int u2, int v2,         // Pixel of interest in image 2
    int n)                  // Half patch size
{
    // Initialize correlation
    double c = 0;
    // For each pixel displacement in patch
    for (int x = -n; x <= n; x++)
        for(int y = -n; y <= n; y++)
            // Sum correllation
            c += (I1(u1+x,v1+y) - I1M(u1,v1)) * (I2(u2+x,v2+y) - I2M(u2,v2));
    // Return correlation normalized by patch size
    return c / ((2*n+1)*(2*n+1));
}

// Compute ZNCC between two patches in images 1 and 2
double zncc(
    const doubleImage& I1,  // Image 1
    const doubleImage& I1M, // Image of mean intensity value over patch
    const doubleImage& I2,  // Image2
    const doubleImage& I2M, // Image of mean intensity value over patch
    int u1, int v1,         // Pixel of interest in image 1
    int u2, int v2,         // Pixel of interest in image 2
    int n)                  // Half patch size
{
    // Compute variance of patch in image 1
    double var1 = correl(I1, I1M, I1, I1M, u1, v1, u1, v1, n);
    if (var1 == 0)
    return 0;
    // Compute variance of patch in image 2
    double var2 = correl(I2, I2M, I2, I2M, u2, v2, u2, v2, n);
    if (var2 == 0)
    return 0;
    // Return normalized cross correlation
    return correl(I1, I1M, I2, I2M, u1, v1, u2, v2, n) / sqrt(var1 * var2);
}
//
double rho(double c){
    return (c < 0.0) ? 1 : sqrt(1-c);
}
//
// Load two rectified images.
// Compute the disparity of image 2 w.r.t. image 1.
// Display disparity map.
// Display 3D mesh of corresponding depth map.
//
// Images are clipped to focus on pixels visible in both images.
// OPTIMIZATION: to make the program faster, a zoom factor is used to
// down-sample the input images on the fly. For an image of size WxH, you will
// only look at pixels (n+zoom*i,n+zoom*j) with n the radius of patch.
int main()
{
    cout << "Loading images... " << flush;
    byteImage I;
    doubleImage I1,I2;
    load(I, srcPath("face00R.png"));
    //load(I, srcPath("img31.png"));
    //load(I, srcPath("img3.png"));
    I1 = I.getSubImage(IntPoint2(20,30), IntPoint2(430,420));
    load(I, srcPath("face01R.png"));
    //load(I, srcPath("img21.png"));
    //load(I, srcPath("img2.png"));
    I2 = I.getSubImage(IntPoint2(20,30), IntPoint2(480,420));
    cout << "done" << endl;

    cout << "Setting parameters... " << flush;
    // Generic parameters
    int zoom = 2;      // Zoom factor (to speedup computations)
    int n = 3;         // Consider correlation patches of size (2n+1)*(2n+1)
    float lambdaf = 0.1;//0.1; // Weight of regularization (smoothing) term
    // for lambda = 20 labdaf=1, lambda = 10 labdaf=0.5,lambda = 2 labdaf=0.1,lambda = 1 labdaf=0.05, lambda = 0 labdaf=0
    int wcc = max(1+int(1/lambdaf),20); // Energy discretization precision [as we build a graph with 'int' weights]
    int lambda = lambdaf * wcc; // Weight of regularization (smoothing) term [must be >= 1]
    float sigma = 3;   // Gaussian blur parameter for disparity
    // Image-specific, hard-coded parameters for approximate 3D reconstruction,
    // as real geometry before rectification is not known
    int dmin = 5;    // Minmum disparity #10 dMin=-30, dMax=-7
    int dmax = 100;     // Maximum disparity #55
    float fB = 40000;  // Depth factor #40000
    float db = 100;    // Disparity base #100
    cout << "done" << endl;

    cout << "Displaying images... " << flush;
    int w1 = I1.width(), w2 = I2.width(), h = I1.height();
    openWindow(w1+w2, h);
    display(grey(I1));
    display(grey(I2), w1, 0);
    cout << "done" << endl;

    cout << "Constructing graph (be patient)... " << flush;
    // Precompute images of mean intensity value over patch
    doubleImage I1M = meanImage(I1,n), I2M = meanImage(I2,n);
    // Zoomed image dimension, disregarding borders (strips of width equal to patch half-size)
    int nx = (w1 - 2*n) / zoom, ny = (h - 2*n) / zoom;
    int nd = dmax-dmin; // Disparity range
    int INF=1000000; // "Infinite" value

    // Value of the K penalizer
    int penal = 1 + nd * 4 * lambda;

    // Create graph
    // The graph library works with node numbers. To clarify the setting, create
    // a formula to associate a unique node number to a triplet (x,y,d) of pixel
    // coordinates and disparity.
    // The library assumes an edge consists of a pair of oriented edges, one in
    // each direction. Put correct weights to the edges, such as 0, INF, or a
    // an intermediate weight.
    /////  BEGIN CODE TO COMPLETE: define appropriate graph G

    auto node = [nx, ny, nd](int x, int y, int d){
        return d + nd * (x + nx * y);
    };

    int nPixel = nx * ny;
    int nNode = nPixel * nd ;
    int nEdge = nPixel * ( 3 * nd - 1 ) - nd * (nx+ny);
    Graph<int,int,int> G(nNode,nEdge);
    G.add_node(nNode);

    // Building the graph
    const int refreshStep = nx*5/100;
    for (int x = 0; x < nx; x++){
        if((x-n-1)/refreshStep != (x-n)/refreshStep)
            std::cout << "Constructing the graph... " << 5*(x-n)/refreshStep <<"%          \r"<<std::flush;
        for (int y = 0; y < ny; y++){
            for (int d = 0; d < nd; d++){
                // Edges to neighbors
                for(int i=0; i<nN; i++) {
                    int xn=x+dx[i], yn=y+dy[i];
                    if ((xn < nx) && (yn < ny)){
                        G.add_edge(node(x,y,d),node(xn, yn, d),lambda,lambda);
                    }
                }
                // Edges between levels of disparity
                int cap = penal;
                double ZNcc = zncc(I1, I1M, I2, I2M, n+zoom*x, n+zoom*y, n+zoom*x + d + dmin, n+zoom*y, n);
                cap += wcc * rho(ZNcc);
                if (d == (nd - 1)){
                    G.add_tweights(node(x,y,d), 0, cap);
                }
                if (d == 0){
                    G.add_tweights(node(x,y,0), cap, 0);
                }
                else {
                    G.add_edge(node(x,y,d-1), node(x,y,d), cap, 0);
                }
            }
        }
    }
    // Done
    cout << "Constructing the graph... done" << endl;
    /////  END CODE TO BE COMPLETED
    /////------------------------------------------------------------

    cout << "Computing minimum cut... " << flush;
    int f = G.maxflow();
    cout << "done" << endl<< "  max flow = " << f << endl;

    cout << "Extracting disparity map from minimum cut... " << flush;
    doubleImage D(nx,ny);
    // For each pixel
    const int source = Graph<int,int,int>::SOURCE;
    for (int i=0;i<nx;i++) {
        for (int j=0;j<ny;j++) {
            ///// Extract disparity from minimum cut
            int disp = 0;
            while ((disp < nd) && (G.what_segment(node(i,j,disp))==source)){
                disp++;
            }
            D(i,j) = dmin + disp;
        }
    }
    cout << "done" << endl;

    cout << "Displaying disparity map... " << flush;
    display(enlarge(grey(D),zoom), n, n);
    // Done
    cout << "done" << endl;

    cout << "Click to compute and display blured disparity map... " << flush;
    click();
    D=blur(D,sigma);
    display(enlarge(grey(D),zoom),n,n);
    // Done
    cout << "done" << endl;

    cout << "Click to compute depth map and 3D mesh renderings... " << flush;
    click();
    setActiveWindow( openWindow3D(512, 512, "3D") );
    //Window W = openWindow3D(512, 512, "3D");
    //setActiveWindow(W);
    ///// Compute 3D points
    Array<FloatPoint3> p(nx*ny);
    Array<Color> pcol(nx*ny);
    // For each pixel in image 1
    for (int i = 0;i < nx; i++)
        for (int j = 0; j < ny; j++) {
            // Compute depth: magic constants depending on camera pose
            float depth = fB / (db + D(i,j)-dmin);
            // Create 3D point
            p[i+nx*j] = FloatPoint3(float(i), float(j), -depth);
            // Get point color in original image
            byte g = byte( I1(n+i*zoom,n+j*zoom) );
            pcol[i+nx*j] = Color(g,g,g);
    }
    ///// Create mesh from 3D points
    Array<Triangle> t(2*(nx-1)*(ny-1));
    Array<Color> tcol(2*(nx-1)*(ny-1));
    // For each pixel in image 1 (but last line/column)
    for (int i = 0; i < nx-1; i++)
        for (int j = 0; j < ny-1; j++) {
            // Create triangles with next pixels in line/column
            t[2*(i+j*(nx-1))] = Triangle(i+nx*j, i+1+nx*j, i+nx*(j+1));
            t[2*(i+j*(nx-1))+1] = Triangle(i+1+nx*j, i+1+nx*(j+1), i+nx*(j+1));
            tcol[2*(i+j*(nx-1))] = pcol[i+nx*j];
            tcol[2*(i+j*(nx-1))+1] = pcol[i+nx*j];
    }
    // Create first mesh as textured with colors taken from original image
    Mesh Mt(p.data(), nx*ny, t.data(), 2*(nx-1)*(ny-1), 0, 0, FACE_COLOR);
    Mt.setColors(TRIANGLE, tcol.data());
    // Create second mesh with artificial light
    Mesh Mg(p.data(), nx*ny, t.data(), 2*(nx-1)*(ny-1), 0, 0,
        CONSTANT_COLOR, SMOOTH_SHADING);
    cout << "done" << endl;


    ///// Display 3D mesh renderings
    cout << "***** 3D mesh renderings *****" << endl;
    cout << "- Button 1: toggle textured or gray rendering" << endl;
    cout << "- SHIFT+Button 1: rotate" << endl;
    cout << "- SHIFT+Button 3: translate" << endl;
    cout << "- Mouse wheel: zoom" << endl;
    cout << "- SHIFT+a: zoom out" << endl;
    cout << "- SHIFT+z: zoom in" << endl;
    cout << "- SHIFT+r: recenter camera" << endl;
    cout << "- SHIFT+m: toggle solid/wire/points mode" << endl;
    cout << "- Button 3: exit" << endl;
    // Display textured mesh
    showMesh(Mt);
    bool textured = true;
    //Event evt;
    while (true) {
        Event evt;
        getEvent(5, evt);
        // On mouse button 1
        if (evt.type == EVT_BUT_ON && evt.button == 1) {
        // Toggle textured rendering and gray rendering
            if (textured) {
            hideMesh(Mt,false);
            showMesh(Mg,false);
            } else {
            hideMesh(Mg,false);
            showMesh(Mt,false);
            }
            textured = !textured;
        }
        // On mouse button 3
        if (evt.type == EVT_BUT_ON && evt.button == 3)
            break;
    }
    endGraphics();
    return 0;
}
