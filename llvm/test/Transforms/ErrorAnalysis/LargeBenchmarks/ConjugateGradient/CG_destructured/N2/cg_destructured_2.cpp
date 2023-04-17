#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <time.h>
using namespace std;

#define TYPE float
#define PRINT_PRECISION_TYPE "%0.7f"

// Inputs
#define N 100                       // dimension of the problem
#define A00 20
#define A01 1
#define A10 1
#define A11 3
#define B0 1
#define B1 2

// Setting a tolerance level which will be used as a termination condition for this algorithm
const TYPE TOLERANCE = 0.0000000001;
const TYPE NEARZERO = 0.0000000001;       // interpretation of "zero"

using vec    = vector<TYPE>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)

// Prototypes
void print( string title, const vec &V );
void print( string title, const matrix &A );
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( double a, const vec &U, double b, const vec &V );
double innerProduct( const vec &U, const vec &V );
double vectorNorm( const vec &V );
void conjugateGradientSolver( TYPE A_0_0, TYPE A_0_1,
                             TYPE A_1_0, TYPE A_1_1,
                             TYPE B_0, TYPE B_1 );


//======================================================================


int main() {
  // Calculate Time
  clock_t t;
  t = clock();
  conjugateGradientSolver(A00, A01,
                                  A10, A11,
                                  B0, B1);
  t = clock() - t;
  printf("Time: %f\n", ((double)t)/CLOCKS_PER_SEC);
  cout << "Solves AX = B\n";
//  print( "\nA:", A );
//  print( "\nB:", B );
//  print( "\nX:", X );
//  print( "\nCheck AX:", matrixTimesVector( A, X ) );
}


//======================================================================

// Prints the vector V
void print( string title, const vec &V )
{
  cout << title << '\n';

  int n = V.size();
  for ( int i = 0; i < n; i++ )
  {
    double x = V[i];   if ( abs( x ) < NEARZERO ) x = 0.0;
    cout << x << '\t';
  }
  cout << '\n';
}


//======================================================================

// Prints the matrix A
void print( string title, const matrix &A )
{
  cout << title << '\n';

  int m = A.size(), n = A[0].size();                      // A is an m x n matrix
  for ( int i = 0; i < m; i++ )
  {
    for ( int j = 0; j < n; j++ )
    {
      double x = A[i][j];   if ( abs( x ) < NEARZERO ) x = 0.0;
      cout << x << '\t';
    }
    cout << '\n';
  }
}


//======================================================================

// Inner product of the matrix A with vector V returned as C (a vector)
vec matrixTimesVector( const matrix &A, const vec &V )     // Matrix times vector
{
  int n = A.size();
  vec C( n );
  for ( int i = 0; i < n; i++ ) C[i] = innerProduct( A[i], V );
  return C;
}


//======================================================================

// Returns the Linear combination of aU+bV as a vector W.
vec vectorCombination( double a, const vec &U, double b, const vec &V )        // Linear combination of vectors
{
  int n = U.size();
  vec W( n );
  for ( int j = 0; j < n; j++ ) W[j] = a * U[j] + b * V[j];
  return W;
}


//======================================================================

// Returns the inner product of vector U with V.
double innerProduct( const vec &U, const vec &V )          // Inner product of U and V
{
  return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}


//======================================================================

// Computes and returns the Euclidean/2-norm of the vector V.
double vectorNorm( const vec &V )                          // Vector norm
{
  return sqrt( innerProduct( V, V ) );
}


//======================================================================

// The conjugate gradient solving algorithm.
__attribute__((noinline))
void conjugateGradientSolver(TYPE A_0_0, TYPE A_0_1,
                            TYPE A_1_0, TYPE A_1_1,
                            TYPE B_0, TYPE B_1) {
  // Initializing vector X which will be set to the solution by this algorithm.
  TYPE X_0 = 0.0, X_1 = 0.0;

  // Vector Initializations
  TYPE R_0 = B_0, R_1 = B_1;
  TYPE P_0 = B_0, P_1 = B_1;

  int k = 0;

  while ( k < N ) {
    TYPE Rold_0 = R_0, Rold_1 = R_1;
    TYPE AP_0, AP_1;


//    for ( int i = 0; i < n; i++ ) {
//      AP[i] = 0;
//      for (int j = 0; j < int(A[0].size()); ++j)
//        AP[i] += A[i][j]*P[j];
//    }

    AP_0 = AP_1 = 0;

    AP_0 += A_0_0*P_0 + A_0_1*P_1;
    AP_1 += A_1_0*P_0 + A_1_1*P_1;

    TYPE NormOfR = 0;
//    for (int j = 0; j < n; ++j)
//      NormOfR += R[j]*R[j];

    NormOfR += R_0*R_0 + R_1*R_1;


    TYPE P_AP_Product = 0;
//    for (int j = 0; j < n; ++j)
//      P_AP_Product += P[j]*AP[j];

    P_AP_Product += P_0*AP_0 + P_1*AP_1;

    TYPE alpha = NormOfR / max( P_AP_Product, NEARZERO );


//    for ( int j = 0; j < n; j++ )
//      X[j] = X[j] + alpha * P[j];                             // Next estimate of solution
//      R[j] = R[j] - alpha * AP[j];                            // Residual
//    if ( vectorNorm( R ) < TOLERANCE ) break;             // Convergence test

    X_0 += alpha * P_0;
    X_1 += alpha * P_1;

    R_0 -= alpha * AP_0;
    R_1 -= alpha * AP_1;

    NormOfR = 0;
    NormOfR += R_0*R_0 + R_1*R_1;
    if (NormOfR < TOLERANCE) break;


    TYPE NormOfRold = 0;
//    for (int j = 0; j < n; ++j)
//      NormOfRold += Rold[j] * Rold[j];

    NormOfRold += Rold_0*Rold_0 + Rold_1*Rold_1;

//    double beta = NormOfR / max( NormOfRold, NEARZERO );

    TYPE beta = NormOfR / max( NormOfRold, NEARZERO );

//    for ( int j = 0; j < n; j++ )
//      P[j] = R[j] + beta * P[j];                            // Next gradient

    P_0 = R_0 + beta * P_0;
    P_1 = R_1 + beta * P_1;

    k++;
  }

  printf("Number of iterations: %d\n", k);

  vec X( 2, 0.0);
  X[0] = X_0;
  X[1] = X_1;

  return ;
}