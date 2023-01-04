#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

const double NEARZERO = 1.0e-10;       // interpretation of "zero"

using vec    = vector<double>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)

// Prototypes
void print( string title, const vec &V );
void print( string title, const matrix &A );
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( double a, const vec &U, double b, const vec &V );
double innerProduct( const vec &U, const vec &V );
double vectorNorm( const vec &V );
vec conjugateGradientSolver( const matrix &A, const vec &B );


//======================================================================


int main()
{
  matrix A = { { 4, 1 }, { 1, 3 } };
  vec B = { 1, 2 };

  vec X = conjugateGradientSolver( A, B );

  cout << "Solves AX = B\n";
  print( "\nA:", A );
  print( "\nB:", B );
  print( "\nX:", X );
  print( "\nCheck AX:", matrixTimesVector( A, X ) );
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
vec conjugateGradientSolver( const matrix &A, const vec &B )
{
  // Setting a tolerance level which will be used as a termination condition for this algorithm
  double TOLERANCE = 1.0e-10;

  // Number of vectors/rows in the matrix A.
  int n = A.size();
  int iterations = 2000;
  
  // Initializing vector X which will be set to the solution by this algorithm.
  vec X( n, 0.0 );

  
  vec R = B;
  vec P = R;
  int k = 0;

  while ( k < iterations )
  {
    vec Rold = R;
    vec AP( n );
    for ( int i = 0; i < n; i++ ) {
      AP[i] = 0;
      for (int j = 0; j < int(A[0].size()); ++j)
        AP[i] += A[i][j]*P[j];
    }

    double NormOfR = 0;
    for (int j = 0; j < n; ++j)
      NormOfR += R[j]*R[j];

    double P_AP_Product = 0;
    for (int j = 0; j < n; ++j)
      P_AP_Product += P[j]*AP[j];

    double alpha = NormOfR / max( P_AP_Product, NEARZERO );


    for ( int j = 0; j < n; j++ )
      X[j] = X[j] + alpha * P[j];                             // Next estimate of solution
    for ( int j = 0; j < n; j++ )
      R[j] = R[j] - alpha * AP[j];                            // Residual
    if ( vectorNorm( R ) < TOLERANCE ) break;             // Convergence test


    NormOfR = 0;
    for (int j = 0; j < n; ++j)
      NormOfR += R[j]*R[j];

    double NormOfRold = 0;
    for (int j = 0; j < n; ++j)
      NormOfRold += Rold[j]*Rold[j];

    double beta = NormOfR / max( NormOfRold, NEARZERO );

    for ( int j = 0; j < n; j++ )
      P[j] = R[j] + beta * P[j];                            // Next gradient
    k++;
  }

  return X;
}