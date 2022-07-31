//
// Created by tanmay on 7/27/22.
//
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

int main ( );
double *lorenz_rhs ( double t, int m, double x[] );
double *r8vec_linspace_new ( int n, double a, double b );
double *rk4vec ( double t0, int n, double u0[], double dt,
               double *f ( double t, int n, double u[] ) );
void timestamp ( );

/******************************************************************************/

int main ( )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for LORENZ_ODE.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 October 2013

  Author:

    John Burkardt
*/
{
  char command_filename[] = "lorenz_ode_commands.txt";
  FILE *command_unit;
  char data_filename[] = "lorenz_ode_data.txt";
  FILE *data_unit;
  double dt;
  int i;
  int j;
  int m = 3;
  int n = 200000;
  double *t;
  double t_final;
  double *x;
  double *xnew;

  timestamp ( );
  printf ( "\n" );
  printf ( "LORENZ_ODE\n" );
  printf ( "  C version\n" );
  printf ( "  Compute solutions of the Lorenz system.\n" );
  printf ( "  Write data to a file for use by gnuplot.\n" );
  /*
    Data
  */
  t_final = 40.0;
  dt = t_final / ( double ) ( n );
  /*
    Store the initial conditions in entry 0.
  */
  t = r8vec_linspace_new ( n + 1, 0.0, t_final );
  x = ( double * ) malloc ( m * ( n + 1 ) * sizeof ( double ) );
  x[0+0*m] = 8.0;
  x[0+1*m] = 1.0;
  x[0+2*m] = 1.0;
  /*
    Compute the approximate solution at equally spaced times.
  */
  for ( j = 0; j < n; j++ )
  {
    xnew = rk4vec ( t[j], m, x+j*m, dt, lorenz_rhs );
    for ( i = 0; i < m; i++ )
    {
      x[i+(j+1)*m] = xnew[i];
    }
    free ( xnew );
  }
  /*
    Create the plot data file.
  */
  data_unit = fopen ( data_filename, "wt" );
  for ( j = 0; j <= n; j = j + 50 )
  {
    fprintf ( data_unit, "  %14.6g  %14.6g  %14.6g  %14.6g\n",
            t[j], x[0+j*m], x[1+j*m], x[2+j*m] );
  }
  fclose ( data_unit );
  printf ( "  Created data file \"%s\".\n", data_filename );
  /*
    Create the plot command file.
  */
  command_unit = fopen ( command_filename, "wt" );
  fprintf ( command_unit, "# %s\n", command_filename );
  fprintf ( command_unit, "#\n" );
  fprintf ( command_unit, "# Usage:\n" );
  fprintf ( command_unit, "#  gnuplot < %s\n", command_filename );
  fprintf ( command_unit, "#\n" );
  fprintf ( command_unit, "set term png\n" );
  fprintf ( command_unit, "set output 'xyz_time.png'\n" );
  fprintf ( command_unit, "set xlabel '<--- T --->'\n" );
  fprintf ( command_unit, "set ylabel '<--- X(T), Y(T), Z(T) --->'\n" );
  fprintf ( command_unit, "set title 'X, Y and Z versus Time'\n" );
  fprintf ( command_unit, "set grid\n" );
  fprintf ( command_unit, "set style data lines\n" );
  fprintf ( command_unit, "plot '%s' using 1:2 lw 3 linecolor rgb 'blue',",
          data_filename );
  fprintf ( command_unit, "'' using 1:3 lw 3 linecolor rgb 'red'," );
  fprintf ( command_unit, "'' using 1:4 lw 3 linecolor rgb 'green'\n" );
  fprintf ( command_unit, "set output 'xyz_3d.png'\n" );
  fprintf ( command_unit, "set xlabel '<--- X(T) --->'\n" );
  fprintf ( command_unit, "set ylabel '<--- Y(T) --->'\n" );
  fprintf ( command_unit, "set zlabel '<--- Z(T) --->'\n" );
  fprintf ( command_unit, "set title '(X(T),Y(T),Z(T)) trajectory'\n" );
  fprintf ( command_unit, "set grid\n" );
  fprintf ( command_unit, "set style data lines\n" );
  fprintf ( command_unit,
          "splot '%s' using 2:3:4 lw 1 linecolor rgb 'blue'\n", data_filename );
  fprintf ( command_unit, "quit\n" );
  fclose ( command_unit );
  printf ( "  Created command file '%s'\n", command_filename );
  /*
    Terminate.
  */
  printf ( "\n" );
  printf ( "LORENZ_ODE:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

double *lorenz_rhs ( double t, int m, double x[] )

/******************************************************************************/
/*
  Purpose:

    LORENZ_RHS evaluates the right hand side of the Lorenz ODE.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    10 October 2013

  Author:

    John Burkardt

  Parameters:

    Input, double T, the value of the independent variable.

    Input, int M, the spatial dimension.

    Input, double X[M], the values of the dependent variables
    at time T.

    Output, double DXDT[M], the values of the derivatives
    of the dependent variables at time T.
*/
{
  double beta = 8.0 / 3.0;
  double *dxdt;
  double rho = 28.0;
  double sigma = 10.0;

  dxdt = ( double * ) malloc ( m * sizeof ( double ) );

  dxdt[0] = sigma * ( x[1] - x[0] );
  dxdt[1] = x[0] * ( rho - x[2] ) - x[1];
  dxdt[2] = x[0] * x[1] - beta * x[2];

  return dxdt;
}
/******************************************************************************/

double *r8vec_linspace_new ( int n, double a, double b )

/******************************************************************************/
/*
  Purpose:

    R8VEC_LINSPACE_NEW creates a vector of linearly spaced values.

  Discussion:

    An R8VEC is a vector of R8's.

    4 points evenly spaced between 0 and 12 will yield 0, 4, 8, 12.

    In other words, the interval is divided into N-1 even subintervals,
    and the endpoints of intervals are used as the points.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 March 2011

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the vector.

    Input, double A, B, the first and last entries.

    Output, double R8VEC_LINSPACE_NEW[N], a vector of linearly spaced data.
*/
{
  int i;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( n == 1 )
  {
    x[0] = ( a + b ) / 2.0;
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      x[i] = ( ( double ) ( n - 1 - i ) * a
              + ( double ) (         i ) * b )
             / ( double ) ( n - 1     );
    }
  }
  return x;
}
/******************************************************************************/

double *rk4vec ( double t0, int m, double u0[], double dt,
               double *f ( double t, int n, double u[] ) )

/******************************************************************************/
/*
  Purpose:

    RK4VEC takes one Runge-Kutta step for a vector ODE.

  Discussion:

    It is assumed that an initial value problem, of the form

      du/dt = f ( t, u )
      u(t0) = u0

    is being solved.

    If the user can supply current values of t, u, a stepsize dt, and a
    function to evaluate the derivative, this function can compute the
    fourth-order Runge Kutta estimate to the solution at time t+dt.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    09 October 2013

  Author:

    John Burkardt

  Parameters:

    Input, double T0, the current time.

    Input, int M, the dimension of the space.

    Input, double U0[M], the solution estimate at the current time.

    Input, double DT, the time step.

    Input, double *F ( double T, int M, double U[] ), a function which evaluates
    the derivative, or right hand side of the problem.

    Output, double RK4VEC[M], the fourth-order Runge-Kutta solution estimate
    at time T0+DT.
*/
{
  double *f0;
  double *f1;
  double *f2;
  double *f3;
  int i;
  double t1;
  double t2;
  double t3;
  double *u;
  double *u1;
  double *u2;
  double *u3;
  /*
    Get four sample values of the derivative.
  */
  f0 = f ( t0, m, u0 );

  t1 = t0 + dt / 2.0;
  u1 = ( double * ) malloc ( m * sizeof ( double ) );
  for ( i = 0; i < m; i++ )
  {
    u1[i] = u0[i] + dt * f0[i] / 2.0;
  }
  f1 = f ( t1, m, u1 );

  t2 = t0 + dt / 2.0;
  u2 = ( double * ) malloc ( m * sizeof ( double ) );
  for ( i = 0; i < m; i++ )
  {
    u2[i] = u0[i] + dt * f1[i] / 2.0;
  }
  f2 = f ( t2, m, u2 );

  t3 = t0 + dt;
  u3 = ( double * ) malloc ( m * sizeof ( double ) );
  for ( i = 0; i < m; i++ )
  {
    u3[i] = u0[i] + dt * f2[i];
  }
  f3 = f ( t3, m, u3 );
  /*
    Combine them to estimate the solution.
  */
  u = ( double * ) malloc ( m * sizeof ( double ) );
  for ( i = 0; i < m; i++ )
  {
    u[i] = u0[i] + dt * ( f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i] ) / 6.0;
  }
  /*
    Free memory.
  */
  free ( f0 );
  free ( f1 );
  free ( f2 );
  free ( f3 );
  free ( u1 );
  free ( u2 );
  free ( u3 );

  return u;
}
/******************************************************************************/

void timestamp ( )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    24 September 2003

  Author:

    John Burkardt
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}