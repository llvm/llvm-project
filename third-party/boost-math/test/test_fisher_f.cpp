// test_fisher_squared.cpp

// Copyright Paul A. Bristow 2006.
// Copyright John Maddock 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/config.hpp>
#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;
#endif

#include <boost/math/distributions/fisher_f.hpp> // for fisher_f_distribution
using boost::math::fisher_f_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE
#include "test_out_of_range.hpp"

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

template <class RealType>
RealType naive_pdf(RealType df1, RealType df2, RealType x)
{
   //
   // Calculate the PDF naively using direct evaluation
   // of equation 2 from http://mathworld.wolfram.com/F-Distribution.html
   //
   // Our actual PDF implementation uses a completely different method,
   // so this is a good sanity check that our math is correct.
   //
   using namespace std; // For ADL of std functions.
   RealType e = boost::math::lgamma((df1 + df2) / 2);
   e += log(df1) * df1 / 2;
   e += log(df2) * df2 / 2;
   e += log(x) * ((df1 / 2) - 1);
   e -= boost::math::lgamma(df1 / 2);
   e -= boost::math::lgamma(df2 / 2);
   e -= log(df2 + x * df1) * (df1 + df2) / 2;
   return exp(e);
}

template <class RealType>
void test_spot(
     RealType df1,    // Degrees of freedom 1
     RealType df2,    // Degrees of freedom 2
     RealType cs,    // Chi Square statistic
     RealType P,     // CDF
     RealType Q,     // Complement of CDF
     RealType tol)   // Test tolerance
{
   boost::math::fisher_f_distribution<RealType> dist(df1, df2);
   BOOST_CHECK_CLOSE(
      cdf(dist, cs), P, tol);
   BOOST_CHECK_CLOSE(
      pdf(dist, cs), naive_pdf(dist.degrees_of_freedom1(), dist.degrees_of_freedom2(), cs), tol);
   if((P < 0.999) && (Q < 0.999))
   {
      //
      // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is free of error:
      //
      BOOST_CHECK_CLOSE(
         cdf(complement(dist, cs)), Q, tol);
      BOOST_CHECK_CLOSE(
            quantile(dist, P), cs, tol);
      BOOST_CHECK_CLOSE(
            quantile(complement(dist, Q)), cs, tol);
   }
}

//
// This test data is taken from the tables of upper
// critical values of the F distribution available
// at http://www.itl.nist.gov/div898/handbook/eda/section3/eda3673.htm
//
double q[] = { 0.10, 0.05, 0.025, 0.01, 0.001 };
double upper_critical_values[][10] = {
   { 161.448,199.500,215.707,224.583,230.162,233.986,236.768,238.882,240.543,241.882 },
   { 18.513, 19.000, 19.164, 19.247, 19.296, 19.330, 19.353, 19.371, 19.385, 19.396 },
   { 10.128,  9.552,  9.277,  9.117,  9.013,  8.941,  8.887,  8.845,  8.812,  8.786 },
   { 7.709,  6.944,  6.591,  6.388,  6.256,  6.163,  6.094,  6.041,  5.999,  5.964 },
   { 6.608,  5.786,  5.409,  5.192,  5.050,  4.950,  4.876,  4.818,  4.772,  4.735 },
   { 5.987,  5.143,  4.757,  4.534,  4.387,  4.284,  4.207,  4.147,  4.099,  4.060 },
   { 5.591,  4.737,  4.347,  4.120,  3.972,  3.866,  3.787,  3.726,  3.677,  3.637 },
   { 5.318,  4.459,  4.066,  3.838,  3.687,  3.581,  3.500,  3.438,  3.388,  3.347 },
   { 5.117,  4.256,  3.863,  3.633,  3.482,  3.374,  3.293,  3.230,  3.179,  3.137 },
   { 4.965,  4.103,  3.708,  3.478,  3.326,  3.217,  3.135,  3.072,  3.020,  2.978 },
   { 4.844,  3.982,  3.587,  3.357,  3.204,  3.095,  3.012,  2.948,  2.896,  2.854 },
   { 4.747,  3.885,  3.490,  3.259,  3.106,  2.996,  2.913,  2.849,  2.796,  2.753 },
   { 4.667,  3.806,  3.411,  3.179,  3.025,  2.915,  2.832,  2.767,  2.714,  2.671 },
   { 4.600,  3.739,  3.344,  3.112,  2.958,  2.848,  2.764,  2.699,  2.646,  2.602 },
   { 4.543,  3.682,  3.287,  3.056,  2.901,  2.790,  2.707,  2.641,  2.588,  2.544 },
   { 4.494,  3.634,  3.239,  3.007,  2.852,  2.741,  2.657,  2.591,  2.538,  2.494 },
   { 4.451,  3.592,  3.197,  2.965,  2.810,  2.699,  2.614,  2.548,  2.494,  2.450 },
   { 4.414,  3.555,  3.160,  2.928,  2.773,  2.661,  2.577,  2.510,  2.456,  2.412 },
   { 4.381,  3.522,  3.127,  2.895,  2.740,  2.628,  2.544,  2.477,  2.423,  2.378 },
   { 4.351,  3.493,  3.098,  2.866,  2.711,  2.599,  2.514,  2.447,  2.393,  2.348 },
   { 4.325,  3.467,  3.072,  2.840,  2.685,  2.573,  2.488,  2.420,  2.366,  2.321 },
   { 4.301,  3.443,  3.049,  2.817,  2.661,  2.549,  2.464,  2.397,  2.342,  2.297 },
   { 4.279,  3.422,  3.028,  2.796,  2.640,  2.528,  2.442,  2.375,  2.320,  2.275 },
   { 4.260,  3.403,  3.009,  2.776,  2.621,  2.508,  2.423,  2.355,  2.300,  2.255 },
   { 4.242,  3.385,  2.991,  2.759,  2.603,  2.490,  2.405,  2.337,  2.282,  2.236 },
   { 4.225,  3.369,  2.975,  2.743,  2.587,  2.474,  2.388,  2.321,  2.265,  2.220 },
   { 4.210,  3.354,  2.960,  2.728,  2.572,  2.459,  2.373,  2.305,  2.250,  2.204 },
   { 4.196,  3.340,  2.947,  2.714,  2.558,  2.445,  2.359,  2.291,  2.236,  2.190 },
   { 4.183,  3.328,  2.934,  2.701,  2.545,  2.432,  2.346,  2.278,  2.223,  2.177 },
   { 4.171,  3.316,  2.922,  2.690,  2.534,  2.421,  2.334,  2.266,  2.211,  2.165 },
   { 4.160,  3.305,  2.911,  2.679,  2.523,  2.409,  2.323,  2.255,  2.199,  2.153 },
   { 4.149,  3.295,  2.901,  2.668,  2.512,  2.399,  2.313,  2.244,  2.189,  2.142 },
   { 4.139,  3.285,  2.892,  2.659,  2.503,  2.389,  2.303,  2.235,  2.179,  2.133 },
   { 4.130,  3.276,  2.883,  2.650,  2.494,  2.380,  2.294,  2.225,  2.170,  2.123 },
   { 4.121,  3.267,  2.874,  2.641,  2.485,  2.372,  2.285,  2.217,  2.161,  2.114 },
   { 4.113,  3.259,  2.866,  2.634,  2.477,  2.364,  2.277,  2.209,  2.153,  2.106 },
   { 4.105,  3.252,  2.859,  2.626,  2.470,  2.356,  2.270,  2.201,  2.145,  2.098 },
   { 4.098,  3.245,  2.852,  2.619,  2.463,  2.349,  2.262,  2.194,  2.138,  2.091 },
   { 4.091,  3.238,  2.845,  2.612,  2.456,  2.342,  2.255,  2.187,  2.131,  2.084 },
   { 4.085,  3.232,  2.839,  2.606,  2.449,  2.336,  2.249,  2.180,  2.124,  2.077 },
   { 4.079,  3.226,  2.833,  2.600,  2.443,  2.330,  2.243,  2.174,  2.118,  2.071 },
   { 4.073,  3.220,  2.827,  2.594,  2.438,  2.324,  2.237,  2.168,  2.112,  2.065 },
   { 4.067,  3.214,  2.822,  2.589,  2.432,  2.318,  2.232,  2.163,  2.106,  2.059 },
   { 4.062,  3.209,  2.816,  2.584,  2.427,  2.313,  2.226,  2.157,  2.101,  2.054 },
   { 4.057,  3.204,  2.812,  2.579,  2.422,  2.308,  2.221,  2.152,  2.096,  2.049 },
   { 4.052,  3.200,  2.807,  2.574,  2.417,  2.304,  2.216,  2.147,  2.091,  2.044 },
   { 4.047,  3.195,  2.802,  2.570,  2.413,  2.299,  2.212,  2.143,  2.086,  2.039 },
   { 4.043,  3.191,  2.798,  2.565,  2.409,  2.295,  2.207,  2.138,  2.082,  2.035 },
   { 4.038,  3.187,  2.794,  2.561,  2.404,  2.290,  2.203,  2.134,  2.077,  2.030 },
   { 4.034,  3.183,  2.790,  2.557,  2.400,  2.286,  2.199,  2.130,  2.073,  2.026 },
   { 4.030,  3.179,  2.786,  2.553,  2.397,  2.283,  2.195,  2.126,  2.069,  2.022 },
   { 4.027,  3.175,  2.783,  2.550,  2.393,  2.279,  2.192,  2.122,  2.066,  2.018 },
   { 4.023,  3.172,  2.779,  2.546,  2.389,  2.275,  2.188,  2.119,  2.062,  2.015 },
   { 4.020,  3.168,  2.776,  2.543,  2.386,  2.272,  2.185,  2.115,  2.059,  2.011 },
   { 4.016,  3.165,  2.773,  2.540,  2.383,  2.269,  2.181,  2.112,  2.055,  2.008 },
   { 4.013,  3.162,  2.769,  2.537,  2.380,  2.266,  2.178,  2.109,  2.052,  2.005 },
   { 4.010,  3.159,  2.766,  2.534,  2.377,  2.263,  2.175,  2.106,  2.049,  2.001 },
   { 4.007,  3.156,  2.764,  2.531,  2.374,  2.260,  2.172,  2.103,  2.046,  1.998 },
   { 4.004,  3.153,  2.761,  2.528,  2.371,  2.257,  2.169,  2.100,  2.043,  1.995 },
   { 4.001,  3.150,  2.758,  2.525,  2.368,  2.254,  2.167,  2.097,  2.040,  1.993 },
   { 3.998,  3.148,  2.755,  2.523,  2.366,  2.251,  2.164,  2.094,  2.037,  1.990 },
   { 3.996,  3.145,  2.753,  2.520,  2.363,  2.249,  2.161,  2.092,  2.035,  1.987 },
   { 3.993,  3.143,  2.751,  2.518,  2.361,  2.246,  2.159,  2.089,  2.032,  1.985 },
   { 3.991,  3.140,  2.748,  2.515,  2.358,  2.244,  2.156,  2.087,  2.030,  1.982 },
   { 3.989,  3.138,  2.746,  2.513,  2.356,  2.242,  2.154,  2.084,  2.027,  1.980 },
   { 3.986,  3.136,  2.744,  2.511,  2.354,  2.239,  2.152,  2.082,  2.025,  1.977 },
   { 3.984,  3.134,  2.742,  2.509,  2.352,  2.237,  2.150,  2.080,  2.023,  1.975 },
   { 3.982,  3.132,  2.740,  2.507,  2.350,  2.235,  2.148,  2.078,  2.021,  1.973 },
   { 3.980,  3.130,  2.737,  2.505,  2.348,  2.233,  2.145,  2.076,  2.019,  1.971 },
   { 3.978,  3.128,  2.736,  2.503,  2.346,  2.231,  2.143,  2.074,  2.017,  1.969 },
   { 3.976,  3.126,  2.734,  2.501,  2.344,  2.229,  2.142,  2.072,  2.015,  1.967 },
   { 3.974,  3.124,  2.732,  2.499,  2.342,  2.227,  2.140,  2.070,  2.013,  1.965 },
   { 3.972,  3.122,  2.730,  2.497,  2.340,  2.226,  2.138,  2.068,  2.011,  1.963 },
   { 3.970,  3.120,  2.728,  2.495,  2.338,  2.224,  2.136,  2.066,  2.009,  1.961 },
   { 3.968,  3.119,  2.727,  2.494,  2.337,  2.222,  2.134,  2.064,  2.007,  1.959 },
   { 3.967,  3.117,  2.725,  2.492,  2.335,  2.220,  2.133,  2.063,  2.006,  1.958 },
   { 3.965,  3.115,  2.723,  2.490,  2.333,  2.219,  2.131,  2.061,  2.004,  1.956 },
   { 3.963,  3.114,  2.722,  2.489,  2.332,  2.217,  2.129,  2.059,  2.002,  1.954 },
   { 3.962,  3.112,  2.720,  2.487,  2.330,  2.216,  2.128,  2.058,  2.001,  1.953 },
   { 3.960,  3.111,  2.719,  2.486,  2.329,  2.214,  2.126,  2.056,  1.999,  1.951 },
   { 3.959,  3.109,  2.717,  2.484,  2.327,  2.213,  2.125,  2.055,  1.998,  1.950 },
   { 3.957,  3.108,  2.716,  2.483,  2.326,  2.211,  2.123,  2.053,  1.996,  1.948 },
   { 3.956,  3.107,  2.715,  2.482,  2.324,  2.210,  2.122,  2.052,  1.995,  1.947 },
   { 3.955,  3.105,  2.713,  2.480,  2.323,  2.209,  2.121,  2.051,  1.993,  1.945 },
   { 3.953,  3.104,  2.712,  2.479,  2.322,  2.207,  2.119,  2.049,  1.992,  1.944 },
   { 3.952,  3.103,  2.711,  2.478,  2.321,  2.206,  2.118,  2.048,  1.991,  1.943 },
   { 3.951,  3.101,  2.709,  2.476,  2.319,  2.205,  2.117,  2.047,  1.989,  1.941 },
   { 3.949,  3.100,  2.708,  2.475,  2.318,  2.203,  2.115,  2.045,  1.988,  1.940 },
   { 3.948,  3.099,  2.707,  2.474,  2.317,  2.202,  2.114,  2.044,  1.987,  1.939 },
   { 3.947,  3.098,  2.706,  2.473,  2.316,  2.201,  2.113,  2.043,  1.986,  1.938 },
   { 3.946,  3.097,  2.705,  2.472,  2.315,  2.200,  2.112,  2.042,  1.984,  1.936 },
   { 3.945,  3.095,  2.704,  2.471,  2.313,  2.199,  2.111,  2.041,  1.983,  1.935 },
   { 3.943,  3.094,  2.703,  2.470,  2.312,  2.198,  2.110,  2.040,  1.982,  1.934 },
   { 3.942,  3.093,  2.701,  2.469,  2.311,  2.197,  2.109,  2.038,  1.981,  1.933 },
   { 3.941,  3.092,  2.700,  2.467,  2.310,  2.196,  2.108,  2.037,  1.980,  1.932 },
   { 3.940,  3.091,  2.699,  2.466,  2.309,  2.195,  2.106,  2.036,  1.979,  1.931 },
   { 3.939,  3.090,  2.698,  2.465,  2.308,  2.194,  2.105,  2.035,  1.978,  1.930 },
   { 3.938,  3.089,  2.697,  2.465,  2.307,  2.193,  2.104,  2.034,  1.977,  1.929 },
   { 3.937,  3.088,  2.696,  2.464,  2.306,  2.192,  2.103,  2.033,  1.976,  1.928 },
   { 3.936,  3.087,  2.696,  2.463,  2.305,  2.191,  2.103,  2.032,  1.975,  1.927 }
};


template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
  // Basic sanity checks, test data is to three decimal places only
  // so set tolerance to 0.002 expressed as a percentage.  Note that
  // we can't even get full 3 digit accuracy since the data we're
  // using as input has *already been rounded*, leading to even
  // greater differences in output.  As an accuracy test this is
  // pretty useless, but it is an excellent sanity check.

  RealType tolerance = 0.002f * 100;
  cout << "Tolerance = " << tolerance << "%." << endl;

  using boost::math::fisher_f_distribution;
  using  ::boost::math::fisher_f;
  using  ::boost::math::cdf;
  using  ::boost::math::pdf;

  for(unsigned i = 0; i < sizeof(upper_critical_values) / sizeof(upper_critical_values[0]); ++i)
  {
     for(unsigned j = 0; j < sizeof(upper_critical_values[0])/sizeof(upper_critical_values[0][0]); ++j)
     {
        test_spot(
           static_cast<RealType>(j+1),   // degrees of freedom 1
           static_cast<RealType>(i+1),   // degrees of freedom 2
           static_cast<RealType>(upper_critical_values[i][j]), // test statistic F
           static_cast<RealType>(0.95),       // Probability of result (CDF), P
           static_cast<RealType>(0.05),       // Q = 1 - P
           tolerance);
     }
  }

   // http://www.vias.org/simulations/simusoft_distcalc.html
   // Distcalc version 1.2 Copyright 2002 H Lohninger, TU Wein
   // H.Lohninger: Teach/Me Data Analysis, Springer-Verlag, Berlin-New York-Tokyo, 1999. ISBN 3-540-14743-8
   // The Windows calculator is available zipped distcalc.exe for download at:
   // http://www.vias.org/simulations/simu_stat.html

   // This interactive Windows program was used to find some combination for which the
   // result appears to be exact.  No doubt this can be done analytically too,
   // by mathematicians!

   // Some combinations for which the result is 'exact', or at least is to 40 decimal digits.
   // 40 decimal digits includes 128-bit significand User Defined Floating-Point types.
   // These all pass tests at near epsilon accuracy for the floating-point type.
   tolerance = boost::math::tools::epsilon<RealType>() * 5 * 100;
   cout << "Tolerance = " << tolerance << "%." << endl;
   BOOST_CHECK_CLOSE(
      cdf(fisher_f_distribution<RealType>(
         static_cast<RealType>(1.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(2.)/static_cast<RealType>(3.) ),  // F
      static_cast<RealType>(0.5), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(1.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(1.6L))),  // F
      static_cast<RealType>(0.333333333333333333333333333333333333L), // probability.
      tolerance * 100); // needs higher tolerance at 128-bit precision - value not exact?

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(1.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(6.5333333333333333333333333333333333L))),  // F
      static_cast<RealType>(0.125L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(1.))),  // F
      static_cast<RealType>(0.5L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(3.))),  // F
      static_cast<RealType>(0.25L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(3.))),  // F
      static_cast<RealType>(0.25L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(7.))),  // F
      static_cast<RealType>(0.125L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(9.))),  // F
      static_cast<RealType>(0.1L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(19.))),  // F
      static_cast<RealType>(0.05L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(29.))),  // F
      static_cast<RealType>(0.03333333333333333333333333333333333333333L), // probability.
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(99.))),  // F
      static_cast<RealType>(0.01L), // probability. 
      tolerance);

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(4.),  // df1
         static_cast<RealType>(4.)),  // df2
         static_cast<RealType>(9.))),  // F
      static_cast<RealType>(0.028L), // probability. 
      tolerance*10);   // not quite exact???

   BOOST_CHECK_CLOSE(
      cdf(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(8.),  // df1
         static_cast<RealType>(8.)),  // df2
         static_cast<RealType>(1.))),  // F
      static_cast<RealType>(0.5L), // probability. 
      tolerance);

// Inverse tests

      BOOST_CHECK_CLOSE(
      quantile(complement(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(0.03333333333333333333333333333333333333333L))),  // probability
      static_cast<RealType>(29.), // F expected.
      tolerance*10);

      BOOST_CHECK_CLOSE(
      quantile(fisher_f_distribution<RealType>(
         static_cast<RealType>(2.),  // df1
         static_cast<RealType>(2.)),  // df2
         static_cast<RealType>(1.0L - 0.03333333333333333333333333333333333333333L)),  // probability
      static_cast<RealType>(29.), // F expected.
      tolerance*10);


// Also note limit cases for F(1, infinity) == normal distribution
// F(1, n2) == Student's t distribution
// F(n1, infinity) == Chisq distribution

// These might allow some further cross checks?

    RealType tol2 = boost::math::tools::epsilon<RealType>() * 5 * 100;  // 5 eps as a percent
    cout << "Tolerance = " << tol2 << "%." << endl;
    fisher_f_distribution<RealType> dist(static_cast<RealType>(8), static_cast<RealType>(6));
    RealType x = 7;
    using namespace std; // ADL of std names.
    // mean:
    BOOST_CHECK_CLOSE(
       mean(dist)
       , static_cast<RealType>(6)/static_cast<RealType>(4), tol2);
    // variance:
    BOOST_CHECK_CLOSE(
       variance(dist)
       , static_cast<RealType>(2 * 6 * 6 * (8 + 6 - 2)) / static_cast<RealType>(8 * 16 * 2), tol2);
    // std deviation:
    BOOST_CHECK_CLOSE(
       standard_deviation(dist)
       , sqrt(static_cast<RealType>(2 * 6 * 6 * (8 + 6 - 2)) / static_cast<RealType>(8 * 16 * 2)), tol2);
    // hazard:
    BOOST_CHECK_CLOSE(
       hazard(dist, x)
       , pdf(dist, x) / cdf(complement(dist, x)), tol2);
    // cumulative hazard:
    BOOST_CHECK_CLOSE(
       chf(dist, x)
       , -log(cdf(complement(dist, x))), tol2);
    // coefficient_of_variation:
    BOOST_CHECK_CLOSE(
       coefficient_of_variation(dist)
       , standard_deviation(dist) / mean(dist), tol2);
    BOOST_CHECK_CLOSE(
       mode(dist)
       , static_cast<RealType>(6*6)/static_cast<RealType>(8*8), tol2);

    fisher_f_distribution<RealType> dist2(static_cast<RealType>(8), static_cast<RealType>(12));
    BOOST_CHECK_CLOSE(
       skewness(dist2)
       , static_cast<RealType>(26 * sqrt(64.0L)) / (12*6), tol2);
    BOOST_CHECK_CLOSE(
       kurtosis_excess(dist2)
       , static_cast<RealType>(6272) * 12 / 3456, tol2);
    BOOST_CHECK_CLOSE(
       kurtosis(dist2)
       , static_cast<RealType>(6272) * 12 / 3456 + 3, tol2);
    // special cases:
    BOOST_MATH_CHECK_THROW(
       pdf(
          fisher_f_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(1)),
          static_cast<RealType>(0)), std::overflow_error
       );
    BOOST_CHECK_EQUAL(
       pdf(fisher_f_distribution<RealType>(2, 2), static_cast<RealType>(0))
       , static_cast<RealType>(1.0f));
    BOOST_CHECK_EQUAL(
       pdf(fisher_f_distribution<RealType>(3, 3), static_cast<RealType>(0))
       , static_cast<RealType>(0.0f));
    BOOST_CHECK_EQUAL(
       cdf(fisher_f_distribution<RealType>(1, 1), static_cast<RealType>(0))
       , static_cast<RealType>(0.0f));
    BOOST_CHECK_EQUAL(
       cdf(fisher_f_distribution<RealType>(2, 2), static_cast<RealType>(0))
       , static_cast<RealType>(0.0f));
    BOOST_CHECK_EQUAL(
       cdf(fisher_f_distribution<RealType>(3, 3), static_cast<RealType>(0))
       , static_cast<RealType>(0.0f));
    BOOST_CHECK_EQUAL(
       cdf(complement(fisher_f_distribution<RealType>(1, 1), static_cast<RealType>(0)))
       , static_cast<RealType>(1));
    BOOST_CHECK_EQUAL(
       cdf(complement(fisher_f_distribution<RealType>(2, 2), static_cast<RealType>(0)))
       , static_cast<RealType>(1));
    BOOST_CHECK_EQUAL(
       cdf(complement(fisher_f_distribution<RealType>(3, 3), static_cast<RealType>(0)))
       , static_cast<RealType>(1));

    BOOST_MATH_CHECK_THROW(
       pdf(
          fisher_f_distribution<RealType>(-1, 2),
          static_cast<RealType>(1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       pdf(
          fisher_f_distribution<RealType>(1, -1),
          static_cast<RealType>(1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       pdf(
          fisher_f_distribution<RealType>(8, 2),
          static_cast<RealType>(-1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       cdf(
          fisher_f_distribution<RealType>(-1, 1),
          static_cast<RealType>(1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       cdf(
          fisher_f_distribution<RealType>(8, 4),
          static_cast<RealType>(-1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       cdf(complement(
          fisher_f_distribution<RealType>(-1, 2),
          static_cast<RealType>(1))), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       cdf(complement(
          fisher_f_distribution<RealType>(8, 4),
          static_cast<RealType>(-1))), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       quantile(
          fisher_f_distribution<RealType>(-1, 2),
          static_cast<RealType>(0.5)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       quantile(
          fisher_f_distribution<RealType>(8, 8),
          static_cast<RealType>(-1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       quantile(
          fisher_f_distribution<RealType>(8, 8),
          static_cast<RealType>(1.1)), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       quantile(complement(
          fisher_f_distribution<RealType>(2, -1),
          static_cast<RealType>(0.5))), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       quantile(complement(
          fisher_f_distribution<RealType>(8, 8),
          static_cast<RealType>(-1))), std::domain_error
       );
    BOOST_MATH_CHECK_THROW(
       quantile(complement(
          fisher_f_distribution<RealType>(8, 8),
          static_cast<RealType>(1.1))), std::domain_error
       );
   check_out_of_range<fisher_f_distribution<RealType> >(2, 3);
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{

  // Check that can generate fisher distribution using the two convenience methods:
   boost::math::fisher_f myf1(1., 2); // Using typedef
   fisher_f_distribution<> myf2(1., 2); // Using default RealType double.


  // Basic sanity-check spot values.

  // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float.
  test_spots(0.0); // Test double.
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output is:

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_fisher.exe"
Running 1 test case...
Tolerance = 0.2%.
Tolerance = 5.96046e-005%.
Tolerance = 5.96046e-005%.
Tolerance = 0.2%.
Tolerance = 1.11022e-013%.
Tolerance = 1.11022e-013%.
Tolerance = 0.2%.
Tolerance = 1.11022e-013%.
Tolerance = 1.11022e-013%.
Tolerance = 0.2%.
Tolerance = 1.11022e-013%.
Tolerance = 1.11022e-013%.
*** No errors detected

*/



