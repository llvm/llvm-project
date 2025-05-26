//  Copyright Paul A. Bristow 2013.
//  Copyright Nakhar Agrawal 2013.
//  Copyright John Maddock 2013.
//  Copyright Christopher Kormanyos 2013.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma warning (disable : 4100) // unreferenced formal parameter.
#pragma warning (disable : 4127) // conditional expression is constant.

//#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/bernoulli.hpp>

#include <iostream>

/* First 50 from 2 to 100 inclusive: */
/* TABLE[N[BernoulliB[n], 200], {n,2,100,2}] */

//SC_(0.1666666666666666666666666666666666666666), 
//SC_(-0.0333333333333333333333333333333333333333), 
//SC_(0.0238095238095238095238095238095238095238), 
//SC_(-0.0333333333333333333333333333333333333333), 
//SC_(0.0757575757575757575757575757575757575757), 
//SC_(-0.2531135531135531135531135531135531135531), 
//SC_(1.1666666666666666666666666666666666666666), 
//SC_(-7.0921568627450980392156862745098039215686), 
//SC_(54.9711779448621553884711779448621553884711), 

int main()
{
  //[bernoulli_example_1

/*`A simple example computes the value of B[sub 4] where the return type is `double`,
note that the argument to bernoulli_b2n is ['2] not ['4] since it computes B[sub 2N].


*/ 
  try
  { // It is always wise to use try'n'catch blocks around Boost.Math functions
    // so that any informative error messages can be displayed in the catch block.
  std::cout
    << std::setprecision(std::numeric_limits<double>::digits10)
    << boost::math::bernoulli_b2n<double>(2) << std::endl;

/*`So B[sub 4] == -1/30 == -0.0333333333333333 

If we use Boost.Multiprecision and its 50 decimal digit floating-point type `cpp_dec_float_50`,
we can calculate the value of much larger numbers like B[sub 200]
and also obtain much higher precision.
*/

  std::cout
    << std::setprecision(std::numeric_limits<boost::multiprecision::cpp_dec_float_50>::digits10)
    << boost::math::bernoulli_b2n<boost::multiprecision::cpp_dec_float_50>(100) << std::endl;
 
//] //[/bernoulli_example_1]

//[bernoulli_example_2
/*`We can compute and save all the float-precision Bernoulli numbers from one call.
*/
  std::vector<float> bn; // Space for 32-bit `float` precision Bernoulli numbers.

  // Start with Bernoulli number 0.
  boost::math::bernoulli_b2n<float>(0, 32, std::back_inserter(bn)); // Fill vector with even Bernoulli numbers.

  for(size_t i = 0; i < bn.size(); i++)
  { // Show vector of even Bernoulli numbers, showing all significant decimal digits.
      std::cout << std::setprecision(std::numeric_limits<float>::digits10)
          << i*2 << ' '           
          << bn[i]
          << std::endl;
  }
//] //[/bernoulli_example_2]

  }
  catch(const std::exception& ex)
  {
     std::cout << "Thrown Exception caught: " << ex.what() << std::endl;
  }


//[bernoulli_example_3    
/*`Of course, for any floating-point type, there is a maximum Bernoulli number that can be computed
  before it overflows the exponent.
  By default policy, if we try to compute too high a Bernoulli number, an exception will be thrown.
*/
  try
  {
    std::cout
    << std::setprecision(std::numeric_limits<float>::digits10)
    << "Bernoulli number " << 33 * 2 <<std::endl;

    std::cout << boost::math::bernoulli_b2n<float>(33) << std::endl;
  }
  catch (std::exception ex)
  {
    std::cout << "Thrown Exception caught: " << ex.what() << std::endl;
  }

/*`
and we will get a helpful error message (provided try'n'catch blocks are used).
*/

//] //[/bernoulli_example_3]

//[bernoulli_example_4
/*For example:
*/
   std::cout << "boost::math::max_bernoulli_b2n<float>::value = "  << boost::math::max_bernoulli_b2n<float>::value << std::endl;
   std::cout << "Maximum Bernoulli number using float is " << boost::math::bernoulli_b2n<float>( boost::math::max_bernoulli_b2n<float>::value) << std::endl;
   std::cout << "boost::math::max_bernoulli_b2n<double>::value = "  << boost::math::max_bernoulli_b2n<double>::value << std::endl;
   std::cout << "Maximum Bernoulli number using double is " << boost::math::bernoulli_b2n<double>( boost::math::max_bernoulli_b2n<double>::value) << std::endl;
  //] //[/bernoulli_example_4]

//[tangent_example_1

/*`We can compute and save a few Tangent numbers.
*/
  std::vector<float> tn; // Space for some `float` precision Tangent numbers.

  // Start with Bernoulli number 0.
  boost::math::tangent_t2n<float>(1, 6, std::back_inserter(tn)); // Fill vector with even Tangent numbers.

  for(size_t i = 0; i < tn.size(); i++)
  { // Show vector of even Tangent numbers, showing all significant decimal digits.
      std::cout << std::setprecision(std::numeric_limits<float>::digits10)
          << " "
          << tn[i];
  }
  std::cout << std::endl;

//] [/tangent_example_1]

// 1, 2, 16, 272, 7936, 353792, 22368256, 1903757312 



} // int main()

/*

//[bernoulli_output_1
  -3.6470772645191354362138308865549944904868234686191e+215
//] //[/bernoulli_output_1]

//[bernoulli_output_2

  0 1
  2 0.166667
  4 -0.0333333
  6 0.0238095
  8 -0.0333333
  10 0.0757576
  12 -0.253114
  14 1.16667
  16 -7.09216
  18 54.9712
  20 -529.124
  22 6192.12
  24 -86580.3
  26 1.42552e+006
  28 -2.72982e+007
  30 6.01581e+008
  32 -1.51163e+010
  34 4.29615e+011
  36 -1.37117e+013
  38 4.88332e+014
  40 -1.92966e+016
  42 8.41693e+017
  44 -4.03381e+019
  46 2.11507e+021
  48 -1.20866e+023
  50 7.50087e+024
  52 -5.03878e+026
  54 3.65288e+028
  56 -2.84988e+030
  58 2.38654e+032
  60 -2.14e+034
  62 2.0501e+036
//] //[/bernoulli_output_2]

//[bernoulli_output_3
 Bernoulli number 66
 Thrown Exception caught: Error in function boost::math::bernoulli_b2n<float>(n):
 Overflow evaluating function at 33
//] //[/bernoulli_output_3]
//[bernoulli_output_4
  boost::math::max_bernoulli_b2n<float>::value = 32
  Maximum Bernoulli number using float is -2.0938e+038
  boost::math::max_bernoulli_b2n<double>::value = 129
  Maximum Bernoulli number using double is 1.33528e+306
//] //[/bernoulli_output_4]

  
//[tangent_output_1
   1 2 16 272 7936 353792
//] [/tangent_output_1]



*/


