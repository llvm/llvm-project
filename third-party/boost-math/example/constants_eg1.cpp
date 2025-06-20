// Copyright Paul Bristow 2013.
// Copyright John Maddock 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/*! \brief Examples of using the enhanced math constants.
    \details This allows for access to constants via functions like @c pi(),
    and also via namespaces, @c using @c namespace boost::math::double_constants;
    called simply @c pi.
*/

#include <boost/math/constants/constants.hpp>

#include <iostream>
using std::cout;
using std::endl;

#include <limits>
using std::numeric_limits;

/*! \brief Examples of a template function using constants.
    \details This example shows using of constants from function calls like @c pi(),
    rather than the 'cute' plain @c pi use in non-template applications.

    \tparam Real radius parameter that can be a built-in like float, double,
      or a user-defined type like multiprecision.
    \returns Area = pi * radius ^ 2
*/

//[math_constants_eg1
template<class Real>
Real area(Real r)
{
  using namespace boost::math::constants;

  return pi<Real>() * r * r;
}
//] [/math_constants_eg1]

int main()
{

  { // Boost.Math constants using function calls like pi().
    // using namespace boost::math::constants;

   using boost::math::constants::pi;
   using boost::math::constants::one_div_two_pi;

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
     std::size_t max_digits10 = 2 + std::numeric_limits<double>::digits * 3010/10000;
#else
   std::size_t max_digits10 = std::numeric_limits<double>::max_digits10;
#endif

  std::cout.precision(max_digits10);
  cout << "double pi =  boost::math::double_constants::pi = " << pi<double>() << endl;
  //   double pi =  boost::math::double_constants::pi = 3.1415926535897931
  double r = 1.234567890123456789;
  double d = pi<double>() * r * r;

  cout << "d = " << d << ", r = " << r << endl;

  float rf = 0.987654321987654321f;

  float pif = boost::math::constants::pi<float>();
  cout << "pidf =  boost::math::constants::pi() = " << pif << endl;
  //   pidf =  boost::math::float_constants::pi = 3.1415927410125732

    //float df = pi * rf * rf; // conversion from 'const double' to 'float', possible loss of data.
  float df = pif * rf * rf;

  cout << "df = " << df << ", rf = " << rf << endl;

  cout << "one_div_two_pi " << one_div_two_pi<double>() << endl;

  using boost::math::constants::one_div_two_pi;

  cout << "one_div_root_two_pi " << one_div_two_pi<double>() << endl;
  }

  { // Boost math new constants using namespace selected values, like pi.

    //using namespace boost::math::float_constants;
    using namespace boost::math::double_constants;

    double my2pi = two_pi; // Uses boost::math::double_constants::two_pi;

    cout << "double my2pi = " << my2pi << endl;

    using boost::math::float_constants::e;
    float my_e = e;
    cout << "float my_e  " << my_e << endl;

    double my_pi = boost::math::double_constants::pi;
    cout << "double my_pi = boost::math::double_constants::pi =  " << my_pi << endl;

    // If you try to use two namespaces, this may, of course, create ambiguity:
    // it is not too difficult to do this inadvertently.
    using namespace boost::math::float_constants;
    //cout << pi << endl; // error C2872: 'pi' : ambiguous symbol.

  }
  {

//[math_constants_ambiguity
     // If you use more than one namespace, this will, of course, create ambiguity:
     using namespace boost::math::double_constants;
     using namespace boost::math::constants;

      //double my_pi = pi(); // error C2872: 'pi' : ambiguous symbol
      //double my_pi2 = pi; // Context does not allow for disambiguation of overloaded function

     // It is also possible to create ambiguity inadvertently,
     // perhaps in other peoples code,
     // by making the scope of a namespace declaration wider than necessary,
     // therefore is it prudent to avoid this risk by localising the scope of such definitions.
//] [/math_constants_ambiguity]

  }

  { // You can, of course, use both methods of access if both are fully qualified, for examples:

    //cout.precision(std::numeric_limits<double>::max_digits10);// Ideally.
    cout.precision(2 + std::numeric_limits<double>::digits * 3010/10000); // If no max_digits10.

    double my_pi1 = boost::math::constants::pi<double>();
    double my_pid = boost::math::double_constants::pi;
    cout << "boost::math::constants::pi<double>() = " << my_pi1 << endl
         << "boost::math::double_constants::pi = " << my_pid << endl;

    // cout.precision(std::numeric_limits<float>::max_digits10); // Ideally.
    cout.precision(2 + std::numeric_limits<double>::digits * 3010/10000); // If no max_digits10.
    float my_pif = boost::math::float_constants::pi;
    cout << "boost::math::float_constants::pi = " << my_pif << endl;

  }

  { // Use with templates

    // \warning it is important to be very careful with the type provided as parameter.
    // For example, naively providing an @b integer instead of a floating-point type can be disastrous.
    // cout << "Area = " << area(2) << endl; // warning : 'return' : conversion from 'double' to 'int', possible loss of data
    // Failure to heed this warning can lead to very wrong answers!
    //  Area = 12 !!  = 3 * 2 * 2
//[math_constants_template_integer_type
    //cout << "Area = " << area(2) << endl; //   Area = 12!
    cout << "Area = " << area(2.) << endl;  //   Area = 12.566371

    // You can also avoid this by being explicit about the type of @c area.
    cout << "Area = " << area<double>(2) << endl;

//] [/math_constants_template_integer_type]


  }
/*
{
    using  boost::math::constants::pi;
    //double my_pi3 = pi<double>(); // OK
    //double my_pi4 = pi<>(); cannot find template type.
    //double my_pi4 = pi(); // Can't find a function.

  }
*/

} // int main()

/*[constants_eq1_output

Output:

  double pi =  boost::math::double_constants::pi = 3.1415926535897931
  d = 4.7882831840285398, r = 1.2345678901234567
  pidf =  boost::math::constants::pi() = 3.1415927410125732
  df = 3.0645015239715576, rf = 0.98765432834625244
  one_div_two_pi 0.15915494309189535
  one_div_root_two_pi 0.15915494309189535
  double my2pi = 6.2831853071795862
  float my_e  2.7182817459106445
  double my_pi = boost::math::double_constants::pi =  3.1415926535897931
  boost::math::constants::pi<double>() = 3.1415926535897931
  boost::math::double_constants::pi = 3.1415926535897931
  boost::math::float_constants::pi = 3.1415927410125732
  Area = 12.566370614359172
  Area = 12.566370614359172


] [/constants_eq1_output]
*/
