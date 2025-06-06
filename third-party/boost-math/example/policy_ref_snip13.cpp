//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <boost/config.hpp>
#ifdef _MSC_VER
#  pragma warning (disable : 4189) //  'd' : local variable is initialized but not referenced
#endif
#ifdef BOOST_GCC
#  pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#include <iostream>
using std::cout; using std::endl;

#include <stdexcept>
using std::domain_error;

//[policy_ref_snip13

#include <boost/math/distributions/cauchy.hpp>

namespace myspace
{ // using namespace boost::math::policies; // May be convenient in myspace.

  // Define a policy called my_policy to use.
  using boost::math::policies::policy;
  
// In this case we want all the distribution accessor functions to compile,
// even if they are mathematically undefined, so
// make the policy assert_undefined.
  using boost::math::policies::assert_undefined;

typedef policy<assert_undefined<false> > my_policy;

// Finally apply this policy to type double.
BOOST_MATH_DECLARE_DISTRIBUTIONS(double, my_policy)
} // namespace myspace

// Now we can use myspace::cauchy etc, which will use policy
// myspace::mypolicy:
//
// This compiles but throws a domain error exception at runtime.
// Caution! If you omit the try'n'catch blocks, 
// it will just silently terminate, giving no clues as to why! 
// So try'n'catch blocks are very strongly recommended.

void test_cauchy()
{
   try
   {
      double d = mean(myspace::cauchy());  // Cauchy does not have a mean!
      (void) d;
   }
   catch(const std::domain_error& e)
   {
      cout << e.what() << endl;
   }
}

//] //[/policy_ref_snip13]

int main()
{
   test_cauchy();
}

/*

Output:

policy_snip_13.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\policy_snip_13.exe
  Error in function boost::math::mean(cauchy<double>&): The Cauchy distribution does not have a mean: the only possible return value is 1.#QNAN.

  */

