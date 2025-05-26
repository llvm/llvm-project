//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[policy_ref_snip1

#include <boost/math/special_functions/gamma.hpp>
using boost::math::tgamma;

//using namespace boost::math::policies; may also be convenient.
using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;

// Define a policy:
typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
> my_policy;

double my_value = 0.; // 

// Call the function applying my_policy:
double t1 = tgamma(my_value, my_policy());

// Alternatively (and equivalently) we could use helpful function
// make_policy and define everything at the call site:
double t2 = tgamma(my_value,
  make_policy(
    domain_error<errno_on_error>(), 
    pole_error<errno_on_error>(),
    overflow_error<errno_on_error>(),
    evaluation_error<errno_on_error>() )
  );
//]

#include <iostream>
using std::cout; using std::endl;

int main()
{
  cout << "my_value = " << my_value << endl;
  try
  { // First with default policy - throw an exception.
     cout << "tgamma(my_value) = " << tgamma(my_value) << endl; 
  }
  catch(const std::exception& e)
  {
     cout <<"\n""Message from thrown exception was:\n   " << e.what() << endl;
  }

  cout << "tgamma(my_value, my_policy() = " << t1 << endl;
  cout << "tgamma(my_value, make_policy(domain_error<errno_on_error>(), pole_error<errno_on_error>(), overflow_error<errno_on_error>(), evaluation_error<errno_on_error>() ) = " << t2 << endl;
}

/*
Output:
  my_value = 0
  
  Message from thrown exception was:
     Error in function boost::math::tgamma<long double>(long double): Evaluation of tgamma at a negative integer 0.
  tgamma(my_value, my_policy() = 1.#QNAN
  tgamma(my_value, make_policy(domain_error<errno_on_error>(), pole_error<errno_on_error>(), overflow_error<errno_on_error>(), evaluation_error<errno_on_error>() ) = 1.#QNAN
*/
