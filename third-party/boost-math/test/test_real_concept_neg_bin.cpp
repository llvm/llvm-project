// test_real_concept.cpp

// Copyright Paul A. Bristow 2010.
// Copyright John Maddock 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tests real_concept for Negative Binomial and Geometric Distribution.
// find_upper_bound ...

#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;

#include <boost/math/distributions/geometric.hpp> // for geometric_distribution
using boost::math::geometric_distribution;
using boost::math::geometric; // using typedef for geometric_distribution<double> 

#include <boost/math/distributions/negative_binomial.hpp> // for some comparisons.

#include <iostream>
using std::cout;
using std::endl;
using std::setprecision;
using std::showpoint;
#include <limits>
using std::numeric_limits;

template <class RealType>
void test_spot(RealType)
{
    using boost::math::negative_binomial_distribution;

    // NOT boost::math::negative_binomial or boost::math::geometric 
    // - because then you get the default negative_binomial_distribution<double>!!!

    RealType k = static_cast<RealType>(2.L);
    RealType alpha = static_cast<RealType>(0.05L);
    RealType p = static_cast<RealType>(0.5L);
    RealType result;
    result = negative_binomial_distribution<RealType>::find_lower_bound_on_p(static_cast<RealType>(k), static_cast<RealType>(1), static_cast<RealType>(alpha));
    result = negative_binomial_distribution<RealType>::find_lower_bound_on_p(k, 1, alpha);
    result = geometric_distribution<RealType>::find_lower_bound_on_p(k, alpha);
}

int main()
{
  test_spot(boost::math::concepts::real_concept(0.)); // Test real concept.
  return 0;
} 
