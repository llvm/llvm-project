//  Copyright John Maddock 2007.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/traits.hpp>
#include <boost/math/distributions.hpp>

using namespace boost::math;

static_assert(::boost::math::tools::is_distribution<double>::value == false, "double is erroneously identified as a distribution");
static_assert(::boost::math::tools::is_distribution<int>::value == false, "int is erroneously identified as a distribution");
static_assert(::boost::math::tools::is_distribution<bernoulli>::value, "bernoulli distribution should be identified as a distribution");
static_assert(::boost::math::tools::is_distribution<beta_distribution<> >::value, "beta distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<binomial>::value, "binomial distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<cauchy>::value, "cauchy distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<chi_squared>::value, "chi squared distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<exponential>::value, "exponential distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<extreme_value>::value, "extreme value distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<fisher_f>::value, "fisher f distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<gamma_distribution<> >::value, "gamma distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<lognormal>::value, "lognormal distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<negative_binomial>::value, "negative distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<normal>::value, "normal distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<pareto>::value, "pareto distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<poisson>::value, "poisson distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<rayleigh>::value, "rayleigh distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<students_t>::value, "students t distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<triangular>::value, "triangular distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<uniform>::value, "uniform distribution should be identified a distribution");
static_assert(::boost::math::tools::is_distribution<weibull>::value, "weibull distribution should be identified a distribution");

static_assert(::boost::math::tools::is_scaled_distribution<double>::value == false, "double is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<int>::value == false, "int double is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<bernoulli>::value == false, "bernoulli distribution is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<beta_distribution<> >::value == false, "beta distribution is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<binomial>::value == false, "binomial distribution should be identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<cauchy>::value, "cauchy distribution should be identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<chi_squared>::value == false, "chi squared is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<exponential>::value == false, "exponential distribution is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<extreme_value>::value, "extreme value distribution should be identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<fisher_f>::value == false, "fisher f is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<gamma_distribution<> >::value == false, "gamma distribution is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<lognormal>::value, "lognormal distribution should be identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<negative_binomial>::value == false, "negative binomial is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<normal>::value, "normal distribution should be identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<pareto>::value == false, "pareto is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<poisson>::value == false, "poisson is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<rayleigh>::value == false, "rayleigh is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<students_t>::value == false, "students t is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<triangular>::value == false, "triangular is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<uniform>::value == false, "uniform distribution is erroneously identified as a scaled distribution");
static_assert(::boost::math::tools::is_scaled_distribution<weibull>::value == false, "weibull distribution is erroneously identified as a scaled distribution");
