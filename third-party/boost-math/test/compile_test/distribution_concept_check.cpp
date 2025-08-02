//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS
#include <boost/math/distributions.hpp>
#include <boost/math/concepts/distributions.hpp>

template <class RealType>
void instantiate(RealType)
{
   using namespace boost;
   using namespace boost::math;
   using namespace boost::math::concepts;

   typedef policies::policy<policies::digits2<std::numeric_limits<RealType>::digits - 2> > custom_policy;

   function_requires<DistributionConcept<bernoulli_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<beta_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<binomial_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<cauchy_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<chi_squared_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<exponential_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<extreme_value_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<fisher_f_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<gamma_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<geometric_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<hypergeometric_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<hypergeometric_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<inverse_chi_squared_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<inverse_gamma_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<inverse_gaussian_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<kolmogorov_smirnov_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<laplace_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<logistic_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<lognormal_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<negative_binomial_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<non_central_beta_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<non_central_chi_squared_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<non_central_f_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<non_central_t_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<normal_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<pareto_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<poisson_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<rayleigh_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<skew_normal_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<students_t_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<triangular_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<uniform_distribution<RealType, custom_policy> > >();
   function_requires<DistributionConcept<weibull_distribution<RealType, custom_policy> > >();
}

#else // Standalone mode

template <typename T>
void instantiate(T) {}

#endif 

int main()
{
   instantiate(float(0));
   instantiate(double(0));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   instantiate((long double)(0));
#endif
}

