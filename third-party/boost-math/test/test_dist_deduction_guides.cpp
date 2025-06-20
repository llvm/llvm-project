//  (C) Copyright John Maddock 2022.
//  (C) Copyright James Folberth 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Issue 754
// Check that the class template argument deduction guides properly promote
// integral ctor args to a real floating point type.

#include <boost/math/distributions/arcsine.hpp>
#include <boost/math/distributions/bernoulli.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/cauchy.hpp>
#include <boost/math/distributions/chi_squared.hpp>
//#include <boost/math/distributions/empirical_cumulative_distribution_function.hpp>
#include <boost/math/distributions/exponential.hpp>
#include <boost/math/distributions/extreme_value.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/geometric.hpp>
//#include <boost/math/distributions/hyperexponential.hpp>
//#include <boost/math/distributions/hypergeometric.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>
#include <boost/math/distributions/inverse_gaussian.hpp>
#include <boost/math/distributions/kolmogorov_smirnov.hpp>
#include <boost/math/distributions/laplace.hpp>
#include <boost/math/distributions/logistic.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/non_central_beta.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/distributions/non_central_f.hpp>
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/pareto.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/rayleigh.hpp>
#include <boost/math/distributions/skew_normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/triangular.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/weibull.hpp>

// Instantiate a DistType object with the parameter pack given by Types.
// Then verify that the `RealType` template parameter of DistType (stored in
// in value_type) is promoted correctly according to the deduction guides.
template <template<class, class> class DistType, class PromType = double, class... Types>
void test_deduction_guide(Types... types)
{
   DistType d(types...);
   static_assert(std::is_same<typename decltype(d)::value_type, PromType>::value);
}

int main()
{
   using namespace boost::math;

   test_deduction_guide<arcsine_distribution>(0);
   test_deduction_guide<arcsine_distribution>(0, 1);

   test_deduction_guide<bernoulli_distribution>(0);

   test_deduction_guide<beta_distribution>(1);
   test_deduction_guide<beta_distribution>(1, 1);

   test_deduction_guide<binomial_distribution>(1);
   test_deduction_guide<binomial_distribution>(1, 0);

   test_deduction_guide<cauchy_distribution>(0);
   test_deduction_guide<cauchy_distribution>(0, 1);

   test_deduction_guide<chi_squared_distribution>(2);

   test_deduction_guide<exponential_distribution>(1);

   test_deduction_guide<extreme_value_distribution>(0);
   test_deduction_guide<extreme_value_distribution>(0, 1);

   test_deduction_guide<fisher_f_distribution>(1, 2);

   test_deduction_guide<gamma_distribution>(1);
   test_deduction_guide<gamma_distribution>(1, 1);

   test_deduction_guide<geometric_distribution>(1);

   test_deduction_guide<inverse_chi_squared_distribution>(1);
   test_deduction_guide<inverse_chi_squared_distribution>(1, 1);

   test_deduction_guide<inverse_gamma_distribution>(1);
   test_deduction_guide<inverse_gamma_distribution>(1, 1);

   test_deduction_guide<inverse_gaussian_distribution>(1);
   test_deduction_guide<inverse_gaussian_distribution>(1, 1);

   test_deduction_guide<kolmogorov_smirnov_distribution>(1);

   test_deduction_guide<laplace_distribution>(0);
   test_deduction_guide<laplace_distribution>(0, 1);

   test_deduction_guide<logistic_distribution>(0);
   test_deduction_guide<logistic_distribution>(0, 1);

   test_deduction_guide<lognormal_distribution>(0);
   test_deduction_guide<lognormal_distribution>(0, 1);

   test_deduction_guide<negative_binomial_distribution>(1, 1);

   test_deduction_guide<non_central_beta_distribution>(1, 1, 1);

   test_deduction_guide<non_central_chi_squared_distribution>(1, 1);

   test_deduction_guide<non_central_f_distribution>(1, 1, 1);

   test_deduction_guide<non_central_t_distribution>(1, 1);

   test_deduction_guide<normal_distribution>(2);
   test_deduction_guide<normal_distribution>(2, 3);

   test_deduction_guide<pareto_distribution>(2);
   test_deduction_guide<pareto_distribution>(2, 3);

   test_deduction_guide<poisson_distribution>(1);

   test_deduction_guide<rayleigh_distribution>(1);

   test_deduction_guide<skew_normal_distribution>(0);
   test_deduction_guide<skew_normal_distribution>(0, 1);
   test_deduction_guide<skew_normal_distribution>(0, 1, 0);

   test_deduction_guide<students_t_distribution>(2);

   test_deduction_guide<triangular_distribution>(-1);
   test_deduction_guide<triangular_distribution>(-1, 0);
   test_deduction_guide<triangular_distribution>(-1, 0, 1);

   test_deduction_guide<uniform_distribution>(0);
   test_deduction_guide<uniform_distribution>(0, 1);

   test_deduction_guide<weibull_distribution>(1);
   test_deduction_guide<weibull_distribution>(1, 1);
}
