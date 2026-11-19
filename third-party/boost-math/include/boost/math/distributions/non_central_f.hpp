// boost\math\distributions\non_central_f.hpp

// Copyright John Maddock 2008.
// Copyright Matt Borland 2024.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_SPECIAL_NON_CENTRAL_F_HPP
#define BOOST_MATH_SPECIAL_NON_CENTRAL_F_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/distributions/non_central_beta.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/distributions/detail/generic_mode.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/distributions/complement.hpp> // complements

namespace boost
{
   namespace math
   {
      namespace detail
      {
         template <class RealType, class Policy>
         struct non_centrality_finder_f
         {
            BOOST_MATH_GPU_ENABLED non_centrality_finder_f(const RealType x_, const RealType v1_, const RealType v2_, const RealType p_, const bool c)
               : x(x_), v1(v1_), v2(v2_), p(p_), comp(c) {}

            BOOST_MATH_GPU_ENABLED RealType operator()(RealType nc) const
            {
               non_central_f_distribution<RealType, Policy> d(v1, v2, nc);
               return comp ?
                  RealType(p - cdf(complement(d, x)))
                  : RealType(cdf(d, x) - p);
            }
         private:
            RealType x, v1, v2, p;
            bool comp; 
         };
         
         template <class RealType, class Policy>
         BOOST_MATH_GPU_ENABLED RealType find_non_centrality_f(const RealType x, const RealType v1, const RealType v2, const RealType p, const RealType q, const RealType p_q_precision, const Policy& pol)         
         {
            constexpr auto function = "non_central_f<%1%>::find_non_centrality";

            if ( p == 0 || q == 0) {
               return policies::raise_domain_error<RealType>(function, "Can't find non centrality parameter when the probability is <=0 or >=1, only possible answer is %1%", // LCOV_EXCL_LINE
                  RealType(boost::math::numeric_limits<RealType>::quiet_NaN()), Policy()); // LCOV_EXCL_LINE
            }

            // Check if nc = 0 (which is just the F-distribution)
            non_centrality_finder_f<RealType, Policy> f(x, v1, v2, p < q ? p : q, p < q ? false : true);
            // This occurs when the root finder would need to find a result smaller than
            // tools::min_value (which it cannot do).  Note that we have to add in a small
            // amount of "tolerance" since the subtraction in our termination condition
            // implies a small amount of wobble in the result which should be of the
            // order p * eps (note not q * eps, since q is calculated as 1-p).
            // Also note that p_q_precision is passed down from our caller as the
            // epsilon of the original called values, and not after possible promotion.
            if (f(tools::min_value<RealType>()) <= 20 * p_q_precision * p){
               return 0;
            }

            RealType guess = RealType(10);                       // Starting guess.
            RealType factor = RealType(2);                       // How big steps to take when searching.
            boost::math::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
            tools::eps_tolerance<RealType> tol(policies::digits<RealType, Policy>());

            boost::math::pair<RealType, RealType> result_bracket = tools::bracket_and_solve_root(
                                 f, guess, factor, false, tol, max_iter, pol);
            
            RealType result = result_bracket.first + (result_bracket.second - result_bracket.first)/2;
            if (max_iter >= policies::get_max_root_iterations<Policy>()) {
               return policies::raise_evaluation_error<RealType>(function, "Unable to locate solution in a reasonable time:" // LCOV_EXCL_LINE
                  " or there is no answer to problem.  Current best guess is %1%", result, Policy()); // LCOV_EXCL_LINE
            }
            return result;
         }
         
         template <class RealType, class Policy>
         struct f_degrees_of_freedom_finder
         {
            f_degrees_of_freedom_finder(
               RealType x_, RealType v_, RealType nc_, bool find_v1_, RealType p_, bool c)
               : x(x_), v(v_), nc(nc_), find_v1(find_v1_), p(p_), comp(c) {}

            RealType operator()(const RealType& input_v)
            {
               RealType v1 = find_v1 ? input_v : v; 
               RealType v2 = find_v1 ? v : input_v; 
               non_central_f_distribution<RealType, Policy> d(v1, v2, nc);
               return comp ?
                  p - cdf(complement(d, x))
                  : cdf(d, x) - p;
            }
         private:
            RealType x;
            RealType v;
            RealType nc;
            bool find_v1;
            RealType p;
            bool comp;
         };

         template <class RealType, class Policy>
         RealType large_v2_approximation(RealType x, RealType v1, RealType p, RealType q, RealType nc)
         { // For v2 -> inf approximate f_degreese_of_freedome_finder with chi squared distribution 
           // with degrees of freedom v1 at the cdf at x * v1
               bool comp = p < q ? false : true;
               RealType pval =  p < q ? p : q;
               non_central_chi_squared_distribution<RealType, Policy> d(v1, nc);
               return comp ? pval - cdf(complement(d, x*v1)) : cdf(d, x*v1) - pval;
         }

         template <class RealType, class Policy>
         inline RealType find_degrees_of_freedom_f(
            const RealType x, const RealType v, const RealType nc, const bool find_v1, const RealType p, const RealType q, const Policy& pol)
         {
            BOOST_MATH_STD_USING
            using std::fabs;
            const char* function = "non_central_f<%1%>::find_degrees_of_freedom_f";
            if((p == 0) || (q == 0))
            {
               //
               // Can't find a thing if one of p and q is zero:
               //
               return policies::raise_domain_error<RealType>(function, "Can't find degrees of freedom when the probability is 0 or 1, only possible answer is %1%",
                  RealType(std::numeric_limits<RealType>::quiet_NaN()), Policy()); // LCOV_EXCL_LINE
            }
            if (x < tools::epsilon<RealType>())
            {
               return policies::raise_evaluation_error<RealType>(function, "Can't find degrees of freedom when the abscissa value is very close to zero as all degrees of freedom generate the same CDF at x=0: try again further out in the tails!!",
                  RealType(std::numeric_limits<RealType>::quiet_NaN()), Policy()); // LCOV_EXCL_LINE
            }
            f_degrees_of_freedom_finder<RealType, Policy> f(x, v, nc, find_v1, p < q ? p : q, p < q ? false : true);

            // There are times when f has two roots - thus, two degrees of freedom can
            // be found. We find this case by checking if the sign of f for large and 
            // small values of v have the same sign. If the sign is the same, then there 
            // are an even number of roots. If the signs differ, there is only one root
            // and we can safely find the root.
            RealType vLarge = sqrt(boost::math::tools::max_value<RealType>());
            RealType vSmall = 1 / vLarge;

            // As v2 -> infinity, noncentral f converges to chi-squared distribution.
            // Rather than evaluating f(vLarge), we can use the more stable chi-squared approximation
            RealType large_difference;
            if (find_v1)
            {
               large_difference = large_v2_approximation<RealType, Policy>(x, v, p, q, nc);
            }
            else
               large_difference = f(vLarge);

            if ((large_difference < 0) == (f(vSmall) < 0)){
               return policies::raise_evaluation_error<RealType>(function, "Can't find degrees of freedom because two degrees of freedom can be found using the given parameters",
                  RealType(std::numeric_limits<RealType>::quiet_NaN()), Policy()); // LCOV_EXCL_LINE 
            }

            tools::eps_tolerance<RealType> tol(policies::digits<RealType, Policy>());
            std::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
            //
            // Pick an initial guess:
            //
            RealType guess = 1;
            std::pair<RealType, RealType> ir = tools::bracket_and_solve_root(
               f, guess, RealType(2), f(guess) < 0 ? true : false, tol, max_iter, pol);
            RealType result = ir.first + (ir.second - ir.first) / 2;
            if(max_iter >= policies::get_max_root_iterations<Policy>())
            {
               return policies::raise_evaluation_error<RealType>(function, "Unable to locate solution in a reasonable time:" // LCOV_EXCL_LINE
                  " or there is no answer to problem. Current best guess is %1%", result, Policy()); // LCOV_EXCL_LINE
            }
            return result;
         }
      } // namespace detail

      template <class RealType = double, class Policy = policies::policy<> >
      class non_central_f_distribution
      {
      public:
         typedef RealType value_type;
         typedef Policy policy_type;

         BOOST_MATH_GPU_ENABLED non_central_f_distribution(RealType v1_, RealType v2_, RealType lambda) : v1(v1_), v2(v2_), ncp(lambda)
         {
            constexpr auto function = "boost::math::non_central_f_distribution<%1%>::non_central_f_distribution(%1%,%1%)";
            RealType r;
            detail::check_df(
               function,
               v1, &r, Policy());
            detail::check_df(
               function,
               v2, &r, Policy());
            detail::check_non_centrality(
               function,
               lambda,
               &r,
               Policy());
         } // non_central_f_distribution constructor.

         BOOST_MATH_GPU_ENABLED RealType degrees_of_freedom1()const
         {
            return v1;
         }
         BOOST_MATH_GPU_ENABLED RealType degrees_of_freedom2()const
         {
            return v2;
         }
         BOOST_MATH_GPU_ENABLED RealType non_centrality() const
         { // Private data getter function.
            return ncp;
         }
         BOOST_MATH_GPU_ENABLED static RealType find_non_centrality(const RealType x, const RealType v1, const RealType v2, const RealType p)
         {
            constexpr auto function = "non_central_f_distribution<%1%>::find_non_centrality";
            typedef typename policies::evaluation<RealType, Policy>::type eval_type;
            typedef typename policies::normalise<
               Policy,
               policies::promote_float<false>,
               policies::promote_double<false>,
               policies::discrete_quantile<>,
               policies::assert_undefined<> >::type forwarding_policy;
            eval_type result = detail::find_non_centrality_f(
               static_cast<eval_type>(x),
               static_cast<eval_type>(v1),
               static_cast<eval_type>(v2),
               static_cast<eval_type>(p),
               static_cast<eval_type>(1-p),
               static_cast<eval_type>(tools::epsilon<RealType>()),
               forwarding_policy());
            return policies::checked_narrowing_cast<RealType, forwarding_policy>(
               result,
               function);
         }
         template <class A, class B, class C, class D>
         BOOST_MATH_GPU_ENABLED static RealType find_non_centrality(const complemented4_type<A,B,C, D>& c)
         {
            constexpr auto function = "non_central_f_distribution<%1%>::find_non_centrality";
            typedef typename policies::evaluation<RealType, Policy>::type eval_type;
            typedef typename policies::normalise<
               Policy,
               policies::promote_float<false>,
               policies::promote_double<false>,
               policies::discrete_quantile<>,
               policies::assert_undefined<> >::type forwarding_policy;
            eval_type result = detail::find_non_centrality_f(
               static_cast<eval_type>(c.dist),
               static_cast<eval_type>(c.param1),
               static_cast<eval_type>(c.param2),
               static_cast<eval_type>(1-c.param3),
               static_cast<eval_type>(c.param3),
               static_cast<eval_type>(tools::epsilon<RealType>()),
               forwarding_policy());
            return policies::checked_narrowing_cast<RealType, forwarding_policy>(
               result,
               function);
         }
         BOOST_MATH_GPU_ENABLED static RealType find_v1(const RealType x, const RealType v2, const RealType nc, const RealType p)
         {
            constexpr auto function = "non_central_f_distribution<%1%>::find_v1";
            typedef typename policies::evaluation<RealType, Policy>::type eval_type;
            typedef typename policies::normalise<
               Policy,
               policies::promote_float<false>,
               policies::promote_double<false>,
               policies::discrete_quantile<>,
               policies::assert_undefined<> >::type forwarding_policy;
            eval_type result = detail::find_degrees_of_freedom_f(
               static_cast<eval_type>(x),
               static_cast<eval_type>(v2),
               static_cast<eval_type>(nc),
               true,
               static_cast<eval_type>(p),
               static_cast<eval_type>(1-p),
               forwarding_policy());
            return policies::checked_narrowing_cast<RealType, forwarding_policy>(
               result,
               function);
         }
         template <class A, class B, class C, class D>
         BOOST_MATH_GPU_ENABLED static RealType find_v1(const complemented4_type<A,B,C, D>& c)
         {
            constexpr auto function = "non_central_f_distribution<%1%>::find_non_centrality";
            typedef typename policies::evaluation<RealType, Policy>::type eval_type;
            typedef typename policies::normalise<
               Policy,
               policies::promote_float<false>,
               policies::promote_double<false>,
               policies::discrete_quantile<>,
               policies::assert_undefined<> >::type forwarding_policy;
            eval_type result = detail::find_degrees_of_freedom_f(
               static_cast<eval_type>(c.dist),
               static_cast<eval_type>(c.param1),
               static_cast<eval_type>(c.param2),
               true,
               static_cast<eval_type>(1-c.param3),
               static_cast<eval_type>(c.param3),
               forwarding_policy());
            return policies::checked_narrowing_cast<RealType, forwarding_policy>(
               result,
               function);
         }
                  BOOST_MATH_GPU_ENABLED static RealType find_v2(const RealType x, const RealType v2, const RealType nc, const RealType p)
         {
            constexpr auto function = "non_central_f_distribution<%1%>::find_v1";
            typedef typename policies::evaluation<RealType, Policy>::type eval_type;
            typedef typename policies::normalise<
               Policy,
               policies::promote_float<false>,
               policies::promote_double<false>,
               policies::discrete_quantile<>,
               policies::assert_undefined<> >::type forwarding_policy;
            eval_type result = detail::find_degrees_of_freedom_f(
               static_cast<eval_type>(x),
               static_cast<eval_type>(v2),
               static_cast<eval_type>(nc),
               false,
               static_cast<eval_type>(p),
               static_cast<eval_type>(1-p),
               forwarding_policy());
            return policies::checked_narrowing_cast<RealType, forwarding_policy>(
               result,
               function);
         }
         template <class A, class B, class C, class D>
         BOOST_MATH_GPU_ENABLED static RealType find_v2(const complemented4_type<A,B,C, D>& c)
         {
            constexpr auto function = "non_central_f_distribution<%1%>::find_non_centrality";
            typedef typename policies::evaluation<RealType, Policy>::type eval_type;
            typedef typename policies::normalise<
               Policy,
               policies::promote_float<false>,
               policies::promote_double<false>,
               policies::discrete_quantile<>,
               policies::assert_undefined<> >::type forwarding_policy;
            eval_type result = detail::find_degrees_of_freedom_f(
               static_cast<eval_type>(c.dist),
               static_cast<eval_type>(c.param1),
               static_cast<eval_type>(c.param2),
               false,
               static_cast<eval_type>(1-c.param3),
               static_cast<eval_type>(c.param3),
               forwarding_policy());
            return policies::checked_narrowing_cast<RealType, forwarding_policy>(
               result,
               function);
         }
      private:
         // Data member, initialized by constructor.
         RealType v1;   // alpha.
         RealType v2;   // beta.
         RealType ncp; // non-centrality parameter
      }; // template <class RealType, class Policy> class non_central_f_distribution

      typedef non_central_f_distribution<double> non_central_f; // Reserved name of type double.

      #ifdef __cpp_deduction_guides
      template <class RealType>
      non_central_f_distribution(RealType,RealType,RealType)->non_central_f_distribution<typename boost::math::tools::promote_args<RealType>::type>;
      #endif

      // Non-member functions to give properties of the distribution.

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const non_central_f_distribution<RealType, Policy>& /* dist */)
      { // Range of permissible values for random variable k.
         using boost::math::tools::max_value;
         return boost::math::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>());
      }

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const non_central_f_distribution<RealType, Policy>& /* dist */)
      { // Range of supported values for random variable k.
         // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
         using boost::math::tools::max_value;
         return boost::math::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>());
      }

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType mean(const non_central_f_distribution<RealType, Policy>& dist)
      {
         constexpr auto function = "mean(non_central_f_distribution<%1%> const&)";
         RealType v1 = dist.degrees_of_freedom1();
         RealType v2 = dist.degrees_of_freedom2();
         RealType l = dist.non_centrality();
         RealType r;
         if(!detail::check_df(
            function,
            v1, &r, Policy())
               ||
            !detail::check_df(
               function,
               v2, &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               l,
               &r,
               Policy()))
                  return r;
         if(v2 <= 2)
            return policies::raise_domain_error(
               function,
               "Second degrees of freedom parameter was %1%, but must be > 2 !",
               v2, Policy());
         return v2 * (v1 + l) / (v1 * (v2 - 2));
      } // mean

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType mode(const non_central_f_distribution<RealType, Policy>& dist)
      { // mode.
         constexpr auto function = "mode(non_central_chi_squared_distribution<%1%> const&)";

         RealType n = dist.degrees_of_freedom1();
         RealType m = dist.degrees_of_freedom2();
         RealType l = dist.non_centrality();
         RealType r;
         if(!detail::check_df(
            function,
            n, &r, Policy())
               ||
            !detail::check_df(
               function,
               m, &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               l,
               &r,
               Policy()))
                  return r;
         RealType guess = m > 2 ? RealType(m * (n + l) / (n * (m - 2))) : RealType(1);
         return detail::generic_find_mode(
            dist,
            guess,
            function);
      }

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType variance(const non_central_f_distribution<RealType, Policy>& dist)
      { // variance.
         constexpr auto function = "variance(non_central_f_distribution<%1%> const&)";
         RealType n = dist.degrees_of_freedom1();
         RealType m = dist.degrees_of_freedom2();
         RealType l = dist.non_centrality();
         RealType r;
         if(!detail::check_df(
            function,
            n, &r, Policy())
               ||
            !detail::check_df(
               function,
               m, &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               l,
               &r,
               Policy()))
                  return r;
         if(m <= 4)
            return policies::raise_domain_error(
               function,
               "Second degrees of freedom parameter was %1%, but must be > 4 !",
               m, Policy());
         RealType result = 2 * m * m * ((n + l) * (n + l)
            + (m - 2) * (n + 2 * l));
         result /= (m - 4) * (m - 2) * (m - 2) * n * n;
         return result;
      }

      // RealType standard_deviation(const non_central_f_distribution<RealType, Policy>& dist)
      // standard_deviation provided by derived accessors.

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType skewness(const non_central_f_distribution<RealType, Policy>& dist)
      { // skewness = sqrt(l).
         constexpr auto function = "skewness(non_central_f_distribution<%1%> const&)";
         BOOST_MATH_STD_USING
         RealType n = dist.degrees_of_freedom1();
         RealType m = dist.degrees_of_freedom2();
         RealType l = dist.non_centrality();
         RealType r;
         if(!detail::check_df(
            function,
            n, &r, Policy())
               ||
            !detail::check_df(
               function,
               m, &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               l,
               &r,
               Policy()))
                  return r;
         if(m <= 6)
            return policies::raise_domain_error(
               function,
               "Second degrees of freedom parameter was %1%, but must be > 6 !",
               m, Policy());
         RealType result = 2 * constants::root_two<RealType>();
         result *= sqrt(m - 4);
         result *= (n * (m + n - 2) *(m + 2 * n - 2)
            + 3 * (m + n - 2) * (m + 2 * n - 2) * l
            + 6 * (m + n - 2) * l * l + 2 * l * l * l);
         result /= (m - 6) * pow(n * (m + n - 2) + 2 * (m + n - 2) * l + l * l, RealType(1.5f));
         return result;
      }

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const non_central_f_distribution<RealType, Policy>& dist)
      {
         constexpr auto function = "kurtosis_excess(non_central_f_distribution<%1%> const&)";
         BOOST_MATH_STD_USING
         RealType n = dist.degrees_of_freedom1();
         RealType m = dist.degrees_of_freedom2();
         RealType l = dist.non_centrality();
         RealType r;
         if(!detail::check_df(
            function,
            n, &r, Policy())
               ||
            !detail::check_df(
               function,
               m, &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               l,
               &r,
               Policy()))
                  return r;
         if(m <= 8)
            return policies::raise_domain_error(
               function,
               "Second degrees of freedom parameter was %1%, but must be > 8 !",
               m, Policy());
         RealType l2 = l * l;
         RealType l3 = l2 * l;
         RealType l4 = l2 * l2;
         RealType result = (3 * (m - 4) * (n * (m + n - 2)
            * (4 * (m - 2) * (m - 2)
            + (m - 2) * (m + 10) * n
            + (10 + m) * n * n)
            + 4 * (m + n - 2) * (4 * (m - 2) * (m - 2)
            + (m - 2) * (10 + m) * n
            + (10 + m) * n * n) * l + 2 * (10 + m)
            * (m + n - 2) * (2 * m + 3 * n - 4) * l2
            + 4 * (10 + m) * (-2 + m + n) * l3
            + (10 + m) * l4))
            /
            ((-8 + m) * (-6 + m) * boost::math::pow<2>(n * (-2 + m + n)
            + 2 * (-2 + m + n) * l + l2));
            return result;
      } // kurtosis_excess

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const non_central_f_distribution<RealType, Policy>& dist)
      {
         return kurtosis_excess(dist) + 3;
      }

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType pdf(const non_central_f_distribution<RealType, Policy>& dist, const RealType& x)
      { // Probability Density/Mass Function.
         typedef typename policies::evaluation<RealType, Policy>::type value_type;
         typedef typename policies::normalise<
            Policy,
            policies::promote_float<false>,
            policies::promote_double<false>,
            policies::discrete_quantile<>,
            policies::assert_undefined<> >::type forwarding_policy;

         value_type alpha = dist.degrees_of_freedom1() / 2;
         value_type beta = dist.degrees_of_freedom2() / 2;
         value_type y = x * alpha / beta;
         value_type r = pdf(boost::math::non_central_beta_distribution<value_type, forwarding_policy>(alpha, beta, dist.non_centrality()), y / (1 + y));
         return policies::checked_narrowing_cast<RealType, forwarding_policy>(
            r * (dist.degrees_of_freedom1() / dist.degrees_of_freedom2()) / ((1 + y) * (1 + y)),
            "pdf(non_central_f_distribution<%1%>, %1%)");
      } // pdf

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED RealType cdf(const non_central_f_distribution<RealType, Policy>& dist, const RealType& x)
      {
         constexpr auto function = "cdf(const non_central_f_distribution<%1%>&, %1%)";
         RealType r;
         if(!detail::check_df(
            function,
            dist.degrees_of_freedom1(), &r, Policy())
               ||
            !detail::check_df(
               function,
               dist.degrees_of_freedom2(), &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               dist.non_centrality(),
               &r,
               Policy()))
                  return r;

         if((x < 0) || !(boost::math::isfinite)(x))
         {
            return policies::raise_domain_error<RealType>(
               function, "Random Variable parameter was %1%, but must be > 0 !", x, Policy());
         }

         RealType alpha = dist.degrees_of_freedom1() / 2;
         RealType beta = dist.degrees_of_freedom2() / 2;
         RealType y = x * alpha / beta;
         RealType c = y / (1 + y);
         RealType cp = 1 / (1 + y);
         //
         // To ensure accuracy, we pass both x and 1-x to the
         // non-central beta cdf routine, this ensures accuracy
         // even when we compute x to be ~ 1:
         //
         r = detail::non_central_beta_cdf(c, cp, alpha, beta,
            dist.non_centrality(), false, Policy());
         return r;
      } // cdf

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED RealType cdf(const complemented2_type<non_central_f_distribution<RealType, Policy>, RealType>& c)
      { // Complemented Cumulative Distribution Function
         constexpr auto function = "cdf(complement(const non_central_f_distribution<%1%>&, %1%))";
         RealType r;
         if(!detail::check_df(
            function,
            c.dist.degrees_of_freedom1(), &r, Policy())
               ||
            !detail::check_df(
               function,
               c.dist.degrees_of_freedom2(), &r, Policy())
               ||
            !detail::check_non_centrality(
               function,
               c.dist.non_centrality(),
               &r,
               Policy()))
                  return r;

         if((c.param < 0) || !(boost::math::isfinite)(c.param))
         {
            return policies::raise_domain_error<RealType>(
               function, "Random Variable parameter was %1%, but must be > 0 !", c.param, Policy());
         }

         RealType alpha = c.dist.degrees_of_freedom1() / 2;
         RealType beta = c.dist.degrees_of_freedom2() / 2;
         RealType y = c.param * alpha / beta;
         RealType x = y / (1 + y);
         RealType cx = 1 / (1 + y);
         //
         // To ensure accuracy, we pass both x and 1-x to the
         // non-central beta cdf routine, this ensures accuracy
         // even when we compute x to be ~ 1:
         //
         r = detail::non_central_beta_cdf(x, cx, alpha, beta,
            c.dist.non_centrality(), true, Policy());
         return r;
      } // ccdf

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType quantile(const non_central_f_distribution<RealType, Policy>& dist, const RealType& p)
      { // Quantile (or Percent Point) function.
         RealType alpha = dist.degrees_of_freedom1() / 2;
         RealType beta = dist.degrees_of_freedom2() / 2;
         RealType x = quantile(boost::math::non_central_beta_distribution<RealType, Policy>(alpha, beta, dist.non_centrality()), p);
         if(x == 1)
            return policies::raise_overflow_error<RealType>(
               "quantile(const non_central_f_distribution<%1%>&, %1%)",
               "Result of non central F quantile is too large to represent.",
               Policy());
         return (x / (1 - x)) * (dist.degrees_of_freedom2() / dist.degrees_of_freedom1());
      } // quantile

      template <class RealType, class Policy>
      BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<non_central_f_distribution<RealType, Policy>, RealType>& c)
      { // Quantile (or Percent Point) function.
         RealType alpha = c.dist.degrees_of_freedom1() / 2;
         RealType beta = c.dist.degrees_of_freedom2() / 2;
         RealType x = quantile(complement(boost::math::non_central_beta_distribution<RealType, Policy>(alpha, beta, c.dist.non_centrality()), c.param));
         if(x == 1)
            return policies::raise_overflow_error<RealType>(
               "quantile(complement(const non_central_f_distribution<%1%>&, %1%))",
               "Result of non central F quantile is too large to represent.",
               Policy());
         return (x / (1 - x)) * (c.dist.degrees_of_freedom2() / c.dist.degrees_of_freedom1());
      } // quantile complement.
   } // namespace math
} // namespace boost

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif // BOOST_MATH_SPECIAL_NON_CENTRAL_F_HPP



