//  (C) Copyright John Maddock 2005-2006.
//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_FRACTION_INCLUDED
#define BOOST_MATH_TOOLS_FRACTION_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/complex.hpp>
#include <boost/math/tools/cstdint.hpp>

namespace boost{ namespace math{ namespace tools{

namespace detail
{

   template <typename T>
   struct is_pair : public boost::math::false_type{};

   template <typename T, typename U>
   struct is_pair<boost::math::pair<T,U>> : public boost::math::true_type{};

   template <typename Gen>
   struct fraction_traits_simple
   {
      using result_type = typename Gen::result_type;
      using  value_type = typename Gen::result_type;

      BOOST_MATH_GPU_ENABLED static result_type a(const value_type&) BOOST_MATH_NOEXCEPT(value_type)
      {
         return 1;
      }
      BOOST_MATH_GPU_ENABLED static result_type b(const value_type& v) BOOST_MATH_NOEXCEPT(value_type)
      {
         return v;
      }
   };

   template <typename Gen>
   struct fraction_traits_pair
   {
      using  value_type = typename Gen::result_type;
      using result_type = typename value_type::first_type;

      BOOST_MATH_GPU_ENABLED static result_type a(const value_type& v) BOOST_MATH_NOEXCEPT(value_type)
      {
         return v.first;
      }
      BOOST_MATH_GPU_ENABLED static result_type b(const value_type& v) BOOST_MATH_NOEXCEPT(value_type)
      {
         return v.second;
      }
   };

   template <typename Gen>
   struct fraction_traits
       : public boost::math::conditional<
         is_pair<typename Gen::result_type>::value,
         fraction_traits_pair<Gen>,
         fraction_traits_simple<Gen>>::type
   {
   };

   template <typename T, bool = is_complex_type<T>::value>
   struct tiny_value
   {
      // For float, double, and long double, 1/min_value<T>() is finite.
      // But for mpfr_float and cpp_bin_float, 1/min_value<T>() is inf.
      // Multiply the min by 16 so that the reciprocal doesn't overflow.
      BOOST_MATH_GPU_ENABLED static T get() {
         return 16*tools::min_value<T>();
      }
   };
   template <typename T>
   struct tiny_value<T, true>
   {
      using value_type = typename T::value_type;
      BOOST_MATH_GPU_ENABLED static T get() {
         return 16*tools::min_value<value_type>();
      }
   };

} // namespace detail

namespace detail {

//
// continued_fraction_b
// Evaluates:
//
// b0 +       a1
//      ---------------
//      b1 +     a2
//           ----------
//           b2 +   a3
//                -----
//                b3 + ...
//
// Note that the first a0 returned by generator Gen is discarded.
//

template <typename Gen, typename U>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_b_impl(Gen& g, const U& factor, boost::math::uintmax_t& max_terms)
      noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
      #ifndef BOOST_MATH_HAS_GPU_SUPPORT
      // SYCL can not handle this condition so we only check float on that platform
      && noexcept(std::declval<Gen>()())
      #endif
      )
{
   BOOST_MATH_STD_USING // ADL of std names

   using traits = detail::fraction_traits<Gen>;
   using result_type = typename traits::result_type;
   using value_type = typename traits::value_type;
   using integer_type = typename integer_scalar_type<result_type>::type;
   using scalar_type = typename scalar_type<result_type>::type;

   integer_type const zero(0), one(1);

   result_type tiny = detail::tiny_value<result_type>::get();
   scalar_type terminator = abs(factor);

   value_type v = g();

   result_type f, C, D, delta;
   f = traits::b(v);
   if(f == zero)
      f = tiny;
   C = f;
   D = 0;

   boost::math::uintmax_t counter(max_terms);
   do{
      v = g();
      D = traits::b(v) + traits::a(v) * D;
      if(D == result_type(0))
         D = tiny;
      C = traits::b(v) + traits::a(v) / C;
      if(C == zero)
         C = tiny;
      D = one/D;
      delta = C*D;
      f = f * delta;
   }while((abs(delta - one) > terminator) && --counter);

   max_terms = max_terms - counter;

   return f;
}

} // namespace detail

template <typename Gen, typename U>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_b(Gen& g, const U& factor, boost::math::uintmax_t& max_terms)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
         #ifndef BOOST_MATH_HAS_GPU_SUPPORT
         && noexcept(std::declval<Gen>()())
         #endif
         )
{
   return detail::continued_fraction_b_impl(g, factor, max_terms);
}

template <typename Gen, typename U>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_b(Gen& g, const U& factor)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   boost::math::uintmax_t max_terms = (boost::math::numeric_limits<boost::math::uintmax_t>::max)();
   return detail::continued_fraction_b_impl(g, factor, max_terms);
}

template <typename Gen>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_b(Gen& g, int bits)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   BOOST_MATH_STD_USING // ADL of std names

   using traits = detail::fraction_traits<Gen>;
   using result_type = typename traits::result_type;

   result_type factor = ldexp(1.0f, 1 - bits); // 1 / pow(result_type(2), bits);
   boost::math::uintmax_t max_terms = (boost::math::numeric_limits<boost::math::uintmax_t>::max)();
   return detail::continued_fraction_b_impl(g, factor, max_terms);
}

template <typename Gen>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_b(Gen& g, int bits, boost::math::uintmax_t& max_terms)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   BOOST_MATH_STD_USING // ADL of std names

   using traits = detail::fraction_traits<Gen>;
   using result_type = typename traits::result_type;

   result_type factor = ldexp(1.0f, 1 - bits); // 1 / pow(result_type(2), bits);
   return detail::continued_fraction_b_impl(g, factor, max_terms);
}

namespace detail {

//
// continued_fraction_a
// Evaluates:
//
//            a1
//      ---------------
//      b1 +     a2
//           ----------
//           b2 +   a3
//                -----
//                b3 + ...
//
// Note that the first a1 and b1 returned by generator Gen are both used.
//
template <typename Gen, typename U>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_a_impl(Gen& g, const U& factor, boost::math::uintmax_t& max_terms)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   BOOST_MATH_STD_USING // ADL of std names

   using traits = detail::fraction_traits<Gen>;
   using result_type = typename traits::result_type;
   using value_type = typename traits::value_type;
   using integer_type = typename integer_scalar_type<result_type>::type;
   using scalar_type = typename scalar_type<result_type>::type;

   integer_type const zero(0), one(1);

   result_type tiny = detail::tiny_value<result_type>::get();
   scalar_type terminator = abs(factor);

   value_type v = g();

   result_type f, C, D, delta, a0;
   f = traits::b(v);
   a0 = traits::a(v);
   if(f == zero)
      f = tiny;
   C = f;
   D = 0;

   boost::math::uintmax_t counter(max_terms);

   do{
      v = g();
      D = traits::b(v) + traits::a(v) * D;
      if(D == zero)
         D = tiny;
      C = traits::b(v) + traits::a(v) / C;
      if(C == zero)
         C = tiny;
      D = one/D;
      delta = C*D;
      f = f * delta;
   }while((abs(delta - one) > terminator) && --counter);

   max_terms = max_terms - counter;

   return a0/f;
}

} // namespace detail

template <typename Gen, typename U>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_a(Gen& g, const U& factor, boost::math::uintmax_t& max_terms)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   return detail::continued_fraction_a_impl(g, factor, max_terms);
}

template <typename Gen, typename U>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_a(Gen& g, const U& factor)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type)
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   boost::math::uintmax_t max_iter = (boost::math::numeric_limits<boost::math::uintmax_t>::max)();
   return detail::continued_fraction_a_impl(g, factor, max_iter);
}

template <typename Gen>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_a(Gen& g, int bits)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   BOOST_MATH_STD_USING // ADL of std names

   typedef detail::fraction_traits<Gen> traits;
   typedef typename traits::result_type result_type;

   result_type factor = ldexp(1.0f, 1-bits); // 1 / pow(result_type(2), bits);
   boost::math::uintmax_t max_iter = (boost::math::numeric_limits<boost::math::uintmax_t>::max)();

   return detail::continued_fraction_a_impl(g, factor, max_iter);
}

template <typename Gen>
BOOST_MATH_GPU_ENABLED inline typename detail::fraction_traits<Gen>::result_type continued_fraction_a(Gen& g, int bits, boost::math::uintmax_t& max_terms)
   noexcept(BOOST_MATH_IS_FLOAT(typename detail::fraction_traits<Gen>::result_type) 
   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   && noexcept(std::declval<Gen>()())
   #endif
   )
{
   BOOST_MATH_STD_USING // ADL of std names

   using traits = detail::fraction_traits<Gen>;
   using result_type = typename traits::result_type;

   result_type factor = ldexp(1.0f, 1-bits); // 1 / pow(result_type(2), bits);
   return detail::continued_fraction_a_impl(g, factor, max_terms);
}

} // namespace tools
} // namespace math
} // namespace boost

#endif // BOOST_MATH_TOOLS_FRACTION_INCLUDED
