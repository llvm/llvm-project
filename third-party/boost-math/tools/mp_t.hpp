//  (C) Copyright John Maddock 2014.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_MP_T
#define BOOST_MATH_TOOLS_MP_T

#ifndef BOOST_MATH_PRECISION
#define BOOST_MATH_PRECISION 1000
#endif

#if defined(BOOST_MATH_USE_MPFR)

#include <boost/multiprecision/mpfr.hpp>

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<BOOST_MATH_PRECISION *301L / 1000L> > mp_t;

#else

#include <boost/multiprecision/cpp_bin_float.hpp>

typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<BOOST_MATH_PRECISION, boost::multiprecision::digit_base_2> > mp_t;

#endif

inline mp_t ConvPrec(mp_t arg, int digits)
{
   int e1, e2;
   mp_t t = frexp(arg, &e1);
   t = frexp(floor(ldexp(t, digits + 1)), &e2);
   return ldexp(t, e1);
}

inline mp_t relative_error(mp_t a, mp_t b)
{
   mp_t min_val = boost::math::tools::min_value<mp_t>();
   mp_t max_val = boost::math::tools::max_value<mp_t>();

   if((a != 0) && (b != 0))
   {
      // mp_tODO: use isfinite:
      if(fabs(b) >= max_val)
      {
         if(fabs(a) >= max_val)
            return 0;  // one infinity is as good as another!
      }
      // If the result is denormalised, treat all denorms as equivalent:
      if((a < min_val) && (a > 0))
         a = min_val;
      else if((a > -min_val) && (a < 0))
         a = -min_val;
      if((b < min_val) && (b > 0))
         b = min_val;
      else if((b > -min_val) && (b < 0))
         b = -min_val;
      return (std::max)(fabs((a - b) / a), fabs((a - b) / b));
   }

   // Handle special case where one or both are zero:
   if(min_val == 0)
      return fabs(a - b);
   if(fabs(a) < min_val)
      a = min_val;
   if(fabs(b) < min_val)
      b = min_val;
   return (std::max)(fabs((a - b) / a), fabs((a - b) / b));
}


#endif // BOOST_MATH_TOOLS_MP_T
