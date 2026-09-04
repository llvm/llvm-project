//  Copyright John Maddock 2016.
//  Copyright Matt Borland 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_CONVERT_FROM_STRING_INCLUDED
#define BOOST_MATH_TOOLS_CONVERT_FROM_STRING_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <type_traits>
#ifndef BOOST_MATH_STANDALONE

#if defined(_MSC_VER) || defined(__GNUC__)
# pragma push_macro( "I" )
# undef I
#endif

#include <boost/lexical_cast.hpp>

#if defined(_MSC_VER) || defined(__GNUC__)
# pragma pop_macro( "I" )
#endif

#endif

namespace boost{ namespace math{ namespace tools{

   template <class T>
   struct convert_from_string_result
   {
      typedef typename std::conditional<std::is_constructible<T, const char*>::value, const char*, T>::type type;
   };

   template <class Real>
   Real convert_from_string(const char* p, const std::false_type&)
   {
      #ifdef BOOST_MATH_NO_LEXICAL_CAST

      // This function should not compile, we don't have the necessary functionality to support it:
      static_assert(sizeof(Real) == 0, "boost.lexical_cast is not supported in standalone mode.");
      (void)p; // Suppresses -Wunused-parameter
      return Real(0);

      #elif defined(BOOST_MATH_USE_CHARCONV_FOR_CONVERSION)

      if constexpr (std::is_arithmetic_v<Real>)
      {
         Real v {};
         std::from_chars(p, p + std::strlen(p), v);

         return v;
      }
      else
      {
         return boost::lexical_cast<Real>(p);
      }

      #else

      return boost::lexical_cast<Real>(p);

      #endif
   }
   template <class Real>
   constexpr const char* convert_from_string(const char* p, const std::true_type&) noexcept
   {
      return p;
   }
   template <class Real>
   constexpr typename convert_from_string_result<Real>::type convert_from_string(const char* p) noexcept((std::is_constructible<Real, const char*>::value))
   {
      return convert_from_string<Real>(p, std::is_constructible<Real, const char*>());
   }

} // namespace tools
} // namespace math
} // namespace boost

#endif // BOOST_MATH_TOOLS_CONVERT_FROM_STRING_INCLUDED

