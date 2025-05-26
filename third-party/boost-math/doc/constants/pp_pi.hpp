//[preprocessed_pi

// Preprocessed pi constant, annotated.

namespace boost
{
  namespace math
  {
    namespace constants
    {
      namespace detail
      {
        template <class T> struct constant_pi
        {
          private:
            // Default implementations from string of decimal digits:
            static inline T get_from_string()
            {
            static const T result
               = detail::convert_from_string<T>("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00",
               std::is_convertible<const char*, T>());
              return result;
            }
            template <int N> static T compute();

          public:
            // Default implementations from string of decimal digits:
            static inline T get(const std::integral_constant<int, construct_from_string>&)
            {
              constant_initializer<T, & constant_pi<T>::get_from_string >::do_nothing();
              return get_from_string();
            }
            // Float, double and long double versions:
            static inline T get(const std::integral_constant<int, construct_from_float>)
            {
              return 3.141592653589793238462643383279502884e+00F;
            }
            static inline  T get(const std::integral_constant<int, construct_from_double>&)
            {
              return 3.141592653589793238462643383279502884e+00;
            }
            static inline  T get(const std::integral_constant<int, construct_from_long_double>&)
            {
              return 3.141592653589793238462643383279502884e+00L;
            }
            // For very high precision that is nonetheless can be calculated at compile time:
            template <int N> static inline T get(const std::integral_constant<int, N>& n)
            {
              constant_initializer2<T, N, & constant_pi<T>::template compute<N> >::do_nothing();
              return compute<N>();
            }
            //For true arbitrary precision, which may well vary at runtime.
            static inline T get(const std::integral_constant<int, 0>&)
            {
              return tools::digits<T>() > max_string_digits ? compute<0>() : get(std::integral_constant<int, construct_from_string>());
            }
         }; // template <class T> struct constant_pi
      } //  namespace detail

      // The actual forwarding function (including policy to control precision).
      template <class T, class Policy> inline T pi( )
      {
        return detail:: constant_pi<T>::get(typename construction_traits<T, Policy>::type());
      }
      // The actual forwarding function (using default policy to control precision).
      template <class T> inline  T pi()
      {
        return pi<T, boost::math::policies::policy<> >()
      }
    } //     namespace constants

    // Namespace specific versions, for the three built-in floats:
    namespace float_constants
    {
      static const float pi = 3.141592653589793238462643383279502884e+00F;
    }
    namespace double_constants
    {
      static const double pi = 3.141592653589793238462643383279502884e+00;
    }
    namespace long_double_constants
    {
      static const long double pi = 3.141592653589793238462643383279502884e+00L;
    }
    namespace constants{;
    } // namespace constants
  } // namespace math
} // namespace boost

//] [/preprocessed_pi]

/*
  Copyright 2012 John Maddock and Paul A. Bristow.
  Distributed under the Boost Software License, Version 1.0.
  (See accompanying file LICENSE_1_0.txt or copy at
  http://www.boost.org/LICENSE_1_0.txt)
*/


