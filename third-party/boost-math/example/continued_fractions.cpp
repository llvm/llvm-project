//  (C) Copyright John Maddock 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/fraction.hpp>
#include <iostream>
#include <complex>
#include <boost/multiprecision/cpp_complex.hpp>

//[golden_ratio_1
template <class T>
struct golden_ratio_fraction
{
   typedef T result_type;

   result_type operator()()
   {
      return 1;
   }
};
//]

//[cf_tan_fraction
template <class T>
struct tan_fraction
{
private:
   T a, b;
public:
   tan_fraction(T v)
      : a(-v * v), b(-1)
   {}

   typedef std::pair<T, T> result_type;

   std::pair<T, T> operator()()
   {
      b += 2;
      return std::make_pair(a, b);
   }
};
//]
//[cf_tan
template <class T>
T tan(T a)
{
   tan_fraction<T> fract(a);
   return a / continued_fraction_b(fract, std::numeric_limits<T>::epsilon());
}
//]
//[cf_expint_fraction
template <class T>
struct expint_fraction
{
   typedef std::pair<T, T> result_type;
   expint_fraction(unsigned n_, T z_) : b(z_ + T(n_)), i(-1), n(n_) {}
   std::pair<T, T> operator()()
   {
      std::pair<T, T> result = std::make_pair(-static_cast<T>((i + 1) * (n + i)), b);
      b += 2;
      ++i;
      return result;
   }
private:
   T b;
   int i;
   unsigned n;
};
//]
//[cf_expint
template <class T>
inline std::complex<T> expint_as_fraction(unsigned n, std::complex<T> const& z)
{
   std::uintmax_t max_iter = 1000;
   expint_fraction<std::complex<T> > f(n, z);
   std::complex<T> result = boost::math::tools::continued_fraction_b(
      f,
      std::complex<T>(std::numeric_limits<T>::epsilon()),
      max_iter);
   result = exp(-z) / result;
   return result;
}
//]
//[cf_upper_gamma_fraction
template <class T>
struct upper_incomplete_gamma_fract
{
private:
   typedef typename T::value_type scalar_type;
   T z, a;
   int k;
public:
   typedef std::pair<T, T> result_type;

   upper_incomplete_gamma_fract(T a1, T z1)
      : z(z1 - a1 + scalar_type(1)), a(a1), k(0)
   {
   }

   result_type operator()()
   {
      ++k;
      z += scalar_type(2);
      return result_type(scalar_type(k) * (a - scalar_type(k)), z);
   }
};
//]
//[cf_gamma_Q
template <class T>
inline std::complex<T> gamma_Q_as_fraction(const std::complex<T>& a, const std::complex<T>& z)
{
   upper_incomplete_gamma_fract<std::complex<T> > f(a, z);
   std::complex<T> eps(std::numeric_limits<T>::epsilon());
   return pow(z, a) / (exp(z) *(z - a + T(1) + boost::math::tools::continued_fraction_a(f, eps)));
}
//]
inline boost::multiprecision::cpp_complex_50 gamma_Q_as_fraction(const boost::multiprecision::cpp_complex_50& a, const boost::multiprecision::cpp_complex_50& z)
{
   upper_incomplete_gamma_fract<boost::multiprecision::cpp_complex_50> f(a, z);
   boost::multiprecision::cpp_complex_50 eps(std::numeric_limits<boost::multiprecision::cpp_complex_50::value_type>::epsilon());
   return pow(z, a) / (exp(z) * (z - a + 1 + boost::math::tools::continued_fraction_a(f, eps)));
}


int main()
{
   using namespace boost::math::tools;

   //[cf_gr
   golden_ratio_fraction<double> func;
   double gr = continued_fraction_a(
      func,
      std::numeric_limits<double>::epsilon());
   std::cout << "The golden ratio is: " << gr << std::endl;
   //]

   std::cout << tan(0.5) << std::endl;

   std::complex<double> arg(3, 2);
   std::cout << expint_as_fraction(5, arg) << std::endl;

   std::complex<double> a(3, 3), z(3, 2);
   std::cout << gamma_Q_as_fraction(a, z) << std::endl;

   boost::multiprecision::cpp_complex_50 am(3, 3), zm(3, 2);
   std::cout << gamma_Q_as_fraction(am, zm) << std::endl;

   return 0;
}
