//  (C) Copyright John Maddock 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/series.hpp>
#include <boost/math/tools/assert.hpp>

#include <iostream>
#include <complex>
#include <cassert>

//[series_log1p
template <class T>
struct log1p_series
{
   // we must define a result_type typedef:
   typedef T result_type;

   log1p_series(T x)
      : k(0), m_mult(-x), m_prod(-1) {}

   T operator()()
   {
      // This is the function operator invoked by the summation
      // algorithm, the first call to this operator should return
      // the first term of the series, the second call the second
      // term and so on.
      m_prod *= m_mult;
      return m_prod / ++k;
   }

private:
   int k;
   const T m_mult;
   T m_prod;
};
//]

//[series_log1p_func
template <class T>
T log1p(T x)
{
   // We really should add some error checking on x here!
   BOOST_MATH_ASSERT(std::fabs(x) < 1);

   // Construct the series functor:
   log1p_series<T> s(x);
   // Set a limit on how many iterations we permit:
   std::uintmax_t max_iter = 1000;
   // Add it up, with enough precision for full machine precision:
   return boost::math::tools::sum_series(s, std::numeric_limits<T>::epsilon(), max_iter);
}
//]

//[series_clog1p_func
template <class T>
struct log1p_series<std::complex<T> >
{
   // we must define a result_type typedef:
   typedef std::complex<T> result_type;

   log1p_series(std::complex<T> x)
      : k(0), m_mult(-x), m_prod(-1) {}

   std::complex<T> operator()()
   {
      // This is the function operator invoked by the summation
      // algorithm, the first call to this operator should return
      // the first term of the series, the second call the second
      // term and so on.
      m_prod *= m_mult;
      return m_prod / T(++k);
   }

private:
   int k;
   const std::complex<T> m_mult;
   std::complex<T> m_prod;
};


template <class T>
std::complex<T> log1p(std::complex<T> x)
{
   // We really should add some error checking on x here!
   BOOST_MATH_ASSERT(abs(x) < 1);

   // Construct the series functor:
   log1p_series<std::complex<T> > s(x);
   // Set a limit on how many iterations we permit:
   std::uintmax_t max_iter = 1000;
   // Add it up, with enough precision for full machine precision:
   return boost::math::tools::sum_series(s, std::complex<T>(std::numeric_limits<T>::epsilon()), max_iter);
}
//]

int main()
{
   using namespace boost::math::tools;

   std::cout << log1p(0.25) << std::endl;

   std::cout << log1p(std::complex<double>(0.25, 0.25)) << std::endl;

   return 0;
}
