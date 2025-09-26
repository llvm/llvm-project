//  Copyright John Maddock 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

template <class T>
void test()
{
   auto f = [](const T& x){ return exp(-x * x / 2); };

   T error = 0;

   auto r = boost::math::quadrature::gauss_kronrod<T, 15>::integrate(f, 0, 1, 0, 0, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss_kronrod<T, 21>::integrate(f, 0, 1, 0, 0, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss_kronrod<T, 31>::integrate(f, 0, 1, 0, 0, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss_kronrod<T, 41>::integrate(f, 0, 1, 0, 0, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss_kronrod<T, 51>::integrate(f, 0, 1, 0, 0, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss_kronrod<T, 61>::integrate(f, 0, 1, 0, 0, &error);
   std::cout << r << std::endl;

   r = boost::math::quadrature::gauss<T, 7>::integrate(f, 0, 1, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss<T, 15>::integrate(f, 0, 1, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss<T, 20>::integrate(f, 0, 1, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss<T, 25>::integrate(f, 0, 1, &error);
   std::cout << r << std::endl;
   r = boost::math::quadrature::gauss<T, 30>::integrate(f, 0, 1, &error);
   std::cout << r << std::endl;
}


int main()
{
   test<boost::multiprecision::cpp_bin_float_double>();
   test<boost::multiprecision::cpp_bin_float_double_extended>();
   test<boost::multiprecision::cpp_bin_float_quad>();

   using strange = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<50, boost::multiprecision::backends::digit_base_2, void, std::int16_t, -16382, 16383>, boost::multiprecision::et_off>;
   test<strange>();

   return 0;
}
