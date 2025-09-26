// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <limits>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/constants/constants.hpp>
#include "math_unit_test.hpp"

// numerically evaluate the integral for Stefan-Boltzmann Law from Planck's Law
// https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law#Derivation_from_Planck's_law
template<typename Real>
Real StefanBoltzmann() 
{
   constexpr auto h = Real(6.62607015e-34);
   constexpr auto c = Real(299792458.0);
   constexpr auto kB = Real(1.380649e-23);

   constexpr auto c1 = Real(2) * boost::math::constants::pi<Real>() * h * c * c;
   constexpr auto c2 = h * c / kB;
   constexpr auto T = 1000;

   auto integrand = [&](const Real l)
   {
      return c1 / (std::pow(l, Real(5)) * (std::exp(c2 / (T * l)) - Real(1)));
   };

   auto integrator = boost::math::quadrature::tanh_sinh<Real>(15);
   return static_cast<Real>(integrator.integrate(integrand, Real(0.0), std::numeric_limits<Real>::infinity()));
}

int main() {

   constexpr auto sigma = 56703.74419;

   // Double Precision
   CHECK_MOLLIFIED_CLOSE(sigma, StefanBoltzmann<double>(), 1e-5);

   // integate using single precision. It gets the right answer, but there is integer underflow
   // if this line is commented out, there is no underflow
   CHECK_MOLLIFIED_CLOSE(sigma, StefanBoltzmann<float>(), 1e-5f);

   return boost::math::test::report_errors();
}
