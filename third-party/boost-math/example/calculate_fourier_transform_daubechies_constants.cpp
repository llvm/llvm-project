//  (C) Copyright Nick Thompson 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <utility>
#include <boost/math/filters/daubechies.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/constants/constants.hpp>

using std::pow;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::filters::daubechies_scaling_filter;
using boost::math::tools::polynomial;
using boost::math::constants::half;
using boost::math::constants::root_two;

template<typename Real, size_t N>
std::vector<Real> get_constants() {
   auto h = daubechies_scaling_filter<cpp_bin_float_100, N>();
   auto p = polynomial<cpp_bin_float_100>(h.begin(), h.end());

   auto q = polynomial({half<cpp_bin_float_100>(), half<cpp_bin_float_100>()});
   q = pow(q, N);
   auto l = p/q;
   return l.data(); 
}

template<typename Real>
void print_constants(std::vector<Real> const & l) {
   std::cout << std::setprecision(std::numeric_limits<Real>::digits10 -10);
   std::cout << "return std::array<Real, " << l.size() << ">{";
   for (size_t i = 0; i < l.size() - 1; ++i) {
       std::cout << "BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits, " << l[i]/root_two<Real>() << "), ";
   }
   std::cout << "BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits, " << l.back()/root_two<Real>() << ")};\n";
}

int main() {
   auto constants = get_constants<cpp_bin_float_100, 1>();
   print_constants(constants);
}
