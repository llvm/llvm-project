//  (C) Copyright Christopher Kormanyos 2025.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See also: https://godbolt.org/z/nhMsKb8Yr

#include <boost/core/lightweight_test.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include <cmath>
#include <limits>

namespace local
{
  template<typename FloatType>
  auto test() -> void
  {
    using float_type = FloatType;

    using std::ldexp;

    // N[BesselY[-1, (2875/1000)  (2^(-128))], 51]
    // 7.53497332069250908152363321534337029122980775749906*10^37

    const float_type x { static_cast<float_type>(ldexp(2.875L, -128)) };

    constexpr float_type ctrl { static_cast<float_type>(7.53497332069250908152363321534337029122980775749906E37L) };

    const float_type result = ::boost::math::cyl_neumann(-1, x);

    constexpr float_type tol = std::numeric_limits<float_type>::epsilon() * 16;

    using ::std::fabs;

    BOOST_TEST(result > 0);
    BOOST_TEST(fabs(1 - (result / ctrl)) < tol);
  }
} // namespace local

auto main() -> int
{
  local::test<double>();
  local::test<long double>();

  return boost::report_errors();
}
