// test file for quaternion.hpp

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <boost/math/quaternion.hpp>

typedef boost::math::quaternion<double> qt;
typedef std::complex<double> ct;

#ifndef BOOST_NO_CXX14_CONSTEXPR

constexpr qt full_constexpr_test(qt a, qt b, double d, ct c)
{
   a.swap(b);
   qt result(a), t;
   result += d;
   result += c;
   result += b;
   t = result;
   t = d;
   t = c;
   result -= d;
   result -= c;
   result -= a;
   result *= d;
   result *= c;
   result *= a;
   result /= d;
   result /= c;
   result /= b;

   result += a + d;
   result += d + a;
   result += a + c;
   result += c + a;
   result += a + b;

   result += a - d;
   result += d - a;
   result += a - c;
   result += c - a;
   result += a - b;

   result += a * d;
   result += d * a;
   result += a * c;
   result += c * a;
   result += a * b;

   result += a / d;
   result += d / a;
   result += a / c;
   result += c / a;
   result += a / b;

   result += norm(a);
   result += conj(a);

   return result;
}

#endif

int main()
{
#ifndef BOOST_NO_CXX11_CONSTEXPR

   constexpr qt q1;
   constexpr qt q2(2.0);
   constexpr qt q3(2.0, 3.0);
   constexpr qt q4(2.0, 3.9, 3.0);
   constexpr qt q5(2.0, 3.9, 3.0, 5.);

   constexpr ct c1(2., 3.);
   constexpr qt q6(c1);
   constexpr qt q7(c1, c1);
   constexpr qt q8(q1);

   constexpr double d1 = q5.real();
   constexpr qt q9 = q1.unreal();
   constexpr double d2 = q1.R_component_1();
   constexpr double d3 = q1.R_component_2();
   constexpr double d4 = q1.R_component_3();
   constexpr double d5 = q1.R_component_4();
   constexpr ct c2 = q1.C_component_1();
   constexpr ct c3 = q1.C_component_1();

   constexpr qt q10 = q1 + d1;
   constexpr qt q11 = d1 + q1;
   constexpr qt q14 = q1 + q2;

   constexpr qt q15 = q1 - d1;
   constexpr qt q16 = d1 - q1;
   constexpr qt q19 = q1 - q2;

   constexpr qt q20 = q1 * d1;
   constexpr qt q21 = d1 * q1;
   constexpr qt q22 = q5 / d1;

   constexpr double d6 = real(q5);
   constexpr qt q23 = unreal(q1);

   constexpr bool b1 = q1 == d1;
   constexpr bool b2 = d1 == q1;
   constexpr bool b3 = q1 != d1;
   constexpr bool b4 = d1 != q1;

   constexpr bool b5 = q1 == c2;
   constexpr bool b6 = c2 == q1;
   constexpr bool b7 = q1 != c2;
   constexpr bool b8 = c2 != q1;
   constexpr bool b9 = q2 == q1;
   constexpr bool b10 = q1 != q2;

   (void)q9;
   (void)d2;
   (void)d3;
   (void)d4;
   (void)d6;
   (void)d5;
   (void)c3;
   (void)q10;
   (void)q11;
   (void)q14;
   (void)q15;
   (void)q16;
   (void)q19;
   (void)q20;
   (void)q21;
   (void)q22;
   (void)q23;
   (void)b1;
   (void)b2;
   (void)b3;
   (void)b4;
   (void)b5;
   (void)b6;
   (void)b7;
   (void)b8;
   (void)b9;
   (void)b10;
   (void)q3;
   (void)q4;
   (void)q6;
   (void)q7;
   (void)q8;

#endif

#ifndef BOOST_NO_CXX14_CONSTEXPR

   constexpr qt q12 = c2 + q1;
   constexpr qt q13 = q1 + c2;

   constexpr qt q17 = c2 - q1;
   constexpr qt q18 = q1 - c2;

   constexpr qt q24 = full_constexpr_test(q5, q5 + 1, 3.2, q5.C_component_1());

   (void)q12;
   (void)q13;
   (void)q17;
   (void)q18;
   (void)q24;

#endif

   return 0;
}
