//  (C) Copyright John Maddock 2006.
//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/special_functions/digamma.hpp>
#include "math_unit_test.hpp"

template <class T>
void test_spots(T, const char* t)
{
   std::cout << "Testing basic sanity checks for type " << t << std::endl;
   //
   // Basic sanity checks, tolerance is 3 epsilon:
   //
   T tolerance = 3;
   //
   // Special tolerance (200eps) for when we're very near the root,
   // and T has more than 64-bits in it's mantissa:
   //
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(0.125)), static_cast<T>(-8.3884926632958548678027429230863430000514460424495L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(0.5)), static_cast<T>(-1.9635100260214234794409763329987555671931596046604L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(1)), static_cast<T>(-0.57721566490153286060651209008240243104215933593992L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(1.5)), static_cast<T>(0.036489973978576520559023667001244432806840395339566L), tolerance * 40);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(1.5) - static_cast<T>(1)/32), static_cast<T>(0.00686541147073577672813890866512415766586241385896200579891429L), tolerance * 200);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(2)), static_cast<T>(0.42278433509846713939348790991759756895784066406008L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(8)), static_cast<T>(2.0156414779556099965363450527747404261006978069172L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(12)), static_cast<T>(2.4426616799758120167383652547949424463027180089374L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(22)), static_cast<T>(3.0681430398611966699248760264450329818421699570581L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(50)), static_cast<T>(3.9019896734278921969539597028823666609284424880275L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(500)), static_cast<T>(6.2136077650889917423827750552855712637776544784569L), tolerance);
   //
   // negative values:
   //
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(-0.125)), static_cast<T>(7.1959829284523046176757814502538535827603450463013L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(-10.125)), static_cast<T>(9.9480538258660761287008034071425343357982429855241L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(-10.875)), static_cast<T>(-5.1527360383841562620205965901515879492020193154231L), tolerance);
   CHECK_ULP_CLOSE(::boost::math::digamma(static_cast<T>(-1.5)), static_cast<T>(0.70315664064524318722569033366791109947350706200623L), tolerance);
}

int main()
{
   test_spots(0.0F, "float");
   test_spots(0.0, "double");

   return boost::math::test::report_errors();
}


