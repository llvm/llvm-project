//  (C) Copyright John Maddock, 2024
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/math/special_functions/bessel_iterators.hpp>


BOOST_AUTO_TEST_CASE(test_main)
{
   using boost::math::bessel_j_backwards_iterator;
   using boost::math::bessel_i_backwards_iterator;
   using boost::math::bessel_i_forwards_iterator;

   BOOST_CHECK_THROW(bessel_j_backwards_iterator<double>(-2.3, 5.0), std::domain_error);
   BOOST_CHECK_THROW(bessel_j_backwards_iterator<double>(-2.3, 5.0, 2.0), std::domain_error);
   BOOST_CHECK_THROW(bessel_j_backwards_iterator<double>(-2.3, 5.0, 2.0, 2.0), std::domain_error);

   double tolerance = std::numeric_limits<double>::epsilon() * 5;

   {
      bessel_j_backwards_iterator<double> i(12, 2.0);
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_j(8, 2.0), tolerance);
      double v = *i++;
      BOOST_CHECK_CLOSE_FRACTION(v, boost::math::cyl_bessel_j(8, 2.0), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_j(7, 2.0), tolerance);
   }
   {
      bessel_j_backwards_iterator<double> i(12, 2.0, boost::math::cyl_bessel_j(12, 2.0));
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_j(8, 2.0), tolerance);
   }
   {
      bessel_j_backwards_iterator<double> i(12, 2.0, boost::math::cyl_bessel_j(13, 2.0), boost::math::cyl_bessel_j(12, 2.0));
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_j(8, 2.0), tolerance);
   }

   {
      bessel_i_backwards_iterator<double> i(12, 2.0);
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(8, 2.0), tolerance);
      double v = *i++;
      BOOST_CHECK_CLOSE_FRACTION(v, boost::math::cyl_bessel_i(8, 2.0), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(7, 2.0), tolerance);

      BOOST_CHECK_THROW(bessel_i_backwards_iterator<double>(-1.1, 2.0), std::domain_error);
      BOOST_CHECK_THROW(bessel_i_backwards_iterator<double>(-1.1, 2.0, 1.0), std::domain_error);
      BOOST_CHECK_THROW(bessel_i_backwards_iterator<double>(-1.1, 2.0, 1.0, 1.0), std::domain_error);
   }
   {
      bessel_i_backwards_iterator<double> i(12, 2.0, boost::math::cyl_bessel_i(12, 2.0));
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(8, 2.0), tolerance);
   }
   {
      bessel_i_backwards_iterator<double> i(12, 2.0, boost::math::cyl_bessel_i(13, 2.0), boost::math::cyl_bessel_i(12, 2.0));
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(8, 2.0), tolerance);
   }

   {
      bessel_i_forwards_iterator<double> i(-12, 2.0);
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(-8, 2.0), tolerance);
      double v = *i++;
      BOOST_CHECK_CLOSE_FRACTION(v, boost::math::cyl_bessel_i(-8, 2.0), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(-7, 2.0), tolerance);

      BOOST_CHECK_THROW(bessel_i_forwards_iterator<double>(1.1, 2.0), std::domain_error);
      BOOST_CHECK_THROW(bessel_i_forwards_iterator<double>(1.1, 2.0, 1.0), std::domain_error);
      BOOST_CHECK_THROW(bessel_i_forwards_iterator<double>(1.1, 2.0, 1.0, 1.0), std::domain_error);
   }
   {
      bessel_i_forwards_iterator<double> i(-12, 2.0, boost::math::cyl_bessel_i(-12, 2.0));
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(-8, 2.0), tolerance);
   }
   {
      bessel_i_forwards_iterator<double> i(-12, 2.0, boost::math::cyl_bessel_i(-13, 2.0), boost::math::cyl_bessel_i(-12, 2.0));
      for (unsigned j = 0; j < 4; ++j)
         ++i;
      BOOST_CHECK_CLOSE_FRACTION(*i, boost::math::cyl_bessel_i(-8, 2.0), tolerance);
   }
}
