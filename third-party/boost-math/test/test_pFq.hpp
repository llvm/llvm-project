// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/special_functions/relative_difference.hpp>

#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

#ifndef SC_
#define SC_(x) BOOST_MATH_BIG_CONSTANT(T, 1000000, x)
#endif

template <class Seq>
bool is_small_a(const Seq& a)
{
   if (a.size() == 1)
   {
      auto v = *a.begin();
      if ((v > -14) && (v < 1))
         return true;
   }
   return false;
}

template <class Seq>
bool has_negative_ab(const Seq& a, const Seq& b)
{
   for(auto p = a.begin(); p != a.end(); ++p)
   {
      if(*p < 0)
         return true;
   }
   for(auto p = b.begin(); p != b.end(); ++p)
   {
      if(*p < 0)
         return true;
   }
   return false;
}

template <class T>
void check_pFq_result(const T& result, const T& norm, const T& expect, const std::initializer_list<T>& a, const std::initializer_list<T>& b, const T& z)
{
   //
   // Ideally the error rate we calculate from comparing norm to result
   // should be larger than the actual error.  However, in practice even
   // if all the terms are positive and norm == result there will still
   // be a small error from the actual summation (we could work out how
   // much from the number of terms summed, but that's overkill for this)
   // so we add a small fudge factor when comparing errors:
   //
   T err = boost::math::relative_difference(result, expect);
   T found_err = norm / fabs(result);
   T fudge_factor = 25;
   if (is_small_a(a))
      fudge_factor *= 4;  // not sure why??
   if ((has_negative_ab(a, b)) || ((a.size() == 2) && (b.size() == 1)) || (boost::math::tools::epsilon<T>() < boost::math::tools::epsilon<double>()))
   {
      T min_err = boost::math::tools::epsilon<T>() * 600 / found_err;
      fudge_factor = (std::max)(fudge_factor, min_err);
   }
   if ((((err > fudge_factor * found_err) && (found_err < 1)) || (boost::math::isnan)(found_err)) && (!(boost::math::isinf)(result)))
   {
      std::cout << "Found error = " << err << " error from norm = " << found_err << std::endl;
      std::cout << "Testing fudge factor = " << fudge_factor << std::endl;
      std::cout << "  a = ";
      for (auto pa = a.begin(); pa != a.end(); ++pa)
         std::cout << *pa << ",";
      std::cout << "\n  b = ";
      for (auto pb = b.begin(); pb != b.end(); ++pb)
         std::cout << *pb << ",";
      std::cout << "\n  z = " << z << std::endl;
      //
      // This will fail if we've got here:
      //
      BOOST_CHECK_LE(err, fudge_factor * found_err);
      BOOST_CHECK(!(boost::math::isnan)(found_err));
   }
}

template <class T>
void test_spots_1F0(T, const char*)
{
   using std::pow;
   
   T tolerance = boost::math::tools::epsilon<T>() * 1000;

   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(2)), T(-1), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(4)), T(-27), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(0.5)), T(0.125), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(3) }, {}, T(0.5)), T(8), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(3) }, {}, T(2)), T(-1), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(3) }, {}, T(4)), T(T(-1) / 27), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(3) }, {}, T(-0.5)), pow(T(1.5), -3), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(3) }, {}, T(-2)), T(1 / T(27)), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(3) }, {}, T(-4)), T(T(1) / 125), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(-0.5)), pow(T(1.5), 3), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(-2)), T(27), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(-4)), T(125), tolerance);

   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({ T(3) }, {}, T(1)), std::domain_error);
   //BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({ T(-3) }, {}, T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({ T(3.25) }, {}, T(1)), std::domain_error);
   //BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({ T(-3.25) }, {}, T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({ T(3.25) }, {}, T(2)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({ T(-3.25) }, {}, T(2)), std::domain_error);
}

template <class T>
void test_spots_0F1(T, const char*)
{
   T tolerance = boost::math::tools::epsilon<T>() * 50000;

   BOOST_CHECK_EQUAL(boost::math::hypergeometric_pFq({},  { T(3) }, T(0)), 1);
   BOOST_CHECK_EQUAL(boost::math::hypergeometric_pFq({}, { T(-3) }, T(0)), 1);
   //BOOST_CHECK_EQUAL(boost::math::hypergeometric_pFq({}, { T(0) }, T(0)), 1);

   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({}, { T(0) }, T(-1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({}, { T(-1) }, T(-1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_pFq({}, { T(-10) }, T(-5)), std::domain_error);

   static const std::array<std::array<T, 3>, 35> hypergeometric_pFq_integer_data = { {
      { SC_(4.0), SC_(-20.0),  SC_(-0.012889714201783047561923257996127233830940165138385) },
      { SC_(8.0), SC_(-20.0),  SC_(0.046498609282365144223175012935939437508273248399881) },
      { SC_(12.0), SC_(-20.0),  SC_(0.16608847431869756642136191351311569335145459224622) },
      { SC_(16.0), SC_(-20.0),  SC_(0.27230484709157170329168048388841880599105216477631) },
      //{ SC_(20.0), SC_(-20.0),  SC_(0.35865872656868844615709101792040025253126126604266) },
      { SC_(4.0), SC_(-16.0),  SC_(-0.027293644412433023379286103818840667403690937153604) },
      { SC_(8.0), SC_(-16.0),  SC_(0.098618710511372349330666801041676087431136532039702) },
      { SC_(12.0), SC_(-16.0),  SC_(0.24360114226383905073379763460037817885919457531523) },
      //{ SC_(16.0), SC_(-16.0),  SC_(0.35635186318802906043824855864337727878754460163525) },
      //{ SC_(20.0), SC_(-16.0),  SC_(0.44218381382689101428948260613085371477815110358789) },
      { SC_(4.0), SC_(-12.0),  SC_(-0.021743572290699436419371120781513860006290363262907) },
      { SC_(8.0), SC_(-12.0),  SC_(0.19025625754362006866949730683824627505504067855043) },
      //{ SC_(12.0), SC_(-12.0),  SC_(0.35251228238278927379621049815222218665165551016489) },
      //{ SC_(16.0), SC_(-12.0),  SC_(0.46415411486674623230458980010115972932474705884865) },
      //{ SC_(20.0), SC_(-12.0),  SC_(0.54394918325286018927327004362535051310016558628741) },
      { SC_(4.0), SC_(-8.0),  SC_(0.056818744289274872033266550620647787396712125304880) },
      //{ SC_(8.0), SC_(-8.0),  SC_(0.34487371876996263249797701802458885718691612997456) },
      //{ SC_(12.0), SC_(-8.0),  SC_(0.50411654015891701804499796523449656998841355305043) },
      //{ SC_(16.0), SC_(-8.0),  SC_(0.60191459981670594041254437708158847428118361245442) },
      //{ SC_(20.0), SC_(-8.0),  SC_(0.66770752550930138035694866478078941681114294465418) },
      //{ SC_(4.0), SC_(-4.0),  SC_(0.32262860540671645526863760914000166725449779629143) },
      //{ SC_(8.0), SC_(-4.0),  SC_(0.59755773349355150397404772151441126513126998265958) },
      //{ SC_(12.0), SC_(-4.0),  SC_(0.71337465206009117934071859694314971137807212605147) },
      //{ SC_(16.0), SC_(-4.0),  SC_(0.77734333649378860739496954157535257278092349684783) },
      //{ SC_(20.0), SC_(-4.0),  SC_(0.81794177985447769150469288350369205683856312760890) },

      { SC_(4.0), SC_(4.0),  SC_(2.5029568338152582758923890008139391395035041790831) },
      { SC_(8.0), SC_(4.0),  SC_(1.6273673128576761227855719910743734060605725722129) },
      { SC_(12.0), SC_(4.0),  SC_(1.3898419290864057799739567227851793491657442624207) },
      { SC_(16.0), SC_(4.0),  SC_(1.2817098157957427946677711269410726972209834860612) },
      { SC_(20.0), SC_(4.0),  SC_(1.2202539302152377230940386181201477276788392792437) },
      { SC_(4.0), SC_(8.0),  SC_(5.5616961007411965409200003309686924059253894118586) },
      { SC_(8.0), SC_(8.0),  SC_(2.5877053985451664722152913482683136948296873738479) },
      { SC_(12.0), SC_(8.0),  SC_(1.9166410733572697158003086323981583993970490592046) },
      { SC_(16.0), SC_(8.0),  SC_(1.6370675016890669952237854163997946987362497613701) },
      { SC_(20.0), SC_(8.0),  SC_(1.4862852701827990444915220582410007454379891584086) },
      { SC_(4.0), SC_(12.0),  SC_(11.419268276211177842169936131590385979116019595164) },
      { SC_(8.0), SC_(12.0),  SC_(4.0347215359576567066789638314925802225312840819037) },
      { SC_(12.0), SC_(12.0),  SC_(2.6242497527837800417573064942486918368886996538285) },
      { SC_(16.0), SC_(12.0),  SC_(2.0840468784170876805932772732753387258909164486511) },
      { SC_(20.0), SC_(12.0),  SC_(1.8071042457762091748544382847762106786633952487005) },
      { SC_(4.0), SC_(16.0),  SC_(22.132051970576036053853444648907108439504682530918) },
      { SC_(8.0), SC_(16.0),  SC_(6.1850485247748975008808779795786699492711191898792) },
      { SC_(12.0), SC_(16.0),  SC_(3.5694322843488018916484224923627864928705138154372) },
      { SC_(16.0), SC_(16.0),  SC_(2.6447371137201451261118187672029372265909501355722) },
      { SC_(20.0), SC_(16.0),  SC_(2.1934058398888071720297525592515838555602675797235) },
      { SC_(4.0), SC_(20.0),  SC_(41.021743268279206331672552645354782698296383424328) },
      { SC_(8.0), SC_(20.0),  SC_(9.3414225299809886395081381945971250426599939097753) },
      { SC_(12.0), SC_(20.0),  SC_(4.8253866205826406499959001774187695527272168375992) },
      { SC_(16.0), SC_(20.0),  SC_(3.3462305133519485784864062004430532216764447939942) },
      { SC_(20.0), SC_(20.0),  SC_(2.6578698872220394617444624241257799193518140676691) },
      } };

   for (auto row = hypergeometric_pFq_integer_data.begin(); row != hypergeometric_pFq_integer_data.end(); ++row)
   {
      BOOST_CHECK_CLOSE(boost::math::hypergeometric_pFq({},  { (*row)[0] }, (*row)[1]), (*row)[2], tolerance);
   }
}

template <class T>
void test_spots_1F1(T, const char*)
{
#include "hypergeometric_1F1.ipp"

   for (auto row = hypergeometric_1F1.begin(); row != hypergeometric_1F1.end(); ++row)
   {
      try {
         T norm;
         T result = boost::math::hypergeometric_pFq({ (*row)[0] }, { (*row)[1] }, (*row)[2], &norm);
         check_pFq_result(result, norm, (*row)[3], { (*row)[0] }, { (*row)[1] }, (*row)[2]);
      }
      catch (const boost::math::evaluation_error&) {}
   }
}

template <class T>
void test_spots_1F1_b(T, const char*)
{
#include "hypergeometric_1F1_big.ipp"

   for (auto row = hypergeometric_1F1_big.begin(); row != hypergeometric_1F1_big.end(); ++row)
   {
      try {
         T norm;
         T result = boost::math::hypergeometric_pFq({ (*row)[0] }, { (*row)[1] }, (*row)[2], &norm);
         check_pFq_result(result, norm, (*row)[3], { (*row)[0] }, { (*row)[1] }, (*row)[2]);
      }
      catch (const boost::math::evaluation_error&) {}
   }
}

template <class T>
void test_spots_2F1(T, const char*)
{
#include "hypergeometric_2F1.ipp"

   for (auto row = hypergeometric_2F1.begin(); row != hypergeometric_2F1.end(); ++row)
   {
      try {
         T norm;
         T result = boost::math::hypergeometric_pFq({ (*row)[0], (*row)[1] }, { (*row)[2] }, (*row)[3], &norm);
         check_pFq_result(result, norm, (*row)[4], { (*row)[0], (*row)[1] }, { (*row)[2] }, (*row)[3]);
      }
      catch (const boost::math::evaluation_error&) {}
   }
}

template <class T>
void test_spots_0F2(T, const char*)
{
#include "hypergeometric_0F2.ipp"

   for (auto row = hypergeometric_0F2.begin(); row != hypergeometric_0F2.end(); ++row)
   {
      try {
         T norm;
         T result = boost::math::hypergeometric_pFq({}, { (*row)[0], (*row)[1] }, (*row)[2], &norm);
         check_pFq_result(result, norm, (*row)[3], {}, { (*row)[0], (*row)[1] }, (*row)[2]);
      }
      catch (const boost::math::evaluation_error&) {}
   }
}

template <class T>
void test_spots_1F2(T, const char*)
{
#include "hypergeometric_1F2.ipp"

   for (auto row = hypergeometric_1F2.begin(); row != hypergeometric_1F2.end(); ++row)
   {
      try {
         T norm;
         T result = boost::math::hypergeometric_pFq({ (*row)[0] }, { (*row)[1], (*row)[2] }, (*row)[3], &norm);
         check_pFq_result(result, norm, (*row)[4], { (*row)[0] }, { (*row)[1], (*row)[2] }, (*row)[3]);
      }
      catch (const boost::math::evaluation_error&) {}
   }
}

template <class T>
void test_spots_2F2(T, const char*)
{
#include "hypergeometric_2F2.ipp"

   for (auto row = hypergeometric_2F2.begin(); row != hypergeometric_2F2.end(); ++row)
   {
      try {
         T norm;
         T result = boost::math::hypergeometric_pFq({ (*row)[0], (*row)[1] }, { (*row)[2], (*row)[3] }, (*row)[4], &norm);
         check_pFq_result(result, norm, (*row)[5], { (*row)[0], (*row)[1] }, { (*row)[2], (*row)[3] }, (*row)[4]);
      }
      catch (const boost::math::evaluation_error&) {}
   }
}

template <class T>
void test_spots(T z, const char* type_name)
{
   test_spots_1F0(z, type_name);
   test_spots_0F1(z, type_name);
   test_spots_1F1(z, type_name);
   test_spots_1F1_b(z, type_name);
   test_spots_0F2(z, type_name);
   test_spots_1F2(z, type_name);
   test_spots_2F2(z, type_name);
   test_spots_2F1(z, type_name);
}
