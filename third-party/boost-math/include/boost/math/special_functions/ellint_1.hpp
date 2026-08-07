//  Copyright (c) 2006 Xiaogang Zhang
//  Copyright (c) 2006 John Maddock
//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  History:
//  XZ wrote the original of this file as part of the Google
//  Summer of Code 2006.  JM modified it to fit into the
//  Boost.Math conceptual framework better, and to ensure
//  that the code continues to work no matter how many digits
//  type T has.

#ifndef BOOST_MATH_ELLINT_1_HPP
#define BOOST_MATH_ELLINT_1_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/workaround.hpp>
#include <boost/math/special_functions/round.hpp>

// Elliptic integrals (complete and incomplete) of the first kind
// Carlson, Numerische Mathematik, vol 33, 1 (1979)

namespace boost { namespace math {

template <class T1, class T2, class Policy>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_1(T1 k, T2 phi, const Policy& pol);

namespace detail{

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_k_imp(T k, const Policy& pol, boost::math::integral_constant<int, 0> const&);
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_k_imp(T k, const Policy& pol, boost::math::integral_constant<int, 1> const&);
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_k_imp(T k, const Policy& pol, boost::math::integral_constant<int, 2> const&);
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_k_imp(T k, const Policy& pol, T one_minus_k2);

// Elliptic integral (Legendre form) of the first kind
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED T ellint_f_imp(T phi, T k, const Policy& pol, T one_minus_k2)
{
    BOOST_MATH_STD_USING
    using namespace boost::math::tools;
    using namespace boost::math::constants;

    constexpr auto function = "boost::math::ellint_f<%1%>(%1%,%1%)";
    BOOST_MATH_INSTRUMENT_VARIABLE(phi);
    BOOST_MATH_INSTRUMENT_VARIABLE(k);
    BOOST_MATH_INSTRUMENT_VARIABLE(function);

    bool invert = false;
    if(phi < 0)
    {
       BOOST_MATH_INSTRUMENT_VARIABLE(phi);
       phi = fabs(phi);
       invert = true;
    }

    T result;

    if(phi >= tools::max_value<T>())
    {
       // Need to handle infinity as a special case:
       result = policies::raise_overflow_error<T>(function, nullptr, pol);
       BOOST_MATH_INSTRUMENT_VARIABLE(result);
    }
    else if(phi > 1 / tools::epsilon<T>())
    {
       // Phi is so large that phi%pi is necessarily zero (or garbage),
       // just return the second part of the duplication formula:
       result = 2 * phi * ellint_k_imp(k, pol, one_minus_k2) / constants::pi<T>();
       BOOST_MATH_INSTRUMENT_VARIABLE(result);
    }
    else
    {
       // Carlson's algorithm works only for |phi| <= pi/2,
       // use the integrand's periodicity to normalize phi
       //
       // Xiaogang's original code used a cast to long long here
       // but that fails if T has more digits than a long long,
       // so rewritten to use fmod instead:
       //
       BOOST_MATH_INSTRUMENT_CODE("pi/2 = " << constants::pi<T>() / 2);
       T rphi = boost::math::tools::fmod_workaround(phi, T(constants::half_pi<T>()));
       BOOST_MATH_INSTRUMENT_VARIABLE(rphi);
       T m = boost::math::round((phi - rphi) / constants::half_pi<T>());
       BOOST_MATH_INSTRUMENT_VARIABLE(m);
       int s = 1;
       if(boost::math::tools::fmod_workaround(m, T(2)) > T(0.5))
       {
          m += 1;
          s = -1;
          rphi = constants::half_pi<T>() - rphi;
          BOOST_MATH_INSTRUMENT_VARIABLE(rphi);
       }
       T sinp = sin(rphi);
       sinp *= sinp;
       if (sinp * k * k >= 1)
       {
          return policies::raise_domain_error<T>(function,
             "Got k^2 * sin^2(phi) = %1%, but the function requires this < 1", sinp * k * k, pol);
       }
       T cosp = cos(rphi);
       cosp *= cosp;
       BOOST_MATH_INSTRUMENT_VARIABLE(sinp);
       BOOST_MATH_INSTRUMENT_VARIABLE(cosp);
       if(sinp > tools::min_value<T>())
       {
          BOOST_MATH_ASSERT(rphi != 0); // precondition, can't be true if sin(rphi) != 0.
          //
          // Use http://dlmf.nist.gov/19.25#E5, note that
          // c-1 simplifies to cot^2(rphi) which avoids cancellation.
          // Likewise c - k^2 is the same as (c - 1) + (1 - k^2).
          //
          T c = 1 / sinp;
          T c_minus_one = cosp / sinp;
          T arg2;
          if (k != 0)
          {
             T cross = fabs(c / (k * k));
             if ((cross > 0.9f) && (cross < 1.1f))
                arg2 = c_minus_one + one_minus_k2;
             else
                arg2 = c - k * k;
          }
          else
             arg2 = c;
          result = static_cast<T>(s * ellint_rf_imp(c_minus_one, arg2, c, pol));
       }
       else
          result = s * sin(rphi);
       BOOST_MATH_INSTRUMENT_VARIABLE(result);
       if(m != 0)
       {
          result += m * ellint_k_imp(k, pol, one_minus_k2);
          BOOST_MATH_INSTRUMENT_VARIABLE(result);
       }
    }
    return invert ? T(-result) : result;
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED inline T ellint_f_imp(T phi, T k, const Policy& pol)
{
   return ellint_f_imp(phi, k, pol, T(1 - k * k));
}

// Complete elliptic integral (Legendre form) of the first kind
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED T ellint_k_imp(T k, const Policy& pol, T one_minus_k2)
{
    BOOST_MATH_STD_USING
    using namespace boost::math::tools;

    constexpr auto function = "boost::math::ellint_k<%1%>(%1%)";

    if (abs(k) > 1)
    {
       return policies::raise_domain_error<T>(function, "Got k = %1%, function requires |k| <= 1", k, pol);
    }
    if (abs(k) == 1)
    {
       return policies::raise_overflow_error<T>(function, nullptr, pol);
    }

    T x = 0;
    T z = 1;
    T value = ellint_rf_imp(x, one_minus_k2, z, pol);

    return value;
}
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED inline T ellint_k_imp(T k, const Policy& pol, boost::math::integral_constant<int, 2> const&)
{
   return ellint_k_imp(k, pol, T(1 - k * k));
}

//
// Special versions for double and 80-bit long double precision,
// double precision versions use the coefficients from:
// "Fast computation of complete elliptic integrals and Jacobian elliptic functions",
// Celestial Mechanics and Dynamical Astronomy, April 2012.
// 
// Higher precision coefficients for 80-bit long doubles can be calculated
// using for example:
// Table[N[SeriesCoefficient[ EllipticK [ m ], { m, 875/1000, i} ], 20], {i, 0, 24}]
// and checking the value of the first neglected term with:
// N[SeriesCoefficient[ EllipticK [ m ], { m, 875/1000, 24} ], 20] * (2.5/100)^24
// 
// For m > 0.9 we don't use the method of the paper above, but simply call our
// existing routines.  The routine used in the above paper was tried (and is
// archived in the code below), but was found to have slightly higher error rates.
//
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_k_imp(T k, const Policy& pol, boost::math::integral_constant<int, 0> const&)
{
   BOOST_MATH_STD_USING
   using namespace boost::math::tools;

   T m = k * k;

   switch (static_cast<int>(m * 20))
   {
   case 0:
   case 1:
      //if (m < 0.1)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.591003453790792180),
         static_cast<T>(0.416000743991786912),
         static_cast<T>(0.245791514264103415),
         static_cast<T>(0.179481482914906162),
         static_cast<T>(0.144556057087555150),
         static_cast<T>(0.123200993312427711),
         static_cast<T>(0.108938811574293531),
         static_cast<T>(0.098853409871592910),
         static_cast<T>(0.091439629201749751),
         static_cast<T>(0.085842591595413900),
         static_cast<T>(0.081541118718303215),
         static_cast<T>(0.078199656811256481910)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.05));
   }
   case 2:
   case 3:
      //else if (m < 0.2)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.635256732264579992),
         static_cast<T>(0.471190626148732291),
         static_cast<T>(0.309728410831499587),
         static_cast<T>(0.252208311773135699),
         static_cast<T>(0.226725623219684650),
         static_cast<T>(0.215774446729585976),
         static_cast<T>(0.213108771877348910),
         static_cast<T>(0.216029124605188282),
         static_cast<T>(0.223255831633057896),
         static_cast<T>(0.234180501294209925),
         static_cast<T>(0.248557682972264071),
         static_cast<T>(0.266363809892617521)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.15));
   }
   case 4:
   case 5:
      //else if (m < 0.3)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.685750354812596043),
         static_cast<T>(0.541731848613280329),
         static_cast<T>(0.401524438390690257),
         static_cast<T>(0.369642473420889090),
         static_cast<T>(0.376060715354583645),
         static_cast<T>(0.405235887085125919),
         static_cast<T>(0.453294381753999079),
         static_cast<T>(0.520518947651184205),
         static_cast<T>(0.609426039204995055),
         static_cast<T>(0.724263522282908870),
         static_cast<T>(0.871013847709812357),
         static_cast<T>(1.057652872753547036)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.25));
   }
   case 6:
   case 7:
      //else if (m < 0.4)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.744350597225613243),
         static_cast<T>(0.634864275371935304),
         static_cast<T>(0.539842564164445538),
         static_cast<T>(0.571892705193787391),
         static_cast<T>(0.670295136265406100),
         static_cast<T>(0.832586590010977199),
         static_cast<T>(1.073857448247933265),
         static_cast<T>(1.422091460675497751),
         static_cast<T>(1.920387183402304829),
         static_cast<T>(2.632552548331654201),
         static_cast<T>(3.652109747319039160),
         static_cast<T>(5.115867135558865806),
         static_cast<T>(7.224080007363877411)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.35));
   }
   case 8:
   case 9:
      //else if (m < 0.5)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.813883936816982644),
         static_cast<T>(0.763163245700557246),
         static_cast<T>(0.761928605321595831),
         static_cast<T>(0.951074653668427927),
         static_cast<T>(1.315180671703161215),
         static_cast<T>(1.928560693477410941),
         static_cast<T>(2.937509342531378755),
         static_cast<T>(4.594894405442878062),
         static_cast<T>(7.330071221881720772),
         static_cast<T>(11.87151259742530180),
         static_cast<T>(19.45851374822937738),
         static_cast<T>(32.20638657246426863),
         static_cast<T>(53.73749198700554656),
         static_cast<T>(90.27388602940998849)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.45));
   }
   case 10:
   case 11:
      //else if (m < 0.6)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.898924910271553526),
         static_cast<T>(0.950521794618244435),
         static_cast<T>(1.151077589959015808),
         static_cast<T>(1.750239106986300540),
         static_cast<T>(2.952676812636875180),
         static_cast<T>(5.285800396121450889),
         static_cast<T>(9.832485716659979747),
         static_cast<T>(18.78714868327559562),
         static_cast<T>(36.61468615273698145),
         static_cast<T>(72.45292395127771801),
         static_cast<T>(145.1079577347069102),
         static_cast<T>(293.4786396308497026),
         static_cast<T>(598.3851815055010179),
         static_cast<T>(1228.420013075863451),
         static_cast<T>(2536.529755382764488)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.55));
   }
   case 12:
   case 13:
      //else if (m < 0.7)
   {
      constexpr T coef[] =
      {
         static_cast<T>(2.007598398424376302),
         static_cast<T>(1.248457231212347337),
         static_cast<T>(1.926234657076479729),
         static_cast<T>(3.751289640087587680),
         static_cast<T>(8.119944554932045802),
         static_cast<T>(18.66572130873555361),
         static_cast<T>(44.60392484291437063),
         static_cast<T>(109.5092054309498377),
         static_cast<T>(274.2779548232413480),
         static_cast<T>(697.5598008606326163),
         static_cast<T>(1795.716014500247129),
         static_cast<T>(4668.381716790389910),
         static_cast<T>(12235.76246813664335),
         static_cast<T>(32290.17809718320818),
         static_cast<T>(85713.07608195964685),
         static_cast<T>(228672.1890493117096),
         static_cast<T>(612757.2711915852774)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.65));
   }
   case 14:
   case 15:
      //else if (m < static_cast<T>(0.8))
   {
      constexpr T coef[] =
      {
         static_cast<T>(2.156515647499643235),
         static_cast<T>(1.791805641849463243),
         static_cast<T>(3.826751287465713147),
         static_cast<T>(10.38672468363797208),
         static_cast<T>(31.40331405468070290),
         static_cast<T>(100.9237039498695416),
         static_cast<T>(337.3268282632272897),
         static_cast<T>(1158.707930567827917),
         static_cast<T>(4060.990742193632092),
         static_cast<T>(14454.00184034344795),
         static_cast<T>(52076.66107599404803),
         static_cast<T>(189493.6591462156887),
         static_cast<T>(695184.5762413896145),
         static_cast<T>(2567994.048255284686),
         static_cast<T>(9541921.966748386322),
         static_cast<T>(35634927.44218076174),
         static_cast<T>(133669298.4612040871),
         static_cast<T>(503352186.6866284541),
         static_cast<T>(1901975729.538660119),
         static_cast<T>(7208915015.330103756)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.75));
   }
   case 16:
      //else if (m < static_cast<T>(0.85))
   {
      constexpr T coef[] =
      {
         static_cast<T>(2.318122621712510589),
         static_cast<T>(2.616920150291232841),
         static_cast<T>(7.897935075731355823),
         static_cast<T>(30.50239715446672327),
         static_cast<T>(131.4869365523528456),
         static_cast<T>(602.9847637356491617),
         static_cast<T>(2877.024617809972641),
         static_cast<T>(14110.51991915180325),
         static_cast<T>(70621.44088156540229),
         static_cast<T>(358977.2665825309926),
         static_cast<T>(1847238.263723971684),
         static_cast<T>(9600515.416049214109),
         static_cast<T>(50307677.08502366879),
         static_cast<T>(265444188.6527127967),
         static_cast<T>(1408862325.028702687),
         static_cast<T>(7515687935.373774627)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.825));
   }
   case 17:
      //else if (m < static_cast<T>(0.90))
   {
      constexpr T coef[] =
      {
         static_cast<T>(2.473596173751343912),
         static_cast<T>(3.727624244118099310),
         static_cast<T>(15.60739303554930496),
         static_cast<T>(84.12850842805887747),
         static_cast<T>(506.9818197040613935),
         static_cast<T>(3252.277058145123644),
         static_cast<T>(21713.24241957434256),
         static_cast<T>(149037.0451890932766),
         static_cast<T>(1043999.331089990839),
         static_cast<T>(7427974.817042038995),
         static_cast<T>(53503839.67558661151),
         static_cast<T>(389249886.9948708474),
         static_cast<T>(2855288351.100810619),
         static_cast<T>(21090077038.76684053),
         static_cast<T>(156699833947.7902014),
         static_cast<T>(1170222242422.439893),
         static_cast<T>(8777948323668.937971),
         static_cast<T>(66101242752484.95041),
         static_cast<T>(499488053713388.7989),
         static_cast<T>(37859743397240299.20)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.875));
   }
   default:
      //
      // This handles all cases where m > 0.9, 
      // including all error handling:
      //
      return ellint_k_imp(k, pol, boost::math::integral_constant<int, 2>());
#if 0
   else
   {
      T lambda_prime = (1 - sqrt(k)) / (2 * (1 + sqrt(k)));
      T k_prime = ellint_k(sqrt((1 - k) * (1 + k))); // K(m')
      T lambda_prime_4th = boost::math::pow<4>(lambda_prime);
      T q_prime = ((((((20910 * lambda_prime_4th) + 1707) * lambda_prime_4th + 150) * lambda_prime_4th + 15) * lambda_prime_4th + 2) * lambda_prime_4th + 1) * lambda_prime;
      /*T q_prime_2 = lambda_prime
         + 2 * boost::math::pow<5>(lambda_prime)
         + 15 * boost::math::pow<9>(lambda_prime)
         + 150 * boost::math::pow<13>(lambda_prime)
         + 1707 * boost::math::pow<17>(lambda_prime)
         + 20910 * boost::math::pow<21>(lambda_prime);*/
      return -log(q_prime) * k_prime / boost::math::constants::pi<T>();
   }
#endif
   }
}
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_k_imp(T k, const Policy& pol, boost::math::integral_constant<int, 1> const&)
{
   BOOST_MATH_STD_USING
   using namespace boost::math::tools;

   T m = k * k;
   switch (static_cast<int>(m * 20))
   {
   case 0:
   case 1:
   {
      constexpr T coef[] =
      {
         1.5910034537907921801L,
         0.41600074399178691174L,
         0.24579151426410341536L,
         0.17948148291490616181L,
         0.14455605708755514976L,
         0.12320099331242771115L,
         0.10893881157429353105L,
         0.098853409871592910399L,
         0.091439629201749751268L,
         0.085842591595413899672L,
         0.081541118718303214749L,
         0.078199656811256481910L,
         0.075592617535422415648L,
         0.073562939365441925050L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.05L);
   }
   case 2:
   case 3:
   {
      constexpr T coef[] =
      {
         1.6352567322645799924L,
         0.47119062614873229055L,
         0.30972841083149958708L,
         0.25220831177313569923L,
         0.22672562321968464974L,
         0.21577444672958597588L,
         0.21310877187734890963L,
         0.21602912460518828154L,
         0.22325583163305789567L,
         0.23418050129420992492L,
         0.24855768297226407136L,
         0.26636380989261752077L,
         0.28772845215611466775L,
         0.31290024539780334906L,
         0.34223105446381299902L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.15L);
   }
   case 4:
   case 5:
   {
      constexpr T coef[] =
      {
         1.6857503548125960429L,
         0.54173184861328032882L,
         0.40152443839069025682L,
         0.36964247342088908995L,
         0.37606071535458364462L,
         0.40523588708512591863L,
         0.45329438175399907924L,
         0.52051894765118420473L,
         0.60942603920499505544L,
         0.72426352228290886975L,
         0.87101384770981235737L,
         1.0576528727535470365L,
         1.2945970872087764321L,
         1.5953368253888783747L,
         1.9772844873556364793L,
         2.4628890581910021287L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.25L);
   }
   case 6:
   case 7:
   {
      constexpr T coef[] =
      {
         1.7443505972256132429L,
         0.63486427537193530383L,
         0.53984256416444553751L,
         0.57189270519378739093L,
         0.67029513626540610034L,
         0.83258659001097719939L,
         1.0738574482479332654L,
         1.4220914606754977514L,
         1.9203871834023048288L,
         2.6325525483316542006L,
         3.6521097473190391602L,
         5.1158671355588658061L,
         7.2240800073638774108L,
         10.270306349944787227L,
         14.685616935355757348L,
         21.104114212004582734L,
         30.460132808575799413L,
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.35L);
   }
   case 8:
   case 9:
   {
      constexpr T coef[] =
      {
         1.8138839368169826437L,
         0.76316324570055724607L,
         0.76192860532159583095L,
         0.95107465366842792679L,
         1.3151806717031612153L,
         1.9285606934774109412L,
         2.9375093425313787550L,
         4.5948944054428780618L,
         7.3300712218817207718L,
         11.871512597425301798L,
         19.458513748229377383L,
         32.206386572464268628L,
         53.737491987005546559L,
         90.273886029409988491L,
         152.53312130253275268L,
         259.02388747148299086L,
         441.78537518096201946L,
         756.39903981567380952L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.45L);
   }
   case 10:
   case 11:
   {
      constexpr T coef[] =
      {
         1.8989249102715535257L,
         0.95052179461824443490L,
         1.1510775899590158079L,
         1.7502391069863005399L,
         2.9526768126368751802L,
         5.2858003961214508892L,
         9.8324857166599797471L,
         18.787148683275595622L,
         36.614686152736981447L,
         72.452923951277718013L,
         145.10795773470691023L,
         293.47863963084970259L,
         598.38518150550101790L,
         1228.4200130758634505L,
         2536.5297553827644880L,
         5263.9832725075189576L,
         10972.138126273491753L,
         22958.388550988306870L,
         48203.103373625406989L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.55L);
   }
   case 12:
   case 13:
   {
      constexpr T coef[] =
      {
         2.0075983984243763017L,
         1.2484572312123473371L,
         1.9262346570764797287L,
         3.7512896400875876798L,
         8.1199445549320458022L,
         18.665721308735553611L,
         44.603924842914370633L,
         109.50920543094983774L,
         274.27795482324134804L,
         697.55980086063261629L,
         1795.7160145002471293L,
         4668.3817167903899100L,
         12235.762468136643348L,
         32290.178097183208178L,
         85713.076081959646847L,
         228672.18904931170958L,
         612757.27119158527740L,
         1.6483233976504668314e6L,
         4.4492251046211960936e6L,
         1.2046317340783185238e7L,
         3.2705187507963254185e7L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.65L);
   }
   case 14:
   case 15:
   {
      constexpr T coef[] =
      {
         2.1565156474996432354L,
         1.7918056418494632425L,
         3.8267512874657131470L,
         10.386724683637972080L,
         31.403314054680702901L,
         100.92370394986954165L,
         337.32682826322728966L,
         1158.7079305678279173L,
         4060.9907421936320917L,
         14454.001840343447947L,
         52076.661075994048028L,
         189493.65914621568866L,
         695184.57624138961450L,
         2.5679940482552846861e6L,
         9.5419219667483863221e6L,
         3.5634927442180761743e7L,
         1.3366929846120408712e8L,
         5.0335218668662845411e8L,
         1.9019757295386601192e9L,
         7.2089150153301037563e9L,
         2.7398741806339510931e10L,
         1.0439286724885300495e11L,
         3.9864875581513728207e11L,
         1.5254661585564745591e12L,
         5.8483259088850315936e12
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.75L);
   }
   case 16:
   {
      constexpr T coef[] =
      {
         2.3181226217125105894L,
         2.6169201502912328409L,
         7.8979350757313558232L,
         30.502397154466723270L,
         131.48693655235284561L,
         602.98476373564916170L,
         2877.0246178099726410L,
         14110.519919151803247L,
         70621.440881565402289L,
         358977.26658253099258L,
         1.8472382637239716844e6L,
         9.6005154160492141090e6L,
         5.0307677085023668786e7L,
         2.6544418865271279673e8L,
         1.4088623250287026866e9L,
         7.5156879353737746270e9L,
         4.0270783964955246149e10L,
         2.1662089325801126339e11L,
         1.1692489201929996116e12L,
         6.3306543358985679881e12
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.825L);
   }
   case 17:
   {
      constexpr T coef[] =
      {
         2.4735961737513439120L,
         3.7276242441180993105L,
         15.607393035549304964L,
         84.128508428058877470L,
         506.98181970406139349L,
         3252.2770581451236438L,
         21713.242419574342564L,
         149037.04518909327662L,
         1.0439993310899908390e6L,
         7.4279748170420389947e6L,
         5.3503839675586611510e7L,
         3.8924988699487084738e8L,
         2.8552883511008106195e9L,
         2.1090077038766840525e10L,
         1.5669983394779020136e11L,
         1.1702222424224398927e12L,
         8.7779483236689379709e12L,
         6.6101242752484950408e13L,
         4.9948805371338879891e14L,
         3.7859743397240299201e15L,
         2.8775996123036112296e16L,
         2.1926346839925760143e17L,
         1.6744985438468349361e18L,
         1.2814410112866546052e19L,
         9.8249807041031260167e19
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.875L);
   }
   default:
      //
      // All cases where m > 0.9
      // including all error handling:
      //
      return ellint_k_imp(k, pol, boost::math::integral_constant<int, 2>());
   }
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T>::type ellint_1(T k, const Policy& pol, const boost::math::true_type&)
{
   typedef typename tools::promote_args<T>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   typedef boost::math::integral_constant<int, 
#if defined(__clang_major__) && (__clang_major__ == 7)
      2
#else
      boost::math::is_floating_point<T>::value && boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 54) ? 0 :
      boost::math::is_floating_point<T>::value && boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 64) ? 1 : 2
#endif
   > precision_tag_type;
   return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_k_imp(static_cast<value_type>(k), pol, precision_tag_type()), "boost::math::ellint_1<%1%>(%1%)");
}

template <class T1, class T2>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_1(T1 k, T2 phi, const boost::math::false_type&)
{
   return boost::math::ellint_1(k, phi, policies::policy<>());
}

} // namespace detail

// Elliptic integral (Legendre form) of the first kind
template <class T1, class T2, class Policy>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_1(T1 k, T2 phi, const Policy& pol)  // LCOV_EXCL_LINE gcc misses this but sees the function body, strange!
{
   typedef typename tools::promote_args<T1, T2>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_f_imp(static_cast<value_type>(phi), static_cast<value_type>(k), pol), "boost::math::ellint_1<%1%>(%1%,%1%)");
}

template <class T1, class T2>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_1(T1 k, T2 phi)
{
   typedef typename policies::is_policy<T2>::type tag_type;
   return detail::ellint_1(k, phi, tag_type());
}

// Complete elliptic integral (Legendre form) of the first kind
template <typename T>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T>::type ellint_1(T k)
{
   return ellint_1(k, policies::policy<>());
}

}} // namespaces

#endif // BOOST_MATH_ELLINT_1_HPP

