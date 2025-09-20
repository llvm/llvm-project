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

#ifndef BOOST_MATH_ELLINT_2_HPP
#define BOOST_MATH_ELLINT_2_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <boost/math/special_functions/ellint_rg.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/workaround.hpp>
#include <boost/math/special_functions/round.hpp>

// Elliptic integrals (complete and incomplete) of the second kind
// Carlson, Numerische Mathematik, vol 33, 1 (1979)

namespace boost { namespace math {

template <class T1, class T2, class Policy>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi, const Policy& pol);

namespace detail{

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_e_imp(T k, const Policy& pol, const boost::math::integral_constant<int, 0>&);
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_e_imp(T k, const Policy& pol, const boost::math::integral_constant<int, 1>&);
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_e_imp(T k, const Policy& pol, const boost::math::integral_constant<int, 2>&);

// Elliptic integral (Legendre form) of the second kind
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED T ellint_e_imp(T phi, T k, const Policy& pol)
{
    BOOST_MATH_STD_USING
    using namespace boost::math::tools;
    using namespace boost::math::constants;

    bool invert = false;
    if (phi == 0)
       return 0;

    if(phi < 0)
    {
       phi = fabs(phi);
       invert = true;
    }

    T result;

    if(phi >= tools::max_value<T>())
    {
       // Need to handle infinity as a special case:
       result = policies::raise_overflow_error<T>("boost::math::ellint_e<%1%>(%1%,%1%)", nullptr, pol);
    }
    else if(phi > 1 / tools::epsilon<T>())
    {
       typedef boost::math::integral_constant<int,
          boost::math::is_floating_point<T>::value&& boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 54) ? 0 :
          boost::math::is_floating_point<T>::value && boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 64) ? 1 : 2
       > precision_tag_type;
       // Phi is so large that phi%pi is necessarily zero (or garbage),
       // just return the second part of the duplication formula:
       result = 2 * phi * ellint_e_imp(k, pol, precision_tag_type()) / constants::pi<T>();
    }
    else if(k == 0)
    {
       return invert ? T(-phi) : phi;
    }
    else if(fabs(k) == 1)
    {
       //
       // For k = 1 ellipse actually turns to a line and every pi/2 in phi is exactly 1 in arc length
       // Periodicity though is in pi, curve follows sin(pi) for 0 <= phi <= pi/2 and then
       // 2 - sin(pi- phi) = 2 + sin(phi - pi) for pi/2 <= phi <= pi, so general form is:
       //
       // 2n + sin(phi - n * pi) ; |phi - n * pi| <= pi / 2
       //
       T m = boost::math::round(phi / boost::math::constants::pi<T>());
       T remains = phi - m * boost::math::constants::pi<T>();
       T value = 2 * m + sin(remains);

       // negative arc length for negative phi
       return invert ? -value : value;
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
       T rphi = boost::math::tools::fmod_workaround(phi, T(constants::half_pi<T>()));
       T m = boost::math::round((phi - rphi) / constants::half_pi<T>());
       int s = 1;
       if(boost::math::tools::fmod_workaround(m, T(2)) > T(0.5))
       {
          m += 1;
          s = -1;
          rphi = constants::half_pi<T>() - rphi;
       }
       T k2 = k * k;
       if(boost::math::pow<3>(rphi) * k2 / 6 < tools::epsilon<T>() * fabs(rphi))
       {
          // See http://functions.wolfram.com/EllipticIntegrals/EllipticE2/06/01/03/0001/
          result = s * rphi;
       }
       else
       {
          // http://dlmf.nist.gov/19.25#E10
          T sinp = sin(rphi);
          if (k2 * sinp * sinp >= 1)
          {
             return policies::raise_domain_error<T>("boost::math::ellint_2<%1%>(%1%, %1%)", "The parameter k is out of range, got k = %1%", k, pol);
          }
          T cosp = cos(rphi);
          T c = 1 / (sinp * sinp);
          T cm1 = cosp * cosp / (sinp * sinp);  // c - 1
          result = s * ((1 - k2) * ellint_rf_imp(cm1, T(c - k2), c, pol) + k2 * (1 - k2) * ellint_rd(cm1, c, T(c - k2), pol) / 3 + k2 * sqrt(cm1 / (c * (c - k2))));
       }
       if (m != 0)
       {
          typedef boost::math::integral_constant<int,
             boost::math::is_floating_point<T>::value&& boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 54) ? 0 :
             boost::math::is_floating_point<T>::value && boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 64) ? 1 : 2
          > precision_tag_type;
          result += m * ellint_e_imp(k, pol, precision_tag_type());
       }
    }
    return invert ? T(-result) : result;
}

// Complete elliptic integral (Legendre form) of the second kind
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED T ellint_e_imp(T k, const Policy& pol, boost::math::integral_constant<int, 2> const&)
{
    BOOST_MATH_STD_USING
    using namespace boost::math::tools;

    if (abs(k) > 1)
    {
       return policies::raise_domain_error<T>("boost::math::ellint_e<%1%>(%1%)", "Got k = %1%, function requires |k| <= 1", k, pol);
    }
    if (abs(k) == 1)
    {
        return static_cast<T>(1);
    }

    T x = 0;
    T t = k * k;
    T y = 1 - t;
    T z = 1;
    T value = 2 * ellint_rg_imp(x, y, z, pol);

    return value;
}
//
// Special versions for double and 80-bit long double precision,
// double precision versions use the coefficients from:
// "Fast computation of complete elliptic integrals and Jacobian elliptic functions",
// Celestial Mechanics and Dynamical Astronomy, April 2012.
// 
// Higher precision coefficients for 80-bit long doubles can be calculated
// using for example:
// Table[N[SeriesCoefficient[ EllipticE [ m ], { m, 875/1000, i} ], 20], {i, 0, 24}]
// and checking the value of the first neglected term with:
// N[SeriesCoefficient[ EllipticE [ m ], { m, 875/1000, 24} ], 20] * (2.5/100)^24
// 
// For m > 0.9 we don't use the method of the paper above, but simply call our
// existing routines.
//
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_e_imp(T k, const Policy& pol, boost::math::integral_constant<int, 0> const&)
{
   BOOST_MATH_STD_USING
   using namespace boost::math::tools;

   T m = k * k;
   switch (static_cast<int>(20 * m))
   {
   case 0:
   case 1:
   //if (m < 0.1)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.550973351780472328),
         -static_cast<T>(0.400301020103198524),
         -static_cast<T>(0.078498619442941939),
         -static_cast<T>(0.034318853117591992),
         -static_cast<T>(0.019718043317365499),
         -static_cast<T>(0.013059507731993309),
         -static_cast<T>(0.009442372874146547),
         -static_cast<T>(0.007246728512402157),
         -static_cast<T>(0.005807424012956090),
         -static_cast<T>(0.004809187786009338),
         -static_cast<T>(0.004086399233255150)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.05));
   }
   case 2:
   case 3:
   //else if (m < 0.2)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.510121832092819728),
         -static_cast<T>(0.417116333905867549),
         -static_cast<T>(0.090123820404774569),
         -static_cast<T>(0.043729944019084312),
         -static_cast<T>(0.027965493064761785),
         -static_cast<T>(0.020644781177568105),
         -static_cast<T>(0.016650786739707238),
         -static_cast<T>(0.014261960828842520),
         -static_cast<T>(0.012759847429264803),
         -static_cast<T>(0.011799303775587354),
         -static_cast<T>(0.011197445703074968)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.15));
   }
   case 4:
   case 5:
   //else if (m < 0.3)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.467462209339427155),
         -static_cast<T>(0.436576290946337775),
         -static_cast<T>(0.105155557666942554),
         -static_cast<T>(0.057371843593241730),
         -static_cast<T>(0.041391627727340220),
         -static_cast<T>(0.034527728505280841),
         -static_cast<T>(0.031495443512532783),
         -static_cast<T>(0.030527000890325277),
         -static_cast<T>(0.030916984019238900),
         -static_cast<T>(0.032371395314758122),
         -static_cast<T>(0.034789960386404158)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.25));
   }
   case 6:
   case 7:
   //else if (m < 0.4)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.422691133490879171),
         -static_cast<T>(0.459513519621048674),
         -static_cast<T>(0.125250539822061878),
         -static_cast<T>(0.078138545094409477),
         -static_cast<T>(0.064714278472050002),
         -static_cast<T>(0.062084339131730311),
         -static_cast<T>(0.065197032815572477),
         -static_cast<T>(0.072793895362578779),
         -static_cast<T>(0.084959075171781003),
         -static_cast<T>(0.102539850131045997),
         -static_cast<T>(0.127053585157696036),
         -static_cast<T>(0.160791120691274606)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.35));
   }
   case 8:
   case 9:
   //else if (m < 0.5)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.375401971871116291),
         -static_cast<T>(0.487202183273184837),
         -static_cast<T>(0.153311701348540228),
         -static_cast<T>(0.111849444917027833),
         -static_cast<T>(0.108840952523135768),
         -static_cast<T>(0.122954223120269076),
         -static_cast<T>(0.152217163962035047),
         -static_cast<T>(0.200495323642697339),
         -static_cast<T>(0.276174333067751758),
         -static_cast<T>(0.393513114304375851),
         -static_cast<T>(0.575754406027879147),
         -static_cast<T>(0.860523235727239756),
         -static_cast<T>(1.308833205758540162)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.45));
   }
   case 10:
   case 11:
   //else if (m < 0.6)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.325024497958230082),
         -static_cast<T>(0.521727647557566767),
         -static_cast<T>(0.194906430482126213),
         -static_cast<T>(0.171623726822011264),
         -static_cast<T>(0.202754652926419141),
         -static_cast<T>(0.278798953118534762),
         -static_cast<T>(0.420698457281005762),
         -static_cast<T>(0.675948400853106021),
         -static_cast<T>(1.136343121839229244),
         -static_cast<T>(1.976721143954398261),
         -static_cast<T>(3.531696773095722506),
         -static_cast<T>(6.446753640156048150),
         -static_cast<T>(11.97703130208884026)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.55));
   }
   case 12:
   case 13:
   //else if (m < 0.7)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.270707479650149744),
         -static_cast<T>(0.566839168287866583),
         -static_cast<T>(0.262160793432492598),
         -static_cast<T>(0.292244173533077419),
         -static_cast<T>(0.440397840850423189),
         -static_cast<T>(0.774947641381397458),
         -static_cast<T>(1.498870837987561088),
         -static_cast<T>(3.089708310445186667),
         -static_cast<T>(6.667595903381001064),
         -static_cast<T>(14.89436036517319078),
         -static_cast<T>(34.18120574251449024),
         -static_cast<T>(80.15895841905397306),
         -static_cast<T>(191.3489480762984920),
         -static_cast<T>(463.5938853480342030),
         -static_cast<T>(1137.380822169360061)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.65));
   }
   case 14:
   case 15:
   //else if (m < 0.8)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.211056027568459525),
         -static_cast<T>(0.630306413287455807),
         -static_cast<T>(0.387166409520669145),
         -static_cast<T>(0.592278235311934603),
         -static_cast<T>(1.237555584513049844),
         -static_cast<T>(3.032056661745247199),
         -static_cast<T>(8.181688221573590762),
         -static_cast<T>(23.55507217389693250),
         -static_cast<T>(71.04099935893064956),
         -static_cast<T>(221.8796853192349888),
         -static_cast<T>(712.1364793277635425),
         -static_cast<T>(2336.125331440396407),
         -static_cast<T>(7801.945954775964673),
         -static_cast<T>(26448.19586059191933),
         -static_cast<T>(90799.48341621365251),
         -static_cast<T>(315126.0406449163424),
         -static_cast<T>(1104011.344311591159)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.75));
   }
   case 16:
   //else if (m < 0.85)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.161307152196282836),
         -static_cast<T>(0.701100284555289548),
         -static_cast<T>(0.580551474465437362),
         -static_cast<T>(1.243693061077786614),
         -static_cast<T>(3.679383613496634879),
         -static_cast<T>(12.81590924337895775),
         -static_cast<T>(49.25672530759985272),
         -static_cast<T>(202.1818735434090269),
         -static_cast<T>(869.8602699308701437),
         -static_cast<T>(3877.005847313289571),
         -static_cast<T>(17761.70710170939814),
         -static_cast<T>(83182.69029154232061),
         -static_cast<T>(396650.4505013548170),
         -static_cast<T>(1920033.413682634405)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.825));
   }
   case 17:
   //else if (m < 0.90)
   {
      constexpr T coef[] =
      {
         static_cast<T>(1.124617325119752213),
         -static_cast<T>(0.770845056360909542),
         -static_cast<T>(0.844794053644911362),
         -static_cast<T>(2.490097309450394453),
         -static_cast<T>(10.23971741154384360),
         -static_cast<T>(49.74900546551479866),
         -static_cast<T>(267.0986675195705196),
         -static_cast<T>(1532.665883825229947),
         -static_cast<T>(9222.313478526091951),
         -static_cast<T>(57502.51612140314030),
         -static_cast<T>(368596.1167416106063),
         -static_cast<T>(2415611.088701091428),
         -static_cast<T>(16120097.81581656797),
         -static_cast<T>(109209938.5203089915),
         -static_cast<T>(749380758.1942496220),
         -static_cast<T>(5198725846.725541393),
         -static_cast<T>(36409256888.12139973)
      };
      return boost::math::tools::evaluate_polynomial(coef, m - static_cast<T>(0.875));
   }
   default:
      //
      // All cases where m > 0.9
      // including all error handling:
      //
      return ellint_e_imp(k, pol, boost::math::integral_constant<int, 2>());
   }
}
template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED BOOST_MATH_FORCEINLINE T ellint_e_imp(T k, const Policy& pol, boost::math::integral_constant<int, 1> const&)
{
   BOOST_MATH_STD_USING
   using namespace boost::math::tools;

   T m = k * k;
   switch (static_cast<int>(20 * m))
   {
   case 0:
   case 1:
      //if (m < 0.1)
   {
      constexpr T coef[] =
      {
         1.5509733517804723277L,
         -0.40030102010319852390L,
         -0.078498619442941939212L,
         -0.034318853117591992417L,
         -0.019718043317365499309L,
         -0.013059507731993309191L,
         -0.0094423728741465473894L,
         -0.0072467285124021568126L,
         -0.0058074240129560897940L,
         -0.0048091877860093381762L,
         -0.0040863992332551506768L,
         -0.0035450302604139562644L,
         -0.0031283511188028336315L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.05L);
   }
   case 2:
   case 3:
      //else if (m < 0.2)
   {
      constexpr T coef[] =
      {
         1.5101218320928197276L,
         -0.41711633390586754922L,
         -0.090123820404774568894L,
         -0.043729944019084311555L,
         -0.027965493064761784548L,
         -0.020644781177568105268L,
         -0.016650786739707238037L,
         -0.014261960828842519634L,
         -0.012759847429264802627L,
         -0.011799303775587354169L,
         -0.011197445703074968018L,
         -0.010850368064799902735L,
         -0.010696133481060989818L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.15L);
   }
   case 4:
   case 5:
      //else if (m < 0.3L)
   {
      constexpr T coef[] =
      {
         1.4674622093394271555L,
         -0.43657629094633777482L,
         -0.10515555766694255399L,
         -0.057371843593241729895L,
         -0.041391627727340220236L,
         -0.034527728505280841188L,
         -0.031495443512532782647L,
         -0.030527000890325277179L,
         -0.030916984019238900349L,
         -0.032371395314758122268L,
         -0.034789960386404158240L,
         -0.038182654612387881967L,
         -0.042636187648900252525L,
         -0.048302272505241634467
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.25L);
   }
   case 6:
   case 7:
      //else if (m < 0.4L)
   {
      constexpr T coef[] =
      {
         1.4226911334908791711L,
         -0.45951351962104867394L,
         -0.12525053982206187849L,
         -0.078138545094409477156L,
         -0.064714278472050001838L,
         -0.062084339131730310707L,
         -0.065197032815572476910L,
         -0.072793895362578779473L,
         -0.084959075171781003264L,
         -0.10253985013104599679L,
         -0.12705358515769603644L,
         -0.16079112069127460621L,
         -0.20705400012405941376L,
         -0.27053164884730888948L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.35L);
   }
   case 8:
   case 9:
      //else if (m < 0.5L)
   {
      constexpr T coef[] =
      {
         1.3754019718711162908L,
         -0.48720218327318483652L,
         -0.15331170134854022753L,
         -0.11184944491702783273L,
         -0.10884095252313576755L,
         -0.12295422312026907610L,
         -0.15221716396203504746L,
         -0.20049532364269733857L,
         -0.27617433306775175837L,
         -0.39351311430437585139L,
         -0.57575440602787914711L,
         -0.86052323572723975634L,
         -1.3088332057585401616L,
         -2.0200280559452241745L,
         -3.1566019548237606451L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.45L);
   }
   case 10:
   case 11:
      //else if (m < 0.6L)
   {
      constexpr T coef[] =
      {
         1.3250244979582300818L,
         -0.52172764755756676713L,
         -0.19490643048212621262L,
         -0.17162372682201126365L,
         -0.20275465292641914128L,
         -0.27879895311853476205L,
         -0.42069845728100576224L,
         -0.67594840085310602110L,
         -1.1363431218392292440L,
         -1.9767211439543982613L,
         -3.5316967730957225064L,
         -6.4467536401560481499L,
         -11.977031302088840261L,
         -22.581360948073964469L,
         -43.109479829481450573L,
         -83.186290908288807424L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.55L);
   }
   case 12:
   case 13:
      //else if (m < 0.7L)
   {
      constexpr T coef[] =
      {
         1.2707074796501497440L,
         -0.56683916828786658286L,
         -0.26216079343249259779L,
         -0.29224417353307741931L,
         -0.44039784085042318909L,
         -0.77494764138139745824L,
         -1.4988708379875610880L,
         -3.0897083104451866665L,
         -6.6675959033810010645L,
         -14.894360365173190775L,
         -34.181205742514490240L,
         -80.158958419053973056L,
         -191.34894807629849204L,
         -463.59388534803420301L,
         -1137.3808221693600606L,
         -2820.7073786352269339L,
         -7061.1382244658715621L,
         -17821.809331816437058L,
         -45307.849987201897801L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.65L);
   }
   case 14:
   case 15:
      //else if (m < 0.8L)
   {
      constexpr T coef[] =
      {
         1.2110560275684595248L,
         -0.63030641328745580709L,
         -0.38716640952066914514L,
         -0.59227823531193460257L,
         -1.2375555845130498445L,
         -3.0320566617452471986L,
         -8.1816882215735907624L,
         -23.555072173896932503L,
         -71.040999358930649565L,
         -221.87968531923498875L,
         -712.13647932776354253L,
         -2336.1253314403964072L,
         -7801.9459547759646726L,
         -26448.195860591919335L,
         -90799.483416213652512L,
         -315126.04064491634241L,
         -1.1040113443115911589e6L,
         -3.8998018348056769095e6L,
         -1.3876249116223745041e7L,
         -4.9694982823537861149e7L,
         -1.7900668836197342979e8L,
         -6.4817399873722371964e8L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.75L);
   }
   case 16:
      //else if (m < 0.85L)
   {
      constexpr T coef[] =
      {
         1.1613071521962828360L,
         -0.70110028455528954752L,
         -0.58055147446543736163L,
         -1.2436930610777866138L,
         -3.6793836134966348789L,
         -12.815909243378957753L,
         -49.256725307599852720L,
         -202.18187354340902693L,
         -869.86026993087014372L,
         -3877.0058473132895713L,
         -17761.707101709398174L,
         -83182.690291542320614L,
         -396650.45050135481698L,
         -1.9200334136826344054e6L,
         -9.4131321779500838352e6L,
         -4.6654858837335370627e7L,
         -2.3343549352617609390e8L,
         -1.1776928223045913454e9L,
         -5.9850851892915740401e9L,
         -3.0614702984618644983e10L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.825L);
   }
   case 17:
      //else if (m < 0.90L)
   {
      constexpr T coef[] =
      {
         1.1246173251197522132L,
         -0.77084505636090954218L,
         -0.84479405364491136236L,
         -2.4900973094503944527L,
         -10.239717411543843601L,
         -49.749005465514798660L,
         -267.09866751957051961L,
         -1532.6658838252299468L,
         -9222.3134785260919507L,
         -57502.516121403140303L,
         -368596.11674161060626L,
         -2.4156110887010914281e6L,
         -1.6120097815816567971e7L,
         -1.0920993852030899148e8L,
         -7.4938075819424962198e8L,
         -5.1987258467255413931e9L,
         -3.6409256888121399726e10L,
         -2.5711802891217393544e11L,
         -1.8290904062978796996e12L,
         -1.3096838781743248404e13L,
         -9.4325465851415135118e13L,
         -6.8291980829471896669e14L
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.875L);
   }
   default:
      //
      // All cases where m > 0.9
      // including all error handling:
      //
      return ellint_e_imp(k, pol, boost::math::integral_constant<int, 2>());
   }
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T>::type ellint_2(T k, const Policy& pol, const boost::math::true_type&)
{
   typedef typename tools::promote_args<T>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   typedef boost::math::integral_constant<int,
      boost::math::is_floating_point<T>::value&& boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 54) ? 0 :
      boost::math::is_floating_point<T>::value && boost::math::numeric_limits<T>::digits && (boost::math::numeric_limits<T>::digits <= 64) ? 1 : 2
   > precision_tag_type;
   return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_e_imp(static_cast<value_type>(k), pol, precision_tag_type()), "boost::math::ellint_2<%1%>(%1%)");
}

// Elliptic integral (Legendre form) of the second kind
template <class T1, class T2>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi, const boost::math::false_type&)
{
   return boost::math::ellint_2(k, phi, policies::policy<>());
}

} // detail

// Elliptic integral (Legendre form) of the second kind
template <class T1, class T2>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi)
{
   typedef typename policies::is_policy<T2>::type tag_type;
   return detail::ellint_2(k, phi, tag_type());
}

template <class T1, class T2, class Policy>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi, const Policy& pol)  // LCOV_EXCL_LINE gcc misses this but sees the function body, strange!
{
   typedef typename tools::promote_args<T1, T2>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_e_imp(static_cast<value_type>(phi), static_cast<value_type>(k), pol), "boost::math::ellint_2<%1%>(%1%,%1%)");
}


// Complete elliptic integral (Legendre form) of the second kind
template <typename T>
BOOST_MATH_GPU_ENABLED typename tools::promote_args<T>::type ellint_2(T k)
{
   return ellint_2(k, policies::policy<>());
}


}} // namespaces

#endif // BOOST_MATH_ELLINT_2_HPP

