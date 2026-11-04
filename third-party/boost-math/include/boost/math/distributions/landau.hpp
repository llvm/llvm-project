//  Copyright Takuma Yoshimura 2024.
//  Copyright Matt Borland 2024
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_LANDAU_HPP
#define BOOST_STATS_LANDAU_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/distributions/detail/derived_accessors.hpp>
#include <boost/math/tools/rational.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/promotion.hpp>

#ifndef BOOST_MATH_HAS_NVRTC
#include <boost/math/distributions/fwd.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <utility>
#include <cmath>
#endif

namespace boost { namespace math {
template <class RealType, class Policy>
class landau_distribution;

namespace detail {

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 6.1179e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.62240126375351657026e-1),
            static_cast<RealType>(3.37943593381366824691e-1),
            static_cast<RealType>(1.53537606095123787618e-1),
            static_cast<RealType>(3.01423783265555668011e-2),
            static_cast<RealType>(2.66982581491576132363e-3),
            static_cast<RealType>(-1.57344124519315009970e-5),
            static_cast<RealType>(3.46237168332264544791e-7),
            static_cast<RealType>(2.54512306953704347532e-8),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.61596691542333069131e0),
            static_cast<RealType>(1.31560197919990191004e0),
            static_cast<RealType>(6.37865139714920275881e-1),
            static_cast<RealType>(1.99051021258743986875e-1),
            static_cast<RealType>(3.73788085017437528274e-2),
            static_cast<RealType>(3.72580876403774116752e-3),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if(x < 2){
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 2.1560e-17
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(1.63531240868022603476e-1),
            static_cast<RealType>(1.42818648212508067982e-1),
            static_cast<RealType>(4.95816076364679661943e-2),
            static_cast<RealType>(8.59234710489723831273e-3),
            static_cast<RealType>(5.76649181954629544285e-4),
            static_cast<RealType>(-5.66279925274108366994e-7),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.41478104966077351483e0),
            static_cast<RealType>(9.41180365857002724714e-1),
            static_cast<RealType>(3.65084346985789448244e-1),
            static_cast<RealType>(8.77396986274371571301e-2),
            static_cast<RealType>(1.24233749817860139205e-2),
            static_cast<RealType>(8.57476298543168142524e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 9.1732e-19
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(9.55242261334771588094e-2),
            static_cast<RealType>(6.66529732353979943139e-2),
            static_cast<RealType>(1.80958840194356287100e-2),
            static_cast<RealType>(2.34205449064047793618e-3),
            static_cast<RealType>(1.16859089123286557482e-4),
            static_cast<RealType>(-1.48761065213531458940e-7),
            static_cast<RealType>(4.37245276130361710865e-9),
            static_cast<RealType>(-8.10479404400603805292e-11),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.21670723402658089612e0),
            static_cast<RealType>(6.58224466688607822769e-1),
            static_cast<RealType>(2.00828142796698077403e-1),
            static_cast<RealType>(3.64962053761472303153e-2),
            static_cast<RealType>(3.76034152661165826061e-3),
            static_cast<RealType>(1.74723754509505656326e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 7.6621e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(3.83643820409470770350e-2),
            static_cast<RealType>(1.97555000044256883088e-2),
            static_cast<RealType>(3.71748668368617282698e-3),
            static_cast<RealType>(3.04022677703754827113e-4),
            static_cast<RealType>(8.76328889784070114569e-6),
            static_cast<RealType>(-3.34900379044743745961e-9),
            static_cast<RealType>(5.36581791174380716937e-11),
            static_cast<RealType>(-5.50656207669255770963e-13),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.09290785092251223006e-1),
            static_cast<RealType>(3.49404120360701349529e-1),
            static_cast<RealType>(7.23730835206014275634e-2),
            static_cast<RealType>(8.47875744543245845354e-3),
            static_cast<RealType>(5.28021165718081084884e-4),
            static_cast<RealType>(1.33941126695887244822e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 6.6311e-19
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.12656323880287532947e-2),
            static_cast<RealType>(2.87311140580416132088e-3),
            static_cast<RealType>(2.61788674390925516376e-4),
            static_cast<RealType>(9.74096895307400300508e-6),
            static_cast<RealType>(1.19317564431052244154e-7),
            static_cast<RealType>(-6.99543778035110375565e-12),
            static_cast<RealType>(4.33383971045699197233e-14),
            static_cast<RealType>(-1.75185581239955717728e-16),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.94430267268436822392e-1),
            static_cast<RealType>(1.00370783567964448346e-1),
            static_cast<RealType>(1.05989564733662652696e-2),
            static_cast<RealType>(6.04942184472254239897e-4),
            static_cast<RealType>(1.72741008294864428917e-5),
            static_cast<RealType>(1.85398104367945191152e-7),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 5.6459e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.83847488747490686627e-3),
            static_cast<RealType>(4.95641151588714788287e-4),
            static_cast<RealType>(2.79159792287747766415e-5),
            static_cast<RealType>(5.93951761884139733619e-7),
            static_cast<RealType>(3.89602689555407749477e-9),
            static_cast<RealType>(-4.86595415551823027835e-14),
            static_cast<RealType>(9.68524606019510324447e-17),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.01847536766892219351e-1),
            static_cast<RealType>(3.63152433272831196527e-2),
            static_cast<RealType>(2.20938897517130866817e-3),
            static_cast<RealType>(7.05424834024833384294e-5),
            static_cast<RealType>(1.09010608366510938768e-6),
            static_cast<RealType>(6.08711307451776092405e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 6.5205e-17
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(6.85767880395157523315e-4),
            static_cast<RealType>(4.08288098461672797376e-5),
            static_cast<RealType>(8.10640732723079320426e-7),
            static_cast<RealType>(6.10891161505083972565e-9),
            static_cast<RealType>(1.37951861368789813737e-11),
            static_cast<RealType>(-1.25906441382637535543e-17),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.23722380864018634550e-1),
            static_cast<RealType>(6.05800403141772433527e-3),
            static_cast<RealType>(1.47809654123655473551e-4),
            static_cast<RealType>(1.84909364620926802201e-6),
            static_cast<RealType>(1.08158235309005492372e-8),
            static_cast<RealType>(2.16335841791921214702e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(x) < 8) {
        RealType t = log2(ldexp(x, -6));

        // Rational Approximation
        // Maximum Relative Error: 3.5572e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(6.78613480244945294595e-1),
            static_cast<RealType>(9.61675759893298556080e-1),
            static_cast<RealType>(3.45159462006746978086e-1),
            static_cast<RealType>(6.32803373041761027814e-2),
            static_cast<RealType>(6.93646175256407852991e-3),
            static_cast<RealType>(4.69867700169714338273e-4),
            static_cast<RealType>(1.76219117171149694118e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.44693640094228656726e0),
            static_cast<RealType>(5.46298626321591162873e-1),
            static_cast<RealType>(1.01572892952421447864e-1),
            static_cast<RealType>(1.04982575345680980744e-2),
            static_cast<RealType>(7.65591730392359463367e-4),
            static_cast<RealType>(2.69383817793665674679e-5),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 16) {
        RealType t = log2(ldexp(x, -8));

        // Rational Approximation
        // Maximum Relative Error: 5.7408e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.51438485661317103070e-1),
            static_cast<RealType>(2.67941671074735988081e-1),
            static_cast<RealType>(5.18564629295719783781e-2),
            static_cast<RealType>(6.18976337233135940231e-3),
            static_cast<RealType>(5.08042228681335953236e-4),
            static_cast<RealType>(2.97268230746003939324e-5),
            static_cast<RealType>(1.24283200336057908183e-6),
            static_cast<RealType>(3.35670921544537716055e-8),
            static_cast<RealType>(5.06987792821954864905e-10),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.23792506680780833665e-1),
            static_cast<RealType>(8.17040643791396371682e-2),
            static_cast<RealType>(9.63961713981621216197e-3),
            static_cast<RealType>(8.06584713485725204135e-4),
            static_cast<RealType>(4.62050471704120102023e-5),
            static_cast<RealType>(1.96919734048024406173e-6),
            static_cast<RealType>(5.23890369587103685278e-8),
            static_cast<RealType>(7.99399970089366802728e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 32) {
        RealType t = log2(ldexp(x, -16));

        // Rational Approximation
        // Maximum Relative Error: 1.0195e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(6.36745544906925230102e-1),
            static_cast<RealType>(2.06319686601209029700e-1),
            static_cast<RealType>(3.27498059700133287053e-2),
            static_cast<RealType>(3.30913729536910108000e-3),
            static_cast<RealType>(2.34809665750270531592e-4),
            static_cast<RealType>(1.21234086846551635407e-5),
            static_cast<RealType>(4.55253563898240922019e-7),
            static_cast<RealType>(1.17544434819877511707e-8),
            static_cast<RealType>(1.76754192209232807941e-10),
            static_cast<RealType>(-2.78616504641875874275e-17),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.24145654925686670201e-1),
            static_cast<RealType>(5.14350019501887110402e-2),
            static_cast<RealType>(5.19867984016649969928e-3),
            static_cast<RealType>(3.68798608372265018587e-4),
            static_cast<RealType>(1.90449594112666257344e-5),
            static_cast<RealType>(7.15068261954120746192e-7),
            static_cast<RealType>(1.84646096630493837656e-8),
            static_cast<RealType>(2.77636277083994601941e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 64) {
        RealType t = log2(ldexp(x, -32));

        // Rational Approximation
        // Maximum Relative Error: 8.0433e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.36619776379492082324e-1),
            static_cast<RealType>(2.68158440168597706495e-1),
            static_cast<RealType>(5.49040993767853738389e-2),
            static_cast<RealType>(7.23458585096723552751e-3),
            static_cast<RealType>(6.85438876301780090281e-4),
            static_cast<RealType>(4.84561891424380633578e-5),
            static_cast<RealType>(2.82092117716081590941e-6),
            static_cast<RealType>(9.57557353473514565245e-8),
            static_cast<RealType>(5.16773829224576217348e-9),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.21222294324039934056e-1),
            static_cast<RealType>(8.62431574655015481812e-2),
            static_cast<RealType>(1.13640608906815986975e-2),
            static_cast<RealType>(1.07668486873466248474e-3),
            static_cast<RealType>(7.61148039258802068270e-5),
            static_cast<RealType>(4.43109262308946031382e-6),
            static_cast<RealType>(1.50412757354817481381e-7),
            static_cast<RealType>(8.11746432728995551732e-9),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else{
        result = 2 / (constants::pi<RealType>() * x * x);
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 7.4629e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62240126375351657025589608183516471315e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.94698530837122818345222883832757839888e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06728003509081587907620543204047536319e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.41256254272104786752190871391781331271e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34420233794664437979710204055323742199e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55021337841765667713712845735938627884e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.90557752737535583908921594594761570259e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.89899202021818926241643215600800085123e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.19635143827754893815649685600837995626e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.90989458941330917626663002392683325107e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92038069341802550019371049232152823407e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.40251964644989324856906264776204142653e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.55873076454666680466531097660277995317e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.80771940886011613393622410616035955976e-13),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.35771004134750535117224809381897395331e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.37002484862962406489509174332580745411e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.40833952846707180337506160933176158766e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.81709029902887471895588386777029652661e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98824705588020901032379932614151640505e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.83767868823957223030472664574235892682e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35718995485026064249286377096427165287e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.37305148463792922843850823142976586205e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.06575764439154972544253668821920460826e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07663693811543002088092708395572161856e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09711221791106684926377106608027279057e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91302186546138009232520527964387543006e-6),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 6.6684e-38
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63531240868022603475813051802104652763e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.17803013130262393286657457221415701909e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77780575692956605214628767143941600132e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44224824965135546671876867759691622832e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.93294212655117265065191070995706405837e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16021988737209938284910541133167243163e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89245591723934954825306673917695058577e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09614731993308746343064543583426077485e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48578173962833046113032690615443901556e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.91098199913613774034789276073191721350e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.46788618410999858374206722394998550706e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.14296339768511312584670061679121003569e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.52631422678659858574974085885146420544e-15),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.48481735580594347909096198787726314434e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91598585888012869317473155570063821216e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12672162924784178863164220170459406872e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06981909640884405591730537337036849744e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.89767326897694369071250285702215471082e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05098647402530640576816174680275844283e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.10454903166951593161839822697382452489e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08850649343579977859251275585834901546e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.21168773136767495960695426112972188729e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21420361560900449851206650427538430926e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.84456961344035545134425261150891935402e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46462389440125559723382692664970874255e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 6.3397e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.55242261334771588093967856464157010584e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48866040463435403672044647455806606078e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04241715667984551487882549843428953917e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.32030608366022483736940428739436921577e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17209924605508887793687609139940354371e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.16808856405217460367038406337257561698e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75466331296758720822164534334356742122e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35657250222166360635152712608912585973e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28870137478821561164537700376942753108e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.07556331078347991810236646922418944687e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.18067019247793233704208913546277631267e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.96745094401496364651919224112160111958e-12),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07735872062601280828576861757316683396e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00667909426245388114411629440735066799e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18840123665979969294228925712434860653e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79233661359264185181083948452464063323e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38221013998193410441723488211346327478e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91365002115280149925615665651486504495e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.50379182630668701710656913597366961277e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.03946139315999749917224356955071595508e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95417998434227083224840824790387887539e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05109028829536837163462811783445124876e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.33125282515685091345480270760501403655e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.58127838888839012133236453180928291822e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.64781659622256824499981528095809140284e-12),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 8.0238e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83643820409470770350079809236512802618e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.02996762669868036727057860510914079553e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88220267784864518806154823373656292346e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.12677705163934102871251710968247891123e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.96642570169484318623869835991454809217e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04358807405587072010621764865118316919e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09461879230275452416933096674703383719e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06823998699058163165831211561331795518e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24129479811279469256914665585439417704e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01799222004929573125167949870797564244e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27744716755834439008073010185921331093e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.64210356143729930758657624381557123115e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11666384975358223644665199669986358056e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.30202644697506464624965700043476935471e-22),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44479208003384373099160875893986831861e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.54290037675901616362332580709754113529e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79815821498858750185823401350096868195e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01076480676864621093034009679744852375e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88607467767854661547920709472888000469e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51572461182263866462295745828009170865e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39843444671402317250813055670653845815e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60546478324160472036295355872288494327e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.25462551353792877506974677628167909695e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05915328498722701961972258866550409117e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20632869761578411246344533841556350518e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99438347491752820345051091574883391217e-12),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 3.2541e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12656323880287532946687856443190592955e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.31374972240605659239154788518240221417e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.10776910971729651587578902049263096117e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.53872632372452909103332647334935138324e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.81756147611150151751911596225474463602e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75302607308223110644722612796766590029e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33839913867469199941739467004997833889e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.32115127487193219555283158969582307620e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.90766547421015851413713511917307214275e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08939895797457378361211153362169024503e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.88597187949354708113046662952288249250e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62829447082637808482463811005771133942e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.65525705592205245661726488519562256000e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60698835222044786453848932477732972928e-26),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.88605948104664828377228254521124685930e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.58594705700945215121673591119784576258e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.67113091918430152113322758216774649130e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39583889554372147091140765508385042797e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57139043074134496391251233307552940106e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26451960029396455805403758307828624817e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.30400557427446929311350088728080667203e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.99617890540456503276038942480115937467e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.50232186816498003232143065883536003942e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59310652872918546431499274822722004981e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82203579442241682923277858553949327687e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10345359368438386945407402887625511801e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55225829972215033873365516486524181445e-17),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 4.1276e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83847488747490686627461184914507143000e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.61220392257287638364190361688188696363e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42217711448675893329072184826328300776e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20597728166467972373586650878478687059e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.46404433551447410467051774706080733051e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27909145305324391651548849043874549520e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.33564789388635859003082815215888382619e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.18456219811686603951886248687349029515e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.92730718471866912036453008101994816885e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.51773776414973336511129801645901922234e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32371094281803507447435352076735970857e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44775294242071078601023962869394690897e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.94920633206242554892676642458535141153e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.18030442958390399095902441284074544279e-31),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.65871972115253665568580046072625013145e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.74531522538358367003224536101724206626e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20716479628426451344205712137554469781e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.83247584368619500260722365812456197226e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.52931189426842216323461406426803698335e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19343566926626449933230814579037896037e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.16243058880148231471744235009435586353e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21344088555713979086041331387697053780e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63246599173435592817113618949498524238e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43426263963680589288791782556801934305e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.62386317351298917459659548443220451300e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13281535580097407374477446521496074453e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27187882784316306216858933778750811182e-21),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 1.8458e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.85767880395157523314894776472286059373e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07684379950498990874449661385130414967e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.29181715091139597455177955800910928786e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78745116935613858188145093313446961899e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.61522707085521545633529621526418843836e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00989556810424018339768632204186394735e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94136605359672888838088037894401904574e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.15203266224687619299892471650072720579e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.25349098945982074415471295859193558426e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.31874620165906020409111024866737082384e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19330888204484008667352280840160186671e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.89951131249530265518610784629981482444e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35979606245171162602352579985003194602e-33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.50946115943875327149319867495704969908e-36),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21212467547297045538111676107434471585e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17663841151156626845609176694801024524e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25478800461954401173897968683982253458e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.69831763649657690166671862562231448718e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19712058726935472913461138967922524612e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11423395018514913507624349385447326009e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.58664605420655866109404476637021322838e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15398721299264752103644541934654351463e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17567858878427250079920401604119982576e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.92808825029184923713064129493385469531e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.06007644624654848502783947087038305433e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.01246784499782934986619755015082182398e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(x) < 8) {
        RealType t = log2(ldexp(x, -6));

        // Rational Approximation
        // Maximum Relative Error: 2.6634e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.78613480244945294594505480426643613242e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07362312709864018864207848733814857157e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.47727521897653923649758175033206259109e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04183129813120998456717217121703605830e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09978729224187570508825456585418357590e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98739784100617344335742510102186570437e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08596635852958074572320481325030046975e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34947456497875218771996878497766058580e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31766866003171430205401377671093088134e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.29444683984117745298484117924452498776e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34885173277203843795065094551227568738e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30306828175920576070486704404727265760e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.05908347665846652276910544097430115068e-13),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07218191317166728296013167220324207427e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.38908532499742180532814291654329829544e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63676664387672566455490461784630320677e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31302647779056928216789214742790688980e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.69477260342662648574925942030720482689e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.82918424748192763052497731722563414651e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69244295675395948278971027618145225216e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08928780307959133484802547123672997757e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.11055350627948183551681634293425028439e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22066081452382450191191677443527136733e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78025987104169227624653323808131280009e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.93164997733174955208299290433803918816e-13),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 16) {
        RealType t = log2(ldexp(x, -8));

        // Rational Approximation
        // Maximum Relative Error: 6.1919e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.51438485661317103069553924870169052838e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.29652867028564588922931020456447362877e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.90557738902930002845457640269863338815e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.47622170600415955276436226439948455362e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75198213226024095368607442455597948634e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73010116224706573149404022585502812698e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.33440551266376466187512220300943206212e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27556365758364667507686872656121131255e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.63395763346533783414747536236033733143e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75408632486279069728789506666930014630e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.74194099205847568739445023334735086627e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.52462367172968216583968200390021647482e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75367408334713835736514158797013854282e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.62633983586253025227038002631010874719e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46717630077826649018810277799043037738e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00642537643332236333695338824014611799e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47351714774371338348451112020520067028e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15896012319823666881998903857141624070e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.62176014448801854863922778456328119208e-25),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.79042471052521112984740498925369905803e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.55058068535501327896327971200536085268e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33143443551335870264469963604049242325e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75325348141376361676246108294525717629e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.28871858542582365161221803267369985933e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.23702867786056336210872367019916245663e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.14513776996445072162386201808986222616e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13763070277828149031445006534179375988e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75529866599039195417128499359378019030e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53029524184341515115464886126119582515e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.10598685541492162454676538516969294049e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75587930183994618721688808612207567233e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.83150895141383746641924725237948860959e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30675015193353451939138512698571954110e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.71774361582156518394662911172142577047e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.03397072601182597002547703682673198965e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.52666999314026491934445577764441483687e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 32) {
        RealType t = log2(ldexp(x, -16));

        // Rational Approximation
        // Maximum Relative Error: 1.2411e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36745544906925230101752563433306496000e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73688900814770369626527563956988302379e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.81718746296195151971617726268038570065e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.13663059680440438907042970413471861121e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.40004645275531255402942177790836798523e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.80059489775751412372432345156902685277e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47699576477278882708291693658669435536e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.45226121992756638990044029871581321461e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13406331882918393195342615955627442395e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46682598893946975917562485374893408094e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50450743907497671918301557074470352707e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.33121239192492785826422815650499088833e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05998176182038788839361491871608950696e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17857918044922309623941523489531919822e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.67865547879145051715131144371287619666e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.89654931108624296326740455618289840327e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.73017950634516660552375272495618707905e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68519137981001059472024985205381913202e-24),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.29948066505039082395951244410552705780e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.13730690908098361287472898564563217987e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27810872138103132689695155123062073221e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31948058845675193039732511839435290811e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06822729610151747708260147063757668707e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.03250522904270408071762059653475885811e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.85197262150009124871794386644476067020e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78139536405831228129042087771755615472e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.01642938314578533660138738069251610818e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36328724659833107203404258336776286146e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.80342430290059616305921915291683180697e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66502051110007556897014898713746069491e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42209715361911856322028597714105225748e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.77842582605458905635718323117222788078e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.69147628396460384758492682185049535079e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.43015110519230289924122344324563890953e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21788819753161690674882271896091269356e-24),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 64) {
        RealType t = log2(ldexp(x, -32));

        // Rational Approximation
        // Maximum Relative Error: 2.0348e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619776379492082323649724050601750141e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29818158612993476124594583743266388964e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.07736315744724186061845512973085067283e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72566458808745644851080213349673559756e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.01243670706840752914099834172565920736e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65306557791300593593488790517297048902e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41751291649776832705247036453540452119e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.26652535267657618112731521308564571490e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32742926765578976373764178875983383214e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.32948532312961882464151446137719196209e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96536595631611560703804402181953334762e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48463581600017734001916804890205661347e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.65588239861378749665334852913775575615e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39462290798829172203386678450961569536e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83049279786679854738508318703604392055e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87131679136229094080572090496960701828e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35977519905679446758726709186381481753e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.50639358104925465711435411537609380290e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.47593981758247424082096107205150226114e-40),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60997521267746350015610841742718472657e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.40470704349086277215167519790809981379e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.42305660178694704379572259575557934523e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.30272082429322808188807034927827414359e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.16742567294582284534194935923915261582e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22662407906450293978092195442686428843e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.84343501655116670387608730076359018869e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.65591734166216912475609790035240582537e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15131286290573570519912674341226377625e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08718962387679715644203327604824250850e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.04444946843492647477476784817227903589e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35966282749098189010715902284098451987e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19066854132814661112207991393498039851e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87533136296192957063599695937632598999e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93945754223094281767677343057286164777e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13592988790740273103099465658198617078e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.64942281110142621080966631872844557766e-26),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else if (ilogb(x) < 128) {
        RealType t = log2(ldexp(x, -64));

        // Rational Approximation
        // Maximum Relative Error: 4.3963e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619772367581344984274685280416528592e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72417390936686577479751162141499390532e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74319117326966091295365258834959120634e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.94269681742277805376258823511210253023e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09354876913180019634171748490068797632e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.46986612543101357465265079580805403382e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21726753043764920243710352514279216684e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29971756326232375757519588897328507962e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06770117983967828996891025614645348127e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27141668055392041978388268556174062945e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48383887723476619460217715361289178429e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.49530301203157403427315504054500005836e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18668427867427341566476567665953082312e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73377083349017331494144334612902128610e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.32380647653444581710582396517056104063e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.29865827039123699411352876626634361936e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07464506614287925844993490382319608619e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60173555862875972119871402681133785088e-23),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27912237038396638341492536677313983747e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.02138359905285600768927677649467546192e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24763589856532154099789305018886222841e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27133166772875885088000073325642460162e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01628419446817660009223289575239926907e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.62446834592284424116329218260348474201e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61238790103816844895453935630752859272e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67714109140674398508739253084218270557e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70952563202454851902810005226033501692e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33080865791583428494353408816388908148e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.06120545912923145572220606396715398781e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86403930600680015325844027465766431761e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.29419718354538719350803683985104818654e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.36261565790718847159482447247645891176e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46062982552515416754702177333530968405e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68804852250549346018535616711418533423e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51600033199082754845231795160728350588e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x * x);
    }
    else {
        result = 2 / (constants::pi<RealType>() * x * x);
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if(x >= -1){
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 9.3928e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(2.21762208692280384264e-1),
            static_cast<RealType>(7.10041055270973473923e-1),
            static_cast<RealType>(8.66556480457430718380e-1),
            static_cast<RealType>(4.78718713740071686348e-1),
            static_cast<RealType>(1.03670563650247405820e-1),
            static_cast<RealType>(4.31699263023057628473e-3),
            static_cast<RealType>(1.72029926636215817416e-3),
            static_cast<RealType>(-2.76271972015177236271e-4),
            static_cast<RealType>(1.89483904652983701680e-5),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1),
            static_cast<RealType>(2.18155995697310361937e0),
            static_cast<RealType>(2.53173077603836285217e0),
            static_cast<RealType>(1.91802065831309251416e0),
            static_cast<RealType>(9.94481663032480077373e-1),
            static_cast<RealType>(3.72037148486473195054e-1),
            static_cast<RealType>(8.85828240211801048938e-2),
            static_cast<RealType>(1.41354784778520560313e-2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 2.4742e-18
        BOOST_MATH_STATIC const RealType P[11] = {
            static_cast<RealType>(6.50763682207511020789e-3),
            static_cast<RealType>(5.73790055136022120436e-2),
            static_cast<RealType>(2.22375662069496257066e-1),
            static_cast<RealType>(4.92288611166073916396e-1),
            static_cast<RealType>(6.74552077334695078716e-1),
            static_cast<RealType>(5.75057550963763663751e-1),
            static_cast<RealType>(2.85690710485234671432e-1),
            static_cast<RealType>(6.73776735655426117231e-2),
            static_cast<RealType>(3.80321995712675339999e-3),
            static_cast<RealType>(1.09503400950148681072e-3),
            static_cast<RealType>(-9.00045301380982997382e-5),
        };
        BOOST_MATH_STATIC const RealType Q[11] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.07919389927659014373e0),
            static_cast<RealType>(2.56142472873207168042e0),
            static_cast<RealType>(1.68357271228504881003e0),
            static_cast<RealType>(2.23924151033591770613e0),
            static_cast<RealType>(9.05629695159584880257e-1),
            static_cast<RealType>(8.94372028246671579022e-1),
            static_cast<RealType>(1.98616842716090037437e-1),
            static_cast<RealType>(1.70142519339469434183e-1),
            static_cast<RealType>(1.46288923980509020713e-2),
            static_cast<RealType>(1.26171654901120724762e-2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        const static RealType lambda_bias = static_cast<RealType>(1.45158270528945486473); // (= log(pi/2)+1)

        RealType sigma = exp(-x * constants::pi<RealType>() / 2 - lambda_bias);
        RealType s = exp(-sigma) * sqrt(sigma);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 5.8685e-18
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(6.31126317567898819465e-1),
                static_cast<RealType>(5.28493759149515726917e-1),
                static_cast<RealType>(3.28301410420682938866e-1),
                static_cast<RealType>(1.31682639578153092699e-1),
                static_cast<RealType>(3.86573798047656547423e-2),
                static_cast<RealType>(7.77797337463414935830e-3),
                static_cast<RealType>(9.97883658430364658707e-4),
                static_cast<RealType>(6.05131104440018116255e-5),
            };
            BOOST_MATH_STATIC const RealType Q[8] = {
                static_cast<RealType>(1),
                static_cast<RealType>(8.47781139548258655981e-1),
                static_cast<RealType>(5.21797290075642096762e-1),
                static_cast<RealType>(2.10939174293308469446e-1),
                static_cast<RealType>(6.14856955543769263502e-2),
                static_cast<RealType>(1.24427885618560158811e-2),
                static_cast<RealType>(1.58973907730896566627e-3),
                static_cast<RealType>(9.66647686344466292608e-5),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -5.1328125) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 3.2532e-17
            BOOST_MATH_STATIC const RealType P[9] = {
                static_cast<RealType>(6.26864481454444278646e-1),
                static_cast<RealType>(5.10647753508714204745e-1),
                static_cast<RealType>(1.98551443303285119497e-1),
                static_cast<RealType>(4.71644854289800143386e-2),
                static_cast<RealType>(7.71285919105951697285e-3),
                static_cast<RealType>(8.93551020612017939395e-4),
                static_cast<RealType>(6.97020145401946303751e-5),
                static_cast<RealType>(4.17249760274638104772e-6),
                static_cast<RealType>(7.73502439313710606153e-12),
            };
            BOOST_MATH_STATIC const RealType Q[8] = {
                static_cast<RealType>(1),
                static_cast<RealType>(8.15124079722976906223e-1),
                static_cast<RealType>(3.16755852188961901369e-1),
                static_cast<RealType>(7.52819418000330690962e-2),
                static_cast<RealType>(1.23053506566779662890e-2),
                static_cast<RealType>(1.42615273721494498141e-3),
                static_cast<RealType>(1.11211928184477279204e-4),
                static_cast<RealType>(6.65899898061789485757e-6),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else {
            result = 0;
        }
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 1.2803e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21762208692280384264052188465103527015e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07121154108880017947709737976750200391e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34036993772851526455115746887751392080e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.06347688547967680654012636399459376006e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68662427153576049083876306225433068713e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67496398036468361727297056409545434117e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.69289909624425652939466055042210850769e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65649060232973461318206716040181929160e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.93006819232611588097575675157841312689e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34514211575975820725706925256381036061e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86184594939834946952489805173559003431e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66982890863184520310462776294335540260e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.28944885271022303878175622411438230193e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.47136245900831864668353768185407977846e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98034330388999615249606466662289782222e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.67931741921878993598048665757824165533e-12),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.81019852414657529520034272090632311645e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.51602582973416348091361820936922274106e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.87246706500788771729605610442552651673e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.55758863380051182011815572544985924963e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.16921634066377885762356020006515057786e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.28590978860106110644638308039189352463e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07182688002603587927920766666962846169e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.14413931232875917473403467095618397172e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59534588679183116305361784906322155131e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.62788361787003488572546802835677555151e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32291670834750583053201239125839728061e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97300476673137879475887158731166178829e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99801949382703479169010768105376163814e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09234481837537672361990844588166022791e-5),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 3.8590e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.50763682207511020788551990942118742910e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.35160148798611192350830963080055471564e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.85567614778755464918744664468938413626e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24395902843792338723377508551415399267e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75803588325237557939443967923337822799e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44751743702858358960016891543930028989e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.38771920793989989423514808134997891434e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.99899457801652012757624005300136548027e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.59668432891116320233415536189782241116e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.02521376213276025040458141317737977692e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.00511857068867825025582508627038721402e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19031970665203475373248353773765801546e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.03203906044415590651592066934331209362e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01354553335348149914596284286907046333e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40077709279222086527834844446288408059e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.07036291955272673946830858788691198641e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75229595324028909877518859428663744660e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.51522041748753421579496885726802106514e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.28554063325397021905295499768922434904e-10),
        };
        BOOST_MATH_STATIC const RealType Q[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.55889733194498836168215560931863059152e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.45050534010127542130960211621894286688e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39437268390909980446225806216001154876e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85370557677145869100298813360909127310e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99358236671478050470186012149124879556e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82914467302553175692644992910876515874e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42426383410763382224410804289834740252e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.69477085497572590673874940261777949808e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.69832833104494997844651343499526754631e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95708391432781281454592429473451742972e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32541987059874996779040445020449508142e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.24889827757289516008834701298899804535e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76326709965329347689033555841964826234e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.19652942193884551681987290472603208296e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22987197033955835618810845653379470109e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51893290463268547258382709202599507274e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.43575882043846146581825453522967678538e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06683418138599962787868832158681391673e-5),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        const static RealType lambda_bias = BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.451582705289454864726195229894882143572); // (= log(pi/2)+1)

        RealType sigma = exp(-x * constants::pi<RealType>() / 2 - lambda_bias);
        RealType s = exp(-sigma) * sqrt(sigma);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 7.0019e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[18] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.31126317567898819464557840628449107915e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.31008645911415314700225107327351636697e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.60743397071713227215207831174512626190e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69243936604887410595461520921270733657e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93778117053417749769040328795824088196e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.04718815412035890861219665332918840537e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41914050146414549019258775115663029791e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17147074474397510167661838243237386450e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31006358624990533313832878493963971249e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.31424805670861981190416637260176493218e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71604447221961082506919140038819715820e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.01796816886825676412069047911936154422e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.16975381608692872525287947181531051179e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47194712963929503930146780326366215579e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19469248860267489980690249379132289464e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22272545853285700254948346226514762534e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.05432616288832680241611577865488417904e-13),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08723511461992818779941378551362882730e-14),
            };
            BOOST_MATH_STATIC const RealType Q[16] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01021278581037282130358759075689669228e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.91783545335316986601746168681457332835e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.90025337163174587593060864843160047245e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.09029833197792884728968597136867674585e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44295160726145715084515736090313329125e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.46416375246465800703437031839310870287e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86521610039165178072099210670199368231e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.29830357713744587265637686549132688965e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32187562202835921333177458294507064946e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75034541113922116856456794810138543224e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79216314818261657918748858010817570215e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.69179323869133503169292092727333289999e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.09019623876540244217038375274802731869e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13900582194674129200395213522524183495e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92590979457175565666605415984496551246e-9),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -6.875) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 6.4095e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[18] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.26864481454444278645937156746132802908e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35052316263030534355724898036735352905e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46701697626917441774916114124028252971e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03805679118924248671851611170709699862e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.29457230118834515743802694404620370943e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04992992250026414994541561073467805333e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.21521951889983113700615967351903983850e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50611640491200231504944279876023072268e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.96007721851412367657495076592244098807e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.76876967456744990483799856564174838073e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.34285198828980523126745002596084187049e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.98811180672843179022928339476420108494e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36933707823930146448761204037985193905e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.76515121042989743198432939393805252169e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87259915481622487665138935922067520210e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34703958446785695676542385299325713141e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53199672688507288037695102377982544434e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.97283413733676690377949556457649405210e-14),
            };
            BOOST_MATH_STATIC const RealType Q[18] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15492787140203223641846510939273526038e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34095796298757853634036909432345998054e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65650140652391522296109869665871008634e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.44894089102275258806976831589022821974e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27121975866547045393504246592187721233e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.91803733484503004520983723890062644122e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40341451263971324381655967408519161854e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72360046810103129487529493828280649599e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.60986435254173073868329335245110986549e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01216966786091058959421242465309838187e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11514619470960373138100691463949937779e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01639426441970732201346798259534312372e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.60411422906070056043690129326288757143e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.58398956202137709744885774931524547894e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14956902064425256856583295469934064903e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.23201118234279642321630988607491208515e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43185798646451225275728735761433082676e-13),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else {
            result = 0;
        }
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53> &tag) {
    if (x >= 0) {
        return landau_pdf_plus_imp_prec<RealType>(x, tag);
    }
    else if (x <= 0) {
        return landau_pdf_minus_imp_prec<RealType>(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>& tag) {
    if (x >= 0) {
        return landau_pdf_plus_imp_prec<RealType>(x, tag);
    }
    else if (x <= 0) {
        return landau_pdf_minus_imp_prec<RealType>(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType landau_pdf_imp(const landau_distribution<RealType, Policy>& dist, const RealType& x) {
    //
    // This calculates the pdf of the Landau distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::pdf(landau<%1%>&, %1%)";
    RealType result = 0;
    RealType location = dist.location();
    RealType scale = dist.scale();
    RealType bias = dist.bias();

    if (false == detail::check_location(function, location, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_scale(function, scale, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_x(function, x, &result, Policy()))
    {
        return result;
    }

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Landau distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale + bias;

    result = landau_pdf_imp_prec(u, tag_type()) / scale;

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 2.7348e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(6.34761298487625202628e-1),
            static_cast<RealType>(7.86558857265845597915e-1),
            static_cast<RealType>(4.30220871807399303399e-1),
            static_cast<RealType>(1.26410946316538340541e-1),
            static_cast<RealType>(2.09346669713191648490e-2),
            static_cast<RealType>(1.48926177023501002834e-3),
            static_cast<RealType>(-5.93750588554108593271e-7),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.65227304522196452589e0),
            static_cast<RealType>(1.29276828719607419526e0),
            static_cast<RealType>(5.93815051307098615300e-1),
            static_cast<RealType>(1.69165968013666952456e-1),
            static_cast<RealType>(2.84272940328510367574e-2),
            static_cast<RealType>(2.28001970477820696422e-3),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 6.1487e-17
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(4.22133240358047652363e-1),
            static_cast<RealType>(3.48421126689016131480e-1),
            static_cast<RealType>(1.15402429637790321091e-1),
            static_cast<RealType>(1.90374044978864005061e-2),
            static_cast<RealType>(1.26628667888851698698e-3),
            static_cast<RealType>(-5.75103242931559285281e-7),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.21277435324167238159e0),
            static_cast<RealType>(6.38324046905267845243e-1),
            static_cast<RealType>(1.81723381692749892660e-1),
            static_cast<RealType>(2.80457012073363245106e-2),
            static_cast<RealType>(1.93749385908189487538e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 3.2975e-17
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(2.95892137955791216378e-1),
            static_cast<RealType>(2.29083899043580095868e-1),
            static_cast<RealType>(7.09374171394372356009e-2),
            static_cast<RealType>(1.08774274442674552229e-2),
            static_cast<RealType>(7.69674715320139398655e-4),
            static_cast<RealType>(1.63486840000680408991e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.09704883482087441931e0),
            static_cast<RealType>(5.10139057077147935327e-1),
            static_cast<RealType>(1.27055234007499238241e-1),
            static_cast<RealType>(1.74542139987310825683e-2),
            static_cast<RealType>(1.18944143641885993718e-3),
            static_cast<RealType>(2.55296292914537992309e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 2.6740e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.73159318667565938776e-1),
            static_cast<RealType>(6.95847424776057206679e-2),
            static_cast<RealType>(1.04513924567165899506e-2),
            static_cast<RealType>(6.35094718543965631442e-4),
            static_cast<RealType>(1.04166111154771164657e-5),
            static_cast<RealType>(1.43633490646363733467e-9),
            static_cast<RealType>(-4.55493341295654514558e-11),
            static_cast<RealType>(6.71119091495929467041e-13),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(6.23409270429130114247e-1),
            static_cast<RealType>(1.54791925441839372663e-1),
            static_cast<RealType>(1.85626981728559445893e-2),
            static_cast<RealType>(1.01414235673220405086e-3),
            static_cast<RealType>(1.63385654535791481980e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 7.6772e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(8.90469147411748292410e-2),
            static_cast<RealType>(2.76033447621178662228e-2),
            static_cast<RealType>(3.26577485081539607943e-3),
            static_cast<RealType>(1.77755752909150255339e-4),
            static_cast<RealType>(4.20716551767396206445e-6),
            static_cast<RealType>(3.19415703637929092564e-8),
            static_cast<RealType>(-1.79900915228302845362e-13),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.36499987260915480890e-1),
            static_cast<RealType>(7.67544181756713372678e-2),
            static_cast<RealType>(6.83535263652329633233e-3),
            static_cast<RealType>(3.15983778969051850073e-4),
            static_cast<RealType>(6.84144567273078698399e-6),
            static_cast<RealType>(5.00300197147417963939e-8),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 1.5678e-20
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(4.35157264931262089762e-2),
            static_cast<RealType>(8.46833474333913742597e-3),
            static_cast<RealType>(6.43769318301002170686e-4),
            static_cast<RealType>(2.39440197089740502223e-5),
            static_cast<RealType>(4.45572968892675484685e-7),
            static_cast<RealType>(3.76071815793351687179e-9),
            static_cast<RealType>(1.04851094362145160445e-11),
            static_cast<RealType>(-8.50646541795105885254e-18),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1),
            static_cast<RealType>(2.59832721225510968607e-1),
            static_cast<RealType>(2.75929030381330309762e-2),
            static_cast<RealType>(1.53115657043391090526e-3),
            static_cast<RealType>(4.70173086825204710446e-5),
            static_cast<RealType>(7.76185172490852556883e-7),
            static_cast<RealType>(6.10512879655564540102e-9),
            static_cast<RealType>(1.64522607881748812093e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 2.2534e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.11253031965493064317e-2),
            static_cast<RealType>(1.36656844320536022509e-3),
            static_cast<RealType>(2.99036224749763963099e-5),
            static_cast<RealType>(2.54538665523638998222e-7),
            static_cast<RealType>(6.79286608893558228264e-10),
            static_cast<RealType>(-6.92803349600061706079e-16),
            static_cast<RealType>(5.47233092767314029032e-19),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(9.71506209641408410168e-2),
            static_cast<RealType>(3.52744690483830496158e-3),
            static_cast<RealType>(5.85142319429623560735e-5),
            static_cast<RealType>(4.29686638196055795330e-7),
            static_cast<RealType>(1.06586221304077993137e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(x) < 8) {
        RealType t = log2(ldexp(x, -6));

        // Rational Approximation
        // Maximum Relative Error: 3.8057e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(6.60754766433212615409e-1),
            static_cast<RealType>(2.47190065739055522599e-1),
            static_cast<RealType>(4.17560046901040308267e-2),
            static_cast<RealType>(3.71520821873148657971e-3),
            static_cast<RealType>(2.03659383008528656781e-4),
            static_cast<RealType>(2.52070598577347523483e-6),
            static_cast<RealType>(-1.63741595848354479992e-8),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(3.92836792184266080580e-1),
            static_cast<RealType>(6.64332913820571574875e-2),
            static_cast<RealType>(5.59456053716889879620e-3),
            static_cast<RealType>(3.44201583106671507027e-4),
            static_cast<RealType>(2.74554105716911980435e-6),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 16) {
        RealType t = log2(ldexp(x, -8));

        // Rational Approximation
        // Maximum Relative Error: 1.5585e-18
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.44802371584831601817e-1),
            static_cast<RealType>(2.74177359656349204309e-1),
            static_cast<RealType>(5.53659240731871433983e-2),
            static_cast<RealType>(6.97653365560511851744e-3),
            static_cast<RealType>(6.17058143529799037402e-4),
            static_cast<RealType>(3.94979574476108021136e-5),
            static_cast<RealType>(1.88315864113369221822e-6),
            static_cast<RealType>(6.10941845734962836501e-8),
            static_cast<RealType>(1.39403332890347813312e-9),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.32345127287830884682e-1),
            static_cast<RealType>(8.70500634789942065799e-2),
            static_cast<RealType>(1.09253956356393590470e-2),
            static_cast<RealType>(9.72576825490118007977e-4),
            static_cast<RealType>(6.18656322285414147985e-5),
            static_cast<RealType>(2.96375876501823390564e-6),
            static_cast<RealType>(9.58622809886777038970e-8),
            static_cast<RealType>(2.19059124630695181004e-9),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 32) {
        RealType t = log2(ldexp(x, -16));

        // Rational Approximation
        // Maximum Relative Error: 8.4773e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.36685748306554972132e-1),
            static_cast<RealType>(2.22217783148381285219e-1),
            static_cast<RealType>(3.79173960692559280353e-2),
            static_cast<RealType>(4.13394722917837684942e-3),
            static_cast<RealType>(3.18141233442663766089e-4),
            static_cast<RealType>(1.79745613243740552736e-5),
            static_cast<RealType>(7.47632665728046334131e-7),
            static_cast<RealType>(2.18258684729250152138e-8),
            static_cast<RealType>(3.93038365129320422968e-10),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(3.49087806008685701060e-1),
            static_cast<RealType>(5.95568283529034601477e-2),
            static_cast<RealType>(6.49386742119035055908e-3),
            static_cast<RealType>(4.99721374204563274865e-4),
            static_cast<RealType>(2.82348248031305043777e-5),
            static_cast<RealType>(1.17436903872210815656e-6),
            static_cast<RealType>(3.42841159307801319359e-8),
            static_cast<RealType>(6.17382517100568714012e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 64) {
        RealType t = log2(ldexp(x, -32));

        // Rational Approximation
        // Maximum Relative Error: 4.1441e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.36619774420718062663e-1),
            static_cast<RealType>(2.68594096777677177874e-1),
            static_cast<RealType>(5.50713044649497737064e-2),
            static_cast<RealType>(7.26574134143434960446e-3),
            static_cast<RealType>(6.89173530168387629057e-4),
            static_cast<RealType>(4.87688310559244353811e-5),
            static_cast<RealType>(2.84218580121660744969e-6),
            static_cast<RealType>(9.65240367429172366675e-8),
            static_cast<RealType>(5.21722720068664704240e-9),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.21906621389193043384e-1),
            static_cast<RealType>(8.65058026826346828750e-2),
            static_cast<RealType>(1.14129998157398060009e-2),
            static_cast<RealType>(1.08255124950652385121e-3),
            static_cast<RealType>(7.66059006900869004871e-5),
            static_cast<RealType>(4.46449501653114622960e-6),
            static_cast<RealType>(1.51619602364037777665e-7),
            static_cast<RealType>(8.19520132288940649002e-9),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else {
        result = 2 / (constants::pi<RealType>() * x);
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 2.6472e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.34761298487625202628055609797763667089e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67589195401255255724121983550745957195e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07502511824371206858547365520593277966e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58354381655514028012912292026393699991e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.26470588572701739953294573496059174764e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.09494168186680012705692462031819276746e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.47385718073281027400744626077865581325e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69107567947502492044754464589464306928e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39641345689672620514703813504927833352e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27003930699448633502508661352994055898e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26124673422692247711088651516214728305e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.92103390710025598612731036700549416611e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.49572523814120679048097861755172556652e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.50719933268462244255954307285373705456e-13),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05332427324361912631483249892199461926e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46280417679002004953145547112352398783e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.10429833573651169023447466152999802738e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.63535585818618617796313647799029559407e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24103322502244219003850826414302390557e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.38359438431541204276767900393091886363e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16687583686405832820912406970664239423e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31451667102532056871497958974899742424e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31646175307279119467894327494418625431e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.22334681489114534492425036698050444462e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86326948577818727263376488455223120476e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53867591038308710930446815360572461884e-7),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 1.2387e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.22133240358047652363270514524313049653e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.35860518549481281929441026718420080571e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.89900189271177970319691370395978805326e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84682995288088652145572170736339265315e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.10748045562955323875797887939420022326e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00246325517647746481631710824413702051e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02998394686245118431020407235000441722e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.06095284318730009040434594746639110387e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91754425158654496372516241124447726889e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.37288564874584819097890713305968351561e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.77487285800889132325390488044487626942e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.41654614425073025870130302460301244273e-13),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13058739144695658589427075788960660400e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11792528400843967390452475642793635419e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28794756779085737559146475126886069030e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.29339015472607099189295465796550367819e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53434372685847620864540166752049026834e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.17610372643685730837081191600424913542e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.64455304425865128680681864919048610730e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.62357689170951502920019033576939977973e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.89912258835489782923345357128779660633e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.76323449710934127736624596886862488066e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21524231900555452527639738371019517044e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 1.2281e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95892137955791216377776422765473500279e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.65957634570689820998348206103212047458e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.34657686985192350529330481818991619730e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43985500841002490334046057189458709493e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.09876223028004323158413173719329449720e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.04194660038290410425299531094974709019e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09780604136364125990393172827373829860e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02676079027875648517286351062161581740e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.30298199082321832830328345832636435982e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33633965123855006982811143987691483957e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46384114966020719170903077536685621119e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.07058773850795175564735754911699285828e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.76765053309825506619419451346428518606e-16),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89758965744489334954041814073547951925e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65985298582650601001220682594742473012e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.81017086203232617734714711306180675445e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14481301672800918591822984940714490526e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.90605450026850685321372623938646722657e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42447999818015246265718131846902731574e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83426770079526980292392341278413549820e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65073357441521690641768959521412898756e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.90437453546925074707222505750595530773e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.63458145595422196447107547750737429872e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.94457070577990681786301801930765271001e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.80986568964737305842778359322566801845e-11),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 5.3269e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73159318667565938775602634998889798568e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95149372103869634275319490207451722385e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.87504411659823400690797222216564651939e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.94571385159717824767058200278511014560e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.71656210265434934399632978675652106638e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.51248899957476233641240573020681464290e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.74490600490886011190565727721143414249e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.07537323853509621126318424069471060527e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59167561354023258538869598891502822922e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.72608361427131857269675430568328018022e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.54143016370650707528704927655983490119e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07936446902207128577031566135957311260e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.73506415766100115673754920344659223382e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66918494546396383814682000746818494148e-21),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34854858486201481385140426291984169791e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.25379978655428608198799717171321453517e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.01905621587554903438286661709763596137e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.31293339647901753103699339801273898688e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22793491714510746538048140924864505813e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45360205736839126407568005196865547577e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20918479556021574336548106785887700883e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.91450617548036413606169102407934734864e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59940586452863361281618661053014404930e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.93918243796178165623395356401173295690e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.14578198844767847381800490360878776998e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.25951924258762195043744665124187621023e-13),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 4.8719e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.90469147411748292410422813492550092930e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.20196598836093298098360769875443462143e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92579652651763461802771336515384878994e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.50439147419887323351995227585244144060e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13214742069751393867851080954754449610e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29648648382394801501422003194522139519e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.80420399625810952886117129805960917210e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.73059844436212109742132138573157222143e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66835461298243901306176013397428732836e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.61808423521250921041207160217989047728e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39300098366988229510997966682317724011e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09447823064238788960158765421669935819e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76962456941948786610101052244821659252e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.56004343709960620209823076030906442732e-25),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.22996422556023111037354479836605618488e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05244279198013248402385148537421114680e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72477335169177427114629223821992187549e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.76404568980852320252614006021707040788e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.79959793426748071158513573279263946303e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.93524788220877416643145672816678561612e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.46692111397574773931528693806744007042e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20764040991846422990601664181377937629e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84452460717254884659858711994943474216e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30317379590981344496250492107505244036e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83736710938966780518785861828424593249e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72943283576264035508862984899450025895e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77791087299927741360821362607419036797e-18),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 5.3269e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35157264931262089761621934621402648954e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51407493866635569361305338029611888082e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30886132894858313459359493329266696766e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.02488735053241778868198537544867092626e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12676055870976203566712705442945186614e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.14136757304740001515364737551021389293e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01514327671186735593984375829685709678e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63783530594707477852365258482782354261e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67623013776194044717097141295482922572e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01397549144502050693284434189497148608e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.16720246008161901837639496002941412533e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70776057051329137176494230292143483874e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.02838174509144355795908173352005717435e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.07343581702278433243268463675468320030e-30),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13166129191902183515154099741529804400e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.58587877615076239769720197025023333190e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.14619910799508944167306046977187889556e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.66671218029939293563302720748492945618e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67192565058098643223751044962155343554e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.68397391192695060767615969382391508636e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94009678375859797198831431154760916459e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91901267471125881702216121486397689200e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83697721782125852878533856266722593909e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65409094893730117412328297801448869154e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.04202174160401885595563150562438901685e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.82104711207466136473754349696286794448e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 1.0937e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11253031965493064317003259449214452745e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66886306590939856622089350675801752704e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77786526684921036345823450504680078696e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20343424607276252128027697088363135591e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29196073776799916444272401212341853981e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.92132293422644089278551376756604946339e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07465707745270914645735055945940815947e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.54424785613626844024154493717770471131e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74829439628215654062512023453584521531e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.03470347880592072854295353687395319489e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21113051919776165865529140783521696702e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.83719812218384126931626509884648891889e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.41009036423458926116066353864843586169e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04965807080681693416200699806159303323e-34),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06133417626680943824361625182288165823e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87991711814130682492211639336942588926e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.98342969282034680444232201546039059255e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41940454945139684365514171982891170420e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.90481418770909949109210069475433304086e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25451856391453896652473393039014954572e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36349120987010174609224867075354225138e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94716367816033715208164909918572061643e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.67466470065187852967064897686894407151e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31509787633232139845762764472649607555e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93147765040455324545205202900563337981e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07370974123835247210519262324524537634e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(x) < 8) {
        RealType t = log2(ldexp(x, -6));

        // Rational Approximation
        // Maximum Relative Error: 3.1671e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.60754766433212615408805486898847664740e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.76143516602438873568296501921670869526e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.25254763859315398784817302471631188095e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.58650277225655302085863010927524053686e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92227773746592457803942136197158658110e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77170481512334811333255898903061802339e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.97864282716826576471164657368231427231e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.44243747123065035356982629201975914275e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54592817957461998135980337838429682406e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.52110831321633404722419425039513444319e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75030698219998735693228347424295850790e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.08894662488905377548940479566994482806e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11472306961184868827300852021969296872e-12),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04202386226609593823214781180612848612e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.07648212684952405730772649955008739292e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50646784687432427178774105515508540021e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02521400964223268224629095722841793118e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35374201758795213489427690294679848997e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.67290684433876221744005507243460683585e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83834977086311601362115427826807705185e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43236252790815493406777552261402865674e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17693398496807851224497995174884274919e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34161978291568722756523120609497435933e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10819003833429876218381886615930538464e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75674945131892236663189757353419870796e-12),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 16) {
        RealType t = log2(ldexp(x, -8));

        // Rational Approximation
        // Maximum Relative Error: 6.8517e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.44802371584831601817146389426921705500e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.32962058761590152378007743852342151897e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93461601407042255925193793376118641680e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.55533612685775705468614711945893908392e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76358919439168503100357154639460097607e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74131615534562303144125602950691629908e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.34470363614899824502654995633001232079e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29020592733459891982428815398092077306e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65794017754267756566941255128608603072e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78550878208007836345763926019855723350e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00524022519953193863682806155339574713e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.75977976583947697667784048133959750133e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89287460618943500291479647438555099783e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.27647727947590174240069836749437647626e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70582660582766959108625375415057711766e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.68499175244574169768386088971844067765e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84493774639724473576782806157757824413e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.03526708437207438952843827018631758857e-20),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.88770963972332750838571146142568699263e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.60273143287476497658795203149608758815e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34393017137025732376113353720493995469e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77086140458900002000076127880391602253e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30613589576665986239534705717153313682e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.25355055770024448240128702278455001334e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.16820392686312531160900884133254461634e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.17518073757769640772428097588524967431e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80469112215756035261419102003591533407e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57898123952200478396366475124854317231e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.47675840885248141425130440389244781221e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.97339213584115778189141444065113447170e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.85845602624148484344802432304264064957e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67971825826765902713812866354682255811e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.78795543918651402032912195982033010270e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.18166403020940538730241286150437447698e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.47995312365747437038996228794650773820e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 32) {
        RealType t = log2(ldexp(x, -16));

        // Rational Approximation
        // Maximum Relative Error: 6.5315e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36685748306554972131586673701426039950e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75892450098649456865500477195142009984e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.77167300709199375935767980419262418694e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.76657987434662206916119089733639111866e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.55003354250569146980730594644539195376e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.64102805555049236216024194001407792885e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36247488122195469059567496833809879653e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63710218182036673197103906176200862606e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.60629446465979003842091012679929186607e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22712292003775206105713577811447961965e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.28539646473359376707298867613704501434e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47893806904387088760579412952474847897e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.86809622035928392542821045232270554753e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47935154807866802001012566914901169147e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.40623855123515207599160827187101517978e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.60738876249485914019826585865464103800e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29257874466803586327275841282905821499e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.06781616867813418930916928811492801723e-31),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.33391038950576592915531240096703257292e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.06598147816889621749840662500099582486e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21997213397347849640608088189055469954e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18596259982438670449688554459343971428e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.86085649929528605647139297483281849158e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28178198640304770056854166079598253406e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57155234493390297220982397633114062827e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03771216318243964850930846579433365529e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.49837117625052865973189772546210716556e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.87302306206908338457432167186661027909e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32312467403555290915110093627622951484e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.50516769529388616534895145258103120804e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.03618834064000582669276279973634033592e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.49205787058220657972891114812453768100e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.23730041323226753771240724078738146658e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60115971929371066362271909482282503973e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 64) {
        RealType t = log2(ldexp(x, -32));

        // Rational Approximation
        // Maximum Relative Error: 1.0538e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619774420718062663274858007687066488e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30268560944740805268408378762250557522e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.09208091036436297425427953080968023835e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.74943166696408995577495065480328455423e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.03759498310898586326086395411203400316e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67261192787197720215143001944093963953e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42934578939412238889174695091726883834e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.32436794923711934610724023467723195718e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35077626369701051583611707128788137675e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.40836846523442062397035620402082560833e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98783806012035285862106614557391807137e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53869567415427145730376778932236900838e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.76521311340629419738016523643187305675e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41298928566351198899106243930173421965e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85552706372195482059144049293491755419e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89780987301820615664133438159710338126e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37965575161090804572561349091024723962e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.58944184323201470938493323680744408698e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35050162268451658064792430214910233545e-40),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61705010674524952791495931314010679992e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.42782564900556152436041716057503104160e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.46038982970912591009739894441944631471e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.34223936001873800295785537132905986678e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.19812900355867749521882613003222797586e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24521111398180921205229795007228494287e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.93429394949368809594897465724934596442e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69259071867341718986156650672535675726e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16370379759046264903196063336023488714e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.12248872253003553623419554868303473929e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12936649424676303532477421399492615666e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37683645610869385656713212194971883914e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21951837982136344238516771475869548147e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91465509588513270823718962232280739302e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98107277753797219142748868489983891831e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.16715818685246698314459625236675887448e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.77987471623502330881961633434056523159e-26),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else if (ilogb(x) < 128) {
        RealType t = log2(ldexp(x, -64));

        // Rational Approximation
        // Maximum Relative Error: 2.2309e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619772367581344040890134127619524371e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72522424358877592972375801826826390634e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74749058021341871895402838175268752603e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.95136532385982168410320513292144834602e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.10500792867575154180588502397506694341e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.48101840822895419487033057691746982216e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.22577211783918426674527460572438843266e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30499751609793470641331626931224574780e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07043764292750472900578756659402327450e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.28345765853059515246787820662932506931e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48838700086419232178247558529254516870e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.51015062047870581993810118835353083110e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19087584836023628483830612541904830502e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74405125338538967114280887107628943111e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.34493122865874905104884954420903910585e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.33794243240353561095702650271950891264e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07931799710240978706633227327649731325e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60935135769719557933955672887720342220e-23),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.28077223152164982690351137450174450926e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.02813709168750724641877726632095676090e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24899754437233214579860634420198464016e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27313166830073839667108783881090842820e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01803599095361490387188839658640162684e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.63782732057408167492988043000134055952e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.62068163155799642595981598061154626504e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68143951757351157612001096085234448512e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72843955600132743549395255544732133507e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33795283380674591910584657171640729449e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.08452802793967494363851669977089389376e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87062340827301546650149031133192913586e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.31034562935470222182311379138749593572e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.39579834094849668082821388907704276985e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46680056726416759571957577951115309094e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69538874529209016624246362786229032706e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52796320119313458991885552944744518437e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * x);
    }
    else {
        result = 2 / (constants::pi<RealType>() * x);
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 4.8279e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(9.61609610406317335842e-2),
            static_cast<RealType>(3.91836314722738553695e-1),
            static_cast<RealType>(6.79862925205625107133e-1),
            static_cast<RealType>(6.52516594941817706368e-1),
            static_cast<RealType>(3.78594163612581127974e-1),
            static_cast<RealType>(1.37741592243008345389e-1),
            static_cast<RealType>(3.16100502353317199197e-2),
            static_cast<RealType>(3.94935603975622336575e-3),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.76863983252615276767e0),
            static_cast<RealType>(1.81486018095087241378e0),
            static_cast<RealType>(1.17295504548962999723e0),
            static_cast<RealType>(5.33998066342362562313e-1),
            static_cast<RealType>(1.66508320794082632235e-1),
            static_cast<RealType>(3.42192028846565504290e-2),
            static_cast<RealType>(3.94691613177524994796e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 2.3675e-17
        BOOST_MATH_STATIC const RealType P[11] = {
            static_cast<RealType>(7.07114056489178077423e-4),
            static_cast<RealType>(7.35277969197058909845e-3),
            static_cast<RealType>(3.45402694579204809691e-2),
            static_cast<RealType>(9.62849773112695332289e-2),
            static_cast<RealType>(1.75738736725818007992e-1),
            static_cast<RealType>(2.18309266582058485951e-1),
            static_cast<RealType>(1.85680388782727289455e-1),
            static_cast<RealType>(1.06177394398691169291e-1),
            static_cast<RealType>(3.94880388335722224211e-2),
            static_cast<RealType>(9.46543177731050647162e-3),
            static_cast<RealType>(1.50949646857411896396e-3),
        };
        BOOST_MATH_STATIC const RealType Q[11] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.19520021153535414164e0),
            static_cast<RealType>(2.24057032777744601624e0),
            static_cast<RealType>(1.63635577968560162720e0),
            static_cast<RealType>(1.58952087228427876880e0),
            static_cast<RealType>(7.63062254749311648018e-1),
            static_cast<RealType>(4.65805990343825931327e-1),
            static_cast<RealType>(1.45821531714775598887e-1),
            static_cast<RealType>(5.42393925507104531351e-2),
            static_cast<RealType>(9.84276292481407168381e-3),
            static_cast<RealType>(1.54787649925009672534e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        const static RealType lambda_bias = static_cast<RealType>(1.45158270528945486473); // (= log(pi/2)+1)

        RealType sigma = exp(-x * constants::pi<RealType>() / 2 - lambda_bias);
        RealType s = exp(-sigma) / sqrt(sigma);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 6.6532e-17
            BOOST_MATH_STATIC const RealType P[9] = {
                static_cast<RealType>(3.71658823632747235572e-1),
                static_cast<RealType>(2.81493346318174084721e-1),
                static_cast<RealType>(1.80052521696460721846e-1),
                static_cast<RealType>(7.65907659636944822120e-2),
                static_cast<RealType>(2.33352148213280934280e-2),
                static_cast<RealType>(5.02308701022480574067e-3),
                static_cast<RealType>(6.29239919421134075502e-4),
                static_cast<RealType>(8.36993181707604609065e-6),
                static_cast<RealType>(-8.38295154747385945293e-6),
            };
            BOOST_MATH_STATIC const RealType Q[9] = {
                static_cast<RealType>(1),
                static_cast<RealType>(6.62107509936390708604e-1),
                static_cast<RealType>(4.72501892305147483696e-1),
                static_cast<RealType>(1.84446743813050604353e-1),
                static_cast<RealType>(5.99971792581573339487e-2),
                static_cast<RealType>(1.24751029844082800143e-2),
                static_cast<RealType>(1.56705297654475773870e-3),
                static_cast<RealType>(2.36392472352050487445e-5),
                static_cast<RealType>(-2.11667044716450080820e-5),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -5.1328125) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 2.6331e-17
            BOOST_MATH_STATIC const RealType P[10] = {
                static_cast<RealType>(3.97500903816385095134e-1),
                static_cast<RealType>(5.08559630146730380854e-1),
                static_cast<RealType>(2.99190443368166803486e-1),
                static_cast<RealType>(1.07339363365158174786e-1),
                static_cast<RealType>(2.61694301269384158162e-2),
                static_cast<RealType>(4.58386867966451237870e-3),
                static_cast<RealType>(5.80610284231484509069e-4),
                static_cast<RealType>(5.07249042503156949021e-5),
                static_cast<RealType>(2.91644292826084281875e-6),
                static_cast<RealType>(9.75453868235609527534e-12),
            };
            BOOST_MATH_STATIC const RealType Q[9] = {
                static_cast<RealType>(1),
                static_cast<RealType>(1.27376091725485414303e0),
                static_cast<RealType>(7.49829208702328578188e-1),
                static_cast<RealType>(2.69157374996960976399e-1),
                static_cast<RealType>(6.55795320040378662663e-2),
                static_cast<RealType>(1.14912646428788757804e-2),
                static_cast<RealType>(1.45541420582309879973e-3),
                static_cast<RealType>(1.27135040794481871472e-4),
                static_cast<RealType>(7.31138551538712031061e-6),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else {
            result = 0;
        }
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 1.2055e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.61609610406317335842332400044553397267e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74152295981095898203847178356629061821e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58642905042588731020840168744866124345e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69370085525311304330141932309908104187e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14888713497930800611167630826754270499e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69123861559106636252620023643265102867e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74273532954853421626852458737661546439e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.73534665976007761924923962996725209700e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42543389723715037640714282663089570985e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05120903211852044362181935724880384488e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49586169587615171270941258051088627885e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46047939521303565932576405363107506886e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.68248726161641913236972878212857788320e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.60663638253775180681171554635861859625e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76463460016745893121574217030494989443e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.08380585744336744543979680558024295296e-12),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66458574743150749245922924142120646408e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.87010262350733534202724862784081296105e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.51107149980251214963849267707173045433e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.71158207369578457239679595370389431171e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.37188705505573668092513124472448362633e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95647530096628718695081507038921183627e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30278895428001081342301218278371140110e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.61322060563420594659487640090297303892e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30529729106312748824241317854740876915e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.90465740298431311519387111139787971960e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.92760416706194729215037805873466599319e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02070496615845146626690561655353212151e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72080705566714681586449384371609107346e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76433504120625478720883079263866245392e-6),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 3.4133e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07114056489178077422539043012078031613e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.18006784954579394004360967455655021959e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.60309646092161147676756546417366564213e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13479499932401667065782086621368143322e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.68587439643060549883916236839613331692e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12366494749830793876926914920462629077e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70739124754664545339208363069646589169e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04073482998938337661285862393345731336e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94833709787596305918524943438549684109e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50214412821697972546222929550410139790e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.43105005523280337071698704765973602884e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.85396789833278250392015217207198739243e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05690359993570736607428746439280858381e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.17297815188944531843360083791153470475e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.03913601629627587800587620822216769010e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.61963034255210565218722882961703473760e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.99502258440875586452963094474829571000e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.66563884565518965562535171848480872267e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.23954896921292896539048530795544784261e-6),
        };
        BOOST_MATH_STATIC const RealType Q[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77934931846682015134812629288297137499e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.85052416252910403272283619201501701345e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45276747409453182009917448097687214033e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87717215449690275562288513806049961791e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96583424263422661540930513525639950307e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73001838976297286477856104855182595364e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29209801725936746054703603946844929105e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.31809396176316042818100839595926947461e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.62125101720695030847208519302530333864e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22912823173107974750307098204717046200e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.28404310708078592866397210871397836013e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.33433860799478110495440617696667578486e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01779942752411055394079990371203135494e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.60870827161929649807734240735205100749e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.43275518144078080917466090587075581039e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.80287554756375373913082969626543154342e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00697535360590561244468004025972321465e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.23883308105457761862174623664449205327e-6),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        const static RealType lambda_bias = BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.451582705289454864726195229894882143572); // (= log(pi/2)+1)

        RealType sigma = exp(-x * constants::pi<RealType>() / 2 - lambda_bias);
        RealType s = exp(-sigma) / sqrt(sigma);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 9.2619e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[19] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.71658823632747235572391863987803415545e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20402452680758356732340074285765302037e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53870483364594487885882489517365212394e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.73525449564340671962525942038149851804e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.67339872142847248852186397385576389802e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.60644488744851390946293970736919678433e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33042051950636491987775324999025538357e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13846819893538329440033115143593487041e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41648498082970622389678372669789346515e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.74006867625631068946791714035394785978e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.12238896415831258936563475509362795783e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88070293465108791701905953972140154151e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.24813015654516014181209691083399092303e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.64092873079064926551281731026589848877e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.09892207654972883190432072151353819511e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.86990125202059013860642688739159455800e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62986611607135348214220687891374676368e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07567013469555215514702758084138467446e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.84619752008414239602732630339626773669e-14),
            };
            BOOST_MATH_STATIC const RealType Q[17] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.28669950018285475182750690468224641923e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12421557061005325313661189943328446480e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68376064122323574208976258468929505299e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22010354939562426718305463635398985290e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13795955314742199207524303721722785075e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.90452274425830801819532524004271355513e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.38324283887272345859359008873739301544e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15232844484261129757743512155821350773e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.79562237779621711674853020864686436450e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64370777996591099856555782918006739330e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.02327782881305686529414731684464770990e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27181015755595543140221119020333695667e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01121287947061613072815935956604529157e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44038164966032378909755215752715620878e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.39138685106442954199109662617641745618e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.83317957765031605023198891326325990178e-10),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -6.875) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 4.9208e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[20] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97500903816385095134217223320239082420e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02058997410109156148729828665298333233e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30492992901887465108077581566548743407e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08695332228530157560495896731847709498e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.54469321766529692240388930552986490213e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00543201281990041935310905273146022998e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08633547932070289660163851972658637916e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15432691192536747268886307936712580254e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.46179071338871656505293487217938889935e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45295210106393905833273975344579255175e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34638105523514101671944454719592801562e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15786069528793080046638424661219527619e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.54781306296697568446848038567723598851e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31977279631544580423883461084970429143e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.56616743805004179430469197497030496870e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60913959062328670735884196858280987356e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.91123354712008822789348244888916948822e-11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82453513391091361890763400931018529659e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.12671859603774617133607658779709622453e-14),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03211544596001317143519388487481133891e-20),
            };
            BOOST_MATH_STATIC const RealType Q[19] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.56188463983858614833914386500628633184e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27273165410457713017446497319550252691e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72495122287308474449946195751088057230e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64049710819255633163836824600620426349e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.53329810455612298967902432399110414761e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72302144446588066369304547920758875106e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.90680157119357595265085115978578965640e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87039785683949322939618337154059874729e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.64199530594973983893552925652598080310e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.88147828823178863054226159776600116931e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.91569503818223078110818909039307983575e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.89289385694964650198403071737653842880e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.32154679053642509246603754078168127853e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.43239674842248090516375370051832849701e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.03349866320207008385913232167927124115e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98307302768178927108235662166752511325e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07996042577029996321821937863373306901e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53576500935979732855511826033727522138e-13),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else {
            result = 0;
        }
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 53>& tag) {
    if (x >= 0) {
        return complement ? landau_cdf_plus_imp_prec(x, tag) : 1 - landau_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - landau_cdf_minus_imp_prec(x, tag) : landau_cdf_minus_imp_prec(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 113>& tag) {
    if (x >= 0) {
        return complement ? landau_cdf_plus_imp_prec(x, tag) : 1 - landau_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - landau_cdf_minus_imp_prec(x, tag) : landau_cdf_minus_imp_prec(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType landau_cdf_imp(const landau_distribution<RealType, Policy>& dist, const RealType& x, bool complement) {
    //
    // This calculates the cdf of the Landau distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::cdf(landau<%1%>&, %1%)";
    RealType result = 0;
    RealType location = dist.location();
    RealType scale = dist.scale();
    RealType bias = dist.bias();

    if (false == detail::check_location(function, location, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_scale(function, scale, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_x(function, x, &result, Policy()))
    {
        return result;
    }

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Landau distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale + bias;

    result = landau_cdf_imp_prec(u, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_lower_imp_prec(const RealType& p, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.375) {
        RealType t = p - static_cast < RealType>(0.375);

        // Rational Approximation
        // Maximum Absolute Error: 3.0596e-17
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(3.74557416577759554506e-2),
            static_cast<RealType>(3.87808262376545756299e0),
            static_cast<RealType>(4.03092288183382979104e0),
            static_cast<RealType>(-1.65221829710249468257e1),
            static_cast<RealType>(-6.99689838230114367276e0),
            static_cast<RealType>(1.51123479911771488314e1),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.37863773851525662884e-1),
            static_cast<RealType>(-6.35020262707816744534e0),
            static_cast<RealType>(3.07646508389502660442e-1),
            static_cast<RealType>(9.72566583784248877260e0),
            static_cast<RealType>(-2.72338088170674280735e0),
            static_cast<RealType>(-1.58608957980133006476e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - static_cast < RealType>(0.25);

        // Rational Approximation
        // Maximum Absolute Error: 5.2780e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(-4.17764764050720190117e-1),
            static_cast<RealType>(1.27887601021900963655e0),
            static_cast<RealType>(1.80329928265996817279e1),
            static_cast<RealType>(2.35783605878556791719e1),
            static_cast<RealType>(-2.67160590411398800149e1),
            static_cast<RealType>(-2.36192101013335692266e1),
            static_cast<RealType>(8.30396110938939237358e0),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(5.37459525158081633669e0),
            static_cast<RealType>(2.35696607501498012129e0),
            static_cast<RealType>(-1.71117034150268575909e1),
            static_cast<RealType>(-6.72278235529877170403e0),
            static_cast<RealType>(1.27763043804603299034e1),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - static_cast < RealType>(0.125);

        // Rational Approximation
        // Maximum Absolute Error: 6.3254e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-8.77109518013577785811e-1),
            static_cast<RealType>(-1.03442936529923615496e1),
            static_cast<RealType>(-1.03389868296950570121e1),
            static_cast<RealType>(2.01575691867458616553e2),
            static_cast<RealType>(4.59115079925618829199e2),
            static_cast<RealType>(-3.38676271744958577802e2),
            static_cast<RealType>(-5.38213647878547918506e2),
            static_cast<RealType>(1.99214574934960143349e2),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.64177607733998839003e1),
            static_cast<RealType>(8.10042194014991761178e1),
            static_cast<RealType>(7.61952772645589839171e1),
            static_cast<RealType>(-2.52698871224510918595e2),
            static_cast<RealType>(-1.95365983250723202416e2),
            static_cast<RealType>(2.61928845964255538379e2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 3.5192e-18
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(-8.77109518013577852585e-1),
            static_cast<RealType>(-1.08703720146608358678e0),
            static_cast<RealType>(-4.34198537684719253325e-1),
            static_cast<RealType>(-6.97264194535092564620e-2),
            static_cast<RealType>(-4.20721933993302797971e-3),
            static_cast<RealType>(-6.27420063107527426396e-5),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(8.38688797993971740640e-1),
            static_cast<RealType>(2.47558526682310722526e-1),
            static_cast<RealType>(3.03952783355954712472e-2),
            static_cast<RealType>(1.39226078796010665644e-3),
            static_cast<RealType>(1.43993679246435688244e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 1.1196e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-1.16727296241754548410e0),
            static_cast<RealType>(-1.12325365855062172009e0),
            static_cast<RealType>(-3.96403456954867129566e-1),
            static_cast<RealType>(-6.50024588048629862189e-2),
            static_cast<RealType>(-5.08582387678609504048e-3),
            static_cast<RealType>(-1.71657051345258316598e-4),
            static_cast<RealType>(-1.81536405273085024830e-6),
            static_cast<RealType>(-9.65262938333207656548e-10),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(7.55271574611337871389e-1),
            static_cast<RealType>(2.16323131117540100488e-1),
            static_cast<RealType>(2.92693206540519768049e-2),
            static_cast<RealType>(1.89396907936678571916e-3),
            static_cast<RealType>(5.20017914327360594265e-5),
            static_cast<RealType>(4.18896774212993675707e-7),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 1.0763e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-1.78348038398799868409e0),
            static_cast<RealType>(-7.74779087785346936524e-1),
            static_cast<RealType>(-1.27121601027522656374e-1),
            static_cast<RealType>(-9.86675785835385622362e-3),
            static_cast<RealType>(-3.69510132425310943600e-4),
            static_cast<RealType>(-6.00811940375633438805e-6),
            static_cast<RealType>(-3.06397799506512676163e-8),
            static_cast<RealType>(-7.34821360521886161256e-12),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(3.76606062137668223823e-1),
            static_cast<RealType>(5.37821995022686641494e-2),
            static_cast<RealType>(3.62736078766811383733e-3),
            static_cast<RealType>(1.16954398984720362997e-4),
            static_cast<RealType>(1.59917906784160311385e-6),
            static_cast<RealType>(6.41144889614705503307e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 9.9936e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-2.32474749499506229415e0),
            static_cast<RealType>(-4.81681429397597263092e-1),
            static_cast<RealType>(-3.79696253130015182335e-2),
            static_cast<RealType>(-1.42328672650093755545e-3),
            static_cast<RealType>(-2.58335052925986849305e-5),
            static_cast<RealType>(-2.03945574260603170161e-7),
            static_cast<RealType>(-5.04229972664978604816e-10),
            static_cast<RealType>(-5.49506755992282162712e-14),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.87186049570056737301e-1),
            static_cast<RealType>(1.32852903862611979806e-2),
            static_cast<RealType>(4.45262195863310928309e-4),
            static_cast<RealType>(7.13306978839226580931e-6),
            static_cast<RealType>(4.84555343060572391776e-8),
            static_cast<RealType>(9.65086092007764297450e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 9.2449e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-2.82318656228158372998e0),
            static_cast<RealType>(-2.84346379198027589453e-1),
            static_cast<RealType>(-1.09194719815749710073e-2),
            static_cast<RealType>(-1.99728160102967185378e-4),
            static_cast<RealType>(-1.77069359938827653381e-6),
            static_cast<RealType>(-6.82828539186572955883e-9),
            static_cast<RealType>(-8.22634582905944543176e-12),
            static_cast<RealType>(-4.10585514777842307175e-16),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(9.29910333991046040738e-2),
            static_cast<RealType>(3.27860300729204691815e-3),
            static_cast<RealType>(5.45852206475929614010e-5),
            static_cast<RealType>(4.34395271645812189497e-7),
            static_cast<RealType>(1.46600782366946777467e-9),
            static_cast<RealType>(1.45083131237841500574e-12),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 8.6453e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-3.29700011190686231229e0),
            static_cast<RealType>(-1.62920309130909343601e-1),
            static_cast<RealType>(-3.07152472866757852259e-3),
            static_cast<RealType>(-2.75922040607620211449e-5),
            static_cast<RealType>(-1.20144242264703283024e-7),
            static_cast<RealType>(-2.27410079849018964454e-10),
            static_cast<RealType>(-1.34109445298156050256e-13),
            static_cast<RealType>(-3.08843378675512185582e-18),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.62324092774919223927e-2),
            static_cast<RealType>(8.10410923007867515072e-4),
            static_cast<RealType>(6.70843016241177926470e-6),
            static_cast<RealType>(2.65459014339231700938e-8),
            static_cast<RealType>(4.45531791525831169724e-11),
            static_cast<RealType>(2.19324401673412172456e-14),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -256) {
        RealType t = -log2(ldexp(p, 128));

        // Rational Approximation
        // Maximum Relative Error: 8.2028e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-3.75666995985336008568e0),
            static_cast<RealType>(-9.15751436135409108392e-2),
            static_cast<RealType>(-8.51745858385908954959e-4),
            static_cast<RealType>(-3.77453552696508401182e-6),
            static_cast<RealType>(-8.10504146884381804474e-9),
            static_cast<RealType>(-7.55871397276946580837e-12),
            static_cast<RealType>(-2.19023097542770265117e-15),
            static_cast<RealType>(-2.34270094396556916060e-20),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(2.30119177073875808729e-2),
            static_cast<RealType>(2.00787377759037971795e-4),
            static_cast<RealType>(8.27382543511838001513e-7),
            static_cast<RealType>(1.62997898759733931959e-9),
            static_cast<RealType>(1.36215810410261098317e-12),
            static_cast<RealType>(3.33957268115953023683e-16),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -512) {
        RealType t = -log2(ldexp(p, 256));

        // Rational Approximation
        // Maximum Relative Error: 7.8900e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-4.20826069989721597050e0),
            static_cast<RealType>(-5.07864788729928381957e-2),
            static_cast<RealType>(-2.33825872475869133650e-4),
            static_cast<RealType>(-5.12795917403072758309e-7),
            static_cast<RealType>(-5.44657955194364350768e-10),
            static_cast<RealType>(-2.51001805474510910538e-13),
            static_cast<RealType>(-3.58448226638949307172e-17),
            static_cast<RealType>(-1.79092368272097571876e-22),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(1.14671758705641048135e-2),
            static_cast<RealType>(4.98614103841229871806e-5),
            static_cast<RealType>(1.02397186002860292625e-7),
            static_cast<RealType>(1.00544286633906421384e-10),
            static_cast<RealType>(4.18843275058038084849e-14),
            static_cast<RealType>(5.11960642868907665857e-18),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -1024) {
        RealType t = -log2(ldexp(p, 512));

        // Rational Approximation
        // Maximum Relative Error: 7.6777e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-4.65527239540648658214e0),
            static_cast<RealType>(-2.78834161568280967534e-2),
            static_cast<RealType>(-6.37014695368461940922e-5),
            static_cast<RealType>(-6.92971221299243529202e-8),
            static_cast<RealType>(-3.64900562915285147191e-11),
            static_cast<RealType>(-8.32868843440595945586e-15),
            static_cast<RealType>(-5.87602374631705229119e-19),
            static_cast<RealType>(-1.37812578498484605190e-24),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(5.72000087046224585566e-3),
            static_cast<RealType>(1.24068329655043560901e-5),
            static_cast<RealType>(1.27105410419102416943e-8),
            static_cast<RealType>(6.22649556008196699310e-12),
            static_cast<RealType>(1.29416254332222127404e-15),
            static_cast<RealType>(7.89365027125866583275e-20),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else{
        result = -boost::math::numeric_limits<RealType>::infinity();
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_lower_imp_prec(const RealType& p, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.375) {
        RealType t = p - 0.375;

        // Rational Approximation
        // Maximum Absolute Error: 2.5723e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.74557416577759248536854968412794870581e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.04379368253541440583870397314012269006e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12622841210720956864564105821904588447e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.57744422491408570970393103737579322242e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.13509711945094517370264490591904074504e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.18322789179144512109337184576079775889e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21447613719864832622177316196592738866e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49076304733407444404640803736504398642e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.96654951892056950374719952752959986017e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.73083458872938872583408218098970368331e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.22584946471889320670122404162385347867e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98534922151507267157370682137856253991e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.09159286510191893522643172277831735606e0),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.86204686129323171601167115178777357431e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.43698274248278918649234376575855135232e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.75240332521434608696943994815649748669e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.31438891446345558658756610288653829009e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10716029191240549289948990305434475528e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.10878330779477313404660683539265890549e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.52360069933886703736010179403700697679e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.15864312939821257811853678185928982258e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.10341116017481903631605786613604619909e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29121822170912306719250697890270750964e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.56489746112937744052098794310386515793e1),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - 0.25;

        // Rational Approximation
        // Maximum Absolute Error: 6.1583e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.17764764050720242897742634974454113395e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.80044093802431965072543552425830082205e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23613318632011593171919848575560968064e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77438013844838858458786448973516177604e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.62569530523012138862025718052954558264e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.02005706260864894793795986187582916504e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.29383609355165614630538852833671831839e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.09367754001841471839736367284852087164e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45744413840415901080013900562654222567e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.41920296534143581978760545125050148256e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.94857580745127596732818606388347624241e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.02847586753967876900858299686189155164e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.29953583375818707785500963989580066735e1),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27455303165341271216882778791555788609e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.41762124591820618604790027888328605963e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.30845760165840203715852751405553821601e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.00827370048057599908445731563638383351e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.19621193929561206904250173267823637982e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.10514757798726932158537558200005910184e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.79738493761540403010052092523396617472e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.94101664430520833603032182296078344870e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.31586575577250608890806988616823861649e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.93650751613703379272667745729529916084e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.52472388998113562780767055981852228229e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.01428305018551686265238906201345171425e0),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - 0.125;

        // Rational Approximation
        // Maximum Absolute Error: 1.3135e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.77109518013577849065583862782160121458e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.05813204052660740589813216397258899528e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.19628607167020425528944673039894592264e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.67162644860799051148361885190022738759e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.05921446080443979618622123764941760355e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.26685085062411656483492973256809500654e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17117538916032273474332064444853786788e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.45059470468014721314631799845029715639e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28952226224720891553119529857430570919e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.98502296814963504284919407719496390478e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.10876326351879104392865586365509749012e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.70358021544406445036220918341411271912e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.49724346845064961378591039928633169443e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.23815021378788622035604969476085727123e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.17262073948257994617723369387261569086e4),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.94901665980514882602824575757494472790e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.54328910175180674300123471690771017388e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84847502738788846487698327848593567941e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98451502799612368808473649408471338893e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.13744760159877712051088928513298431905e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.20745061658519699732567732006176366700e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.68622317228909264645937229979147883985e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.96020751551679746882793283955926871655e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.88860541272346724142574740580038834720e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.73454107207588310809238143625482857512e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.23165643368613191971938741926948857263e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.94832163019509140191456686231012184524e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.26616234097287315007047356261933409072e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24686019847093806280148917466062407447e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 2.0498e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.77109518013577849065583862782160155093e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.77585398076895266354686007069850894777e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.47522378123968853907102309276280187353e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.63343576432650242131602396758195296288e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.77801189859227220359806456829683498508e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.93221663334563259732178473649683953515e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.95272757466323942599253855146019408376e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.73624853556509653351605530630788087166e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.41317699770351712612969089634227647374e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.34187895701093934279414993393750297714e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.64090928155753225614302094820737249510e-10),
        };
        BOOST_MATH_STATIC const RealType Q[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62401464973350962823995096121206419019e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11979822811128264831341485706314465894e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27257342406829987209876262928379300361e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.85505879705365729768944032174855501091e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40983451000610516082352700421098499905e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23459897865681009685618192649929504121e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.28925214684463186484928824536992032740e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67647262682850294124662856194944728023e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88173142080572819772032615169461689904e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.07756799117728455728056041053803769069e-11),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 6.7643e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.16727296241754547290632950718657117630e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.97822895738734630842909028778257589627e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.45580723831325060656664869189975355503e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.13835678647158936819843386298690513648e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.64536831064884519168892017327822018961e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.93786616484143556451247457584976578832e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.55770899078184683328915310751857391073e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.91778173446401005072425460365992356304e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.59064619930808759325013814591048817325e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.54786673836080683521554567617693797315e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.15917340396537949894051711038346411232e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.29633344043292285568750868731529586549e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.27785620133198676852587951604694784533e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.89999814745618370028655821500875451178e-16),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48772690114094395052120751771215809418e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.72281013057830222881716429522080327421e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.67186370229687087768391373818683340542e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.86988148601521223503040043124617333773e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43321724586909919175166704060749343677e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57428821868404424742036582321713763151e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17165774858274087452172407067668213010e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.78674439389954997342198692571336875222e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82045374895858670592647375231115294575e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40152058277291349447734231472872126483e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.34789603687129472952627586273206671442e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38087376350052845654180435966624948994e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34945081364333330292720602508979680233e-16),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 6.4987e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.78348038398799867332294266481364810762e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.42913983922316889357725662957488617770e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.02077376277824482097703213549730657663e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.01799479940825547859103232846394236067e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.31954083060883245879038709103320778401e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.14437110578260816704498035546280169833e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.75434713435598124790021625988306358726e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.70722283097111675839403787383067403199e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.22792548204908895458622068271940298849e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.23652632092726261134927067083229843867e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.26848751206698811476021875382152874517e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.96683933776920842966962054618493551480e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.65547426464916480144982028081303670013e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.01788104318587272115031165074724363239e-19),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.43507070588695242714872431565299762416e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.42808541175677232789532731946043918868e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.58154336417481327293949514291626832622e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.52883128062761272825364005132296437324e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46220303655089035098911370014929809787e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.44776253795594076489612438705019750179e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.09607872267766585503592561222987444825e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24270418154050297788150584301311027023e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.52138350835458198482199500102799185922e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.28330565098807415367837423320898722351e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61220858078610415609826514581165467762e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31680570822471881148008283775281806658e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61638868324981393463928986484698110415e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 6.4643e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.32474749499506228416012679106564727824e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.11125026189437033131539969177846635890e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.56906722402983201196890012041528422765e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.56242546565817333757522889497509484980e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.96189353402888611791301502740835972176e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.25518459970705638772495930203869523701e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.23831474024265607073689937590604367113e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.44925744847701733694636991148083680863e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.22891322042392818013643347840386719351e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.72860750698838897533843164259437533533e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.38276123679972197567738586890856461530e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.75010927807240165715236750369730131837e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.39435252454410259267870094713230289131e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.32672767938414655620839066142834241506e-23),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.71913035066927544877255131988977106466e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.07499674325721771035402891723823952963e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.73304002376509252638426379643927595435e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.45986195188119302051678426047947808068e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39631771214004792103186529415117786213e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.82972151546053891838685817022915476363e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.11484161875982352879422494936862579004e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.76886416872139526041488219568768973343e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.88160764330501845206576873052377420740e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20653899535657202009579871085255085820e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.86752135706343102514753706859178940399e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.08670633989984379551412930443791478495e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96869107941293302786688580824755244599e-24),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 6.2783e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.82318656228158372073367735499501003484e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.46261040951642110189344545942990712460e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.64741560190892266676648641695426188913e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.28753551974093682831398870653055328683e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.21312013770915263838500863217194379134e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.52436958859473873340733176333088176566e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.19550238139736009251193868269757013675e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.14964971787780037500173882363122301527e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.40304301938210548254468386306034204388e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.86982233973109416660769999752508002999e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.47229710624085810190563630948355644978e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.97511060097659395674010001155696382091e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.14321784268659603072523892366718901165e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.69804409248161357472540739283978368871e-27),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85763741109198600677877934140774914793e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51555423561034635648725665049090572375e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.14334282485948451530639961260946534734e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15303265564789411158928907568898290494e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.33945229806307308687045028827126348382e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.94373901322371782367428404051188999662e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.51420073260465851038482922686870398511e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.39366317896256472225488167609473929757e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32986474655329330922243678847674164814e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.09408217872473269288530036223761068322e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79051953285476930547217173280519421410e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94530899348454778842122895096072361105e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36452993460830805591166007621343447892e-28),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 6.0123e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.29700011190686230364493911161520668302e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.16175031776740080906111179721128106011e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.33343982195432985864570319341790342784e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.25414682801788504282484273374052405406e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.08812659343240279665150323243172015853e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.33251452861660571881208437468957953698e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.80894766863868081020089830941243893253e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.84955155823472122347227298177346716657e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.98637322645260158088125181176106901234e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.24174383760514163336627039277792172744e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.54369979866464292009398761404242103210e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.02572051048819721089874338860693952304e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.93656169061287808919601714139458074543e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.14930159772574816086864316805656403181e-31),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27154900915819978649344191118112870943e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77527908332591966425460814882436207182e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.88097249712649070373643439940164263005e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33593973311650359460519742789132084170e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34383186845963127931313004467487408932e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.18631088001587612168708294926967112654e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.25338215226314856456799568077385137286e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30644713290591280849926388043887647219e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55508263112797212356530850090635211577e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.96694528841324480583957017533192805939e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81411886190142822899424539396403206677e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.64683991040772975824276994623053932566e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81924597500766743545654858597960153152e-32),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -256) {
        RealType t = -log2(ldexp(p, 128));

        // Rational Approximation
        // Maximum Relative Error: 5.7624e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.75666995985336007747791649448887723610e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.78960399079208663111712385988217075907e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.74942252057371678208959612011771010491e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.54567765510065203543937772001248399869e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.53093894540157655856029322335609764674e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.83833601054721321664219768559444646069e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.52281007055180941965172296953524749452e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.57322728543196345563534040700366511864e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.52881564741260266060082523971278782893e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.60440334652864372786302383583725866608e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.60691285483339296337794569661545125426e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.29567560587442907936295101146377006338e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.50976593324256906782731237116487284834e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.71308835356954147218854223581309967814e-35),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.62732785401286024270119905692156750540e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.40382961238668912455720345718267045656e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10406445824749289380797744206585266357e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.28896702052362503156922190248503561966e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.16141168910009886089186579048301366151e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41978644147717141591105056152782456952e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.28101353275172857831967521183323237520e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.02996252940600644617348281599332256544e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.91006255647885778937252519693385130907e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84585864559619959844425689120130028450e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.68573963627097356380969264657086640713e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11059307697054035905630311480256015939e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36363494270701950295678466437393953964e-36),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -512) {
        RealType t = -log2(ldexp(p, 256));

        // Rational Approximation
        // Maximum Relative Error: 5.5621e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.20826069989721596260510558511263035942e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.97440158261228371765435988840257904642e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.03971528248920108158059927256206438162e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.27123766722395421727031536104546382045e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.42341191105097202061646583288627536471e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.27644514375284202188806395834379509517e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.10772944192965679212172315655880689287e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.32875098791800400229370712119075696952e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.06204614360238210805757647764525929969e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.43745006810807466452260414216858795476e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.66712970893511330059273629445122037896e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.72840198778128683137250377883245540424e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.91906782399731224228792112460580813901e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.42769091263044979075875010403899574987e-39),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31008507886426704374911618340654350029e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34370110384866123378972324145883460422e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37370811166006065198348108499624387519e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.14880753458828334658200185014547794333e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29049942195929206183214601044522500821e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19814793427532184357255406261941946071e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53609759199568827596069048758012402352e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94113467521833827559558236675876398395e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.46066673213431758610437384053309779874e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73781952388557106045597803110890418919e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.17225106466605017267996611448679124342e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66384334839761400228111118435077786644e-35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.02877806111195383689496741738320318348e-40),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -1024) {
        RealType t = -log2(ldexp(p, 512));

        // Rational Approximation
        // Maximum Relative Error: 5.4128e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.65527239540648657446629479052874029563e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.49609214609793557370425343404734771058e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.85355312961840000203681352424632999367e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.57243631623079865238801420669247289633e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.64978384343879316016184643597712973486e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.12776782513387823319217102727637716531e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.86985041780323969283076332449881856202e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.11665149267826038417038582618446201377e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.44259259232002496618805591961855219612e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.13186466395317710362065595347401054176e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.72627240737786709568584848420972570566e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.40450635670659803069555960816203368299e-33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.90550919589933206991152832258558972394e-38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.33785768143117121220383154455316199086e-43),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15365252334339030944695314405853064901e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84519641047962864523571386561993045416e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71097850431873211384168229175171958023e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20268329795802836663630276028274915013e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.00891848515558877833795613956071967566e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41031229424613259381704686657785733606e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96506914235910190020798805190634423572e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.51190756655665636680121123277286815188e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82855686000721415124702578998188630945e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64298533757673219241102013167519737553e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01176104624443909516274664414542493718e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.41571947895162847564926590304679876888e-39),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.84682590163505511580949151048092123923e-44),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -2048) {
        RealType t = -log2(ldexp(p, 1024));

        // Rational Approximation
        // Maximum Relative Error: 5.3064e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.09971143249822249471944441552701756051e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.00154235169065403254826962372636417554e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.76859552294270710004718457715250134998e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.16331901379268792872208226779641113312e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.11590258438815173520561213981966313758e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.17278804462968109983985217400233347654e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.14112976645884560534267524918610371127e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.34652102658577790471066054415469309178e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.85242987373551062800089607781071064493e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.35051904844102317261572436130886083833e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.78478298776769981726834169566536801689e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.22532973433435489030532261530565473605e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.25433604872532935232490414753194993235e-41),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.49792182967344082832448065912949074241e-47),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.76316274347013095030195725596822418859e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45872499993438633169552184478587544165e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13309566903496793786045158442686362533e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99468690853840997883815075627545315449e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24734617022827960185483615293575601906e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.30099852343633243897084627428924039959e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52598626985708878790452436052924637029e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91432956461466900007096548587800675801e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.54421383015859327468201269268335476713e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55939743284103455997584863292829252782e-33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.73331214275752923691778067125447148395e-38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55089353084326800338273098565932598679e-42),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.85408276119483460035366338145310798737e-48),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4096) {
        RealType t = -log2(ldexp(p, 2048));

        // Rational Approximation
        // Maximum Relative Error: 5.2337e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.54271778755494231572464179212263718102e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.62737121212473668543011440432166267791e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.10099492629239750693134803100262740506e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.56925359477960645026399648793960646858e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.50756287005636861300081510456668184335e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.40657453971177017986596834420774251809e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.25518001919157628924245515302669097090e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.79618511101781942757791021761865762100e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.70242241511341924787722778791482800736e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.87078860748428154402226644449936091766e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.84256560347986567120140826597805016470e-35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.71419536977123330712095123316879755172e-40),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.20746149769511232987820552765701234564e-45),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.97544543671003989410397788518265345930e-51),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87980995989632171985079518382705421728e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.64234691529227024725728122489224211774e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66149363392892604040036997518509803848e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24365427902918684575287447585802611012e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.88619663977804926166359181945671853793e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.25299846770395565237726328268659386749e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18720305257346902130922082357712771134e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13302092710568005396855019882472656722e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.88571994886818976015465466797965950164e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.48486836614948668196092864992423643733e-36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72247229252387482782783442901266890088e-41),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76046592638280288324495546006105696670e-46),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.09378739037162732758860377477607829024e-52),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -8192) {
        RealType t = -log2(ldexp(p, 4096));

        // Rational Approximation
        // Maximum Relative Error: 5.1864e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.98493298246627952401490656857159302716e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.76990949843357898517869703626917264559e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.65042794324685841303461715489845834903e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.10605446678026983843303253925148000808e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.02762962429283889606329831562937730874e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.04105882385534634676234513866095562877e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.76001901462155366759952792570076976049e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.48255362964603267691139956218580946011e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.19422119466925125740484046268759113569e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.00719439924828639148906078835399693640e-33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.89921842231783558951433534621837291030e-38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.90738353848476619269054038082927243972e-43),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.19893980415902021846066305054394089887e-49),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.85434486590981105149494168639321627061e-55),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43932168599456260558411716919165161381e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.09851997458503734167541584552305867433e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.32285843258966417340522520711168738158e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.76041266755635729156773747720864677283e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21202253509959946958614664659473305613e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28647335562574024550800155417747339700e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.24953333571478743858014647649207040423e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.41206199962423704137133875822618501173e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34018702380092542910629787632780530080e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41732943566503750356718429150708698018e-39),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29626943299239081309470153019011607254e-44),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.13947437500822384369637881437951570653e-50),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.61766557173110449434575883392084129710e-56),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -16384) {
        RealType t = -log2(ldexp(p, 8192));

        // Rational Approximation
        // Maximum Relative Error: 5.1568e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.42671464308364892089984144203590292562e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.70165333325375920690660683988390032004e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.51230594210711745541592189387307516997e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.81387249912672866168782835177116953008e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.35308063526816559199325906123032162155e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.67672500455361049516022171111707553191e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.45533142942305626136621399056034449775e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.37422341389432268402917477004312957781e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.32123176403616347106899307416474970831e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.07816837508332884935917946618577512264e-36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.95418937882563343895280651308376855123e-41),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.78256110112636303941842779721479313701e-47),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.09423327107440352766843873264503717048e-52),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.12386630925835960782702757402676887380e-58),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.19477412398085422408065302795208098500e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27347517804649548179786994390985841531e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15042399607786347684366638940822746311e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.84537882580411074097888848210083177973e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.78283331111405789359863743531858801963e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00711555012725961640684514298170252743e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.21370151454170604715234671414141850094e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72008007024350635082914256163415892454e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.61182095564217124712889821368695320635e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35498047010165964231841033788823033461e-42),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11828030216193307885831734256233140264e-47),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22499308298315468568520585583666049073e-53),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04877018522402283597555167651619229959e-59),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        result = -boost::math::numeric_limits<RealType>::infinity();
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.375) {
        RealType t = p - static_cast < RealType>(0.375);

        // Rational Approximation
        // Maximum Relative Error: 5.1286e-20
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.31348919222343858178e0),
            static_cast<RealType>(-1.06646675961352786791e0),
            static_cast<RealType>(-1.80946160022120488884e1),
            static_cast<RealType>(-1.53457017598330440033e0),
            static_cast<RealType>(4.71260102173048370028e1),
            static_cast<RealType>(4.61048467818771410732e0),
            static_cast<RealType>(-2.80957284947853532418e1),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.71007453129016317772e0),
            static_cast<RealType>(1.31946404969596908872e0),
            static_cast<RealType>(-1.70321827414586880227e1),
            static_cast<RealType>(-1.11253495615474018666e1),
            static_cast<RealType>(1.62659086449959446986e1),
            static_cast<RealType>(7.37109203295032098763e0),
            static_cast<RealType>(-2.43898047338699777337e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - static_cast < RealType>(0.25);

        // Rational Approximation
        // Maximum Relative Error: 3.4934e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.55081568282045924981e0),
            static_cast<RealType>(5.38750533719526696218e0),
            static_cast<RealType>(-2.32797421725187349036e1),
            static_cast<RealType>(-3.96043566411306749784e1),
            static_cast<RealType>(3.80609941977115436545e1),
            static_cast<RealType>(3.35014421131920266346e1),
            static_cast<RealType>(-1.17490458743273503838e1),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(7.52439409918350484765e0),
            static_cast<RealType>(1.34784954182866689668e1),
            static_cast<RealType>(-9.21002543625052363446e0),
            static_cast<RealType>(-2.67378141317474265949e1),
            static_cast<RealType>(2.10158795079902783094e0),
            static_cast<RealType>(5.90098096212203282798e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - static_cast < RealType>(0.125);

        // Rational Approximation
        // Maximum Relative Error: 4.0795e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(5.68160868054034111703e0),
            static_cast<RealType>(1.06098927525586705381e2),
            static_cast<RealType>(5.74509518025029027944e2),
            static_cast<RealType>(4.91117375866809056969e2),
            static_cast<RealType>(-2.92607000654635606895e3),
            static_cast<RealType>(-3.82912009541683403499e3),
            static_cast<RealType>(2.49195208452006100935e3),
            static_cast<RealType>(1.29413301335116683836e3),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1),
            static_cast<RealType>(2.69603865809599480308e1),
            static_cast<RealType>(2.63378422475372461819e2),
            static_cast<RealType>(1.09903493506098212946e3),
            static_cast<RealType>(1.60315072092792425370e3),
            static_cast<RealType>(-5.44710468198458322870e2),
            static_cast<RealType>(-1.76410218726878681387e3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 4.4618e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(7.10201085067542566037e-1),
            static_cast<RealType>(6.70042401812679849451e-1),
            static_cast<RealType>(2.42799404088685074098e-1),
            static_cast<RealType>(4.80613880364042262227e-2),
            static_cast<RealType>(6.04473313360581797461e-3),
            static_cast<RealType>(5.09172911021654842046e-4),
            static_cast<RealType>(-6.63145317984529265677e-6),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1),
            static_cast<RealType>(9.18649629646213969612e-1),
            static_cast<RealType>(3.66343989541898286306e-1),
            static_cast<RealType>(8.01010534748206001446e-2),
            static_cast<RealType>(1.00553335007168823115e-2),
            static_cast<RealType>(6.30966763237332075752e-4),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 5.8994e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(7.06147398566773538296e-1),
            static_cast<RealType>(4.26802162741800814387e-1),
            static_cast<RealType>(1.32254436707168800420e-1),
            static_cast<RealType>(2.86055054496737936396e-2),
            static_cast<RealType>(3.63373131686703931514e-3),
            static_cast<RealType>(3.84438945816411937013e-4),
            static_cast<RealType>(1.67768561420296743529e-5),
            static_cast<RealType>(8.76982374043363061978e-7),
            static_cast<RealType>(-1.99744396595921347207e-8),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(6.28190787856605587324e-1),
            static_cast<RealType>(2.10992746593815791546e-1),
            static_cast<RealType>(4.44397672327578790713e-2),
            static_cast<RealType>(6.02768341661155914525e-3),
            static_cast<RealType>(5.46578619531721658923e-4),
            static_cast<RealType>(3.11116573895074296750e-5),
            static_cast<RealType>(1.17729007979018602786e-6),
            static_cast<RealType>(-2.78441865351376040812e-8),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 8.8685e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.48209596014908359251e-1),
            static_cast<RealType>(2.52611824671691390768e-1),
            static_cast<RealType>(4.65114070477803399291e-2),
            static_cast<RealType>(5.23373513313686849909e-3),
            static_cast<RealType>(3.83113384161076881958e-4),
            static_cast<RealType>(1.96230077517629530809e-5),
            static_cast<RealType>(5.83117485120890819338e-7),
            static_cast<RealType>(6.92614450423703079737e-9),
            static_cast<RealType>(-3.89531123166658723619e-10),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(3.99413988076189200840e-1),
            static_cast<RealType>(7.32068638518417765776e-2),
            static_cast<RealType>(8.15517102642752348889e-3),
            static_cast<RealType>(6.09126071418098074914e-4),
            static_cast<RealType>(3.03794079468789962611e-5),
            static_cast<RealType>(9.32109079205017197662e-7),
            static_cast<RealType>(1.05435710482490499583e-8),
            static_cast<RealType>(-6.08748435983193979360e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 1.0253e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(6.36719010559816164896e-1),
            static_cast<RealType>(2.06504115804034148753e-1),
            static_cast<RealType>(3.28085429275407182582e-2),
            static_cast<RealType>(3.31676417519020335859e-3),
            static_cast<RealType>(2.35502578757551086372e-4),
            static_cast<RealType>(1.21652240566662139418e-5),
            static_cast<RealType>(4.57039495420392748658e-7),
            static_cast<RealType>(1.18090959236399583940e-8),
            static_cast<RealType>(1.77492646969597480221e-10),
            static_cast<RealType>(-2.19331267300885448673e-17),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(3.24422807416528490276e-1),
            static_cast<RealType>(5.15290129833049138552e-2),
            static_cast<RealType>(5.21051235888272287209e-3),
            static_cast<RealType>(3.69895399249472399625e-4),
            static_cast<RealType>(1.91103139437893226482e-5),
            static_cast<RealType>(7.17882574725373091636e-7),
            static_cast<RealType>(1.85502934977316481559e-8),
            static_cast<RealType>(2.78798057565507249164e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 8.1705e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(6.36619775525705206992e-1),
            static_cast<RealType>(2.68335698140634792041e-1),
            static_cast<RealType>(5.49803347535070103650e-2),
            static_cast<RealType>(7.25018344556356907109e-3),
            static_cast<RealType>(6.87753481255849254220e-4),
            static_cast<RealType>(4.86155006277788340253e-5),
            static_cast<RealType>(2.84604768310787862450e-6),
            static_cast<RealType>(9.56133960810049319917e-8),
            static_cast<RealType>(5.26850116571886385248e-9),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1),
            static_cast<RealType>(4.21500730173440590900e-1),
            static_cast<RealType>(8.63629077498258325752e-2),
            static_cast<RealType>(1.13885615328098640032e-2),
            static_cast<RealType>(1.08032064178130906887e-3),
            static_cast<RealType>(7.63650498196064792408e-5),
            static_cast<RealType>(4.47056124637379045275e-6),
            static_cast<RealType>(1.50189171357721423127e-7),
            static_cast<RealType>(8.27574227882033707932e-9),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else {
        result = 2 / (constants::pi<RealType>() * p);
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.4375) {
        RealType t = p - 0.4375;

        // Rational Approximation
        // Maximum Relative Error: 1.4465e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.08338732735341567163440035550389989556e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.27245731792290848390848202647311435023e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.29317169036386848462079766136373749420e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36342136825575317326816540539659955416e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.31108700679715257074164180252148868348e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.81863611749256385875333154189074054367e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.11618233433781722149749739225688743102e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45241854625686954669050322459035410227e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.09780430233523239228350030812868983054e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.42232005306623465126477816911649683789e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.24816048952817367950452675590290535540e0),
        };
        BOOST_MATH_STATIC const RealType Q[10] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80464069267458650284548842830642770344e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.28240205449280944407125436342013240876e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.94145088402407692372903806765594642452e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30062294376971843436236253827463203953e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47118047660686070998671803800237836970e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.00643263133479482753298910520340235765e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.79460803824650509439313928266686172255e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32647058691746306769699006355256099134e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.59208938705683333141038012302171324544e0),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.375) {
        RealType t = p - 0.375;

        // Rational Approximation
        // Maximum Relative Error: 5.1929e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31348919222343858173602105619413801018e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.02800226274700443079521563669609776285e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.02091675505570786434803291987263553778e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50141943970885120432710080552941486001e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.93099903417013423125762526465625227789e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.56412922160141953385088141936082249641e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.47026602535072645589119440784669747242e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.01068960815396205074336853052832780888e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.86591619131639705495877493344047777421e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.26390836417639942474165178280649450755e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.18212484486162942333407102351878915285e0),
        };
        BOOST_MATH_STATIC const RealType Q[10] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97802777458574322604171035748634755981e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.33277809211107726455308655998819166901e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.76555481647551088626503871996617234475e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.33146828123660043197526014404644087069e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.65159900182434446550785415837526228592e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.32391192521438191878041140980983374411e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.12112886240590711980064990996002999330e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.93964809733838306198746831833843897743e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53948309965401603055162465663290204205e1),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - 0.25;

        // Rational Approximation
        // Maximum Relative Error: 3.2765e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55081568282045925871949387822806890848e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21080883686702131458668798583937913025e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15083151599213113740932148510289036342e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.94190629930345397070104862391009053509e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.40768205403470729468297576291723141480e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00001008242667338579153437084294876585e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70900785394455368299616221471466320407e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48947677419760753410122194475234527150e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.01826174001050912355357867446431955195e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.55833657916143927452986099130671173511e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.32953617526068647169047596631564287934e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.32825234826729794599233825734928884074e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.47352171888649528242284500266830013906e1),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40793887011403443604922082103267036101e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.04348824299115035210088417095305744248e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19680004238557953382868629429538716069e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.31172263627566980203163658640597441741e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.07390429662527773449936608284938592773e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.94877589960261706923147291496752293313e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.94903802003585398809229608695623474341e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.80417437710146805538675929521229778181e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.23364098614130091185959973343748897970e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.12975537807357019330268041620753617442e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.36592279898578127130605391750428961301e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18495624730372864715421146607185990918e1),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - 0.125;

        // Rational Approximation
        // Maximum Relative Error: 1.8007e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.68160868054034088524891526884683014057e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85165791469635551063850795991424359350e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.42938802867742165917839659578485422534e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59273512668331194186228996665355137458e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.91680503091725091370507732042764517726e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85642348415580865994863513727308578556e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.90181935466760294413877600892013910183e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.89141276256233344773677083034724024215e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00250514074918631367419468760920281159e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28168216451109123143492880695546179794e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14996399533648172721538646235459709807e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.58122093722347315498230864294015130011e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.25168985723506298009849577846542992545e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01179759985059408785527092464505889999e5),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08766677593618443545489115711858395831e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.05163374816838964338807027995515659842e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.62582103160439981904537982068579322820e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.62170991799612186300694554812291085206e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11013837158432827711075385018851760313e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45458895395245243570930804678601511371e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.08336489932795411216528182314354971403e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.11314692423102333551299419575616734987e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.43287683964711678082430107025218057096e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.62052814931825182298493472041247278475e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91440920656902450957296030252809476245e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.54913345383745613446952578605023052270e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.76034827722473399290702590414091767416e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94027684838690965214346010602354223752e3),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 6.1905e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.10201085067542610656114408605853786551e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.04725580445598482170291458376577106746e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.35945839005443673797792325217359695272e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15894004364989372373490772246381545906e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.54550169514753150042231386414687368032e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50389998399729913427837945242228928632e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75018554725308784191307050896936055909e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95901705695908219804887362154169268380e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34386856794684798098717884587473860604e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.89025399683852111061217430321882178699e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19044156703773954109232310846984749672e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11932910013840927659486142481532276176e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64064398716881126082770692219937093427e-10),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24909572944428286558287313527068259394e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.70912720447370835699164559729287157119e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.21998644852982625437008410769048682388e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.95906385698373052547496572397097325447e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35344144061390771459100718852878517200e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34168669072527413734185948498168454149e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.24488907049996230177518311480230131257e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.92059624838630990024209986717533470508e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84464614954263838504154559314144088371e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.67874815200287308180777775077428545024e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65919857481420519138294080418011981524e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.31466713452016682217190521435479677133e-10),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 8.5157e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.06147398566773479301585022897491054494e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06137881154706023038556659418303323027e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.00274868819366386235164897614448662308e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.03481313941011533876096564688041226638e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50172569438851062169493372974287427240e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.33370725278950299189434839636002761850e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.97566905908106543054773229070602272718e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85701973515993932384374087677862623215e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81956143385351702288398705969037130205e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.49975572102999645354655667945479202048e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.54665400959860442558683245665801873530e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.94292402413454232307556797758030774716e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98038791388715925556623187510676330309e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11242951548709169234296005470944661995e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.92636379295018831848234711132457626676e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77389296072621088586880199705598178518e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.57808410784300002747916947756919004207e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.93860773322862111592582321183379587624e-16),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52683694883265337797012770275040297516e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17837082293165509684677505408307814500e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.06195236296471366891670923430225774487e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29459155224640682509948954218044556307e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.71350726081102446771887145938865551618e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55986063168260695680927535587363081713e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91996892322204645930710038043021675160e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.43907073162091303683795779882887569537e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.50034830055055263363497137448887884379e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.56615898355501904078935686679056442496e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.61099855362387625880067378834775577974e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.12940315230564635808566630258463831421e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73572881409271303264226007333510301220e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.77786420070246087920941454352749186288e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77914406265766625938477137082940482898e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.19708585422668069396821478975324123588e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40406059898292960948942525697075698413e-15),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 7.6812e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.48209596014908270566135466727658374314e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02026332003132864886056710532156370366e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68941634461905013212266453851941196774e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.61650792370551069313309111250434438540e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52611930219013953260661961529732777539e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29488123972430683478601278003510200360e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68806175827491046693183596144172426378e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51806782259569842628995584152985951836e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92353868262961486571527005289554589652e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21494769586031703137329731447673056499e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.39837421784601055804920937629607771973e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82216155524308827738242486229625170158e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04275785296896148301798836366902456306e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19999929939765873468528448012634122362e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.03583326787146398902262502660879425573e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59755092249701477917281379650537907903e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32583227076029470589713734885690555562e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.18237323554153660947807202150429686004e-20),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.38459552164692902984228821988876295376e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21584899508575302641780901222203752951e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19656836695824518143414401720590693544e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39821085943818944882332778361549212756e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60484296824768079700823824408428524933e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.22173385695010329771921985088956556771e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95541259523416810836752584764202086573e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02184255281138028802991551275755427743e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90825805251143907045903671893185297007e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00501277755608081163250456177637280682e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.43387099521800224735155351696799358451e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63751106922299101655071906417624415019e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.02796982349519589339629488980132546290e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26197249278457937947269910907701176956e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50981935956236238709523457678017928506e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.65309560070040982176772709693008187384e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42817828965851841104270899392956866435e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 2.8388e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36719010559816175149447242695581604280e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14714772485724956396126176973339095223e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.47792450677638612907408723539943311437e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14084804538576805298420530820092167411e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25784891219227004394312050838763762669e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06837168825575413225975778906503529455e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.26908306638706189702624634771158355088e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06396335535135452379658152785541731746e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89854018431899039966628599727721422261e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.48974049316978526855972339306215972434e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50886538662952684349385729585856778829e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.14095970401472469264258565259303801322e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71915162586912203234023473966563445362e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.46099196574734038609354417874908346873e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.69944075002490023348175340827135133316e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.37340205792165863440617831987825515203e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.87812199530402923085142356622707924805e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76810877067601573471489978907720495511e-24),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.94373217074550329856398644558576545146e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17462725343185049507839058445338783693e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79202779096887355136298419604918306868e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97583473532621831662838256679872014292e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67819154370257505016693473230060726722e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14182379349642191946237975301363902175e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.38367234191828732305257162934647076311e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98221340505887984555143894024281550376e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17648765147609962405833802498013198305e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.94091261341172666220769613477202626517e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12169979717068598708585414568018667622e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70043694579268983742161305612636042906e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.43651270200498902307944806310116446583e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.95266223996097470768947426604723764300e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15821111112681530432702452073811996961e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08041298058041360645934320138765284054e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.91893114159827950553463154758337724676e-24),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 1.8746e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619775525705288697351261475419832625e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29882145587771350744255724773409752285e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.07952726597277085327360888304737411175e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72928496414816922167597110591366081416e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.01641163277458693633771532254570177776e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65627339211110756774878685166318417370e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41953343652571732907631074381749818724e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.27682202874503433884090203197149318368e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33177176779158737868498722222027162030e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.34485618544363735547395633416797591537e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96996761199233617188435782568975757378e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.49631247632674130553464740647053162499e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.68090516971007163491968659797593218680e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39910262557283449853923535586722968539e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83704888007521886644896435914745476741e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87884425419276681417666064027484555860e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36600092466902449189685791563990733005e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.53604301472332155307661986064796109517e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.65664588229982894587678197374867153136e-40),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61098031370834273919229478584740981117e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.40810642301361416278392589243623940154e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.42874346984605660407576451987840217534e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.30896462654364903689199648803900475405e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.17246449391141576955714059812811712587e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22979790521806964047777145482613709395e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.85960899519336488582042102184331670230e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66273852553220863665584472398487539899e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15372731217983084923067673501176233172e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.09441788794783860366430915309857085224e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.06279112323261126652767146380404236150e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36359339522621405197747209968637035618e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19770526521305519813109395521868217810e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.88562963284557433336083678206625018948e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95128165317597657325539450957778690578e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14570923483883184645242764315877865073e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.69599603258626408321886443187629340033e-26),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 3.9915e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619772367581344576326594951209529606e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72456363182667891167613558295097711432e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74486567435450138741058930951301644059e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.94624522781897679952110594449134468564e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09848623985771449914778668831103210333e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.47493285141689711937343304940229517457e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.22134975575390048261922652492143139174e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30240148387764167235466713023950979069e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06917824188001432265980161955665997666e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27834220508404489112697949450988070802e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48663447630051388468872352628795428134e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.50514504588736921389704370029090421684e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18965303814265217659151418619980209487e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74200654214326267651127117044008493519e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.34060054352573532839373386456991657111e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.34240516843783954067548886404044879120e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07803703545135964499326712080667886449e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61500479431085205124031101160332446432e-23),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27973454499231032893774072677004977154e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.02401389920613749641292661572240166038e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24819328156695252221821935845914708591e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27210724381675120281861717194783977895e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01708007392492681238863778030115281961e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.63088069045476088736355784718397594807e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61660379368211892821215228891806384883e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67946125503415200067055797463173521598e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72040422051759599096448422858046040086e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33519997465950200122152159780364149268e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07666528975810553712124845533861745455e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86870262247486708096341722190198527508e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30713380444621290817686989936029997572e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.38899571664905345700275460272815357978e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46750157220118157937510816924752429685e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69337661543585547694652989893297703060e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53684359865963395505791671817598669527e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * p);
    }
    else {
        result = 2 / (constants::pi<RealType>() * p);
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 53>& tag)
{
    if (p > 0.5) 
    {
        return !complement ? landau_quantile_upper_imp_prec(1 - p, tag) : landau_quantile_lower_imp_prec(1 - p, tag);
    }

    return complement ? landau_quantile_upper_imp_prec(p, tag) : landau_quantile_lower_imp_prec(p, tag);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 113>& tag)
{
    if (p > 0.5) 
    {
        return !complement ? landau_quantile_upper_imp_prec(1 - p, tag) : landau_quantile_lower_imp_prec(1 - p, tag);
    }

    return complement ? landau_quantile_upper_imp_prec(p, tag) : landau_quantile_lower_imp_prec(p, tag);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType landau_quantile_imp(const landau_distribution<RealType, Policy>& dist, const RealType& p, bool complement)
{
    // This routine implements the quantile for the Landau distribution,
    // the value p may be the probability, or its complement if complement=true.

    constexpr auto function = "boost::math::quantile(landau<%1%>&, %1%)";
    BOOST_MATH_STD_USING // for ADL of std functions

    RealType result = 0;
    RealType scale = dist.scale();
    RealType location = dist.location();
    RealType bias = dist.bias();

    if (false == detail::check_location(function, location, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_scale(function, scale, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_probability(function, p, &result, Policy()))
    {
        return result;
    }

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Landau distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * (landau_quantile_imp_prec(p, complement, tag_type()) - bias);

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_mode_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(-0.42931452986133525017);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_mode_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, -0.42931452986133525016556463510885028346);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType landau_mode_imp(const landau_distribution<RealType, Policy>& dist)
{
    // This implements the mode for the Landau distribution,

    constexpr auto function = "boost::math::mode(landau<%1%>&, %1%)";
    BOOST_MATH_STD_USING // for ADL of std functions

    RealType result = 0;
    RealType scale = dist.scale();
    RealType location = dist.location();
    RealType bias = dist.bias();

    if (false == detail::check_location(function, location, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_scale(function, scale, &result, Policy()))
    {
        return result;
    }

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Landau distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * (landau_mode_imp_prec<RealType>(tag_type()) - bias);

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_median_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(0.57563014394507821440);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_median_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, 0.57563014394507821439627930892257517269);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType landau_median_imp(const landau_distribution<RealType, Policy>& dist)
{
    // This implements the median for the Landau distribution,

    constexpr auto function = "boost::math::median(landau<%1%>&, %1%)";
    BOOST_MATH_STD_USING // for ADL of std functions

    RealType result = 0;
    RealType scale = dist.scale();
    RealType location = dist.location();
    RealType bias = dist.bias();

    if (false == detail::check_location(function, location, &result, Policy()))
    {
        return result;
    }
    if (false == detail::check_scale(function, scale, &result, Policy()))
    {
        return result;
    }

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Landau distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * (landau_median_imp_prec<RealType>(tag_type()) - bias);

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_entropy_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(2.37263644000448182448);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType landau_entropy_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.3726364400044818244844049010588577710);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType landau_entropy_imp(const landau_distribution<RealType, Policy>& dist)
{
    // This implements the entropy for the Landau distribution,

    constexpr auto function = "boost::math::entropy(landau<%1%>&, %1%)";
    BOOST_MATH_STD_USING // for ADL of std functions

    RealType result = 0;
    RealType scale = dist.scale();

    if (false == detail::check_scale(function, scale, &result, Policy()))
    {
        return result;
    }

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Landau distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = landau_entropy_imp_prec<RealType>(tag_type()) + log(scale);

    return result;
}

} // detail

template <class RealType = double, class Policy = policies::policy<> >
class landau_distribution
{
    public:
    typedef RealType value_type;
    typedef Policy policy_type;

    BOOST_MATH_GPU_ENABLED landau_distribution(RealType l_location = 0, RealType l_scale = 1)
        : mu(l_location), c(l_scale)
    {
        BOOST_MATH_STD_USING
        
        constexpr auto function = "boost::math::landau_distribution<%1%>::landau_distribution";
        RealType result = 0;
        detail::check_location(function, l_location, &result, Policy());
        detail::check_scale(function, l_scale, &result, Policy());

        location_bias = -2 / constants::pi<RealType>() * log(l_scale);
    } // landau_distribution

    BOOST_MATH_GPU_ENABLED RealType location()const
    {
        return mu;
    }
    BOOST_MATH_GPU_ENABLED RealType scale()const
    {
        return c;
    }
    BOOST_MATH_GPU_ENABLED RealType bias()const
    {
        return location_bias;
    }

    private:
    RealType mu;    // The location parameter.
    RealType c;     // The scale parameter.
    RealType location_bias;  // = -2 / pi * log(c)
};

typedef landau_distribution<double> landau;

#ifdef __cpp_deduction_guides
template <class RealType>
landau_distribution(RealType) -> landau_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
landau_distribution(RealType, RealType) -> landau_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const landau_distribution<RealType, Policy>&)
{ // Range of permissible values for random variable x.
    BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<RealType>::has_infinity)
    {
        return boost::math::pair<RealType, RealType>(-boost::math::numeric_limits<RealType>::infinity(), boost::math::numeric_limits<RealType>::infinity()); // - to + infinity.
    }
    else
    { // Can only use max_value.
        using boost::math::tools::max_value;
        return boost::math::pair<RealType, RealType>(-max_value<RealType>(), max_value<RealType>()); // - to + max.
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const landau_distribution<RealType, Policy>&)
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
    BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<RealType>::has_infinity)
    {
        return boost::math::pair<RealType, RealType>(-boost::math::numeric_limits<RealType>::infinity(), boost::math::numeric_limits<RealType>::infinity()); // - to + infinity.
    }
    else
    { // Can only use max_value.
        using boost::math::tools::max_value;
        return boost::math::pair<RealType, RealType>(-tools::max_value<RealType>(), max_value<RealType>()); // - to + max.
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType pdf(const landau_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::landau_pdf_imp(dist, x);
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const landau_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::landau_cdf_imp(dist, x, false);
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const landau_distribution<RealType, Policy>& dist, const RealType& p)
{
    return detail::landau_quantile_imp(dist, p, false);
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<landau_distribution<RealType, Policy>, RealType>& c)
{
    return detail::landau_cdf_imp(c.dist, c.param, true);
} //  cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<landau_distribution<RealType, Policy>, RealType>& c)
{
    return detail::landau_quantile_imp(c.dist, c.param, true);
} // quantile complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const landau_distribution<RealType, Policy>&)
{  // There is no mean:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Landau Distribution has no mean");

    return policies::raise_domain_error<RealType>(
        "boost::math::mean(landau<%1%>&)",
        "The Landau distribution does not have a mean: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const landau_distribution<RealType, Policy>& /*dist*/)
{
    // There is no variance:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Landau Distribution has no variance");

    return policies::raise_domain_error<RealType>(
        "boost::math::variance(landau<%1%>&)",
        "The Landau distribution does not have a variance: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const landau_distribution<RealType, Policy>& dist)
{
    return detail::landau_mode_imp(dist);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const landau_distribution<RealType, Policy>& dist)
{
    return detail::landau_median_imp(dist);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const landau_distribution<RealType, Policy>& /*dist*/)
{
    // There is no skewness:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Landau Distribution has no skewness");

    return policies::raise_domain_error<RealType>(
        "boost::math::skewness(landau<%1%>&)",
        "The Landau distribution does not have a skewness: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy()); // infinity?
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const landau_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Landau Distribution has no kurtosis");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis(landau<%1%>&)",
        "The Landau distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const landau_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis excess:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Landau Distribution has no kurtosis excess");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis_excess(landau<%1%>&)",
        "The Landau distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const landau_distribution<RealType, Policy>& dist)
{
    return detail::landau_entropy_imp(dist);
}

}} // namespaces


#endif // BOOST_STATS_LANDAU_HPP
