//  Copyright Takuma Yoshimura 2024.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_SASPOINT5_HPP
#define BOOST_STATS_SASPOINT5_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/distributions/detail/derived_accessors.hpp>
#include <boost/math/tools/rational.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>

#ifndef BOOST_MATH_HAS_NVRTC
#include <boost/math/distributions/fwd.hpp>
#include <cmath>
#endif

namespace boost { namespace math {
template <class RealType, class Policy>
class saspoint5_distribution;

namespace detail {

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 0.125) {
        // Rational Approximation
        // Maximum Relative Error: 7.8747e-17
        BOOST_MATH_STATIC const RealType P[13] = {
            static_cast<RealType>(6.36619772367581343076e-1),
            static_cast<RealType>(2.17275699713513462507e2),
            static_cast<RealType>(3.49063163361344578910e4),
            static_cast<RealType>(3.40332906932698464252e6),
            static_cast<RealType>(2.19485577044357440949e8),
            static_cast<RealType>(9.66086435948730562464e9),
            static_cast<RealType>(2.90571833690383003932e11),
            static_cast<RealType>(5.83089315593106044683e12),
            static_cast<RealType>(7.37911022713775715766e13),
            static_cast<RealType>(5.26757196603002476852e14),
            static_cast<RealType>(1.75780353683063527570e15),
            static_cast<RealType>(1.85883041942144306222e15),
            static_cast<RealType>(4.19828222275972713819e14),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.41295871011779138155e2),
            static_cast<RealType>(5.48907134827349102297e4),
            static_cast<RealType>(5.36641455324410261980e6),
            static_cast<RealType>(3.48045461004960397915e8),
            static_cast<RealType>(1.54920747349701741537e10),
            static_cast<RealType>(4.76490595358644532404e11),
            static_cast<RealType>(1.00104823128402735005e13),
            static_cast<RealType>(1.39703522470411802507e14),
            static_cast<RealType>(1.23724881334160220266e15),
            static_cast<RealType>(6.47437580921138359461e15),
            static_cast<RealType>(1.77627318260037604066e16),
            static_cast<RealType>(2.04792815832538146160e16),
            static_cast<RealType>(7.45102534638640681964e15),
            static_cast<RealType>(3.68496090049571174527e14),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 0.25) {
        RealType t = x - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Relative Error: 2.1471e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(4.35668401768623200524e-1),
            static_cast<RealType>(7.12477357389655327116e0),
            static_cast<RealType>(4.02466317948738993787e1),
            static_cast<RealType>(9.04888497628205955839e1),
            static_cast<RealType>(7.56175387288619211460e1),
            static_cast<RealType>(1.26950253999694502457e1),
            static_cast<RealType>(-6.59304802132933325219e-1),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.98623818041545101115e1),
            static_cast<RealType>(1.52856383017632616759e2),
            static_cast<RealType>(5.70706902111659740041e2),
            static_cast<RealType>(1.06454927680197927878e3),
            static_cast<RealType>(9.13160352749764887791e2),
            static_cast<RealType>(2.58872466837209126618e2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 0.5) {
        RealType t = x - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Relative Error: 5.3265e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.95645445681747568732e-1),
            static_cast<RealType>(2.23779537590791610124e0),
            static_cast<RealType>(5.01302198171248036052e0),
            static_cast<RealType>(2.76363131116340641935e0),
            static_cast<RealType>(1.18134858311074670327e-1),
            static_cast<RealType>(2.00287083462139382715e-2),
            static_cast<RealType>(-7.53979800555375661516e-3),
            static_cast<RealType>(1.37294648777729527395e-3),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.02879626214781666701e1),
            static_cast<RealType>(3.85125274509784615691e1),
            static_cast<RealType>(6.18474367367800231625e1),
            static_cast<RealType>(3.77100050087302476029e1),
            static_cast<RealType>(5.41866360740066443656e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 1) {
        RealType t = x - static_cast <RealType>(0.5);

        // Rational Approximation
        // Maximum Relative Error: 2.7947e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.70762401725206223811e-1),
            static_cast<RealType>(8.43343631021918972436e-1),
            static_cast<RealType>(1.39703819152564365627e0),
            static_cast<RealType>(8.75843324574692085009e-1),
            static_cast<RealType>(1.86199552443747562584e-1),
            static_cast<RealType>(7.35858280181579907616e-3),
            static_cast<RealType>(-1.03693607694266081126e-4),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.73363440952557318819e0),
            static_cast<RealType>(1.74288966619209299976e1),
            static_cast<RealType>(2.15943268035083671893e1),
            static_cast<RealType>(1.29818726981381859879e1),
            static_cast<RealType>(3.40707211426946022041e0),
            static_cast<RealType>(2.80229012541729457678e-1),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 1.7051e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(8.61071469126041183247e-2),
            static_cast<RealType>(1.69689585946245345838e-1),
            static_cast<RealType>(1.09494833291892212033e-1),
            static_cast<RealType>(2.76619622453130604637e-2),
            static_cast<RealType>(2.44972748006913061509e-3),
            static_cast<RealType>(4.09853605772288438003e-5),
            static_cast<RealType>(-2.63561415158954865283e-7),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.04082856018856244947e0),
            static_cast<RealType>(3.52558663323956252986e0),
            static_cast<RealType>(1.94795523079701426332e0),
            static_cast<RealType>(5.23956733400745421623e-1),
            static_cast<RealType>(6.19453597593998871667e-2),
            static_cast<RealType>(2.31061984192347753499e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 2.9247e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(3.91428580496513429479e-2),
            static_cast<RealType>(4.07162484034780126757e-2),
            static_cast<RealType>(1.43342733342753081931e-2),
            static_cast<RealType>(2.01622178115394696215e-3),
            static_cast<RealType>(1.00648013467757737201e-4),
            static_cast<RealType>(9.51545046750892356441e-7),
            static_cast<RealType>(-3.56598940936439037087e-9),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.63904431617187026619e0),
            static_cast<RealType>(1.03812003196677309121e0),
            static_cast<RealType>(3.18144310790210668797e-1),
            static_cast<RealType>(4.81930155615666517263e-2),
            static_cast<RealType>(3.25435391589941361778e-3),
            static_cast<RealType>(7.01626957128181647457e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 2.6547e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.65057384221262866484e-2),
            static_cast<RealType>(8.05429762031495873704e-3),
            static_cast<RealType>(1.35249234647852784985e-3),
            static_cast<RealType>(9.18685252682786794440e-5),
            static_cast<RealType>(2.23447790937806602674e-6),
            static_cast<RealType>(1.03176916111395079569e-8),
            static_cast<RealType>(-1.94913182592441292094e-11),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.10113554189626079232e-1),
            static_cast<RealType>(2.54175325409968367580e-1),
            static_cast<RealType>(3.87119072807894983910e-2),
            static_cast<RealType>(2.92520770162792443587e-3),
            static_cast<RealType>(9.89094130526684467420e-5),
            static_cast<RealType>(1.07148513311070719488e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 2.5484e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(6.60044810497290557553e-3),
            static_cast<RealType>(1.59342644994950292031e-3),
            static_cast<RealType>(1.32429706922966110874e-4),
            static_cast<RealType>(4.45378136978435909660e-6),
            static_cast<RealType>(5.36409958111394628239e-8),
            static_cast<RealType>(1.22293787679910067873e-10),
            static_cast<RealType>(-1.16300443044165216564e-13),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.10446485803039594111e-1),
            static_cast<RealType>(6.51887342399859289520e-2),
            static_cast<RealType>(5.02151225308643905366e-3),
            static_cast<RealType>(1.91741179639551137839e-4),
            static_cast<RealType>(3.27316600311598190022e-6),
            static_cast<RealType>(1.78840301213102212857e-8),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 2.9866e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.54339461777955741686e-3),
            static_cast<RealType>(3.10069525357852579756e-4),
            static_cast<RealType>(1.30082682796085732756e-5),
            static_cast<RealType>(2.20715868479255585050e-7),
            static_cast<RealType>(1.33996659756026452288e-9),
            static_cast<RealType>(1.53505360463827994365e-12),
            static_cast<RealType>(-7.42649416356965421308e-16),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.09203384450859785642e-1),
            static_cast<RealType>(1.69422626897631306130e-2),
            static_cast<RealType>(6.65649059670689720386e-4),
            static_cast<RealType>(1.29654785666009849481e-5),
            static_cast<RealType>(1.12886139474560969619e-7),
            static_cast<RealType>(3.14420104899170413840e-10),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 3.3581e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(9.55085695067883584460e-4),
            static_cast<RealType>(5.86125496733202756668e-5),
            static_cast<RealType>(1.23753971325810931282e-6),
            static_cast<RealType>(1.05643819745933041408e-8),
            static_cast<RealType>(3.22502949410095015524e-11),
            static_cast<RealType>(1.85366144680157942079e-14),
            static_cast<RealType>(-4.53975807317403152058e-18),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.05980850386474826374e-1),
            static_cast<RealType>(4.34966042652000070674e-3),
            static_cast<RealType>(8.66341538387446465700e-5),
            static_cast<RealType>(8.55608082202236124363e-7),
            static_cast<RealType>(3.77719968378509293354e-9),
            static_cast<RealType>(5.33287361559571716670e-12),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x);

        // Rational Approximation
        // Maximum Relative Error: 4.7450e-19
        BOOST_MATH_STATIC const RealType P[5] = {
            static_cast<RealType>(1.99471140200716338970e-1),
            static_cast<RealType>(-1.93310094131437487158e-2),
            static_cast<RealType>(-8.44282614309073196195e-3),
            static_cast<RealType>(3.47296024282356038069e-3),
            static_cast<RealType>(-4.05398011689821941383e-4),
        };
        BOOST_MATH_STATIC const RealType Q[5] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.00973251258577238892e-1),
            static_cast<RealType>(2.66969681258835723157e-1),
            static_cast<RealType>(5.51785147503612200456e-2),
            static_cast<RealType>(6.50130030979966274341e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t / x;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 0.0625) {
        // Rational Approximation
        // Maximum Relative Error: 8.8841e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[27] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36619772367581343075535053490057448138e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.57459506929453385798277946154823008327e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46717322844023441698710451505816706570e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71501459971530549476153273173061194095e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.76700973495278431084530045707075552432e10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01328150775099946510145440412520620021e13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70028222513668830210058353057559790101e15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29641781943744384078006991488193839955e17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52611994112742436432957758588495082163e19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27833177267552931459542318826727288124e21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68946162731840551853993619351896931533e23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02965010233956763504899745874128908220e25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.14128569264874914146628076133997950655e26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09103580386900060922163883603492216942e28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.86778299087452621293332172137014749128e29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.80029712249744334924217328667885673985e31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70890080432228368476255091774238573277e32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.88600513999992354909078399482884993261e33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.01189178534848836605739139176681647755e34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.06531475170803043941021113424602440078e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.64956999370443524098457423629252855270e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44276098283517934229787916584447559248e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.45856704224433991524661028965741649584e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47263237190968408624388275549716907309e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66186300951901408251743228798832386260e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.48064966533519934186356663849904556319e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64877082086372991309408001661535573441e35),
        };
        BOOST_MATH_STATIC const RealType Q[28] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18981461118065892086304195732751798634e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.01761929839041982958990681130944341399e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69465252239913021973760046507387620537e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.49221044103838155300076098325950584061e10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59327386289821190042576978177896481082e13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67528179803224728786405503232064643870e15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61672367849271591791062829736720884633e17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98395893917909208201801908435620016552e19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60025693881358827551113845076726845495e21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67730578745705562356709169493821118109e23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63843883526710042156562706339553092312e25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.23075214698024188140971761421762265880e26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.37775321923937393366376907114580842429e28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12444724625354796650300159037364355605e30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00860835602766063447009568106012449767e31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.39614080159468893509273006948526469708e32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06547095715472468415058181351212520255e34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36755997709303811764051969789337337957e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32519530489892818585066019217287415587e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.45230390606834183602522256278256501404e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.80344475131699029428900627020022801971e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66469314795307459840482483320814279444e38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70209065673736156218117594311801487932e38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.84490531246108754748100009460860427732e38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30215083398643966091721732133851539475e38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.58032845332990262754766784625271262271e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.50461648438613634025964361513066059697e36),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 0.125) {
        RealType t = x - static_cast <RealType>(0.0625);

        // Rational Approximation
        // Maximum Relative Error: 3.4585e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.46416716200748206779925127900698754119e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.41771273526123373239570033672829787791e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56142610225585235535211648703534340871e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15655694129872563686497490176725921724e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.00791883661952751945853742455643714995e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30252591667828615354689186280704562254e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76168115448224677276551213052798322583e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.88534624532179841393387625270218172719e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14740447137831585842166880265350244623e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14904082614021239315925958812100948136e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.76866867279164114004579652405104553404e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53475339598769347326916978463911377965e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.88896160275915786487519266368539625326e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05543800791717482823610940401201712196e4),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.44407579416524903840331499438398472639e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15911780811299460009161345260146251462e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.88457596285725454686358792906273558406e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.66501639812506059997744549411633476528e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.12674134216028769532305433586266118000e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.87676063477990584593444083577765264392e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.56084282739608760299329382263598821653e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.34250986378665047914811630036201995871e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31288233106689286803200674021353188597e9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33621494302241474082474689597125896975e9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.63379428046258653791600947328520263412e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14558538557562267533922961110917101850e8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 0.25) {
        RealType t = x - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Relative Error: 6.9278e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35668401768623200524372663239480799018e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30066509937988171489091367354416214000e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.05924026744937322690717755156090122074e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.12998524955326375684693500551926325112e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.52237930808361186011042950178715609183e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.10734809597587633852077152938985998879e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.20796157836149826988172603622242119074e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12398478061053302537736799402801934778e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17841330491647012385157454335820786724e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.46281413765362795389526259057436151953e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52220357379402116641048490644093497829e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.51130316105543847380510577656570543736e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.32201781975497810173532067354797097401e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.96874547436310030183519174847668703774e0),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.63164311578114868477819520857286165076e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34964379844144961683927306966955217328e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82966031793809959278519002412667883288e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56215285850856046267451500310816276675e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.81046679663412610005501878092824281161e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.33868038251479411246071640628518434659e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.46262495881941625571640264458627940579e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40052628730443097561652737049917920495e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44394803828297754346261138417756941544e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.56647617803506258343236509255155360957e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53513095899009948733175317927025056561e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69130675750530663088963759279778748696e5),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 0.5) {
        RealType t = x - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Relative Error: 6.9378e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95645445681747568731488283573032414811e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.83246437763964151893665752064650172391e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.85333417559435252576820440080930004674e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.90974714199542064991001365628659054084e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39707205668285805800884524044738261436e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.24814598419826565698241508792385416075e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.95012897118808793886195172068123345314e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.87265743900139300849404272909665705025e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98795164648056126707212245325405968413e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07012128790318535418330629467906917213e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99797198893523173981812955075412130913e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55029227544167913873724286459253168886e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54064889901609722583601330171719819660e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.72254289950537680833853394958874977464e-3),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.58291085070053442257438623486099473087e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95618461039379226195473938654286975682e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97427161745150579714266897556974326502e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.52730436681412535198281529590508861106e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49521185356761585062135933350225236726e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.03881178612341724262911142022761966061e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.02360046338629039644581819847209730553e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.65580339066083507998465454599272345735e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15462499626138125314518636645472893045e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61951767959774678843021179589300545717e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60745557054877240279811529503888551492e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91061555870569579915258835459255406575e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43045229010040855016672246098687100063e1),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 1) {
        RealType t = x - static_cast <RealType>(0.5);

        // Rational Approximation
        // Maximum Relative Error: 6.4363e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70762401725206223811383500786268939645e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19353011456197635663058525904929358535e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22974648900600015961253465796487372402e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91951696059042324975935209295355569292e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.79039444119906169910281912009369164227e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00089963120992100860902142265631127046e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.37108883429306700857182028809960789020e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.49586566873564432788366931251358248417e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49790521605774884174840168128255220471e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.90660338063979435668763608259382712726e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93409982383888149064797608605579930804e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22802459215932860445033185874876812040e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07739227340181463034286653569468171767e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.02669738424010290973023004028523684766e-7),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46404480283267324138113869370306506431e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.54550184643308468933661600211579108422e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.63602410602063476726031476852965502123e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.94463479638213888403144706176973026333e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48607087483870766806529883069123352339e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69715692924508994524755312953665710218e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33237849965272853370191827043868842100e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.08460086451666825383009487734769646087e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.47365552394788536087148438788608689300e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.38010282703940184371247559455167674975e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.67219842525655806370702248122668214685e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.01852843874982199859775136086676841910e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.14767043526088185802569803397824432028e-3),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 9.1244e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.61071469126041183247373313827161939454e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35837460186564880289965856498718321896e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.47783071967681246738651796742079530382e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16019502727107539284403003943433359877e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.80510046274709592896987229782879937271e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30456542768955299533391113704078540955e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36539167913428133313942008990965988621e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76450743657913389896743235938695682829e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42847090205575096649865021874905747106e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41380341540026027117735179862124402398e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.40549721587212773424211923602910622515e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09089653391032945883918434200567278139e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21403900721572475664926557233205232491e-10),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.13172035933794917563324458011617112124e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65687100738157412154132860910003018338e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59672433683883998168388916533196510994e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.61469557815097583209668778301921207455e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77070955301136405523492329700943077340e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.20825570431301943907348077675777546304e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.60136197167727810483751794121979805142e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.53723076053642006159503073104152703814e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.63397465217490984394478518334313362490e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.40577918603319523990542237990107206371e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.94376458316662573143947719026985667328e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.09333568224541559157192543410988474886e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.59947287428695057506683902409023760438e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 8.1110e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91428580496513429479068747515164587814e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.69015019070193436467106672180804948494e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03147451266231819912643754579290008651e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.18825170881552297150779588545792258740e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30548850262278582401286533053286406505e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.54315108501815531776138839512564427279e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.66434584176931077662201101557716482514e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.66158632576958238392567355014249971287e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.31365802206301246598393821671437863818e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85378389166807263837732376845556856416e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20375363151456683883984823721339648679e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06637401794693307359898089790558771957e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08663671047376684678494625068451888284e-14),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.07443397096591141329212291707948432414e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.16665031056584124503224711639009530348e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27666060511630720485121299731204403783e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65646979169107732387032821262953301311e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.63594064986880863092994744424349361396e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31360114173642293100378020953197965181e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09489929949457075237756409511944811481e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24574519309785870806550506199124944514e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56048486483867679310086683710523566607e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.60417286783794818094722636906776809193e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53154117367296710469692755461431646999e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60041713691072903334637560080298818163e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.77381528950794767694352468734042252745e-12),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 2.5228e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65057384221262866484014802392420311075e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92801687242885330588201777283015178448e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.65508815862861196424333614846876229064e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71545573465295958468808641544341412235e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72077718130407940498710469661947719216e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.26299620525538984108147098966692839348e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77971404992990847565880351976461271350e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71176235845517643695464740679643640241e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.64603919225244695533557520384631958897e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.85274347406803894317891882905083368489e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48564096627181435612831469651920186491e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.90886715044580341917806394089282500340e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.39396206221935864416563232680283312796e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.37760675743046300528308203869876086823e-22),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49023284463742780238035958819642738891e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.76284367953836866133894756472541395734e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69932155343422362573146811195224195135e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.97593541520549770519034085640975455763e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45862809001322359249894968573830094537e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61348135835522976885804369721316193713e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21069949470458047530981551232427019037e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.03132437580490629136144285669590192597e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91030348024641585284338958059030520141e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.56320479309161046934628280237629402373e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.39524198476052364627683067034422502163e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18666081063885228839052386515073873844e-13),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 9.6732e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.60044810497290557552736366450372523266e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27034183360438185616541260923634443241e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19813403884333707962156711479716066536e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91346554854771687970018076643044998737e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91975837766081548424458764226669789039e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26031304514411902758114277797443618334e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.47127194811140370123712253347211626753e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55248861254135821097921903190564312000e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78340847719683652633864722047250151066e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95888612422041337572422846394029849086e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66363005792960308636467394552324255493e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.93244800648299424751906591077496534948e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95046217952146113063614290717113024410e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.46784746963816915795587433372284530785e-25),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.16012189991825507132967712656930682478e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95202772611563835130347051925062280272e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.23801477561401113332870463345197159418e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.54022665579711946784722766000062263305e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.94266182294627770206082679848878391116e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.11782839184878848480753630961211685630e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.28827067686094594197542725283923947812e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.00220719177374237332018587370837457299e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42250513143925626748132661121749401409e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82007216963767723991309138907689681422e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34214834652884406013489167210936679359e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.85519293212465087373898447546710143008e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.96728437809303144188312623363453475831e-19),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 1.0113e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54339461777955741686401041938275102207e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.17747085249877439037826121862689145081e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14104576580586095462211756659036062930e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.06903778663262313120049231822412184382e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53115958954246158081703822428768781010e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48225007017630665357941682179157662142e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20810829523286181556951002345409843125e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.54070972719909957155251432996372246019e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06258623970363729581390609798632080752e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15603641527498625694677136504611545743e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.05376970060354261667000502105893106009e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.14727542705613448694396750352455931731e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76883960167449461476228984331517762578e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.03558202009465610972808653993060437679e-29),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08809672969012756295937194823378109391e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.41148083436617376855422685448827300528e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.85101541143091590863368934606849033688e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.38984899982960112626157576750593711628e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51437845497783812562009857096371643785e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.12891276596072815764119699444334380521e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82412500887161687329929693518498698716e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80215715026891688444965605768621763721e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.85838684678780184082810752634454259831e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83675729736846176693608812315852523556e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.80347165008408134158968403924819637224e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23639219622240634094606955067799349447e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.63446235885036169537726818244420509024e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 9.7056e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.55085695067883584460317653567009454037e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52919532248638251721278667010429548877e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06266842295477991789450356745903177571e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.20671609948319334255323512011575892813e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04692714549374449244320605137676408001e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70605481454469287545965803970738264158e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83960996572005209177458712170004097587e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29732261733491885750067029092181853751e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.78385693918239619309147428897790440735e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52969197316398995616879018998891661712e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14063120299947677255281707434419044806e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25957675329657493245893497219459256248e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55238112862817593053765898004447484717e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.93970406521541790658675747195982964585e-34),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04722757068068234153968603374387493579e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85854131835804458353300285777969427206e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.85809281481040288085436275150792074968e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.38860750164285700051427698379841626305e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.91463283601681120487987016215594255423e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28104952818420195583669572450494959042e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43912720109615655035554724090181888734e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10668954229813492117417896681856998595e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.65093571330749369067212003571435698558e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81758227619561958470583781325371429458e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36970757752002915423191164330598255294e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.06487673393164724939989217811068656932e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.47121057452822097779067717258050172115e-27),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x);

        // Rational Approximation
        // Maximum Relative Error: 7.1032e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[8] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99471140200716338969973029967190934238e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.82846732476244747063962056024672844211e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.69724475658159099827638225237895868258e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21259630917863228526439367416146293173e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.13469812721679130825429547254346177005e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.73237434182338329541631611908947123606e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72986150007117100707304201395140411630e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.53567129749337040254350979652515879881e-7),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.89815449697874475254942178935516387239e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.21223228867921988134838870379132038419e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79514417558927397512722128659468888701e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.43331254539687594239741585764730095049e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.99078779616201786316256750758748178864e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04590833634768023225748107112347131311e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.17497990182339853998751740288392648984e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53420609011698705803549938558385779137e-6),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t / x;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53> &tag) {
    BOOST_MATH_STD_USING // for ADL of std functions

    return saspoint5_pdf_plus_imp_prec<RealType>(abs(x), tag);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>& tag) {
    BOOST_MATH_STD_USING // for ADL of std functions

    return saspoint5_pdf_plus_imp_prec<RealType>(abs(x), tag);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_pdf_imp(const saspoint5_distribution<RealType, Policy>& dist, const RealType& x) {
    //
    // This calculates the pdf of the Saspoint5 distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::pdf(saspoint5<%1%>&, %1%)";
    RealType result = 0;
    RealType location = dist.location();
    RealType scale = dist.scale();

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

    static_assert(tag_type::value, "The SaS point5 distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale;

    result = saspoint5_pdf_imp_prec(u, tag_type()) / scale;

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 0.5) {
        // Rational Approximation
        // Maximum Relative Error: 2.6225e-17
        BOOST_MATH_STATIC const RealType P[16] = {
            static_cast<RealType>(5.0e-1),
            static_cast<RealType>(1.11530082549581486148e2),
            static_cast<RealType>(1.18564167533523512811e4),
            static_cast<RealType>(7.51503793077701705413e5),
            static_cast<RealType>(3.05648233678438482191e7),
            static_cast<RealType>(8.12176734530090957088e8),
            static_cast<RealType>(1.39533182836234507573e10),
            static_cast<RealType>(1.50394359286077974212e11),
            static_cast<RealType>(9.79057903542935575811e11),
            static_cast<RealType>(3.73800992855150140014e12),
            static_cast<RealType>(8.12697090329432868343e12),
            static_cast<RealType>(9.63154058643818290870e12),
            static_cast<RealType>(5.77714904017642642181e12),
            static_cast<RealType>(1.53321958252091815685e12),
            static_cast<RealType>(1.36220966258718212359e11),
            static_cast<RealType>(1.70766655065405022702e9),
        };
        BOOST_MATH_STATIC const RealType Q[16] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.24333404643898143947e2),
            static_cast<RealType>(2.39984636687021023600e4),
            static_cast<RealType>(1.53353791432086858132e6),
            static_cast<RealType>(6.30764952479861776476e7),
            static_cast<RealType>(1.70405769169309597488e9),
            static_cast<RealType>(3.00381227010195289341e10),
            static_cast<RealType>(3.37519046677507392667e11),
            static_cast<RealType>(2.35001610518109063314e12),
            static_cast<RealType>(9.90961948200767679416e12),
            static_cast<RealType>(2.47066673978544828258e13),
            static_cast<RealType>(3.51442593932882610556e13),
            static_cast<RealType>(2.68891431106117733130e13),
            static_cast<RealType>(9.99723484253582494535e12),
            static_cast<RealType>(1.49190229409236772612e12),
            static_cast<RealType>(5.68752980146893975323e10),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 1) {
        RealType t = x - static_cast <RealType>(0.5);

        // Rational Approximation
        // Maximum Relative Error: 9.2135e-19
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(3.31309550000758082456e-1),
            static_cast<RealType>(1.63012162307622129396e0),
            static_cast<RealType>(2.97763161467248770571e0),
            static_cast<RealType>(2.49277948739575294031e0),
            static_cast<RealType>(9.49619262302649586821e-1),
            static_cast<RealType>(1.38360148984087584165e-1),
            static_cast<RealType>(4.00812864075652334798e-3),
            static_cast<RealType>(-4.82051978765960490940e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.43565383128046471592e0),
            static_cast<RealType>(1.13265160672130133152e1),
            static_cast<RealType>(1.13352316246726435292e1),
            static_cast<RealType>(5.56671465170409694873e0),
            static_cast<RealType>(1.21011708389501479550e0),
            static_cast<RealType>(8.34618282872428849500e-2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 6.4688e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.71280312689343248819e-1),
            static_cast<RealType>(7.44610837974139249205e-1),
            static_cast<RealType>(7.17844128359406982825e-1),
            static_cast<RealType>(2.98789060945288850507e-1),
            static_cast<RealType>(5.22747411439102272576e-2),
            static_cast<RealType>(3.06447984437786430265e-3),
            static_cast<RealType>(2.60407071021044908690e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.06221257507188300824e0),
            static_cast<RealType>(3.44827372231472308047e0),
            static_cast<RealType>(1.78166113338930668519e0),
            static_cast<RealType>(4.25580478492907232687e-1),
            static_cast<RealType>(4.09983847731128510426e-2),
            static_cast<RealType>(1.04343172183467651240e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 8.2289e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.13928162275383716645e-1),
            static_cast<RealType>(2.35139109235828185307e-1),
            static_cast<RealType>(9.35967515134932733243e-2),
            static_cast<RealType>(1.64310489592753858417e-2),
            static_cast<RealType>(1.23186728989215889119e-3),
            static_cast<RealType>(3.13500969261032539402e-5),
            static_cast<RealType>(1.17021346758965979212e-7),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.28212183177829510267e0),
            static_cast<RealType>(6.17321009406850420793e-1),
            static_cast<RealType>(1.38400318019319970893e-1),
            static_cast<RealType>(1.44994794535896837497e-2),
            static_cast<RealType>(6.17774446282546623636e-4),
            static_cast<RealType>(7.00521050169239269819e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 3.7284e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.63772802979087193656e-1),
            static_cast<RealType>(9.69009603942214234119e-2),
            static_cast<RealType>(2.08261725719828138744e-2),
            static_cast<RealType>(1.97965182693146960970e-3),
            static_cast<RealType>(8.05499273532204276894e-5),
            static_cast<RealType>(1.11401971145777879684e-6),
            static_cast<RealType>(2.25932082770588727842e-9),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.92463563872865541733e-1),
            static_cast<RealType>(1.80720987166755982366e-1),
            static_cast<RealType>(2.20416647324531054557e-2),
            static_cast<RealType>(1.26052070140663063778e-3),
            static_cast<RealType>(2.93967534265875431639e-5),
            static_cast<RealType>(1.82706995042259549615e-7),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 4.9609e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.22610122564874280532e-1),
            static_cast<RealType>(3.70273222121572231593e-2),
            static_cast<RealType>(4.06083618461789591121e-3),
            static_cast<RealType>(1.96898134215932126299e-4),
            static_cast<RealType>(4.08421066512186972853e-6),
            static_cast<RealType>(2.87707419853226244584e-8),
            static_cast<RealType>(2.96850126180387702894e-11),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.55825191301363023576e-1),
            static_cast<RealType>(4.77251766176046719729e-2),
            static_cast<RealType>(2.99136605131226103925e-3),
            static_cast<RealType>(8.78895785432321899939e-5),
            static_cast<RealType>(1.05235770624006494709e-6),
            static_cast<RealType>(3.35423877769913468556e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 5.6559e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(9.03056141356415077080e-2),
            static_cast<RealType>(1.37568904417652631821e-2),
            static_cast<RealType>(7.60947271383247418831e-4),
            static_cast<RealType>(1.86048302967560067128e-5),
            static_cast<RealType>(1.94537860496575427218e-7),
            static_cast<RealType>(6.90524093915996283104e-10),
            static_cast<RealType>(3.58808434477817122371e-13),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.80501347735272292079e-1),
            static_cast<RealType>(1.22807958286146936376e-2),
            static_cast<RealType>(3.90421541115275676253e-4),
            static_cast<RealType>(5.81669449234915057779e-6),
            static_cast<RealType>(3.53005415676201803667e-8),
            static_cast<RealType>(5.69883025435873921433e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 6.0653e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(6.57333571766941474226e-2),
            static_cast<RealType>(5.02795551798163084224e-3),
            static_cast<RealType>(1.39633616037997111325e-4),
            static_cast<RealType>(1.71386564634533872559e-6),
            static_cast<RealType>(8.99508156357247137439e-9),
            static_cast<RealType>(1.60229460572297160486e-11),
            static_cast<RealType>(4.17711709622960498456e-15),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.10198637347368265508e-2),
            static_cast<RealType>(3.12263472357578263712e-3),
            static_cast<RealType>(5.00524795130325614005e-5),
            static_cast<RealType>(3.75913188747149725195e-7),
            static_cast<RealType>(1.14970132098893394023e-9),
            static_cast<RealType>(9.34957119271300093120e-13),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x);

        // Rational Approximation
        // Maximum Relative Error: 2.0104e-20
        BOOST_MATH_STATIC const RealType P[5] = {
            static_cast<RealType>(3.98942280401432677940e-1),
            static_cast<RealType>(8.12222388783621449146e-2),
            static_cast<RealType>(1.68515703707271703934e-2),
            static_cast<RealType>(2.19801627205374824460e-3),
            static_cast<RealType>(-5.63321705854968264807e-5),
        };
        BOOST_MATH_STATIC const RealType Q[5] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.02536240902768558315e-1),
            static_cast<RealType>(1.99284471400121092380e-1),
            static_cast<RealType>(3.48012577961755452113e-2),
            static_cast<RealType>(3.38545004473058881799e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 0.125) {
        // Rational Approximation
        // Maximum Relative Error: 6.9340e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[30] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.0e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.25520067710293108163697513129883130648e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70866020657515874782126804139443323023e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.00865235319309486225795793030882782077e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15226363537737769449645357346965170790e10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.90371247243851280277289046301838071764e12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.55124590509169425751300134399513503679e14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.31282020412787511681760982839078664474e16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.81134278666896523873256421982740565131e18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.36154530125229747305141034242362609073e20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67793867640429875837167908549938345465e22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34584264816825205490037614178084070903e24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.52622279567059369718208827282730379468e25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.84678324511679577282571711018484545185e27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.99412564257799793932936828924325638617e28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08467105431111959283045453636520222779e30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87466808926544728702827204697734995611e31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55020252231174414164534905191762212055e32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69582736077420504345389671165954321163e33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.18203860972249826626461130638196586188e34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32955733788770318392204091471121129386e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.97972270315674052071792562126668438695e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93941537398987201071027348577636994465e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40818708062034138095495206258366082481e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.76833406751769751643745383413977973530e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.67873467711368838525239991688791162617e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.94179310584115437584091984619858795365e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24348215908456320362232906012152922949e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71625432346533320597285660433110657670e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.54662474187354179772157464533408058525e33),
        };
        BOOST_MATH_STATIC const RealType Q[31] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05231337496532137901354609636674085703e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.43071888317491317900094470796567113997e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.80864482202910830302921131771345102044e8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.32755297215862998181755216820621285536e10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.86251123527611073428156549377791985741e12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12025543961949466786297141758805461421e15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27681657695574252637426145112570596483e17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17850553865715973904162289375819555884e19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.87285897504702686250962844939736867339e20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.46852796231948446334549476317560711795e22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76101689878844725930808096548998198853e24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14017776727845251567032313915953239178e26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83738954971390158348334918235614003163e27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04701345216121451992682705965658316871e29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29994190638467725374533751141434904865e30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.03334024242845994501493644478442360593e31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.59094378123268840693978620156028975277e32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.05630254163426327113368743426054256780e33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.06008195534030444387061989883493342898e34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.21262490304347036689874956206774563906e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52353024633841796119920505314785365242e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28839143293381125956284415313626962263e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31074057704096457802547386358094338369e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24762412200364040971704861346921094354e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55663903116458425420509083471048286114e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81839283802391753865642022579846918253e37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.40026559327708207943879092058654410696e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48767474646810049293505781106444169229e36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10591353097667671736865938428051885499e35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26980708896893794012677171239610721832e33),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 0.25) {
        RealType t = x - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Relative Error: 9.6106e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.31887921568009055676985827521151969069e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62791448964529380666250180886203090183e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.18238045199893937316918299064825702894e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.77274519306540522227493503092956314136e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.89424638466340765479970877448972418958e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84027004420207996285174223581748706097e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.84633134142285937075423713704784530853e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76780579189423063605715733542379494552e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19812409802969581112716039533798357401e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60039008588877024309600768114757310858e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.10268260529501421009222937882726290612e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.72169594688819848498039471657587836720e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27181379647139697258984772894869505788e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.73617450590346508706222885401965820190e1),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.18558935411552146390814444666395959919e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49210559503096368944407109881023223654e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93959323596111340518285858313038058302e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53590607436758691037825792660167970938e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.82700985983018132572589829602100319330e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62137033935442506086127262036686905276e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.76014299715348555304267927238963139228e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.12336796972134088340556958396544477713e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.01952132024838508233050167059872220508e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41846547214877387780832317250797043384e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.02083431572388097955901208994308271581e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.30401057171447074343957754855656724141e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 0.5) {
        RealType t = x - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Relative Error: 3.1519e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.87119665000174806422420129219814467874e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.60769554551148293079169764245570645155e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.49181979810834706538329284478129952168e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15722765491675871778645250624425739489e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.65973147084701923411221710174830072860e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93709338011482232037110656459951914303e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57393131299425403017769538642434714791e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24110491141294379107651487490031694257e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23670394514211681515965192338544032862e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06141024932329394052395469123628405389e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.08599362073145455095790192415468286304e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.22746783794652085925801188098270888502e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.19652234873414609727168969049557770989e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.73529976407853894192156335785920329181e-4),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04157569499592889296640733909653747983e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82248883130787159161541119440215325308e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.35191216924911901198168794737654512677e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05099314677808235578577204150229855903e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.61081069236463123032873733048661305746e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.65340645555368229718826047069323437201e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.44526681322128674428653420882660351679e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.03963804195353853550682049993122898950e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40399236835577953127465726826981753422e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.46764755170079991793106428011388637748e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.06384106042490712972156545051459068443e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27614406724572981099586665536543423891e0),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 1) {
        RealType t = x - static_cast <RealType>(0.5);

        // Rational Approximation
        // Maximum Relative Error: 7.1196e-37
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31309550000758082761278726632760756847e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.07242222531117199094690544171275415854e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.18286763141875580859241637334381199648e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72102024869298528501604761974348686708e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31748399999514540052066169132819656757e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72168003284748405703923567644025252608e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52648991506052496046447777354251378257e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.16777263528764704804758173026143295383e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66044453196259367950849328889468385159e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.35095952392355288307377427145581700484e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43011308494452327007589069222668324337e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16582092138863383294685790744721021189e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.02261914949200575965813000131964695720e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93943913630044161720796150617166047233e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.76395009419307902351328300308365369814e-8),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28073115520716780203055949058270715651e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.20245752585870752942356137496087189194e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34548337034735803039553186623067144497e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.90925817267776213429724248532378895039e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.92883822651628140083115301005227577059e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72868136219107985834601503784789993218e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.50498744791568911029110559017896701095e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05178276667813671578581259848923964311e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.16880263792490095344135867620645018480e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16199396397514668672304602774610890666e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25543193822942088303609988399416145281e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.62522294286034117189844614005500278984e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18342889744790118595835138444372660676e-3),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 9.5605e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71280312689343266367958859259591541365e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49628636612698702680819948707479820292e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61090930375686902075245639803646265081e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.01191924051756106307211298794294657688e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.42496510376427957390465373165464672088e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59577769624139820954046058289100998534e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02664809521258420718170586857797408674e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72258711278476951299824066502536249701e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72578941800687566921553416498339481887e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.12553368488232553360765667155702324159e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20749770911901442251726681861858323649e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00621121212654384864006297569770703900e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18976033102817074104109472578202752346e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22093548539863254922531707899658394458e-10),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.83305694892673455436552817409325835774e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49922543669955056754932640312490112609e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23488972536322019584648241457582608908e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14051527527038669918848981363974859889e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37891280136777182304388426277537358346e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.08058775103864815769223385606687612117e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83305488980337433132332401784292281716e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.71072208215804671719811563659227630554e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.86332040813989094594982937011005305263e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.87698178237970337664105782546771501188e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69113555019737313680732855691540088318e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31888539972217875242352157306613891243e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85633766164682554126992822326956560433e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 1.1494e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13928162275383718405630406427822960090e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.90742307267701162395764574873947997211e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.13821826367941514387521090205756466068e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.96186879146063565484800486550739025293e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19438785955706463753454881511977831603e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.44969124820016994689518539612465708536e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27841835070651018079759230944461773079e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69952429132675045239242077293594666305e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47919853099168659881487026035933933068e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.04644774117864306055402364094681541437e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52604718870921084048756263996119841957e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97610950633031564892821158058978809537e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35934159016861180185992558083703785765e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21044098237798939057079316997065892072e-14),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.94437702178976797218081686254875998984e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.82068837586514484653828718675654460991e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87606058269189306593797764456467061128e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39130528408903116343256483948950693356e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.52792074489091396425713962375223436022e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00892233011840867583848470677898363716e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53717105060592851173320646706141911461e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57293857675930200001382624769341451561e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04219251796696135508847408131139677925e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.20053006131133304932740325113068767057e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.26384707028090985155079342718673255493e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.19146442700994823924806249608315505708e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59879298772002950043508762057850408213e-12),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 1.9710e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63772802979087199762340235165979751298e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31941534705372320785274994658709390116e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43643960022585762678456016437621064500e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.11415302325466272779041471612529728187e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15767742459744253874067896740220951622e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74049309186016489825053763513176160256e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76349898574685150849080543168157785281e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19757202370729036627932327405149840205e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.32067727965321839898287320520750897894e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47636340015260789807543414080472136575e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36236727340568181129875213546468908164e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89379573960280486883733996547662506245e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.71832232038263988173042637335112603365e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72380245500539326441037770757072641975e-18),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51702400281458104713682413542736419584e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01375164846907815766683647295932603968e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.92796007869834847612192314006582598557e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.77580441164023725582659445614058463183e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63592331843149724480258804892989851727e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87334158717610115008450674967492650941e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46596056941432875244263245821845070102e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.69980051560936361597177347949112822752e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.61690034211585843423761830218320365457e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.40619773800285766355596852314940341504e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.10624533319804091814643828283820958419e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.10009654621246392691126133176423833259e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48070591106986983088640496621926852293e-16),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 5.2049e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22610122564874286614786819620499101143e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.30352481858382230273216195795534959290e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.45933050053542949214164590814846222512e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.18776888646200567321599584635465632591e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53421996228923143480455729204878676265e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.26075686557831306993734433164305349875e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58045762501721375879877727645933749122e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13469629033419341069106781092024950086e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09157226556088521407323375433512662525e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44961350323527660188267669752380722085e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11052101325523147964890915835024505324e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.26421404354976214191891992583151033361e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.17133505681224996657291059553060754343e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41010917905686427164414364663355769988e-22),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.31062773739451672808456319166347015167e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35386721434011881226168110614121649232e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.39357338312443465616015226804775178232e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.26630144036271792027494677957363535353e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.24340476859846183414651435036807677467e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33917136421389571662908749253850939876e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.80972141456523767244381195690041498939e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22653625120465488656616983786525028119e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.64072452032620505897896978124863889812e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61842001579321492488462230987972104386e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96631619425501661980194304605724632777e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.80392324086028812772385536034034039168e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.34254502871215949266781048808984963366e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 1.7434e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.03056141356415128156562790092782153630e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15331583242023443256381237551843296356e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.81847913073640285776566199343276995613e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.23578443960486030170636772457627141406e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36906354316016270165240908809929957836e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.80584421020238085239890207672296651219e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.20726437845755296397071540583729544203e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.71042563703818585243207722641746283288e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08307373360265947158569900625482137206e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80776566500233755365518221977875432763e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11405351639704510305055492207286172753e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21575609293568296049921888011966327905e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66081982641748223969990279975752576675e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96483118060215455299182487430511998831e-26),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77347004038951368607085827825968614455e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.27559305780716801070924630708599448466e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.05452382624230160738008550961679711827e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.75369667590360521677018734348769796476e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56551290985905942229892419848093494661e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46972102361871185271727958608184616388e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.72423917010499649257775199140781647069e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14337306905269302583746182007852069459e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.69949885309711859563395555285232232606e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.00318719634300754237920041312234711548e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41139664927184402637020651515172315287e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79013190225240505774959477465594797961e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.46536966503325413797061462062918707370e-24),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 2.0402e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.57333571766941514095434647381791040479e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15416685251021339933358981066948923001e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.86806753164417557035166075399588122481e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91972616817770660098405128729991574724e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10225768760715861978198010761036882002e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05986998674039047865566990469266534338e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59572646670205456333051888086612875871e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19347294198055585461131949159508730257e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21328285448498841418774425071549974153e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.19331596847283822557042655221763459728e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.36128017817576942059191451016251062072e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.35600223942735523925477855247725326228e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14658948592500290756690769268766876322e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86984591055448991335081550609451649866e-30),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90112824856612652807095815199496602262e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59291524937386142936420775839969648652e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.74245361925275011235694006013677228467e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41828589449615478387532599798645159282e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.08088176420557205743676774127863572768e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.30760429417424419297000535744450830697e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.18464910867914234357511605329900284981e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.74255540513281299503596269087176674333e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.02616028440371294233330747672966435921e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.26276597941744408946918920573146445795e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.47463109867603732992337779860914933775e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.77217411888267832243050973915295217582e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.67397425207383164084527830512920206074e-28),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x);

        // Rational Approximation
        // Maximum Relative Error: 9.2612e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[9] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98942280401432677939946059934381868476e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33908701314796522684603310107061150444e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.92120397142832495974006972404741124398e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.15463147603421962834297353867930971657e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44488751006069172847577645328482300099e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44057582804743599116332797864164802887e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.02968018188491417839349438941039867033e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.75092244933846337077999183310087492887e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35099582728548602389917143511323566818e-8),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.34601617336219074065534356705298927390e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82954035780824611941899463895040327299e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70929001162671283123255408612494541378e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05508596604210030533747793197422815105e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02913299057943756875992272236063124608e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.37824426836648736125759177846682556245e-5),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t;
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 53>& tag) {
    if (x >= 0) {
        return complement ? saspoint5_cdf_plus_imp_prec(x, tag) : 1 - saspoint5_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - saspoint5_cdf_plus_imp_prec(-x, tag) : saspoint5_cdf_plus_imp_prec(-x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 113>& tag) {
    if (x >= 0) {
        return complement ? saspoint5_cdf_plus_imp_prec(x, tag) : 1 - saspoint5_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - saspoint5_cdf_plus_imp_prec(-x, tag) : saspoint5_cdf_plus_imp_prec(-x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_cdf_imp(const saspoint5_distribution<RealType, Policy>& dist, const RealType& x, bool complement) {
    //
    // This calculates the cdf of the Saspoint5 distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::cdf(saspoint5<%1%>&, %1%)";
    RealType result = 0;
    RealType location = dist.location();
    RealType scale = dist.scale();

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

    static_assert(tag_type::value, "The SaS point5 distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale;

    result = saspoint5_cdf_imp_prec(u, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (ilogb(p) >= -2) {
        RealType u = -log2(ldexp(p, 1));

        if (u < 0.125) {
            // Rational Approximation
            // Maximum Relative Error: 4.2616e-17
            BOOST_MATH_STATIC const RealType P[13] = {
                static_cast<RealType>(1.36099130643975127045e-1),
                static_cast<RealType>(2.19634434498311523885e1),
                static_cast<RealType>(1.70276954848343179287e3),
                static_cast<RealType>(8.02187341786354339306e4),
                static_cast<RealType>(2.48750112198456813443e6),
                static_cast<RealType>(5.20617858300443231437e7),
                static_cast<RealType>(7.31202030685167303439e8),
                static_cast<RealType>(6.66061403138355591915e9),
                static_cast<RealType>(3.65687892725590813998e10),
                static_cast<RealType>(1.06061776220305595494e11),
                static_cast<RealType>(1.23930642673461465346e11),
                static_cast<RealType>(1.49986408149520127078e10),
                static_cast<RealType>(-6.17325587219357123900e8),
            };
            BOOST_MATH_STATIC const RealType Q[13] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(1.63111146753825227716e2),
                static_cast<RealType>(1.27864461509685444043e4),
                static_cast<RealType>(6.10371533241799228037e5),
                static_cast<RealType>(1.92422115963507708309e7),
                static_cast<RealType>(4.11544185502250709497e8),
                static_cast<RealType>(5.95343302992055062258e9),
                static_cast<RealType>(5.65615858889758369947e10),
                static_cast<RealType>(3.30833154992293143503e11),
                static_cast<RealType>(1.06032392136054207216e12),
                static_cast<RealType>(1.50071282012095447931e12),
                static_cast<RealType>(5.43552396263989180433e11),
                static_cast<RealType>(9.57434915768660935004e10),
            };

            result = u * tools::evaluate_polynomial(P, u) / (tools::evaluate_polynomial(Q, u) * (p * p));
        }
        else if (u < 0.25) {
            RealType t = u - static_cast <RealType>(0.125);

            // Rational Approximation
            // Maximum Relative Error: 2.3770e-19
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(1.46698650748920243698e-2),
                static_cast<RealType>(3.58380131788385557227e-1),
                static_cast<RealType>(3.39153750029553194566e0),
                static_cast<RealType>(1.55457424873957272207e1),
                static_cast<RealType>(3.44403897039657057261e1),
                static_cast<RealType>(3.01881531964962975320e1),
                static_cast<RealType>(2.77679052294606319767e0),
                static_cast<RealType>(-7.76665288232972435969e-2),
            };
            BOOST_MATH_STATIC const RealType Q[7] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(1.72584280323876188464e1),
                static_cast<RealType>(1.11983518800147654866e2),
                static_cast<RealType>(3.25969893054048132145e2),
                static_cast<RealType>(3.91978809680672051666e2),
                static_cast<RealType>(1.29874252720714897530e2),
                static_cast<RealType>(2.08740114519610102248e1),
            };

            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
        }
        else if (u < 0.5) {
            RealType t = u - static_cast <RealType>(0.25);

            // Rational Approximation
            // Maximum Relative Error: 9.2445e-18
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(2.69627866689346445458e-2),
                static_cast<RealType>(3.23091180507445216811e-1),
                static_cast<RealType>(1.42164019533549860681e0),
                static_cast<RealType>(2.74613170828120023406e0),
                static_cast<RealType>(2.07865023346180997996e0),
                static_cast<RealType>(2.53267176863740856907e-1),
                static_cast<RealType>(-2.55816250186301841152e-2),
                static_cast<RealType>(3.02683750470398342224e-3),
            };
            BOOST_MATH_STATIC const RealType Q[6] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(8.55049920135376003042e0),
                static_cast<RealType>(2.48726119139047911316e1),
                static_cast<RealType>(2.79519589592198994574e1),
                static_cast<RealType>(9.88212916161823866098e0),
                static_cast<RealType>(1.39749417956251951564e0),
            };

            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
        }
        else {
            RealType t = u - static_cast <RealType>(0.5);

            // Rational Approximation
            // Maximum Relative Error: 2.2918e-20
            BOOST_MATH_STATIC const RealType P[9] = {
                static_cast<RealType>(4.79518653373241051274e-2),
                static_cast<RealType>(3.81837125793765918564e-1),
                static_cast<RealType>(1.13370353708146321188e0),
                static_cast<RealType>(1.55218145762186846509e0),
                static_cast<RealType>(9.60938271141036509605e-1),
                static_cast<RealType>(2.11811755464425606950e-1),
                static_cast<RealType>(8.84533960603915742831e-3),
                static_cast<RealType>(1.73314614571009160225e-3),
                static_cast<RealType>(-3.63491208733876986098e-5),
            };
            BOOST_MATH_STATIC const RealType Q[8] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(6.36954463000253710936e0),
                static_cast<RealType>(1.40601897306833147611e1),
                static_cast<RealType>(1.33838075106916667084e1),
                static_cast<RealType>(5.60958095533108032859e0),
                static_cast<RealType>(1.11796035623375210182e0),
                static_cast<RealType>(1.12508482637488861060e-1),
                static_cast<RealType>(5.18503975949799718538e-3),
            };

            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
        }
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 2));

        // Rational Approximation
        // Maximum Relative Error: 4.2057e-18
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(8.02395484493329835881e-2),
            static_cast<RealType>(2.46132933068351274622e-1),
            static_cast<RealType>(2.81820176867119231101e-1),
            static_cast<RealType>(1.47754061028371025893e-1),
            static_cast<RealType>(3.54638964490281023406e-2),
            static_cast<RealType>(3.99998730093393774294e-3),
            static_cast<RealType>(3.81581928434827040262e-4),
            static_cast<RealType>(1.82520920154354221101e-5),
            static_cast<RealType>(-2.06151396745690348445e-7),
            static_cast<RealType>(6.77986548138011345849e-9),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.39244329037830026691e0),
            static_cast<RealType>(2.12683465416376620896e0),
            static_cast<RealType>(9.02612272334554457823e-1),
            static_cast<RealType>(2.06667959191488815314e-1),
            static_cast<RealType>(2.79328968525257867541e-2),
            static_cast<RealType>(2.28216286216537879937e-3),
            static_cast<RealType>(1.04195690531437767679e-4),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 3.3944e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(1.39293493266195561875e-1),
            static_cast<RealType>(1.26741380938661691592e-1),
            static_cast<RealType>(4.31117040307200265931e-2),
            static_cast<RealType>(7.50528269269498076949e-3),
            static_cast<RealType>(8.63100497178570310436e-4),
            static_cast<RealType>(6.75686286034521991703e-5),
            static_cast<RealType>(3.11102625473120771882e-6),
            static_cast<RealType>(9.63513655399980075083e-8),
            static_cast<RealType>(-6.40223609013005302318e-11),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.11234548272888947555e-1),
            static_cast<RealType>(2.63525516991753831892e-1),
            static_cast<RealType>(4.77118226533147280522e-2),
            static_cast<RealType>(5.46090741266888954909e-3),
            static_cast<RealType>(4.15325425646862026425e-4),
            static_cast<RealType>(2.02377681998442384863e-5),
            static_cast<RealType>(5.79823311154876056655e-7),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 4.1544e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(1.57911660613037760235e-1),
            static_cast<RealType>(5.59740955695099219682e-2),
            static_cast<RealType>(8.92895854008560399142e-3),
            static_cast<RealType>(8.88795299273855801726e-4),
            static_cast<RealType>(5.66358335596607738071e-5),
            static_cast<RealType>(2.46733195253941569922e-6),
            static_cast<RealType>(6.44829870181825872501e-8),
            static_cast<RealType>(7.62193242864380357931e-10),
            static_cast<RealType>(-7.82035413331699873450e-14),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.49007782566002620811e-1),
            static_cast<RealType>(5.65303702876260444572e-2),
            static_cast<RealType>(5.54316442661801299351e-3),
            static_cast<RealType>(3.58498995501703237922e-4),
            static_cast<RealType>(1.53872913968336341278e-5),
            static_cast<RealType>(4.08512152326482573624e-7),
            static_cast<RealType>(4.72959615756470826429e-9),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 8.5877e-18
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(1.59150086070234563099e-1),
            static_cast<RealType>(6.07144002506911115092e-2),
            static_cast<RealType>(1.10026443723891740392e-2),
            static_cast<RealType>(1.24892739209332398698e-3),
            static_cast<RealType>(9.82922518655171276487e-5),
            static_cast<RealType>(5.58366837526347222893e-6),
            static_cast<RealType>(2.29005408647580194007e-7),
            static_cast<RealType>(6.44325718317518336404e-9),
            static_cast<RealType>(1.05110361316230054467e-10),
            static_cast<RealType>(1.48083450629432857655e-18),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.81470315977341203351e-1),
            static_cast<RealType>(6.91330250512167919573e-2),
            static_cast<RealType>(7.84712209182587717077e-3),
            static_cast<RealType>(6.17595479676821181012e-4),
            static_cast<RealType>(3.50829361179041199953e-5),
            static_cast<RealType>(1.43889153071571504712e-6),
            static_cast<RealType>(4.04840254888235877998e-8),
            static_cast<RealType>(6.60429636407045050112e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 8.7254e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(1.59154943017783026201e-1),
            static_cast<RealType>(6.91506515614472069475e-2),
            static_cast<RealType>(1.44590186111155933843e-2),
            static_cast<RealType>(1.92616138327724025421e-3),
            static_cast<RealType>(1.79640147906775699469e-4),
            static_cast<RealType>(1.30852535070639833809e-5),
            static_cast<RealType>(5.55259657884038297268e-7),
            static_cast<RealType>(3.50107118687544980820e-8),
            static_cast<RealType>(-1.47102592933729597720e-22),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.34486357752330500669e-1),
            static_cast<RealType>(9.08486933075320995164e-2),
            static_cast<RealType>(1.21024289017243304241e-2),
            static_cast<RealType>(1.12871233794777525784e-3),
            static_cast<RealType>(8.22170725751776749123e-5),
            static_cast<RealType>(3.48879932410650101194e-6),
            static_cast<RealType>(2.19978790407451988423e-7),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else {
        result = 1 / (p * p * constants::two_pi<RealType>());
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (ilogb(p) >= -2) {
        RealType u = -log2(ldexp(p, 1));

        if (u < 0.125) {
            // Rational Approximation
            // Maximum Relative Error: 2.5675e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[31] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36099130643975133156293056139850872219e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.03940482189350763127508703926866548690e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00518276893354880480781640750482315271e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.55844903094077096941027360107304259099e6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04507684135310729583474324660276395831e9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28519957085041757616278379578781441623e11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26054173986187219679917530171252145632e13),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00693075272502479915569708465960917906e15),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.64153695410984136395853200311209462775e16),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.64993034609287363745840801813540992383e18),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68080300629977787949474098413155901197e20),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.50632142671665246974634799849090331338e21),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11943753054362349397013211631038480307e23),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.80601829873419334580289886671478701625e24),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33441650581633426542372642262736818512e26),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.56279427934163518272441555879970370340e27),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08899113985387092689705022477814364717e28),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.37750989391907347952902900750138805007e29),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.76961267256299304213687639380275530721e30),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.98417586455955659885944915688130612888e31),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40932923796679251232655132670811114351e32),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.80810239916688876216017180714744912573e33),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.23907429566810200929293428832485038147e33),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11441754640405256305951569489818422227e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30534222360394829628175800718529342304e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.73301799323855143458670230536670073483e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53142592196246595846485130434777396548e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81719621726393542967303806360105998384e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00188544550531824809437206713326495544e33),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62706943144847786115732327787879709587e32),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32129438774563059735783287456769609571e31),
            };
            BOOST_MATH_STATIC const RealType Q[32] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.65910866673514847742559406762379054364e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21954860438789969160116317316418373146e5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.85684385746348850219351196129081986508e7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.76131116920014625994371306210585646224e9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.57402411617965582839975369786525269977e11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.42213951996062253608905591667405322835e13),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.55477693883842522631954327528060778834e15),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.00397907346473927493255003955380711046e17),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76305959503723486331556274939198109922e19),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27926540483498824808520492399128682366e21),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.98253913105291675445666919447864520248e22),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63457445658532249936389003141915626894e24),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.51446616633910582673057455450707805902e25),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04744823698010333311911891992022528040e27),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.03400927415310540137351756981742318263e28),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.28761940359662123632247441327784689568e29),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.39016138777648624292953560568071708327e30),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79639567867465767764785448609833337532e31),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.23406781975678544311073661662680006588e32),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.97261483656310352862554580475760827374e33),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62715040832592600542933595577003951697e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.77359945057399130202830211722221279906e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07842295432910751940058270741081867701e35),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.51739306780247334064265249344359460675e35),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.60574331076505049588401700048488577194e35),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.08286808700840316336961663635580879141e35),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27661033115008662284071342245200272702e35),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00465576791024249023365007797010262700e35),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83311248273885136105510175099322638440e34),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96635220211386288597285960837372073054e33),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23849744128418288892902205619933047730e32),
            };
            // LCOV_EXCL_STOP
            result = u * tools::evaluate_polynomial(P, u) / (tools::evaluate_polynomial(Q, u) * (p * p));
        }
        else if (u < 0.25) {
            RealType t = u - static_cast <RealType>(0.125);

            // Rational Approximation
            // Maximum Relative Error: 9.0663e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46698650748920243663487731226111319705e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.39021286045890143123252180276484388346e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21933242816562043224009451007344301143e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33741547463966207206741888477702151242e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.29556944160837955334643715180923663741e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.25261081330476435844217173674285740857e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28690563577245995896389783271544510833e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51764495004238264050843085122188741180e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.00501773552098137637598813101153206656e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93991776883375928647775429233323885440e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13059418708769178567954713937745050279e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41565791250614170744069436181282300453e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.35838723365672196069179944509778281549e1),
            };
            BOOST_MATH_STATIC const RealType Q[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.63888803456697300467924455320638435538e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74785179836182339383932806919167693991e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.15133301804008879476562749311747788645e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87361675398393057971764841741518474061e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02992617475892211368309739891693879676e5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36356854400440662641546588001882412251e5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.35807552915245783626759227539698719908e5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75959389290929178190646034566377062463e5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18514088996371641206828142820042918681e5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54881978220293930450469794941944831047e4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83958740186543542804045767758191509433e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26637084978098507405883170227585648985e2),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
        }
        else if (u < 0.5) {
            RealType t = u - static_cast <RealType>(0.25);

            // Rational Approximation
            // Maximum Relative Error: 7.1265e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[14] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69627866689346442965083437425920959525e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.28948812330446670380449765578224539665e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.38832694133021352110245148952631526683e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70206624753427831733487031852769976576e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34677850226082773550206949299306677736e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.18657422004942861459539366963056149110e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.90933843076824719761937043667767333536e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.78597771586582252472927601403235921029e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76489020985978559079198751910122765603e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.37662018494780327201390375334403954354e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11303058491765900888068268844399186476e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.38147649159947518976483710606042789880e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.81260575060831053615857196033574207714e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.26783311530618626413866321968979725353e-3),
            };
            BOOST_MATH_STATIC const RealType Q[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98941943311823528497840052715295329781e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70142252619301982454969690308614487433e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.17472255695869018956165466705137979540e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.42016169942136311355803413981032780219e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.55874385736597452997483327962434131932e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45782883079400958761816030672202996788e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.05272877129840019671123017296056938361e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79833037593794381103412381177370862105e3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.67388248713896792948592889733513376054e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.19952164110429183557842014635391021832e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.43813483967503071358907030110791934870e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.21327682641358836049127780506729428797e-1),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
        }
        else {
            RealType t = u - static_cast <RealType>(0.5);

            // Rational Approximation
            // Maximum Relative Error: 2.7048e-37
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79518653373241051262822702930040975338e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.62230291299220868262265687829866364204e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87315544620612697712513318458226575394e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.38993950875334507399211313740958438201e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54257654902026056547861805085572437922e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85673656862223617197701693270067722169e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47842193222521213922734312546590337064e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.76640627287007744941009407221495229316e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71884893887802925773271837595143776207e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.87432154629995817972739015224205530101e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44664933176248007092868241686074743562e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32094739938150047092982705610586287965e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.18537678581395571564129512698022192316e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99365265557355974918712592061740510276e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.66467016868206844419002547523627548705e-6),
            };
            BOOST_MATH_STATIC const RealType Q[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01315080955831561204744043759079263546e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.43409077070585581955481063438385546913e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09863540097812452102765922256432103612e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69971336507400724019217277303598318934e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71444880426858110981683485927452024652e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15252748520663939799185721687082682973e2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28399989835264172624148638350889215004e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73464700365199500083227290575797895127e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40421770918884020099427978511354197438e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.97023025282119988988976542004620759235e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38774609088015115009880504176630591783e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52748138528630655371589047000668876440e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13088455793478303045390386135591069087e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.18220605549460262119565543089703387122e-5),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
        }
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 2));

        // Rational Approximation
        // Maximum Relative Error: 3.8969e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.02395484493329839255216366819344305871e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.02703992140456336967688958960484716694e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38779662796374026809611637926067177436e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22326903547451397450399124548020897393e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29321119874906326000117036864856138032e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60045794013093831332658415095234082115e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.75863216252160126657107771372004587438e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65658718311497180532644775193008407069e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.53225259384404343896446164609240157391e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17876243295156782920260122855798305258e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59647007234516896762020830535717539733e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64519789656979327339865975091579252352e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29566724776730544346201080459027524931e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.40979492647851412567441418477263395917e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78894057948338305679452471174923939381e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.97064244496171921075006182915678263370e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13496588267213644899739513941375650458e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.75224691413667093006312591320754720811e-14),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.59000688626663121310675150262772434285e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37518060227321498297232252379976917550e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96559443266702775026538144474892076437e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81865210018244220041408788510705356696e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15188291931842064325756652570456168425e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.16909307081950035111952362482113369939e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68454509269150307761046136063890222011e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06391236761753712424925832120306727169e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.35744804731044427608283991933125506859e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00655928177646208520006978937806043639e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04093230988242553633939757013466501271e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.07921031269974885975846184199640060403e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.61356596082773699708092475561216104426e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83968520269928804453766899533464507543e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44620973323561344735660659502096499899e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.84398925760354259350870730551452956164e-11),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 4.0176e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39293493266195566603513288406748830312e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75724665658983779947977436518056682748e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.42437549740894393207094008058345312893e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26189619865771499663660627120168211026e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.38952430871711360962228087792821341859e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09604487371653920602809626594722822237e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06215021409396534038209460967790566899e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02245531075243838209245241246011523536e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.52822482024384335373072062232322682354e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.32527687997718638700761890588399465467e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54799997015944073019842889902521208940e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59368565314052950335981455903474908073e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09459594346367728583560281313278117879e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30296867679720593932307487485758431355e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04594079707862644415224596859620253913e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40274507498190913768918372242285652373e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.48644117815971872777609922455371868747e-16),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88079839671202113888025645668230104601e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58898753182105924446845274197682915131e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.05418719178760837974322764299800701708e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76439568495464423890950166804368135632e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.87661284201828717694596419805804620767e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29462021166220769918154388930589492957e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89960014717788045459266868996575581278e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21759236630028632465777310665839652757e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08884467282860764261728614542418632608e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60098870889198704716300891829788260654e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00120123451682223443624210304146589040e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.08868117923724451329261971335574401646e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07346130275947166224129347124306950150e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.57848230665832873347797099944091265220e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50841502849442327828534131901583916707e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19165038770000448560339443014882434202e-15),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 4.1682e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57911660613037766795694241662819364797e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28799302413396670477035614399187456630e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.87488304496324715063356722168914018093e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15106082041721012436439208357739139578e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91744691940169259573871742836817806248e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40707390548486625606656777332664791183e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.37047148097688601398129659532643297674e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88039545021930711122085375901243257574e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22254460725736448552173288004145978774e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.58462349007293730244197837509157696852e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95242372547984999431208546685672497090e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10113734998651793201123616276573169622e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.38963677413425618019569452771868834246e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.45242599273032563942546507899265865936e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64855118157117311049698715635863670233e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31679318790012894619592273346600264199e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97289727214495789126072009268721022605e-20),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.10184661848812835285809771940181522329e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.06179300560230499194426573196970342618e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.23171547302923911058112454487643162794e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20486436116678834807354529081908850425e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51239574861351183874145649960640500707e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48939385253081273966380467344920741615e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18148716720470800170115047757600735127e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.68156131480770927662478944117713742978e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13720275846166334505537351224097058812e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85505701632948614345319635028225905820e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.91876669388212587242659571229471930880e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12971661051277278610784329698988278013e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.31096179726750865531615367639563072055e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03579138802970748888093188937926461893e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45570688568663643410924100311054014175e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23959804461200982866930072222355142173e-19),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 6.2158e-37
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59150086070234561732507586188017224084e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.80849532387385837583114307010320459997e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41158479406270598752210344238285334672e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88824037165656723581890282427897772492e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.82912940787568736176030025420621547622e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36469458704261637785603215754389736108e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.13801486421774537025334682673091205328e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97058432407176502984043208925327069250e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60823277541385163663463406307766614689e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45016369260792040947272022706860047646e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54846457278644736871929319230398689553e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.67291749890916930953794688556299297735e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.18742803398417392282841454979723852423e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13337431668170547244474715030235433597e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37782648734897338547414800391203459036e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17863064141234633971470839644872485483e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.58768205048500915346781559321978174829e-24),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27782279546086824129750042200649907991e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.86934772625607907724733810228981095894e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18640923531164938140838239032346416143e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14927915778694317602192656254187608215e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.57462121236985574785071163024761935943e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.11326475640883361512750692176665937785e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49479333407835032831192117009600622344e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01048164044583907219965175201239136609e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42444147056333448589159611785359705792e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.72928365136507710372724683325279209464e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30776323450676114149657959931149982200e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.51599272091669693373558919762006698549e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12120236145539526122748260176385899774e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.65713807525694136400636427188839379484e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.40555520515542383952495965093730818381e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.51084407433793180162386990118245623958e-23),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 9.8515e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59154943017783040087729009335921759322e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.35955784629344586058432079844665517425e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24333525582177610783141409282489279582e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58257137499954581519132407255793210808e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47191495695958634792434622715063010854e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06408464185207904662485396901099847317e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.20796977470988464880970001894205834196e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.99451680244976178843047944033382023574e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21331607817814211329055723244764031561e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.16997758215752306644496702331954449485e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22151810180865778439184946086488092970e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05017329554372903197056366604190738772e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.45919279055502465343977575104142733356e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14611865933281087898817644094411667861e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66574579315129285098834562564888533591e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90098275536617376789480602467351545227e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56200324658873566425094389271790730206e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.35526648761411463124801128103381691418e-26),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.99582804063194774835771688139366152937e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.81210581320456331960539046132284190053e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94358920922810097599951120081974275145e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.24831443209858422294319043037419780210e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.68584098673893150929178892200446909375e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.90058244779420124535106788512940199547e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88151039746201934320258158884886191886e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.62348975553160355852344937226490493460e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62007418751593938350474825754731470453e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.67502458979132962529935588245058477825e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91648040348401277706598232576212305626e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.05843052379331618504561151714467880641e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.20127592059771206959014911028588129920e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04661894930286305556240859086772458465e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19442269177165740287568170417649762849e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.81435584873372180820418114652670864136e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.62145023253666168339801687459484937001e-25),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 2.2157e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59154943091895335751628149866310390641e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.91164927854420277537616294413463565970e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47557801928232619499125670863084398577e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06172621625221091203249391660455847328e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11720157411653968975956625234656001375e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70086412127379161257840749700428137407e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11069186177775505692019195793079552937e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.04581765901792649215653828121992908775e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78996797234624395264657873201296117159e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10365978021268853654282661591051834468e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76744621013787434243259445839624450867e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11110170303355425599446515240949433934e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83669090335022069229153919882930282425e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.09633460833089193733622172621696983652e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16200852052266861122422190933586966917e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47795810090424252745150042033544310609e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35722092370326505616747155207965300634e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.98381676423023212724768510437325359364e-51),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34271731953273239691423485699928257808e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27133013035186849140772481980360839559e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29542078693828543560388747333519393752e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33027698228265344561650492600955601983e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06868444562964057795387778972002636261e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.97868278672593071151212650783507879919e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79869926850283188785885503049178903204e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75298857713475428388051708713106549590e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.93449891515741631942400181171061740767e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36715626731277089044713008494971829270e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.98125789528264426960496891930942311971e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78234546049400950544724588355539821858e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83044000387150792693434128054414524740e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.30111486296552039483196431122555524886e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.28628462422858135083238952377446358986e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48108558735886480298604270981393793162e-21),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * (p * p));
    }
    else {
        result = 1 / (p * p * constants::two_pi<RealType>());
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 53>& tag)
{
    if (p > 0.5) {
        return !complement ? saspoint5_quantile_upper_imp_prec(1 - p, tag) : -saspoint5_quantile_upper_imp_prec(1 - p, tag);
    }

    return complement ? saspoint5_quantile_upper_imp_prec(p, tag) : -saspoint5_quantile_upper_imp_prec(p, tag);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 113>& tag)
{
    if (p > 0.5) {
        return !complement ? saspoint5_quantile_upper_imp_prec(1 - p, tag) : -saspoint5_quantile_upper_imp_prec(1 - p, tag);
    }

    return complement ? saspoint5_quantile_upper_imp_prec(p, tag) : -saspoint5_quantile_upper_imp_prec(p, tag);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_quantile_imp(const saspoint5_distribution<RealType, Policy>& dist, const RealType& p, bool complement)
{
    // This routine implements the quantile for the Saspoint5 distribution,
    // the value p may be the probability, or its complement if complement=true.

    constexpr auto function = "boost::math::quantile(saspoint5<%1%>&, %1%)";
    BOOST_MATH_STD_USING // for ADL of std functions

    RealType result = 0;
    RealType scale = dist.scale();
    RealType location = dist.location();

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

    static_assert(tag_type::value, "The SaS point5 distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * saspoint5_quantile_imp_prec(p, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_entropy_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(3.63992444568030649573);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_entropy_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.6399244456803064957308496039071853510);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType saspoint5_entropy_imp(const saspoint5_distribution<RealType, Policy>& dist)
{
    // This implements the entropy for the Saspoint5 distribution,

    constexpr auto function = "boost::math::entropy(saspoint5<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The SaS point5 distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = saspoint5_entropy_imp_prec<RealType>(tag_type()) + log(scale);

    return result;
}

} // detail

template <class RealType = double, class Policy = policies::policy<> >
class saspoint5_distribution
{
    public:
    typedef RealType value_type;
    typedef Policy policy_type;

    BOOST_MATH_GPU_ENABLED saspoint5_distribution(RealType l_location = 0, RealType l_scale = 1)
        : mu(l_location), c(l_scale)
    {
        constexpr auto function = "boost::math::saspoint5_distribution<%1%>::saspoint5_distribution";
        RealType result = 0;
        detail::check_location(function, l_location, &result, Policy());
        detail::check_scale(function, l_scale, &result, Policy());
    } // saspoint5_distribution

    BOOST_MATH_GPU_ENABLED RealType location()const
    {
        return mu;
    }
    BOOST_MATH_GPU_ENABLED RealType scale()const
    {
        return c;
    }

    private:
    RealType mu;    // The location parameter.
    RealType c;     // The scale parameter.
};

typedef saspoint5_distribution<double> saspoint5;

#ifdef __cpp_deduction_guides
template <class RealType>
saspoint5_distribution(RealType) -> saspoint5_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
saspoint5_distribution(RealType, RealType) -> saspoint5_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const saspoint5_distribution<RealType, Policy>&)
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
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const saspoint5_distribution<RealType, Policy>&)
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
BOOST_MATH_GPU_ENABLED inline RealType pdf(const saspoint5_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::saspoint5_pdf_imp(dist, x);
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const saspoint5_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::saspoint5_cdf_imp(dist, x, false);
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const saspoint5_distribution<RealType, Policy>& dist, const RealType& p)
{
    return detail::saspoint5_quantile_imp(dist, p, false);
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<saspoint5_distribution<RealType, Policy>, RealType>& c)
{
    return detail::saspoint5_cdf_imp(c.dist, c.param, true);
} //  cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<saspoint5_distribution<RealType, Policy>, RealType>& c)
{
    return detail::saspoint5_quantile_imp(c.dist, c.param, true);
} // quantile complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const saspoint5_distribution<RealType, Policy> &dist)
{
    // There is no mean:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The SaS point5 Distribution has no mean");

    return policies::raise_domain_error<RealType>(
        "boost::math::mean(saspoint5<%1%>&)",
        "The SaS point5 distribution does not have a mean: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const saspoint5_distribution<RealType, Policy>& /*dist*/)
{
    // There is no variance:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The SaS point5 Distribution has no variance");

    return policies::raise_domain_error<RealType>(
        "boost::math::variance(saspoint5<%1%>&)",
        "The SaS point5 distribution does not have a variance: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const saspoint5_distribution<RealType, Policy>& dist)
{
    return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const saspoint5_distribution<RealType, Policy>& dist)
{
    return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const saspoint5_distribution<RealType, Policy>& /*dist*/)
{
    // There is no skewness:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The SaS point5 Distribution has no skewness");

    return policies::raise_domain_error<RealType>(
        "boost::math::skewness(saspoint5<%1%>&)",
        "The SaS point5 distribution does not have a skewness: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy()); // infinity?
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const saspoint5_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The SaS point5 Distribution has no kurtosis");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis(saspoint5<%1%>&)",
        "The SaS point5 distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const saspoint5_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis excess:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The SaS point5 Distribution has no kurtosis excess");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis_excess(saspoint5<%1%>&)",
        "The SaS point5 distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const saspoint5_distribution<RealType, Policy>& dist)
{
    return detail::saspoint5_entropy_imp(dist);
}

}} // namespaces


#endif // BOOST_STATS_SASPOINT5_HPP
