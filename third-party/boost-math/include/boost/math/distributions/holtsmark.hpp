//  Copyright Takuma Yoshimura 2024.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_HOLTSMARK_HPP
#define BOOST_STATS_HOLTSMARK_HPP

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
#include <boost/math/special_functions/cbrt.hpp>
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
class holtsmark_distribution;

namespace detail {

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 4.7894e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.87352751452164445024e-1),
            static_cast<RealType>(1.18577398160636011811e-3),
            static_cast<RealType>(-2.16526599226820153260e-2),
            static_cast<RealType>(2.06462093371223113592e-3),
            static_cast<RealType>(2.43382128013710116747e-3),
            static_cast<RealType>(-2.15930711444603559520e-4),
            static_cast<RealType>(-1.04197836740809694657e-4),
            static_cast<RealType>(1.74679078247026597959e-5),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.12654472808214997252e-3),
            static_cast<RealType>(2.93891863033354755743e-1),
            static_cast<RealType>(8.70867222155141724171e-3),
            static_cast<RealType>(3.15027515421842640745e-2),
            static_cast<RealType>(2.11141832312672190669e-3),
            static_cast<RealType>(1.23545521355569424975e-3),
            static_cast<RealType>(1.58181113865348637475e-4),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 3.0925e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.02038159607840130389e-1),
            static_cast<RealType>(-1.20368541260123112191e-2),
            static_cast<RealType>(-3.19235497414059987151e-3),
            static_cast<RealType>(8.88546222140257289852e-3),
            static_cast<RealType>(-5.37287599824602316660e-4),
            static_cast<RealType>(-2.39059149972922243276e-4),
            static_cast<RealType>(9.19551014849109417931e-5),
            static_cast<RealType>(-8.45210544648986348854e-6),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.11634701234079515138e-1),
            static_cast<RealType>(4.39922162828115412952e-1),
            static_cast<RealType>(1.73609068791154078128e-1),
            static_cast<RealType>(6.15831808473403962054e-2),
            static_cast<RealType>(1.64364949550314788638e-2),
            static_cast<RealType>(2.94399615562137394932e-3),
            static_cast<RealType>(4.99662797033514776061e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 1.4499e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(8.45396231261375200568e-2),
            static_cast<RealType>(-9.15509628797205847643e-3),
            static_cast<RealType>(1.82052933284907579374e-2),
            static_cast<RealType>(-2.44157914076021125182e-4),
            static_cast<RealType>(8.40871885414177705035e-4),
            static_cast<RealType>(7.26592615882060553326e-5),
            static_cast<RealType>(-1.87768359214600016641e-6),
            static_cast<RealType>(1.65716961206268668529e-6),
            static_cast<RealType>(-1.73979640146948858436e-7),
            static_cast<RealType>(7.24351142163396584236e-9),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.88099527896838765666e-1),
            static_cast<RealType>(6.53896948546877341992e-1),
            static_cast<RealType>(2.96296982585381844864e-1),
            static_cast<RealType>(1.14107585229341489833e-1),
            static_cast<RealType>(3.08914671331207488189e-2),
            static_cast<RealType>(7.03139384769200902107e-3),
            static_cast<RealType>(1.01201814277918577790e-3),
            static_cast<RealType>(1.12200113270398674535e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 6.5259e-17
        BOOST_MATH_STATIC const RealType P[11] = {
            static_cast<RealType>(1.36729417918039395222e-2),
            static_cast<RealType>(1.19749117683408419115e-2),
            static_cast<RealType>(6.26780921592414207398e-3),
            static_cast<RealType>(1.84846137440857608948e-3),
            static_cast<RealType>(3.39307829797262466829e-4),
            static_cast<RealType>(2.73606960463362090866e-5),
            static_cast<RealType>(-1.14419838471713498717e-7),
            static_cast<RealType>(1.64552336875610576993e-8),
            static_cast<RealType>(-7.95501797873739398143e-10),
            static_cast<RealType>(2.55422885338760255125e-11),
            static_cast<RealType>(-4.12196487201928768038e-13),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.61334003864149486454e0),
            static_cast<RealType>(1.28348868912975898501e0),
            static_cast<RealType>(6.36594545291321210154e-1),
            static_cast<RealType>(2.11478937436277242988e-1),
            static_cast<RealType>(4.71550897200311391579e-2),
            static_cast<RealType>(6.64679677197059316835e-3),
            static_cast<RealType>(4.93706832858615742810e-4),
            static_cast<RealType>(9.26919465059204396228e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 3.5084e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.90649774685568282390e-3),
            static_cast<RealType>(7.43708409389806210196e-4),
            static_cast<RealType>(9.53777347766128955847e-5),
            static_cast<RealType>(3.79800193823252979170e-6),
            static_cast<RealType>(2.84836656088572745575e-8),
            static_cast<RealType>(-1.22715411241721187620e-10),
            static_cast<RealType>(8.56789906419220801109e-13),
            static_cast<RealType>(-4.17784858891714869163e-15),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.29383849235788831455e-1),
            static_cast<RealType>(2.16287201867831015266e-1),
            static_cast<RealType>(3.28789040872705709070e-2),
            static_cast<RealType>(2.64660789801664804789e-3),
            static_cast<RealType>(1.03662724048874906931e-4),
            static_cast<RealType>(1.47658125632566407978e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 1.4660e-19
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(3.07231582988207590928e-4),
            static_cast<RealType>(5.16108848485823513911e-5),
            static_cast<RealType>(3.05776014220862257678e-6),
            static_cast<RealType>(7.64787444325088143218e-8),
            static_cast<RealType>(7.40426355029090813961e-10),
            static_cast<RealType>(1.57451122102115077046e-12),
            static_cast<RealType>(-2.14505675750572782093e-15),
            static_cast<RealType>(5.11204601013038698192e-18),
            static_cast<RealType>(-9.00826023095223871551e-21),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.28966789835486457746e-1),
            static_cast<RealType>(4.46981634258601621625e-2),
            static_cast<RealType>(3.22521297380474263906e-3),
            static_cast<RealType>(1.31985203433890010111e-4),
            static_cast<RealType>(3.01507121087942156530e-6),
            static_cast<RealType>(3.47777238523841835495e-8),
            static_cast<RealType>(1.50780503777979189972e-10),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 4.2292e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(5.25741312407933720817e-5),
            static_cast<RealType>(2.34425802342454046697e-6),
            static_cast<RealType>(3.30042747965497652847e-8),
            static_cast<RealType>(1.58564820095683252738e-10),
            static_cast<RealType>(1.54070758384735212486e-13),
            static_cast<RealType>(-8.89232435250437247197e-17),
            static_cast<RealType>(8.14099948000080417199e-20),
            static_cast<RealType>(-4.61828164399178360925e-23),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.23544974283127158019e-1),
            static_cast<RealType>(6.01210465184576626802e-3),
            static_cast<RealType>(1.45390926665383063500e-4),
            static_cast<RealType>(1.80594709695117864840e-6),
            static_cast<RealType>(1.06088985542982155880e-8),
            static_cast<RealType>(2.20287881724613104903e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x * x * x);

        // Rational Approximation
        // Maximum Relative Error: 2.3004e-17
        BOOST_MATH_STATIC const RealType P[4] = {
            static_cast<RealType>(2.99206710301074508455e-1),
            static_cast<RealType>(-8.62469397757826072306e-1),
            static_cast<RealType>(1.74661995423629075890e-1),
            static_cast<RealType>(8.75909164947413479137e-1),
        };
        BOOST_MATH_STATIC const RealType Q[3] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(-6.07405848111002255020e0),
            static_cast<RealType>(1.34068401972703571636e1),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t / x;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 4.5215e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87352751452164445024482162286994868262e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.07622509000285763173795736744991173600e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75004930885780661923539070646503039258e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.72358602484766333657370198137154157310e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80082654994455046054228833198744292689e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.53887200727615005180492399966262970151e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07684195532179300820096260852073763880e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.39151986881253768780523679256708455051e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.31700721746247708002568205696938014069e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.52538425285394123789751606057231671946e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.13997198703138372752313576244312091598e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74788965317036115104204201740144738267e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.18994723428163008965406453309272880204e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.49208308902369087634036371223527932419e-11),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.07053963271862256947338846403373278592e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30146528469038357598785392812229655811e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.22168809220570888957518451361426420755e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.30911708477464424748895247790513118077e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.32037605861909345291211474811347056388e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.37380742268959889784160508321242249326e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.17777859396994816599172003124202701362e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.69357597449425742856874347560067711953e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.22061268498705703002731594804187464212e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.03685918248668999775572498175163352453e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.42037705933347925911510259098903765388e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.13651251802353350402740200231061151003e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.15390928968620849348804301589542546367e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.96186359077726620124148756657971390386e-9),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 1.3996e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.02038159607840130388931544845552929992e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.85240836242909590376775233472494840074e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.92928437142375928121954427888812334305e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.56075992368354834619445578502239925632e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85410663490566091471288623735720924369e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.09160661432404033681463938555133581443e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60290555290385646856693819798655258098e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24420942563054709904053017769325945705e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.06370233020823161157791461691510091864e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.51562554221298564845071290898761434388e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.77361020844998296791409508640756247324e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.10768937536097342883548728871352580308e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.97810512763454658214572490850146305033e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.77430867682132459087084564268263825239e-11),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.30030169049261634787262795838348954434e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45935676273909940847479638179887855033e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14724239378269259016679286177700667008e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21580123796578745240828564510740594111e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70287348745451818082884807214512422940e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46859813604124308580987785473592196488e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.49627445316021031361394030382456867983e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05157712406194406440213776605199788051e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91541875103990251411297099611180353187e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.47960462287955806798879139599079388744e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.80126815763067695392857052825785263211e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04569118116204820761181992270024358122e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.63024381269503801668229632579505279520e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00967434338725770754103109040982001783e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 1.6834e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.45396231261375200568114750897618690566e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.83107635287140466760500899510899613385e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71690205829238281191309321676655995475e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.95995611963950467634398178757261552497e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.52444689050426648467863527289016233648e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.40423239472181137610649503303203209123e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72181273738390251101985797318639680476e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.11423032981781501087311583401963332916e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37255768388351332508195641748235373885e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.25140171472943043666747084376053803301e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98925617316135247540832898350427842870e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.27532592227329144332335468302536835334e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.25846339430429852334026937219420930290e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.17852693845678292024334670662803641322e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60008761860786244203651832067697976835e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.85474213475378978699789357283744252832e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.05561259222780127064607109581719435800e-15),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08902510590064634965634560548380735284e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.60127698266075086782895988567899172787e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.73299227011247478433171171063045855612e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.94019328695445269130845646745771017029e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21478511930928822349285105322914093227e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.42888485420705779382804725954524839381e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36839484685440714657854206969200824442e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77082068469251728028552451884848161629e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.92625563541021144576900067220082880950e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88302521658522279293312672887766072876e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.37703703342287521257351386589629343948e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.32454189932655869016489443530062686013e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.81822848072558151338694737514507945151e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.40176559099032106726456059226930240477e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.55722115663529425797132143276461872035e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.18236697046568703899375072798708359035e-10),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 5.6207e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[20] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36729417918039395222067998266923903488e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05780369334958736210688756060527042344e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88449456199223796440901487003885388570e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20213624124017393492512893302682417041e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.95009975955570002297453163471062373746e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35668345583965001606910217518443864382e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.69006847702829685253055277085000792826e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08366922884479491780654020783735539561e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.71834368599657597252633517017213868956e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.88269472722301903965736220481240654265e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37797139843759131750966129487745639531e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72390971590654495025982276782257590019e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68354503497961090303189233611418754374e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20749461042713568368181066233478264894e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.71167265100639100355339812752823628805e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37497033071709741762372104386727560387e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.08992504249040731356693038222581843266e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.03311745412603363076896897060158476094e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.89266184062176002518506060373755160893e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.22157263424086267338486564980223658130e-22),
        };
        BOOST_MATH_STATIC const RealType Q[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24254809760594824834854946949546737102e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66740386908805016172202899592418717176e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17175023341071972435947261868288366592e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33939409711833786730168591434519989589e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.58859674176126567295417811572162232222e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66346764121676348703738437519493817401e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.00687534341032230207422557716131339293e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57352381181825892637055619366793541271e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.23955067096868711061473058513398543786e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28279376429637301814743591831507047825e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22380760186302431267562571014519501842e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.21421839279245792393425090284615681867e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.80151544531415207189620615654737831345e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.57177992740786529976179511261318869505e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54223623314672019530719165336863142227e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26447311109866547647645308621478963788e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76514314007336173875469200193103772775e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.63785420481380041892410849615596985103e-13),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 6.8882e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90649774685568282389553481307707005425e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70151946710788532273869130544473159961e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76188245008605985768921328976193346788e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.94997481586873355765607596415761713534e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83556339450065349619118429405554762845e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.39766178753196196595432796889473826698e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48835240264191055418415753552383932859e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.23205178959384483669515397903609703992e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80665018951397281836428650435128239368e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27113208299726105096854812628329439191e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.75272882929773945317046764560516449105e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.73174017370926101455204470047842394787e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.55548825213165929101134655786361059720e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.79786015549170518239230891794588988732e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73060731998834750292816218696923192789e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.62842837946576938669447109511449827857e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33878078951302606409419167741041897986e-26),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75629880937514507004822969528240262723e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43883005193126748135739157335919076027e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.26826935326347315479579835343751624245e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52263130214924169696993839078084050641e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.34708681216662922818631865761136370252e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19079618273418070513605131981401070622e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68812668867590621701228940772852924670e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81323523265546812020317698573638573275e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46655191174052062382710487986225631851e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.79864553144116347379916608661549264281e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.81866770335021233700248077520029108331e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15408288688082935176022095799735538723e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29421875915133979067465908221270435168e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74564282803894180881025348633912184161e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69782249847887916810010605635064672269e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.85875986197737611300062229945990879767e-18),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 2.7988e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.07231582988207590928480356376941073734e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35574911514921623999866392865480652576e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.60219401814297026945664630716309317015e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84927222345566515103807882976184811760e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96327408363203008584583124982694689234e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.86684048703029160378252571846517319101e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65469175974819997602752600929172261626e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.21842057555380199566706533446991680612e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.53555106309423641769303386628162522042e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.92686543698369260585325449306538016446e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01838615452860702770059987567879856504e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.65492535746962514730615062374864701860e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53395563720606494853374354984531107080e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.99957357701259203151690416786669242677e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46357124817620384236108395837490629563e-31),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02259092175256156108200465685980768901e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63438230616954606028022008517920766366e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.63880061357592661176130881772975919418e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81911305852397235014131637306820512975e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09690724408294608306577482852270088377e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11275552068434583356476295833517496456e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.24681861037105338446379750828324925566e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16034379416965004687140768474445096709e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23234481703249409689976894391287818596e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93297387560911081670605071704642179017e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50338428974314371000017727660753886621e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27897854868353937080739431205940604582e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.37798740524930029176790562876868493344e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29920082153439260734550295626576101192e-22),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 6.9688e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.25741312407933720816582583160953651639e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.04434146174674791036848306058526901384e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.68959516304795838166182070164492846877e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78859935261158263390023581309925613858e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.21854067989018450973827853792407054510e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20573856697340412957421887367218135538e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30843538021351383101589538141878424462e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.05991458689384045976214216819611949900e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.82253708752556965233757129893944884411e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.97645331663303764054986066027964294209e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.69353366461654917577775981574517182648e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59050144462227302681332505386238071973e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.85165507189649330971049854127575847359e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70711310565669331853925519429988855964e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.72047006026700174884151916064158941262e-38),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50985661940624198574968436548711898948e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81705882167596649186405364717835589894e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86537779048672498307196786015602357729e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09555188550938733096253930959407749063e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41930442687159455334801545898059105733e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.09084284266255183930305946875294557622e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.58122754063904909636061457739518406730e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.91800215912676651584368499126132687326e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.66413330532845384974993669138524203429e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65919563020196445006309683624384862816e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.61596083414169579692212575079167989319e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16321386033703806802403099255708972015e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.90892719803158002834365234646982537288e-25),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x * x * x);

        // Rational Approximation
        // Maximum Relative Error: 3.0545e-39
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[8] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.99206710301074508454959544950786401357e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.75243304700875633383991614142545185173e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.69652690455351600373808930804785330828e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.36233941060408773406522171349397343951e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.28958973553713980463808202034854958375e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.55704950313835982743029388151551925282e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.28767698270323629107775935552991333781e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.80591252844738626580182351673066365090e1),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.57593243741246726197476469913307836496e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.99458751269722094414105565700775283458e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.91043982880665229427553316951582511317e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.99054490423334526438490907473548839751e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.36948968143124830402744607365089118030e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13781639547150826385071482161074041168e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t / x;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53> &tag) {
    BOOST_MATH_STD_USING // for ADL of std functions

    return holtsmark_pdf_plus_imp_prec<RealType>(abs(x), tag);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>& tag) {
    BOOST_MATH_STD_USING // for ADL of std functions

    return holtsmark_pdf_plus_imp_prec<RealType>(abs(x), tag);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_pdf_imp(const holtsmark_distribution<RealType, Policy>& dist, const RealType& x) {
    //
    // This calculates the pdf of the Holtsmark distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::pdf(holtsmark<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Holtsmark distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale;

    result = holtsmark_pdf_imp_prec(u, tag_type()) / scale;

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 0.5) {
        // Rational Approximation
        // Maximum Relative Error: 1.3147e-17
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(5.0e-1),
            static_cast<RealType>(-1.34752580674786639030e-1),
            static_cast<RealType>(1.86318418252163378528e-2),
            static_cast<RealType>(1.04499798132512381447e-2),
            static_cast<RealType>(-1.60831910014592923855e-3),
            static_cast<RealType>(1.38823662364438342844e-4),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.05200341554753776087e-1),
            static_cast<RealType>(2.12663999430421346175e-1),
            static_cast<RealType>(7.23836000984872591553e-2),
            static_cast<RealType>(1.67941072412796299986e-2),
            static_cast<RealType>(4.71213644318790580839e-3),
            static_cast<RealType>(5.86825130959777535991e-4),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 1) {
        RealType t = x - 0.5f;

        // Rational Approximation
        // Maximum Relative Error: 1.6265e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(3.60595773518728397351e-1),
            static_cast<RealType>(5.75238626843218819756e-1),
            static_cast<RealType>(-3.31245319943021227117e-1),
            static_cast<RealType>(1.48132966310216368831e-1),
            static_cast<RealType>(-2.32875122617713403365e-2),
            static_cast<RealType>(2.08038303148835575624e-3),
            static_cast<RealType>(6.01511310581302829460e-6),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.32264360456739861886e0),
            static_cast<RealType>(6.39715443864749851087e-1),
            static_cast<RealType>(5.03940458163958921325e-1),
            static_cast<RealType>(8.84780893031413729292e-2),
            static_cast<RealType>(3.01497774031208621961e-2),
            static_cast<RealType>(3.45886005612108195390e-3),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 7.4398e-20
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.43657975600729535515e-1),
            static_cast<RealType>(-6.02286263626532324632e-2),
            static_cast<RealType>(4.68361231392743283350e-2),
            static_cast<RealType>(-1.13497179885838883972e-3),
            static_cast<RealType>(1.20141595689136205012e-3),
            static_cast<RealType>(3.02402304689333413256e-4),
            static_cast<RealType>(-1.22652173865646814676e-6),
            static_cast<RealType>(2.29521832683440044997e-6),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.82002427359748247121e-1),
            static_cast<RealType>(3.96529686558825119743e-1),
            static_cast<RealType>(1.49690294526117385174e-1),
            static_cast<RealType>(5.15049953937764895435e-2),
            static_cast<RealType>(1.30218216530450637564e-2),
            static_cast<RealType>(2.53640337919037463659e-3),
            static_cast<RealType>(3.79575042317720710311e-4),
            static_cast<RealType>(2.94034997185982139717e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 5.6148e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(1.05039829654829164883e-1),
            static_cast<RealType>(1.66621813028423002562e-2),
            static_cast<RealType>(2.93820049104275137099e-2),
            static_cast<RealType>(3.36850260303189378587e-3),
            static_cast<RealType>(2.27925819398326978014e-3),
            static_cast<RealType>(1.66394162680543987783e-4),
            static_cast<RealType>(4.51400415642703075050e-5),
            static_cast<RealType>(2.12164734714059446913e-7),
            static_cast<RealType>(1.69306881760242775488e-8),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.63461239051296108254e-1),
            static_cast<RealType>(6.54183344973801096611e-1),
            static_cast<RealType>(2.92007762594247903696e-1),
            static_cast<RealType>(1.00918751132022401499e-1),
            static_cast<RealType>(2.55899135910670703945e-2),
            static_cast<RealType>(4.85740416919283630358e-3),
            static_cast<RealType>(6.11435190489589619906e-4),
            static_cast<RealType>(4.10953248859973756440e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 6.5866e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(3.05754562114095142887e-2),
            static_cast<RealType>(3.25462617990002726083e-2),
            static_cast<RealType>(1.78205524297204753048e-2),
            static_cast<RealType>(5.61565369088816402420e-3),
            static_cast<RealType>(1.05695297340067353106e-3),
            static_cast<RealType>(9.93588579804511250576e-5),
            static_cast<RealType>(2.94302107205379334662e-6),
            static_cast<RealType>(1.09016076876928010898e-8),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.51164395622515150122e0),
            static_cast<RealType>(1.09391911233213526071e0),
            static_cast<RealType>(4.77950346062744800732e-1),
            static_cast<RealType>(1.34082684956852773925e-1),
            static_cast<RealType>(2.37572579895639589816e-2),
            static_cast<RealType>(2.41806218388337284640e-3),
            static_cast<RealType>(1.10378140456646280084e-4),
            static_cast<RealType>(1.31559373832822136249e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 5.6575e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(9.47408470248235718880e-3),
            static_cast<RealType>(4.70888722333356024081e-3),
            static_cast<RealType>(8.66397831692913140221e-4),
            static_cast<RealType>(7.11721056656424862090e-5),
            static_cast<RealType>(2.56320582355149253994e-6),
            static_cast<RealType>(3.37749186035552101702e-8),
            static_cast<RealType>(8.32182844837952178153e-11),
            static_cast<RealType>(-8.80541360484428526226e-14),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.98261117346347123707e-1),
            static_cast<RealType>(1.97823959738695249267e-1),
            static_cast<RealType>(2.89311735096848395080e-2),
            static_cast<RealType>(2.30087055379997473849e-3),
            static_cast<RealType>(9.60592522700377510007e-5),
            static_cast<RealType>(1.84474415187428058231e-6),
            static_cast<RealType>(1.14339998084523151203e-8),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 1.4164e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(3.19610991747326729867e-3),
            static_cast<RealType>(5.11880074251341162590e-4),
            static_cast<RealType>(2.80704092977662888563e-5),
            static_cast<RealType>(6.31310155466346114729e-7),
            static_cast<RealType>(5.29618446795457166842e-9),
            static_cast<RealType>(9.20292337847562746519e-12),
            static_cast<RealType>(-9.16761719448360345363e-15),
            static_cast<RealType>(1.20433396121606479712e-17),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.56283944667056551858e-1),
            static_cast<RealType>(2.56811818304462676948e-2),
            static_cast<RealType>(1.26678062261253559927e-3),
            static_cast<RealType>(3.17001344827541091252e-5),
            static_cast<RealType>(3.68737201224811007437e-7),
            static_cast<RealType>(1.47625352605312785910e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 9.2537e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.11172037056341397612e-3),
            static_cast<RealType>(7.84545643188695076893e-5),
            static_cast<RealType>(1.94862940242223222641e-6),
            static_cast<RealType>(2.02704958737259525509e-8),
            static_cast<RealType>(7.99772378955335076832e-11),
            static_cast<RealType>(6.62544230949971310060e-14),
            static_cast<RealType>(-3.18234118727325492149e-17),
            static_cast<RealType>(2.03424457039308806437e-20),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.17861198759233241198e-1),
            static_cast<RealType>(5.45962263583663240699e-3),
            static_cast<RealType>(1.25274651876378267111e-4),
            static_cast<RealType>(1.46857544539612002745e-6),
            static_cast<RealType>(8.06441204620771968579e-9),
            static_cast<RealType>(1.53682779460286464073e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType x_cube = x * x * x;
        RealType t = static_cast<RealType>((boost::math::isnormal)(x_cube) ? 1 / sqrt(x_cube) : 1 / pow(sqrt(x), 3));

        // Rational Approximation
        // Maximum Relative Error: 4.2897e-18
        BOOST_MATH_STATIC const RealType P[4] = {
            static_cast<RealType>(1.99471140200716338970e-1),
            static_cast<RealType>(-6.90933799347184400422e-1),
            static_cast<RealType>(4.30385245884336871950e-1),
            static_cast<RealType>(3.52790131116013716885e-1),
        };
        BOOST_MATH_STATIC const RealType Q[3] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(-5.05959751628952574534e0),
            static_cast<RealType>(8.04408113719341786819e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t;
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 0.5) {
        // Rational Approximation
        // Maximum Relative Error: 8.6635e-36
       // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.0e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.48548242430636907136192799540229598637e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31541453581608245475805834922621529866e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.16579064508490250336159593502955219069e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61598809551362112011328341554044706550e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.15119245273512554325709429759983470969e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02145196753734867721148927112307708045e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.90817224464950088663183617156145065001e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69596202760983052482358128481956242532e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.50461337222845025623869078372182437091e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.62777995800923647521692709390412901586e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.63937253747323898965514197114021890186e-8),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.76090180430550757765787254935343576341e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.07685236907561593034104428156351640194e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27770556484351179553611274487979706736e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.99201460869149634331004096815257398515e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70139000408086498153685620963430185837e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74682544708653069148470666809094453722e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57607114117485446922700160080966856243e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01069214414741946409122492979083487977e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.19996282759031441186748256811206136921e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60933466092746543579699079418115420013e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.92780739162611243933581782562159603862e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 1) {
        RealType t = x - 0.5;

        // Rational Approximation
        // Maximum Relative Error: 7.1235e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60595773518728397925852903878144761766e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.46999595154527091473427440379143006753e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36962313432466566724352608642383560211e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08387290167105915393692028475888846796e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.34156151832478939276011262838869269011e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15970594471853166393830585755485842021e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.47022841547527682761332752928069503835e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.01955019188793323293925482112543902560e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.03069493388735516695142799880566783261e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61367662035593735709965982000611000987e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.62800430658278408539398798888955969345e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.22300086876618079439960709120163780513e-8),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19740977756009966244249035150363085180e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39394884078938560974435920719979860046e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.97107758486905601309707335353809421910e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.36594079604957733960211938310153276332e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.85712904264673773213248691029253356702e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.87605080555629969548037543637523346061e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.26356599628579249350545909071984757938e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.79582114368994462181480978781382155103e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.00970375323007336435151032145023199020e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.06528824060244313614177859412028348352e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.13914667697998291289987140319652513139e-7),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 6.7659e-38
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43657975600729535499895880792984203140e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.37090874182351552816526775008685285108e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70793783828569126853147999925198280654e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.27295555253412802819195403503721983066e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.95916890788873842705597506423512639342e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93625795791721417553345795882983866640e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.73237387099610415336810752053706403935e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08118655139419640900853055479087235138e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74920069862339840183963818219485580710e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59015304773612605296533206093582658838e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57256820413579442950151375512313072105e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.36240848333000575199740403759568680951e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53890585580518120552628221662318725825e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59245311730292556271235324976832000740e-10),
        };
        BOOST_MATH_STATIC const RealType Q[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.49800491033591771256676595185869442663e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.35827615015880595229881139361463765537e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41657125931991211322147702760511651998e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11782602975553967179829921562737846592e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79410805176258968660086532862367842847e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22872839892405613311532856773434270554e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.23742349724658114137235071924317934569e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.80350762663884259375711329227548815674e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59501693037547119094683008622867020131e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.86068186167498269806443077840917848151e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.36940342373887783231154918541990667741e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.48911186460768204167014270878839691938e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.55051094964993052272146587430780404904e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.96312716130620326771080033656930839768e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45496951385730104726429368791951742738e-10),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 9.9091e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05039829654829170780787685299556996311e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28948022754388615368533934448107849329e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.34139151583225691775740839359914493385e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13366377215523066657592295006960955345e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08045462837998791188853367062130086996e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.37648565386728404881404199616182064711e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14881702523183566448187346081007871684e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73169022445183613027772635992366708052e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.86434609673325793686202636939208406356e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20865083025640755296377488921536984172e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.24550087063009488023243811976147518386e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78763689691843975658550702147832072016e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.53901449493513509116902285044951137217e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64133451376958243174967226929215155126e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78021916681275593923355425070000331160e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40116391931116431686557163556034777896e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.43891156389092896219387988411277617045e-15),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30840297297890638941129884491157396207e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16059271948787750556465175239345182035e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.32333703228724830516425197803770832978e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.74722711058640395885914966387546141874e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57544653090705553268164186689966671940e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.65943099435809995745673109708218670077e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74158626875895095042054345316232575354e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.65978318533667031874695821156329945501e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07907034178758316909655424935083792468e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.16769901831316460137104511711073411646e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.72764558714782436683712413015421717627e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.42494185105694341746192094740530489313e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.47668761140694808076322373887857100882e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.06395948884595166425357861427667353718e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.04398743651684916010743222115099630062e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47852251142917253705233519146081069006e-10),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 3.2255e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[20] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05754562114095147060025732340404111260e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.29082907781747007723015304584383528212e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.15736486393536930535038719804968063752e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.47619683293773846642359668429058772885e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78777185267549567154655052281449528836e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.32280474402180284471490985942690221861e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.45430564625797085273267452885960070105e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.81643129239005795245093568930666448817e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57851748656417804512189330871167578685e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.04264676511381380381909064283066657450e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.84536783037391183433322642273799250079e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.27169201994160924743393109705813711010e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.42623512076200527099335832138825884729e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.98298083389459839517970895839114237996e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71357920034737751299594537655948527288e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.98563999354325930973228648080876368296e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36248172644168880316722905969876969074e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.61071663749398045880261823483568866904e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.95933262363502031836408613043245164787e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23007623135952181561484264810647517912e-21),
        };
        BOOST_MATH_STATIC const RealType Q[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17760389606658547971193065026711073898e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49565543987559264712057768584303008339e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94822569926563661124528478579051628722e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14676844425183314970062115422221981422e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.35960757354198367535169328826167556715e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04865288305482048252211468989095938024e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.51599632816346741950206107526304703067e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74065824586512487126287762563576185455e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91819078437689679732215988465616022328e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.41675362609023565846569121735444698127e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17176431752708802291177040031150143262e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52367587943529121285938327286926798550e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59405168077254169099025950029539316125e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29448420654438993509041228047289503943e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.70091773726833073512661846603385666642e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.03909417984236210307694235586859612592e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.59098698207309055890188845050700901852e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28146456709550379493162440280752828165e-14),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 2.0174e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.47408470248235665279366712356669210597e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32149712567170349164953101675315481096e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.39806230477579028722350422669222849223e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19665271447867857827798702851111114658e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.06773237553503696884546088197977608676e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41294370314265386485116359052296796357e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74848600628353761723457890991084017928e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52963427970210468265870547940464851481e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.33389244528769791436454176079341120973e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.86702000100897346192018772319301428852e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04192907586200235211623448416582655030e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70804269459077260463819507381406529187e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52665761996923502719902050367236108720e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.01866635015788942430563628065687465455e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.46658865059509532456423012727042498365e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.05806999626031246519161395419216393127e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.37645700309533972676063947195650607935e-26),
        };
        BOOST_MATH_STATIC const RealType Q[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59608758824065179587008165265773042260e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17347162462484266250945490058846704988e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.24511137251392519285309985668265122633e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58497164094526279145784765183039854604e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40787701096334660711443654292041286786e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.34615029717812271556414485397095293077e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.17712219229282308306346195001801048971e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.24578142893420308057222282020407949529e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.23429691331344898578916434987129070432e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41486460551571344910835151948209788541e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.23569151219279213399210115101532416912e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.21438860148387356361258237451828377118e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.46770060692933726695086996017149976796e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.58079984178724940266882149462170567147e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19997796316046571607659704855966005180e-17),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 4.5109e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19610991747326725339429696634365932643e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74646611039453235739153286141429338461e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13331430865337412098234177873337036811e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.58947311195482646360642638791970923726e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79226752074485124923797575635082779509e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73081326043094090549807549513512116319e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05408849431691450650464797109033182773e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75716486666270246158606737499459843698e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.81075133718930099703621109350447306080e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.41318403854345256855350755520072932140e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70220987388883118699419526374266655536e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38711669183547686107032286389030018396e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31300491679098874872172866011372530771e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.99223939265527640018203019269955457925e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.18316957049006338447926554380706108087e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.47298013808154174645356607027685011183e-32),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.42561659771176310412113991024326129105e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83353398513931409985504410958429204317e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.07254121026393428163401481487563215753e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36667170168890854756291846167398225330e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54019749685699795075624204463938596069e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35321766966107368759516431698755077175e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13350720091296144188972188966204719103e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38107118390482863395863404555696613407e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.59267757423034664579822257229473088511e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29549090773392058626428205171445962834e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69922128600755513676564327500993739088e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31337037977667816904491472174578334375e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.28088047429043940293455906253037445768e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.01213369826105495256520034997664473667e-22),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 1.2707e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11172037056341396583040940446061501972e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09383362521204903801686281772843962372e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71440982391172647693486692131238237524e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.01685075759372692173396811575536866699e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36574894913423830789864836789988898151e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.59644999935503505576091023207315968623e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.95573282292603122067959656607163690356e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.10361486103428098366627536344769789255e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80946231978997457068033851007899208222e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.39341134002270945594553624959145830111e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72307967968246649714945553177468010263e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41093409238620968003297675770440189200e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70464969040825495565297719377221881609e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.25341184125872354328990441812668510029e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.54663422572657744572284839697818435372e-36),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.35632539169215377884393376342532721825e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.46975491055790597767445011183622230556e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51806800870130779095309105834725930741e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.07403939022350326847926101278370197017e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66046114012817696416892197044749060854e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.16723371111678357128668916130767948114e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.22972796529973974439855811125888770710e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91073180314665062004869985842402705599e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43753004383633382914827301174981384446e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.77313206526206002175298314351042907499e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32850553089285690900825039331456226080e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.85369976595753971532524294793778805089e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28948021485210224442871255909409155592e-25),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType x_cube = x * x * x;
        RealType t = (boost::math::isnormal)(x_cube) ? 1 / sqrt(x_cube) : 1 / pow(sqrt(x), 3);

        // Rational Approximation
        // Maximum Relative Error: 5.4677e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[7] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99471140200716338969973029967190934238e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.48481268366645066801385595379873318648e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64087860141734943856373451877569284231e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.45555576045996041260191574503331698473e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43290677381328916734673040799990923091e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.63011127597770211743774689830589568544e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.61127812511057623691896118746981066174e0),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.90660291309478542795359451748753358123e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60631500002415936739518466837931659008e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.88655117367497147850617559832966816275e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48350179543067311398059386524702440002e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.18873206560757944356169500452181141647e3),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t;
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 53>& tag) {
    if (x >= 0) {
        return complement ? holtsmark_cdf_plus_imp_prec(x, tag) : 1 - holtsmark_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - holtsmark_cdf_plus_imp_prec(-x, tag) : holtsmark_cdf_plus_imp_prec(-x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 113>& tag) {
    if (x >= 0) {
        return complement ? holtsmark_cdf_plus_imp_prec(x, tag) : 1 - holtsmark_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - holtsmark_cdf_plus_imp_prec(-x, tag) : holtsmark_cdf_plus_imp_prec(-x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_cdf_imp(const holtsmark_distribution<RealType, Policy>& dist, const RealType& x, bool complement) {
    //
    // This calculates the cdf of the Holtsmark distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::cdf(holtsmark<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Holtsmark distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale;

    result = holtsmark_cdf_imp_prec(u, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (ilogb(p) >= -2) {
        RealType t = -log2(ldexp(p, 1));

        // Rational Approximation
        // Maximum Relative Error: 5.8068e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(7.59789769759814986929e-1),
            static_cast<RealType>(1.27515008642985381862e0),
            static_cast<RealType>(4.38619247097275579086e-1),
            static_cast<RealType>(-1.25521537863031799276e-1),
            static_cast<RealType>(-2.58555599127223857177e-2),
            static_cast<RealType>(1.20249932437303932411e-2),
            static_cast<RealType>(-1.36753104188136881229e-3),
            static_cast<RealType>(6.57491277860092595148e-5),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.48696501912062288766e0),
            static_cast<RealType>(2.06239370128871696850e0),
            static_cast<RealType>(5.67577904795053902651e-1),
            static_cast<RealType>(-2.89022828087034733385e-2),
            static_cast<RealType>(-2.17207943286085236479e-2),
            static_cast<RealType>(3.14098307020814954876e-4),
            static_cast<RealType>(3.51448381406676891012e-4),
            static_cast<RealType>(5.71995514606568751522e-5),
        };

        result = t * tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -3) {
        RealType t = -log2(ldexp(p, 2));

        // Rational Approximation
        // Maximum Relative Error: 1.0339e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(3.84521387984759064238e-1),
            static_cast<RealType>(4.15763727809667641126e-1),
            static_cast<RealType>(-1.73610240124046440578e-2),
            static_cast<RealType>(-3.89915764128788049837e-2),
            static_cast<RealType>(1.07252911248451890192e-2),
            static_cast<RealType>(7.62613727089795367882e-4),
            static_cast<RealType>(-3.11382403581073580481e-4),
            static_cast<RealType>(3.93093062843177374871e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.76193897442484823754e-1),
            static_cast<RealType>(3.70953499602257825764e-2),
            static_cast<RealType>(-2.84211795745477605398e-2),
            static_cast<RealType>(2.66146101014551209760e-3),
            static_cast<RealType>(1.85436727973937413751e-3),
            static_cast<RealType>(2.00318687649825430725e-4),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 1.4431e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(4.46943301497773314460e-1),
            static_cast<RealType>(-1.07267614417424412546e-2),
            static_cast<RealType>(-7.21097021064631831756e-2),
            static_cast<RealType>(2.93948745441334193469e-2),
            static_cast<RealType>(-7.33259305010485915480e-4),
            static_cast<RealType>(-1.38660725579083612045e-3),
            static_cast<RealType>(2.95410432808739478857e-4),
            static_cast<RealType>(-2.88688017391292485867e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(-2.72809429017073648893e-2),
            static_cast<RealType>(-7.85526213469762960803e-2),
            static_cast<RealType>(2.41360900478283465241e-2),
            static_cast<RealType>(3.44597797125179611095e-3),
            static_cast<RealType>(-8.65046428689780375806e-4),
            static_cast<RealType>(-1.04147382037315517658e-4),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -6) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 4.8871e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(4.25344469980677332786e-1),
            static_cast<RealType>(3.42055470008289997369e-2),
            static_cast<RealType>(9.33607217644370441642e-2),
            static_cast<RealType>(4.57057092587794346086e-2),
            static_cast<RealType>(1.16149976708336017542e-2),
            static_cast<RealType>(6.40479797962035786337e-3),
            static_cast<RealType>(1.58526153828271386329e-3),
            static_cast<RealType>(3.84032908993313260466e-4),
            static_cast<RealType>(6.98960839033991110525e-5),
            static_cast<RealType>(9.66690587477825432174e-6),
        };
        BOOST_MATH_STATIC const RealType Q[10] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.60044610004497775009e-1),
            static_cast<RealType>(2.41675490962065446592e-1),
            static_cast<RealType>(1.13752642382290596388e-1),
            static_cast<RealType>(4.05058759031434785584e-2),
            static_cast<RealType>(1.59432816225295660111e-2),
            static_cast<RealType>(4.79286678946992027479e-3),
            static_cast<RealType>(1.16048151070154814260e-3),
            static_cast<RealType>(2.01755520912887201472e-4),
            static_cast<RealType>(2.82884561026909054732e-5),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 6));

        // Rational Approximation
        // Maximum Relative Error: 4.8173e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(3.68520435599726877886e-1),
            static_cast<RealType>(8.26682725061327242371e-1),
            static_cast<RealType>(6.85235826889543887309e-1),
            static_cast<RealType>(3.28640408399661746210e-1),
            static_cast<RealType>(9.04801242897407528807e-2),
            static_cast<RealType>(1.57470088502958130451e-2),
            static_cast<RealType>(1.61541023176880542598e-3),
            static_cast<RealType>(9.78919203915954346945e-5),
            static_cast<RealType>(9.71371309261213597491e-8),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.29132755303753682133e0),
            static_cast<RealType>(1.95530118226232968288e0),
            static_cast<RealType>(9.55029685883545321419e-1),
            static_cast<RealType>(2.68254036588585643328e-1),
            static_cast<RealType>(4.61398419640231283164e-2),
            static_cast<RealType>(4.66131710581568432246e-3),
            static_cast<RealType>(2.94491397241310968725e-4),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 6.0376e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(3.48432718168951419458e-1),
            static_cast<RealType>(2.99680703419193973028e-1),
            static_cast<RealType>(1.09531896991852433149e-1),
            static_cast<RealType>(2.28766133215975559897e-2),
            static_cast<RealType>(3.09836969941710802698e-3),
            static_cast<RealType>(2.89346186674853481383e-4),
            static_cast<RealType>(1.96344583080243707169e-5),
            static_cast<RealType>(9.48415601271652569275e-7),
            static_cast<RealType>(3.08821091232356755783e-8),
            static_cast<RealType>(5.58003465656339818416e-10),
        };
        BOOST_MATH_STATIC const RealType Q[10] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.73938978582311007855e-1),
            static_cast<RealType>(3.21771888210250878162e-1),
            static_cast<RealType>(6.70432401844821772827e-2),
            static_cast<RealType>(9.05369648218831664411e-3),
            static_cast<RealType>(8.50098390828726795296e-4),
            static_cast<RealType>(5.73568804840571459050e-5),
            static_cast<RealType>(2.78374120155590875053e-6),
            static_cast<RealType>(9.03427646135263412003e-8),
            static_cast<RealType>(1.63556457120944847882e-9),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 2.2804e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(3.41419813138786920868e-1),
            static_cast<RealType>(1.30219412019722274099e-1),
            static_cast<RealType>(2.36047671342109636195e-2),
            static_cast<RealType>(2.67913051721210953893e-3),
            static_cast<RealType>(2.10896260337301129968e-4),
            static_cast<RealType>(1.19804595761611765179e-5),
            static_cast<RealType>(4.91470756460287578143e-7),
            static_cast<RealType>(1.38299844947707591018e-8),
            static_cast<RealType>(2.25766283556816829070e-10),
            static_cast<RealType>(-8.46510608386806647654e-18),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.81461950831351846380e-1),
            static_cast<RealType>(6.91390438866520696447e-2),
            static_cast<RealType>(7.84798596829449138229e-3),
            static_cast<RealType>(6.17735117400536913546e-4),
            static_cast<RealType>(3.50937328177439258136e-5),
            static_cast<RealType>(1.43958654321452532854e-6),
            static_cast<RealType>(4.05109749922716264456e-8),
            static_cast<RealType>(6.61306247924109415113e-10),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 4.8545e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(3.41392032051575965049e-1),
            static_cast<RealType>(1.53372256183388434238e-1),
            static_cast<RealType>(3.33822240038718319714e-2),
            static_cast<RealType>(4.66328786929735228532e-3),
            static_cast<RealType>(4.67981207864367711082e-4),
            static_cast<RealType>(3.48119463063280710691e-5),
            static_cast<RealType>(2.17755850282052679342e-6),
            static_cast<RealType>(7.40424342670289242177e-8),
            static_cast<RealType>(4.61294046336533026640e-9),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.49255524669251621744e-1),
            static_cast<RealType>(9.77826688966262423974e-2),
            static_cast<RealType>(1.36596271675764346980e-2),
            static_cast<RealType>(1.37080296105355418281e-3),
            static_cast<RealType>(1.01970588303201339768e-4),
            static_cast<RealType>(6.37846903580539445994e-6),
            static_cast<RealType>(2.16883897125962281968e-7),
            static_cast<RealType>(1.35121503608967367232e-8),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else {
        const BOOST_MATH_STATIC_LOCAL_VARIABLE RealType c = ldexp(cbrt(constants::pi<RealType>()), 1);

        RealType p_square = p * p;

        if ((boost::math::isnormal)(p_square)) {
            result = 1 / (cbrt(p_square) * c);
        }
        else if (p > 0) {
            result = 1 / (cbrt(p) * cbrt(p) * c);
        }
        else {
            result = boost::math::numeric_limits<RealType>::infinity();
        }
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (ilogb(p) >= -2) {
        RealType u = -log2(ldexp(p, 1));

        if (u < 0.5) {
            // Rational Approximation
            // Maximum Relative Error: 1.7987e-35
           // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[14] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.59789769759815031687162026655576575384e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.23247138049619855169890925442523844619e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.35351935489348780511227763760731136136e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.17321534695821967609074567968260505604e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30930523792327030433989902919481147250e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.47676800034255152477549544991291837378e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.09952071024064609787697026812259269093e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.65479872964217159571026674930672527880e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.30204907832301876030269224513949605725e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.61038349134944320766567917361933431224e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.17242905696479357297850061918336600969e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43640101589433162893041733511239841220e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.39406616773257816628641556843884616119e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54871597065387376666252643921309051097e-7),
            };
            BOOST_MATH_STATIC const RealType Q[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.06310038178166385607814371094968073940e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06144046990424238286303107360481469219e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17860081295611631017119482265353540470e1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.26319639748358310901277622665331115333e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.25962127567362715217159291513550804588e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65543974081934423010588955830131357921e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.80331848633772107482330422252085368575e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.97426948050874772305317056836660558275e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.10722999873793200671617106731723252507e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.68871255379198546500699434161302033826e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70190278641952708999014435335172772138e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.11562497711461468804693130702653542297e-7),
            };
            // LCOV_EXCL_STOP
            result = u * tools::evaluate_polynomial(P, u) / (tools::evaluate_polynomial(Q, u) * cbrt(p * p));
        }
        else {
            RealType t = u - static_cast <RealType>(0.5);

            // Rational Approximation
            // Maximum Relative Error: 2.5554e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.63490994331899195346399558699533994243e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.68682839419340144322747963938810505658e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.63089084712442063245295709191126453412e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24910510426787025593146475670961782647e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.14005632199839351091767181535761567981e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.88144015238275997284082820907124267240e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.12015895125039876623372795832970536355e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.96386756665254981286292821446749025989e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.82855208595003635135641502084317667629e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.18007513930934295792217002090233670917e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.82563310387467580262182864644541746616e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52830681121195099547078704713089681353e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.91383571211375811878311159248551586411e-8),
            };
            BOOST_MATH_STATIC const RealType Q[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96820655322136936855997114940653763917e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30209571878469737819039455443404070107e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.61235660141139249931521613001554108034e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.31683133997030095798635713869616211197e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.20681979279848555447978496580849290723e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.08958899028812330281115719259773001136e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.02478613175545210977059079339657545008e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.68653479132148912896487809682760117627e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35166554499214836086438565154832646441e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95409975934011596023165394669416595582e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.84312112139729518216217161835365265801e-7),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
        }
    }
    else if (ilogb(p) >= -3) {
        RealType u = -log2(ldexp(p, 2));

        if (u < 0.5) {
            // Rational Approximation
            // Maximum Relative Error: 1.0297e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.84521387984759060262188972210005114936e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70837834325236202821328032137877091515e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53856963029219911450181095566096563059e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.97659091653089105048621336944687224192e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.77726241585387617566937892474685179582e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21657224955483589784473724186837316423e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76357400631206366078287330192525531850e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.45967265853745968166172649261385754061e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.08367654892620484522749804048317330020e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41224530727710207304898458924763411052e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.02908228738160003274584644834000176496e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05702214080592377840761032481067834813e-7),
            };
            BOOST_MATH_STATIC const RealType Q[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33954869248363301881659953529609341564e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.73738626674455393272550888585363920917e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.90708494363306682523722238824373341707e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.49559648492983033200126224112060119905e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.07561158260652000950392950266037061167e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30349651195547682860585068738648645100e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.21766408404123861757376277367204136764e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.22181499366766592894880124261171657846e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.74488053046587079829684775540618210211e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90504597668186854963746384968119788469e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45195198322028676384075318222338781298e-7),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, u) / (tools::evaluate_polynomial(Q, u) * cbrt(p * p));
        }
        else {
            RealType t = u - static_cast <RealType>(0.5);

            // Rational Approximation
            // Maximum Relative Error: 1.3688e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34418795581931891732555950599385666106e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.13006013029934051875748102515422669897e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.27990072710518465265454549585803147529e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.82530244963278920355650323928131927272e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05335741422175616606162502617378682462e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71242678756797136217651369710748524650e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.65147398836785709305701073315614307906e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23912765853731378067295654886575185240e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77861910171412622761254991979036167882e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.11971510714149983297022108523700437739e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23649928279010039670034778778065846828e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.99636080473697209793683863161785312159e-8),
            };
            BOOST_MATH_STATIC const RealType Q[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.95056572065373808001002483348789719155e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.55702988004729812458415992666809422570e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.07586989542594910084052301521098115194e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96831670560124470215505714403486118412e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.86445076378084412691927796983792892534e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.75566285003039738258189045863064261980e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.18557444175572723760508226182075127685e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66667716357950609103712975111660496416e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.70999480357934082364999779023268059131e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.14604868719110256415222454908306045416e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.32724040071094913191419223901752642417e-8),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
        }
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 6.6020e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46943301497773318715008398224877079279e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.85403413700924949902626248891615772650e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.02791895890363892816315784780533893399e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.89147412486638444082129846251261616763e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.93382251168424191872267997181870008850e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.68332196426082871660060467570049113632e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.88720436260994811649162949644253306037e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.34099304204778307050211441936900839075e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.42970601149275611131932131801993030928e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.19329425598839605828710629592687495198e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.48826007216547106568423189194739111033e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.47132934846160946190230821709692067279e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34123780321108493820637601375183345528e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.64549285026064221742294542922996905241e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72723306533295983872420985773212608299e-9),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.23756826160440280076231428938184359865e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.46557011055563840763437682311082689407e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18907861669025579159409035585375166964e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.09998981512549500250715800529896557509e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.09496663758959409482213456915225652712e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.37086325651334206453116588474211557676e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65325780110454655811120026458133145750e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.93435549562125602056160657604473721758e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34967558308250784125219085040752451132e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.73883529653464036447550624641291181317e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.88600727347267778330635397957540267359e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.26681383000234695948685993798733295748e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19871610873353691152255428262732390602e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42468017918888155246438948321084323623e-9),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -5) {
        RealType u = -log2(ldexp(p, 4));

        if (u < 0.5) {
            // Rational Approximation
            // Maximum Relative Error: 5.0596e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.25344469980677353573160570139298422046e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.41915371584999983192100443156935649063e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02829239548689190780023994008688591230e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29283473326959885625548350158197923999e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.01078477165670046284950196047161898687e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.02714892887893367912743194877742997622e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.43133417775367444366548711083157149060e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34782994090554432391320506638030058071e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.06742736859237185836735105245477248882e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.55982601406660341132288721616681417444e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.57770758189194396236862269776507019313e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.29311341249565125992213260043135188072e-8),
            };
            BOOST_MATH_STATIC const RealType Q[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.54021943144355190773797361537886598583e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30965787836836308380896385568728211303e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19314242976592846926644622802257778872e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.84123785238634690769817401191138848504e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.75779029464908805680899310810660326192e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15078294915445673781718097749944059134e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84667183003626452412083824490324913477e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.59521438712225874821007396323337016693e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.90446539427779905568600432145715126083e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.21425779911599424040614866482614099753e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.00972806247654369646317764344373036462e-8),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, u) / (tools::evaluate_polynomial(Q, u) * cbrt(p * p));
        }
        else {
            RealType t = u - static_cast <RealType>(0.5);

            // Rational Approximation
            // Maximum Relative Error: 8.3743e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08071367192424306005939751362206079160e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.94625900993512461462097316785202943274e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55970241156822104458842450713854737857e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.07663066299810473476390199553510422731e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.89859986209620592557993828310690990189e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04002735956724252558290154433164340078e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.28754717941144647796091692241880059406e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19307116062867039608045413276099792797e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.02377178609994923303160815309590928289e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.71739619655097982325716241977619135216e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70229045058419872036870274360537396648e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.90495731447121207951661931979310025968e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23210708203609461650368387780135568863e-8),
            };
            BOOST_MATH_STATIC const RealType Q[11] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.93402256203255215539822867473993726421e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42452702043886045884356307934634512995e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.16981055684612802160174937997247813645e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39560623514414816165791968511612762553e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.26014275897567952035148355055139912545e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.42163967753843746501638925686714935099e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.63605648300801696460942201096159808446e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.55933967787268788177266789383155699064e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.41526208021076709058374666903111908743e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.08505866202670144225100385141263360218e-6),
            };
            // LCOV_EXCL_STOP
            result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
        }
    }
    else if (ilogb(p) >= -6) {
        RealType t = -log2(ldexp(p, 5));

        // Rational Approximation
        // Maximum Relative Error: 2.4734e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.92042979500197776619414802317216082414e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.94742044285563829335663810275331541585e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14525306632578654372860377652983462776e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.88893010132758460781753381176593178775e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08491462791290535107958214106528611951e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61374431854187722720094162894017991926e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11641062509116613779440753514902522337e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12474548036763970495563846370119556004e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.48140831258790372410036499310440980121e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.26913338169355215445128368312197650848e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63109797282729701768942543985418804075e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.55296802973076575732233624155433324402e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72108609713971908723724065216410393928e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.93328436272999507339897246655916666269e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72119240610740992234979508242967886200e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17836139198065889244530078295061548097e-10),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78065342260594920160228973261455037923e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.08575070304822733863613657779515344137e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81185785915044621118680763035984134530e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87597191269586886460326897968559867853e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07903258768761230286548634868645339678e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.88395769450457864233486684232536503140e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05678227243099671420442217017131559055e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.24803207742284923122212652186826674987e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06094715338829793088081672723947647238e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.96454433858093590192363331553516923090e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94509901530299070041475386866323617753e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49001710126540196485963921184736711193e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58899179756014192338509671769986887613e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.06916561094749601736592488829778059190e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 6));

        // Rational Approximation
        // Maximum Relative Error: 1.1570e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.68520435599726860132888599110871216319e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.01076105507184082206031922185510102322e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39912455237662038937400667644545834191e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51088991221663244634723139723207272560e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26465949648856746869050310379379898086e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.37079746226805258449355819952819997723e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.49372033421420312720741838903118544951e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95729572745049276972587492142384353131e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.92794840197452838799536047152725573779e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96897979363475104635129765703613472468e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44138843334474914059035559588791041371e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.78076328055619970057667292651627051391e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04182093251998194244585085400876144351e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04392999917657413659748817212746660436e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.76006125565969084470924344826977844710e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21045181507045010640119572995692565368e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61400097324698003962179537436043636306e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.88084230973635340409728710734906398080e-11),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49319798750825059930589954921919984293e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.90218243410186000622818205955425584848e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.25384789213915993855434876209137054104e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.58563858782064482133038568901836564329e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39112608961600614189971858070197609546e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29192895265168981204927382938872469754e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.66418375973954918346810939649929797237e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01606040038159207768769492693779323748e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.75837675697421536953171865636865644576e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30258315910281295093103384193132807400e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.28333635097670841003561009290200071343e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.04871369296490431325621140782944603554e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05077352164673794093561693258318905067e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.28508157403208548483052311164947568580e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22527248376737724147359908626095469985e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.75479484339716254784610505187249810386e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.39990051830081888581639577552526319577e-11),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 1.1362e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[22] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.48432718168951398420402661878962745094e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.55946442453078865766668586202885528338e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.54912640113904816247923987542554486059e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.60852745978561293262851287627328856197e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93256608166097432329211369307994852513e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.94707001299612588571704157159595918562e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40368387009950846525432054396214443833e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62326983889228773089492130483459202197e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97166112600628615762158757484340724056e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.95053681446806610424931810174198926457e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13613767164027076487881255767029235747e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.46627172639536503825606138995804926378e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.26813757095977946534946955553296696736e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01393212063713249666862633388902006492e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04428602119155661411061942866480445477e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.52977051350929618206095556763031195967e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.63092013964238065197415324341392517794e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.77457116423818347179318334884304764609e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10372468210274291890669895933038762772e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85517152798650696598776156882211719502e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01916800572423194619358228507804954863e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.72241483171311778625855302356391965266e-26),
        };
        BOOST_MATH_STATIC const RealType Q[21] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.18341916009800042837726003154518652168e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19215655980509256344434487727207541208e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34380326549827252189214516628038733750e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.65069135930665131327262366757787760402e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74132905027750048531814627726862962404e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.11184893124573373947875834716323223477e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.68456853089572312034718359282699132364e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16340806625223749486884390838046244494e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45007766006724826837429360471785418874e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50443251593190111677537955057976277305e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30829464745241179175728900376502542995e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.57256894336319418553622695416919409120e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.89973763917908403951538315949652981312e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05834675579622824896206540981508286215e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32721343511724613011656816221169980981e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.77569292346900432492044041866264215291e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39903737668944675386972393000746368518e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.23200083621376582032771041306045737695e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.43557805626692790539354751731913075096e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.91326410956582998375100191562832969140e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 9.1729e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41419813138786928653984591611599949126e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94225020281693988785012368481961427155e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.28967134188573605597955859185818311256e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.16617725083935565014535265818666424029e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13379610773944032381149443514208866162e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06508483032198116332154635763926628153e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.89315471210589177037346413966039863126e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72993906450633221200844495419180873066e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32883391567312244751716481903540505335e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.50998889887280885500990101116973130081e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.23247766687180294767338042555173653249e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.12326475887709255500757383109178584638e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11688319088825228685832870139320733695e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95381542569360703428852622701723193645e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.70730387484749668293167350494151199659e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84243405919322052861165273432136993833e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40860917180131228318146854666419586211e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.85122374200561402546731933480737679849e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.79744248200459077556218062241428072826e-32),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.68930884381361438749954611436694811868e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54944129151720429074748655153760118465e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68493670923968273171437877298940102712e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.32109946297461941811102221103314572340e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.11983030120265263999033828442555862122e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31204888358097171713697195034681853057e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38548604936907265274059071726622071821e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.82158855359673890472124017801768455208e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78564556026252472894386810079914912632e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.46854501887011863360558947087254908412e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67236380279070121978196383998000020645e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.20076045812548485396837897240357026254e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15814123143437217877762088763846289858e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67177999717442465582949551415385496304e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71135001552136641449927514544850663366e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.98449056954034104266783180068258117013e-22),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 1.8330e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41392032051575981622151194498090952488e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32651097995974052731414709779952524875e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51927763729719814565225981452897995722e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.11148082477882981299945196621348531180e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.80457559655975695558885644380771202301e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96223001525552834934139567532649816367e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10625449265784963560596299595289620029e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.14785887121654524328854820350425279893e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00832358736396150660417651391240544392e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.63128732906298604011217701767305935851e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.86148432181465165445355560568442172406e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44088921565424320298916604159745842835e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.95220124898384051195673049864765987092e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50531060529388128674128631193212903032e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06119051130826148039530805693452156757e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19849873960405145967462029876325494393e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66833600176986734600260382043861669021e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07829060832934383885234817363480653925e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21485411177823993142696645934560017341e-40),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.88559444380290379529260819350179144435e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.37942717465159991856146428659881557553e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.11409915376157429952160202733757574026e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.21511733003564236929107862750700281202e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74773232555012468159223116269289241483e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.24042271031862389840796415749527818562e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50790246845873571117791557191071320982e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.88274886666078071130557536971927872847e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94242592538917360235050248151146832636e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45262967284548223426004177385213311949e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30081806380053435857465845326686775489e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62226426496757450797456131921060042081e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40933140159573381494354127717542598424e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.03760580312376891985077265621432029857e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.43980683769941233230954109646012150124e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.88686274782816858372719510890126716148e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07336140055510452905474533727353308321e-25),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 5.9085e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41392031627647840832213878541731833340e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48256908849985263191468999842405689327e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.16515822909144946601084169745484248278e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.42246334265547596187501472291026180697e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.54145961608971551335283437288203286104e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.64840354062369555376354747633807898689e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.38246669464526050793398379055335943951e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29684566081664150074215568847731661446e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.98456331768093420851844051941851740455e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36738267296531031235518935656891979319e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.08128287278026286279504717089979753319e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.38334581618709868951669630969696873534e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.08478537820365448038773095902465198679e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30768169494950935152733510713679558562e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.49254243621461466892836128222648688091e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.17026357413798368802986708112771803774e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.05630817682870951728748696694117980745e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.13881361534205323565985756195674181203e-50),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34271731953273239599863811873205236246e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27133013035186849060586077266046297964e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29542078693828543540010668640353491847e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33027698228265344545932885863767276804e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06868444562964057780556916100143215394e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.97868278672593071061800234869603536243e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79869926850283188735312536038469293739e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.75298857713475428365153491580710497759e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.93449891515741631851202042430818496480e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36715626731277089013724968542144140938e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.98125789528264426869121548546848968670e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78234546049400950521459021508632294206e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83044000387150792643468853129175805308e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.30111486296552039388613073915170671881e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.28628462422858134962149154420358876352e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48108558735886480279744474396456699335e-21),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else {
        const BOOST_MATH_STATIC_LOCAL_VARIABLE RealType c = ldexp(cbrt(constants::pi<RealType>()), 1);

        RealType p_square = p * p;

        if ((boost::math::isnormal)(p_square)) {
            result = 1 / (cbrt(p_square) * c);
        }
        else if (p > 0) {
            result = 1 / (cbrt(p) * cbrt(p) * c);
        }
        else {
            result = boost::math::numeric_limits<RealType>::infinity();
        }
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 53>& tag)
{
    if (p > 0.5) {
        return !complement ? holtsmark_quantile_upper_imp_prec(1 - p, tag) : -holtsmark_quantile_upper_imp_prec(1 - p, tag);
    }

    return complement ? holtsmark_quantile_upper_imp_prec(p, tag) : -holtsmark_quantile_upper_imp_prec(p, tag);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 113>& tag)
{
    if (p > 0.5) {
        return !complement ? holtsmark_quantile_upper_imp_prec(1 - p, tag) : -holtsmark_quantile_upper_imp_prec(1 - p, tag);
    }

    return complement ? holtsmark_quantile_upper_imp_prec(p, tag) : -holtsmark_quantile_upper_imp_prec(p, tag);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_quantile_imp(const holtsmark_distribution<RealType, Policy>& dist, const RealType& p, bool complement)
{
    // This routine implements the quantile for the Holtsmark distribution,
    // the value p may be the probability, or its complement if complement=true.

    constexpr auto function = "boost::math::quantile(holtsmark<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Holtsmark distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * holtsmark_quantile_imp_prec(p, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_entropy_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(2.06944850513462440032);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_entropy_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.0694485051346244003155800384542166381);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType holtsmark_entropy_imp(const holtsmark_distribution<RealType, Policy>& dist)
{
    // This implements the entropy for the Holtsmark distribution,

    constexpr auto function = "boost::math::entropy(holtsmark<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Holtsmark distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = holtsmark_entropy_imp_prec<RealType>(tag_type()) + log(scale);

    return result;
}

} // detail

template <class RealType = double, class Policy = policies::policy<> >
class holtsmark_distribution
{
    public:
    typedef RealType value_type;
    typedef Policy policy_type;

    BOOST_MATH_GPU_ENABLED holtsmark_distribution(RealType l_location = 0, RealType l_scale = 1)
        : mu(l_location), c(l_scale)
    {
        constexpr auto function = "boost::math::holtsmark_distribution<%1%>::holtsmark_distribution";
        RealType result = 0;
        detail::check_location(function, l_location, &result, Policy());
        detail::check_scale(function, l_scale, &result, Policy());
    } // holtsmark_distribution

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

typedef holtsmark_distribution<double> holtsmark;

#ifdef __cpp_deduction_guides
template <class RealType>
holtsmark_distribution(RealType) -> holtsmark_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
holtsmark_distribution(RealType, RealType) -> holtsmark_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const holtsmark_distribution<RealType, Policy>&)
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
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const holtsmark_distribution<RealType, Policy>&)
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
BOOST_MATH_GPU_ENABLED inline RealType pdf(const holtsmark_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::holtsmark_pdf_imp(dist, x);
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const holtsmark_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::holtsmark_cdf_imp(dist, x, false);
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const holtsmark_distribution<RealType, Policy>& dist, const RealType& p)
{
    return detail::holtsmark_quantile_imp(dist, p, false);
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<holtsmark_distribution<RealType, Policy>, RealType>& c)
{
    return detail::holtsmark_cdf_imp(c.dist, c.param, true);
} //  cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<holtsmark_distribution<RealType, Policy>, RealType>& c)
{
    return detail::holtsmark_quantile_imp(c.dist, c.param, true);
} // quantile complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const holtsmark_distribution<RealType, Policy> &dist)
{
    return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const holtsmark_distribution<RealType, Policy>& /*dist*/)
{
    return boost::math::numeric_limits<RealType>::infinity();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const holtsmark_distribution<RealType, Policy>& dist)
{
    return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const holtsmark_distribution<RealType, Policy>& dist)
{
    return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const holtsmark_distribution<RealType, Policy>& /*dist*/)
{
    // There is no skewness:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Holtsmark Distribution has no skewness");

    return policies::raise_domain_error<RealType>(
        "boost::math::skewness(holtsmark<%1%>&)",
        "The Holtsmark distribution does not have a skewness: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy()); // infinity?
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const holtsmark_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Holtsmark Distribution has no kurtosis");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis(holtsmark<%1%>&)",
        "The Holtsmark distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const holtsmark_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis excess:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Holtsmark Distribution has no kurtosis excess");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis_excess(holtsmark<%1%>&)",
        "The Holtsmark distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const holtsmark_distribution<RealType, Policy>& dist)
{
    return detail::holtsmark_entropy_imp(dist);
}

}} // namespaces


#endif // BOOST_STATS_HOLTSMARK_HPP
