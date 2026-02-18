//  Copyright Takuma Yoshimura 2024.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_MAPAIRY_HPP
#define BOOST_STATS_MAPAIRY_HPP

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
#include <boost/math/special_functions/cbrt.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/promotion.hpp>

#ifndef BOOST_MATH_HAS_NVRTC
#include <boost/math/distributions/fwd.hpp>
#include <cmath>
#endif

namespace boost { namespace math {
template <class RealType, class Policy>
class mapairy_distribution;

namespace detail {

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 3.7591e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.97516171847191855610e-1),
            static_cast<RealType>(3.67488253628465083737e-2),
            static_cast<RealType>(-9.73242224038828612673e-4),
            static_cast<RealType>(2.32207514136635673061e-3),
            static_cast<RealType>(5.69067907423210669037e-5),
            static_cast<RealType>(-6.02637387141524535193e-5),
            static_cast<RealType>(1.04960324426666933327e-5),
            static_cast<RealType>(-6.58470237954242016920e-7),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.09464351647314165710e-1),
            static_cast<RealType>(3.66413036246461392316e-1),
            static_cast<RealType>(1.10947882302862241488e-1),
            static_cast<RealType>(2.65928486676817177159e-2),
            static_cast<RealType>(3.75507284977386290874e-3),
            static_cast<RealType>(4.03789594641339005785e-4),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 1.5996e-20
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.06251243013238748252e-1),
            static_cast<RealType>(1.38178831205785069108e-2),
            static_cast<RealType>(4.19280374368049006206e-3),
            static_cast<RealType>(8.54607219684690930289e-4),
            static_cast<RealType>(-7.46881084120928210702e-5),
            static_cast<RealType>(1.47110856483345063335e-5),
            static_cast<RealType>(-1.30090180307471994500e-6),
            static_cast<RealType>(5.24801123304330014713e-8),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.10853683888611687140e-1),
            static_cast<RealType>(3.89361261627717143905e-1),
            static_cast<RealType>(1.15124062681082170577e-1),
            static_cast<RealType>(2.38803416611949902468e-2),
            static_cast<RealType>(3.08616898814509065071e-3),
            static_cast<RealType>(2.43760043942846261876e-4),
            static_cast<RealType>(1.34538901435238836768e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 1.1592e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(5.33842514891989443409e-2),
            static_cast<RealType>(1.23301980674903270971e-2),
            static_cast<RealType>(3.45717831433988631923e-3),
            static_cast<RealType>(3.27034449923176875761e-4),
            static_cast<RealType>(1.20406794831890291348e-5),
            static_cast<RealType>(5.77489170397965604669e-7),
            static_cast<RealType>(-1.15255267205685159063e-7),
            static_cast<RealType>(9.15896323073109992939e-9),
            static_cast<RealType>(-3.14068002815368247985e-10),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.08772985520393226044e-1),
            static_cast<RealType>(4.26418573702560818267e-1),
            static_cast<RealType>(1.22033746594868893316e-1),
            static_cast<RealType>(2.27934009200310243172e-2),
            static_cast<RealType>(2.60658999011198623962e-3),
            static_cast<RealType>(1.54461660261435227768e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 9.2228e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.58950538583133457384e-2),
            static_cast<RealType>(7.47835440063141601948e-3),
            static_cast<RealType>(1.81137244353261478410e-3),
            static_cast<RealType>(2.26935565382135588558e-4),
            static_cast<RealType>(1.43877113825683795505e-5),
            static_cast<RealType>(2.08242747557417233626e-7),
            static_cast<RealType>(-1.54976465724771282989e-9),
            static_cast<RealType>(1.30762989300333026019e-11),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.95505437381674174441e-1),
            static_cast<RealType>(4.58882737262511297099e-1),
            static_cast<RealType>(1.25031310192148865496e-1),
            static_cast<RealType>(2.15727229249904102247e-2),
            static_cast<RealType>(2.33597081566665672569e-3),
            static_cast<RealType>(1.45198998318300328562e-4),
            static_cast<RealType>(3.87962234445835345676e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 1.0257e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(3.22517551525042172428e-3),
            static_cast<RealType>(1.12822806030796339659e-3),
            static_cast<RealType>(1.54489389961322571031e-4),
            static_cast<RealType>(9.28479992527909796427e-6),
            static_cast<RealType>(2.06168350199745832262e-7),
            static_cast<RealType>(9.05110751997021418539e-10),
            static_cast<RealType>(-2.15498112371756202097e-12),
            static_cast<RealType>(6.41838355699777435924e-15),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(6.53390465399680164234e-1),
            static_cast<RealType>(1.82759048270449018482e-1),
            static_cast<RealType>(2.80407546367978533849e-2),
            static_cast<RealType>(2.50853443923476718145e-3),
            static_cast<RealType>(1.27671852825846245421e-4),
            static_cast<RealType>(3.28380135691060279203e-6),
            static_cast<RealType>(3.06545317089055335742e-8),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 6.0510e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(5.82527663232857270992e-4),
            static_cast<RealType>(6.89502117025124630567e-5),
            static_cast<RealType>(2.24909795087265741433e-6),
            static_cast<RealType>(2.18576787334972903790e-8),
            static_cast<RealType>(3.39014723444178274435e-11),
            static_cast<RealType>(-9.74481309265612390297e-15),
            static_cast<RealType>(-1.13308546492906818388e-16),
            static_cast<RealType>(5.32472028720777735712e-19),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.74018883667663396766e-1),
            static_cast<RealType>(2.95901195665990089660e-2),
            static_cast<RealType>(1.57901733512147920251e-3),
            static_cast<RealType>(4.24965124147621236633e-5),
            static_cast<RealType>(5.17522027193205842016e-7),
            static_cast<RealType>(2.00522219276570039934e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 7.3294e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.03264853379349880039e-4),
            static_cast<RealType>(5.35256306644392405447e-6),
            static_cast<RealType>(9.00657716972118816692e-8),
            static_cast<RealType>(5.34913574042209793720e-10),
            static_cast<RealType>(6.70752605041678779380e-13),
            static_cast<RealType>(-5.30089923101856817552e-16),
            static_cast<RealType>(7.28133811621687143754e-19),
            static_cast<RealType>(-7.38047553655951666420e-22),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.29920843258164337377e-1),
            static_cast<RealType>(6.75018577147646502386e-3),
            static_cast<RealType>(1.77694968039695671819e-4),
            static_cast<RealType>(2.46428299911920942946e-6),
            static_cast<RealType>(1.67165053157990942546e-8),
            static_cast<RealType>(4.19496974141131087116e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x * x * x);

        // Rational Approximation
        // Maximum Relative Error: 5.6693e-20
        BOOST_MATH_STATIC const RealType P[5] = {
            static_cast<RealType>(5.98413420602149016910e-1),
            static_cast<RealType>(3.14584075817417883086e-5),
            static_cast<RealType>(1.62977928311793051895e1),
            static_cast<RealType>(-4.12903117172994371875e-4),
            static_cast<RealType>(-1.06404478702135751872e2),
        };
        BOOST_MATH_STATIC const RealType Q[3] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.25696892802060720079e-5),
            static_cast<RealType>(4.03600055498020483920e1),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t / x;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 7.8308e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97516171847191855609649452292217911973e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17531822787252717270400174744562144891e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.85115358761409188259685286269086053296e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18029395189535552537870932989876189597e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77412874842522285996566741532939343827e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.77992070255086842672551073580133785334e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.54573264286260796576738952968288691782e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.94764012694602906119831079380500255557e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.97596258932025712802674070104281981323e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45466169112247379589927514614067756956e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.99760415118300349769641418430273526815e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43150486566834492207695241913522311930e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46130347604880355784938321408765318948e-13),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11845869711743584628289654085905424438e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.80391154854347711297249357734993136108e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.75628443538173255184583966965162835227e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41016303833742742212624596040074202424e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.19142300833563644046500846364541891138e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02421707708633106515934651956262614532e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.03973732602338507411104824853671547615e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.35206168908201402570766383018708660819e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.38602606623008690327520130558254165564e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53740175911385378188372963739884519312e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27513004715414297729539702862351044344e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54510493017251997793679126704007098265e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 3.0723e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06251243013238748252181151646220197947e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.92438638323563234519452281479338921158e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83335793178622701784730867677919844599e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.84159075203218824591724451142550478306e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04213732090358859917896442076931334722e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.72388220651785798237487005913708387756e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.36099324022668533012286817710272936865e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74483270731217433628720245792741986795e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.56461597064783966758904403291149549559e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.28590608939674970691948223694855264817e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.81756745849477762773082030302943341729e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.65915115243311285178083515017249358853e-12),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33250387018216706082200927591739589024e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.71707718560216685629188467984384070512e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.81316277289673837399162302797006618384e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78475951599121894570443981591530879087e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.16167801098514576400689883575304687623e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19167794366424137722223009369062644830e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.20831082064982892777497773490792080382e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27196399162146247210036306870401328410e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79335434374966775903734846875100087590e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30825409557870847168672662674521614782e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97296173230649275943984471731360073540e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.48943057909563158917114503727080517958e-9),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 4.0903e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.33842514891989443409465171800884519331e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.53264053296761245408991932692426094424e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23210520807186629205810670362048049836e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.71104271443590027208545022968625306496e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98781446716778138729774954595209697813e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98895829308616657174932023565302947632e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.25993639218721804661037829873135732687e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.64669776700609853276056375742089715662e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.11846243382610611156151291892877027869e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.74830086064868141326053648144496072795e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07549997153431643849551871367000763445e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.10030596535721362628619523622308581344e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19376016170255697546854583591494809062e-13),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52686177278870816414637961315363468426e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19872083945442288336636376283295310445e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.26633866969676511944680471882188527224e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41261867539396133951024374504099977090e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.18852182132645783844766153200510014113e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70152126044106007357033814742158353948e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.23810508827493234517751339979902448944e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.96161313274648769113605163816403306967e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.06693316156193327359541953619174255726e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.79366356086062616343285660797389238271e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.14585835815353770175366834099001313472e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05314631662369743547568064896403143693e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.90325380271096603676911761784650800378e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.36933359079566550212098911224675011839e-12),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 6.5015e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.58950538583133457383574346194006716984e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.25447644411503971725638816502617490834e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47605882774114100209665040117276675598e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.12224864838900383464124716266085521485e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79164249640537972514574059182421325541e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.89668438166714230032406615413991628135e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.44410389750700463263686630222653669837e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.14788978994687095829140113472609739982e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.79821680629333600844514042061772236495e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.49636960435731257154960798035854124639e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70554745768928821263556963261516872171e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.42293994855343109617040824208078534205e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37599287094703195312894833570340165019e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.35248179978735448062307216459232932068e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.53569375838863862590910010617140120876e-18),
        };
        BOOST_MATH_STATIC const RealType Q[17] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94337325681904859647161946168957959628e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77120402023938328899162557073347121463e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01644685191130734907530007424741314392e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.12479655123720440909164080517207084404e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25556010526357752360439314019567992245e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.96143273204038192262150849394970544022e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.50612932318889495209230176354364299236e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.12160918304376427109905628326638480473e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.47696044292604039527013647985997661762e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.64067652576843720823459199100800335854e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.00745166063635113130434111509648306420e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.05398901239421768403763864060147286105e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05698259572340563109985785513355912114e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19362835269415404005406782719825077472e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15444386779802728200716489787161419304e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02452666470008756043350040893761339083e-16),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 2.0995e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.22517551525042172427941302520759668293e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.86576974828476461442549217748945498966e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18419822818191546598384139622512477000e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98396184944524020019688823190946146641e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.06686400532599396487775148973665625687e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.05680178109228687159829475615095925679e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.17554487015345146749705505971350254902e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14774751685364429557883242232797329274e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33266124509168360207594600356349282805e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76332756800842989348756910429214676252e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.60639771339252642992277508068105926919e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.41859490403554144799385471141184829903e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77177795293424055655391515546880774987e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76106923344461402353501262620681801053e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.68829978902134103249656805130103045021e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42496376687241918803028631991083570963e-26),
        };
        BOOST_MATH_STATIC const RealType Q[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19213376162053391168605415200906099633e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.65578261958732385181558047087365997878e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30046653564394292929001223763106276016e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.48388301731958697028701215596777178117e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.50873786049439122933188684993719288258e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23255654647151798865208394342856435797e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20861791399969402003082323686080041040e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.96882049090731653763684812243275884213e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.98669985741073085290012296575736698103e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.03383311816835346577432387682379226740e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.87320682938150375144724980774245810905e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13573468677076838075146150847170057373e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34526045003716422620879156626237175127e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.35681579117696161282979297336282783473e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92944288060269290125987728528698476197e-18),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 2.0937e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.82527663232857270992129793621400616909e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41696401156754081476312871174198295322e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.42036620449365724707919875710197564857e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.67076745288708619632303078677641380627e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.14278954094278648593125010577441869646e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40092485054621853149602511539550254471e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.17755660009065973828053533035808718033e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.23871371557251644837598540542648782066e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04069998646037977439620128812310273053e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.94055978349016208777803296823455779097e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29866428982892883091537921429389750973e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.06056281963023929277728535486590256573e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57963857545037466186123981516026589992e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.81390322233700529779563477285232205886e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52190981930441828041102818178755246228e-31),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.70564782441895707961338319466546005093e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47770566490107388849474183308889339231e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29364672385303439788399215507370006639e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.37279274083988250795581105436675097881e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72124151284421794872333348562536468054e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.96970247774973902625712414297788402746e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.38395055453444011915661055983937917120e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.19605460410208704830882138883730331113e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76945301389475508747530234950023648137e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33624384932503964160642677987886086890e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01155130710615988897664213446593907596e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03959317021567084067518847978890548086e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78213669817351488671519066803835958715e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75492332026736176991870807903277324902e-22),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 1.5856e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03264853379349880038687006045193401399e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79539964604630527636184900467871907171e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34840369549460790638336121351837912308e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.73087351972154879439617719914590729748e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51775493325347153520115736204545037264e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.60104651860674451546102708885530128768e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.90233449697112559539826150932808197444e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06978852724410115655105118663137681992e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.00399855296672416041126220131900937128e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.18139748830278263202087699889457673035e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43070756487288399784700274808326343543e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70126687893706466023887757573369648552e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.29405234560873665664952418690159194840e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.69069082510020066864633718082941688708e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.33468198065176301137949068264633336529e-37),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.51951069241510130465691156908893803280e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84647597299970149588010858770320631739e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90239396588176334117512714878489376365e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.35551585337774834346900776840459179841e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53375746264539501168763602838029023222e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.42421935941736734247914078641324315900e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.23835501607741697737129504173606231513e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.79603272375172813955236187874231935324e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.44624821303153251954931367754173356213e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.10635081308984534416704147448323126303e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.14627867347129520510628554651739571006e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43792928765659831045040802615903432044e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.79856365207259871336606847582889916798e-25),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType t = 1 / sqrt(x * x * x);

        // Rational Approximation
        // Maximum Relative Error: 3.5081e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[8] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.98413420602149016909919089901572802714e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30303835860684077803651094768293625633e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.89097726237252419724261295392691855545e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.12696604472230480273239741428914666511e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84517621403071494824886152940942995151e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.67577378292168927009421205756730205227e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.16343347002845084264982358165052437094e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.59558963351172885545760841064831356701e3),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.51965956124978480521462518750569617550e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.61700833299761977287211297600922591853e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.94988298508869748383898344668918510537e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.52494213749069142804725453333400335525e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20093079283917759611690534481918040882e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.82564796242972192725215815897475246715e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t / x;
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 3.7525e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.76859868856746781256e-1),
            static_cast<RealType>(1.10489814676299003241e-1),
            static_cast<RealType>(-6.25690643488236678667e-3),
            static_cast<RealType>(-1.17905420222527577236e-3),
            static_cast<RealType>(1.27188963720084274122e-3),
            static_cast<RealType>(-7.20575105181207907889e-5),
            static_cast<RealType>(-2.22575633858411851032e-5),
            static_cast<RealType>(2.94270091008508492304e-6),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.98673671503410894284e-1),
            static_cast<RealType>(3.15907666864554716291e-1),
            static_cast<RealType>(8.34463558393629855977e-2),
            static_cast<RealType>(2.71804643993972494173e-2),
            static_cast<RealType>(3.52187050938036578406e-3),
            static_cast<RealType>(7.03072974279509263844e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 4.0995e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(2.14483832832989822788e-1),
            static_cast<RealType>(3.72789690317712876663e-1),
            static_cast<RealType>(1.86473650057086284496e-1),
            static_cast<RealType>(1.31182724166379598907e-2),
            static_cast<RealType>(-9.00695064809774432392e-3),
            static_cast<RealType>(3.46884420664996747052e-4),
            static_cast<RealType>(4.88651392754189961173e-4),
            static_cast<RealType>(-6.13516242712196835055e-5),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.06478618107122200489e0),
            static_cast<RealType>(4.08809060854459518663e-1),
            static_cast<RealType>(2.66617598099501800866e-1),
            static_cast<RealType>(4.53526315786051807494e-2),
            static_cast<RealType>(2.44078693689626940834e-2),
            static_cast<RealType>(1.52822572478697831870e-3),
            static_cast<RealType>(8.69480001029742502197e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType s = exp(2 * x * x * x / 27) * sqrt(-x);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 2.4768e-18
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(2.74308494787955998605e-1),
                static_cast<RealType>(4.87765991440983416392e-1),
                static_cast<RealType>(3.84524365110270427617e-1),
                static_cast<RealType>(1.77409497505926097339e-1),
                static_cast<RealType>(5.25612864287310961520e-2),
                static_cast<RealType>(1.01528615034079765421e-2),
                static_cast<RealType>(1.20417225696161842090e-3),
                static_cast<RealType>(6.97462693097107007719e-5),
            };
            BOOST_MATH_STATIC const RealType Q[8] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(1.81256903248465876424e0),
                static_cast<RealType>(1.43959302060852067876e0),
                static_cast<RealType>(6.65882284117861804351e-1),
                static_cast<RealType>(1.97537712781845593211e-1),
                static_cast<RealType>(3.81732970028510912201e-2),
                static_cast<RealType>(4.52767489928026542226e-3),
                static_cast<RealType>(2.62240194911920120003e-4),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -8) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 1.5741e-17
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(2.67391547707456587286e-1),
                static_cast<RealType>(3.39319035621314371924e-1),
                static_cast<RealType>(1.85434799940724207230e-1),
                static_cast<RealType>(5.63667456320679857693e-2),
                static_cast<RealType>(1.01231164548944177474e-2),
                static_cast<RealType>(1.02501575174439362864e-3),
                static_cast<RealType>(4.60769537123286016400e-5),
                static_cast<RealType>(-4.92754650783224582641e-13),
            };
            BOOST_MATH_STATIC const RealType Q[7] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(1.27271216837333318516e0),
                static_cast<RealType>(6.96551952883867277759e-1),
                static_cast<RealType>(2.11871363524516350422e-1),
                static_cast<RealType>(3.80622887806509632537e-2),
                static_cast<RealType>(3.85400280812991562328e-3),
                static_cast<RealType>(1.73246593953823694311e-4),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -16) {
            RealType t = -x - 8;

            // Rational Approximation
            // Maximum Relative Error: 4.6579e-17
            BOOST_MATH_STATIC const RealType P[6] = {
                static_cast<RealType>(2.66153901932100301337e-1),
                static_cast<RealType>(1.65767350677458230714e-1),
                static_cast<RealType>(4.19801402197670061146e-2),
                static_cast<RealType>(5.39337995172784579558e-3),
                static_cast<RealType>(3.50811247702301287586e-4),
                static_cast<RealType>(9.21758454778883157515e-6),
            };
            BOOST_MATH_STATIC const RealType Q[6] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(6.23092941554668369107e-1),
                static_cast<RealType>(1.57829914506366827914e-1),
                static_cast<RealType>(2.02787979758160988615e-2),
                static_cast<RealType>(1.31903008994475216511e-3),
                static_cast<RealType>(3.46575870637847438219e-5),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -32) {
            RealType t = -x - 16;

            // Rational Approximation
            // Maximum Relative Error: 5.2014e-17
            BOOST_MATH_STATIC const RealType P[5] = {
                static_cast<RealType>(2.65985830928929730672e-1),
                static_cast<RealType>(7.19655029633308583205e-2),
                static_cast<RealType>(7.26293125679558421946e-3),
                static_cast<RealType>(3.24276402295343802262e-4),
                static_cast<RealType>(5.40508013573989841127e-6),
            };
            BOOST_MATH_STATIC const RealType Q[5] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(2.70578525590448009961e-1),
                static_cast<RealType>(2.73082032706004833847e-2),
                static_cast<RealType>(1.21926059813954504560e-3),
                static_cast<RealType>(2.03227900426552177849e-5),
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
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 5.2870e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76859868856746781256050397658493368372e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13037642242224438972685982606987140111e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.93206268361082760254653961897373271146e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12844418906916902333116398594921450782e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36889326770180267250286619759335338794e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.95272615884641416804001553871108995422e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.53808638264746233799776679481568171506e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.92177790427881393122479399837010657693e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93492737815019893169693306410980499366e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.87510085148730083683110532987841223544e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.28469424017979299382094276157986775969e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83693904015623816528442886551032709693e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.77632857558257155545506847333166147492e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00448215148716947837105979735199471601e-11),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.69069814466926608209872727645156315374e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.89657828158127300370734997707096744077e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62713433978940724622996782534485162816e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.91600878366366974062522408704458777166e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.89144035500328704769924414014440238441e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.35263616916053275381069097012458200491e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49136684724986851824746531490006769036e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65912003138912073317982729161392623277e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.65931144405541620572732754508534372034e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40193555853535182510951061797573338442e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.43625211359756249232841566256877823039e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33207781577559817130740123609636060998e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 1.1977e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14483832832989822788477500521594411868e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75657192307644021285091474845448102656e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40437358470633234235031852091608646844e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66609942512054705023295445270747546208e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.54563774151184610728476161049657676321e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51479057544157089574005315379453615537e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.59853789372610909788599341307719626846e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.76919062715142378209907670793921883406e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.58572738466179822770103948740437237476e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.66618046393835590932491510543557226290e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.26253044828460469263564567571249315188e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11130363073235247786909976446790746902e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.49023728251751416730708805268921994420e-10),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.11919889346080886194925406930280687022e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.99082771425048574611745923487528183522e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99525320878512488641033584061027063035e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.20775109959302182467696345673111724657e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67505804311611026128557007926613964162e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.77854913919309273628222660024596583623e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.91661599559554233157812211199256222756e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.83924945472605861063053622956144354568e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.84286353909650034923710426843028632590e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57737060659799463556626420070111210218e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76047305116625604109657617040360402976e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.86975509621224474718728318687795215895e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71646204381423826495116781730719271111e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30359141441663007574346497273327240071e-9),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType s = exp(2 * x * x * x / 27) * sqrt(-x);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 5.4547e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74308494787955998605105974174143750745e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.56767876568276519015214709629156760546e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23402577465454790961498400214198520261e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09577559351834952074671208183548972395e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.76209118910349927892265971592071407626e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.09368637728788364895148841703533651597e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09003822946777310058789032386408519829e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02362804210869367995322279203786166303e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.67210045349462046966360849113168808620e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17170437120510484976042000272825166724e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62068279517157268391045945672600042900e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72238125522303876741011786930129571553e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33906175951716762094473406744654874848e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.88118741063309731598638313174835288433e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78908322579081615215057968216358892954e-9),
            };
            BOOST_MATH_STATIC const RealType Q[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.15777668058369565739250784347385217839e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.58275582332060589146223977924181161908e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08890987062755381429904193744273374370e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53062680969750921573862970262146744660e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15983695707064161504470525373678920004e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.09120624447001177857109399158887656977e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13566107440776375294261717406754395407e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50716565210262652091950832287627406780e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40417354541359829249609883808591989082e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.09285589734746898623782466689035549135e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.47580156629757526370271002425784456931e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.03479533688660179064728081632921439825e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.58728676819719406366664644282113323077e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72685000369623096389026353785111272994e-9),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -8) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 1.8813e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67391547707456587286086623414017962238e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.69944730920904699720804320295067934914e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.80384452804523880914883464295008532437e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.74832028145199140240423863864148009059e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71728522451977382202061046054643165624e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.91023495084678296967637417245526177858e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57730498044529764612538979048001166775e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31940820074475947691746555183209863058e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.54175805821840981842505041345112198286e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.31350452337838677820161124238784043790e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.52175993144502511705213771924810467309e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.85684239411667243910736588216628677445e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.27124210379062272403030391492854565008e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17645475312219452046348851569796494059e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06306499345515479193219487228315566344e-11),
            };
            BOOST_MATH_STATIC const RealType Q[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13521398369589479131299586715604029947e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17680254721938920978999949995837883884e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40693619288419980101309080614788657638e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.44930162913500531579305526795523256972e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22044272115074113804712893993125987243e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.92745159832354238503828226333417152767e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24766164774700476810039401793119553409e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.08325637569571782180723187639357833929e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74954547353553788519997212700557196088e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82800744682204649977844278025855329390e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.20210992299988298543034791887173754015e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22996819257926038785424888617824130286e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.42340212199922656577943251139931264313e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.75700556749505163188370496864513941614e-11),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -16) {
            RealType t = -x - 8;

            // Rational Approximation
            // Maximum Relative Error: 3.7501e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66153901932100301337118653561328446399e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.52542079386371212946566450189144670788e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.17560936304516198261138159102435548430e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.26792904240601626330507068992045446428e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15418212265160879313643948347460896640e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.05247220687656529725024232836519908641e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.64228534097787946289779529645800775231e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.85634097697132464418223150629017524118e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.49585420710073223183176764488210189671e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.48871040740917898530270248991342594461e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.42577266655992039477272273926475476183e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19214263302271253341410568192952269518e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.36635313919771528255819112450043338510e-12),
            };
            BOOST_MATH_STATIC const RealType Q[13] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32484755553196872705775494679365596205e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.17714315014480774542066462899317631393e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.10789882607024692577764888497624620277e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09821963157449764169644456445120769215e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52354198870000121894280965999352991441e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12133327236256081067100384182795121111e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.20187894923874357333806454001674518211e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69039238999927049119096278882765161803e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.35737444680219098802811205475695127060e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.54403624143647064402264521374546365073e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.24233005893817070145949404296998119469e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.89735152971223120087721392400123727326e-12),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -32) {
            RealType t = -x - 16;

            // Rational Approximation
            // Maximum Relative Error: 9.2696e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65985830928929730672052407058361701971e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.80409998734303497641108024806388734755e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.49286120625421787109350223436127409819e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.89160491404149422833016337592047445082e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16725811789351893632348469796802834008e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.43438517439595919021069131504449842238e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.29058184637190638359623120253986595623e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.03288592271246432030980385908922413497e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.12286831076824535034975676306286388291e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.64563161552001551475186730009447111173e-11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13183856815615371136129883169639301710e-13),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.02405342795439598418033139109649640085e-35),
            };
            BOOST_MATH_STATIC const RealType Q[11] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.78286317363568496229516074305435186276e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06519013547074134846431611115576250187e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.71907733798006110542919988654989891098e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.38874744033460851257697736304200953873e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.54724289412996188575775800547576856966e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98922099980447626797646560786207812928e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.64352676367403443733555974471752023206e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92616898324742524009679754162620171773e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87471345773127482399498877510153906820e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92954174836731254818376396170511443820e-12),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -64) {
            RealType t = -x - 32;

            // Rational Approximation
            // Maximum Relative Error: 2.3524e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[10] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65964563346442080104568381680822923977e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.77958685324702990033291591478515962894e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.56419338083136866686699803771820491401e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.82465178504003399087279098324316458608e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92402911374159755476910533154145918079e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.91224450962405933321548581824712789516e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.84063939469145970625490205194192347630e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.15300528698702940691774461674788639801e-11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.85553643603397817535280932672322232325e-13),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.46207029637607033398822620480584537642e-38),
            };
            BOOST_MATH_STATIC const RealType Q[9] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54906717312241693103173902792310528801e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.84408124581401290943345932332007045483e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81403744024723164669745491417804917709e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.23423244618880845765135047598258754409e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.84697524433421334697753031272973192290e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.94803525968789587050040294764458613062e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68948879514200831687856703804327184420e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07366525547027105672618224029122809899e-12),
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
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53> &tag) {
    if (x >= 0) {
        return mapairy_pdf_plus_imp_prec<RealType>(x, tag);
    }
    else if (x <= 0) {
        return mapairy_pdf_minus_imp_prec<RealType>(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>& tag) {
    if (x >= 0) {
        return mapairy_pdf_plus_imp_prec<RealType>(x, tag);
    }
    else if (x <= 0) {
        return mapairy_pdf_minus_imp_prec<RealType>(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_pdf_imp(const mapairy_distribution<RealType, Policy>& dist, const RealType& x) {
    //
    // This calculates the pdf of the Map-Airy distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::pdf(mapairy<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Map-Airy distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale;

    result = mapairy_pdf_imp_prec(u, tag_type()) / scale;

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 2.9194e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(3.33333333333333333333e-1),
            static_cast<RealType>(7.49532137610545010591e-2),
            static_cast<RealType>(9.25326921848155048716e-3),
            static_cast<RealType>(6.59133092365796208900e-3),
            static_cast<RealType>(-5.21942678326323374113e-4),
            static_cast<RealType>(8.22766804917461941348e-5),
            static_cast<RealType>(-3.97941251650023182117e-6),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.17408156824742736411e-1),
            static_cast<RealType>(3.57041011418415988268e-1),
            static_cast<RealType>(1.04580353775369716002e-1),
            static_cast<RealType>(1.87521616934129432292e-2),
            static_cast<RealType>(2.33232161135637085535e-3),
            static_cast<RealType>(7.31285352607895467310e-5),
        };

        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 3.1531e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.84196970581015939888e-1),
            static_cast<RealType>(-1.19398028299089933853e-3),
            static_cast<RealType>(1.21954054797949597854e-2),
            static_cast<RealType>(-9.37912675685073154845e-4),
            static_cast<RealType>(1.66651954077980453212e-4),
            static_cast<RealType>(-1.33271812303025233648e-5),
            static_cast<RealType>(5.35982226125013888796e-7),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.70352826101668448273e-1),
            static_cast<RealType>(1.98852010141232271304e-1),
            static_cast<RealType>(3.64864882318453496161e-2),
            static_cast<RealType>(4.22173125405065522298e-3),
            static_cast<RealType>(1.20079284386796600356e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 1.8348e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.07409273397524124098e-1),
            static_cast<RealType>(3.83900318969331880402e-2),
            static_cast<RealType>(1.17926652359826576790e-2),
            static_cast<RealType>(1.52181625871479030046e-3),
            static_cast<RealType>(1.50703424417132565662e-4),
            static_cast<RealType>(2.10117959279448106308e-6),
            static_cast<RealType>(1.97360985832285866640e-8),
            static_cast<RealType>(-1.06076300080048408251e-9),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.54435380513870673497e-1),
            static_cast<RealType>(3.66021233157880878411e-1),
            static_cast<RealType>(9.42985570806905160687e-2),
            static_cast<RealType>(1.54122343653998564507e-2),
            static_cast<RealType>(1.49849056258932455548e-3),
            static_cast<RealType>(6.94290406268856211707e-5),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 2.6624e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(4.70720199535228802538e-2),
            static_cast<RealType>(2.67200763833749070079e-2),
            static_cast<RealType>(7.37400551855064729769e-3),
            static_cast<RealType>(1.10592441765001623699e-3),
            static_cast<RealType>(9.15846028547400212588e-5),
            static_cast<RealType>(3.17801522553862136789e-6),
            static_cast<RealType>(2.03102753319827713542e-8),
            static_cast<RealType>(-5.16172854149066643529e-11),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.05317644829451086870e-1),
            static_cast<RealType>(3.73713496637025562492e-1),
            static_cast<RealType>(8.94434672792094976627e-2),
            static_cast<RealType>(1.31846542255347106087e-2),
            static_cast<RealType>(1.16680596342421447100e-3),
            static_cast<RealType>(5.44719256441278863300e-5),
            static_cast<RealType>(8.73131209154185067287e-7),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 2.6243e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.74847564444513000450e-2),
            static_cast<RealType>(6.00209162595027323742e-3),
            static_cast<RealType>(7.86550260761375576075e-4),
            static_cast<RealType>(4.46682547335758521734e-5),
            static_cast<RealType>(9.51329761417139273391e-7),
            static_cast<RealType>(4.10313065114362712333e-9),
            static_cast<RealType>(-9.81286503831545640189e-12),
            static_cast<RealType>(2.98763969872672156104e-14),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.27732094554221674504e-1),
            static_cast<RealType>(1.14330643482604301178e-1),
            static_cast<RealType>(1.27722341942374066265e-2),
            static_cast<RealType>(7.54563340152441778517e-4),
            static_cast<RealType>(2.13377039814057925832e-5),
            static_cast<RealType>(2.09670987094350618690e-7),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 5.4684e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(6.22684103170563193015e-3),
            static_cast<RealType>(1.34714356588780958096e-3),
            static_cast<RealType>(9.51289465377874891896e-5),
            static_cast<RealType>(2.64918464474843134081e-6),
            static_cast<RealType>(2.66703857491046795285e-8),
            static_cast<RealType>(5.42037888457985833156e-11),
            static_cast<RealType>(-6.18017115447736427379e-14),
            static_cast<RealType>(9.11626234402148561268e-17),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.09895694991285975774e-1),
            static_cast<RealType>(3.69874670435930773471e-2),
            static_cast<RealType>(2.15708854325146400153e-3),
            static_cast<RealType>(6.35345408451056881884e-5),
            static_cast<RealType>(8.65722805575670770555e-7),
            static_cast<RealType>(4.03153189557220023202e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 6.5947e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(2.20357145727036120652e-3),
            static_cast<RealType>(1.45412555771401325111e-4),
            static_cast<RealType>(3.27819006009093198652e-6),
            static_cast<RealType>(2.96786786716623870006e-8),
            static_cast<RealType>(9.54192199129339742308e-11),
            static_cast<RealType>(5.71421706870777687254e-14),
            static_cast<RealType>(-1.48321866072033823195e-17),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.12851983233980279746e-1),
            static_cast<RealType>(4.94650928817638043712e-3),
            static_cast<RealType>(1.05447405092956497114e-4),
            static_cast<RealType>(1.11578464291338271178e-6),
            static_cast<RealType>(5.27522295397347842625e-9),
            static_cast<RealType>(7.95786524903707645399e-12),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType x_cube = x * x * x;
        RealType t = static_cast<RealType>((boost::math::isnormal)(x_cube) ? 1 / sqrt(x_cube) : 1 / pow(sqrt(x), 3));

        // Rational Approximation
        // Maximum Relative Error: 6.2709e-17
        BOOST_MATH_STATIC const RealType P[4] = {
            static_cast<RealType>(3.98942280401432677940e-1),
            static_cast<RealType>(2.89752186412133782995e-2),
            static_cast<RealType>(4.67360459917040710474e0),
            static_cast<RealType>(-1.26770824563800250704e-1),
        };
        BOOST_MATH_STATIC const RealType Q[3] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.26301023103568827709e-2),
            static_cast<RealType>(1.60899894281099149848e1),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t;
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_plus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x < 1) {
        // Rational Approximation
        // Maximum Relative Error: 4.7720e-37
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.33333333333333333333333333333333333333e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38519736580901276671338330967060054188e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.07012342772403725079487012557507575976e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70163612228825567572185033570526547856e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.16393313438726572630782132625753922397e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.92141312947853945617138019222992750592e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16513062047959961711747864068554379374e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08850391017085844154857927364247623649e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07060872491334153829857156707699441084e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.56961733740920438026573722084839596926e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.93626747947476815631021107726714283086e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32967164823609209711923411113824666288e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.23420723211833268177898025846064230665e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13807083548358335699029971528179486964e-13),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00810772528427939684296334977783425582e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.24383652800043768524894854013745098654e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64696616559657052516796844068580626381e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.62288747679271039067363492752820355369e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19311779292286492714550084942827207241e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48436879303839576521077892946281025894e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.28665316157256311138787387605249076674e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.36350302380845433472593647100484547496e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.05835458213330488018147374864403662878e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.13919959493955187399856105325181806876e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30960533107704070411766556906543316310e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
    }
    else if (x < 2) {
        RealType t = x - 1;

        // Rational Approximation
        // Maximum Relative Error: 1.6297e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.84196970581015939887507434989936103587e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23864910443500344832158256856064580005e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.72066675347648126090497588433854314742e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.81712740200456564860442639192891089515e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.39091197181834765859741334477680768031e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.03759464781707198959689175957603165395e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15298069568149410830642785868857309358e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18910514301176322829267019223946392192e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.16851691488007921400221017970691227149e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.82031940093536875619655849638573432722e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.30042143299959913519747484877532997335e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.19848671456872291336347012756651759817e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00479393063394570750334218362674723065e-13),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24929390929112144560152115661603117364e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.34853762543033883106055186520573363290e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.73783624941936412984356492130276742707e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23224734370942016023173307854505597524e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.11116448823067697039703254343621931158e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.12490054037308798338231679733816982120e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.38701607014856856812627276285445001885e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10075199231657382435402462616587005087e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.43662615015322880941108094510531477066e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.37981396630189761210639158952200945512e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55820444854396304928946970937054949160e-8),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 4) {
        RealType t = x - 2;

        // Rational Approximation
        // Maximum Relative Error: 2.8103e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07409273397524124098315500450332255837e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.98373054365213259465522536994638631699e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.30851284606709136235419547406278197945e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92686617543233900289721448026065555990e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.18056394312484073294780140350522772329e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07058343449035366484618967963264380933e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71636108080692802684712497501670425230e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13155853034615230731719317488499751231e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.00070273388376168880473457782396672044e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35528857373910910704625837069445190727e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.99897218751541535347315078577172104436e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35092090729912631973050415647154137571e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.72220647682193638971237255396233171508e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.45008884108655511268690849420714428764e-15),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42652074703683973183213296310906006173e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03479786698331153607905223548719296572e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.95556520914240562719970700900964416000e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73127917601685318803655745157828471269e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.63007065833918179119250623000791647836e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.70652732923091039268400927316918354628e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.60880782675229297981880241245777122866e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.09979261868403910549978204036056659380e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12085610111710889118562321318284539217e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59811533082647193392924345081953134304e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.37211668706684650035086116219257276925e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62479830409039340826066305367893543134e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.22039803134898937546371285610102850458e-11),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 8) {
        RealType t = x - 4;

        // Rational Approximation
        // Maximum Relative Error: 7.5930e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.70720199535228802537946089633331273434e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.85220706158476482443562698303252970927e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55090221054465759649629178911450010833e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.70398047783095186291450019612979853708e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11846661331973171721224034349719801691e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83195024406409870789088752469490824640e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.23908312140480103249294791529383548724e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.40765128885655152415228193255890859830e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.14294523267278070539100529759317560119e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.26815059429007745850376987481747435820e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.28142945635159623618312928455133399240e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77079683180868753715374495747422819326e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73710011278079325323578951018770847628e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.70140037580287364298206334732060874507e-16),
        };
        BOOST_MATH_STATIC const RealType Q[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36848014038411798213992770858203510748e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.15373052017549822413011375404872359177e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.92705544967513282963463451395766172671e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19899290805598090502434290420047460406e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74002906913724742582773116667380578990e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.80632456977494447641985312297971970632e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.53381530665983535467406445749348183915e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.86606180756179817016240556949228031340e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.49594666942152749850479792402078560469e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.25231012522695972983928740617341887334e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34987086926725472733984045599487947378e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.58286136970918021841189712851698747417e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.12238357666199366902936267515573231037e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82464168044335183356132979380360583444e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40073718480172265670072434562833527076e-17),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 16) {
        RealType t = x - 8;

        // Rational Approximation
        // Maximum Relative Error: 7.3609e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74847564444513000450056174922427854591e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56842503159303803254436983444304764079e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.48504629497687889354406208309334148575e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62327083507366120871877936416427790391e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.72062210557023828776202679230979309963e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19153025667221102770398900522196418041e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66248482185063262034022017727375829162e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.57390218395059632327421809878050974588e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.45520328522839835737631604118833792570e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.76327978880339919462910339138428389322e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.99700625463451418394515481232159889297e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82943476668680389338853032002472541164e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19415284760817575957617090798914089413e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.17080879333540200065368097274334363537e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.09912208206107606750610288716869139753e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.98451733054622166748935243139556132704e-26),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08148065380582488495702136465010348576e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.42385352331252779422725444021027377277e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66510412535270623169792008730183916611e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.47952712144801508762945315513819636452e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.20703092334999244212988997416711617790e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.71658889250345012472529115544710926154e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.63905601023452497974798277091285373919e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76730409484335386334980429532443217982e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19139408077753398896224794522985050607e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.58025872548387600940275201648443410419e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.11369336267349152895272975096509109414e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.56182954522937999103610817174373785571e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.35452907177197742692545044913125982311e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.23587924912460218189929226092439805175e-17),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 32) {
        RealType t = x - 16;

        // Rational Approximation
        // Maximum Relative Error: 9.7192e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.22684103170563193014558918295924551173e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.55222688816852408105912768186300290291e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.60747505331765587662432023547517953629e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.80463770266821887100086895337451846880e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.19190824169154471000496746227725070963e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40646301571395681364881852739555404287e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15408836734496798091749932018121879724e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.13676779930022341958128426888835497781e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02435098103190516418351075792372986932e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.82018920071479061978244972592746216377e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.26435061215428679536159320644587957335e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.05298407883178633891153989998582851270e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.61156860101928352010449210760843428372e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.02156808288545876198121127510075217184e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.65549196385656698597261688277898043367e-30),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.03426141030409708635168766288764563749e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13808987755928828118915442251025992769e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52253239792170999949444502938290297674e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33720936468171204432499390745432338841e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.08713980159883984886576124842631646880e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.43652846144339754840998823540656399165e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02849693617024492825330133490278326951e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.14110017452008167954262319462808192536e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.01462578814695350559338360744897649915e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73495817568046489613308117490508832084e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47445372925844096612021093857581987132e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08200002287534174751275097848899176785e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.15305756373406702253187385797525419287e-21),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x < 64) {
        RealType t = x - 32;

        // Rational Approximation
        // Maximum Relative Error: 9.7799e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.20357145727036120652264700679701054983e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95712324967981162396595365933255312698e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08619492652809635942960438372427086939e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.37140224583881547818087260161723208444e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83073777522092069988595553041062506001e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.00473542739040742110568810201412321512e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.47447289601822506789553624164171452120e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70913574957198131397471307249294758738e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36538119628489354953085829178695645929e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00763343664814257170332492241110173166e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62297585950798764290583627210836077239e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15780217054514513493147192853488153246e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.31961589164397397724611386366339562789e-28),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.26440207646105117747875545474828367516e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.27872879091838733280518786463281413334e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.34256572873114675776148923422025029494e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13595637397535037957995766856628205747e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33745879863685053883024090247009549434e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41792226523670940279016788831933559977e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.03966147662273388060545199475024100492e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.62177951640260313354050335795080248910e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50650165210517365082118441264513277196e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.48413283257020741389298806290302772976e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16439276222123152748426700489921412654e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24969602890963356175782126478237865639e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.08681155203261739689727004641345513984e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.28282024196484688479115133027874255367e-30),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType x_cube = x * x * x;
        RealType t = (boost::math::isnormal)(x_cube) ? 1 / sqrt(x_cube) : 1 / pow(sqrt(x), 3);

        // Rational Approximation
        // Maximum Relative Error: 3.5865e-37
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[8] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98942280401432677939946059934381868476e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12426566605292130233061857505057433291e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.91574528280329492283287073127040983832e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.69914217884224943794012165979483573091e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30178902028403564086640591437738216288e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96515490341559353794378324810127583810e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44343825578434751356083230369361399507e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.07224810408790092272497403739984510394e2),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.32474438135610721926278423612948794250e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27594461167587027771303292526448542806e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.49207539478843628626934249487055017677e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.75094412095634602055738687636893575929e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.51642534474780515366628648516673270623e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.05977615003758056284424652420774587813e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t) * t;
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 1.6964e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(4.23238998449671083670e-1),
            static_cast<RealType>(4.95353582976475183891e-1),
            static_cast<RealType>(2.45823281826037784270e-1),
            static_cast<RealType>(7.29726507468813920788e-2),
            static_cast<RealType>(1.63332856186819713346e-2),
            static_cast<RealType>(2.82514634871307516142e-3),
            static_cast<RealType>(2.66220579589280704089e-4),
            static_cast<RealType>(3.09442180091323751049e-6),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.16241922223786900600e-1),
            static_cast<RealType>(2.75690727171711638879e-1),
            static_cast<RealType>(7.18707184893542884080e-2),
            static_cast<RealType>(1.87136800286819336797e-2),
            static_cast<RealType>(2.38383441176345054929e-3),
            static_cast<RealType>(3.23509126477812051983e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 5.8303e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(1.62598955251978523175e-1),
            static_cast<RealType>(2.30154661502402196205e-1),
            static_cast<RealType>(1.29233975368291684522e-1),
            static_cast<RealType>(3.80919553916980965587e-2),
            static_cast<RealType>(8.17724414618808505948e-3),
            static_cast<RealType>(1.95816800210481122544e-3),
            static_cast<RealType>(3.35259917978421935141e-4),
            static_cast<RealType>(1.22071311320012805777e-5),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(9.63771793313770952352e-2),
            static_cast<RealType>(2.23602260938227310054e-1),
            static_cast<RealType>(9.21944797677283179038e-3),
            static_cast<RealType>(1.82181136341939651516e-2),
            static_cast<RealType>(1.11216849284965970458e-4),
            static_cast<RealType>(5.57446347676836375810e-4),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType s = exp(2 * x * x * x / 27) / sqrt(-x * x * x);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 3.6017e-18
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(8.31806744221966404520e-1),
                static_cast<RealType>(1.34481067378012055850e0),
                static_cast<RealType>(9.12139427469494995264e-1),
                static_cast<RealType>(3.59706159222491124928e-1),
                static_cast<RealType>(9.48836332725688279299e-2),
                static_cast<RealType>(1.68259594978853951234e-2),
                static_cast<RealType>(1.89700733471520162946e-3),
                static_cast<RealType>(1.13854052826846329787e-4),
            };
            BOOST_MATH_STATIC const RealType Q[8] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(1.29694286517571741097e0),
                static_cast<RealType>(7.99686735441213882518e-1),
                static_cast<RealType>(3.08198207583883597188e-1),
                static_cast<RealType>(7.97230139795658588972e-2),
                static_cast<RealType>(1.40742142048849462162e-2),
                static_cast<RealType>(1.58411440546277691506e-3),
                static_cast<RealType>(9.51560785730564046338e-5),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -8) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 1.3504e-17
            BOOST_MATH_STATIC const RealType P[8] = {
                static_cast<RealType>(1.10294551528734705946e0),
                static_cast<RealType>(1.26696377028973554615e0),
                static_cast<RealType>(6.63115985833429688941e-1),
                static_cast<RealType>(2.06289793717379095832e-1),
                static_cast<RealType>(4.11977615717846276227e-2),
                static_cast<RealType>(5.28620928618550859827e-3),
                static_cast<RealType>(4.04328442334023561279e-4),
                static_cast<RealType>(1.42364413902075896503e-5),
            };
            BOOST_MATH_STATIC const RealType Q[8] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(1.09709853682665798542e0),
                static_cast<RealType>(5.63687797989627787500e-1),
                static_cast<RealType>(1.73604358560002859604e-1),
                static_cast<RealType>(3.44985744385890794044e-2),
                static_cast<RealType>(4.41683993064797272821e-3),
                static_cast<RealType>(3.37834206192286709492e-4),
                static_cast<RealType>(1.18951465786445720729e-5),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -16) {
            RealType t = -x - 8;

            // Rational Approximation
            // Maximum Relative Error: 8.8272e-18
            BOOST_MATH_STATIC const RealType P[7] = {
                static_cast<RealType>(1.18246847255744057280e0),
                static_cast<RealType>(8.41320657699741240497e-1),
                static_cast<RealType>(2.55093097377551881478e-1),
                static_cast<RealType>(4.21261576802732715976e-2),
                static_cast<RealType>(3.98805044659990523312e-3),
                static_cast<RealType>(2.04688276265993954527e-4),
                static_cast<RealType>(4.43354791268634655473e-6),
            };
            BOOST_MATH_STATIC const RealType Q[7] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(7.07103973315808077783e-1),
                static_cast<RealType>(2.13664682181055450396e-1),
                static_cast<RealType>(3.52218225168465984709e-2),
                static_cast<RealType>(3.33218664347896435919e-3),
                static_cast<RealType>(1.71025807471868853268e-4),
                static_cast<RealType>(3.70441884597642042665e-6),
            };

            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -32) {
            RealType t = -x - 16;

            // Rational Approximation
            // Maximum Relative Error: 2.6236e-18
            BOOST_MATH_STATIC const RealType P[6] = {
                static_cast<RealType>(1.19497306481411168356e0),
                static_cast<RealType>(3.90497195765498241356e-1),
                static_cast<RealType>(5.13120330037626853257e-2),
                static_cast<RealType>(3.38574023921119491471e-3),
                static_cast<RealType>(1.12075935888344736993e-4),
                static_cast<RealType>(1.48743616420183584738e-6),
            };
            BOOST_MATH_STATIC const RealType Q[6] = {
                static_cast<RealType>(1.),
                static_cast<RealType>(3.26493785348088598123e-1),
                static_cast<RealType>(4.28813205161574223713e-2),
                static_cast<RealType>(2.82893073845390254969e-3),
                static_cast<RealType>(9.36442365966638579335e-5),
                static_cast<RealType>(1.24281651532469125315e-6),
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
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_minus_imp_prec(const RealType& x, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (x >= -1) {
        RealType t = x + 1;

        // Rational Approximation
        // Maximum Relative Error: 1.0688e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.23238998449671083670041452413316011920e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.14900991369455846775267187236501987891e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.19132787054572299485638029221977944555e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87295743700300806662745209398368996653e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.41994520703802035725356673887766112213e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78782099629586443747968633412271291734e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.05200546520666366552864974572901349343e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.51453477916196630939702866688348310208e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15461354910584918402088506199099270742e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43371674256124419899137414410592359185e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35849788347057186916350200082990102088e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.50359296597872967493549820191745700442e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21838020977580479741299141050400953125e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.46723648594704078875476888175530463986e-12),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.98700317671474659677458220091101276158e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.00405631175818416028878082789095587658e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04189939150805562128632256692765842568e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.03621065280443734565418469521814125946e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.85722257874304617269018116436650330070e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.24191409213079401989695901900760076094e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.64269032641964601932953114106294883156e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19289631274036494326058240677240511431e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41389309719775603006897751176159931569e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42000309062533491486426399210996541477e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.02436961569668743353755318268149636644e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.50130875023154569442119099173406269991e-9),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (x >= -2) {
        RealType t = x + 2;

        // Rational Approximation
        // Maximum Relative Error: 5.1815e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62598955251978523174755901843430986522e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08127698872954954678270473317137288772e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.70144997468767751317246482211703706086e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49486603823046766249106014234315835102e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.07186495389828596786579668258622667573e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98334953533562948674335281457057445421e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.44119017374895211020429143034854620303e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27080759819117162456137826659721634882e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.53892796920597912362370019918933112349e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.30530442651657077016130554430933607143e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.04837779538527662990102489150650534390e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.94354615171320374997141684442120888127e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.30746545799073289786965697800049892311e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41870129065056783732691371215602982173e-9),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.75919235734607601884356783586727272494e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.57656678936617227532275100649989944452e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72617552401870454676736869003112018648e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.59238104942208254162102314312757621047e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06040513359343987972917295603514840777e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.26922840063034349024167652148593396307e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25628506630180107357627955876231943531e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81600387497542714853225329159728694926e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08210973846891324886779444820838563800e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.68632477858150229792523037059221563861e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.43542789104866782087701759971538600076e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70594730517167328271953424328890849790e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.30162314557860623869079601905904538470e-9),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        RealType s = exp(2 * x * x * x / 27) / sqrt(-x * x * x);

        if (x >= -4) {
            RealType t = -x - 2;

            // Rational Approximation
            // Maximum Relative Error: 6.4678e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[16] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.31806744221966404520449104514474066823e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50292887071777664663197915067642779665e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.45140067157601150721516139901304901854e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93227973605511286112712730820664209900e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74259108933048973391560053531348126900e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.77677252890665602191818487592154553094e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71843197238558832510595724454548089268e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.62811778285151415483649897138119310816e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74127763877120261698596916683136227034e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.24832552591462216226478550702845438540e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.93381036027487259940171548523889481080e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02261328789519398745019578211081412570e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.75409238451885381267277435341417474231e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.09526311389365895099871581844304449319e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96371756262605118060185816854433322493e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.88472819535099746216179119978362211227e-10),
            };
            BOOST_MATH_STATIC const RealType Q[16] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.68923525157720774962908922391133419863e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40714902096062779527207435671907059131e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73120596883364361220343183559076165363e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.56331267512666685349409906638266569733e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.80956276267438042306216894159447642323e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34213468750936211385520570062547991332e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.50081600968590616549654807511166706919e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47297208240850928379158677132220746750e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.73392976579560287571141938466202325901e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13858821123741335782695407397784840241e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.04103497389656828224053882850778186433e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.81040189127998139689091455192659191796e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42176283104790992634826374270801565123e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64077137545614380065714904794220228239e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.08139724991616322332901357866680220241e-10),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -8) {
            RealType t = -x - 4;

            // Rational Approximation
            // Maximum Relative Error: 3.5975e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[16] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.10294551528734705945662709421382590676e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.26135857114883288617323671166863478751e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.23504465936865651893193560109437792738e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41598983788870071301270649341678962009e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.43871304031224174103636568402522086316e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.22745720458050596514499383658714367529e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.05976431838299244997805790000128175545e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32087500190238014890030606301748111874e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32754445514451500968404092049839985196e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31866016448921762610690552586049011375e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.80197257671079297305525087998125408939e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.44212088947602969374978384512149432847e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.38857170416924025226203571589937286465e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20239999218390467567339789443070294182e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.93965060142992479149039624149602039394e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.36407983918582149239548869529460234702e-12),
            };
            BOOST_MATH_STATIC const RealType Q[16] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99867960957804580209868321228347067213e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.94236623527818880544030470097296139679e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21644866845440678050425616384656052588e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.48653919287388803523090727546630060490e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88696531788490258477870877792341909659e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.11157356307921406032115084386689196255e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11071149696069503480091810333521267753e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95274844731679437274609760294652465905e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.77971280158253322431071249000491659536e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.18092258028773913076132483326275839950e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.87773088535057996947643657676843842076e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.99609277781492599950063871899582711550e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00465660598924300723542908245498229301e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.29174652982710100418474261697035968379e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.31746082236506935340972706820707017875e-12),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -16) {
            RealType t = -x - 8;

            // Rational Approximation
            // Maximum Relative Error: 2.6792e-35
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[15] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18246847255744057280356900905660312795e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77955886026107125189834586992142580148e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24948425302263641813107623611637262126e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.42593662659560333324287312162818766556e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62714138002904073145045478360748042164e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.56008984285541289474850396553042124777e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84858048549330525583286373950733005244e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.30578460156038467943968005946143934751e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.93974868941529258700281962314167648967e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.95086664204515648622431580749060079100e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57811968176644056830002158465591081929e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27814751838906948007289825582251221538e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06762893426725920159998333647896590440e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15388861641344998301210173677051088515e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.83956842740198388242245209024484381888e-29),
            };
            BOOST_MATH_STATIC const RealType Q[14] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50056124032615852703112365430040751173e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05112559537845833793684655693572118348e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.55609497026127521043140534271852131858e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36430652394614121156238070755223942728e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.98167215021940993097697777547641188697e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.89418831310297071347013983522734394061e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.10980717618462843498917526227524790487e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.80119618735773019675212434416594954984e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13748676086657187580746476165248613583e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.15424395860921826755718081823964568760e-7),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.75228896859720124469916340725146705309e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.72759238269315282789451836388878919387e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79966603543593799412565926418879689461e-12),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -32) {
            RealType t = -x - 16;

            // Rational Approximation
            // Maximum Relative Error: 2.1744e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19497306481411168355692832231058399132e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.15593166681833539521403250736661720488e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.54020260738207743315755235213180652303e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.76467972857585566189917087631621063058e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.97922795572348613358915532172847895070e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.26998967192207380100354278434037095729e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.32827180395699855050424881575240362199e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50587178182571637802022891868380669565e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.78252548290929962236994183546354358888e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01519297007773622283120166415145520855e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29602226691665918537895803270497291716e-11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.53666531487585211574942518181922132884e-14),
            };
            BOOST_MATH_STATIC const RealType Q[12] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.82230649578130958108098853863277631065e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.12412482738973738235656376802445565005e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98320116955422615960870363549721494683e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.99756654189000467678223166815845628725e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40414942475279981724792023159180203408e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78118445466942812088955228016254912391e-5),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.25827002637577602812624580692342616301e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.99604467789028963216078448884632489822e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.48237134334492289420105516726562561260e-10),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08288201960155447241423587030002372229e-11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.29720612489952110448407063201146274502e-14),
            };
            // LCOV_EXCL_STOP
            result = s * tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
        }
        else if (x >= -64) {
            RealType t = -x - 32;

            // Rational Approximation
            // Maximum Relative Error: 3.4699e-36
            // LCOV_EXCL_START
            BOOST_MATH_STATIC const RealType P[10] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.19659414007358083585943280640656311534e0),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36969140730640253987817932335415532846e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21946928005759888612066397569236165853e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08341720579009422518863704766395201498e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44908614491286780138818989614277172709e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.54172482866925057749338312942859761961e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.49281630950104861570255344237175124548e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.27586759709416364899010676712546639820e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00054716479138657682306851175059678989e-11),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.48798342894235412426464893852098239746e-14),
            };
            BOOST_MATH_STATIC const RealType Q[10] = {
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.81588658109851219975949691772676519853e-1),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.52583331848892383968186924120872369151e-2),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.57644670426430994363913234422346706991e-3),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21080164634428298820141591419770346977e-4),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.79484074949580980980061103238709314326e-6),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.93167250146504946763386377338487557826e-8),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06604193118724797924138056151582242604e-9),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.35999937789324222934257460080153249173e-12),
                BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.91435929481043135336094426837156247599e-14),
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
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 53>& tag) {
    if (x >= 0) {
        return complement ? mapairy_cdf_plus_imp_prec(x, tag) : 1 - mapairy_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - mapairy_cdf_minus_imp_prec(x, tag) : mapairy_cdf_minus_imp_prec(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_imp_prec(const RealType& x, bool complement, const boost::math::integral_constant<int, 113>& tag) {
    if (x >= 0) {
        return complement ? mapairy_cdf_plus_imp_prec(x, tag) : 1 - mapairy_cdf_plus_imp_prec(x, tag);
    }
    else if (x <= 0) {
        return complement ? 1 - mapairy_cdf_minus_imp_prec(x, tag) : mapairy_cdf_minus_imp_prec(x, tag);
    }
    else {
        return boost::math::numeric_limits<RealType>::quiet_NaN();
    }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_cdf_imp(const mapairy_distribution<RealType, Policy>& dist, const RealType& x, bool complement) {
    //
    // This calculates the cdf of the Map-Airy distribution and/or its complement.
    //

    BOOST_MATH_STD_USING // for ADL of std functions
    constexpr auto function = "boost::math::cdf(mapairy<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Map-Airy distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    RealType u = (x - location) / scale;

    result = mapairy_cdf_imp_prec(u, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_lower_imp_prec(const RealType& p, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.375) {
        RealType t = p - static_cast <RealType>(0.375);

        // Rational Approximation
        // Maximum Relative Error: 1.5488e-18
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(-1.17326074020471664075e0),
            static_cast<RealType>(1.51461298154568349598e0),
            static_cast<RealType>(1.19979368094343490487e1),
            static_cast<RealType>(-5.94882121521324108164e0),
            static_cast<RealType>(-2.20619749774447254528e1),
            static_cast<RealType>(7.17766543775229176131e0),
            static_cast<RealType>(4.79284243496552841508e0),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.76268072706610602584e0),
            static_cast<RealType>(-4.88492535243404839734e0),
            static_cast<RealType>(-5.67524172432687656881e0),
            static_cast<RealType>(6.83327389947131710596e0),
            static_cast<RealType>(2.91338085774159042709e0),
            static_cast<RealType>(-1.41108918944159283950e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Relative Error: 7.5181e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(-1.63281240925531302762e0),
            static_cast<RealType>(-4.92351310795930780147e0),
            static_cast<RealType>(1.43448529253101759409e1),
            static_cast<RealType>(3.33182629948094299473e1),
            static_cast<RealType>(-3.06679026539368582747e1),
            static_cast<RealType>(-2.87298447423841965301e1),
            static_cast<RealType>(1.31575930750093554120e1),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.38761652244702318296e0),
            static_cast<RealType>(2.40932080746189543284e0),
            static_cast<RealType>(-1.69465870062123632126e1),
            static_cast<RealType>(-6.39998944283654848809e0),
            static_cast<RealType>(1.27168434054332272391e1),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Relative Error: 2.3028e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-2.18765177572396469657e0),
            static_cast<RealType>(-3.65752788934974426531e1),
            static_cast<RealType>(-1.81144810822028903904e2),
            static_cast<RealType>(-1.22434531262312950288e2),
            static_cast<RealType>(8.99451018491165823831e2),
            static_cast<RealType>(9.11333307522308410858e2),
            static_cast<RealType>(-8.76285742384616909177e2),
            static_cast<RealType>(-2.33786726970025938837e2),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.91797638291395345792e1),
            static_cast<RealType>(1.24293724082506952768e2),
            static_cast<RealType>(2.82393116012902543276e2),
            static_cast<RealType>(-1.80472369158936285558e1),
            static_cast<RealType>(-5.31764390192922827093e2),
            static_cast<RealType>(-5.60586018315854885788e1),
            static_cast<RealType>(1.21284324755968033098e2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 6.1147e-18
        BOOST_MATH_STATIC const RealType P[6] = {
            static_cast<RealType>(-2.18765177572396470773e0),
            static_cast<RealType>(-2.19887766409334094428e0),
            static_cast<RealType>(-7.77080107207360785208e-1),
            static_cast<RealType>(-1.15551765136654549650e-1),
            static_cast<RealType>(-6.64711321022529990367e-3),
            static_cast<RealType>(-9.74212491048543799073e-5),
        };
        BOOST_MATH_STATIC const RealType Q[6] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.91919722132624625590e-1),
            static_cast<RealType>(2.17415447268626558639e-1),
            static_cast<RealType>(2.41474762519410575392e-2),
            static_cast<RealType>(9.41084107182696904714e-4),
            static_cast<RealType>(6.65754108797614202364e-6),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 2.0508e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-2.59822399410385085335e0),
            static_cast<RealType>(-2.24306757759003016244e0),
            static_cast<RealType>(-7.36208578161752060979e-1),
            static_cast<RealType>(-1.15130762650287391576e-1),
            static_cast<RealType>(-8.77652386123688618995e-3),
            static_cast<RealType>(-2.96358888256575251437e-4),
            static_cast<RealType>(-3.33661282483762192446e-6),
            static_cast<RealType>(-4.19292241201527861927e-9),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.23065798041556418844e-1),
            static_cast<RealType>(1.96731305131315877264e-1),
            static_cast<RealType>(2.49952034298034383781e-2),
            static_cast<RealType>(1.49149568322111062242e-3),
            static_cast<RealType>(3.66010398525593921460e-5),
            static_cast<RealType>(2.46857713549279930857e-7),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 2.1997e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-3.67354365380697580447e0),
            static_cast<RealType>(-1.52181685844845957618e0),
            static_cast<RealType>(-2.40883948836320845233e-1),
            static_cast<RealType>(-1.82424079258401987512e-2),
            static_cast<RealType>(-6.75844978572417703979e-4),
            static_cast<RealType>(-1.11273358356809152121e-5),
            static_cast<RealType>(-6.12797605223700996671e-8),
            static_cast<RealType>(-3.78061321691170114390e-11),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(3.57770840766081587688e-1),
            static_cast<RealType>(4.81290550545412209056e-2),
            static_cast<RealType>(3.02079969075162071807e-3),
            static_cast<RealType>(8.89589626547135423615e-5),
            static_cast<RealType>(1.07618717290978464257e-6),
            static_cast<RealType>(3.57383804712249921193e-9),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 2.4331e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-4.92187819510636697128e0),
            static_cast<RealType>(-9.94924018698264727979e-1),
            static_cast<RealType>(-7.69914962772717316098e-2),
            static_cast<RealType>(-2.85558010159310978248e-3),
            static_cast<RealType>(-5.19022578720207406789e-5),
            static_cast<RealType>(-4.19975546950263453259e-7),
            static_cast<RealType>(-1.13886013623971006760e-9),
            static_cast<RealType>(-3.46758191090170732580e-13),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.77270673840643360017e-1),
            static_cast<RealType>(1.18099604045834575786e-2),
            static_cast<RealType>(3.66889581757166584963e-4),
            static_cast<RealType>(5.34484782554469770841e-6),
            static_cast<RealType>(3.19694601727035291809e-8),
            static_cast<RealType>(5.24649233511937214948e-11),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 2.7742e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-6.41443550638291133784e0),
            static_cast<RealType>(-6.38369359780748328332e-1),
            static_cast<RealType>(-2.43420704406734621618e-2),
            static_cast<RealType>(-4.45274771094277987075e-4),
            static_cast<RealType>(-3.99529078051262843241e-6),
            static_cast<RealType>(-1.59758677464731620413e-8),
            static_cast<RealType>(-2.14338367751477432622e-11),
            static_cast<RealType>(-3.23343844538964435927e-15),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.79845511272943785289e-2),
            static_cast<RealType>(2.90839059356197474893e-3),
            static_cast<RealType>(4.48172838083912540123e-5),
            static_cast<RealType>(3.23770691025690100895e-7),
            static_cast<RealType>(9.60156044379859908674e-10),
            static_cast<RealType>(7.81134095049301988435e-13),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 3.2451e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-8.23500806363233610938e0),
            static_cast<RealType>(-4.05652655284908839003e-1),
            static_cast<RealType>(-7.65978833819859622912e-3),
            static_cast<RealType>(-6.94194676058731901672e-5),
            static_cast<RealType>(-3.08771646223818451436e-7),
            static_cast<RealType>(-6.12443207313641110962e-10),
            static_cast<RealType>(-4.07882839359528825925e-13),
            static_cast<RealType>(-3.05720104049292610799e-17),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.37395212065018405474e-2),
            static_cast<RealType>(7.18654254114820140590e-4),
            static_cast<RealType>(5.50371158026951899491e-6),
            static_cast<RealType>(1.97583864365011234715e-8),
            static_cast<RealType>(2.91169706068202431036e-11),
            static_cast<RealType>(1.17716830382540977039e-14),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -256) {
        RealType t = -log2(ldexp(p, 128));

        // Rational Approximation
        // Maximum Relative Error: 3.8732e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-1.04845570631944023913e1),
            static_cast<RealType>(-2.56502856165700644836e-1),
            static_cast<RealType>(-2.40615394566347412600e-3),
            static_cast<RealType>(-1.08364601171893250764e-5),
            static_cast<RealType>(-2.39603255140022514289e-8),
            static_cast<RealType>(-2.36344017673944676435e-11),
            static_cast<RealType>(-7.83146284114485675414e-15),
            static_cast<RealType>(-2.92218240202835807955e-19),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.17740414929742679904e-2),
            static_cast<RealType>(1.78084231709097280884e-4),
            static_cast<RealType>(6.78870668961146609668e-7),
            static_cast<RealType>(1.21313439060489363960e-9),
            static_cast<RealType>(8.89917934953781122884e-13),
            static_cast<RealType>(1.79115540847944524599e-16),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -512) {
        RealType t = -log2(ldexp(p, 256));

        // Rational Approximation
        // Maximum Relative Error: 4.6946e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-1.32865827226175698181e1),
            static_cast<RealType>(-1.61802434199627472010e-1),
            static_cast<RealType>(-7.55642602577784211259e-4),
            static_cast<RealType>(-1.69457608092375302291e-6),
            static_cast<RealType>(-1.86612389867293722402e-9),
            static_cast<RealType>(-9.17015770142364635163e-13),
            static_cast<RealType>(-1.51422473889348610974e-16),
            static_cast<RealType>(-2.81661279271583206526e-21),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(1.08518414679241420227e-2),
            static_cast<RealType>(4.42335224797004486239e-5),
            static_cast<RealType>(8.40387821972524402121e-8),
            static_cast<RealType>(7.48486746424527560620e-11),
            static_cast<RealType>(2.73676810622938942041e-14),
            static_cast<RealType>(2.74588200481263214866e-18),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -1024) {
        RealType t = -log2(ldexp(p, 512));

        // Rational Approximation
        // Maximum Relative Error: 5.7586e-17
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(-1.67937186583822375593e1),
            static_cast<RealType>(-1.01958138247797604098e-1),
            static_cast<RealType>(-2.37409774265951876695e-4),
            static_cast<RealType>(-2.65483321307104128810e-7),
            static_cast<RealType>(-1.45803536947907216594e-10),
            static_cast<RealType>(-3.57375116523338994342e-14),
            static_cast<RealType>(-2.94401318006358820268e-18),
            static_cast<RealType>(-2.73260616170245224789e-23),
        };
        BOOST_MATH_STATIC const RealType Q[7] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.41357843707822974161e-3),
            static_cast<RealType>(1.10082540037527566536e-5),
            static_cast<RealType>(1.04338126042963003178e-8),
            static_cast<RealType>(4.63619608458569600346e-12),
            static_cast<RealType>(8.45781310395535984099e-16),
            static_cast<RealType>(4.23432554226506409568e-20),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else {
        result = -boost::math::numeric_limits<RealType>::infinity();
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_lower_imp_prec(const RealType& p, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.4375) {
        RealType t = p - static_cast <RealType>(0.4375);

        // Rational Approximation
        // Maximum Relative Error: 4.2901e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[10] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.48344198262277235851026749871350753173e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.18249834490570496537675012473572546187e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20191368895639224466285643454767208281e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.88388953372157636908236843798588258539e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.59477796311326067051769635858472572709e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.88799146700484120781026039104654730797e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.15708831983930955608517858269193800412e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.01389336086567891484877690859385409842e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.16683694881010716925933071465043323946e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.04356966421177683585461937085598186805e1),
        };
        BOOST_MATH_STATIC const RealType Q[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.75444066345435020043849341970820565274e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.95105673975812427406540024601734210826e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.20381124524894051002242766595737443257e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.48370658634610329590305283520183480026e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.52213602242009530270284305006282822794e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.91028722773916006242187843372209197705e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.76130245344411748356977700519731978720e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30834721900169773543149860814908904224e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.37863084758381651884340710544840951679e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.46880981703613838666108664771931239970e0),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.375) {
        RealType t = p - static_cast <RealType>(0.375);

        // Rational Approximation
        // Maximum Relative Error: 2.8433e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.17326074020471664204142312429732771661e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.23412560010002723970559712941124583385e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83665111310407767293290698145068379130e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.38459476870110655357485107373883403534e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.28751995328228442619291346921055105808e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.31663592034507247231393516167247241037e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.13629333446941271397790762651183997586e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.80674058829101054663235662701823250421e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.53226182094253065852552393446365315319e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.14713948941614711932063053969010219677e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.62979741122708118776725634304028246971e0),
        };
        BOOST_MATH_STATIC const RealType Q[10] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.10550060286464202595779024353437346419e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.15893254630199957990897452211066782021e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.58964066823516762861256609311733069353e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.73352515261971291505497909338586980605e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.64737859211974163695241658186141083513e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.79137714768236053008878088337762178011e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.71851514659301019977259792564627124877e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.37210093190088984630526671624779422232e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06793750951779308425209267821815264457e1),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Relative Error: 5.9072e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.63281240925531315038207673147576291783e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.72733898245766165408685147762489513406e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.48666841594842113608962500631836790675e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.38711336213357101067420572773139678571e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.19536066931882831915715343914510496760e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70911330354860558400876197129777829223e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.46138758869321272507090399082047865434e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.42653825421465476333482312795245170700e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.68040069633027903153088221686431049116e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.63017854949929226947577854802720988740e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.57362168966659376351959631576588023516e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.48386631313725080746815524770260451090e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.03293129698111279047104766073456412318e1),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29511778027351594854005887702013466376e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.66155745848864270109281703659789474448e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.25628362783798417463294553777015370203e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.93162726153946899828589402901015679821e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.51582398149308841534372162372276623400e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55512400116480727630652657714109740448e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.11949742749256615588470329024257669470e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.28090154738508864776480712360731968283e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.44307717248171941824014971579691790721e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.33595130666758203099507440236958725924e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.76156378002782668186592725145930755636e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.70446647862725980215630194019740606935e0),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Relative Error: 9.9092e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.18765177572396470161180571018467019660e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.16991481696416567893311341049825218287e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.25497491118598918048058751362064598010e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.00259915194411316966036757165146681474e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.20350803730836873687692010728689867756e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.75441278117456011071671644613599089820e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.18844967505497310645091822621081741562e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.84771867850847121528386231811667556346e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.78112436422992766542256241612018834150e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.82617957794395420193751983521804760378e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.77260227244465268981198741998181334875e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.61918290776044518321561351472048170874e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.10309368217936941851272359946015001037e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.68917274690585744147547352309416731690e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.96914697182030973966321601422851730384e4),
        };
        BOOST_MATH_STATIC const RealType Q[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.98063872144867195074924232601423646991e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.65911346382127464683324945513128779971e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02451223307009464199634546540152067898e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.12662676019712475980273334769644047369e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.77637305574675655673572303462430608857e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.65900204382557635710258095712789133767e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.61649498173261886264315880770449636676e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.29867325788870863753779283018061152414e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.31375646045453788071216808289409712455e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.91053361331987954531162452163243245571e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.30917504462260061766689326034981496723e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.95779171217851232246427282884386844906e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.07234815204245866330282860014624832711e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21353269292094971546479026200435095695e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 3.9653e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.18765177572396470161180571018467025793e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.94718878144788678915739777385667044494e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.01760622104142726407095836139719210570e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.27585388152893587017559610649258643106e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.26494992809545184138230791849722703452e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.19962820888633928632710264415572027960e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.10249328404135065767844288193594496173e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.94733898567966142295343935527193851633e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.10350856810280579594619259121755788797e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.23852908701349250480831167491889740823e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.61008160195204632725691076288641221707e-10),
        };
        BOOST_MATH_STATIC const RealType Q[11] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59109221235949005113322202980300291082e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07119192591092503378838510797916225920e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97313678065269932508447079892684333156e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.85743007214453288049750256975889151838e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21504290861269099964963866156493713716e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00880286431998922077891394903879512720e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.78806057460269900288838437267359072282e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15390284861815831078443996558014864171e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09877166004503937701692216421704042881e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.01544823753120969225271131241177003165e-11),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 6.7872e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.59822399410385083283727681965013517187e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.30676107401101401386206170152508285083e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.14999782004905950712290914501961213222e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.33941786334132569296061539102765384372e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.67220197146826865151515598496049341734e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.80553562310354708419148501358813792297e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.69365728863553037992854715314245847166e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.66571611322125393586164383361858996769e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.60112572827002965346926427208336150737e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.82368436189138780270310776927920829805e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.53112953085778860983110669544602606343e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.47363651755817041383574210879856850108e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.56959356177318833325064543662295824581e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.16259632533790175212174199386945953139e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.51102540845397821195190063256442894688e-18),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.51733661576324699382035973518172469602e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.01435607980568082538278883569729476204e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.93274478117803447229185270863587786287e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.79695020433868416640781960667235896490e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64179255575983014759473815411232853821e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.88411076775875459504324039642698874213e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47924359965537384942568979646011627522e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.80915944234873904741224397674033111178e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.67530059257263142305079790717032648103e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.59778634436027309520742387952911132163e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.45772841855129835242992919296397034883e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.40356170422999016176996652783329671363e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.31127037096552892520323998665757802862e-16),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 7.6679e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.67354365380697578246790709817724831418e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.98681625368564496421038758088088788795e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.07310677200866436332040539417232353673e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.24429836496456823308103613923194387860e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.02985399190977938381625172095575017346e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.76782196972948240235456454778537838123e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.74449002785643398012450514191731166637e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.59599894044461264694825875303563328822e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.25326249969126313897827328136779597159e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.41714701672521323699602179629851858792e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.43197925999694667433180053594831915164e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.84385683045691486021670951412023644809e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.34484740544060627138216383389282372695e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.69717198111130468014375331439613690658e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.54464319402577486444841981479085908190e-22),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.56568169258142086426383908572004868200e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.52144703581828715720555168887427064424e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.87268567039210776554113754014224005739e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.04907480436107533324385788289629535047e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.05063218951887755000006493061952858632e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.88707869862323507236241669797692957827e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12876335369047582931728838551780783006e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.96657547014655679104867167083078285517e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.05799908632250375607393338998205481867e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.26543514080500125624753383852230043206e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02102501711063497529014782040893679505e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.91577723105757841509716090936343311518e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.13824283239122976911704652025193325941e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 8.6323e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.92187819510636694694450607724165689649e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.97763858433958798529675258052376253402e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.51315096979196319830162238831654165509e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.63462872759639470195664268077372442947e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.42843455093529447002457295994721102683e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.09839699044798405685726233251577671229e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.42918585783783247934440868680748693033e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.39927851612709063686969934343256912856e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.08811978806833318962489621493456773153e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.05790315329766847040100971840989677130e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.45479905512618918078640786598987515012e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.26725031195260767541308143946590024995e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.93921283405116349017396651678347306610e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.46989755992471397407520449698676945629e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.84098833615882764168840211033822541979e-26),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76933186134913044021577076301874622292e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.25794095712489484166470336696962749356e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.02367257449622569623375613790000874499e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.72422948147678152291655497515112236849e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.54841786738844966378222670550160421679e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40690766057741905625149753730480294357e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.57722740290698689456097239435447030950e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12202417878861628530322715231540006386e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.52053248659561670052645118655279630611e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.93156242508301535729374373870786335203e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40767922796889118151219837068812449420e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83472702205100162081157644960354192597e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.07575352729625387198150555665307193572e-24),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 9.8799e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.41443550638291131009585191506467028820e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.27887396494095561461365753577189254063e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.12752633002089885479040650194288302309e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.79073355729340068320968150408320521772e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.92104558368762745368896313096467732214e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.31521786147036353766882145733055166296e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.69209485282228501578601478546441260206e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.17440286764020598209076590905417295956e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.24323354132100896221825450145208350291e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.53395893728239001663242998169841168859e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.68737623513761169307963299679882178852e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.34293243065056704017609034428511365032e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.16077944943653158589001897848048630079e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.62195354664281744711336666974567406606e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.34400562353945663460416286570988365992e-30),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.87838171063584994806998206766890809367e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.55395225505766120991458653457272783334e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.45282436421694718640472363162421055686e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29570159526914149727970023744213510609e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.75502365627517415214497786524319814617e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.74140267916464056408693097635546173776e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.53554952415602030474322188152855226456e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.25818606585833092910042975933757268581e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.79945579370700383986587672831350689541e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.64188969985517050219881299229805701044e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.59811800080967790439078895802795103852e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.28225402720788909349967839966304553864e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.22313199865592821923485268860178384308e-28),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -128) {
        RealType t = -log2(ldexp(p, 64));

        // Rational Approximation
        // Maximum Relative Error: 1.1548e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.23500806363233607692361021471929016922e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.16747938711332885952564885548768072606e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.58252672740225210061031286151136185818e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.15477223656783400505701577048140375949e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.51130817987857130928725357067032472382e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.68955185187558206711296837951129048907e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.30418343536824247801373327173028702308e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.96029044368524575193330776540736319950e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.53259926294848786440686413056632823519e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.09476918795964320022985872737468492126e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.20864569882477440934325776445799604204e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.01321468510611172635388570487951907905e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.45079221107904118975166347269173516170e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.36559452694774774399884349254686988041e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.92396730553947142987611521115040472261e-35),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.36599681872296382199486815169747516110e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.86330544720458446620644055149504593514e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.23793435123936109978741252388806998743e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41867848381419339587560909646887411175e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.46493580168298395584601073832432583371e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.03537470107933172406224278121518518287e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.00375884189083338846655326801486182158e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62515041281615938158455493618149216047e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.42337949780187570810208014464208536484e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.40196890950995648233637422892711654146e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.11894256193449973803773962108906527772e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.00900961462911160554915139090711911885e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.65825716532665817972751320034032284421e-32),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -256) {
        RealType t = -log2(ldexp(p, 128));

        // Rational Approximation
        // Maximum Relative Error: 1.3756e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.04845570631944023525776899386112795330e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.18146685146173151383718092529868406030e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.13255476532916847606354932879190731233e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.44228173780288838603949849889291143631e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.18663599891768480607165516401619315227e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.61193813386433438633008774630150180359e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.54402603714659392010463991032389692959e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.76816632514967325885563032378775486543e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.22774672528068516513970610441705738842e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.47312348271366325243169398780745416279e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.10984325972747808970318612951079014854e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.81620524028936785168005732104270722618e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.03443227423068771484783389914203726108e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.02313300749670214384591200940841254958e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.40321396496046206171642334628524367374e-39),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.67292043485384876322219919215413286868e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.61652550158809553935603664087740554258e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14722810796821047167211543031044501921e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.78954660078305461714050086730116257387e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.52794892087101750452585357544956835504e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59652255206657812422503741672829368618e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.84914442937017449248597857220675602148e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.23662183613814475007146598734598810102e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.55388618781592901470236982277678753407e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.20418178834057564300014964843066904024e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.48606639104883413456676877330419513129e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39845313960416564778273486179935754019e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14538443937605324316706211070799970095e-35),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -512) {
        RealType t = -log2(ldexp(p, 256));

        // Rational Approximation
        // Maximum Relative Error: 1.6639e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.32865827226175697711590794217590458484e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.27551166309670994513910580518431041518e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.57161482253140058637495100797888501265e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.26908727392152312216118985392395130974e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.31391169101865809627389212651592902649e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.58926174475498352884244229017384309804e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.97074898765380614681225071978849430802e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.59852103341093122669197704225736036199e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.97287954178083606552531325613580819555e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.57634773176875526612407357244997035312e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.34467233210713881817055138794482883359e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.91236170873875898506577053309622472122e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.12220990075567880730037575497818287435e-33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.20881889651487527801970182542596258873e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.56848486878078288956741060120464349537e-43),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33267370502089423930888060969568705647e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39632352029900752622967578086289898150e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.42703739831525305516280300008439396218e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.45767617531685380458878368024246654652e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40344962756110545138002101382437142038e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.47015258371290492450093115369080460499e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.97280918936580227687603348219414768787e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40441024286579579491205384492088325576e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26130000914236204012152918399995098882e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.03893990417552709151955156348527062863e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12687002255767114781771099969545907763e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74218645918961186861014420578277888513e-35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.36897549168687040570349702061165281706e-39),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -1024) {
        RealType t = -log2(ldexp(p, 512));

        // Rational Approximation
        // Maximum Relative Error: 2.0360e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.67937186583822375017526293948703697225e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.06681764321003187068904973985967908140e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.12508887240195683379004033347903251977e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.56847823157999127998977939588643284176e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.31281869105767454049413701029676766275e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.01498951390503036399081209706853095793e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.72866867304007090391517734634589972858e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.44819806993104486828983054294866921869e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.91441365806306460165885645136864045231e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.71364347097027340365042558634044496149e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.63601648491144929836375956218857970640e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.76948956673676441236280803645598939735e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.47367651137203634311843318915161504046e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.83187517541957887917067558455828915184e-41),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.21480186561579326423946788448005430367e-47),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16494591376838053609854716130343599036e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.97651667629616309497454026431358820357e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.77741398674456235952879526959641925087e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.39478109667902532743651043316724748827e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.35965905056304225411108295866332882930e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83206524996481183422082802793852630990e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30320324476590103123012385840054658401e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.30322948189714718819437477682869360798e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43729790298535728717477691270336818161e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.89788697114251966298674871919685298106e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.43510901856942238937717065880365530871e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.38232043389918216652459244727737381677e-38),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64599465785019268214108345671361994702e-43),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -2048) {
        RealType t = -log2(ldexp(p, 1024));

        // Rational Approximation
        // Maximum Relative Error: 2.5130e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.11959316095291435774375635827672517008e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.30292575371366023255165927527306483022e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.54260157439166096303943109715675142318e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.61232819199170639867079290977704351939e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.74481972848503486840161528924694379655e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.98283577906243441829434029827766571263e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.90691715740583828769850056130458574520e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.31354728925505346732804698899977180508e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.42545288372836698650371589645832759416e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.12901629676328680681102537492164204387e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.99690040253176100731314099573187027309e-31),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.07899506956133955785140496937520311210e-35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.88607240095256436460507438213387199067e-40),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.78815984551154095621830792130401294111e-45),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.20516030880916148179297554212609531432e-51),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.81973541224235020744673910266545976833e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49156990107280109344880219729275939242e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.21606730563542176411852745162267260946e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11449329529433741944366607648360521674e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35659138507940819452801802756409587220e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.95708516718550485872934856595725983907e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78871290972777009292563576533612002908e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.60955318960542258732596447917271198307e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.72437581728117963125690670426402194936e-29),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77473938855958487119889840032590783232e-33),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.66198494713809467076392278745811981500e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.34142897042941614778352280692901008538e-42),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.98791658647635156162347063765388728959e-47),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4096) {
        RealType t = -log2(ldexp(p, 2048));

        // Rational Approximation
        // Maximum Relative Error: 3.1220e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.67307564006689676593687414536012112755e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.21005690250741024744367516466433711478e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.11537345869365655126739041291119096558e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.82910297156061391001507891422501792453e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.51574933516708249049824513935386420692e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.56433757960363802088718489136097249753e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.74248235865301086829849817739500215149e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.19151506367084295119369434315371762091e-22),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.98518959000360170320183116510466814569e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.21330422702314763225472001861559380186e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.44346922987964428874014866468161821471e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.59773305152191273570416120169527607421e-39),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.81894412321532723356954669501665983316e-44),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.25766346856943928908756472385992861288e-49),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.44601448994095786447982489957909713982e-55),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.90814053669730896497462224007523900520e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.72450689508305973756255440356759005330e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.76517273086984384225845151573287252506e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31844251885317815627707511621078762352e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.22689324865113257769413663010725436393e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27524826460469866934006001123700331335e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39172850948322201614266822896191911031e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.40343613937272428197414545004329993769e-27),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.17936773028976355507339458927541970545e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.66512598049860260933817550698863263184e-36),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.06432209670481882442649684139775366719e-41),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.10247886333916820534393624270217678968e-46),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40899321122058714028548211810431871877e-51),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -8192) {
        RealType t = -log2(ldexp(p, 4096));

        // Rational Approximation
        // Maximum Relative Error: 3.8974e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.36960987939726803544369406181770745475e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.17239311695985070524235502979761682692e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.51192208811996535244435318068035492922e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.38938553034896173195617671475670860841e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.55156910900732478717648524688116855303e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.14905040433940475292279950923000597280e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -7.35237464492052939771487320880614968639e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.07937000518607459141766382199896010484e-24),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.42797358100745086706362563988598447929e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.27871637324128856529004325499921407260e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.99553669724530906250814559570819049401e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.04272644552406682186928100080598582627e-42),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.70093315570856725077212325128817808000e-47),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.51767319872105260145583037426067406953e-53),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.87159268409640967747617639113346310759e-59),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.45350026842128595165328480395513258721e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.30398532102631290226106936127181928207e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.45242264812189519858570105609209495630e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.22745462510042159972414219082495434039e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31834645038348794443252730265421155969e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44590392528760847619123404904356730177e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.08436623062305193311891193246627599030e-25),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.46542554766048266351202730449796918707e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.78672067064478133389628198943161640913e-34),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.56573050761582685018467077197376031818e-39),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.92145137276530136848088270840255715047e-44),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.96933863894082533505471662180541379922e-49),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.92661972206232945959915223259585457082e-55),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -16384) {
        RealType t = -log2(ldexp(p, 8192));

        // Rational Approximation
        // Maximum Relative Error: 4.8819e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.24662793339079714510108682543625432532e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.25841960642102016210295419419373971750e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.10589156998251704634852108689102850747e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.18697614924486382142056819421294206504e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.79445222262445726654186491785652765635e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.41847338407338901513049755299049551186e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.44550500540299259432401029904726959214e-21),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.97463434518480676079167684683604645092e-26),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.68404349202062958045327516688040625516e-30),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.14018837476359778654965300153810397742e-35),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.67726222606571327724434861967972555751e-40),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.48082398191886705229604237754446294033e-45),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.28534456209055262678153908192583037946e-51),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.99466700145428173772768099494881455874e-57),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.43473066278196981345209422626769148425e-63),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.26570199429958856038191879713341034013e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.32484776300757286079244074394356908390e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.31234182027812869096733088981702059020e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13711443044675425837293030288097468867e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.11480036828082409994688474687120865023e-19),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.25592803132287127389756949487347562847e-23),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.45726006695535760451195102271978072855e-28),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13082170504731110487003517418453709982e-32),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.48217827031663836930337143509338210426e-37),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.47389053144555736191304002865419453269e-42),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.90980968229123572201281013063229644814e-47),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79448544326289688123648457587797649323e-53),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.56179793347045575604935927245529360950e-59),
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
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 53>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.25) {
        RealType t = p - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Absolute Error: 1.8559e-18
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(4.81512108276093785320e-1),
            static_cast<RealType>(-2.74296316128959647914e0),
            static_cast<RealType>(-3.29973875964825685757e1),
            static_cast<RealType>(-4.87536980816224603581e1),
            static_cast<RealType>(8.22233203036734027999e1),
            static_cast<RealType>(1.21654607908452130093e2),
            static_cast<RealType>(-6.66681853240657307279e1),
            static_cast<RealType>(-4.28101952511581488588e1),
        };
        BOOST_MATH_STATIC const RealType Q[10] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.20189490825315245036e0),
            static_cast<RealType>(1.63469912146101848441e1),
            static_cast<RealType>(-1.52740920318273920072e1),
            static_cast<RealType>(-5.41684560257839409762e1),
            static_cast<RealType>(6.51733677169299416471e0),
            static_cast<RealType>(3.93092001388102589237e1),
            static_cast<RealType>(-9.59983666140749481195e-1),
            static_cast<RealType>(-9.95648827557655863699e-1),
            static_cast<RealType>(-1.32007124426778083829e0),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Absolute Error: 4.6019e-17
        BOOST_MATH_STATIC const RealType P[7] = {
            static_cast<RealType>(1.70276979914029733585e0),
            static_cast<RealType>(2.09991992116646276165e1),
            static_cast<RealType>(2.26775403775298867998e1),
            static_cast<RealType>(-4.85384304722129472833e2),
            static_cast<RealType>(-1.47107146466495573999e3),
            static_cast<RealType>(-7.08748473959943943929e1),
            static_cast<RealType>(1.54245210917147215257e3),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(2.13092357122115486375e1),
            static_cast<RealType>(1.57318281834689144053e2),
            static_cast<RealType>(4.42261730187813035957e2),
            static_cast<RealType>(2.10814431586717588454e2),
            static_cast<RealType>(-6.36700983439599552504e2),
            static_cast<RealType>(-2.82923881266630617596e2),
            static_cast<RealType>(1.36613971025062750340e2),
        };

        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 1.2193e-19
        BOOST_MATH_STATIC const RealType P[8] = {
            static_cast<RealType>(4.25692449785074345588e-1),
            static_cast<RealType>(3.10963501706596356267e-1),
            static_cast<RealType>(2.91357806215297069863e-2),
            static_cast<RealType>(2.34716342676849303244e-2),
            static_cast<RealType>(5.83137296293361915583e-3),
            static_cast<RealType>(3.71792415497884868748e-4),
            static_cast<RealType>(1.59538372221030642757e-4),
            static_cast<RealType>(4.74040834029330213692e-6),
        };
        BOOST_MATH_STATIC const RealType Q[8] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(4.14801234415100707213e-1),
            static_cast<RealType>(1.04693730144480856638e-1),
            static_cast<RealType>(3.81581484862997435076e-2),
            static_cast<RealType>(8.95334009127358617362e-3),
            static_cast<RealType>(1.43316686981760147226e-3),
            static_cast<RealType>(1.81367766024620080990e-4),
            static_cast<RealType>(1.54779999748286671973e-5),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 4.4418e-17
        BOOST_MATH_STATIC const RealType P[11] = {
            static_cast<RealType>(5.07341098045260541890e-1),
            static_cast<RealType>(3.11771145411143166935e-1),
            static_cast<RealType>(1.74515601081894060888e-1),
            static_cast<RealType>(8.46576990174024231338e-2),
            static_cast<RealType>(2.57510090204322149315e-2),
            static_cast<RealType>(8.26605326867021684811e-3),
            static_cast<RealType>(1.73081423934722046819e-3),
            static_cast<RealType>(3.36314161099011673569e-4),
            static_cast<RealType>(4.50990441180388912803e-5),
            static_cast<RealType>(4.53513191985642134268e-6),
            static_cast<RealType>(2.62304611053075404923e-7),
        };
        BOOST_MATH_STATIC const RealType Q[11] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(5.28225379952156944029e-1),
            static_cast<RealType>(3.49662079845715371907e-1),
            static_cast<RealType>(1.45408903426879603625e-1),
            static_cast<RealType>(5.06773501409016231879e-2),
            static_cast<RealType>(1.45385556714043243731e-2),
            static_cast<RealType>(3.31235831325018043744e-3),
            static_cast<RealType>(6.06977554525543056050e-4),
            static_cast<RealType>(8.42406730405209749492e-5),
            static_cast<RealType>(8.32337989541696717905e-6),
            static_cast<RealType>(4.84923196546857128337e-7),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 5.7932e-17
        BOOST_MATH_STATIC const RealType P[10] = {
            static_cast<RealType>(5.41774626094491510395e-1),
            static_cast<RealType>(4.11060141334529017898e-1),
            static_cast<RealType>(1.48195601801946264526e-1),
            static_cast<RealType>(3.33881552814492855873e-2),
            static_cast<RealType>(5.20893974732203890418e-3),
            static_cast<RealType>(5.84734765774178832854e-4),
            static_cast<RealType>(4.71028150898133935445e-5),
            static_cast<RealType>(2.59185739450631464618e-6),
            static_cast<RealType>(7.77428184258777394627e-8),
            static_cast<RealType>(2.51255632629650930196e-14),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(7.58341767924960527280e-1),
            static_cast<RealType>(2.73511775500642961539e-1),
            static_cast<RealType>(6.16011987856129890130e-2),
            static_cast<RealType>(9.61296002312356116021e-3),
            static_cast<RealType>(1.07890675777726076554e-3),
            static_cast<RealType>(8.69223632953458271977e-5),
            static_cast<RealType>(4.78248875031756169279e-6),
            static_cast<RealType>(1.43460852065144859304e-7),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 9.0396e-17
        BOOST_MATH_STATIC const RealType P[9] = {
            static_cast<RealType>(5.41926067826974905066e-1),
            static_cast<RealType>(4.86926556246548518715e-1),
            static_cast<RealType>(2.11963908288176005856e-1),
            static_cast<RealType>(5.92200639925655576883e-2),
            static_cast<RealType>(1.18859816815542567438e-2),
            static_cast<RealType>(1.76833662992855443754e-3),
            static_cast<RealType>(2.21226152157950219596e-4),
            static_cast<RealType>(1.50444847316426133872e-5),
            static_cast<RealType>(1.87458213915373906356e-6),
        };
        BOOST_MATH_STATIC const RealType Q[9] = {
            static_cast<RealType>(1.),
            static_cast<RealType>(8.98511036742503939380e-1),
            static_cast<RealType>(3.91130673008184655152e-1),
            static_cast<RealType>(1.09277016228474605069e-1),
            static_cast<RealType>(2.19328471889880028208e-2),
            static_cast<RealType>(3.26305879571349016107e-3),
            static_cast<RealType>(4.08222014684743492069e-4),
            static_cast<RealType>(2.77611385768697969181e-5),
            static_cast<RealType>(3.45911046256304795257e-6),
        };

        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else {
        RealType p_square = p * p;

        if ((boost::math::isnormal)(p_square)) {
            result = 1 / cbrt(p_square * constants::two_pi<RealType>());
        }
        else if (p > 0) {
            result = 1 / (cbrt(p) * cbrt(p) * cbrt(constants::two_pi<RealType>()));
        }
        else {
            result = boost::math::numeric_limits<RealType>::infinity();
        }
    }

    return result;
}


template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_upper_imp_prec(const RealType& p, const boost::math::integral_constant<int, 113>&)
{
    BOOST_MATH_STD_USING
    RealType result;

    if (p >= 0.375) {
        RealType t = p - static_cast <RealType>(0.375);

        // Rational Approximation
        // Maximum Absolute Error: 4.0835e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.00474815142578902619056852805926666121e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.56422290947427848191079775267512708223e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.70103710180837859003070678080056933649e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.08521918131449191445864593768320217287e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.29340655781369686013042530147130581054e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.24198237124638368989049118891909723118e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.43382878809828906953609389440800537385e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.45564809127564867825118566276365267035e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.75881247317499884393790698530115428373e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.55845932095942777602241134226597158364e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.41328261385867825781522154621962338450e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.06758225510372847658316203115073730186e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.10895417312529385966062255102265009972e0),
        };
        BOOST_MATH_STATIC const RealType Q[12] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.88252553879196710256650370298744093367e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.54875259600848880869571364891152935969e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.78589587338618424770295921221996471887e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.15356831947775532414727361010652423453e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12951532118504570745988981200579372124e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.48163841544376327168780999614703092433e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.56786609618056303930232548304847911521e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.25610739352108840474197350343978451729e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.27063786175330237448255839666252978603e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.11941093895004369510720986032269722254e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.51487618026728514833542002963603231101e1),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.25) {
        RealType t = p - static_cast <RealType>(0.25);

        // Rational Approximation
        // Maximum Absolute Error: 5.7633e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.81512108276093787175849069715334402323e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24417080443497141096829831516758083481e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.67006165991083501886186268944009973084e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.74402382755828993223083868408545308340e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.49182541725192134610277727922493871787e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.67273564707254788337557775618297381267e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.73476432616329813096120568871900178919e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31235376166262024838125198332476698090e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.59379285677781413393733801325840617522e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.38151434050794836595564739176884302539e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.33534676810383673962443893459127818078e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.38110822236764293910895765875742805411e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.42750073722992463087082849671338957023e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.54255748148299874514839812717054396793e2),
        };
        BOOST_MATH_STATIC const RealType Q[13] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.64823387375875361292425741663822893626e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.02973633484731117050245517938177308809e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71288209768693917630236009171518272534e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.23837527610546426062625864735895938014e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.05056816585729983223036277071927165555e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.48087477651935811184913947280572029967e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04631058325147527913398256133791276127e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.69813394441679590721342220435891453447e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.92323371456465893290687995174952942311e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.68542430563281320943284015587559056621e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -4.17969051793607842221356465819951568080e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.82773308760283383020168853159163391394e2),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (p >= 0.125) {
        RealType t = p - static_cast <RealType>(0.125);

        // Rational Approximation
        // Maximum Absolute Error: 2.1140e-36
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[15] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70276979914029738186601698003670175907e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.63126762626382548478172664328434577553e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04190225271045202674546813475341133174e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.93523974140998850492859698545966806498e3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.19814006186501010136822066747124777014e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.55931423620290859807616748030589502039e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.78874021192395317496507459296221703565e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.03860533237347587977439662522389465152e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -2.77882648875352690605815508748162607271e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.05498612167816258406694194925933958145e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05326361485692298778330190198630232666e7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.85827791876754731187453265804790139032e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -8.93719378006868242377955041137674308589e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -5.56839957539576784391036362196229047625e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95604329277359828898502487252547842378e6),
        };
        BOOST_MATH_STATIC const RealType Q[16] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.79208640567193066236912382037923299779e1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94775812217734059201656828286490832145e2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16467934643564936346029555887148320030e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.35525720248600096849901920839060920346e4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69760913594243328874861534307039589127e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.32330501005950982838953061458838040612e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.79610639577090112327353399739315606205e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.43314292923292828425630915931385776182e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.97538885038058371436244702169375622661e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -9.48431896958634429210349441846613832681e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.93459449030820736960297236799012798749e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.67200014823529787381847745962773726408e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.37035571075060153491151970623824940994e6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -3.22682822001329636071591164177026394518e5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09781406768816062486819491582960840983e4),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / tools::evaluate_polynomial(Q, t);
    }
    else if (ilogb(p) >= -4) {
        RealType t = -log2(ldexp(p, 3));

        // Rational Approximation
        // Maximum Relative Error: 1.1409e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.25692449785074345466504245009175450649e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.75679137667345136118441108839649360362e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06171803174020856964914440692439080669e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.87798066278592051163038122952593080648e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.20070543183347459409303407166630392077e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.13457391270614708627745403376469848816e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06743974464224003715510181633693539914e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16870984737226212814217822779976770316e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.21845093091651861426944931268861694026e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.85357146081877929591916782097540632519e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -6.19085800299127898508052519062782284785e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.41985644250494046067095909812634573318e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.30857042700765443668305406695750760693e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.10466412567107519640190849286913680449e-10),
        };
        BOOST_MATH_STATIC const RealType Q[14] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.31914248618040435028023418981527961171e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.73578090645412656850163531828709850171e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.57329813782272411333511950903192234311e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.62736127875896578315177123764520823372e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.76809643836078823237530990091078867553e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32026948719622983920194944841520771986e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.45051018027743807545734050620973716634e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.58281707210621813556068724127478674938e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.63884527227517358294732620995363921547e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15973602356223075515067915930205826229e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.35069439950884795002182517078104942615e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.15454119109586223908613596754794988609e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.55273685376557721039847456564342945576e-10),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -8) {
        RealType t = -log2(ldexp(p, 4));

        // Rational Approximation
        // Maximum Relative Error: 1.2521e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[23] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.07341098045260497471001948654506267614e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.16518383383878659278973043343250842753e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.10029094208424121908983949243560936013e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.04771840726172284780129819470963100749e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.34173170868011689830672637082451998700e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.41990262178664512140746911398264330173e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.06779488545758366708787010705581103705e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.41892665233583725631482443019441608726e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.20692306716979208762785454648538891867e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.11097906809673639231336894729060830995e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.37476591232600886363441107536706973169e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.02659053066396720145189153810309784416e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.02209877191642023279303996697953314344e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.56663781532392665205516573323950583901e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95655734237060800145227277584749429063e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.06357695252098035545383649954315685077e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.78759045059235560356343893064681290047e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.95881339136963512103591745337914059651e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.70156441275519927563064848389865812060e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.99745225746277063516394774908346367811e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.45718440382347867317547921045052714102e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.39027665085346558512961348663034579801e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.05739751797738770096482688062542436470e-15),
        };
        BOOST_MATH_STATIC const RealType Q[23] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.43370372582239919321785765900615222895e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.52872159582703775260145036441128318159e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.28243735290178057451806192890274584778e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.93375398009812888642212045868197435998e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.73364866677217419593129631900708646445e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.53645928499107852437053167521160449434e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.74280939589407863107682593092148442428e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.80095449855178765594835180574448729793e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.65924845456946706158946250220103271334e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.52170861715436344002253767944763106994e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.87246437551620484806338690322735878649e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.88631873230311653853089809596759382095e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.84478812152918182782333415475103623486e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.47998768403859674841488325856607782853e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.82364683269852480160620586102339743788e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.65854316058742127585142691993199177898e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11358340340462071552670838135645042498e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.22818744671190957896035448856159685984e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.11038729491846772238262374112315536796e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.21355801166652957655438257794658921155e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.41278271853370874105923461404291742454e-14),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95100373579692323015092323646110838623e-15),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -16) {
        RealType t = -log2(ldexp(p, 8));

        // Rational Approximation
        // Maximum Relative Error: 2.0703e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[21] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.41774626094491452462664949805613444094e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.96383089261273022706449773421031102175e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.16315295073029174376617863024082371446e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.65377894193914426949840018839915119410e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.33210993830236821503160637845009556016e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.69315463529653886947182738378630780083e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.09869947341518160436616160018702590834e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44331691052908906654005398143769791881e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.13131413925652085071882765653750661678e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.64441840437413591336927030249538399459e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.78393581596372725434038621824715039765e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.50239319821178575427758224587858938204e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.92619647697287767235953207451871137149e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.26901081456833267780600560830367533351e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12151768312254597726918329997945574766e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.36907507996686107513673694597817437197e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31699373909892506279113260845246144240e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.11230682511893290562864133995544214588e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.44627067257461788044784631155226503036e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.39869585157420474301450400944478312794e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.82128612844034824876694595066123093042e-27),
        };
        BOOST_MATH_STATIC const RealType Q[20] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.65414405277042133067228113526697909557e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.32179221250476209346757936207079534440e0),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.74217392682100275524983756207618144313e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.45810448055940046896534973720645113799e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.81487408603233765436807980794697048675e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.49442843848941402948883852684502731460e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66330842256792791665907478718489013963e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.93285292223845804061941359223505045576e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.87966347754794288681626114849829710697e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.13711644429711675111080150193733607164e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.61758862007482013187806625777101452737e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.55435556106272558989915248980090731639e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.34166571320580242213843747025082914011e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31411360099525131959755145015018410429e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 6.21684839228785650625270026640716752452e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.43021096301255274530428188746599779008e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.58831290247776456235908211620983180005e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.74309280855806399632683315923592902203e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.58097100528573186098159133443927182780e-18),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -32) {
        RealType t = -log2(ldexp(p, 16));

        // Rational Approximation
        // Maximum Relative Error: 3.4124e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[19] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.41926067826974814669251179264786585885e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.21141529920003643675474888047093566280e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.59964592861304582755436075901659426485e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.95135112971576806260593571877646426022e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.12322024725362032809787183337883163254e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.96758465518847580191799508363466893068e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.12389553946694902774213055563291192175e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.04599236076217479033545023949602272721e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.16143771174487665823565565218797804931e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.38966874413947625866830582082846088427e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.02590325514935982607907975481732376204e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.44376747400143802055827426602151525955e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.82088624006657184426589019067893704020e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.95757210706845964048697237729100056232e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.36096213291559182424937062842308387702e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.14362780521873256616533770657488533993e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.73571098395815275003552523759665474105e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.47286214854389274681661944885238913581e-20),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.73701196181204039400706651811524874455e-34),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.77119890916406072259446489508263892540e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.95177888809731859578167185583119074026e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.29131027214559081111011582466619105016e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.31442657037887347262737789825299661237e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.83928863984637222329515960387531101267e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07389089078167127136964851949662391744e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.93013847797006474150589676891548600820e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.50600573851533884594030683413819219915e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.94539484213971921794449107859541806317e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.58360895032645281635534287874266252341e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.66414102108217999886628042310332365446e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.07411076181287950822375436854492998754e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.61224937285582228022463072515935601355e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.89242339209389981530783624934733098598e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.11030225010379194015550512905872992373e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.20285566539355859922818448335043495666e-18),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.71782855576364068752705740544460766362e-20),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else if (ilogb(p) >= -64) {
        RealType t = -log2(ldexp(p, 32));

        // Rational Approximation
        // Maximum Relative Error: 2.1680e-35
        // LCOV_EXCL_START
        BOOST_MATH_STATIC const RealType P[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.41926070139289008206183757488364846894e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.78434820569480998586988738136492447574e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.07939171933509333571821660328723436210e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.92438439347811482522082798370060349739e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.24288918322433485413615362874371441367e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.04437759300344740815274986587186340509e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74063952231188399929705762263485071234e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.07228849610363181194047955109059900544e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.93120850707001212714821992328252707694e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.40911049607879914351205073608184243057e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.71898232013947717725198847649536278438e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06963706982203753050300400912657068823e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.79849166632277658631839126599110199710e-11),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.74682085785152276503345630444792840850e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09650236336641219916377836114077389212e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.97326394822836529817663710792553753811e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 9.26635728806398747570910072594323836441e-17),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.96470010392255781222480229189380065951e-18),
        };
        BOOST_MATH_STATIC const RealType Q[18] = {
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 8.82841492468725267177870050157374330523e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.83703946702662950408034486958999188355e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.09320896703777230915306208582393356690e-1),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.29346630787642344947323515884281464979e-2),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.77242894492599243245354774839232776944e-3),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.05722029871614922850936250945431594997e-4),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.66920224988248720006255827987385374411e-5),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.40887155754772190509572243444386095560e-6),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 4.44545968319921473942351968892623238920e-7),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 3.17198676140022989760684932594389017027e-8),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.97376935482567419865730773801543995320e-9),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.06997835790265899882151030367297786861e-10),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.06862653266619706928282319356971834957e-12),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.02334307903766790059473763725329176667e-13),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 7.33174535634931487079630169746402085699e-15),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 1.70989324903345102377898775620363767855e-16),
            BOOST_MATH_BIG_CONSTANT(RealType, 113, 5.47067260145014475572799216996976703615e-18),
        };
        // LCOV_EXCL_STOP
        result = tools::evaluate_polynomial(P, t) / (tools::evaluate_polynomial(Q, t) * cbrt(p * p));
    }
    else {
        RealType p_square = p * p;

        if ((boost::math::isnormal)(p_square)) {
            result = 1 / cbrt(p_square * constants::two_pi<RealType>());
        }
        else if (p > 0) {
            result = 1 / (cbrt(p) * cbrt(p) * cbrt(constants::two_pi<RealType>()));
        }
        else {
            result = boost::math::numeric_limits<RealType>::infinity();
        }
    }

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 53>& tag)
{
    if (p > 0.5) {
        return !complement ? mapairy_quantile_upper_imp_prec(1 - p, tag) : mapairy_quantile_lower_imp_prec(1 - p, tag);
    }

    return complement ? mapairy_quantile_upper_imp_prec(p, tag) : mapairy_quantile_lower_imp_prec(p, tag);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_imp_prec(const RealType& p, bool complement, const boost::math::integral_constant<int, 113>& tag)
{
    if (p > 0.5) {
        return !complement ? mapairy_quantile_upper_imp_prec(1 - p, tag) : mapairy_quantile_lower_imp_prec(1 - p, tag);
    }

    return complement ? mapairy_quantile_upper_imp_prec(p, tag) : mapairy_quantile_lower_imp_prec(p, tag);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_quantile_imp(const mapairy_distribution<RealType, Policy>& dist, const RealType& p, bool complement)
{
    // This routine implements the quantile for the Map-Airy distribution,
    // the value p may be the probability, or its complement if complement=true.

    constexpr auto function = "boost::math::quantile(mapairy<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Map-Airy distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * mapairy_quantile_imp_prec(p, complement, tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_mode_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(-1.16158727113597068525);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_mode_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, -1.1615872711359706852500000803029112987);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_mode_imp(const mapairy_distribution<RealType, Policy>& dist)
{
    // This implements the mode for the Map-Airy distribution,

    constexpr auto function = "boost::math::mode(mapairy<%1%>&, %1%)";
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

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Map-Airy distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * mapairy_mode_imp_prec<RealType>(tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_median_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(-0.71671068545502205332);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_median_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, -0.71671068545502205331700196278067230944440);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_median_imp(const mapairy_distribution<RealType, Policy>& dist)
{
    // This implements the median for the Map-Airy distribution,

    constexpr auto function = "boost::math::median(mapairy<%1%>&, %1%)";
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

    typedef typename tools::promote_args<RealType>::type result_type;
    typedef typename policies::precision<result_type, Policy>::type precision_type;
    typedef boost::math::integral_constant<int,
        precision_type::value <= 0 ? 0 :
        precision_type::value <= 53 ? 53 :
        precision_type::value <= 113 ? 113 : 0
    > tag_type;

    static_assert(tag_type::value, "The Map-Airy distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = location + scale * mapairy_median_imp_prec<RealType>(tag_type());

    return result;
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_entropy_imp_prec(const boost::math::integral_constant<int, 53>&)
{
    return static_cast<RealType>(2.00727681841065634600);
}

template <class RealType>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_entropy_imp_prec(const boost::math::integral_constant<int, 113>&)
{
    return BOOST_MATH_BIG_CONSTANT(RealType, 113, 2.0072768184106563460003025875575283708);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mapairy_entropy_imp(const mapairy_distribution<RealType, Policy>& dist)
{
    // This implements the entropy for the Map-Airy distribution,

    constexpr auto function = "boost::math::entropy(mapairy<%1%>&, %1%)";
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

    static_assert(tag_type::value, "The Map-Airy distribution is only implemented for types with known precision, and 113 bits or fewer in the mantissa (ie 128 bit quad-floats");

    result = mapairy_entropy_imp_prec<RealType>(tag_type()) + log(scale);

    return result;
}

} // detail

template <class RealType = double, class Policy = policies::policy<> >
class mapairy_distribution
{
    public:
    typedef RealType value_type;
    typedef Policy policy_type;

    BOOST_MATH_GPU_ENABLED mapairy_distribution(RealType l_location = 0, RealType l_scale = 1)
        : mu(l_location), c(l_scale)
    {
        constexpr auto function = "boost::math::mapairy_distribution<%1%>::mapairy_distribution";
        RealType result = 0;
        detail::check_location(function, l_location, &result, Policy());
        detail::check_scale(function, l_scale, &result, Policy());
    } // mapairy_distribution

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

typedef mapairy_distribution<double> mapairy;

#ifdef __cpp_deduction_guides
template <class RealType>
mapairy_distribution(RealType) -> mapairy_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
mapairy_distribution(RealType, RealType) -> mapairy_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const mapairy_distribution<RealType, Policy>&)
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
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const mapairy_distribution<RealType, Policy>&)
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
BOOST_MATH_GPU_ENABLED inline RealType pdf(const mapairy_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::mapairy_pdf_imp(dist, x);
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const mapairy_distribution<RealType, Policy>& dist, const RealType& x)
{
    return detail::mapairy_cdf_imp(dist, x, false);
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const mapairy_distribution<RealType, Policy>& dist, const RealType& p)
{
    return detail::mapairy_quantile_imp(dist, p, false);
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<mapairy_distribution<RealType, Policy>, RealType>& c)
{
    return detail::mapairy_cdf_imp(c.dist, c.param, true);
} //  cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<mapairy_distribution<RealType, Policy>, RealType>& c)
{
    return detail::mapairy_quantile_imp(c.dist, c.param, true);
} // quantile complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const mapairy_distribution<RealType, Policy> &dist)
{
    return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const mapairy_distribution<RealType, Policy>& /*dist*/)
{
    return boost::math::numeric_limits<RealType>::infinity();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const mapairy_distribution<RealType, Policy>& dist)
{
    return detail::mapairy_mode_imp(dist);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const mapairy_distribution<RealType, Policy>& dist)
{
    return detail::mapairy_median_imp(dist);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const mapairy_distribution<RealType, Policy>& /*dist*/)
{
    // There is no skewness:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Map-Airy Distribution has no skewness");

    return policies::raise_domain_error<RealType>(
        "boost::math::skewness(mapairy<%1%>&)",
        "The Map-Airy distribution does not have a skewness: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy()); // infinity?
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const mapairy_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Map-Airy Distribution has no kurtosis");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis(mapairy<%1%>&)",
        "The Map-Airy distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const mapairy_distribution<RealType, Policy>& /*dist*/)
{
    // There is no kurtosis excess:
    typedef typename Policy::assert_undefined_type assert_type;
    static_assert(assert_type::value == 0, "The Map-Airy Distribution has no kurtosis excess");

    return policies::raise_domain_error<RealType>(
        "boost::math::kurtosis_excess(mapairy<%1%>&)",
        "The Map-Airy distribution does not have a kurtosis: "
        "the only possible return value is %1%.",
        boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const mapairy_distribution<RealType, Policy>& dist)
{
    return detail::mapairy_entropy_imp(dist);
}

}} // namespaces


#endif // BOOST_STATS_MAPAIRY_HPP
