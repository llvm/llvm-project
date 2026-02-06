// boost-no-inspect
/*
 * Copyright Nick Thompson, Matt Borland, 2023
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_MATH_SPECIAL_FOURIER_TRANSFORM_DAUBECHIES_HPP
#define BOOST_MATH_SPECIAL_FOURIER_TRANSFORM_DAUBECHIES_HPP
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/tools/estrin.hpp>

namespace boost::math {

namespace detail {

// See the Table 6.2 of Daubechies, Ten Lectures on Wavelets.
// These constants are precisely those divided by 1/sqrt(2), because otherwise
// we'd immediately just have to divide through by 1/sqrt(2).
// These numbers agree with Table 6.2, but are generated via example/calculate_fourier_transform_daubechies_constants.cpp
template <typename Real, unsigned N> constexpr std::array<Real, N> ft_daubechies_scaling_polynomial_coefficients() {
  static_assert(N >= 1 && N <= 10, "Scaling function only implemented for 1-10 vanishing moments.");
  if constexpr (N == 1) {
    return std::array<Real, 1>{static_cast<Real>(1)};
  }
  if constexpr (N == 2) {
    return {BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                    1.36602540378443864676372317075293618347140262690519031402790348972596650842632007803393058),
            BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                    -0.366025403784438646763723170752936183471402626905190314027903489725966508441952115116994061)};
  }
  if constexpr (N == 3) {
    return std::array<Real, 3>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                1.88186883113665472301331643028468183320710177910151845853383427363197699204347143889269703),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -1.08113883008418966599944677221635926685977756966260841342875242639629721931484516409937898),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.199269998947534942986130341931677433652675790561089954894918152764320227250084833874126086)};
  }
  if constexpr (N == 4) {
    return std::array<Real, 4>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                2.60642742441038678619616138456320274846457112268350230103083547418823666924354637907021821),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -2.33814397690691624172277875654682595239896411009843420976312905955518655953831321619717516),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.851612467139421235087502761217605775743179492713667860409024360383174560120738199344383827),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.119895914642891779560885389233982571808786505298735951676730775016224669960397338539830347)};
  }
  if constexpr (N == 5) {
    return std::array<Real, 5>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                3.62270372133693372237431371824382790538377237674943454540758419371854887218301659611796287),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -4.45042192340421529271926241961545172940077367856833333571968270791760393243895360839974479),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                2.41430351179889241160444590912469777504146155873489898274561148139247721271772284677196254),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.662064156756696785656360678859372223233256033099757083735935493062448802216759690564503751),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.0754788470250859443968634711062982722087957761837568913024225258690266500301041274151679859)};
  }
  if constexpr (N == 6) {
    return std::array<Real, 6>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                5.04775782409284533508504459282823265081102702143912881539214595513121059428213452194161891),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -7.90242489414953082292172067801361411066690749603940036372954720647258482521355701761199),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                5.69062231972011992229557724635729642828799628244009852056657089766265949751788181912632318),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -2.29591465417352749013350971621495843275025605194376564457120763045109729714936982561585742),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.508712486289373262241383448555327418882885930043157873517278143590549199629822225076344289),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.0487530817792802065667748935122839545647456859392192011752401594607371693280512344274717466)};
  }
  if constexpr (N == 7) {
    return std::array<Real, 7>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                7.0463635677199166580912954330590360004554457287730448872409828895500755049108034478397642),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -13.4339028220058085795120274851204982381087988043552711869584397724404274044947626280185946),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                12.0571882966390397563079887516068140052534768286900467252199152570563053103366694003818755),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -6.39124482303930285525880162640679389779540687632321120940980371544051534690730897661850842),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                2.07674879424918331569327229402057948161936796436510457676789758815816492768386639712643599),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.387167532162867697386347232520843525988806810788254462365009860280979111139408537312553398),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.0320145185998394020646198653617061745647219696385406695044576133973761206215673170563538)};
  }
  if constexpr (N == 8) {
    return std::array<Real, 8>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                9.85031962984351656604584909868313752909650830419035084214249929687665775818153930511533915),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -22.1667494032601530437943449172929277733925779301673358406203340024653233856852379126537395),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                23.8272728452144265698978643079553442578633838793866258585693705776047828901217069807060715),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -15.6065825916019064469551268429136774427686552695820632173344334583910793479437661751737998),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                6.63923943761238270605338141020386331691362835005178161341935720370310013774320917891051914),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -1.81462830704498058848677549516134095104668450780318379608495409574150643627578462439190617),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.292393958692487086036895445298600849998803161432207979583488595754566344585039785927586499),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.0212655694557728487977430067729997866644059875083834396749941173411979591559303697954912042)};
  }
  if constexpr (N == 9) {
    return std::array<Real, 9>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                13.7856894948673536752299497816200874595462540239049618127984616645562437295073582057283235),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -35.79362367743347676734569335180426263053917566987500206688713345532850076082533131311371),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                44.8271517576868325408174336351944130389504383168376658969692365144162452669941793147313),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -34.9081281226625998193992072777004811412863069972654446089639166067029872995118090115016879),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                18.2858070519930071738884732413420775324549836290768317032298177553411077249931094333824682),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -6.53714271572640296907117142447372145396492988681610221640307755553450246302777187366825001),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                1.5454286423270706293059630490222623728433659436325762803842722481655127844136128434034519),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.219427682644567750633335191213222483839627852234602683427115193605056655384931679751929029),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.0142452515927832872075875380128473058349984927391158822994546286919376896668596927857450578)};
  }
  if constexpr (N == 10) {
    return std::array<Real, 10>{
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                19.3111846872275854185286532829110292444580572106276740012656292351880418629976266671349603),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -56.8572892818288577904562616825768121532988312416110097001327598719988644787442373891037268),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                81.3040184941182201969442916535886223134891624078921290339772790298979750863332417443823932),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -73.3067370305702272426402835488383512315892354877130132060680994033122368453226804355121917),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                45.5029913577892585869595005785056707790215969761054467083138479721524945862678794713356742),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -20.0048938122958245128650205249242185678760602333821352917865992073643758821417211689052482),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                6.18674372398711325312495154772282340531430890354257911422818567803548535981484584999007723),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -1.29022235346655645559407302793903682217361613280994725979138999393113139183198020070701239),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                0.16380852384056875506684562409582514726612462486206657238854671180228210790016298829595125),
        BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits,
                                -0.00960430880128020906860390254555211461150702751378997239464015046967050703218076318595987803)};
  }
}

} // namespace detail

/*
 * Given ω∈ℝ, computes a numerical approximation to 𝓕[𝜙](ω),
 * where 𝜙 is the Daubechies scaling function.
 * Fast and accurate evaluation of these function seems to me to be a rather involved research project,
 * which I have not endeavored to complete.
 * In particular, recovering ~1ULP evaluation is not possible using the techniques
 * employed here-you should use this with the understanding it is good enough for almost
 * all uses with empirical data, but probably doesn't recover enough accuracy
 * for pure mathematical uses (other than graphing-in which case it's fine).
 * The implementation uses an infinite product of trigonometric polynomials.
 * See Daubechies, 10 Lectures on Wavelets, equation 5.1.17, 5.1.18.
 * It uses the factorization of m₀ shown in Corollary 5.5.4 and equation 5.5.5.
 * See more discusion near equation 6.1.1,
 * as well as efficiency gains from equation 7.1.4.
 */
template <class Real, unsigned p> std::complex<Real> fourier_transform_daubechies_scaling(Real omega) {
  // This arg promotion is kinda sad, but IMO the accuracy is not good enough in
  // float precision using this method. Requesting a better algorithm!
  if constexpr (std::is_same_v<Real, float>) {
    return static_cast<std::complex<float>>(fourier_transform_daubechies_scaling<double, p>(static_cast<double>(omega)));
  }
  using boost::math::constants::one_div_root_two_pi;
  using std::abs;
  using std::exp;
  using std::norm;
  using std::pow;
  using std::sqrt;
  using std::cbrt;
  // Equation 7.1.4 of 10 Lectures on Wavelets is singular at ω=0:
  if (omega == 0) {
     return std::complex<Real>(one_div_root_two_pi<Real>(), 0);
  }
  // For whatever reason, this starts returning NaNs rather than zero for |ω|≫1.
  // But we know that this function decays rather quickly with |ω|,
  // and hence it is "numerically zero", even if in actuality the function does not have compact support.
  // Now, should we probably do a fairly involved, exhaustive calculation to see where exactly we should set this threshold
  // and store them in a table? .... yes.
  if (abs(omega) >= sqrt(std::numeric_limits<Real>::max())) {
       return std::complex<Real>(0, 0);
  }
  auto const constexpr lxi = detail::ft_daubechies_scaling_polynomial_coefficients<Real, p>();
  auto xi = -omega / 2;
  std::complex<Real> phi{one_div_root_two_pi<Real>(), 0};
  do {
    std::complex<Real> arg{0, xi};
    auto z = exp(arg);
    phi *= boost::math::tools::evaluate_polynomial_estrin(lxi, z);
    xi /= 2;
  } while (abs(xi) > std::numeric_limits<Real>::epsilon());
  std::complex<Real> arg{0, omega};
  // There is no std::expm1 for complex numbers.
  // We may therefore be leaving accuracy gains on the table for small |ω|:
  std::complex<Real> prefactor = (Real(1) - exp(-arg))/arg;
  return phi * static_cast<std::complex<Real>>(pow(prefactor, p));
}

template <class Real, unsigned p> std::complex<Real> fourier_transform_daubechies_wavelet(Real omega) {
  // See Daubechies, 10 Lectures on Wavelets, page 193, unlabelled equation in Theorem 6.3.6:
  // 𝓕[ψ](ω) = -exp(-iω/2)m₀(ω/2 + π)^{*}𝓕[𝜙](ω/2)
  if constexpr (std::is_same_v<Real, float>) {
    return static_cast<std::complex<float>>(fourier_transform_daubechies_wavelet<double, p>(static_cast<double>(omega)));
  }

  using std::exp;
  using std::pow;
  auto Fphi = fourier_transform_daubechies_scaling<Real, p>(omega/2);
  auto phase = -exp(std::complex<Real>(0, -omega/2));
  // See Section 6.4 for the sign convention on the argument,
  // as well as Table 6.2:
  auto z = phase; // strange coincidence.
  //auto z = exp(std::complex<Real>(0, -omega/2 - boost::math::constants::pi<Real>()));
  auto constexpr lxi = detail::ft_daubechies_scaling_polynomial_coefficients<Real, p>();
  auto m0 = std::complex<Real>(pow((Real(1) + z)/Real(2), p))*boost::math::tools::evaluate_polynomial_estrin(lxi, z);
  return Fphi*std::conj(m0)*phase;
}

} // namespace boost::math
#endif
