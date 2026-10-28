//===-- AGXMathExpansion.cpp - Math functions without hardware ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The math functions AGX has no instruction for, built from the ones it has
// (sin, cos, exp2, log2, sqrt, division) in float: tangent, the inverse and
// hyperbolic functions and their inverses, and sinpi / cospi / tanpi. Half
// and bfloat values are computed in float. The reductions and polynomials
// are those of the Cephes single precision library (atanf, asinf, acosf,
// sinhf, coshf, tanhf, asinhf, acoshf, atanhf).
//
// Also the precise float sin, cos, exp, exp2, exp10, log, log2, log10 and
// pow, where the hardware's approximations (which the afn forms keep) are
// tens of units in the last place off: the argument reduced exactly (a
// three-part pi / 2, the exponent split from a log or exp argument), Cephes'
// sinf, cosf, expf and logf polynomials, and for pow a log and exp carried
// with a low-order part.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include <cmath>

using namespace llvm;

namespace {
class MathBuilder {
  IRBuilderBase &B;
  Type *Ty; // float or a vector of float

public:
  MathBuilder(IRBuilderBase &B, Type *Ty) : B(B), Ty(Ty) {}

  Value *k(double V) { return ConstantFP::get(Ty, V); }
  Value *call(Intrinsic::ID ID, ArrayRef<Value *> Args) {
    return B.CreateIntrinsic(ID, {Ty}, Args);
  }
  Value *fma(Value *A, Value *X, Value *C) {
    return call(Intrinsic::fma, {A, X, C});
  }
  /// The polynomial with coefficients \p C, highest degree first, at \p Z.
  Value *poly(Value *Z, ArrayRef<double> C) {
    Value *P = k(C[0]);
    for (double V : C.drop_front())
      P = fma(P, Z, k(V));
    return P;
  }
  Value *abs(Value *X) { return call(Intrinsic::fabs, {X}); }
  Value *copysign(Value *Mag, Value *Sign) {
    return call(Intrinsic::copysign, {Mag, Sign});
  }
  Value *sqrt(Value *X) { return call(Intrinsic::sqrt, {X}); }
  Value *exp(Value *X) { return call(Intrinsic::exp, {X}); }
  Value *expPrecise(Value *X) { return expHiLo(X, k(0.0)); }
  Value *log(Value *X) { return call(Intrinsic::log, {X}); }
  Value *sel(Value *C, Value *T, Value *F) { return B.CreateSelect(C, T, F); }
  Value *gt(Value *A, double V) { return B.CreateFCmpOGT(A, k(V)); }
  Value *lt(Value *A, double V) { return B.CreateFCmpOLT(A, k(V)); }
  Value *mul(Value *A, Value *X) { return B.CreateFMul(A, X); }
  Value *add(Value *A, Value *X) { return B.CreateFAdd(A, X); }
  Value *sub(Value *A, Value *X) { return B.CreateFSub(A, X); }
  Value *div(Value *A, Value *X) { return B.CreateFDiv(A, X); }

  /// atan of a value in [0, inf] or NaN: the argument reduced to
  /// [0, tan(pi / 8)] around 0, pi / 4 or pi / 2.
  Value *atanNonNegative(Value *A) {
    Value *Big = gt(A, 2.414213562373095), *Mid = gt(A, 0.4142135623730950);
    Value *X = sel(Big, div(k(-1.0), A),
                   sel(Mid, div(sub(A, k(1.0)), add(A, k(1.0))), A));
    Value *Y0 = sel(Big, k(M_PI_2), sel(Mid, k(M_PI_4), k(0.0)));
    Value *Z = mul(X, X);
    Value *P = poly(Z, {8.05374449538e-2, -1.38776856032e-1, 1.99777106478e-1,
                        -3.33329491539e-1});
    return add(Y0, fma(mul(P, Z), X, X));
  }

  /// asin of a value in [0, 0.5].
  Value *asinSmall(Value *A) {
    Value *Z = mul(A, A);
    Value *P = poly(Z, {4.2163199048e-2, 2.4181311049e-2, 4.5470025998e-2,
                        7.4953002686e-2, 1.6666752422e-1});
    return fma(mul(P, Z), A, A);
  }

  Value *asin(Value *X) {
    Value *A = abs(X);
    // Above 0.5: asin(a) = pi / 2 - 2 asin(sqrt((1 - a) / 2)).
    Value *Big = gt(A, 0.5);
    Value *S = sqrt(mul(k(0.5), sub(k(1.0), A)));
    Value *R = sel(Big, sub(k(M_PI_2), mul(k(2.0), asinSmall(S))),
                   asinSmall(A));
    // |x| > 1: the square root is NaN.
    R = sel(gt(A, 1.0), S, R);
    return copysign(R, X);
  }

  Value *acos(Value *X) {
    Value *A = abs(X);
    // |x| > 0.5: acos(|x|) = 2 asin(sqrt((1 - |x|) / 2)), and
    // acos(-|x|) = pi - acos(|x|); else pi / 2 - asin(x).
    Value *S = sqrt(mul(k(0.5), sub(k(1.0), A)));
    Value *Twice = mul(k(2.0), asinSmall(S));
    Value *Outer = sel(lt(X, 0.0), sub(k(M_PI), Twice), Twice);
    Value *Inner = sub(k(M_PI_2), copysign(asinSmall(A), X));
    Value *R = sel(gt(A, 0.5), Outer, Inner);
    return sel(gt(A, 1.0), S, R);
  }

  Value *atan(Value *X) { return copysign(atanNonNegative(abs(X)), X); }

  Value *atan2(Value *Y, Value *X) {
    Value *AX = abs(X), *AY = abs(Y);
    Value *Swap = B.CreateFCmpOGT(AY, AX);
    Value *Mn = sel(Swap, AX, AY), *Mx = sel(Swap, AY, AX);
    // 0 / 0 is 0 (atan2(+-0, +-0)) and inf / inf is 1 (the diagonals).
    Value *T = div(Mn, Mx);
    T = sel(B.CreateFCmpOEQ(Mx, k(0.0)), k(0.0), T);
    T = sel(B.CreateFCmpOEQ(Mn, k(INFINITY)), k(1.0), T);
    Value *R = atanNonNegative(T);
    R = sel(Swap, sub(k(M_PI_2), R), R);
    // x < 0 or x = -0: the left half plane.
    Value *Left = lt(copysign(k(1.0), X), 0.0);
    R = sel(Left, sub(k(M_PI), R), R);
    R = copysign(R, Y);
    Value *NaN = B.CreateFCmpUNO(X, Y);
    return sel(NaN, add(X, Y), R);
  }

  /// e^a / 2 for a >= 0, without overflow below the largest float.
  Value *halfExp(Value *A) {
    Value *Shifted = exp(sub(A, k(M_LN2)));
    return sel(lt(A, 88.0), mul(k(0.5), exp(A)), Shifted);
  }

  Value *sinh(Value *X) {
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small = fma(mul(poly(Z, {2.03721912945e-4, 8.33028376239e-3,
                                    1.66667160211e-1}),
                           Z),
                       X, X);
    Value *H = halfExp(A);
    Value *Big = copysign(sub(H, div(k(0.25), H)), X);
    return sel(gt(A, 1.0), Big, Small);
  }

  Value *cosh(Value *X) {
    Value *H = halfExp(abs(X));
    return add(H, div(k(0.25), H));
  }

  Value *tanh(Value *X) {
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small =
        fma(mul(poly(Z, {-5.70498872745e-3, 2.06390887954e-2,
                         -5.37397155531e-2, 1.33314422036e-1,
                         -3.33332819422e-1}),
                Z),
            X, X);
    // 1 - 2 / (e^2a + 1), which is 1 in float from a = 9 on.
    Value *S = exp(mul(k(2.0), sel(gt(A, 9.0), k(9.0), A)));
    Value *Big = copysign(sub(k(1.0), div(k(2.0), add(S, k(1.0)))), X);
    return sel(lt(A, 0.625), Small, Big);
  }

  Value *tan(Value *X) {
    return div(call(Intrinsic::sin, {X}), call(Intrinsic::cos, {X}));
  }

  Value *asinh(Value *X) {
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small =
        fma(mul(poly(Z, {2.0122003309e-2, -4.2699340972e-2, 7.4847586088e-2,
                         -1.6666288134e-1}),
                Z),
            X, X);
    // log(2a) once a * a + 1 is a * a in float (and before it overflows).
    Value *Huge = add(log(A), k(M_LN2));
    Value *Mid = log(add(A, sqrt(fma(A, A, k(1.0)))));
    Value *Big = copysign(sel(gt(A, 1500.0), Huge, Mid), X);
    return sel(lt(A, 0.5), Small, Big);
  }

  Value *acosh(Value *X) {
    Value *Z = sub(X, k(1.0));
    // Near 1: sqrt(z) times a polynomial (and NaN below 1).
    Value *Near = mul(sqrt(Z),
                      poly(Z, {1.7596881071e-3, -7.5272886713e-3,
                               2.6454905019e-2, -1.1784741703e-1,
                               1.4142135263e0}));
    Value *Huge = add(log(X), k(M_LN2));
    Value *Mid = log(add(X, sqrt(mul(Z, add(X, k(1.0))))));
    return sel(lt(Z, 0.5), Near, sel(gt(X, 1500.0), Huge, Mid));
  }

  Value *atanh(Value *X) {
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small =
        fma(mul(poly(Z, {1.81740078349e-1, 8.24370301058e-2, 1.46691431730e-1,
                         1.99782164500e-1, 3.33337300303e-1}),
                Z),
            X, X);
    Value *Big = copysign(
        mul(k(0.5), log(div(add(k(1.0), A), sub(k(1.0), A)))), X);
    return sel(lt(A, 0.5), Small, Big);
  }

  /// x - 2 round(x / 2), exactly: in [-1, 1].
  Value *halfTurns(Value *X) {
    return sub(X, mul(k(2.0), call(Intrinsic::roundeven, {mul(X, k(0.5))})));
  }

  /// sin(pi r) and cos(pi r) for |r| <= 1/4, with pi r as a rounded
  /// product and its error.
  std::pair<Value *, Value *> sinCosPi(Value *R) {
    constexpr double PiLo = 3.14159265358979323846 - (double)(float)M_PI;
    Value *Hi = mul(R, k(M_PI));
    Value *Lo = fma(R, k(M_PI), B.CreateFNeg(Hi));
    Lo = fma(R, k(PiLo), Lo);
    auto [S, C] = sinCosPolys(Hi);
    return {fma(Lo, C, S), fma(B.CreateFNeg(Lo), S, C)};
  }

  Value *sinpi(Value *X) {
    // sin(pi r) = sin(pi (sign(r) - r)) folds r into [-1/2, 1/2], and
    // = cos(pi (sign(r) / 2 - r)) beyond 1/4; every step is exact.
    Value *R = halfTurns(X);
    Value *R2 = sel(gt(abs(R), 0.5), sub(copysign(k(1.0), R), R), R);
    Value *UseCos = gt(abs(R2), 0.25);
    Value *R3 = sel(UseCos, sub(copysign(k(0.5), R2), R2), R2);
    auto [S, C] = sinCosPi(R3);
    Value *V = sel(UseCos, copysign(C, R2), S);
    // Integers: a zero with the sign of x.
    return sel(B.CreateFCmpOEQ(X, call(Intrinsic::trunc, {X})),
               copysign(k(0.0), X), V);
  }

  Value *cospi(Value *X) {
    // cos(pi r) = -cos(pi (1 - r)) folds |r| into [0, 1/2], and
    // = sin(pi (1/2 - r)) beyond 1/4, exactly 0 at the half integers.
    Value *R = abs(halfTurns(X));
    Value *Big = gt(R, 0.5);
    Value *R2 = sel(Big, sub(k(1.0), R), R);
    Value *UseSin = gt(R2, 0.25);
    Value *R3 = sel(UseSin, sub(k(0.5), R2), R2);
    auto [S, C] = sinCosPi(R3);
    Value *V = sel(UseSin, S, C);
    return sel(Big, B.CreateFNeg(V), V);
  }

  Value *tanpi(Value *X) { return div(sinpi(X), cospi(X)); }

  // Precise sin, cos, exp and log.

  Type *intTy() { return Ty->getWithNewType(B.getInt32Ty()); }
  Value *ki(int32_t V) { return ConstantInt::get(intTy(), V); }
  Value *isNaN(Value *X) { return B.CreateFCmpUNO(X, X); }
  /// a + b as the rounded sum and its error (Knuth's two-sum).
  std::pair<Value *, Value *> twoSum(Value *A, Value *X) {
    Value *S = add(A, X);
    Value *BB = sub(S, A);
    Value *Err = add(sub(A, sub(S, BB)), sub(X, BB));
    return {S, Err};
  }

  /// 2^n for an integral n in [-126, 127].
  Value *pow2(Value *N) {
    Value *E = B.CreateAdd(B.CreateFPToSI(N, intTy()), ki(127));
    return B.CreateBitCast(B.CreateShl(E, ki(23)), Ty);
  }
  /// v * 2^n for an integral n in [-252, 254], in two steps.
  Value *ldexp(Value *V, Value *N) {
    Value *N1 = call(Intrinsic::trunc, {mul(N, k(0.5))});
    return mul(mul(V, pow2(N1)), pow2(sub(N, N1)));
  }

  /// e^r for |r| <= ln(2) / 2 (Cephes expf).
  Value *expCore(Value *R) {
    Value *P = poly(R, {1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3,
                        4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1});
    return add(fma(P, mul(R, R), R), k(1.0));
  }

  /// e^(hi + lo), lo a small correction: n = round(hi / ln 2) is split off
  /// exactly (ln 2 in two parts) and the rest goes to expCore.
  Value *expHiLo(Value *Hi, Value *Lo) {
    Value *C = sel(gt(Hi, 89.0), k(89.0), sel(lt(Hi, -104.0), k(-104.0), Hi));
    Value *N = call(Intrinsic::roundeven, {mul(C, k(M_LOG2E))});
    Value *R = fma(B.CreateFNeg(N), k(0.693359375), C);
    R = fma(B.CreateFNeg(N), k(-2.12194440e-4), R);
    Value *E = ldexp(expCore(add(R, Lo)), N);
    E = sel(gt(Hi, 89.0), k(INFINITY), sel(lt(Hi, -104.0), k(0.0), E));
    return sel(isNaN(Hi), Hi, E);
  }

  Value *exp2Precise(Value *X) {
    // 2^x = e^(x ln 2), with x ln 2 as a rounded product and its error.
    Value *Hi = mul(X, k(M_LN2));
    Value *Lo = fma(X, k(M_LN2), B.CreateFNeg(Hi));
    Lo = fma(X, k(M_LN2 - (double)(float)M_LN2), Lo);
    return expHiLo(Hi, Lo);
  }
  Value *exp10Precise(Value *X) {
    constexpr double LN10 = 2.302585092994045684;
    Value *Hi = mul(X, k(LN10));
    Value *Lo = fma(X, k(LN10), B.CreateFNeg(Hi));
    Lo = fma(X, k(LN10 - (double)(float)LN10), Lo);
    return expHiLo(Hi, Lo);
  }

  /// A positive finite x as 2^e * (1 + f), 1 + f in [sqrt(1/2), sqrt(2)),
  /// with log(1 + f) = f + y (Cephes logf). Denormals are scaled first.
  struct LogParts {
    Value *E, *F, *Y;
  };
  LogParts logParts(Value *X) {
    Value *Tiny = lt(X, 1.17549435e-38);
    Value *XS = sel(Tiny, mul(X, k(8388608.0)), X);
    Value *Bits = B.CreateBitCast(XS, intTy());
    Value *E = B.CreateSIToFP(
        B.CreateSub(B.CreateLShr(Bits, ki(23)), ki(127)), Ty);
    E = sel(Tiny, sub(E, k(23.0)), E);
    Value *M = B.CreateBitCast(
        B.CreateOr(B.CreateAnd(Bits, ki(0x7fffff)), ki(0x3f800000)), Ty);
    Value *Big = gt(M, 1.41421356);
    M = sel(Big, mul(M, k(0.5)), M);
    E = sel(Big, add(E, k(1.0)), E);
    Value *F = sub(M, k(1.0));
    Value *Z = mul(F, F);
    Value *P = poly(F, {7.0376836292e-2, -1.1514610310e-1, 1.1676998740e-1,
                        -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1,
                        2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1});
    Value *Y = fma(k(-0.5), Z, mul(mul(F, Z), P));
    return {E, F, Y};
  }
  /// The value of log for the special inputs (NaN, negative, 0, infinity),
  /// else \p R.
  Value *logSpecial(Value *X, Value *R) {
    R = sel(B.CreateFCmpOEQ(X, k(INFINITY)), X, R);
    R = sel(B.CreateFCmpOEQ(X, k(0.0)), k(-INFINITY), R);
    return sel(B.CreateFCmpULT(X, k(0.0)), k(NAN), R);
  }
  /// log(x) for a positive finite x as a sum hi + lo.
  std::pair<Value *, Value *> logHiLo(Value *X) {
    LogParts L = logParts(X);
    auto [S, Err] = twoSum(L.F, L.Y);
    // e ln 2 in two parts, the first exact for |e| < 2^15.
    auto [Hi, Err2] = twoSum(mul(L.E, k(0.693359375)), S);
    Value *Lo = add(add(Err, Err2), mul(L.E, k(-2.12194440e-4)));
    return {Hi, Lo};
  }
  Value *logPrecise(Value *X) {
    auto [Hi, Lo] = logHiLo(X);
    return logSpecial(X, add(Hi, Lo));
  }
  /// log2 and log10: e log_b(2) + (f + y) log_b(e), each constant in a
  /// part with few bits and a correction (Cephes log10f).
  Value *logBase(Value *X, double EHi, double ELo, double L2Hi, double L2Lo) {
    LogParts L = logParts(X);
    Value *R = mul(L.Y, k(ELo));
    R = fma(L.F, k(ELo), R);
    R = fma(L.Y, k(EHi), R);
    R = fma(L.F, k(EHi), R);
    R = fma(L.E, k(L2Lo), R);
    R = fma(L.E, k(L2Hi), R);
    return logSpecial(X, R);
  }
  Value *log2Precise(Value *X) {
    // log2(e) = 1.4375 + 0.0051950..., log2(2) = 1.
    return logBase(X, 1.4375, M_LOG2E - 1.4375, 1.0, 0.0);
  }
  Value *log10Precise(Value *X) {
    return logBase(X, 0.43359375, 7.00731903251827651129e-4, 3.0078125e-1,
                   2.48745663981195213739e-4);
  }

  Value *powPrecise(Value *X, Value *Y) {
    Value *AX = abs(X);
    auto [LHi, LLo] = logHiLo(AX);
    Value *Hi = mul(Y, LHi);
    Value *Lo = fma(Y, LHi, B.CreateFNeg(Hi));
    Lo = fma(Y, LLo, Lo);
    Value *R = expHiLo(Hi, Lo);
    // log(0) = -inf and log(inf) = inf: the product is +-inf (or NaN for a
    // zero y, handled below), the result 0 or inf.
    Value *L = logSpecial(AX, LHi);
    Value *Inf = mul(Y, L);
    R = sel(B.CreateFCmpOEQ(abs(Inf), k(INFINITY)),
            sel(gt(Inf, 0.0), k(INFINITY), k(0.0)), R);
    // A negative x: an odd integer y gives the sign, any other finite
    // non-integer a NaN (for x = -inf, the infinity or zero above).
    Value *YInt = B.CreateFCmpOEQ(call(Intrinsic::trunc, {Y}), Y);
    Value *Half = mul(Y, k(0.5));
    Value *YOdd = B.CreateAnd(
        YInt, B.CreateFCmpONE(call(Intrinsic::trunc, {Half}), Half));
    Value *Neg = lt(copysign(k(1.0), X), 0.0);
    R = sel(B.CreateAnd(Neg, YOdd), B.CreateFNeg(R), R);
    Value *YFinite = B.CreateFCmpOLT(abs(Y), k(INFINITY));
    Value *NegFinite = B.CreateAnd(lt(X, 0.0), gt(X, -INFINITY));
    R = sel(B.CreateAnd(B.CreateAnd(NegFinite, YFinite), B.CreateNot(YInt)),
            k(NAN), R);
    // |x| = 1 with y infinite, x = 1, y = 0: 1 (even with a NaN).
    R = sel(B.CreateAnd(B.CreateFCmpOEQ(AX, k(1.0)),
                        B.CreateNot(YFinite)),
            k(1.0), R);
    R = sel(B.CreateFCmpUNO(X, Y), k(NAN), R);
    R = sel(B.CreateFCmpOEQ(X, k(1.0)), k(1.0), R);
    return sel(B.CreateFCmpOEQ(Y, k(0.0)), k(1.0), R);
  }

  /// sin(r) and cos(r) for |r| <= pi / 4 (Cephes sinf and cosf).
  std::pair<Value *, Value *> sinCosPolys(Value *R) {
    Value *Z = mul(R, R);
    Value *S = fma(mul(poly(Z, {-1.9515295891e-4, 8.3321608736e-3,
                                -1.6666654611e-1}),
                       Z),
                   R, R);
    Value *C = fma(mul(Z, Z),
                   poly(Z, {2.443315711809948e-5, -1.388731625493765e-3,
                            4.166664568298827e-2}),
                   fma(k(-0.5), Z, k(1.0)));
    return {S, C};
  }

  /// sin(x) and cos(x): x - j pi / 2 with pi / 2 in three parts, exact for
  /// |x| below 2^22, then Cephes' sinf and cosf polynomials by quadrant.
  /// Beyond that the hardware's reduction (afn) is used.
  std::pair<Value *, Value *> sinAndCos(Value *X) {
    Value *J = call(Intrinsic::roundeven, {mul(X, k(M_2_PI))});
    Value *NJ = B.CreateFNeg(J);
    Value *R = fma(NJ, k(1.57079637050628662109375), X);
    R = fma(NJ, k(-4.371139000189375e-8), R);
    R = fma(NJ, k(-1.7151245100059780e-15), R);
    Value *Small = lt(abs(X), 4194304.0);
    Value *Q = B.CreateFPToSI(sel(Small, J, k(0.0)), intTy());
    auto [S, C] = sinCosPolys(R);
    // sin: quadrants 1 and 3 take the cosine, 2 and 3 the minus sign; cos
    // is a quadrant ahead.
    auto Quadrant = [&](Value *Q) {
      Value *Odd = B.CreateICmpNE(B.CreateAnd(Q, ki(1)), ki(0));
      Value *Minus = B.CreateICmpNE(B.CreateAnd(Q, ki(2)), ki(0));
      Value *V = sel(Odd, C, S);
      return sel(Minus, B.CreateFNeg(V), V);
    };
    Value *Sin = Quadrant(Q), *Cos = Quadrant(B.CreateAdd(Q, ki(1)));
    auto Hw = [&](Intrinsic::ID ID) {
      Value *V = B.CreateIntrinsic(ID, {Ty}, {X});
      cast<Instruction>(V)->setHasApproxFunc(true);
      return V;
    };
    return {sel(Small, Sin, Hw(Intrinsic::sin)),
            sel(Small, Cos, Hw(Intrinsic::cos))};
  }

  /// The half not asked for is left to dead code elimination.
  Value *sinCos(Value *X, bool Cos) {
    auto [S, C] = sinAndCos(X);
    return Cos ? C : S;
  }

};

/// Computes \p Fn in float for values of \p Ty, a floating point type or a
/// vector of one.
template <typename FnT>
static Value *inFloat(IRBuilderBase &B, Type *Ty, ArrayRef<Value *> Args,
                      FnT Fn) {
  Type *F32 = Ty->getWithNewType(B.getFloatTy());
  SmallVector<Value *, 2> Ext;
  for (Value *A : Args)
    Ext.push_back(Ty == F32 ? A : B.CreateFPExt(A, F32));
  MathBuilder M(B, F32);
  Value *R = Fn(M, Ext);
  return Ty == F32 ? R : B.CreateFPTrunc(R, Ty);
}
} // namespace

bool AGX::canExpandMath(Intrinsic::ID ID, Type *Ty, bool Approx) {
  Type *E = Ty->getScalarType();
  switch (ID) {
  case Intrinsic::tan:
  case Intrinsic::asin:
  case Intrinsic::acos:
  case Intrinsic::atan:
  case Intrinsic::atan2:
  case Intrinsic::sinh:
  case Intrinsic::cosh:
  case Intrinsic::tanh:
    return E->isFloatTy() || E->isHalfTy() || E->isBFloatTy();
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::exp10:
  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::pow:
  case Intrinsic::sincos:
    // The hardware's are precise enough for half, and are what afn asks.
    return E->isFloatTy() && !Approx;
  default:
    return false;
  }
}

Value *AGX::expandMath(IRBuilderBase &B, Intrinsic::ID ID,
                       ArrayRef<Value *> Args) {
  return inFloat(B, Args[0]->getType(), Args,
                 [&](MathBuilder &M, ArrayRef<Value *> A) -> Value * {
                   switch (ID) {
                   case Intrinsic::tan:
                     return M.tan(A[0]);
                   case Intrinsic::asin:
                     return M.asin(A[0]);
                   case Intrinsic::acos:
                     return M.acos(A[0]);
                   case Intrinsic::atan:
                     return M.atan(A[0]);
                   case Intrinsic::atan2:
                     return M.atan2(A[0], A[1]);
                   case Intrinsic::sinh:
                     return M.sinh(A[0]);
                   case Intrinsic::cosh:
                     return M.cosh(A[0]);
                   case Intrinsic::tanh:
                     return M.tanh(A[0]);
                   case Intrinsic::sin:
                     return M.sinCos(A[0], false);
                   case Intrinsic::cos:
                     return M.sinCos(A[0], true);
                   case Intrinsic::exp:
                     return M.expPrecise(A[0]);
                   case Intrinsic::exp2:
                     return M.exp2Precise(A[0]);
                   case Intrinsic::exp10:
                     return M.exp10Precise(A[0]);
                   case Intrinsic::log:
                     return M.logPrecise(A[0]);
                   case Intrinsic::log2:
                     return M.log2Precise(A[0]);
                   case Intrinsic::log10:
                     return M.log10Precise(A[0]);
                   case Intrinsic::pow:
                     return M.powPrecise(A[0], A[1]);
                   case Intrinsic::sincos: {
                     auto [S, C] = M.sinAndCos(A[0]);
                     Value *R = PoisonValue::get(StructType::get(
                         S->getType()->getContext(), {S->getType(), C->getType()}));
                     R = B.CreateInsertValue(R, S, 0);
                     return B.CreateInsertValue(R, C, 1);
                   }
                   default:
                     llvm_unreachable("not an expanded math intrinsic");
                   }
                 });
}

Value *AGX::expandMath(IRBuilderBase &B, StringRef Fn, Value *X) {
  using MemberT = Value *(MathBuilder::*)(Value *);
  MemberT Member = StringSwitch<MemberT>(Fn)
                       .Case("asinh", &MathBuilder::asinh)
                       .Case("acosh", &MathBuilder::acosh)
                       .Case("atanh", &MathBuilder::atanh)
                       .Case("sinpi", &MathBuilder::sinpi)
                       .Case("cospi", &MathBuilder::cospi)
                       .Case("tanpi", &MathBuilder::tanpi)
                       .Default(nullptr);
  Type *E = X->getType()->getScalarType();
  if (!Member || !(E->isFloatTy() || E->isHalfTy() || E->isBFloatTy()))
    return nullptr;
  return inFloat(B, X->getType(), {X},
                 [&](MathBuilder &M, ArrayRef<Value *> A) {
                   return (M.*Member)(A[0]);
                 });
}
