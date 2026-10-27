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

  /// x - 2 round(x / 2), exactly: in [-1, 1], a zero with the sign of x.
  Value *halfTurns(Value *X) {
    Value *R = sub(X, mul(k(2.0), call(Intrinsic::roundeven,
                                      {mul(X, k(0.5))})));
    return sel(B.CreateFCmpOEQ(R, k(0.0)), copysign(k(0.0), X), R);
  }

  Value *sinpi(Value *X) {
    Value *R = halfTurns(X);
    Value *S = call(Intrinsic::sin, {mul(R, k(M_PI))});
    // Exact zeros at the integers (with the sign of the reduced argument).
    return sel(B.CreateFCmpOEQ(R, call(Intrinsic::trunc, {R})),
               copysign(k(0.0), R), S);
  }

  Value *cospi(Value *X) {
    // cos(pi r) = sin(pi (1/2 - |r|)), exactly 0 at the half integers.
    return sinpi(sub(k(0.5), abs(halfTurns(X))));
  }

  Value *tanpi(Value *X) { return div(sinpi(X), cospi(X)); }
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

bool AGX::canExpandMath(Intrinsic::ID ID, Type *Ty) {
  switch (ID) {
  case Intrinsic::tan:
  case Intrinsic::asin:
  case Intrinsic::acos:
  case Intrinsic::atan:
  case Intrinsic::atan2:
  case Intrinsic::sinh:
  case Intrinsic::cosh:
  case Intrinsic::tanh:
    break;
  default:
    return false;
  }
  Type *E = Ty->getScalarType();
  return E->isFloatTy() || E->isHalfTy() || E->isBFloatTy();
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
