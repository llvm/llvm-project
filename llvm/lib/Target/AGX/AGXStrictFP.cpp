//===-- AGXStrictFP.cpp - IEEE binary32 and canonical NaNs in software ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The 32-bit floating point ALU flushes denormal operands and results to zero,
// division and square root are not correctly rounded, and NaN results have the
// hardware's bit patterns. Two opt-in subtarget features implement the
// missing behavior in software:
//
//  * ieee-fp32: IEEE 754 binary32 arithmetic in functions whose denormal mode
//    for float is IEEE. Additions, multiplications and fused multiply-adds
//    keep the hardware instruction and recompute the result with integer
//    arithmetic only where the hardware result can be wrong (denormal
//    operands, or a result close to the denormal range). Comparisons,
//    minimum and maximum compare denormals as integers, and rounding
//    functions and the approximate functions (rcp, rsqrt, log2, exp2) treat
//    denormal operands or results explicitly. Division and square root are
//    correctly rounded: refined hardware estimates are corrected with exact
//    remainders for normal operands and results, and computed with integer
//    arithmetic otherwise.
//
//  * canonical-nan: NaN results of floating point arithmetic are the
//    canonical NaN of PTX, 0x7fffffff (0x7fff for half and bfloat).
//
// Exact results are represented as sign * M * 2^E with a 64-bit integer M.
// convert() rounds such a value to nearest even, producing denormals,
// normals or infinities as IEEE 754 requires; values that are not exact keep
// a sticky bit in the low bit of M.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSubtarget.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "agx-strict-fp"

namespace {

class AGXStrictFP : public FunctionPass {
public:
  static char ID;
  AGXStrictFP() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Strict FP"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

/// sign * M * 2^E: Sign is i1, M is i64 and E is i32.
struct Term {
  Value *Sign;
  Value *M;
  Value *E;
};

/// Builds the integer implementations at the builder's insertion point.
class SoftFloat {
  IRBuilder<> &B;
  Type *I1, *I32, *I64, *F32;

public:
  explicit SoftFloat(IRBuilder<> &B)
      : B(B), I1(B.getInt1Ty()), I32(B.getInt32Ty()), I64(B.getInt64Ty()),
        F32(B.getFloatTy()) {}

  Value *c32(uint32_t V) { return B.getInt32(V); }
  Value *c64(uint64_t V) { return B.getInt64(V); }
  Value *bits(Value *F) { return B.CreateBitCast(F, I32); }
  Value *flt(Value *I) { return B.CreateBitCast(I, F32); }
  Value *fconst(float V) { return ConstantFP::get(F32, V); }

  Value *umin(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::umin, A, X);
  }
  Value *smax(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::smax, A, X);
  }
  /// Bit length of a 64-bit value, as i32 (0 for 0).
  Value *bitLength(Value *M) {
    Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
    return B.CreateTrunc(B.CreateSub(c64(64), Lz), I32);
  }
  /// M << S and M >> S for 0 <= S; amounts of 64 or more give 0.
  Value *shl(Value *M, Value *S) {
    Value *R = B.CreateShl(M, B.CreateZExt(umin(S, c32(63)), I64));
    return B.CreateSelect(B.CreateICmpUGT(S, c32(63)), c64(0), R);
  }
  Value *lshr(Value *M, Value *S) {
    Value *R = B.CreateLShr(M, B.CreateZExt(umin(S, c32(63)), I64));
    return B.CreateSelect(B.CreateICmpUGT(S, c32(63)), c64(0), R);
  }
  /// M >> S with the bits shifted out ORed into the low bit, for 0 <= S.
  Value *lshrSticky(Value *M, Value *S) {
    Value *Q = lshr(M, S);
    Value *Lost = B.CreateICmpNE(B.CreateShl(Q, B.CreateZExt(umin(S, c32(63)), I64)), M);
    Lost = B.CreateOr(Lost, B.CreateAnd(B.CreateICmpUGT(S, c32(63)),
                                        B.CreateICmpNE(M, c64(0))));
    return B.CreateOr(Q, B.CreateZExt(Lost, I64));
  }

  /// The exponent field is 0: zero or denormal.
  Value *expZero(Value *Bits) {
    return B.CreateICmpEQ(B.CreateAnd(Bits, c32(0x7f800000)), c32(0));
  }
  Value *isDenormal(Value *Bits) {
    Value *Abs = B.CreateAnd(Bits, c32(0x7fffffff));
    return B.CreateICmpULT(B.CreateSub(Abs, c32(1)), c32(0x7fffff));
  }
  Value *isZero(Value *Bits) {
    return B.CreateICmpEQ(B.CreateAnd(Bits, c32(0x7fffffff)), c32(0));
  }
  Value *isInfOrNaN(Value *Bits) {
    return B.CreateICmpEQ(B.CreateAnd(Bits, c32(0x7f800000)), c32(0x7f800000));
  }
  Value *signBit(Value *Bits) { return B.CreateICmpSLT(Bits, c32(0)); }

  /// A finite value as sign * M * 2^E with M < 2^24. With Normalize, the
  /// most significant bit of a nonzero M is bit 23.
  Term unpack(Value *Bits, bool Normalize) {
    Value *Exp = B.CreateAnd(B.CreateLShr(Bits, 23), c32(0xff));
    Value *Frac = B.CreateAnd(Bits, c32(0x7fffff));
    Value *Den = B.CreateICmpEQ(Exp, c32(0));
    Value *M = B.CreateSelect(Den, Frac, B.CreateOr(Frac, c32(0x800000)));
    Value *E = B.CreateSub(B.CreateSelect(Den, c32(1), Exp), c32(150));
    if (Normalize) {
      Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
      Value *Sh = B.CreateSelect(B.CreateICmpEQ(M, c32(0)), c32(0),
                                 B.CreateSub(Lz, c32(8)));
      M = B.CreateShl(M, Sh);
      E = B.CreateSub(E, Sh);
    }
    return {signBit(Bits), B.CreateZExt(M, I64), E};
  }

  /// Round sign * M * 2^E to nearest even: the bits of the binary32 result.
  Value *convert(const Term &T) {
    Value *L = bitLength(T.M);
    // Shift so that 24 significant bits remain, or fewer for denormals.
    Value *S = smax(B.CreateSub(L, c32(24)), B.CreateSub(c32(-149), T.E));
    Value *Right = B.CreateICmpSGT(S, c32(0));
    Value *K = B.CreateSelect(Right, umin(S, c32(64)), c32(0));
    Value *Q = lshr(T.M, K);
    Value *Rem = B.CreateSub(T.M, shl(Q, K));
    Value *Half = shl(c64(1), B.CreateSub(smax(K, c32(1)), c32(1)));
    Value *Up = B.CreateOr(
        B.CreateICmpUGT(Rem, Half),
        B.CreateAnd(B.CreateICmpEQ(Rem, Half),
                    B.CreateICmpNE(B.CreateAnd(Q, c64(1)), c64(0))));
    Up = B.CreateAnd(Up, Right);
    Value *Rounded = B.CreateAdd(Q, B.CreateZExt(Up, I64));
    Value *Left = shl(T.M, B.CreateSelect(Right, c32(0), B.CreateNeg(S)));
    Value *Mant = B.CreateTrunc(B.CreateSelect(Right, Rounded, Left), I32);
    // The exponent field minus one: the implicit bit of Mant adds the one,
    // and a rounding carry into bit 24 increments the exponent.
    Value *Biased = B.CreateAdd(B.CreateAdd(T.E, S), c32(149));
    Value *Bits = B.CreateAdd(B.CreateShl(Biased, 23), Mant);
    Value *Over = B.CreateOr(B.CreateICmpSGT(Biased, c32(253)),
                             B.CreateICmpUGE(Bits, c32(0x7f800000)));
    Bits = B.CreateSelect(Over, c32(0x7f800000), Bits);
    Bits = B.CreateSelect(B.CreateICmpEQ(T.M, c64(0)), c32(0), Bits);
    return B.CreateOr(Bits, B.CreateShl(B.CreateZExt(T.Sign, I32), 31));
  }

  Term mul(const Term &X, const Term &Y) {
    return {B.CreateXor(X.Sign, Y.Sign), B.CreateMul(X.M, Y.M),
            B.CreateAdd(X.E, Y.E)};
  }

  /// X + Y, exact or with a sticky low bit.
  Term add(const Term &X, const Term &Y) {
    Value *TopX = B.CreateAdd(bitLength(X.M), X.E);
    Value *TopY = B.CreateAdd(bitLength(Y.M), Y.E);
    Value *Swap = B.CreateICmpSGT(TopY, TopX);
    Value *SB = B.CreateSelect(Swap, Y.Sign, X.Sign);
    Value *MB = B.CreateSelect(Swap, Y.M, X.M);
    Value *EB = B.CreateSelect(Swap, Y.E, X.E);
    Value *SS = B.CreateSelect(Swap, X.Sign, Y.Sign);
    Value *MS = B.CreateSelect(Swap, X.M, Y.M);
    Value *ES = B.CreateSelect(Swap, X.E, Y.E);
    // The larger term with its most significant bit at bit 61.
    Value *Up = B.CreateSub(c32(62), bitLength(MB));
    MB = shl(MB, Up);
    EB = B.CreateSub(EB, Up);
    Value *D = B.CreateSub(EB, ES);
    Value *Aligned = B.CreateSelect(B.CreateICmpSGT(D, c32(0)),
                                    lshrSticky(MS, D),
                                    shl(MS, B.CreateNeg(D)));
    Value *Same = B.CreateICmpEQ(SB, SS);
    Value *Sum = B.CreateSelect(Same, B.CreateAdd(MB, Aligned),
                                B.CreateSub(MB, Aligned));
    Value *Neg = B.CreateICmpSLT(Sum, c64(0));
    Value *M = B.CreateSelect(Neg, B.CreateNeg(Sum), Sum);
    // An exact cancellation is +0 when rounding to nearest.
    Value *Sign = B.CreateAnd(B.CreateXor(SB, Neg), B.CreateICmpNE(M, c64(0)));
    // Zero terms: the other term, and -0 + -0 = -0.
    Value *XZero = B.CreateICmpEQ(X.M, c64(0));
    Value *YZero = B.CreateICmpEQ(Y.M, c64(0));
    Value *BothZero = B.CreateAnd(XZero, YZero);
    Sign = B.CreateSelect(BothZero, B.CreateAnd(X.Sign, Y.Sign),
                          B.CreateSelect(XZero, Y.Sign,
                                         B.CreateSelect(YZero, X.Sign, Sign)));
    M = B.CreateSelect(XZero, Y.M, B.CreateSelect(YZero, X.M, M));
    Value *E = B.CreateSelect(XZero, Y.E, B.CreateSelect(YZero, X.E, EB));
    return {Sign, M, E};
  }

  /// A denormal operand replaced by a normal value of the same sign, for the
  /// hardware operations on infinities and NaNs.
  Value *nonzeroNormal(Value *X) {
    Value *Bits = bits(X);
    Value *One = B.CreateOr(B.CreateAnd(Bits, c32(0x80000000)), c32(0x3f800000));
    return flt(B.CreateSelect(isDenormal(Bits), One, Bits));
  }

  /// X * 2^64 for |X| < 2^-62, exact (denormals included).
  Value *scaleUp64(Value *X) {
    Value *Bits = bits(X);
    Value *Frac = B.CreateAnd(Bits, c32(0x7fffff));
    // A denormal is Frac * 2^-149, so X * 2^64 = Frac * 2^-85.
    Value *Den = B.CreateFMul(B.CreateUIToFP(Frac, F32), fconst(0x1p-85f));
    Den = flt(B.CreateOr(bits(Den), B.CreateAnd(Bits, c32(0x80000000))));
    return B.CreateSelect(isDenormal(Bits), Den,
                          B.CreateFMul(X, fconst(0x1p64f)));
  }

  Value *fma(Value *X, Value *Y, Value *Z) {
    return B.CreateIntrinsic(Intrinsic::fma, {F32}, {X, Y, Z});
  }
  /// The neighbors of a positive normal value.
  Value *nextUp(Value *X) { return flt(B.CreateAdd(bits(X), c32(1))); }
  Value *nextDown(Value *X) { return flt(B.CreateSub(bits(X), c32(1))); }

  /// Division of normal operands with a normal result, using the hardware:
  /// the mantissas (in [1, 2)) are divided with a refined reciprocal, and
  /// the quotient is corrected until the exact remainder shows that it is
  /// the correctly rounded one. Returns the result and whether it is valid.
  std::pair<Value *, Value *> fastDiv(Value *A, Value *D) {
    Value *AB = bits(A), *DB = bits(D);
    Value *EA = B.CreateAnd(B.CreateLShr(AB, 23), c32(0xff));
    Value *ED = B.CreateAnd(B.CreateLShr(DB, 23), c32(0xff));
    auto Normal = [&](Value *E) {
      return B.CreateICmpULT(B.CreateSub(E, c32(1)), c32(254));
    };
    Value *Mant = c32(0x7fffff), *One = c32(0x3f800000);
    Value *A1 = flt(B.CreateOr(B.CreateAnd(AB, Mant), One));
    Value *D1 = flt(B.CreateOr(B.CreateAnd(DB, Mant), One));
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {F32}, {D1});
    Value *NegD = B.CreateFNeg(D1);
    Value *Y = fma(fma(NegD, Y0, fconst(1.0f)), Y0, Y0);
    Value *Q0 = B.CreateFMul(A1, Y);
    Value *Q = fma(fma(NegD, Q0, A1), Y, Q0);
    for (int I = 0; I < 2; ++I) {
      // a/b lies in (q - gap below / 2, q + gap above / 2] around the
      // correctly rounded q; the remainder r = a - b * q is exact.
      Value *Up = nextUp(Q), *Down = nextDown(Q);
      Value *R = fma(NegD, Q, A1);
      Value *HalfUp = B.CreateFMul(D1, B.CreateFMul(B.CreateFSub(Up, Q), fconst(0.5f)));
      Value *HalfDown = B.CreateFMul(D1, B.CreateFMul(B.CreateFSub(Q, Down), fconst(0.5f)));
      Value *Odd = B.CreateTrunc(bits(Q), I1);
      Value *GoUp = B.CreateOr(B.CreateFCmpOGT(R, HalfUp),
                               B.CreateAnd(B.CreateFCmpOEQ(R, HalfUp), Odd));
      Value *NegHalfDown = B.CreateFNeg(HalfDown);
      Value *GoDown = B.CreateOr(B.CreateFCmpOLT(R, NegHalfDown),
                                 B.CreateAnd(B.CreateFCmpOEQ(R, NegHalfDown), Odd));
      Q = B.CreateSelect(GoUp, Up, B.CreateSelect(GoDown, Down, Q));
    }
    Value *QB = bits(Q);
    Value *Shift = B.CreateSub(EA, ED);
    Value *Exp = B.CreateAdd(B.CreateAnd(B.CreateLShr(QB, 23), c32(0xff)), Shift);
    Value *Valid = B.CreateAnd(B.CreateAnd(Normal(EA), Normal(ED)), Normal(Exp));
    Value *Sign = B.CreateAnd(B.CreateXor(AB, DB), c32(0x80000000));
    Value *R = B.CreateOr(B.CreateAdd(QB, B.CreateShl(Shift, 23)), Sign);
    return {flt(R), Valid};
  }

  /// Square root of a positive normal value, using the hardware: the root of
  /// the mantissa (in [1, 4)) is refined from rsqrt and corrected until the
  /// midpoint test of Tuckerman shows that it is correctly rounded.
  std::pair<Value *, Value *> fastSqrt(Value *X) {
    Value *XB = bits(X);
    Value *Valid = B.CreateICmpULT(B.CreateSub(XB, c32(0x00800000)),
                                   c32(0x7f000000));
    Value *K = B.CreateSub(B.CreateAnd(B.CreateLShr(XB, 23), c32(0xff)), c32(127));
    Value *Odd = B.CreateAnd(K, c32(1));
    Value *Exp = B.CreateSelect(B.CreateICmpNE(Odd, c32(0)), c32(0x40000000),
                                c32(0x3f800000));
    Value *M = flt(B.CreateOr(B.CreateAnd(XB, c32(0x7fffff)), Exp));
    Value *NegM = B.CreateFNeg(M);
    Value *Y = B.CreateIntrinsic(Intrinsic::agx_rsqrt, {F32}, {M});
    Value *S0 = B.CreateFMul(M, Y);
    Value *S = fma(fma(B.CreateFNeg(S0), S0, M), B.CreateFMul(Y, fconst(0.5f)), S0);
    for (int I = 0; I < 2; ++I) {
      // s is correctly rounded iff s * down(s) < m <= s * up(s) (there are no
      // ties for square roots).
      Value *Up = nextUp(S), *Down = nextDown(S);
      Value *Low = B.CreateFCmpOGE(fma(S, Down, NegM), fconst(0.0f));
      Value *High = B.CreateFCmpOLT(fma(S, Up, NegM), fconst(0.0f));
      S = B.CreateSelect(High, Up, B.CreateSelect(Low, Down, S));
    }
    Value *Half = B.CreateAShr(B.CreateSub(K, Odd), 1);
    Value *R = B.CreateAdd(bits(S), B.CreateShl(Half, 23));
    return {flt(R), Valid};
  }

  /// IEEE division of finite or special operands, correctly rounded.
  Value *div(Value *A, Value *D) {
    Value *AB = bits(A), *DB = bits(D);
    Term TA = unpack(AB, true), TD = unpack(DB, true);
    // Q = floor(MA * 2^26 / MD), with MA, MD in [2^23, 2^24): 26 or 27 bits.
    // A floating point estimate is exact to a few units; two integer
    // corrections make it exact.
    Value *X = B.CreateShl(TA.M, 26);
    Value *FD = B.CreateUIToFP(TD.M, F32);
    Value *Inv = B.CreateIntrinsic(Intrinsic::agx_rcp, {F32}, {FD});
    Value *Est = B.CreateFMul(B.CreateFMul(B.CreateUIToFP(TA.M, F32), Inv),
                              fconst(0x1p26f));
    Value *Q = B.CreateZExt(B.CreateFPToUI(Est, I32), I64);
    Value *R = B.CreateSub(X, B.CreateMul(Q, TD.M));
    Value *C = B.CreateFPToSI(
        B.CreateFMul(B.CreateSIToFP(R, F32), Inv), I32);
    Q = B.CreateAdd(Q, B.CreateSExt(C, I64));
    R = B.CreateSub(R, B.CreateMul(B.CreateSExt(C, I64), TD.M));
    for (int I = 0; I < 2; ++I) {
      Value *Low = B.CreateICmpSLT(R, c64(0));
      Q = B.CreateSelect(Low, B.CreateSub(Q, c64(1)), Q);
      R = B.CreateSelect(Low, B.CreateAdd(R, TD.M), R);
      Value *High = B.CreateICmpSGE(R, TD.M);
      Q = B.CreateSelect(High, B.CreateAdd(Q, c64(1)), Q);
      R = B.CreateSelect(High, B.CreateSub(R, TD.M), R);
    }
    Value *M = B.CreateOr(B.CreateShl(Q, 1),
                          B.CreateZExt(B.CreateICmpNE(R, c64(0)), I64));
    Value *E = B.CreateSub(B.CreateSub(TA.E, TD.E), c32(27));
    Value *Sign = B.CreateXor(TA.Sign, TD.Sign);
    Value *Bits = convert({Sign, M, E});

    // Zeros, infinities and NaNs.
    Value *SignBits = B.CreateShl(B.CreateZExt(Sign, I32), 31);
    Value *AZero = isZero(AB), *DZero = isZero(DB);
    Value *AInf = B.CreateICmpEQ(B.CreateAnd(AB, c32(0x7fffffff)), c32(0x7f800000));
    Value *DInf = B.CreateICmpEQ(B.CreateAnd(DB, c32(0x7fffffff)), c32(0x7f800000));
    Value *NaN = B.CreateOr(B.CreateFCmpUNO(A, D),
                            B.CreateOr(B.CreateAnd(AZero, DZero),
                                       B.CreateAnd(AInf, DInf)));
    Value *Inf = B.CreateOr(AInf, DZero);
    Value *Zero = B.CreateOr(AZero, DInf);
    Bits = B.CreateSelect(Zero, SignBits, Bits);
    Bits = B.CreateSelect(Inf, B.CreateOr(SignBits, c32(0x7f800000)), Bits);
    Bits = B.CreateSelect(NaN, c32(0x7fc00000), Bits);
    return flt(Bits);
  }

  /// IEEE square root, correctly rounded.
  Value *sqrt(Value *X) {
    Value *XB = bits(X);
    Term T = unpack(XB, true);
    // An even exponent: M in [2^23, 2^25).
    Value *Odd = B.CreateTrunc(T.E, I1);
    Value *M = B.CreateSelect(Odd, B.CreateShl(T.M, 1), T.M);
    Value *E = B.CreateSelect(Odd, B.CreateSub(T.E, c32(1)), T.E);
    // S = floor(sqrt(M * 2^28)), 26 or 27 bits.
    Value *W = B.CreateShl(M, 28);
    Value *FM = B.CreateUIToFP(M, F32);
    Value *Root = B.CreateFMul(
        FM, B.CreateIntrinsic(Intrinsic::agx_rsqrt, {F32}, {FM}));
    Value *S = B.CreateZExt(
        B.CreateFPToUI(B.CreateFMul(Root, fconst(0x1p14f)), I32), I64);
    Value *R = B.CreateSub(W, B.CreateMul(S, S));
    Value *Half = B.CreateIntrinsic(
        Intrinsic::agx_rcp, {F32},
        {B.CreateFMul(B.CreateUIToFP(S, F32), fconst(2.0f))});
    Value *C = B.CreateFPToSI(B.CreateFMul(B.CreateSIToFP(R, F32), Half), I32);
    S = B.CreateAdd(S, B.CreateSExt(C, I64));
    R = B.CreateSub(W, B.CreateMul(S, S));
    for (int I = 0; I < 2; ++I) {
      Value *Low = B.CreateICmpSLT(R, c64(0));
      R = B.CreateSelect(Low, B.CreateAdd(R, B.CreateSub(B.CreateShl(S, 1),
                                                         c64(1))),
                         R);
      S = B.CreateSelect(Low, B.CreateSub(S, c64(1)), S);
      Value *High = B.CreateICmpSGT(R, B.CreateShl(S, 1));
      R = B.CreateSelect(High, B.CreateSub(R, B.CreateAdd(B.CreateShl(S, 1),
                                                          c64(1))),
                         R);
      S = B.CreateSelect(High, B.CreateAdd(S, c64(1)), S);
    }
    Value *Root2 = B.CreateOr(B.CreateShl(S, 1),
                              B.CreateZExt(B.CreateICmpNE(R, c64(0)), I64));
    Value *RE = B.CreateSub(B.CreateAShr(B.CreateSub(E, c32(28)), 1), c32(1));
    Value *Bits = convert({B.getFalse(), Root2, RE});
    // sqrt(+-0) = +-0, sqrt(+inf) = +inf, NaN for negative values and NaNs.
    Bits = B.CreateSelect(B.CreateICmpEQ(XB, c32(0x7f800000)), XB, Bits);
    Bits = B.CreateSelect(
        B.CreateOr(B.CreateFCmpUNO(X, X),
                   B.CreateAnd(signBit(XB), B.CreateNot(isZero(XB)))),
        c32(0x7fc00000), Bits);
    Bits = B.CreateSelect(isZero(XB), XB, Bits);
    return flt(Bits);
  }

  /// A comparison operand as a signed integer, for zeros and denormals.
  Value *denormalOrder(Value *Bits) {
    Value *Mag = B.CreateAnd(Bits, c32(0x7fffff));
    return B.CreateSelect(signBit(Bits), B.CreateNeg(Mag), Mag);
  }

  /// fcmp that orders denormals.
  Value *compare(CmpInst::Predicate P, Value *X, Value *Y, Value *HW) {
    Value *XB = bits(X), *YB = bits(Y);
    Value *Both = B.CreateAnd(expZero(XB), expZero(YB));
    Value *VX = denormalOrder(XB), *VY = denormalOrder(YB);
    Value *Int;
    switch (P) {
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_UEQ:
      Int = B.CreateICmpEQ(VX, VY);
      break;
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_UNE:
      Int = B.CreateICmpNE(VX, VY);
      break;
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ULT:
      Int = B.CreateICmpSLT(VX, VY);
      break;
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULE:
      Int = B.CreateICmpSLE(VX, VY);
      break;
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_UGT:
      Int = B.CreateICmpSGT(VX, VY);
      break;
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGE:
      Int = B.CreateICmpSGE(VX, VY);
      break;
    default:
      return HW; // ord, uno, true, false: denormals change nothing.
    }
    return B.CreateSelect(Both, Int, HW);
  }

  /// X < Y, with denormals.
  Value *less(Value *X, Value *Y) {
    return compare(CmpInst::FCMP_OLT, X, Y, B.CreateFCmpOLT(X, Y));
  }

  /// minimumnum / maximumnum (and minnum / maxnum): a NaN operand is ignored,
  /// -0 < +0. With Propagate (minimum / maximum), NaN operands give NaN.
  Value *minMax(Value *X, Value *Y, bool Min, bool Propagate) {
    Value *XB = bits(X), *YB = bits(Y);
    Value *Lt = less(X, Y), *Gt = less(Y, X);
    Value *Pick = Min ? B.CreateSelect(Lt, XB, B.CreateSelect(Gt, YB, B.CreateOr(XB, YB)))
                      : B.CreateSelect(Gt, XB, B.CreateSelect(Lt, YB, B.CreateAnd(XB, YB)));
    Value *XNaN = B.CreateFCmpUNO(X, X), *YNaN = B.CreateFCmpUNO(Y, Y);
    if (Propagate) {
      Pick = B.CreateSelect(YNaN, YB, Pick);
      Pick = B.CreateSelect(XNaN, XB, Pick);
    } else {
      Pick = B.CreateSelect(YNaN, XB, Pick);
      Pick = B.CreateSelect(XNaN, YB, Pick);
    }
    return flt(Pick);
  }
};

} // namespace

char AGXStrictFP::ID = 0;

INITIALIZE_PASS_BEGIN(AGXStrictFP, DEBUG_TYPE, "AGX Strict FP", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXStrictFP, DEBUG_TYPE, "AGX Strict FP", false, false)

FunctionPass *llvm::createAGXStrictFPPass() { return new AGXStrictFP(); }

static bool isF32(const Value *V) { return V->getType()->isFloatTy(); }

static Intrinsic::ID getIntrinsicID(const Instruction &I) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&I))
    return II->getIntrinsicID();
  return Intrinsic::not_intrinsic;
}

/// Floating point operations whose NaN results are canonicalized.
static bool producesArithmeticNaN(const Instruction &I) {
  if (!I.getType()->isFPOrFPVectorTy())
    return false;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    return true;
  default:
    break;
  }
  switch (getIntrinsicID(I)) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::round:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::canonicalize:
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
    return true;
  default:
    return false;
  }
}

/// Replace the NaN results of I by the canonical NaN.
static void canonicalizeNaN(Instruction &I) {
  Type *Ty = I.getType();
  const fltSemantics &Sem = Ty->getScalarType()->getFltSemantics();
  unsigned Width = APFloat::getSizeInBits(Sem);
  APFloat NaN(Sem, APInt::getSignedMaxValue(Width));
  IRBuilder<> B(I.getNextNode());
  SmallVector<Use *, 4> Uses;
  for (Use &U : I.uses())
    Uses.push_back(&U);
  Value *R = B.CreateSelect(B.CreateFCmpUNO(&I, &I), ConstantFP::get(Ty, NaN),
                            &I);
  for (Use *U : Uses)
    U->set(R);
}

/// Replace I by Fast where Slow is false, and by the value computed by Build
/// in a conditional block otherwise. Cond is built right after I.
template <typename CondFn, typename SlowFn>
static void lowerWithSlowPath(Instruction &I, CondFn BuildCond,
                              SlowFn BuildSlow) {
  SmallVector<Use *, 4> Uses;
  for (Use &U : I.uses())
    Uses.push_back(&U);
  Instruction *Next = I.getNextNode();
  IRBuilder<> B(Next);
  SoftFloat SF(B);
  Value *Cond = BuildCond(SF, B);
  Instruction *ThenTerm = SplitBlockAndInsertIfThen(Cond, Next, false);
  B.SetInsertPoint(ThenTerm);
  Value *Slow = BuildSlow(SF, B);
  BasicBlock *Then = ThenTerm->getParent();
  BasicBlock *Head = Then->getSinglePredecessor();
  B.SetInsertPoint(Next->getParent(), Next->getParent()->begin());
  PHINode *Phi = B.CreatePHI(I.getType(), 2);
  Phi->addIncoming(&I, Head);
  Phi->addIncoming(Slow, Then);
  for (Use *U : Uses)
    U->set(Phi);
}

/// Replace I by the result of BuildFast, or of BuildSlow (in a conditional
/// block) where BuildFast says that its result is not valid.
template <typename FastFn, typename SlowFn>
static void replaceWithSlowPath(Instruction &I, FastFn BuildFast,
                                SlowFn BuildSlow) {
  Instruction *Next = I.getNextNode();
  IRBuilder<> B(Next);
  SoftFloat SF(B);
  auto [Fast, Valid] = BuildFast(SF, B);
  Instruction *ThenTerm =
      SplitBlockAndInsertIfThen(B.CreateNot(Valid), Next, false);
  B.SetInsertPoint(ThenTerm);
  Value *Slow = BuildSlow(SF, B);
  BasicBlock *Then = ThenTerm->getParent();
  BasicBlock *Head = Then->getSinglePredecessor();
  B.SetInsertPoint(Next->getParent(), Next->getParent()->begin());
  PHINode *Phi = B.CreatePHI(I.getType(), 2);
  Phi->addIncoming(Fast, Head);
  Phi->addIncoming(Slow, Then);
  Phi->takeName(&I);
  I.replaceAllUsesWith(Phi);
  I.eraseFromParent();
}

/// Replace I by the value computed by Build right after it.
template <typename BuildFn> static void replace(Instruction &I, BuildFn Build) {
  IRBuilder<> B(I.getNextNode());
  SoftFloat SF(B);
  Value *R = Build(SF, B);
  R->takeName(&I);
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
}

static bool lowerIEEE(Instruction &I) {
  Intrinsic::ID ID = getIntrinsicID(I);
  bool Approx = isa<FPMathOperator>(I) &&
                (I.hasApproxFunc() || I.hasAllowReciprocal());

  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub: {
    if (!isF32(&I))
      return false;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    bool Sub = I.getOpcode() == Instruction::FSub;
    // The hardware result is right unless both operands are below 2^-101:
    // otherwise the result is above 2^-102, so a denormal operand is below
    // half an ulp of it, and a cancellation of normal operands cannot reach
    // the denormal range.
    lowerWithSlowPath(
        I,
        [&](SoftFloat &SF, IRBuilder<> &B) {
          Value *Limit = SF.fconst(0x1p-101f);
          auto Small = [&](Value *V) {
            return B.CreateFCmpOLT(
                B.CreateUnaryIntrinsic(Intrinsic::fabs, V), Limit);
          };
          return B.CreateAnd(Small(X), Small(Y));
        },
        [&](SoftFloat &SF, IRBuilder<> &B) {
          Term TX = SF.unpack(SF.bits(X), false);
          Term TY = SF.unpack(SF.bits(Y), false);
          if (Sub)
            TY.Sign = B.CreateNot(TY.Sign);
          return SF.flt(SF.convert(SF.add(TX, TY)));
        });
    return true;
  }
  case Instruction::FMul: {
    if (!isF32(&I))
      return false;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    lowerWithSlowPath(
        I,
        [&](SoftFloat &SF, IRBuilder<> &B) {
          // A product at most 2^-125 (or NaN) of nonzero operands.
          Value *Tiny = B.CreateFCmpULE(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, &I),
              SF.fconst(0x1p-125f));
          return B.CreateAnd(Tiny, B.CreateNot(B.CreateOr(
                                       SF.isZero(SF.bits(X)),
                                       SF.isZero(SF.bits(Y)))));
        },
        [&](SoftFloat &SF, IRBuilder<> &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y);
          Value *Exact = SF.flt(SF.convert(
              SF.mul(SF.unpack(XB, false), SF.unpack(YB, false))));
          Value *Special = B.CreateFMul(SF.nonzeroNormal(X), SF.nonzeroNormal(Y));
          return B.CreateSelect(B.CreateOr(SF.isInfOrNaN(XB), SF.isInfOrNaN(YB)),
                                Special, Exact);
        });
    return true;
  }
  case Instruction::FDiv: {
    if (!isF32(&I) || Approx)
      return false;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    replaceWithSlowPath(
        I, [&](SoftFloat &SF, IRBuilder<> &) { return SF.fastDiv(X, Y); },
        [&](SoftFloat &SF, IRBuilder<> &) { return SF.div(X, Y); });
    return true;
  }
  case Instruction::FCmp: {
    if (!isF32(I.getOperand(0)))
      return false;
    auto *Cmp = cast<FCmpInst>(&I);
    Value *X = Cmp->getOperand(0), *Y = Cmp->getOperand(1);
    CmpInst::Predicate P = Cmp->getPredicate();
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *R = SF.compare(P, X, Y, &I);
    if (R == &I)
      return false;
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  default:
    break;
  }

  if (!isF32(&I))
    return false;
  auto *II = dyn_cast<IntrinsicInst>(&I);
  switch (ID) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd: {
    Value *X = II->getArgOperand(0), *Y = II->getArgOperand(1),
          *Z = II->getArgOperand(2);
    lowerWithSlowPath(
        I,
        [&](SoftFloat &SF, IRBuilder<> &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y), *ZB = SF.bits(Z);
          Value *Den = B.CreateOr(SF.isDenormal(XB),
                                  B.CreateOr(SF.isDenormal(YB), SF.isDenormal(ZB)));
          Value *Tiny = B.CreateFCmpULE(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, &I), SF.fconst(0x1p-125f));
          Tiny = B.CreateAnd(Tiny, B.CreateNot(B.CreateOr(SF.isZero(XB),
                                                          SF.isZero(YB))));
          return B.CreateOr(Den, Tiny);
        },
        [&](SoftFloat &SF, IRBuilder<> &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y), *ZB = SF.bits(Z);
          Term P = SF.mul(SF.unpack(XB, false), SF.unpack(YB, false));
          Value *Exact = SF.flt(SF.convert(SF.add(P, SF.unpack(ZB, false))));
          Value *Special = B.CreateIntrinsic(
              Intrinsic::fma, {B.getFloatTy()},
              {SF.nonzeroNormal(X), SF.nonzeroNormal(Y), SF.nonzeroNormal(Z)});
          Value *AnySpecial = B.CreateOr(
              SF.isInfOrNaN(XB), B.CreateOr(SF.isInfOrNaN(YB), SF.isInfOrNaN(ZB)));
          return B.CreateSelect(AnySpecial, Special, Exact);
        });
    return true;
  }
  case Intrinsic::sqrt: {
    if (Approx)
      return false;
    Value *X = II->getArgOperand(0);
    replaceWithSlowPath(
        I, [&](SoftFloat &SF, IRBuilder<> &) { return SF.fastSqrt(X); },
        [&](SoftFloat &SF, IRBuilder<> &) { return SF.sqrt(X); });
    return true;
  }
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum: {
    Value *X = II->getArgOperand(0), *Y = II->getArgOperand(1);
    bool Min = ID == Intrinsic::minnum || ID == Intrinsic::minimumnum ||
               ID == Intrinsic::minimum;
    bool Propagate = ID == Intrinsic::minimum || ID == Intrinsic::maximum;
    replace(I, [&](SoftFloat &SF, IRBuilder<> &) {
      return SF.minMax(X, Y, Min, Propagate);
    });
    return true;
  }
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::round: {
    // Denormals round to zero, except away from zero for floor and ceil.
    Value *X = II->getArgOperand(0);
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *XB = SF.bits(X);
    Value *Sign = B.CreateAnd(XB, SF.c32(0x80000000));
    Value *Fix = Sign;
    if (ID == Intrinsic::floor)
      Fix = B.CreateSelect(SF.signBit(XB), SF.c32(0xbf800000), SF.c32(0));
    if (ID == Intrinsic::ceil)
      Fix = B.CreateSelect(SF.signBit(XB), SF.c32(0x80000000), SF.c32(0x3f800000));
    Value *R = B.CreateSelect(SF.isDenormal(XB), SF.flt(Fix), &I);
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
  case Intrinsic::log2: {
    // Denormal operands are scaled by 2^64 first.
    Value *X = II->getArgOperand(0);
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *Scaled = B.CreateIntrinsic(ID, {B.getFloatTy()}, {SF.scaleUp64(X)});
    Value *Fixed;
    if (ID == Intrinsic::agx_rcp)
      Fixed = B.CreateFMul(Scaled, SF.fconst(0x1p64f));
    else if (ID == Intrinsic::agx_rsqrt)
      Fixed = B.CreateFMul(Scaled, SF.fconst(0x1p32f));
    else
      Fixed = B.CreateFSub(Scaled, SF.fconst(64.0f));
    Value *R = B.CreateSelect(SF.isDenormal(SF.bits(X)), Fixed, &I);
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  case Intrinsic::exp2: {
    // Results below 2^-126 are denormals: 2^(x + 64) * 2^-64.
    Value *X = II->getArgOperand(0);
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *Scaled = B.CreateIntrinsic(Intrinsic::exp2, {B.getFloatTy()},
                                      {B.CreateFAdd(X, SF.fconst(64.0f))});
    Value *N = B.CreateUnaryIntrinsic(
        Intrinsic::rint, B.CreateFMul(Scaled, SF.fconst(0x1p85f)));
    Value *Den = SF.flt(B.CreateFPToUI(N, B.getInt32Ty()));
    Value *InRange = B.CreateAnd(B.CreateFCmpOLT(X, SF.fconst(-126.0f)),
                                 B.CreateFCmpOGT(X, SF.fconst(-151.0f)));
    Value *R = B.CreateSelect(InRange, Den, &I);
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  default:
    return false;
  }
}

bool AGXStrictFP::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  bool IEEE = ST.hasIEEEFP32() &&
              F.getDenormalMode(APFloat::IEEEsingle()).Output ==
                  DenormalMode::IEEE;
  bool CanonicalNaN = ST.hasCanonicalNaN();
  if (!IEEE && !CanonicalNaN)
    return false;

  SmallVector<Instruction *, 32> Worklist;
  for (Instruction &I : instructions(F))
    Worklist.push_back(&I);

  bool Changed = false;
  // NaN canonicalization first: the IEEE lowering then replaces the
  // canonicalized value.
  if (CanonicalNaN)
    for (Instruction *I : Worklist)
      if (producesArithmeticNaN(*I)) {
        canonicalizeNaN(*I);
        Changed = true;
      }
  if (IEEE)
    for (Instruction *I : Worklist)
      Changed |= lowerIEEE(*I);
  return Changed;
}
