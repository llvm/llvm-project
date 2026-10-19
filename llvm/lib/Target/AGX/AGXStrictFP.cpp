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
#include "AGXSoftF64.h"
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
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

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

/// A binary floating point format.
struct Format {
  unsigned Precision; // significand bits, including the implicit one
  int Bias;
  unsigned Width;
  int minLSB() const { return 2 - Bias - int(Precision); }
  uint32_t fracMask() const { return (1u << (Precision - 1)) - 1; }
  uint32_t expMask() const { return uint32_t(2 * Bias + 1); }
  uint32_t infBits() const { return expMask() << (Precision - 1); }
  uint32_t signBit() const { return 1u << (Width - 1); }
};
constexpr Format Binary32 = {24, 127, 32};
constexpr Format Binary16 = {11, 15, 16};

/// Rounding directions, numbered like FLT_ROUNDS and llvm.get.rounding.
enum RoundingCode : uint32_t {
  RoundTowardZero = 0,
  RoundNearestEven = 1,
  RoundUp = 2,
  RoundDown = 3,
};

/// sign * M * 2^E: Sign is i1, M is i64 and E is i32.
struct Term {
  Value *Sign;
  Value *M;
  Value *E;
};

/// Builds the integer implementations at the builder's insertion point.
class SoftFloat {
  IRBuilderBase &B;
  Type *I1, *I32, *I64, *F32;

public:
  /// The rounding direction (a RoundingCode, constant or not) of the
  /// results built by convert().
  Value *Rounding;

  explicit SoftFloat(IRBuilderBase &B)
      : B(B), I1(B.getInt1Ty()), I32(B.getInt32Ty()), I64(B.getInt64Ty()),
        F32(B.getFloatTy()), Rounding(B.getInt32(RoundNearestEven)) {}

  Value *roundingIs(RoundingCode C) {
    return B.CreateICmpEQ(Rounding, B.getInt32(C));
  }

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

  /// A finite value (bits in an i32) as sign * M * 2^E with M <
  /// 2^Precision. With Normalize, the most significant bit of a nonzero M is
  /// bit Precision - 1.
  Term unpack(Value *Bits, bool Normalize, Format Fmt = Binary32) {
    unsigned P = Fmt.Precision;
    Value *Exp = B.CreateAnd(B.CreateLShr(Bits, P - 1), c32(Fmt.expMask()));
    Value *Frac = B.CreateAnd(Bits, c32(Fmt.fracMask()));
    Value *Den = B.CreateICmpEQ(Exp, c32(0));
    Value *M = B.CreateSelect(Den, Frac, B.CreateOr(Frac, c32(1u << (P - 1))));
    Value *E = B.CreateSub(B.CreateSelect(Den, c32(1), Exp),
                           c32(Fmt.Bias + P - 1));
    if (Normalize) {
      Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
      Value *Sh = B.CreateSelect(B.CreateICmpEQ(M, c32(0)), c32(0),
                                 B.CreateSub(Lz, c32(32 - P)));
      M = B.CreateShl(M, Sh);
      E = B.CreateSub(E, Sh);
    }
    Value *Sign = B.CreateICmpNE(B.CreateAnd(Bits, c32(Fmt.signBit())), c32(0));
    return {Sign, B.CreateZExt(M, I64), E};
  }

  /// Round sign * M * 2^E in the direction Rounding: the bits of the result
  /// in Fmt (in an i32).
  Value *convert(const Term &T, Format Fmt = Binary32) {
    unsigned P = Fmt.Precision;
    Value *L = bitLength(T.M);
    // Shift so that P significant bits remain, or fewer for denormals.
    Value *S = smax(B.CreateSub(L, c32(P)), B.CreateSub(c32(Fmt.minLSB()), T.E));
    Value *Right = B.CreateICmpSGT(S, c32(0));
    Value *K = B.CreateSelect(Right, umin(S, c32(64)), c32(0));
    Value *Q = lshr(T.M, K);
    Value *Rem = B.CreateSub(T.M, shl(Q, K));
    Value *Half = shl(c64(1), B.CreateSub(smax(K, c32(1)), c32(1)));
    Value *Inexact = B.CreateICmpNE(Rem, c64(0));
    Value *Nearest = B.CreateOr(
        B.CreateICmpUGT(Rem, Half),
        B.CreateAnd(B.CreateICmpEQ(Rem, Half),
                    B.CreateICmpNE(B.CreateAnd(Q, c64(1)), c64(0))));
    Value *Up = B.CreateSelect(
        roundingIs(RoundNearestEven), Nearest,
        B.CreateSelect(roundingIs(RoundUp), B.CreateAnd(Inexact, B.CreateNot(T.Sign)),
                       B.CreateSelect(roundingIs(RoundDown),
                                      B.CreateAnd(Inexact, T.Sign), B.getFalse())));
    Up = B.CreateAnd(Up, Right);
    Value *Rounded = B.CreateAdd(Q, B.CreateZExt(Up, I64));
    Value *Left = shl(T.M, B.CreateSelect(Right, c32(0), B.CreateNeg(S)));
    Value *Mant = B.CreateTrunc(B.CreateSelect(Right, Rounded, Left), I32);
    // The exponent field minus one: the implicit bit of Mant adds the one,
    // and a rounding carry into bit P increments the exponent.
    Value *Biased = B.CreateAdd(B.CreateAdd(T.E, S), c32(Fmt.Bias + P - 2));
    Value *Bits = B.CreateAdd(B.CreateShl(Biased, P - 1), Mant);
    Value *Over = B.CreateOr(B.CreateICmpSGT(Biased, c32(2 * Fmt.Bias - 1)),
                             B.CreateICmpUGE(Bits, c32(Fmt.infBits())));
    // Overflows give infinity, or the largest finite value when rounding
    // toward zero or away from the sign's infinity.
    Value *Max = c32(Fmt.infBits() - 1), *Inf = c32(Fmt.infBits());
    Value *OverBits = B.CreateSelect(
        roundingIs(RoundTowardZero), Max,
        B.CreateSelect(roundingIs(RoundUp), B.CreateSelect(T.Sign, Max, Inf),
                       B.CreateSelect(roundingIs(RoundDown),
                                      B.CreateSelect(T.Sign, Inf, Max), Inf)));
    Bits = B.CreateSelect(Over, OverBits, Bits);
    Bits = B.CreateSelect(B.CreateICmpEQ(T.M, c64(0)), c32(0), Bits);
    return B.CreateOr(Bits, B.CreateSelect(T.Sign, c32(Fmt.signBit()), c32(0)));
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
    // An exact cancellation is +0, or -0 when rounding down.
    Value *Sign = B.CreateSelect(B.CreateICmpNE(M, c64(0)), B.CreateXor(SB, Neg),
                                 roundingIs(RoundDown));
    // Zero terms: the other term; the sum of two zeros is negative if both
    // are, or (rounding down) if either is.
    Value *XZero = B.CreateICmpEQ(X.M, c64(0));
    Value *YZero = B.CreateICmpEQ(Y.M, c64(0));
    Value *BothZero = B.CreateAnd(XZero, YZero);
    Value *ZeroSign = B.CreateSelect(roundingIs(RoundDown),
                                     B.CreateOr(X.Sign, Y.Sign),
                                     B.CreateAnd(X.Sign, Y.Sign));
    Sign = B.CreateSelect(BothZero, ZeroSign,
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

  /// Q, the correctly rounded positive magnitude of a result with sign Neg,
  /// rounded in the direction Rounding instead: Err has the sign of the exact
  /// magnitude minus Q.
  Value *adjustMagnitude(Value *Q, Value *Err, Value *Neg) {
    Value *Above = B.CreateFCmpOGT(Err, fconst(0.0f));
    Value *Below = B.CreateFCmpOLT(Err, fconst(0.0f));
    // Rounding up a positive or down a negative result increases the
    // magnitude; toward zero, down a positive or up a negative one decreases
    // it.
    Value *Away = B.CreateOr(B.CreateAnd(roundingIs(RoundUp), B.CreateNot(Neg)),
                             B.CreateAnd(roundingIs(RoundDown), Neg));
    Value *Toward = B.CreateOr(
        roundingIs(RoundTowardZero),
        B.CreateOr(B.CreateAnd(roundingIs(RoundUp), Neg),
                   B.CreateAnd(roundingIs(RoundDown), B.CreateNot(Neg))));
    Value *R = B.CreateSelect(B.CreateAnd(Away, Above), nextUp(Q), Q);
    return B.CreateSelect(B.CreateAnd(Toward, Below), nextDown(Q), R);
  }

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
    // Directed rounding: the remainder gives the direction of the error of
    // the magnitude.
    Value *Neg = B.CreateICmpSLT(B.CreateXor(AB, DB), c32(0));
    Q = adjustMagnitude(Q, fma(NegD, Q, A1), Neg);
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
    S = adjustMagnitude(S, fma(B.CreateFNeg(S), S, M), B.getFalse());
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

  /// Binary32 X + Y (or X - Y), rounded in the direction Rounding.
  Value *roundedAdd(Value *X, Value *Y, bool Sub) {
    Value *XB = bits(X), *YB = bits(Y);
    Term TX = unpack(XB, false), TY = unpack(YB, false);
    if (Sub)
      TY.Sign = B.CreateNot(TY.Sign);
    Value *Exact = flt(convert(add(TX, TY)));
    // Infinities and NaNs: denormal operands do not change the result.
    Value *HW = Sub ? B.CreateFSub(X, Y) : B.CreateFAdd(X, Y);
    return B.CreateSelect(B.CreateOr(isInfOrNaN(XB), isInfOrNaN(YB)), HW, Exact);
  }

  /// Binary32 X * Y, rounded in the direction Rounding.
  Value *roundedMul(Value *X, Value *Y) {
    Value *XB = bits(X), *YB = bits(Y);
    Value *Exact = flt(convert(mul(unpack(XB, false), unpack(YB, false))));
    Value *HW = B.CreateFMul(nonzeroNormal(X), nonzeroNormal(Y));
    return B.CreateSelect(B.CreateOr(isInfOrNaN(XB), isInfOrNaN(YB)), HW, Exact);
  }

  /// Binary32 X * Y + Z with one rounding in the direction Rounding.
  Value *roundedFma(Value *X, Value *Y, Value *Z) {
    Value *XB = bits(X), *YB = bits(Y), *ZB = bits(Z);
    Term P = mul(unpack(XB, false), unpack(YB, false));
    Value *Exact = flt(convert(add(P, unpack(ZB, false))));
    Value *HW = fma(nonzeroNormal(X), nonzeroNormal(Y), nonzeroNormal(Z));
    Value *Special = B.CreateOr(isInfOrNaN(XB),
                                B.CreateOr(isInfOrNaN(YB), isInfOrNaN(ZB)));
    return B.CreateSelect(Special, HW, Exact);
  }

  /// An integer converted to Fmt (float or half), rounded in the direction
  /// Rounding.
  Value *roundedFromInt(Value *V, bool Signed, Type *Ty) {
    Format Fmt = Ty->isHalfTy() ? Binary16 : Binary32;
    Value *W = Signed ? B.CreateSExtOrTrunc(V, I64) : B.CreateZExtOrTrunc(V, I64);
    Value *Neg = Signed ? B.CreateICmpSLT(W, c64(0)) : B.getFalse();
    Value *M = B.CreateSelect(Neg, B.CreateNeg(W), W);
    Value *Bits = convert({Neg, M, c32(0)}, Fmt);
    return toFloat(Bits, Ty);
  }

  /// A float converted to half, rounded in the direction Rounding.
  Value *roundedToHalf(Value *X) {
    Value *XB = bits(X);
    Value *Exact = toFloat(convert(unpack(XB, false), Binary16), B.getHalfTy());
    Value *HW = B.CreateFPTrunc(X, B.getHalfTy());
    return B.CreateSelect(isInfOrNaN(XB), HW, Exact);
  }

  /// Bits of Fmt in an i32 as a value of type Ty.
  Value *toFloat(Value *Bits, Type *Ty) {
    if (Ty->isHalfTy())
      return B.CreateBitCast(B.CreateTrunc(Bits, B.getInt16Ty()), Ty);
    return flt(Bits);
  }

  /// Round to an integral value in the direction Rounding.
  Value *roundedIntegral(Value *X) {
    auto Call = [&](Intrinsic::ID ID) { return B.CreateUnaryIntrinsic(ID, X); };
    Value *R = B.CreateSelect(roundingIs(RoundDown), Call(Intrinsic::floor),
                              Call(Intrinsic::roundeven));
    R = B.CreateSelect(roundingIs(RoundUp), Call(Intrinsic::ceil), R);
    return B.CreateSelect(roundingIs(RoundTowardZero), Call(Intrinsic::trunc), R);
  }

  /// flush(X) = |X| below the smallest normal ? +-0 : X, for float or half.
  Value *flushDenormal(Value *X) {
    Type *Ty = X->getType();
    APFloat Min = APFloat::getSmallestNormalized(Ty->getFltSemantics());
    Value *Small = B.CreateFCmpOLT(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                                   ConstantFP::get(Ty, Min));
    Value *Zero = B.CreateBinaryIntrinsic(Intrinsic::copysign,
                                          ConstantFP::get(Ty, 0.0), X);
    return B.CreateSelect(Small, Zero, X);
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
template <typename BuilderT = IRBuilder<>, typename FastFn, typename SlowFn>
static void replaceWithSlowPath(Instruction &I, FastFn BuildFast,
                                SlowFn BuildSlow) {
  Instruction *Next = I.getNextNode();
  BuilderT B(Next);
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
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Value *Limit = SF.fconst(0x1p-101f);
          auto Small = [&](Value *V) {
            return B.CreateFCmpOLT(
                B.CreateUnaryIntrinsic(Intrinsic::fabs, V), Limit);
          };
          return B.CreateAnd(Small(X), Small(Y));
        },
        [&](SoftFloat &SF, IRBuilderBase &B) {
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
        [&](SoftFloat &SF, IRBuilderBase &B) {
          // A product at most 2^-125 (or NaN) of nonzero operands.
          Value *Tiny = B.CreateFCmpULE(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, &I),
              SF.fconst(0x1p-125f));
          return B.CreateAnd(Tiny, B.CreateNot(B.CreateOr(
                                       SF.isZero(SF.bits(X)),
                                       SF.isZero(SF.bits(Y)))));
        },
        [&](SoftFloat &SF, IRBuilderBase &B) {
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
        I, [&](SoftFloat &SF, IRBuilderBase &) { return SF.fastDiv(X, Y); },
        [&](SoftFloat &SF, IRBuilderBase &) { return SF.div(X, Y); });
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
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y), *ZB = SF.bits(Z);
          Value *Den = B.CreateOr(SF.isDenormal(XB),
                                  B.CreateOr(SF.isDenormal(YB), SF.isDenormal(ZB)));
          Value *Tiny = B.CreateFCmpULE(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, &I), SF.fconst(0x1p-125f));
          Tiny = B.CreateAnd(Tiny, B.CreateNot(B.CreateOr(SF.isZero(XB),
                                                          SF.isZero(YB))));
          return B.CreateOr(Den, Tiny);
        },
        [&](SoftFloat &SF, IRBuilderBase &B) {
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
        I, [&](SoftFloat &SF, IRBuilderBase &) { return SF.fastSqrt(X); },
        [&](SoftFloat &SF, IRBuilderBase &) { return SF.sqrt(X); });
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
    replace(I, [&](SoftFloat &SF, IRBuilderBase &) {
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

/// Instructions built by the software implementations themselves, which
/// need no further lowering.
static constexpr const char *ExactTag = "agx.exact";
/// Instructions whose denormals follow IEEE although their function flushes
/// them (their operands and results were flushed explicitly if needed).
static constexpr const char *IEEETag = "agx.ieee";

/// Division and square root, which the hardware only approximates.
static bool isCorrectlyRounded(const Instruction &I) {
  if (!isF32(&I) || (isa<FPMathOperator>(I) &&
                     (I.hasApproxFunc() || I.hasAllowReciprocal())))
    return false;
  return I.getOpcode() == Instruction::FDiv ||
         getIntrinsicID(I) == Intrinsic::sqrt;
}

/// A correctly rounded operation in a flushing function: the IEEE lowering
/// on flushed operands, with a flushed result.
static bool lowerFlushing(Instruction &I) {
  IRBuilder<> B(&I);
  SoftFloat SF(B);
  for (Use &Op : I.operands())
    if (Op->getType()->isFloatTy())
      Op.set(SF.flushDenormal(Op));
  SmallVector<Use *, 4> Uses;
  for (Use &U : I.uses())
    Uses.push_back(&U);
  B.SetInsertPoint(I.getNextNode());
  Value *Result = SF.flushDenormal(&I);
  for (Use *U : Uses)
    U->set(Result);
  lowerIEEE(I);
  return true;
}

bool AGXStrictFP::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  bool FunctionIEEE = F.getDenormalMode(APFloat::IEEEsingle()).Output ==
                      DenormalMode::IEEE;
  bool CanonicalNaN = ST.hasCanonicalNaN();
  if (!ST.hasIEEEFP32() && !CanonicalNaN)
    return false;

  SmallVector<Instruction *, 32> Worklist;
  for (Instruction &I : instructions(F))
    if (!I.hasMetadata(ExactTag))
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
  if (ST.hasIEEEFP32())
    for (Instruction *I : Worklist) {
      if (FunctionIEEE || I->hasMetadata(IEEETag))
        Changed |= lowerIEEE(*I);
      else if (isCorrectlyRounded(*I))
        Changed |= lowerFlushing(*I);
    }
  return Changed;
}

//===----------------------------------------------------------------------===//
// AGXLowerFPEnv
//===----------------------------------------------------------------------===//

namespace {

/// Models the floating point mode register of llvm.agx.set.fpmode and
/// llvm.agx.get.fpmode with a value per program point, and lowers the
/// constrained floating point intrinsics: those that round to nearest even
/// become ordinary operations, the others are computed in software with the
/// rounding direction of their metadata or of the mode. Operations whose
/// denormal mode differs from their function's flush their operands and
/// results explicitly, and are tagged agx.ieee.
class AGXLowerFPEnv : public FunctionPass {
public:
  static char ID;
  AGXLowerFPEnv() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Floating Point Environment";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

/// A builder that tags the floating point operations it creates.
struct TaggingBuilder : IRBuilder<ConstantFolder, IRBuilderCallbackInserter> {
  explicit TaggingBuilder(Instruction *IP)
      : IRBuilder(IP->getContext(), ConstantFolder(),
                  IRBuilderCallbackInserter([](Instruction *I) {
                    if (isa<FPMathOperator>(I) || isa<CallBase>(I) ||
                        isa<FPTruncInst>(I))
                      I->setMetadata(ExactTag, MDNode::get(I->getContext(), {}));
                  })) {
    SetInsertPoint(IP);
  }
};

} // namespace

char AGXLowerFPEnv::ID = 0;

INITIALIZE_PASS_BEGIN(AGXLowerFPEnv, "agx-lower-fp-env",
                      "AGX Lower Floating Point Environment", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXLowerFPEnv, "agx-lower-fp-env",
                    "AGX Lower Floating Point Environment", false, false)

FunctionPass *llvm::createAGXLowerFPEnvPass() { return new AGXLowerFPEnv(); }

static constexpr uint32_t ModeHalfShift = 2;
static constexpr uint32_t ModeFloatFlush = 0x10, ModeHalfFlush = 0x20;

static bool flushes(const Function &F, const fltSemantics &Sem) {
  DenormalMode M = F.getDenormalMode(Sem);
  return M.Output != DenormalMode::IEEE && M.Output != DenormalMode::Dynamic;
}

static uint32_t roundingCode(RoundingMode RM) {
  switch (RM) {
  case RoundingMode::TowardZero:
    return RoundTowardZero;
  case RoundingMode::TowardPositive:
    return RoundUp;
  case RoundingMode::TowardNegative:
    return RoundDown;
  default:
    return RoundNearestEven;
  }
}

/// The constrained intrinsic as the ordinary operation.
static Value *lowerToPlain(ConstrainedFPIntrinsic *CI, IRBuilderBase &B) {
  Function &F = *CI->getFunction();
  SmallVector<Value *, 4> Args(CI->args().begin(),
                               CI->args().begin() + CI->getNonMetadataArgCount());
  switch (CI->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:                                                   \
    if (Instruction::isBinaryOp(Instruction::NAME))                            \
      return B.CreateBinOp(Instruction::BinaryOps(Instruction::NAME), Args[0], \
                           Args[1]);                                           \
    return B.CreateCast(Instruction::CastOps(Instruction::NAME), Args[0],      \
                        CI->getType());
#define CMP_INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC, DAGN)              \
  case Intrinsic::INTRINSIC:                                                   \
    return B.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(CI)->getPredicate(),   \
                        Args[0], Args[1]);
#define FUNCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                           \
  case Intrinsic::INTRINSIC: {                                                 \
    SmallVector<Type *, 4> ArgTys;                                             \
    for (Value *A : Args)                                                      \
      ArgTys.push_back(A->getType());                                          \
    SmallVector<Type *, 2> OverloadTys;                                        \
    FunctionType *FT = FunctionType::get(CI->getType(), ArgTys, false);        \
    if (!Intrinsic::isSignatureValid(Intrinsic::NAME, FT, OverloadTys))        \
      return nullptr;                                                          \
    return B.CreateCall(Intrinsic::getOrInsertDeclaration(                     \
                            F.getParent(), Intrinsic::NAME, OverloadTys),      \
                        Args);                                                 \
  }
#include "llvm/IR/ConstrainedOps.def"
  default:
    return nullptr;
  }
}

/// The constrained intrinsic computed in software with the rounding
/// direction Rounding (an i32 RoundingCode), or null if it does not depend on
/// the rounding direction.
static Value *lowerRounded(ConstrainedFPIntrinsic *CI, Value *Rounding,
                           bool CanonicalNaN) {
  Type *Ty = CI->getType();
  Value *R = nullptr;
  auto Arg = [&](unsigned I) { return CI->getArgOperand(I); };
  bool Float = Ty->isFloatTy();
  {
    TaggingBuilder B(CI);
    SoftFloat SF(B);
    SF.Rounding = Rounding;
    switch (CI->getIntrinsicID()) {
    case Intrinsic::experimental_constrained_fadd:
    case Intrinsic::experimental_constrained_fsub:
      if (Float)
        R = SF.roundedAdd(Arg(0), Arg(1), CI->getIntrinsicID() ==
                                              Intrinsic::experimental_constrained_fsub);
      break;
    case Intrinsic::experimental_constrained_fmul:
      if (Float)
        R = SF.roundedMul(Arg(0), Arg(1));
      break;
    case Intrinsic::experimental_constrained_fma:
    case Intrinsic::experimental_constrained_fmuladd:
      if (Float)
        R = SF.roundedFma(Arg(0), Arg(1), Arg(2));
      break;
    case Intrinsic::experimental_constrained_sitofp:
    case Intrinsic::experimental_constrained_uitofp:
      if (Float || Ty->isHalfTy())
        R = SF.roundedFromInt(Arg(0), CI->getIntrinsicID() ==
                                          Intrinsic::experimental_constrained_sitofp,
                              Ty);
      break;
    case Intrinsic::experimental_constrained_fptrunc:
      if (Ty->isHalfTy() && Arg(0)->getType()->isFloatTy())
        R = SF.roundedToHalf(Arg(0));
      break;
    case Intrinsic::experimental_constrained_rint:
    case Intrinsic::experimental_constrained_nearbyint:
      R = SF.roundedIntegral(Arg(0));
      break;
    default:
      break;
    }
  }
  bool DivOrSqrt = Float && (CI->getIntrinsicID() ==
                                 Intrinsic::experimental_constrained_fdiv ||
                             CI->getIntrinsicID() ==
                                 Intrinsic::experimental_constrained_sqrt);
  if (!R && !DivOrSqrt)
    return nullptr;
  if (DivOrSqrt) {
    // Placeholder replaced below by the correctly rounded division or root.
    Value *X = Arg(0);
    Value *Y = CI->getIntrinsicID() == Intrinsic::experimental_constrained_fdiv
                   ? Arg(1)
                   : nullptr;
    IRBuilder<> B(CI);
    Instruction *Stub = cast<Instruction>(B.CreateFreeze(X));
    replaceWithSlowPath<TaggingBuilder>(
        *Stub,
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return Y ? SF.fastDiv(X, Y) : SF.fastSqrt(X);
        },
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return Y ? SF.div(X, Y) : SF.sqrt(X);
        });
    // The phi that replaced the stub is the first instruction of the block
    // that now holds CI.
    R = &*CI->getParent()->begin();
  }
  if (CanonicalNaN) {
    IRBuilder<> B(CI);
    APFloat NaN(Ty->getFltSemantics(),
                APInt::getSignedMaxValue(Ty->getPrimitiveSizeInBits()));
    R = B.CreateSelect(B.CreateFCmpUNO(R, R), ConstantFP::get(Ty, NaN), R);
  }
  return R;
}

bool AGXLowerFPEnv::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);

  SmallVector<IntrinsicInst *, 8> ModeOps;
  SmallVector<ConstrainedFPIntrinsic *, 16> Constrained;
  for (Instruction &I : instructions(F)) {
    if (auto *CI = dyn_cast<ConstrainedFPIntrinsic>(&I))
      Constrained.push_back(CI);
    else if (auto *II = dyn_cast<IntrinsicInst>(&I);
             II && (II->getIntrinsicID() == Intrinsic::agx_set_fpmode ||
                    II->getIntrinsicID() == Intrinsic::agx_get_fpmode))
      ModeOps.push_back(II);
  }
  if (ModeOps.empty() && Constrained.empty())
    return false;

  // The mode as a variable, promoted to SSA values.
  bool FloatFlush = flushes(F, APFloat::IEEEsingle());
  bool HalfFlush = flushes(F, APFloat::IEEEhalf());
  uint32_t Initial = RoundNearestEven | (RoundNearestEven << ModeHalfShift) |
                     (FloatFlush ? ModeFloatFlush : 0) |
                     (HalfFlush ? ModeHalfFlush : 0);
  IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *Mode = Entry.CreateAlloca(Entry.getInt32Ty());
  Entry.CreateStore(Entry.getInt32(Initial), Mode);
  for (IntrinsicInst *II : ModeOps) {
    IRBuilder<> B(II);
    if (II->getIntrinsicID() == Intrinsic::agx_set_fpmode) {
      B.CreateStore(II->getArgOperand(0), Mode);
    } else {
      Value *V = B.CreateLoad(B.getInt32Ty(), Mode);
      II->replaceAllUsesWith(V);
    }
    II->eraseFromParent();
  }
  // The mode at each constrained intrinsic, kept through the promotion by a
  // freeze.
  DenseMap<ConstrainedFPIntrinsic *, FreezeInst *> ModeAt;
  for (ConstrainedFPIntrinsic *CI : Constrained) {
    IRBuilder<> B(CI);
    ModeAt[CI] = cast<FreezeInst>(B.CreateFreeze(B.CreateLoad(B.getInt32Ty(), Mode)));
  }
  DominatorTree DT(F);
  PromoteMemToReg({Mode}, DT);

  for (ConstrainedFPIntrinsic *CI : Constrained) {
    FreezeInst *Fr = ModeAt[CI];
    Value *Env = Fr->getOperand(0);
    IRBuilder<> B(CI);
    // The rounding of the result's format and the denormal mode of the
    // operands' format.
    Type *ResTy = CI->getType()->getScalarType();
    Type *OpTy = CI->getArgOperand(0)->getType()->getScalarType();
    bool ResHalf = !ResTy->isFloatTy();
    Value *Rounding;
    if (std::optional<RoundingMode> RM = CI->getRoundingMode();
        RM && *RM != RoundingMode::Dynamic)
      Rounding = B.getInt32(roundingCode(*RM));
    else
      Rounding = B.CreateAnd(ResHalf ? B.CreateLShr(Env, ModeHalfShift) : Env, 3);
    bool HasFPOperand = OpTy->isFloatingPointTy();
    Type *FlushTy = HasFPOperand ? OpTy : ResTy;
    bool FlushFloat = FlushTy->isFloatTy();
    Value *Flush = B.CreateICmpNE(
        B.CreateAnd(Env, FlushFloat ? ModeFloatFlush : ModeHalfFlush),
        B.getInt32(0));
    bool Default = FlushFloat ? FloatFlush : HalfFlush;
    bool KnownDefault = isa<ConstantInt>(Flush) &&
                        cast<ConstantInt>(Flush)->isOne() == Default;
    bool Mismatch = !KnownDefault &&
                    (FlushTy->isFloatTy() || FlushTy->isHalfTy()) &&
                    !CI->getType()->isVectorTy();
    SoftFloat SF(B);
    // A denormal mode that differs from the function's: flush the operands
    // explicitly (and the result below).
    if (Mismatch)
      for (unsigned I = 0, E = CI->getNonMetadataArgCount(); I < E; ++I)
        if (CI->getArgOperand(I)->getType() == FlushTy)
          CI->setArgOperand(I, B.CreateSelect(Flush,
                                              SF.flushDenormal(CI->getArgOperand(I)),
                                              CI->getArgOperand(I)));

    bool Nearest = isa<ConstantInt>(Rounding) &&
                   cast<ConstantInt>(Rounding)->getZExtValue() == RoundNearestEven;
    bool Double = ResTy->isDoubleTy() || OpTy->isDoubleTy();
    Value *R = Nearest || Double
                   ? nullptr
                   : lowerRounded(CI, Rounding, ST.hasCanonicalNaN());
    if (!R) {
      B.SetInsertPoint(CI);
      R = lowerToPlain(CI, B);
      // Double precision is computed in software in any rounding direction.
      if (Double && !Nearest)
        if (auto *RI = dyn_cast_or_null<Instruction>(R))
          if (Value *D = AGX::lowerDoubleArithmetic(*RI, Rounding,
                                                    ST.hasCanonicalNaN()))
            R = D;
    }
    if (!R)
      continue;
    if (Mismatch) {
      if (auto *RI = dyn_cast<Instruction>(R))
        RI->setMetadata(IEEETag, MDNode::get(F.getContext(), {}));
      if (R->getType() == FlushTy) {
        B.SetInsertPoint(CI);
        R = B.CreateSelect(Flush, SF.flushDenormal(R), R);
      }
    }
    R->takeName(CI);
    CI->replaceAllUsesWith(R);
    CI->eraseFromParent();
  }
  for (auto &[CI, Fr] : ModeAt) {
    Fr->replaceAllUsesWith(Fr->getOperand(0));
    Fr->eraseFromParent();
  }
  return true;
}
