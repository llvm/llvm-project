//===-- AGXSoftF64.cpp - Double precision in software for AGX -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX has no double precision hardware. This pass replaces double precision
// arithmetic with integer arithmetic on the bits of the values: each operand
// becomes an exact sign * M * 2^E with a 128-bit M, the exact result (with a
// sticky bit where bits are dropped) is computed with integer operations, and
// a single rounding produces the bits of the result. The results are those of
// IEEE 754 with denormals, in all four rounding directions. Values that are
// only moved (loads, stores, phis, selects, arguments) stay double, and the
// type legalizer turns them into i64.
//
//===----------------------------------------------------------------------===//

#include "AGXSoftF64.h"
#include "AGX.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"

using namespace llvm;

#define DEBUG_TYPE "agx-soft-f64"

namespace {

/// A binary interchange format.
struct Fmt {
  unsigned P; // significand bits, including the implicit one
  int Bias;
  unsigned W;
  int minLSB() const { return 2 - Bias - int(P); }
  uint64_t expAll() const { return 2 * Bias + 1; }
  uint64_t fracMask() const { return (uint64_t(1) << (P - 1)) - 1; }
  uint64_t infBits() const { return expAll() << (P - 1); }
  uint64_t signBit() const { return uint64_t(1) << (W - 1); }
  uint64_t maxFinite() const { return infBits() - 1; }
  uint64_t quietBit() const { return uint64_t(1) << (P - 2); }
};
constexpr Fmt F64 = {53, 1023, 64};
constexpr Fmt F32 = {24, 127, 32};
constexpr Fmt F16 = {11, 15, 16};
constexpr Fmt BF16 = {8, 127, 16};

enum RoundingCode : uint32_t {
  RoundTowardZero = 0,
  RoundNearestEven = 1,
  RoundUp = 2,
  RoundDown = 3,
};

/// sign * M * 2^E, with Sign i1, M i128 and E i32.
struct Term {
  Value *Sign;
  Value *M;
  Value *E;
};

class SoftDouble {
  IRBuilderBase &B;
  Type *I1, *I32, *I64, *I128;
  Value *Rounding;
  bool CanonicalNaN;

public:
  /// The 128-bit divisions built, which the caller expands into integer
  /// arithmetic once the replacement is complete.
  SmallVector<BinaryOperator *, 1> Divisions;

  SoftDouble(IRBuilderBase &B, Value *Rounding, bool CanonicalNaN)
      : B(B), I1(B.getInt1Ty()), I32(B.getInt32Ty()), I64(B.getInt64Ty()),
        I128(B.getInt128Ty()), Rounding(Rounding), CanonicalNaN(CanonicalNaN) {}

  Value *c32(int64_t V) { return ConstantInt::getSigned(I32, V); }
  Value *c64(uint64_t V) { return B.getInt64(V); }
  Value *c128(uint64_t V) { return ConstantInt::get(I128, V); }
  Value *roundingIs(RoundingCode C) {
    return B.CreateICmpEQ(Rounding, B.getInt32(C));
  }
  Value *select(Value *C, Value *T, Value *F) { return B.CreateSelect(C, T, F); }
  Value *smax(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::smax, A, X);
  }
  Value *umin(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::umin, A, X);
  }

  /// Bit length of a 128-bit value, as i32.
  Value *bitLength(Value *M) {
    Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
    return B.CreateTrunc(B.CreateSub(c128(128), Lz), I32);
  }
  /// M << S and M >> S for an i32 S >= 0; amounts of 128 or more give 0.
  Value *shl(Value *M, Value *S) {
    Value *R = B.CreateShl(M, B.CreateZExt(umin(S, c32(127)), I128));
    return select(B.CreateICmpUGT(S, c32(127)), c128(0), R);
  }
  Value *lshr(Value *M, Value *S) {
    Value *R = B.CreateLShr(M, B.CreateZExt(umin(S, c32(127)), I128));
    return select(B.CreateICmpUGT(S, c32(127)), c128(0), R);
  }
  /// M >> S with the bits shifted out ORed into the lowest bit.
  Value *lshrSticky(Value *M, Value *S) {
    Value *R = lshr(M, S);
    Value *Lost = B.CreateICmpNE(shl(R, S), M);
    return B.CreateOr(R, B.CreateZExt(Lost, I128));
  }

  // Classification of bits (an i64 holding a value of format F).
  Value *magnitude(Value *Bits, Fmt F) {
    return B.CreateAnd(Bits, c64(F.signBit() - 1));
  }
  Value *signOf(Value *Bits, Fmt F) {
    return B.CreateICmpNE(B.CreateAnd(Bits, c64(F.signBit())), c64(0));
  }
  Value *isNaN(Value *Bits, Fmt F) {
    return B.CreateICmpUGT(magnitude(Bits, F), c64(F.infBits()));
  }
  Value *isInf(Value *Bits, Fmt F) {
    return B.CreateICmpEQ(magnitude(Bits, F), c64(F.infBits()));
  }
  Value *isZero(Value *Bits, Fmt F) {
    return B.CreateICmpEQ(magnitude(Bits, F), c64(0));
  }
  Value *isInfOrNaN(Value *Bits, Fmt F) {
    return B.CreateICmpUGE(magnitude(Bits, F), c64(F.infBits()));
  }

  /// The NaN result of an operation with the given operands of format In,
  /// in format Out: the canonical NaN, or the first NaN operand quieted
  /// (the default NaN if there is none).
  Value *nanResult(ArrayRef<Value *> Ops, Fmt In, Fmt Out) {
    if (CanonicalNaN)
      return c64(Out.signBit() - 1);
    Value *R = c64(Out.infBits() | Out.quietBit());
    for (Value *X : reverse(Ops)) {
      Value *Payload = B.CreateAnd(X, c64(In.fracMask()));
      Payload = In.P > Out.P ? B.CreateLShr(Payload, In.P - Out.P)
                             : B.CreateShl(Payload, Out.P - In.P);
      Value *Q = B.CreateOr(B.CreateOr(Payload, c64(Out.infBits() | Out.quietBit())),
                            select(signOf(X, In), c64(Out.signBit()), c64(0)));
      R = select(isNaN(X, In), Q, R);
    }
    return R;
  }

  Term unpack(Value *Bits, Fmt F) {
    Value *Exp = B.CreateAnd(B.CreateLShr(Bits, F.P - 1), c64(F.expAll()));
    Value *Frac = B.CreateAnd(Bits, c64(F.fracMask()));
    Value *Denormal = B.CreateICmpEQ(Exp, c64(0));
    Value *M = select(Denormal, Frac, B.CreateOr(Frac, c64(F.fracMask() + 1)));
    Value *E = B.CreateTrunc(select(Denormal, c64(1), Exp), I32);
    E = B.CreateSub(E, c32(F.Bias + int(F.P) - 1));
    return {signOf(Bits, F), B.CreateZExt(M, I128), E};
  }

  /// T rounded to format F in the rounding direction, as bits in an i64.
  /// T.M may hold a sticky bit below the bits that matter.
  Value *round(Term T, Fmt F) {
    Value *L = bitLength(T.M);
    Value *Shift = smax(B.CreateSub(L, c32(F.P)), B.CreateSub(c32(F.minLSB()), T.E));
    Value *Right = B.CreateICmpSGT(Shift, c32(0));
    Value *RS = smax(Shift, c32(0));
    Value *Q = select(Right, lshr(T.M, RS),
                      shl(T.M, B.CreateNeg(B.CreateBinaryIntrinsic(Intrinsic::smin, Shift, c32(0)))));
    Value *Rem = B.CreateSub(T.M, shl(lshr(T.M, RS), RS));
    Value *GuardS = B.CreateSub(RS, c32(1));
    Value *Guard = B.CreateTrunc(lshr(Rem, umin(GuardS, c32(128))), I1);
    // The bits below the guard bit; all of them if the guard bit is above
    // bit 127.
    Value *Half = shl(c128(1), GuardS);
    Value *Sticky = B.CreateICmpNE(B.CreateAnd(Rem, B.CreateSub(Half, c128(1))), c128(0));
    Guard = B.CreateAnd(Guard, Right);
    Sticky = B.CreateAnd(Sticky, Right);
    Value *Odd = B.CreateTrunc(Q, I1);
    Value *Inexact = B.CreateOr(Guard, Sticky);
    Value *IncNearest = B.CreateAnd(Guard, B.CreateOr(Sticky, Odd));
    Value *Inc = select(roundingIs(RoundNearestEven), IncNearest,
                        select(roundingIs(RoundUp), B.CreateAnd(Inexact, B.CreateNot(T.Sign)),
                               select(roundingIs(RoundDown), B.CreateAnd(Inexact, T.Sign),
                                      B.getFalse())));
    Q = B.CreateAdd(Q, B.CreateZExt(Inc, I128));
    // bits = k << (P - 1) + Q, with k the biased exponent of the lowest bit
    // of Q; a carry out of Q increments the exponent.
    Value *K = B.CreateSub(B.CreateAdd(T.E, Shift), c32(F.minLSB()));
    K = umin(K, c32(F.expAll() + 1));
    Value *Bits = B.CreateAdd(B.CreateShl(B.CreateZExt(K, I64), F.P - 1),
                              B.CreateTrunc(Q, I64));
    Bits = select(B.CreateICmpEQ(T.M, c128(0)), c64(0), Bits);
    Value *Overflow = B.CreateICmpUGE(Bits, c64(F.infBits()));
    // Overflows round to infinity or to the largest finite value.
    Value *ToInf = select(roundingIs(RoundNearestEven), B.getTrue(),
                          select(roundingIs(RoundUp), B.CreateNot(T.Sign),
                                 select(roundingIs(RoundDown), T.Sign, B.getFalse())));
    Bits = select(Overflow, select(ToInf, c64(F.infBits()), c64(F.maxFinite())), Bits);
    return B.CreateOr(Bits, select(T.Sign, c64(F.signBit()), c64(0)));
  }

  /// The exact sum of A and B (with M of at most 126 bits), with a sticky bit.
  Term addTerms(Term X, Term Y) {
    auto Normalize = [&](Term T) {
      Value *S = B.CreateSub(c32(126), bitLength(T.M));
      return Term{T.Sign, shl(T.M, S), B.CreateSub(T.E, S)};
    };
    X = Normalize(X);
    Y = Normalize(Y);
    // Order by magnitude; zeros are the smallest.
    Value *XZero = B.CreateICmpEQ(X.M, c128(0));
    Value *YZero = B.CreateICmpEQ(Y.M, c128(0));
    Value *YBigger = B.CreateOr(
        B.CreateICmpSGT(Y.E, X.E),
        B.CreateAnd(B.CreateICmpEQ(Y.E, X.E), B.CreateICmpUGT(Y.M, X.M)));
    YBigger = B.CreateAnd(B.CreateNot(YZero), B.CreateOr(XZero, YBigger));
    Term Big{select(YBigger, Y.Sign, X.Sign), select(YBigger, Y.M, X.M),
             select(YBigger, Y.E, X.E)};
    Term Small{select(YBigger, X.Sign, Y.Sign), select(YBigger, X.M, Y.M),
               select(YBigger, X.E, Y.E)};
    Value *D = B.CreateSub(Big.E, Small.E);
    Value *SM = select(B.CreateICmpEQ(Small.M, c128(0)), c128(0),
                       lshrSticky(Small.M, D));
    Value *Same = B.CreateICmpEQ(Big.Sign, Small.Sign);
    Value *M = select(Same, B.CreateAdd(Big.M, SM), B.CreateSub(Big.M, SM));
    return {Big.Sign, M, Big.E};
  }

  /// The sign of an exact zero sum of operands with signs SX and SY that
  /// are both zero (BothZero) or cancel.
  Value *zeroSumSign(Value *BothZero, Value *SX, Value *SY) {
    Value *Down = roundingIs(RoundDown);
    return select(BothZero, select(Down, B.CreateOr(SX, SY), B.CreateAnd(SX, SY)), Down);
  }

  Value *withZeroSign(Term T, Value *ZeroSign, Fmt F = F64) {
    // round() gives the sign of T to a zero.
    return round({select(B.CreateICmpEQ(T.M, c128(0)), ZeroSign, T.Sign), T.M, T.E}, F);
  }

  Value *add(Value *X, Value *Y, bool Sub) {
    if (Sub)
      Y = B.CreateXor(Y, c64(F64.signBit()));
    Term TX = unpack(X, F64), TY = unpack(Y, F64);
    Term S = addTerms(TX, TY);
    Value *R = withZeroSign(
        S, zeroSumSign(B.CreateAnd(isZero(X, F64), isZero(Y, F64)), TX.Sign, TY.Sign));
    // Infinities: the infinite operand, or NaN for opposite infinities.
    Value *XInf = isInf(X, F64), *YInf = isInf(Y, F64);
    Value *Special = select(XInf, X, Y);
    Value *NaN = B.CreateOr(B.CreateOr(isNaN(X, F64), isNaN(Y, F64)),
                            B.CreateAnd(B.CreateAnd(XInf, YInf), B.CreateICmpNE(X, Y)));
    R = select(B.CreateOr(XInf, YInf), Special, R);
    Value *OrigY = Sub ? B.CreateXor(Y, c64(F64.signBit())) : Y;
    return select(NaN, nanResult({X, OrigY}, F64, F64), R);
  }

  Value *mul(Value *X, Value *Y) {
    Term TX = unpack(X, F64), TY = unpack(Y, F64);
    Value *Sign = B.CreateXor(TX.Sign, TY.Sign);
    Value *R = round({Sign, B.CreateMul(TX.M, TY.M), B.CreateAdd(TX.E, TY.E)}, F64);
    Value *XInf = isInf(X, F64), *YInf = isInf(Y, F64);
    Value *Inf = B.CreateOr(c64(F64.infBits()), select(Sign, c64(F64.signBit()), c64(0)));
    Value *NaN = B.CreateOr(
        B.CreateOr(isNaN(X, F64), isNaN(Y, F64)),
        B.CreateOr(B.CreateAnd(XInf, isZero(Y, F64)), B.CreateAnd(YInf, isZero(X, F64))));
    R = select(B.CreateOr(XInf, YInf), Inf, R);
    return select(NaN, nanResult({X, Y}, F64, F64), R);
  }

  Value *fma(Value *X, Value *Y, Value *Z, Fmt F = F64) {
    Term TX = unpack(X, F), TY = unpack(Y, F), TZ = unpack(Z, F);
    Term P{B.CreateXor(TX.Sign, TY.Sign), B.CreateMul(TX.M, TY.M),
           B.CreateAdd(TX.E, TY.E)};
    Term S = addTerms(P, TZ);
    Value *PZero = B.CreateOr(isZero(X, F), isZero(Y, F));
    Value *R = withZeroSign(
        S, zeroSumSign(B.CreateAnd(PZero, isZero(Z, F)), P.Sign, TZ.Sign), F);
    Value *XInf = isInf(X, F), *YInf = isInf(Y, F), *ZInf = isInf(Z, F);
    Value *PInf = B.CreateOr(XInf, YInf);
    Value *PInfBits = B.CreateOr(c64(F.infBits()),
                                 select(P.Sign, c64(F.signBit()), c64(0)));
    Value *NaN = B.CreateOr(B.CreateOr(isNaN(X, F), isNaN(Y, F)), isNaN(Z, F));
    NaN = B.CreateOr(NaN, B.CreateOr(B.CreateAnd(XInf, isZero(Y, F)),
                                     B.CreateAnd(YInf, isZero(X, F))));
    NaN = B.CreateOr(NaN, B.CreateAnd(B.CreateAnd(PInf, ZInf),
                                      B.CreateICmpNE(P.Sign, TZ.Sign)));
    R = select(ZInf, Z, R);
    R = select(PInf, PInfBits, R);
    return select(NaN, nanResult({X, Y, Z}, F, F), R);
  }

  Value *div(Value *X, Value *Y) {
    Term TX = unpack(X, F64), TY = unpack(Y, F64);
    Value *Sign = B.CreateXor(TX.Sign, TY.Sign);
    // The dividend with its top bit at 126 over the divisor with its top bit
    // at 63: a quotient of at least 63 bits.
    Value *SX = B.CreateSub(c32(126), bitLength(TX.M));
    Value *SY = B.CreateSub(c32(63), bitLength(TY.M));
    Value *N = shl(TX.M, SX);
    Value *Dv = shl(TY.M, SY);
    Value *SafeDv = select(B.CreateICmpEQ(Dv, c128(0)), c128(1), Dv);
    Value *Q = B.CreateUDiv(N, SafeDv);
    Divisions.push_back(cast<BinaryOperator>(Q));
    Value *Inexact = B.CreateICmpNE(B.CreateMul(Q, SafeDv), N);
    Value *M = B.CreateOr(B.CreateShl(Q, 1), B.CreateZExt(Inexact, I128));
    Value *E = B.CreateSub(B.CreateSub(TX.E, SX), B.CreateSub(TY.E, SY));
    E = B.CreateSub(E, c32(1));
    Value *R = round({Sign, M, E}, F64);
    Value *SignBits = select(Sign, c64(F64.signBit()), c64(0));
    Value *XZero = isZero(X, F64), *YZero = isZero(Y, F64);
    Value *XInf = isInf(X, F64), *YInf = isInf(Y, F64);
    R = select(B.CreateOr(XZero, YInf), SignBits, R);
    R = select(B.CreateOr(XInf, YZero), B.CreateOr(c64(F64.infBits()), SignBits), R);
    Value *NaN = B.CreateOr(B.CreateOr(isNaN(X, F64), isNaN(Y, F64)),
                            B.CreateOr(B.CreateAnd(XZero, YZero), B.CreateAnd(XInf, YInf)));
    return select(NaN, nanResult({X, Y}, F64, F64), R);
  }

  /// The integer square root of the i128 N (and whether it is exact), with a
  /// loop that produces a bit per iteration.
  std::pair<Value *, Value *> isqrt(Value *N) {
    BasicBlock *Pre = B.GetInsertBlock();
    Instruction *At = &*B.GetInsertPoint();
    BasicBlock *Tail = Pre->splitBasicBlock(At->getIterator(), "isqrt.end");
    Function *F = Pre->getParent();
    BasicBlock *Loop = BasicBlock::Create(B.getContext(), "isqrt", F, Tail);
    Pre->getTerminator()->setSuccessor(0, Loop);
    B.SetInsertPoint(Loop);
    PHINode *I = B.CreatePHI(I32, 2);
    PHINode *Rem = B.CreatePHI(I128, 2);
    PHINode *Root = B.CreatePHI(I128, 2);
    I->addIncoming(c32(63), Pre);
    Rem->addIncoming(c128(0), Pre);
    Root->addIncoming(c128(0), Pre);
    Value *Pair = B.CreateAnd(
        B.CreateLShr(N, B.CreateZExt(B.CreateShl(I, 1), I128)), c128(3));
    Value *R2 = B.CreateOr(B.CreateShl(Rem, 2), Pair);
    Value *Rt = B.CreateShl(Root, 1);
    Value *T = B.CreateOr(B.CreateShl(Rt, 1), c128(1));
    Value *Ge = B.CreateICmpUGE(R2, T);
    Value *NewRem = select(Ge, B.CreateSub(R2, T), R2);
    Value *NewRoot = B.CreateOr(Rt, B.CreateZExt(Ge, I128));
    Value *NextI = B.CreateSub(I, c32(1));
    I->addIncoming(NextI, Loop);
    Rem->addIncoming(NewRem, Loop);
    Root->addIncoming(NewRoot, Loop);
    B.CreateCondBr(B.CreateICmpEQ(I, c32(0)), Tail, Loop);
    B.SetInsertPoint(At);
    PHINode *RootOut = B.CreatePHI(I128, 1);
    RootOut->addIncoming(NewRoot, Loop);
    PHINode *RemOut = B.CreatePHI(I128, 1);
    RemOut->addIncoming(NewRem, Loop);
    return {RootOut, B.CreateICmpEQ(RemOut, c128(0))};
  }

  Value *sqrt(Value *X) {
    Term TX = unpack(X, F64);
    // The radicand with its top bit at 125 or 126 and an even exponent.
    Value *S = B.CreateSub(c32(126), bitLength(TX.M));
    Value *Odd = B.CreateTrunc(B.CreateSub(TX.E, S), I1);
    S = B.CreateSub(S, B.CreateZExt(Odd, I32));
    Value *N = shl(TX.M, S);
    Value *E = B.CreateSub(TX.E, S);
    auto [Root, Exact] = isqrt(N);
    Value *M = B.CreateOr(B.CreateShl(Root, 1), B.CreateZExt(B.CreateNot(Exact), I128));
    Value *R = round({B.getFalse(), M, B.CreateSub(B.CreateAShr(E, 1), c32(1))}, F64);
    R = select(B.CreateOr(isZero(X, F64), B.CreateICmpEQ(X, c64(F64.infBits()))), X, R);
    Value *NaN = B.CreateOr(isNaN(X, F64),
                            B.CreateAnd(signOf(X, F64), B.CreateNot(isZero(X, F64))));
    return select(NaN, nanResult({X}, F64, F64), R);
  }

  /// A value of format In converted to format Out.
  Value *convert(Value *X, Fmt In, Fmt Out) {
    Value *R = round(unpack(X, In), Out);
    Value *Inf = B.CreateOr(c64(Out.infBits()),
                            select(signOf(X, In), c64(Out.signBit()), c64(0)));
    R = select(isInf(X, In), Inf, R);
    return select(isNaN(X, In), nanResult({X}, In, Out), R);
  }

  Value *fromInt(Value *V, bool Signed) {
    unsigned N = V->getType()->getIntegerBitWidth();
    Value *Neg = Signed ? B.CreateICmpSLT(V, ConstantInt::get(V->getType(), 0))
                        : B.getFalse();
    Value *Mag = Signed ? select(Neg, B.CreateNeg(V), V) : V;
    Value *M = N < 128 ? B.CreateZExt(Mag, I128) : Mag;
    return round({Neg, M, c32(0)}, F64);
  }

  /// X truncated toward zero to an integer of the type Ty, saturated (NaN
  /// gives 0).
  Value *toInt(Value *X, Type *Ty, bool Signed) {
    unsigned N = Ty->getIntegerBitWidth();
    Term T = unpack(X, F64);
    Value *E = T.E;
    Value *Big = B.CreateICmpSGT(E, c32(int(N)));
    Value *V = select(B.CreateICmpSGE(E, c32(0)),
                      shl(T.M, umin(E, c32(127))),
                      lshr(T.M, B.CreateNeg(B.CreateBinaryIntrinsic(Intrinsic::smin, E, c32(0)))));
    APInt Max = Signed ? APInt::getSignedMaxValue(N) : APInt::getMaxValue(N);
    APInt MinMag = Signed ? APInt::getOneBitSet(N, N - 1) : APInt(N, 0);
    Value *MaxV = ConstantInt::get(I128, Max.zext(128));
    Value *MinMagV = ConstantInt::get(I128, MinMag.zext(128));
    Value *Lim = select(T.Sign, MinMagV, MaxV);
    Value *Mag = select(B.CreateOr(Big, B.CreateICmpUGT(V, Lim)), Lim, V);
    Value *R = B.CreateTrunc(Mag, Ty);
    R = select(T.Sign, B.CreateNeg(R), R);
    if (!Signed)
      R = select(T.Sign, ConstantInt::get(Ty, 0), R);
    return select(isNaN(X, F64), ConstantInt::get(Ty, 0), R);
  }

  enum Integral { Floor, Ceil, Trunc, Nearest, Even, Away };
  Value *integral(Value *X, Integral Mode) {
    Term T = unpack(X, F64);
    Value *NE = B.CreateNeg(B.CreateBinaryIntrinsic(Intrinsic::smin, T.E, c32(0)));
    Value *Q = lshr(T.M, NE);
    Value *Rem = B.CreateSub(T.M, shl(Q, NE));
    Value *Half = shl(c128(1), B.CreateSub(NE, c32(1)));
    Value *HasFrac = B.CreateICmpNE(Rem, c128(0));
    // Below 2^-75 there is no half to reach (Half is then 0).
    Value *HasHalf = B.CreateAnd(B.CreateICmpNE(NE, c32(0)), B.CreateICmpULE(NE, c32(128)));
    Value *AboveHalf = B.CreateAnd(HasHalf, B.CreateICmpUGT(Rem, Half));
    Value *IsHalf = B.CreateAnd(HasHalf, B.CreateICmpEQ(Rem, Half));
    Value *Odd = B.CreateTrunc(Q, I1);
    Value *IncEven = B.CreateOr(AboveHalf, B.CreateAnd(IsHalf, Odd));
    Value *Inc;
    switch (Mode) {
    case Floor:
      Inc = B.CreateAnd(HasFrac, T.Sign);
      break;
    case Ceil:
      Inc = B.CreateAnd(HasFrac, B.CreateNot(T.Sign));
      break;
    case Trunc:
      Inc = B.getFalse();
      break;
    case Even:
      Inc = IncEven;
      break;
    case Away:
      Inc = B.CreateOr(AboveHalf, IsHalf);
      break;
    case Nearest:
      Inc = select(roundingIs(RoundNearestEven), IncEven,
                   select(roundingIs(RoundUp), B.CreateAnd(HasFrac, B.CreateNot(T.Sign)),
                          select(roundingIs(RoundDown), B.CreateAnd(HasFrac, T.Sign),
                                 B.getFalse())));
      break;
    }
    Q = B.CreateAdd(Q, B.CreateZExt(Inc, I128));
    Value *R = round({T.Sign, Q, c32(0)}, F64);
    // Values of at least 2^52 are integers.
    Value *Integer = B.CreateICmpUGE(magnitude(X, F64), c64(uint64_t(1075) << 52));
    R = select(Integer, X, R);
    return select(isNaN(X, F64), nanResult({X}, F64, F64), R);
  }

  /// A signed integer whose order is that of the non-NaN values, with -0 and
  /// +0 equal (or -0 below +0 with ZeroOrder).
  Value *orderKey(Value *X, bool ZeroOrder) {
    Value *Mag = magnitude(X, F64);
    Value *Neg = ZeroOrder ? B.CreateNot(Mag) : B.CreateNeg(Mag);
    return select(signOf(X, F64), Neg, Mag);
  }

  Value *compare(CmpInst::Predicate P, Value *X, Value *Y) {
    Value *Uno = B.CreateOr(isNaN(X, F64), isNaN(Y, F64));
    Value *KX = orderKey(X, false), *KY = orderKey(Y, false);
    Value *R;
    switch (P) {
    case CmpInst::FCMP_FALSE:
      return B.getFalse();
    case CmpInst::FCMP_TRUE:
      return B.getTrue();
    case CmpInst::FCMP_ORD:
      return B.CreateNot(Uno);
    case CmpInst::FCMP_UNO:
      return Uno;
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_UEQ:
      R = B.CreateICmpEQ(KX, KY);
      break;
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_UNE:
      R = B.CreateICmpNE(KX, KY);
      break;
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_UGT:
      R = B.CreateICmpSGT(KX, KY);
      break;
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGE:
      R = B.CreateICmpSGE(KX, KY);
      break;
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ULT:
      R = B.CreateICmpSLT(KX, KY);
      break;
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULE:
      R = B.CreateICmpSLE(KX, KY);
      break;
    default:
      llvm_unreachable("not a floating point predicate");
    }
    return CmpInst::isUnordered(P) ? B.CreateOr(Uno, R)
                                   : B.CreateAnd(B.CreateNot(Uno), R);
  }

  /// minnum, maxnum and the IEEE 754-2019 minimum, maximum, minimumNumber
  /// and maximumNumber.
  Value *minMax(Value *X, Value *Y, bool Min, bool PropagateNaN, bool ZeroOrder) {
    Value *KX = orderKey(X, ZeroOrder), *KY = orderKey(Y, ZeroOrder);
    Value *PickX = Min ? B.CreateICmpSLE(KX, KY) : B.CreateICmpSGE(KX, KY);
    Value *R = select(PickX, X, Y);
    Value *XNaN = isNaN(X, F64), *YNaN = isNaN(Y, F64);
    if (PropagateNaN)
      return select(B.CreateOr(XNaN, YNaN), nanResult({X, Y}, F64, F64), R);
    R = select(XNaN, Y, select(YNaN, X, R));
    return select(B.CreateAnd(XNaN, YNaN), nanResult({X, Y}, F64, F64), R);
  }

  Value *isFPClass(Value *X, uint64_t Mask) {
    Value *Sign = signOf(X, F64);
    Value *Mag = magnitude(X, F64);
    Value *R = B.getFalse();
    auto Add = [&](unsigned Test, Value *C) {
      if (Mask & Test)
        R = B.CreateOr(R, C);
    };
    Value *SNaN = B.CreateAnd(isNaN(X, F64),
                              B.CreateICmpEQ(B.CreateAnd(X, c64(F64.quietBit())), c64(0)));
    Value *QNaN = B.CreateAnd(isNaN(X, F64), B.CreateNot(SNaN));
    Value *Inf = isInf(X, F64);
    Value *Zero = isZero(X, F64);
    Value *Sub = B.CreateAnd(B.CreateNot(Zero), B.CreateICmpULT(Mag, c64(F64.fracMask() + 1)));
    Value *Normal = B.CreateAnd(B.CreateICmpUGE(Mag, c64(F64.fracMask() + 1)),
                                B.CreateICmpULT(Mag, c64(F64.infBits())));
    Value *Pos = B.CreateNot(Sign);
    Add(fcSNan, SNaN);
    Add(fcQNan, QNaN);
    Add(fcNegInf, B.CreateAnd(Inf, Sign));
    Add(fcNegNormal, B.CreateAnd(Normal, Sign));
    Add(fcNegSubnormal, B.CreateAnd(Sub, Sign));
    Add(fcNegZero, B.CreateAnd(Zero, Sign));
    Add(fcPosZero, B.CreateAnd(Zero, Pos));
    Add(fcPosSubnormal, B.CreateAnd(Sub, Pos));
    Add(fcPosNormal, B.CreateAnd(Normal, Pos));
    Add(fcPosInf, B.CreateAnd(Inf, Pos));
    return R;
  }
};

} // namespace

static bool isDouble(const Value *V) {
  return V->getType()->isDoubleTy();
}

static Fmt formatOf(Type *Ty) {
  if (Ty->isFloatTy())
    return F32;
  if (Ty->isBFloatTy())
    return BF16;
  return F16;
}

bool AGX::isDoubleArithmetic(const Instruction &I) {
  if (I.getType()->isVectorTy())
    return false;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FNeg:
  case Instruction::FPTrunc:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    return isDouble(&I) || isDouble(I.getOperand(0));
  case Instruction::FCmp:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::FPExt:
    return isDouble(I.getOperand(0)) || isDouble(&I);
  default:
    break;
  }
  if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
    case Intrinsic::sqrt:
    case Intrinsic::fabs:
    case Intrinsic::copysign:
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum:
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::trunc:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::roundeven:
    case Intrinsic::round:
    case Intrinsic::canonicalize:
      return isDouble(II);
    case Intrinsic::is_fpclass:
    case Intrinsic::fptosi_sat:
    case Intrinsic::fptoui_sat:
      return isDouble(II->getArgOperand(0));
    default:
      break;
    }
  }
  return false;
}

Value *AGX::buildSoftFMA(IRBuilderBase &B, Value *X, Value *Y, Value *Z,
                         Value *Rounding, bool CanonicalNaN) {
  Type *Ty = X->getType();
  Fmt F = Ty->isDoubleTy() ? F64 : formatOf(Ty);
  Type *IntTy = B.getIntNTy(F.W);
  auto Bits = [&](Value *V) {
    return B.CreateZExt(B.CreateBitCast(V, IntTy), B.getInt64Ty());
  };
  SoftDouble SD(B, Rounding, CanonicalNaN);
  Value *R = SD.fma(Bits(X), Bits(Y), Bits(Z), F);
  return B.CreateBitCast(B.CreateTrunc(R, IntTy), Ty);
}

Value *AGX::lowerDoubleArithmetic(Instruction &I, Value *Rounding,
                                  bool CanonicalNaN) {
  if (!isDoubleArithmetic(I))
    return nullptr;
  IRBuilder<> B(&I);
  SoftDouble SD(B, Rounding, CanonicalNaN);
  Type *I64 = B.getInt64Ty();
  auto Bits = [&](unsigned N) {
    Value *V = I.getOperand(N);
    if (V->getType()->isDoubleTy())
      return B.CreateBitCast(V, I64);
    unsigned W = V->getType()->getPrimitiveSizeInBits();
    return B.CreateZExt(B.CreateBitCast(V, B.getIntNTy(W)), I64);
  };
  auto Dbl = [&](Value *V) { return B.CreateBitCast(V, B.getDoubleTy()); };
  Value *R = nullptr;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
    R = Dbl(SD.add(Bits(0), Bits(1), I.getOpcode() == Instruction::FSub));
    break;
  case Instruction::FMul:
    R = Dbl(SD.mul(Bits(0), Bits(1)));
    break;
  case Instruction::FDiv:
    R = Dbl(SD.div(Bits(0), Bits(1)));
    break;
  case Instruction::FNeg:
    R = Dbl(B.CreateXor(Bits(0), SD.c64(F64.signBit())));
    break;
  case Instruction::FPExt: {
    Fmt In = formatOf(I.getOperand(0)->getType());
    R = Dbl(SD.convert(Bits(0), In, F64));
    break;
  }
  case Instruction::FPTrunc: {
    Type *Ty = I.getType();
    Fmt Out = formatOf(Ty);
    Value *V = SD.convert(Bits(0), F64, Out);
    R = B.CreateBitCast(B.CreateTrunc(V, B.getIntNTy(Out.W)), Ty);
    break;
  }
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    if (!isDouble(&I))
      return nullptr;
    R = Dbl(SD.fromInt(I.getOperand(0), I.getOpcode() == Instruction::SIToFP));
    break;
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    R = SD.toInt(Bits(0), I.getType(), I.getOpcode() == Instruction::FPToSI);
    break;
  case Instruction::FCmp:
    R = SD.compare(cast<FCmpInst>(I).getPredicate(), Bits(0), Bits(1));
    break;
  default: {
    auto &II = cast<IntrinsicInst>(I);
    switch (II.getIntrinsicID()) {
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
      R = Dbl(SD.fma(Bits(0), Bits(1), Bits(2)));
      break;
    case Intrinsic::sqrt:
      R = Dbl(SD.sqrt(Bits(0)));
      break;
    case Intrinsic::fabs:
      R = Dbl(SD.magnitude(Bits(0), F64));
      break;
    case Intrinsic::copysign:
      R = Dbl(B.CreateOr(SD.magnitude(Bits(0), F64),
                         B.CreateAnd(Bits(1), SD.c64(F64.signBit()))));
      break;
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
      R = Dbl(SD.minMax(Bits(0), Bits(1), II.getIntrinsicID() == Intrinsic::minnum,
                        false, true));
      break;
    case Intrinsic::minimum:
    case Intrinsic::maximum:
      R = Dbl(SD.minMax(Bits(0), Bits(1), II.getIntrinsicID() == Intrinsic::minimum,
                        true, true));
      break;
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum:
      R = Dbl(SD.minMax(Bits(0), Bits(1),
                        II.getIntrinsicID() == Intrinsic::minimumnum, false, true));
      break;
    case Intrinsic::floor:
      R = Dbl(SD.integral(Bits(0), SoftDouble::Floor));
      break;
    case Intrinsic::ceil:
      R = Dbl(SD.integral(Bits(0), SoftDouble::Ceil));
      break;
    case Intrinsic::trunc:
      R = Dbl(SD.integral(Bits(0), SoftDouble::Trunc));
      break;
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
      R = Dbl(SD.integral(Bits(0), SoftDouble::Nearest));
      break;
    case Intrinsic::roundeven:
      R = Dbl(SD.integral(Bits(0), SoftDouble::Even));
      break;
    case Intrinsic::round:
      R = Dbl(SD.integral(Bits(0), SoftDouble::Away));
      break;
    case Intrinsic::canonicalize: {
      Value *X = Bits(0);
      R = Dbl(SD.select(SD.isNaN(X, F64), SD.nanResult({X}, F64, F64), X));
      break;
    }
    case Intrinsic::is_fpclass:
      R = SD.isFPClass(Bits(0),
                       cast<ConstantInt>(II.getArgOperand(1))->getZExtValue());
      break;
    case Intrinsic::fptosi_sat:
    case Intrinsic::fptoui_sat:
      R = SD.toInt(Bits(0), I.getType(),
                   II.getIntrinsicID() == Intrinsic::fptosi_sat);
      break;
    default:
      return nullptr;
    }
  }
  }
  R->takeName(&I);
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
  for (BinaryOperator *Div : SD.Divisions)
    expandDivision(Div);
  return R;
}

namespace {

class AGXSoftF64 : public FunctionPass {
public:
  static char ID;
  AGXSoftF64() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Soft F64"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char AGXSoftF64::ID = 0;

INITIALIZE_PASS_BEGIN(AGXSoftF64, DEBUG_TYPE, "AGX Soft F64", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXSoftF64, DEBUG_TYPE, "AGX Soft F64", false, false)

FunctionPass *llvm::createAGXSoftF64Pass() { return new AGXSoftF64(); }

bool AGXSoftF64::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  SmallVector<Instruction *, 16> Worklist;
  for (Instruction &I : instructions(F))
    if (AGX::isDoubleArithmetic(I))
      Worklist.push_back(&I);
  bool Changed = false;
  Value *Nearest = ConstantInt::get(Type::getInt32Ty(F.getContext()),
                                    RoundNearestEven);
  for (Instruction *I : Worklist)
    Changed |= AGX::lowerDoubleArithmetic(*I, Nearest, ST.hasCanonicalNaN()) != nullptr;
  return Changed;
}
