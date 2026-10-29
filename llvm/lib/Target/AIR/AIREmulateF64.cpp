//===- AIREmulateF64.cpp - Double precision as soft-f64 -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIR has no double. A module from a front end that writes AIR directly
// (ZLUDA's PTX path) has doubles everywhere: values, arguments, globals,
// arithmetic, conversions, atomics. This pass makes it what agx-raise-to-air
// writes from AGX's IR (llvm/docs/AIR/AIRHints.rst):
//
// * every double is its bits, an i64 (vectors, arrays and structures of
//   them likewise, constants as their bits);
// * every double precision operation is a call of arith-emulate-f64's
//   library (soft_f64.c, AIRSoftF64.inc: __air_f64_add, ...), linked in as
//   internal functions; the code around the calls is inlined at each
//   operation;
// * a constrained floating point intrinsic, of any format, is the AIR
//   operation when it rounds to nearest, and otherwise its operands exactly
//   as doubles, the operation in soft-f64 rounded in its direction (the _r
//   functions) and the result rounded in that direction to its format
//   (rounding twice in one direction rounds once). AIR has no floating point
//   environment: round.dynamic is to nearest, and the exception behavior is
//   dropped (as is strictfp).
//
// bfloat arithmetic of the constrained intrinsics is computed in float (or
// soft-f64 for fma), with the conversions on the bits, so it leaves no
// bfloat operation. The mapping of each operation follows agx-raise-to-air's
// (its emulate()); a module without double or constrained intrinsics is left
// as it is.
//
//===----------------------------------------------------------------------===//

#include "AIREmulateF64.h"
#include "AIR.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "air-emulate-f64"

#include "AIRSoftF64.inc"

namespace {

/// The attribute of an emulation (a function standing for one operation
/// until its body is written and inlined): the operation's key.
constexpr StringLiteral EmulationAttr = "air-emulate-f64";

// Constrained operations with a directed rounding stay hints (llvm/IR/AIR.h):
// a consumer with a floating point mode (the AGX backend) compiles them as
// the operation, which is much cheaper than their exact emulation; the others
// (the Metal pipeline) inline the emulation, the hint's body.
static cl::opt<bool> HintDirected(
    "air-hint-directed-rounding", cl::init(true), cl::Hidden,
    cl::desc("Keep constrained operations with a directed rounding as AIR "
             "hints whose body is their emulation"));

/// An emulation that stays a hint: a constrained operation rounding toward
/// zero, upward or downward, whose key agx-lower-air restores (no "+"
/// operands it would miscount).
/// So are bfloat ones in any rounding (in float or soft-f64 for Metal, whose
/// float denormals may flush; native on AGX). The "+f32-denormals" marker
/// only chooses the emulation, and is not in the hint's key.
static bool keepsAsHint(StringRef Key) {
  if (!HintDirected || !Key.starts_with("call:llvm.experimental.constrained."))
    return false;
  if ((Key.contains(".bf16") || Key.contains(".v2bf16")) &&
      !Key.contains("constrained.fpext") && !Key.contains("constrained.fptrunc"))
    return true;
  if (Key.contains("|+"))
    return false;
  return Key.contains("|round.towardzero") || Key.contains("|round.upward") ||
         Key.contains("|round.downward");
}

static std::string hintKey(StringRef Key) {
  std::string K = Key.str();
  for (StringRef Marker : {"|+f32-denormals"})
    if (size_t At = K.find(Marker.str()); At != std::string::npos)
      K.erase(At, Marker.size());
  return K;
}

bool isDouble(const Type *T) { return T->getScalarType()->isDoubleTy(); }

/// Whether T has a double anywhere in it.
bool hasDouble(Type *T, SmallPtrSetImpl<Type *> &Seen) {
  if (T->isDoubleTy())
    return true;
  if (!Seen.insert(T).second)
    return false;
  if (auto *ST = dyn_cast<StructType>(T); ST && ST->isOpaque())
    return false;
  return any_of(T->subtypes(), [&](Type *S) { return hasDouble(S, Seen); });
}
bool hasDouble(Type *T) {
  SmallPtrSet<Type *, 8> Seen;
  return hasDouble(T, Seen);
}

bool isConstrained(const Function &F) {
  return F.getName().starts_with("llvm.experimental.constrained.");
}

/// Doubles as i64 in every type.
class F64ToBits : public ValueMapTypeRemapper {
  DenseMap<Type *, Type *> Cache;

public:
  Type *remapType(Type *T) override {
    auto It = Cache.find(T);
    if (It != Cache.end())
      return It->second;
    Type *R = T;
    if (T->isDoubleTy()) {
      R = Type::getInt64Ty(T->getContext());
    } else if (auto *VT = dyn_cast<VectorType>(T)) {
      R = VectorType::get(remapType(VT->getElementType()),
                          VT->getElementCount());
    } else if (auto *AT = dyn_cast<ArrayType>(T)) {
      R = ArrayType::get(remapType(AT->getElementType()),
                         AT->getNumElements());
    } else if (auto *FT = dyn_cast<FunctionType>(T)) {
      SmallVector<Type *, 8> Params;
      for (Type *P : FT->params())
        Params.push_back(remapType(P));
      R = FunctionType::get(remapType(FT->getReturnType()), Params,
                            FT->isVarArg());
    } else if (auto *ST = dyn_cast<StructType>(T); ST && !ST->isOpaque()) {
      SmallVector<Type *, 8> Elts;
      for (Type *E : ST->elements())
        Elts.push_back(remapType(E));
      if (!equal(Elts, ST->elements()))
        R = ST->isLiteral()
                ? StructType::get(T->getContext(), Elts, ST->isPacked())
                : StructType::create(T->getContext(), Elts,
                                     ST->getName().str() + ".f64",
                                     ST->isPacked());
    }
    Cache[T] = R;
    return R;
  }
};

/// C with its doubles as i64 bit patterns, or null if it has none.
Constant *doubleBits(Constant *C) {
  Type *T = C->getType();
  LLVMContext &Ctx = T->getContext();
  // (a vector ConstantFP is a splat)
  if (auto *FP = dyn_cast<ConstantFP>(C); FP && isDouble(T)) {
    Type *I64 = Type::getInt64Ty(Ctx);
    if (auto *VT = dyn_cast<VectorType>(T))
      I64 = VectorType::get(I64, VT);
    return ConstantInt::get(I64, FP->getValueAPF().bitcastToAPInt());
  }
  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    if (!CDS->getElementType()->isDoubleTy())
      return nullptr;
    SmallVector<uint64_t, 8> Bits;
    for (unsigned I = 0; I < CDS->getNumElements(); ++I)
      Bits.push_back(
          CDS->getElementAsAPFloat(I).bitcastToAPInt().getZExtValue());
    return isa<VectorType>(T) ? ConstantDataVector::get(Ctx, Bits)
                              : ConstantDataArray::get(Ctx, Bits);
  }
  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    SmallVector<Constant *, 8> Ops;
    bool Changed = false;
    for (Use &U : CA->operands()) {
      Constant *Op = cast<Constant>(U);
      if (Constant *Bits = doubleBits(Op)) {
        Op = Bits;
        Changed = true;
      } else if (isDouble(Op->getType())) {
        // undef, poison or zero doubles
        Type *To = Type::getInt64Ty(Ctx);
        if (auto *VT = dyn_cast<VectorType>(Op->getType()))
          To = VectorType::get(To, VT);
        Op = isa<PoisonValue>(Op)  ? PoisonValue::get(To)
             : isa<UndefValue>(Op) ? UndefValue::get(To)
                                   : Constant::getNullValue(To);
        Changed = true;
      }
      Ops.push_back(Op);
    }
    if (!Changed)
      return nullptr;
    if (isa<ConstantVector>(CA))
      return ConstantVector::get(Ops);
    if (auto *ST = dyn_cast<StructType>(T)) {
      SmallVector<Type *, 8> Tys;
      for (Constant *Op : Ops)
        Tys.push_back(Op->getType());
      return ConstantStruct::get(StructType::get(Ctx, Tys, ST->isPacked()),
                                 Ops);
    }
    return ConstantArray::get(ArrayType::get(Ops[0]->getType(), Ops.size()),
                              Ops);
  }
  return nullptr;
}

/// Constants with doubles, wherever the value mapper meets them: their bits.
class DoubleMaterializer final : public ValueMaterializer {
public:
  Value *materialize(Value *V) override {
    auto *C = dyn_cast<Constant>(V);
    if (!C || isa<GlobalValue>(C))
      return nullptr;
    return doubleBits(C);
  }
};

bool isDoubleOp(const Instruction &I) {
  auto Dbl = [](const Value *V) { return isDouble(V->getType()); };
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FNeg:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FPExt:
    return Dbl(&I);
  case Instruction::FCmp:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::FPTrunc:
    return Dbl(I.getOperand(0));
  default:
    return false;
  }
}

/// The AIR functions of double precision that a front end may call, as the
/// LLVM intrinsics whose emulation they share.
StringRef airAsIntrinsic(StringRef Name) {
  if (!Name.consume_front("air.") || !Name.consume_back(".f64"))
    return "";
  return StringSwitch<StringRef>(Name)
      .Case("fabs", "llvm.fabs.f64")
      .Case("sqrt", "llvm.sqrt.f64")
      .Case("fma", "llvm.fma.f64")
      .Case("fmin", "llvm.minnum.f64")
      .Case("fmax", "llvm.maxnum.f64")
      .Case("floor", "llvm.floor.f64")
      .Case("ceil", "llvm.ceil.f64")
      .Case("trunc", "llvm.trunc.f64")
      .Case("rint", "llvm.rint.f64")
      .Case("round", "llvm.round.f64")
      .Case("copysign", "llvm.copysign.f64")
      .Default("");
}

/// The operation of an intrinsic (or AIR function) on doubles, as the
/// emulation names it: the name without "llvm." and the overloads.
StringRef doubleOpBase(StringRef Name) {
  Name.consume_front("llvm.experimental.constrained.");
  Name.consume_front("llvm.");
  return Name.take_until([](char C) { return C == '.'; });
}

/// The double precision intrinsics that have an emulation.
bool emulatesDouble(StringRef Base) {
  return is_contained(
      {"fadd",    "fsub",       "fmul",       "fdiv",      "fma",
       "fmuladd", "sqrt",       "fabs",       "copysign",  "floor",
       "ceil",    "trunc",      "round",      "roundeven", "rint",
       "nearbyint", "minnum",   "maxnum",     "minimum",   "maximum",
       "minimumnum", "maximumnum", "fcmp",    "fcmps",     "fptosi",
       "fptoui",  "sitofp",     "uitofp",     "fpext",     "fptrunc",
       "canonicalize"},
      Base);
}

/// A vector operation on doubles as one per element (the soft-f64 library
/// is scalar). False if I is not one this splits.
bool scalarizeDoubles(Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  bool Dbl = II ? (isDouble(I.getType()) ||
                   any_of(II->args(),
                          [](const Use &U) { return isDouble(U->getType()); }))
                : isDoubleOp(I);
  if (!Dbl)
    return false;
  FixedVectorType *VT = dyn_cast<FixedVectorType>(I.getType());
  for (Value *Op : (II ? II->args() : I.operands()))
    if (!VT)
      VT = dyn_cast<FixedVectorType>(Op->getType());
  if (!VT)
    return false;
  IRBuilder<> B(&I);
  Type *Ty = I.getType()->getScalarType();
  SmallVector<Value *, 4> Lanes;
  for (unsigned E = 0; E < VT->getNumElements(); ++E) {
    SmallVector<Value *, 4> Ops;
    for (Value *Op : (II ? II->args() : I.operands()))
      Ops.push_back(Op->getType()->isVectorTy() ? B.CreateExtractElement(Op, E)
                                                : Op);
    Value *X;
    if (II) {
      SmallVector<Type *, 4> Tys;
      for (Value *Op : Ops)
        Tys.push_back(Op->getType());
      SmallVector<Type *, 4> Overloads;
      if (!Intrinsic::isSignatureValid(II->getIntrinsicID(),
                                       FunctionType::get(Ty, Tys, false),
                                       Overloads))
        return false; // (no lane has been used yet: the extracts are dead)
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          I.getModule(), II->getIntrinsicID(), Overloads);
      X = B.CreateCall(Decl, Ops);
    } else if (auto *C = dyn_cast<FCmpInst>(&I))
      X = B.CreateFCmp(C->getPredicate(), Ops[0], Ops[1]);
    else if (auto *Cast = dyn_cast<CastInst>(&I))
      X = B.CreateCast(Cast->getOpcode(), Ops[0], Ty);
    else if (I.getOpcode() == Instruction::FNeg)
      X = B.CreateFNeg(Ops[0]);
    else
      X = B.CreateBinOp(cast<BinaryOperator>(I).getOpcode(), Ops[0], Ops[1]);
    if (auto *XI = dyn_cast<Instruction>(X); XI && !II)
      XI->copyIRFlags(&I);
    Lanes.push_back(X);
  }
  Value *R = PoisonValue::get(I.getType());
  for (auto [E, X] : enumerate(Lanes))
    R = B.CreateInsertElement(R, X, E);
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
  return true;
}

/// A key's double mask: one character for the result and each operand of
/// the operation, 'd' where it is a double.
std::string doubleMask(Type *Ret, ArrayRef<Type *> Ops) {
  std::string S(1, isDouble(Ret) ? 'd' : '-');
  for (Type *T : Ops)
    S += isDouble(T) ? 'd' : '-';
  return S;
}

using LibFn = function_ref<Value *(StringRef, Type *, ArrayRef<Value *>)>;

/// bfloat X (or a vector of them) as float, exactly: its bits are the high
/// half of the float's.
Value *bf16ToF32(IRBuilderBase &B, Value *X) {
  Type *T = X->getType();
  Value *I = B.CreateBitCast(X, T->getWithNewType(B.getInt16Ty()));
  I = B.CreateShl(B.CreateZExt(I, T->getWithNewType(B.getInt32Ty())), 16);
  return B.CreateBitCast(I, T->getWithNewType(B.getFloatTy()));
}

/// float X (or a vector) rounded to nearest even as bfloat, on the bits (a
/// NaN keeps its sign and the high bits of its payload, quieted).
Value *f32ToBF16(IRBuilderBase &B, Value *X) {
  Type *T = X->getType();
  Type *I32 = T->getWithNewType(B.getInt32Ty());
  auto C = [&](uint32_t V) { return ConstantInt::get(I32, V); };
  Value *I = B.CreateBitCast(X, I32);
  Value *NaN = B.CreateICmpUGT(B.CreateAnd(I, C(0x7fffffff)), C(0x7f800000));
  Value *Odd = B.CreateAnd(B.CreateLShr(I, 16), C(1));
  Value *R = B.CreateLShr(B.CreateAdd(B.CreateAdd(I, C(0x7fff)), Odd), 16);
  R = B.CreateSelect(NaN, B.CreateOr(B.CreateLShr(I, 16), C(0x40)), R);
  R = B.CreateTrunc(R, T->getWithNewType(B.getInt16Ty()));
  return B.CreateBitCast(R, T->getWithNewType(B.getBFloatTy()));
}

/// The constrained operation Base of a format other than double, on A, in
/// the rounding Rounding (FLT_ROUNDS numbering): the operands as doubles
/// (exact), the operation in soft-f64 rounded in that direction, then
/// rounded to Ret in it (rounding twice in the same direction rounds once,
/// as the values of the format are doubles). Null for an operation that does
/// not round. agx-raise-to-air's directed(), whose float fast path this
/// shares.
Value *directed(IRBuilderBase &B, StringRef Base, ArrayRef<Value *> A,
                Type *Ret, Value *Rounding, LibFn Lib) {
  bool Arith = is_contained(
      {"fadd", "fsub", "fmul", "fdiv", "fma", "fmuladd", "sqrt"}, Base);
  bool FromInt = Base == "sitofp" || Base == "uitofp";
  bool Integral = Base == "rint" || Base == "nearbyint";
  Type *ST = Ret->getScalarType();
  if ((!Arith && !FromInt && !Integral && Base != "fptrunc") ||
      !(ST->isFloatTy() || ST->isHalfTy() || ST->isBFloatTy()) ||
      (FromInt && A[0]->getType()->getScalarSizeInBits() > 64))
    return nullptr;
  if (auto *VT = dyn_cast<FixedVectorType>(Ret)) {
    // Lane by lane.
    Value *R = PoisonValue::get(Ret);
    for (unsigned L = 0, E = VT->getNumElements(); L < E; ++L) {
      SmallVector<Value *, 4> Lane;
      for (Value *X : A)
        Lane.push_back(X->getType()->isVectorTy() ? B.CreateExtractElement(X, L)
                                                  : X);
      R = B.CreateInsertElement(R, directed(B, Base, Lane, ST, Rounding, Lib),
                                L);
    }
    return R;
  }
  Type *I16 = B.getInt16Ty(), *I32 = B.getInt32Ty(), *I64 = B.getInt64Ty();
  auto Is = [&](unsigned C) { return B.CreateICmpEQ(Rounding, B.getInt32(C)); };

  if (Integral) {
    // bfloat through float (exact both ways: the results are integers of
    // the operand's precision).
    Value *X = ST->isBFloatTy() ? bf16ToF32(B, A[0]) : A[0];
    auto Call = [&](Intrinsic::ID ID) {
      return B.CreateUnaryIntrinsic(ID, X);
    };
    Value *R = Call(Intrinsic::roundeven);
    R = B.CreateSelect(Is(0), Call(Intrinsic::trunc), R);
    R = B.CreateSelect(Is(2), Call(Intrinsic::ceil), R);
    R = B.CreateSelect(Is(3), Call(Intrinsic::floor), R);
    return ST->isBFloatTy() ? f32ToBF16(B, R) : R;
  }
  // A float to half or bfloat: rounded to nearest, then one unit in the
  // direction where the result, back as a float (exactly), lies on the wrong
  // side of the operand. Exact everywhere (an overflow to infinity steps back
  // to the largest finite value, a zero to the smallest subnormal); a NaN
  // compares unordered and stays.
  auto TruncDirected = [&](Value *X, Type *To) -> Value * {
    Value *H = To->isHalfTy() ? B.CreateFPTrunc(X, To) : f32ToBF16(B, X);
    Value *Back = To->isHalfTy() ? B.CreateFPExt(H, B.getFloatTy())
                                 : bf16ToF32(B, H);
    Value *Bits = B.CreateBitCast(H, I16);
    Value *Zero = B.CreateICmpEQ(B.CreateAnd(Bits, 0x7fff), B.getInt16(0));
    Value *Neg = B.CreateICmpSLT(Bits, B.getInt16(0));
    // the neighbours of H below and above
    Value *Down = B.CreateSelect(
        Zero, B.getInt16(0x8001),
        B.CreateSelect(Neg, B.CreateAdd(Bits, B.getInt16(1)),
                       B.CreateSub(Bits, B.getInt16(1))));
    Value *Up = B.CreateSelect(
        Zero, B.getInt16(0x0001),
        B.CreateSelect(Neg, B.CreateSub(Bits, B.getInt16(1)),
                       B.CreateAdd(Bits, B.getInt16(1))));
    Value *Above = B.CreateFCmpOGT(Back, X), *Below = B.CreateFCmpOLT(Back, X);
    // toward zero: down from above a positive operand, up from below a
    // negative one
    Value *XNeg = B.CreateFCmpOLT(X, ConstantFP::get(B.getFloatTy(), 0.0));
    Value *RZ = B.CreateSelect(
        B.CreateAnd(Above, B.CreateNot(XNeg)), Down,
        B.CreateSelect(B.CreateAnd(Below, XNeg), Up, Bits));
    Value *RU = B.CreateSelect(Below, Up, Bits);
    Value *RD = B.CreateSelect(Above, Down, Bits);
    Value *R = B.CreateSelect(
        Is(0), RZ, B.CreateSelect(Is(2), RU, B.CreateSelect(Is(3), RD, Bits)));
    return B.CreateBitCast(R, To);
  };
  if (Base == "fptrunc" && A[0]->getType()->isFloatTy() &&
      (ST->isHalfTy() || ST->isBFloatTy()))
    return TruncDirected(A[0], ST);
  // A value of the format as the bits of the double.
  auto Ext = [&](Value *X) {
    Type *T = X->getType();
    Value *F;
    if (T->isFloatTy())
      F = B.CreateBitCast(X, I32);
    else if (T->isHalfTy())
      F = B.CreateBitCast(B.CreateFPExt(X, B.getFloatTy()), I32);
    else // bfloat: the high half of a float
      F = B.CreateShl(B.CreateZExt(B.CreateBitCast(X, I16), I32), 16);
    return Lib("__air_f64_from_f32", I64, {F});
  };
  // add, sub and mul of floats, by error-free transformations (Knuth's
  // TwoSum, Dekker's product with an fma): the result rounded to nearest and
  // the exact sign of its error, then one unit in the direction where the
  // error says so. Exact while operands and result are zero or at least
  // 2^-70 in magnitude and the result is finite; the others take the double
  // precision path below. An exact sum of zero is +0, -0 rounding downward,
  // unless both operands are zeros of one sign (that sign).
  // The fast path's result (and whether it is exact) for floats X and Y.
  auto TwoFold = [&](Value *X, Value *Y, bool Mul) -> std::pair<Value *, Value *> {
    Type *ST = B.getFloatTy();
    Value *Fast;
    Value *S, *Err;
    if (Mul) {
      S = B.CreateFMul(X, Y);
      Err = B.CreateIntrinsic(Intrinsic::fma, {ST}, {X, Y, B.CreateFNeg(S)});
    } else {
      S = B.CreateFAdd(X, Y);
      Value *BV = B.CreateFSub(S, X);
      Value *AV = B.CreateFSub(S, BV);
      Err = B.CreateFAdd(B.CreateFSub(X, AV), B.CreateFSub(Y, BV));
    }
    // (on the bits: a GPU that flushes denormals compares a denormal equal
    // to zero, which would take it for an exact zero)
    auto Safe = [&](Value *V) {
      Value *Abs = B.CreateAnd(B.CreateBitCast(V, I32), 0x7fffffff);
      Value *InRange = B.CreateAnd(
          B.CreateICmpUGE(Abs, B.getInt32(0x1c800000)), // 2^-70
          B.CreateICmpULT(Abs, B.getInt32(0x7f000000)));
      return B.CreateOr(B.CreateICmpEQ(Abs, B.getInt32(0)), InRange);
    };
    // (S not zero: on a GPU that flushes denormals, a zero S may be a
    // flushed denormal result; an exact zero takes the slow path, which has
    // the IEEE sign rules)
    Value *Ok = B.CreateAnd(
        B.CreateAnd(B.CreateAnd(Safe(X), Safe(Y)), Safe(S)),
        B.CreateFCmpONE(S, ConstantFP::get(ST, 0.0)));
    // The step: +1 or -1 on the bits (away from or toward zero), by the
    // signs of the result and of the error.
    Value *SBits = B.CreateBitCast(S, I32);
    Value *Neg = B.CreateICmpSLT(SBits, B.getInt32(0));
    Value *Above = B.CreateFCmpOGT(Err, ConstantFP::get(ST, 0.0)); // exact > S
    Value *Below = B.CreateFCmpOLT(Err, ConstantFP::get(ST, 0.0));
    Value *Toward = B.CreateSelect(Neg, Above, Below); // |exact| < |S|
    // rz: toward zero when |exact| < |S|; ru: up when exact > S; rd: down.
    Value *Dec = B.CreateSelect(
        Is(0), Toward,
        B.CreateSelect(Is(2), B.CreateAnd(Above, Neg),
                       B.CreateAnd(Below, B.CreateNot(Neg))));
    Value *Inc = B.CreateSelect(
        Is(0), B.getFalse(),
        B.CreateSelect(Is(2), B.CreateAnd(Above, B.CreateNot(Neg)),
                       B.CreateAnd(Below, Neg)));
    Value *Step = B.CreateSub(B.CreateZExt(Inc, I32), B.CreateZExt(Dec, I32));
    Fast = B.CreateBitCast(B.CreateAdd(SBits, Step), ST);
    if (!Mul) {
      Value *Zero = ConstantFP::get(ST, 0.0);
      Value *SameZeros = B.CreateAnd(
          B.CreateAnd(B.CreateFCmpOEQ(X, Zero), B.CreateFCmpOEQ(Y, Zero)),
          B.CreateICmpEQ(
              B.CreateICmpSLT(B.CreateBitCast(X, I32), B.getInt32(0)),
              B.CreateICmpSLT(B.CreateBitCast(Y, I32), B.getInt32(0))));
      Value *MinusZero = B.CreateAnd(
          B.CreateAnd(B.CreateFCmpOEQ(S, Zero), Is(3)), B.CreateNot(SameZeros));
      Fast = B.CreateSelect(MinusZero, ConstantFP::getNegativeZero(ST), Fast);
    }
    return {Fast, Ok};
  };
  // fma of bfloats in float, where it is exact: the product of two bfloats
  // is exact in float, TwoSum gives the sum's error; to nearest, the sum
  // rounded to odd (the odd one of the two floats around the exact value
  // where it is inexact) then to nearest bfloat rounds once (float keeps 16
  // more bits); directed, the float fast path's result rounded to bfloat in
  // the same direction. Operands that are bfloat (float) subnormals, and
  // products, addends and sums outside [2^-70, 2^126), which hardware
  // flushing float subnormals could break, take the double precision path,
  // as do infinities and NaNs.
  if (ST->isBFloatTy() && (Base == "fma" || Base == "fmuladd")) {
    Type *F32 = B.getFloatTy();
    auto NotSubnormal = [&](Value *V) {
      Value *I = B.CreateBitCast(V, I16);
      return B.CreateOr(
          B.CreateICmpNE(B.CreateAnd(I, B.getInt16(0x7f80)), B.getInt16(0)),
          B.CreateICmpEQ(B.CreateAnd(I, B.getInt16(0x7f)), B.getInt16(0)));
    };
    Value *Zero = ConstantFP::get(F32, 0.0);
    auto InRange = [&](Value *V) {
      Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, V);
      return B.CreateOr(
          B.CreateFCmpOEQ(V, Zero),
          B.CreateAnd(B.CreateFCmpOGE(Abs, ConstantFP::get(F32, 0x1p-70)),
                      B.CreateFCmpOLT(Abs, ConstantFP::get(F32, 0x1p126))));
    };
    Value *X = bf16ToF32(B, A[0]), *Y = bf16ToF32(B, A[1]),
          *Z = bf16ToF32(B, A[2]);
    Value *P = B.CreateFMul(X, Y); // exact
    // a product that flushed to zero is not one
    Value *PZero = B.CreateOr(B.CreateFCmpOEQ(X, Zero), B.CreateFCmpOEQ(Y, Zero));
    Value *POk = B.CreateAnd(InRange(P),
                             B.CreateOr(PZero, B.CreateFCmpUNE(P, Zero)));
    auto [Dir, DirOk] = TwoFold(P, Z, /*Mul=*/false);
    Value *S = B.CreateFAdd(P, Z);
    Value *BV = B.CreateFSub(S, P);
    Value *AV = B.CreateFSub(S, BV);
    Value *Err = B.CreateFAdd(B.CreateFSub(P, AV), B.CreateFSub(Z, BV));
    Value *Ok = B.CreateAnd(
        B.CreateAnd(B.CreateAnd(NotSubnormal(A[0]), NotSubnormal(A[1])),
                    B.CreateAnd(NotSubnormal(A[2]), POk)),
        B.CreateAnd(B.CreateAnd(InRange(Z), InRange(S)), DirOk));
    // to odd: one unit toward the exact value where S is inexact and even
    Value *SBits = B.CreateBitCast(S, I32);
    Value *Inexact = B.CreateFCmpUNE(Err, Zero);
    Value *Even = B.CreateICmpEQ(B.CreateAnd(SBits, 1), B.getInt32(0));
    Value *Away = B.CreateICmpEQ(B.CreateICmpSLT(SBits, B.getInt32(0)),
                                 B.CreateFCmpOLT(Err, Zero));
    Value *Odd = B.CreateSelect(
        B.CreateAnd(Inexact, Even),
        B.CreateSelect(Away, B.CreateAdd(SBits, B.getInt32(1)),
                       B.CreateSub(SBits, B.getInt32(1))),
        SBits);
    Value *Nearest = f32ToBF16(B, B.CreateBitCast(Odd, F32));
    Value *R = B.CreateSelect(Is(1), Nearest, TruncDirected(Dir, ST));
    BasicBlock *Here = B.GetInsertBlock();
    Function *Fn = Here->getParent();
    BasicBlock *Slow = BasicBlock::Create(B.getContext(), "bf16.fma.slow", Fn);
    BasicBlock *Done = BasicBlock::Create(B.getContext(), "bf16.fma.join", Fn);
    B.CreateCondBr(Ok, Done, Slow);
    B.SetInsertPoint(Slow);
    SmallVector<Value *, 4> Args;
    for (Value *V : {X, Y, Z})
      Args.push_back(
          Lib("__air_f64_from_f32", I64, {B.CreateBitCast(V, I32)}));
    // to nearest, the double rounded to odd (toward zero, the last bit set
    // where it is inexact: where rounding away from zero differs; a result
    // toward zero of zero is exact, as no product of bfloats is below the
    // doubles), as rounding twice to nearest is not rounding once; directed,
    // in the direction
    SmallVector<Value *, 4> TZ(Args), AwayArgs(Args);
    TZ.push_back(B.getInt32(0));
    Value *DTZ = Lib("__air_f64_fma_r", I64, TZ);
    AwayArgs.push_back(B.CreateSelect(B.CreateICmpSLT(DTZ, B.getInt64(0)),
                                  B.getInt32(3), B.getInt32(2)));
    Value *SlowInexact =
        B.CreateICmpNE(DTZ, Lib("__air_f64_fma_r", I64, AwayArgs));
    Value *DOdd = B.CreateOr(DTZ, B.CreateZExt(SlowInexact, I64));
    Args.push_back(Rounding);
    Value *D = B.CreateSelect(Is(1), DOdd, Lib("__air_f64_fma_r", I64, Args));
    Value *SlowR = B.CreateBitCast(
        B.CreateTrunc(Lib("__air_f64_to_bf16_r", I32, {D, Rounding}), I16),
        ST);
    BasicBlock *SlowEnd = B.GetInsertBlock();
    B.CreateBr(Done);
    B.SetInsertPoint(Done);
    PHINode *Phi = B.CreatePHI(ST, 2);
    Phi->addIncoming(R, Here);
    Phi->addIncoming(SlowR, SlowEnd);
    return Phi;
  }
  // add, sub and mul of halves: the float fast path, exact for any halves
  // (their magnitudes and the results' are within its range), rounded in the
  // same direction to half; twice in one direction rounds once. bfloats the
  // same way where the fast path is exact, else as below.
  // (a bfloat's subnormals are float subnormals, which hardware flushing
  // float subnormals would flush in the float operation: bfloats only with a
  // known directed rounding, whose operands outside the fast path's range
  // take the double precision path; to nearest they stay as below)
  auto *KnownRounding = dyn_cast<ConstantInt>(Rounding);
  bool BFloatDirected =
      ST->isBFloatTy() && KnownRounding && !KnownRounding->isOne();
  if ((ST->isHalfTy() || BFloatDirected) &&
      (Base == "fadd" || Base == "fsub" || Base == "fmul")) {
    auto Wide = [&](Value *V) {
      return ST->isHalfTy() ? B.CreateFPExt(V, B.getFloatTy()) : bf16ToF32(B, V);
    };
    Value *X = Wide(A[0]), *Y = Wide(A[1]);
    if (Base == "fsub")
      Y = B.CreateFNeg(Y);
    auto [F, Ok] = TwoFold(X, Y, Base == "fmul");
    // nearest: the operation in float rounds once more to nearest, which is
    // not rounding once; it is the plain operation
    Value *Nearest = Base == "fadd"   ? B.CreateFAdd(A[0], A[1])
                     : Base == "fsub" ? B.CreateFSub(A[0], A[1])
                                      : B.CreateFMul(A[0], A[1]);
    Value *R = B.CreateSelect(Is(1), Nearest, TruncDirected(F, ST));
    if (ST->isHalfTy())
      return R;
    // bfloats outside the fast path's range: the double precision path
    BasicBlock *Here = B.GetInsertBlock();
    Function *Fn = Here->getParent();
    BasicBlock *Slow = BasicBlock::Create(B.getContext(), "directed.slow", Fn);
    BasicBlock *Done = BasicBlock::Create(B.getContext(), "directed.join", Fn);
    B.CreateCondBr(B.CreateOr(Ok, Is(1)), Done, Slow);
    B.SetInsertPoint(Slow);
    Value *Ext0 = Lib("__air_f64_from_f32", I64, {B.CreateBitCast(X, I32)});
    Value *Ext1 = Lib("__air_f64_from_f32", I64, {B.CreateBitCast(Y, I32)});
    Value *D = Lib(Base == "fmul" ? "__air_f64_mul_r" : "__air_f64_add_r", I64,
                   {Ext0, Ext1, Rounding});
    Value *SlowR = B.CreateBitCast(
        B.CreateTrunc(Lib("__air_f64_to_bf16_r", I32, {D, Rounding}), I16), ST);
    BasicBlock *SlowEnd = B.GetInsertBlock();
    B.CreateBr(Done);
    B.SetInsertPoint(Done);
    PHINode *P = B.CreatePHI(ST, 2);
    P->addIncoming(R, Here);
    P->addIncoming(SlowR, SlowEnd);
    return P;
  }
  BasicBlock *FastBB = nullptr, *Join = nullptr;
  Value *Fast = nullptr;
  if (ST->isFloatTy() && (Base == "fadd" || Base == "fsub" || Base == "fmul")) {
    Value *X = A[0], *Y = Base == "fsub" ? B.CreateFNeg(A[1]) : A[1];
    Value *Ok;
    std::tie(Fast, Ok) = TwoFold(X, Y, Base == "fmul");
    FastBB = B.GetInsertBlock();
    Function *Fn = FastBB->getParent();
    BasicBlock *Slow = BasicBlock::Create(B.getContext(), "directed.slow", Fn);
    Join = BasicBlock::Create(B.getContext(), "directed.join", Fn);
    B.CreateCondBr(B.CreateOr(Ok, Is(1)), Join, Slow);
    B.SetInsertPoint(Slow);
  }
  Value *D;
  if (FromInt) {
    bool Signed = Base == "sitofp";
    Value *X = Signed ? B.CreateSExt(A[0], I64) : B.CreateZExt(A[0], I64);
    StringRef FromName =
        Signed ? "__air_f64_from_i64_r" : "__air_f64_from_u64_r";
    if (auto *RC = dyn_cast<ConstantInt>(Rounding); RC && RC->isOne()) {
      // To nearest, an integer of more than 53 bits would round twice: the
      // double toward zero, odd when inexact (rounding to odd keeps what
      // rounding to nearest at 8 to 24 bits needs).
      D = Lib(FromName, I64, {X, B.getInt32(0)});
      Value *Back =
          Lib(Signed ? "__air_f64_to_i64" : "__air_f64_to_u64", I64, {D});
      D = B.CreateOr(D, B.CreateZExt(B.CreateICmpNE(Back, X), I64));
    } else {
      D = Lib(FromName, I64, {X, Rounding});
    }
  } else if (Base == "fptrunc") {
    D = Ext(A[0]);
  } else {
    auto [Name, N] = StringSwitch<std::pair<StringRef, unsigned>>(Base)
                         .Case("fadd", {"add", 2})
                         .Case("fsub", {"sub", 2})
                         .Case("fmul", {"mul", 2})
                         .Case("fdiv", {"div", 2})
                         .Case("sqrt", {"sqrt", 1})
                         .Default({"fma", 3});
    SmallVector<Value *, 4> Args;
    for (Value *X : A.take_front(N))
      Args.push_back(Ext(X));
    Args.push_back(Rounding);
    D = Lib(("__air_f64_" + Name + "_r").str(), I64, Args);
  }
  StringRef To = ST->isFloatTy()  ? "__air_f64_to_f32_r"
                 : ST->isHalfTy() ? "__air_f64_to_f16_r"
                                  : "__air_f64_to_bf16_r";
  Value *Bits = Lib(To, I32, {D, Rounding});
  if (!ST->isFloatTy())
    Bits = B.CreateTrunc(Bits, I16);
  Value *R = B.CreateBitCast(Bits, ST);
  if (Join) {
    BasicBlock *SlowEnd = B.GetInsertBlock();
    B.CreateBr(Join);
    B.SetInsertPoint(Join);
    PHINode *P = B.CreatePHI(ST, 2);
    P->addIncoming(Fast, FastBB);
    P->addIncoming(R, SlowEnd);
    R = P;
  }
  return R;
}

class EmulateF64 {
  Module &M;
  LLVMContext &Ctx;
  bool Failed = false;
  bool NeedSoftF64 = false;
  unsigned HintCount = 0;
  /// Emulations by key and type, before the types change.
  std::map<std::pair<std::string, FunctionType *>, Function *> Emulations;

public:
  EmulateF64(Module &M) : M(M), Ctx(M.getContext()) {}
  bool run();

private:
  void error(const Function *F, const Twine &Msg) {
    Ctx.diagnose(DiagnosticInfoUnsupported(F ? *F : *M.begin(),
                                           "air-emulate-f64: " + Msg));
    Failed = true;
  }
  bool hasWork();
  Function *emulation(StringRef Key, FunctionType *FT);
  void replaceWithEmulation(Instruction &I, StringRef Key,
                            ArrayRef<Value *> Args);
  void replaceOperations(Function &F);
  void doublesToBits();
  void emulate(Function &F, StringRef Key);
  void linkSoftF64();
  void inlineEmulations();
};

/// Whether M has a double or a constrained intrinsic (otherwise the pass
/// leaves it as it is).
/// The module asks for IEEE f32 (a front end's strict mode: the module flag
/// air.ieee-fp32) and F keeps f32 denormals: its f32 arithmetic must give
/// denormal results and read denormal operands, which a GPU that flushes them
/// (Metal's) does not.
static bool ieeeF32(const Function &F) {
  auto *Flag = mdconst::extract_or_null<ConstantInt>(
      F.getParent()->getModuleFlag("air.ieee-fp32"));
  return Flag && !Flag->isZero() &&
         F.getDenormalMode(APFloat::IEEEsingle()).Output == DenormalMode::IEEE;
}

/// An f32 operation that ieee32 emulates: add, sub, mul, div, fma, sqrt.
static StringRef ieeeF32Op(const Instruction &I) {
  if (!I.getType()->isFloatTy())
    return "";
  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    switch (BO->getOpcode()) {
    case Instruction::FAdd:
      return "add";
    case Instruction::FSub:
      return "sub";
    case Instruction::FMul:
      return "mul";
    case Instruction::FDiv:
      return "div";
    default:
      return "";
    }
  }
  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    if (II->getIntrinsicID() == Intrinsic::fma ||
        II->getIntrinsicID() == Intrinsic::fmuladd)
      return "fma";
    if (II->getIntrinsicID() == Intrinsic::sqrt && !II->hasApproxFunc())
      return "sqrt";
    // the constrained forms rounding to nearest (a front end's module in
    // constrained mode): the same operations
    if (auto *C = dyn_cast<ConstrainedFPIntrinsic>(II)) {
      std::optional<RoundingMode> R = C->getRoundingMode();
      if (!R || (*R != RoundingMode::NearestTiesToEven &&
                 *R != RoundingMode::Dynamic))
        return "";
      switch (C->getIntrinsicID()) {
      case Intrinsic::experimental_constrained_fadd:
        return "add";
      case Intrinsic::experimental_constrained_fsub:
        return "sub";
      case Intrinsic::experimental_constrained_fmul:
        return "mul";
      case Intrinsic::experimental_constrained_fdiv:
        return "div";
      case Intrinsic::experimental_constrained_fma:
      case Intrinsic::experimental_constrained_fmuladd:
        return "fma";
      case Intrinsic::experimental_constrained_sqrt:
        return "sqrt";
      default:
        return "";
      }
    }
  }
  return "";
}

bool EmulateF64::hasWork() {
  for (Function &F : M)
    if (!F.isDeclaration() && ieeeF32(F))
      for (Instruction &I : instructions(F))
        if (!ieeeF32Op(I).empty())
          return true;
  for (GlobalVariable &GV : M.globals())
    if (hasDouble(GV.getValueType()))
      return true;
  for (Function &F : M) {
    if (hasDouble(F.getFunctionType()) && (!F.isDeclaration() || !F.use_empty()))
      return true;
    if (isConstrained(F) && !F.use_empty())
      return true;
    for (Instruction &I : instructions(F)) {
      if (hasDouble(I.getType()))
        return true;
      for (Value *Op : I.operands())
        if (hasDouble(Op->getType()))
          return true;
      if (auto *AI = dyn_cast<AllocaInst>(&I);
          AI && hasDouble(AI->getAllocatedType()))
        return true;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I);
          GEP && hasDouble(GEP->getSourceElementType()))
        return true;
    }
  }
  return false;
}

Function *EmulateF64::emulation(StringRef Key, FunctionType *FT) {
  Function *&F = Emulations[{Key.str(), FT}];
  if (F)
    return F;
  std::string Name = ("air.emulate." + Key).str();
  for (char &C : Name)
    if (!isAlnum(C) && C != '.' && C != '_')
      C = '_';
  F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, &M);
  F->addFnAttr(EmulationAttr, Key);
  return F;
}

void EmulateF64::replaceWithEmulation(Instruction &I, StringRef Key,
                                      ArrayRef<Value *> Args) {
  SmallVector<Type *, 4> Tys;
  for (Value *A : Args)
    Tys.push_back(A->getType());
  FunctionType *FT = FunctionType::get(I.getType(), Tys, false);
  CallInst *New =
      CallInst::Create(emulation(Key, FT), Args, "", I.getIterator());
  New->takeName(&I);
  New->setDebugLoc(I.getDebugLoc());
  I.replaceAllUsesWith(New);
  I.eraseFromParent();
}

/// The double precision operations of F and its constrained intrinsics
/// become calls of emulations.
void EmulateF64::replaceOperations(Function &F) {
  for (Instruction &I : make_early_inc_range(instructions(F)))
    scalarizeDoubles(I);
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    // No fast-math flags on what becomes an integer (a phi, a select, a
    // call of doubles).
    if (isa<FPMathOperator>(&I) && hasDouble(I.getType()) && !isDoubleOp(I))
      I.copyFastMathFlags(FastMathFlags());
    if (StringRef Name = ieeeF32Op(I); !Name.empty() && ieeeF32(F)) {
      // "ieee32:<op>" (before the calls below: fma and sqrt are intrinsic calls): the f32 operation with denormal operands and results
      SmallVector<Value *, 3> Ops;
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        for (Value *A : II->args())
          if (!isa<MetadataAsValue>(A)) // (a constrained form's rounding)
            Ops.push_back(A);
      } else
        Ops.append(I.op_begin(), I.op_end());
      replaceWithEmulation(I, ("ieee32:" + Name).str(), Ops);
      continue;
    }
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      StringRef Name = Callee->getName();
      bool Dbl = isDouble(CI->getType()) ||
                 any_of(CI->args(),
                        [](const Use &U) { return isDouble(U->getType()); });
      if (StringRef As = airAsIntrinsic(Name); !As.empty() && Dbl)
        Name = As;
      else if (!Callee->isIntrinsic())
        continue;
      bool Constrained = Name.starts_with("llvm.experimental.constrained.");
      if (!Constrained && !(Dbl && emulatesDouble(doubleOpBase(Name))))
        continue;
      // "call:<intrinsic>|<double mask>[|<metadata operand>...]"
      SmallVector<Value *, 4> Args;
      SmallVector<Type *, 4> Tys;
      std::string MD;
      for (Value *A : CI->args()) {
        if (auto *MV = dyn_cast<MetadataAsValue>(A)) {
          if (auto *S = dyn_cast<MDString>(MV->getMetadata()))
            MD += "|" + S->getString().str();
          continue;
        }
        Args.push_back(A);
        Tys.push_back(A->getType());
      }
      std::string Key =
          ("call:" + Name + "|" + doubleMask(CI->getType(), Tys) + MD).str();
      // bfloat is computed in float where the function keeps float
      // denormals (emulate()).
      bool BF16 = CI->getType()->getScalarType()->isBFloatTy() ||
                  any_of(Tys, [](Type *T) {
                    return T->getScalarType()->isBFloatTy();
                  });
      if (Constrained && BF16 &&
          F.getDenormalMode(APFloat::IEEEsingle()) == DenormalMode::getIEEE())
        Key += "|+f32-denormals";
      replaceWithEmulation(*CI, Key, Args);
      continue;
    }
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
        RMW && RMW->getValOperand()->getType()->isDoubleTy()) {
      // "atomic:<operation>|d-d|<ordering>|<sync scope>|<align>"
      SmallVector<StringRef, 4> Scopes;
      Ctx.getSyncScopeNames(Scopes);
      std::string Key =
          ("atomic:" + AtomicRMWInst::getOperationName(RMW->getOperation()) +
           "|d-d|" + Twine(unsigned(RMW->getOrdering())) + "|" +
           Scopes[RMW->getSyncScopeID()] + "|" +
           Twine(RMW->getAlign().value()))
              .str();
      replaceWithEmulation(*RMW, Key,
                           {RMW->getPointerOperand(), RMW->getValOperand()});
      continue;
    }
    if (isDoubleOp(I)) {
      // "inst:<opcode>[:<predicate>]|<double mask>"
      SmallVector<Value *, 3> Ops(I.operands());
      SmallVector<Type *, 3> Tys;
      for (Value *O : Ops)
        Tys.push_back(O->getType());
      std::string Key = ("inst:" + Twine(I.getOpcodeName())).str();
      if (auto *C = dyn_cast<FCmpInst>(&I))
        Key += ":" + CmpInst::getPredicateName(C->getPredicate()).str();
      Key += "|" + doubleMask(I.getType(), Tys);
      replaceWithEmulation(I, Key, Ops);
    }
  }
}

/// Every double of the module becomes an i64: the globals and functions
/// whose types have doubles are made anew with their bits, the bodies with
/// doubles cloned into them.
void EmulateF64::doublesToBits() {
  F64ToBits Types;
  ValueToValueMapTy VMap;
  DoubleMaterializer Materializer;
  SmallVector<std::pair<GlobalVariable *, GlobalVariable *>, 8> Globals;
  SmallVector<std::pair<Function *, Function *>, 8> Functions;
  SmallVector<GlobalVariable *, 8> OldGlobals;
  for (GlobalVariable &GV : M.globals())
    OldGlobals.push_back(&GV);
  SmallVector<Function *, 16> OldFunctions;
  for (Function &F : M)
    OldFunctions.push_back(&F);
  for (GlobalVariable *GV : OldGlobals) {
    Type *T = Types.remapType(GV->getValueType());
    if (T == GV->getValueType())
      continue;
    auto *NewGV = new GlobalVariable(
        M, T, GV->isConstant(), GV->getLinkage(), nullptr,
        GV->getName() + ".f64", nullptr, GV->getThreadLocalMode(),
        GV->getAddressSpace());
    NewGV->copyAttributesFrom(GV);
    VMap[GV] = NewGV;
    Globals.push_back({GV, NewGV});
  }
  // A function changes when its type or its body has doubles.
  auto BodyHasDouble = [](Function &F) {
    for (Instruction &I : instructions(F)) {
      if (hasDouble(I.getType()) ||
          any_of(I.operands(),
                 [](const Use &U) { return hasDouble(U->getType()); }))
        return true;
      if (auto *AI = dyn_cast<AllocaInst>(&I);
          AI && hasDouble(AI->getAllocatedType()))
        return true;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I);
          GEP && hasDouble(GEP->getSourceElementType()))
        return true;
    }
    return false;
  };
  for (Function *F : OldFunctions) {
    auto *FT = cast<FunctionType>(Types.remapType(F->getFunctionType()));
    if (F->isIntrinsic()) {
      if (FT == F->getFunctionType())
        continue;
      SmallVector<Type *, 4> Overloads;
      if (F->use_empty())
        continue;
      if (!Intrinsic::isSignatureValid(F->getIntrinsicID(), FT, Overloads)) {
        // Reported at each function that calls it; the calls go, so that
        // the other operations are checked too.
        SmallPtrSet<Function *, 4> Reported;
        for (User *U : make_early_inc_range(F->users()))
          if (auto *CI = dyn_cast<CallInst>(U)) {
            if (Reported.insert(CI->getFunction()).second)
              error(CI->getFunction(), "no emulation of " + F->getName());
            CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
            CI->eraseFromParent();
          }
        continue;
      }
      VMap[F] =
          Intrinsic::getOrInsertDeclaration(&M, F->getIntrinsicID(), Overloads);
      continue;
    }
    if (FT == F->getFunctionType() && (F->isDeclaration() || !BodyHasDouble(*F)))
      continue;
    // (a declaration of doubles nothing calls any more: an AIR function of
    // double precision that has its emulation now)
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      continue;
    }
    Function *NewF = Function::Create(FT, F->getLinkage(), F->getAddressSpace(),
                                      F->getName() + ".f64", &M);
    NewF->copyAttributesFrom(F);
    VMap[F] = NewF;
    Functions.push_back({F, NewF});
  }
  // Initializers, with their doubles as bits.
  std::function<Constant *(Constant *)> MapInit = [&](Constant *C) {
    if (Constant *Bits = doubleBits(C))
      return Bits;
    if (isa<ConstantAggregate>(C)) {
      SmallVector<Constant *, 8> Ops;
      for (Use &U : C->operands())
        Ops.push_back(MapInit(cast<Constant>(U)));
      Type *T = Types.remapType(C->getType());
      if (auto *AT = dyn_cast<ArrayType>(T))
        return ConstantArray::get(AT, Ops);
      if (auto *ST = dyn_cast<StructType>(T))
        return ConstantStruct::get(ST, Ops);
      return ConstantVector::get(Ops);
    }
    return MapValue(C, VMap, RF_None, &Types, &Materializer);
  };
  for (auto [Old, New] : Globals)
    if (Old->hasInitializer())
      New->setInitializer(MapInit(Old->getInitializer()));
  for (auto [Old, New] : Functions) {
    if (Old->isDeclaration())
      continue;
    auto NewArg = New->arg_begin();
    for (Argument &A : Old->args()) {
      NewArg->setName(A.getName());
      VMap[&A] = &*NewArg++;
    }
    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(New, Old, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr, &Types, &Materializer);
  }
  // The old ones go; what else refers to them (other functions, metadata,
  // initializers) takes the new ones, which are pointers as they were.
  for (auto [Old, New] : Functions)
    Old->dropAllReferences();
  for (auto [Old, New] : Globals)
    Old->setInitializer(nullptr);
  for (auto [Old, New] : Functions) {
    std::string Name = Old->getName().str();
    Old->replaceAllUsesWith(New);
    // The new function takes the old one's place (the order of a module's
    // functions is the order of its AIR).
    New->removeFromParent();
    M.getFunctionList().insert(Old->getIterator(), New);
    Old->eraseFromParent();
    New->setName(Name);
  }
  for (auto [Old, New] : Globals) {
    std::string Name = Old->getName().str();
    Old->replaceAllUsesWith(New);
    New->removeFromParent();
    M.insertGlobalVariable(Old->getIterator(), New);
    Old->eraseFromParent();
    New->setName(Name);
  }
  for (Function &F : make_early_inc_range(M))
    if (F.isIntrinsic() && F.use_empty())
      F.eraseFromParent();
}

/// The body of emulation F, of the operation Key.
void EmulateF64::emulate(Function &F, StringRef Key) {
  F.setLinkage(GlobalValue::InternalLinkage);
  BasicBlock *BB = BasicBlock::Create(Ctx, "", &F);
  IRBuilder<> B(BB);
  SmallVector<Value *, 4> A;
  for (Argument &Arg : F.args())
    A.push_back(&Arg);
  Type *Ret = F.getReturnType();
  Type *I32 = B.getInt32Ty(), *I64 = B.getInt64Ty(), *F32 = B.getFloatTy();
  auto Lib = [&](StringRef Name, Type *R, ArrayRef<Value *> Args) -> Value * {
    // A declaration, which linkSoftF64 defines once the bodies are written.
    SmallVector<Type *, 3> Tys;
    for (Value *A : Args)
      Tys.push_back(A->getType());
    NeedSoftF64 = true;
    return B.CreateCall(
        M.getOrInsertFunction(Name, FunctionType::get(R, Tys, false)), Args);
  };
  auto Fin = [&](Value *V) {
    if (Ret->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(V ? V : PoisonValue::get(Ret));
  };
  auto NoEmulation = [&] {
    while (!BB->empty())
      BB->back().eraseFromParent();
    for (BasicBlock &Other : make_early_inc_range(drop_begin(F)))
      Other.eraseFromParent();
    B.SetInsertPoint(BB);
    for (User *U : F.users())
      if (auto *CI = dyn_cast<CallInst>(U)) {
        error(CI->getFunction(), "no emulation of " + Key);
        break;
      }
    Fin(nullptr);
  };

  SmallVector<StringRef, 4> Parts;
  Key.split(Parts, '|');
  StringRef Op = Parts[0], Mask = Parts.size() > 1 ? Parts[1] : "";
  if (StringRef Name = Op; Name.consume_front("ieee32:")) {
    // The hardware's result where no operand is a denormal and the result is
    // normal (an exponent field that is not zero: also infinities and NaNs),
    // where flushing changes nothing; else soft-f64 on the exact operands,
    // rounded to odd (toward zero, the last bit set where rounding away from
    // zero differs) and then to nearest f32, which rounds once (53 >= 24 + 2
    // bits).
    auto Bits = [&](Value *V) { return B.CreateBitCast(V, I32); };
    auto Denormal = [&](Value *V) {
      Value *X = Bits(V);
      return B.CreateAnd(
          B.CreateICmpEQ(B.CreateAnd(X, 0x7f800000), B.getInt32(0)),
          B.CreateICmpNE(B.CreateAnd(X, 0x7fffff), B.getInt32(0)));
    };
    Value *Fast;
    if (Name == "add")
      Fast = B.CreateFAdd(A[0], A[1]);
    else if (Name == "sub")
      Fast = B.CreateFSub(A[0], A[1]);
    else if (Name == "mul")
      Fast = B.CreateFMul(A[0], A[1]);
    else if (Name == "div")
      Fast = B.CreateFDiv(A[0], A[1]);
    else if (Name == "fma")
      Fast = B.CreateIntrinsic(Intrinsic::fma, {F32}, {A[0], A[1], A[2]});
    else if (Name == "sqrt")
      Fast = B.CreateUnaryIntrinsic(Intrinsic::sqrt, A[0]);
    else
      return NoEmulation();
    Value *Ok = B.CreateICmpNE(B.CreateAnd(Bits(Fast), 0x7f800000),
                               B.getInt32(0));
    for (Value *X : A)
      Ok = B.CreateAnd(Ok, B.CreateNot(Denormal(X)));
    BasicBlock *Slow = BasicBlock::Create(Ctx, "ieee32.slow", &F);
    BasicBlock *Done = BasicBlock::Create(Ctx, "ieee32.done", &F);
    BasicBlock *FastEnd = B.GetInsertBlock();
    B.CreateCondBr(Ok, Done, Slow);
    B.SetInsertPoint(Slow);
    SmallVector<Value *, 4> D;
    for (Value *X : A)
      D.push_back(Lib("__air_f64_from_f32", I64, {Bits(X)}));
    std::string Lname = ("__air_f64_" + Name + "_r").str();
    auto Call = [&](Value *Rounding) {
      SmallVector<Value *, 4> Args(D);
      Args.push_back(Rounding);
      return Lib(Lname, I64, Args);
    };
    Value *TZ = Call(B.getInt32(0));
    Value *Away = Call(B.CreateSelect(B.CreateICmpSLT(TZ, B.getInt64(0)),
                                      B.getInt32(3), B.getInt32(2)));
    Value *Odd = B.CreateOr(TZ, B.CreateZExt(B.CreateICmpNE(TZ, Away), I64));
    // (the paths meet on the bits: a flushing GPU may flush a float on its
    // way through a phi)
    Value *SlowR = Lib("__air_f64_to_f32_r", I32, {Odd, B.getInt32(1)});
    BasicBlock *SlowEnd = B.GetInsertBlock();
    B.CreateBr(Done);
    B.SetInsertPoint(FastEnd->getTerminator());
    Value *FastBits = Bits(Fast);
    B.SetInsertPoint(Done);
    PHINode *P = B.CreatePHI(I32, 2);
    P->addIncoming(FastBits, FastEnd);
    P->addIncoming(SlowR, SlowEnd);
    return Fin(B.CreateBitCast(P, F32));
  }
  if (StringRef RMW = Op; RMW.consume_front("atomic:")) {
    // A double precision read-modify-write as a compare-exchange loop of its
    // bits, with the operation in soft-f64 (LLVM's 64-bit cmpxchg, which
    // the AIR consumers have or emulate).
    unsigned Order = 0, Alignment = 8;
    if (Parts.size() != 5 || Parts[2].getAsInteger(10, Order) ||
        Parts[4].getAsInteger(10, Alignment) || A.size() != 2 ||
        !is_contained({"fadd", "fsub", "fmax", "fmin", "xchg"}, RMW))
      return NoEmulation();
    Value *Ptr = A[0], *X = A[1];
    auto Ordering = AtomicOrdering(Order);
    SyncScope::ID Scope = Ctx.getOrInsertSyncScopeID(Parts[3]);
    if (RMW == "xchg")
      return Fin(B.CreateAtomicRMW(AtomicRMWInst::Xchg, Ptr, X,
                                   Align(Alignment), Ordering, Scope));
    Value *Init = B.CreateAlignedLoad(I64, Ptr, Align(Alignment), true);
    BasicBlock *Loop = BasicBlock::Create(Ctx, "loop", &F);
    BasicBlock *Done = BasicBlock::Create(Ctx, "done", &F);
    B.CreateBr(Loop);
    B.SetInsertPoint(Loop);
    PHINode *Old = B.CreatePHI(I64, 2);
    Old->addIncoming(Init, BB);
    Value *New;
    if (RMW == "fadd" || RMW == "fsub") {
      New = Lib(RMW == "fadd" ? "__air_f64_add" : "__air_f64_sub", I64,
                {Old, X});
    } else {
      // maxnum / minnum: the other operand of a NaN.
      Value *Lt = B.CreateICmpNE(
          Lib("__air_f64_lt", I32,
              RMW == "fmin" ? ArrayRef<Value *>{X, Old}
                            : ArrayRef<Value *>{Old, X}),
          B.getInt32(0));
      Value *OldNaN = B.CreateICmpNE(Lib("__air_f64_unord", I32, {Old, Old}),
                                     B.getInt32(0));
      New = B.CreateSelect(B.CreateOr(Lt, OldNaN), X, Old);
    }
    AtomicCmpXchgInst *CX =
        B.CreateAtomicCmpXchg(Ptr, Old, New, Align(Alignment), Ordering,
                              AtomicOrdering::Monotonic, Scope);
    Value *Loaded = B.CreateExtractValue(CX, 0);
    Old->addIncoming(Loaded, Loop);
    B.CreateCondBr(B.CreateExtractValue(CX, 1), Done, Loop);
    B.SetInsertPoint(Done);
    return Fin(Loaded);
  }
  bool DblIn = Mask.size() > 1 && Mask[1] == 'd';
  bool DblOut = !Mask.empty() && Mask[0] == 'd';
  // The rounding direction, numbered like FLT_ROUNDS (0 toward zero, 1 to
  // nearest even, 2 upward, 3 downward): the metadata's. AIR has no
  // floating point environment: a dynamic rounding is to nearest.
  unsigned RoundingMode = 1;
  for (StringRef P : drop_begin(Parts, 2))
    if (P.consume_front("round."))
      RoundingMode = StringSwitch<unsigned>(P)
                         .Case("towardzero", 0)
                         .Case("upward", 2)
                         .Case("downward", 3)
                         .Default(1);
  Value *Rounding = B.getInt32(RoundingMode);
  bool Nearest = RoundingMode == 1;
  // The operation of an intrinsic or of an instruction.
  StringRef Base = Op;
  if (Base.consume_front("call:"))
    Base = doubleOpBase(Base);
  else
    Base.consume_front("inst:");

  auto Pred = [&]() -> CmpInst::Predicate {
    // fcmp/fcmps metadata operand ("oeq"), or the instruction's predicate
    StringRef P = Op.contains("constrained.fcmp")
                      ? (Parts.size() > 2 ? Parts[2] : StringRef())
                      : Op.rsplit(':').second;
    return StringSwitch<CmpInst::Predicate>(P)
        .Case("false", CmpInst::FCMP_FALSE)
        .Case("oeq", CmpInst::FCMP_OEQ)
        .Case("ogt", CmpInst::FCMP_OGT)
        .Case("oge", CmpInst::FCMP_OGE)
        .Case("olt", CmpInst::FCMP_OLT)
        .Case("ole", CmpInst::FCMP_OLE)
        .Case("one", CmpInst::FCMP_ONE)
        .Case("ord", CmpInst::FCMP_ORD)
        .Case("uno", CmpInst::FCMP_UNO)
        .Case("ueq", CmpInst::FCMP_UEQ)
        .Case("ugt", CmpInst::FCMP_UGT)
        .Case("uge", CmpInst::FCMP_UGE)
        .Case("ult", CmpInst::FCMP_ULT)
        .Case("ule", CmpInst::FCMP_ULE)
        .Case("une", CmpInst::FCMP_UNE)
        .Case("true", CmpInst::FCMP_TRUE)
        .Default(CmpInst::BAD_FCMP_PREDICATE);
  };

  if (DblIn || DblOut) {
    // Double precision, in the soft-f64 library.
    StringRef B2 = Base.starts_with("fcmp") ? "fcmp" : Base;
    Value *R = nullptr;
    auto Bool = [&](Value *V) { return B.CreateICmpNE(V, B.getInt32(0)); };
    const uint64_t SignBit = 0x8000000000000000ull;
    const uint64_t AbsMask = 0x7fffffffffffffffull;
    // The function rounded to nearest, or its _r form in the rounding
    // direction.
    auto RLib = [&](StringRef Name, Type *R, ArrayRef<Value *> Args) {
      if (Nearest)
        return Lib(Name, R, Args);
      SmallVector<Value *, 4> RArgs(Args);
      RArgs.push_back(Rounding);
      return Lib((Name + "_r").str(), R, RArgs);
    };
    auto Floor = [&](Value *X) { return Lib("__air_f64_floor", I64, {X}); };
    auto Trunc = [&](Value *X) {
      return B.CreateOr(Floor(B.CreateAnd(X, B.getInt64(AbsMask))),
                        B.CreateAnd(X, B.getInt64(SignBit)));
    };
    auto Ceil = [&](Value *X) {
      return B.CreateXor(Floor(B.CreateXor(X, B.getInt64(SignBit))),
                         B.getInt64(SignBit));
    };
    if (B2 == "fadd")
      R = RLib("__air_f64_add", I64, {A[0], A[1]});
    else if (B2 == "fsub")
      R = RLib("__air_f64_sub", I64, {A[0], A[1]});
    else if (B2 == "fmul")
      R = RLib("__air_f64_mul", I64, {A[0], A[1]});
    else if (B2 == "fdiv")
      R = RLib("__air_f64_div", I64, {A[0], A[1]});
    else if (B2 == "fma" || B2 == "fmuladd")
      R = RLib("__air_f64_fma", I64, {A[0], A[1], A[2]});
    else if (B2 == "sqrt")
      R = RLib("__air_f64_sqrt", I64, {A[0]});
    else if (B2 == "fneg")
      R = B.CreateXor(A[0], B.getInt64(SignBit));
    else if (B2 == "fabs")
      R = B.CreateAnd(A[0], B.getInt64(AbsMask));
    else if (B2 == "copysign")
      R = B.CreateOr(B.CreateAnd(A[0], B.getInt64(AbsMask)),
                     B.CreateAnd(A[1], B.getInt64(SignBit)));
    else if (B2 == "canonicalize")
      R = A[0];
    else if (B2 == "floor")
      R = Floor(A[0]);
    else if (B2 == "ceil")
      R = Ceil(A[0]);
    else if (B2 == "trunc")
      R = Trunc(A[0]);
    else if (B2 == "round") {
      // Half way cases away from zero: |x| rounded down, and up when the
      // (exact) fraction is at least 1/2.
      Value *Abs = B.CreateAnd(A[0], B.getInt64(AbsMask));
      Value *Down = Floor(Abs);
      Value *Frac = Lib("__air_f64_sub", I64, {Abs, Down});
      Value *Below = Bool(Lib("__air_f64_lt", I32,
                              {Frac, B.getInt64(0x3fe0000000000000ull)}));
      Value *Up = Lib("__air_f64_add", I64,
                      {Down, B.getInt64(0x3ff0000000000000ull)});
      R = B.CreateOr(B.CreateSelect(Below, Down, Up),
                     B.CreateAnd(A[0], B.getInt64(SignBit)));
    } else if (B2 == "roundeven")
      R = Lib("__air_f64_roundeven", I64, {A[0]});
    else if (B2 == "rint" || B2 == "nearbyint") {
      // In the rounding direction.
      R = Lib("__air_f64_roundeven", I64, {A[0]});
      if (!Nearest)
        R = RoundingMode == 0 ? Trunc(A[0])
            : RoundingMode == 2 ? Ceil(A[0])
                                : Floor(A[0]);
    } else if (B2 == "minnum" || B2 == "maxnum") {
      // The non-NaN operand when one is NaN.
      bool Min = B2 == "minnum";
      Value *Lt = Bool(Lib("__air_f64_lt", I32,
                           Min ? ArrayRef<Value *>{A[0], A[1]}
                               : ArrayRef<Value *>{A[1], A[0]}));
      Value *ANaN = Bool(Lib("__air_f64_unord", I32, {A[0], A[0]}));
      R = B.CreateSelect(ANaN, A[1], B.CreateSelect(Lt, A[0], A[1]));
    } else if (B2 == "minimum" || B2 == "maximum" || B2 == "minimumnum" ||
               B2 == "maximumnum") {
      // -0 orders below +0; minimum / maximum return a NaN operand (quiet),
      // the *num forms the other operand.
      bool Min = B2.starts_with("min");
      Value *X = A[0], *Y = A[1];
      Value *Lt = Bool(Lib("__air_f64_lt", I32,
                           Min ? ArrayRef<Value *>{X, Y}
                               : ArrayRef<Value *>{Y, X}));
      Value *Eq = Bool(Lib("__air_f64_eq", I32, {X, Y}));
      // equal values: the one with the sign bit for min, without for max
      Value *XNeg = B.CreateICmpSLT(X, B.getInt64(0));
      Value *PickX =
          B.CreateOr(Lt, B.CreateAnd(Eq, Min ? XNeg : B.CreateNot(XNeg)));
      Value *XNaN = Bool(Lib("__air_f64_unord", I32, {X, X}));
      Value *YNaN = Bool(Lib("__air_f64_unord", I32, {Y, Y}));
      R = B.CreateSelect(PickX, X, Y);
      Value *Quiet = B.getInt64(0x0008000000000000ull);
      if (B2.ends_with("num")) {
        R = B.CreateSelect(YNaN, X, R);
        R = B.CreateSelect(XNaN, Y, R);
      } else {
        R = B.CreateSelect(YNaN, B.CreateOr(Y, Quiet), R);
        R = B.CreateSelect(XNaN, B.CreateOr(X, Quiet), R);
      }
    } else if (B2 == "fcmp") {
      Value *Uno = Bool(Lib("__air_f64_unord", I32, {A[0], A[1]}));
      Value *Eq = Bool(Lib("__air_f64_eq", I32, {A[0], A[1]}));
      Value *Lt = Bool(Lib("__air_f64_lt", I32, {A[0], A[1]}));
      Value *Gt = Bool(Lib("__air_f64_lt", I32, {A[1], A[0]}));
      switch (Pred()) {
      case CmpInst::FCMP_FALSE: R = B.getFalse(); break;
      case CmpInst::FCMP_OEQ: R = Eq; break;
      case CmpInst::FCMP_OGT: R = Gt; break;
      case CmpInst::FCMP_OGE: R = B.CreateOr(Gt, Eq); break;
      case CmpInst::FCMP_OLT: R = Lt; break;
      case CmpInst::FCMP_OLE: R = B.CreateOr(Lt, Eq); break;
      case CmpInst::FCMP_ONE: R = B.CreateOr(Lt, Gt); break;
      case CmpInst::FCMP_ORD: R = B.CreateNot(Uno); break;
      case CmpInst::FCMP_UNO: R = Uno; break;
      case CmpInst::FCMP_UEQ: R = B.CreateOr(Uno, Eq); break;
      case CmpInst::FCMP_UGT: R = B.CreateOr(Uno, Gt); break;
      case CmpInst::FCMP_UGE: R = B.CreateNot(Lt); break;
      case CmpInst::FCMP_ULT: R = B.CreateOr(Uno, Lt); break;
      case CmpInst::FCMP_ULE: R = B.CreateNot(Gt); break;
      case CmpInst::FCMP_UNE: R = B.CreateNot(Eq); break;
      case CmpInst::FCMP_TRUE: R = B.getTrue(); break;
      default: break;
      }
    } else if ((B2 == "fptosi" || B2 == "fptoui") && Ret->isIntegerTy() &&
               Ret->getIntegerBitWidth() <= 64) {
      // Saturating, NaN to 0 (llvm.fpto[su]i.sat; out of range values are
      // poison for the instructions).
      bool Signed = B2 == "fptosi";
      R = Lib(Signed ? "__air_f64_to_i64" : "__air_f64_to_u64", I64, {A[0]});
      unsigned W = Ret->getIntegerBitWidth();
      if (W < 64) {
        R = Signed
                ? B.CreateBinaryIntrinsic(
                      Intrinsic::smax,
                      B.CreateBinaryIntrinsic(
                          Intrinsic::smin, R,
                          B.getInt64(APInt::getSignedMaxValue(W).getSExtValue())),
                      B.getInt64(APInt::getSignedMinValue(W).getSExtValue()))
                : B.CreateBinaryIntrinsic(
                      Intrinsic::umin, R,
                      B.getInt64(APInt::getMaxValue(W).getZExtValue()));
        R = B.CreateTrunc(R, Ret);
      }
    } else if ((B2 == "sitofp" || B2 == "uitofp") &&
               A[0]->getType()->getIntegerBitWidth() <= 64) {
      bool Signed = B2 == "sitofp";
      Value *X = Signed ? B.CreateSExt(A[0], I64) : B.CreateZExt(A[0], I64);
      R = RLib(Signed ? "__air_f64_from_i64" : "__air_f64_from_u64", I64, {X});
    } else if (B2 == "fpext") {
      Value *X = A[0];
      if (X->getType()->isBFloatTy())
        X = bf16ToF32(B, X);
      else if (!X->getType()->isFloatTy())
        X = B.CreateFPExt(X, F32);
      R = Lib("__air_f64_from_f32", I64, {B.CreateBitCast(X, I32)});
    } else if (B2 == "fptrunc") {
      // Rounded once, to the result's format.
      if (Ret->isFloatTy())
        R = B.CreateBitCast(RLib("__air_f64_to_f32", I32, {A[0]}), F32);
      else if (Ret->isHalfTy() || Ret->isBFloatTy())
        R = B.CreateBitCast(
            B.CreateTrunc(Lib(Ret->isHalfTy() ? "__air_f64_to_f16_r"
                                              : "__air_f64_to_bf16_r",
                              I32, {A[0], Rounding}),
                          B.getInt16Ty()),
            Ret);
    }
    if (!R)
      return NoEmulation();
    return Fin(R);
  }

  // A constrained intrinsic of another format: to nearest the AIR
  // operation; directed, computed in double precision in that direction and
  // rounded again in it (exact: the values of the format are doubles).
  if (!Op.starts_with("call:llvm.experimental.constrained."))
    return NoEmulation();
  if (!Nearest)
    if (Value *R = directed(B, Base, A, Ret, Rounding, Lib))
      return Fin(R);
  // bfloat: in float, exactly (the operands are floats, and a result of an
  // operation of bfloat operands rounded to float and then to bfloat is
  // rounded once), or soft-f64 (fma, from integers).
  Type *ArgTy = A.empty() ? Ret : A[0]->getType();
  bool BF16 = ArgTy->getScalarType()->isBFloatTy() ||
              Ret->getScalarType()->isBFloatTy();
  if (BF16) {
    if (Base == "fpext" && Ret->getScalarType()->isFloatTy())
      return Fin(bf16ToF32(B, A[0]));
    if (Base == "fptrunc" && ArgTy->getScalarType()->isFloatTy())
      return Fin(f32ToBF16(B, A[0]));
    // In float, a bfloat denormal (a float denormal) would flush where the
    // function's float denormals do; bfloat has no such mode. There the
    // arithmetic is soft-f64's.
    bool FloatKeepsDenormals = is_contained(Parts, "+f32-denormals");
    bool Arith = is_contained({"fadd", "fsub", "fmul", "fdiv", "sqrt"}, Base);
    if (Base == "fma" || Base == "fmuladd" || Base == "sitofp" ||
        Base == "uitofp" || Base == "fptrunc" ||
        (Arith && !FloatKeepsDenormals)) {
      if (Value *R = directed(B, Base, A, Ret, Rounding, Lib))
        return Fin(R);
      return NoEmulation();
    }
  }
  SmallVector<Value *, 4> X(A);
  Type *RetTy = Ret;
  if (BF16) {
    for (Value *&V : X)
      if (V->getType()->getScalarType()->isBFloatTy())
        V = bf16ToF32(B, V);
    if (Ret->getScalarType()->isBFloatTy())
      RetTy = Ret->getWithNewType(F32);
  }
  Value *R = nullptr;
  if (Base == "fadd")
    R = B.CreateFAdd(X[0], X[1]);
  else if (Base == "fsub")
    R = B.CreateFSub(X[0], X[1]);
  else if (Base == "fmul")
    R = B.CreateFMul(X[0], X[1]);
  else if (Base == "fdiv")
    R = B.CreateFDiv(X[0], X[1]);
  else if (Base == "frem")
    R = B.CreateFRem(X[0], X[1]);
  else if (Base == "fptosi")
    R = B.CreateFPToSI(X[0], RetTy);
  else if (Base == "fptoui")
    R = B.CreateFPToUI(X[0], RetTy);
  else if (Base == "sitofp")
    R = B.CreateSIToFP(X[0], RetTy);
  else if (Base == "uitofp")
    R = B.CreateUIToFP(X[0], RetTy);
  else if (Base == "fpext")
    R = B.CreateFPExt(X[0], RetTy);
  else if (Base == "fptrunc")
    R = B.CreateFPTrunc(X[0], RetTy);
  else if (Base == "fcmp" || Base == "fcmps")
    R = B.CreateFCmp(Pred(), X[0], X[1]);
  else {
    // The intrinsic of the same name (the AIR function it becomes): fma,
    // sqrt, the roundings to integers, min / max, and the functions that
    // do not round exactly (exp2, sin ...).
    Intrinsic::ID IID = StringSwitch<Intrinsic::ID>(Base)
                            .Case("fmuladd", Intrinsic::fma)
                            .Case("rint", Intrinsic::roundeven)
                            .Case("nearbyint", Intrinsic::roundeven)
                            .Default(Intrinsic::lookupIntrinsicID(
                                ("llvm." + Base).str()));
    SmallVector<Type *, 4> Tys;
    for (Value *V : X)
      Tys.push_back(V->getType());
    SmallVector<Type *, 4> Overloads;
    if (IID != Intrinsic::not_intrinsic &&
        Intrinsic::isSignatureValid(IID, FunctionType::get(RetTy, Tys, false),
                                    Overloads))
      R = B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, IID, Overloads),
                       X);
  }
  if (!R)
    return NoEmulation();
  if (RetTy != Ret)
    R = f32ToBF16(B, R);
  return Fin(R);
}

void EmulateF64::linkSoftF64() {
  SMDiagnostic Err;
  std::unique_ptr<Module> Lib = parseAssemblyString(AIRSoftF64IR, Err, Ctx);
  if (!Lib) {
    error(nullptr, "soft-f64 library: " + Err.getMessage());
    return;
  }
  Lib->setDataLayout(M.getDataLayout());
  Lib->setTargetTriple(M.getTargetTriple());
  if (Linker::linkModules(M, std::move(Lib), Linker::Flags::LinkOnlyNeeded)) {
    error(nullptr, "cannot link the soft-f64 library");
    return;
  }
  for (Function &F : M)
    if (F.getName().starts_with("__air_f64_") && !F.isDeclaration())
      F.setLinkage(GlobalValue::InternalLinkage);
}

/// The emulations, inlined at each call and simplified there (the rounding
/// is a constant at the call), so the module has only AIR operations and
/// calls of the soft-f64 library.
void EmulateF64::inlineEmulations() {
  // AIR has no floating point environment: no strictfp (whose callers would
  // turn the inlined emulations into constrained intrinsics).
  for (Function &F : M) {
    F.removeFnAttr(Attribute::StrictFP);
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        CB->removeFnAttr(Attribute::StrictFP);
  }
  SmallPtrSet<Function *, 8> Callers;
  for (Function &F : make_early_inc_range(M)) {
    if (!F.hasFnAttribute(EmulationAttr) || F.isDeclaration())
      continue;
    StringRef Key = F.getFnAttribute(EmulationAttr).getValueAsString();
    if (keepsAsHint(Key)) {
      // the hint: its body simplified below, as the inlined emulations are
      std::string Name = (air::HintPrefix + Twine(HintCount++)).str();
      F.setName(Name);
      F.addFnAttr(air::HintAttr, hintKey(Key));
      F.removeFnAttr(EmulationAttr);
      F.addFnAttr(Attribute::NoInline);
      F.setLinkage(GlobalValue::InternalLinkage);
      Callers.insert(&F);
      continue;
    }
    F.setName(""); // (its name would be left in the inlined blocks')
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &F)
        continue;
      Callers.insert(CI->getFunction());
      InlineFunctionInfo IFI;
      if (!InlineFunction(*CI, IFI).isSuccess())
        error(CI->getFunction(), "cannot inline the emulation of " +
                                     F.getFnAttribute(EmulationAttr)
                                         .getValueAsString());
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
  // Fold what the constants at the calls decide (dead soft-f64 calls are
  // memory(none), so they go too).
  const DataLayout &DL = M.getDataLayout();
  for (Function *F : Callers) {
    for (bool Changed = true; Changed;) {
      Changed = false;
      for (BasicBlock &BB : *F)
        for (Instruction &I : make_early_inc_range(BB)) {
          if (Value *V = simplifyInstruction(&I, SimplifyQuery(DL, &I))) {
            I.replaceAllUsesWith(V);
            Changed = true;
          }
          if (isInstructionTriviallyDead(&I)) {
            I.eraseFromParent();
            Changed = true;
          }
        }
      for (BasicBlock &BB : *F)
        Changed |= ConstantFoldTerminator(&BB, true);
      Changed |= removeUnreachableBlocks(*F);
    }
    for (BasicBlock &BB : make_early_inc_range(*F))
      MergeBlockIntoPredecessor(&BB);
  }
  // Library functions nothing calls any more, and the intrinsics (the
  // constrained ones) nothing calls.
  for (bool Erased = true; Erased;) {
    Erased = false;
    for (Function &F : make_early_inc_range(M))
      if (F.use_empty() &&
          ((F.hasLocalLinkage() && F.getName().starts_with("__air_f64_")) ||
           (F.isIntrinsic() && F.isDeclaration()))) {
        F.eraseFromParent();
        Erased = true;
      }
  }
}

bool EmulateF64::run() {
  if (!hasWork())
    return false;
  for (Function &F : make_early_inc_range(M))
    if (!F.isDeclaration() && !F.hasFnAttribute(EmulationAttr))
      replaceOperations(F);
  doublesToBits();
  // Bodies for the emulations, now with i64 for double.
  for (Function &F : make_early_inc_range(M))
    if (F.isDeclaration() && F.hasFnAttribute(EmulationAttr))
      emulate(F, F.getFnAttribute(EmulationAttr).getValueAsString());
  if (NeedSoftF64)
    linkSoftF64();
  inlineEmulations();
  return true;
}

} // namespace

PreservedAnalyses AIREmulateF64Pass::run(Module &M, ModuleAnalysisManager &) {
  return EmulateF64(M).run() ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

bool AIREmulateF64Legacy::runOnModule(Module &M) {
  return EmulateF64(M).run();
}

char AIREmulateF64Legacy::ID = 0;

INITIALIZE_PASS(AIREmulateF64Legacy, DEBUG_TYPE,
                "AIR: double precision as soft-f64", false, false)

ModulePass *llvm::createAIREmulateF64LegacyPass() {
  return new AIREmulateF64Legacy();
}
