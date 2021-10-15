//===-- AMDGPUCodeGenPrepare.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass does misc. AMDGPU optimizations on IR before instruction
/// selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"

#define DEBUG_TYPE "amdgpu-codegenprepare"

using namespace llvm;

namespace {

static cl::opt<bool> WidenLoads(
  "amdgpu-codegenprepare-widen-constant-loads",
  cl::desc("Widen sub-dword constant address space loads in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(false));

static cl::opt<bool> Widen16BitOps(
  "amdgpu-codegenprepare-widen-16-bit-ops",
  cl::desc("Widen uniform 16-bit instructions to 32-bit in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(true));

static cl::opt<bool> UseMul24Intrin(
  "amdgpu-codegenprepare-mul24",
  cl::desc("Introduce mul24 intrinsics in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(true));

// Legalize 64-bit division by using the generic IR expansion.
static cl::opt<bool> ExpandDiv64InIR(
  "amdgpu-codegenprepare-expand-div64",
  cl::desc("Expand 64-bit division in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(false));

// Leave all division operations as they are. This supersedes ExpandDiv64InIR
// and is used for testing the legalizer.
static cl::opt<bool> DisableIDivExpand(
  "amdgpu-codegenprepare-disable-idiv-expansion",
  cl::desc("Prevent expanding integer division in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(false));

class AMDGPUCodeGenPrepare : public FunctionPass,
                             public InstVisitor<AMDGPUCodeGenPrepare, bool> {
  const GCNSubtarget *ST = nullptr;
  AssumptionCache *AC = nullptr;
  DominatorTree *DT = nullptr;
  LegacyDivergenceAnalysis *DA = nullptr;
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;
  bool HasUnsafeFPMath = false;
  bool HasFP32Denormals = false;

  /// Copies exact/nsw/nuw flags (if any) from binary operation \p I to
  /// binary operation \p V.
  ///
  /// \returns Binary operation \p V.
  /// \returns \p T's base element bit width.
  unsigned getBaseElementBitWidth(const Type *T) const;

  /// \returns Equivalent 32 bit integer type for given type \p T. For example,
  /// if \p T is i7, then i32 is returned; if \p T is <3 x i12>, then <3 x i32>
  /// is returned.
  Type *getI32Ty(IRBuilder<> &B, const Type *T) const;

  /// \returns True if binary operation \p I is a signed binary operation, false
  /// otherwise.
  bool isSigned(const BinaryOperator &I) const;

  /// \returns True if the condition of 'select' operation \p I comes from a
  /// signed 'icmp' operation, false otherwise.
  bool isSigned(const SelectInst &I) const;

  /// \returns True if type \p T needs to be promoted to 32 bit integer type,
  /// false otherwise.
  bool needsPromotionToI32(const Type *T) const;

  /// Promotes uniform binary operation \p I to equivalent 32 bit binary
  /// operation.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by sign or zero extending operands to
  /// 32 bits, replacing \p I with equivalent 32 bit binary operation, and
  /// truncating the result of 32 bit binary operation back to \p I's original
  /// type. Division operation is not promoted.
  ///
  /// \returns True if \p I is promoted to equivalent 32 bit binary operation,
  /// false otherwise.
  bool promoteUniformOpToI32(BinaryOperator &I) const;

  /// Promotes uniform 'icmp' operation \p I to 32 bit 'icmp' operation.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by sign or zero extending operands to
  /// 32 bits, and replacing \p I with 32 bit 'icmp' operation.
  ///
  /// \returns True.
  bool promoteUniformOpToI32(ICmpInst &I) const;

  /// Promotes uniform 'select' operation \p I to 32 bit 'select'
  /// operation.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by sign or zero extending operands to
  /// 32 bits, replacing \p I with 32 bit 'select' operation, and truncating the
  /// result of 32 bit 'select' operation back to \p I's original type.
  ///
  /// \returns True.
  bool promoteUniformOpToI32(SelectInst &I) const;

  /// Promotes uniform 'bitreverse' intrinsic \p I to 32 bit 'bitreverse'
  /// intrinsic.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by zero extending the operand to 32
  /// bits, replacing \p I with 32 bit 'bitreverse' intrinsic, shifting the
  /// result of 32 bit 'bitreverse' intrinsic to the right with zero fill (the
  /// shift amount is 32 minus \p I's base element bit width), and truncating
  /// the result of the shift operation back to \p I's original type.
  ///
  /// \returns True.
  bool promoteUniformBitreverseToI32(IntrinsicInst &I) const;


  unsigned numBitsUnsigned(Value *Op, unsigned ScalarSize) const;
  unsigned numBitsSigned(Value *Op, unsigned ScalarSize) const;

  /// Replace mul instructions with llvm.amdgcn.mul.u24 or llvm.amdgcn.mul.s24.
  /// SelectionDAG has an issue where an and asserting the bits are known
  bool replaceMulWithMul24(BinaryOperator &I) const;

  /// Perform same function as equivalently named function in DAGCombiner. Since
  /// we expand some divisions here, we need to perform this before obscuring.
  bool foldBinOpIntoSelect(BinaryOperator &I) const;

  bool divHasSpecialOptimization(BinaryOperator &I,
                                 Value *Num, Value *Den) const;
  int getDivNumBits(BinaryOperator &I,
                    Value *Num, Value *Den,
                    unsigned AtLeast, bool Signed) const;

  /// Expands 24 bit div or rem.
  Value* expandDivRem24(IRBuilder<> &Builder, BinaryOperator &I,
                        Value *Num, Value *Den,
                        bool IsDiv, bool IsSigned) const;

  Value *expandDivRem24Impl(IRBuilder<> &Builder, BinaryOperator &I,
                            Value *Num, Value *Den, unsigned NumBits,
                            bool IsDiv, bool IsSigned) const;

  /// Expands 32 bit div or rem.
  Value* expandDivRem32(IRBuilder<> &Builder, BinaryOperator &I,
                        Value *Num, Value *Den) const;

  Value *shrinkDivRem64(IRBuilder<> &Builder, BinaryOperator &I,
                        Value *Num, Value *Den) const;
  void expandDivRem64(BinaryOperator &I) const;

  /// Widen a scalar load.
  ///
  /// \details \p Widen scalar load for uniform, small type loads from constant
  //  memory / to a full 32-bits and then truncate the input to allow a scalar
  //  load instead of a vector load.
  //
  /// \returns True.

  bool canWidenScalarExtLoad(LoadInst &I) const;

public:
  static char ID;

  AMDGPUCodeGenPrepare() : FunctionPass(ID) {}

  bool visitFDiv(BinaryOperator &I);
  bool visitXor(BinaryOperator &I);

  bool visitInstruction(Instruction &I) { return false; }
  bool visitBinaryOperator(BinaryOperator &I);
  bool visitLoadInst(LoadInst &I);
  bool visitICmpInst(ICmpInst &I);
  bool visitSelectInst(SelectInst &I);

  bool visitIntrinsicInst(IntrinsicInst &I);
  bool visitBitreverseIntrinsicInst(IntrinsicInst &I);

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "AMDGPU IR optimizations"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<LegacyDivergenceAnalysis>();

    // FIXME: Division expansion needs to preserve the dominator tree.
    if (!ExpandDiv64InIR)
      AU.setPreservesAll();
 }
};

} // end anonymous namespace

unsigned AMDGPUCodeGenPrepare::getBaseElementBitWidth(const Type *T) const {
  assert(needsPromotionToI32(T) && "T does not need promotion to i32");

  if (T->isIntegerTy())
    return T->getIntegerBitWidth();
  return cast<VectorType>(T)->getElementType()->getIntegerBitWidth();
}

Type *AMDGPUCodeGenPrepare::getI32Ty(IRBuilder<> &B, const Type *T) const {
  assert(needsPromotionToI32(T) && "T does not need promotion to i32");

  if (T->isIntegerTy())
    return B.getInt32Ty();
  return FixedVectorType::get(B.getInt32Ty(), cast<FixedVectorType>(T));
}

bool AMDGPUCodeGenPrepare::isSigned(const BinaryOperator &I) const {
  return I.getOpcode() == Instruction::AShr ||
      I.getOpcode() == Instruction::SDiv || I.getOpcode() == Instruction::SRem;
}

bool AMDGPUCodeGenPrepare::isSigned(const SelectInst &I) const {
  return isa<ICmpInst>(I.getOperand(0)) ?
      cast<ICmpInst>(I.getOperand(0))->isSigned() : false;
}

bool AMDGPUCodeGenPrepare::needsPromotionToI32(const Type *T) const {
  if (!Widen16BitOps)
    return false;

  const IntegerType *IntTy = dyn_cast<IntegerType>(T);
  if (IntTy && IntTy->getBitWidth() > 1 && IntTy->getBitWidth() <= 16)
    return true;

  if (const VectorType *VT = dyn_cast<VectorType>(T)) {
    // TODO: The set of packed operations is more limited, so may want to
    // promote some anyway.
    if (ST->hasVOP3PInsts())
      return false;

    return needsPromotionToI32(VT->getElementType());
  }

  return false;
}

// Return true if the op promoted to i32 should have nsw set.
static bool promotedOpIsNSW(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::Shl:
  case Instruction::Add:
  case Instruction::Sub:
    return true;
  case Instruction::Mul:
    return I.hasNoUnsignedWrap();
  default:
    return false;
  }
}

// Return true if the op promoted to i32 should have nuw set.
static bool promotedOpIsNUW(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::Shl:
  case Instruction::Add:
  case Instruction::Mul:
    return true;
  case Instruction::Sub:
    return I.hasNoUnsignedWrap();
  default:
    return false;
  }
}

bool AMDGPUCodeGenPrepare::canWidenScalarExtLoad(LoadInst &I) const {
  Type *Ty = I.getType();
  const DataLayout &DL = Mod->getDataLayout();
  int TySize = DL.getTypeSizeInBits(Ty);
  Align Alignment = DL.getValueOrABITypeAlignment(I.getAlign(), Ty);

  return I.isSimple() && TySize < 32 && Alignment >= 4 && DA->isUniform(&I);
}

bool AMDGPUCodeGenPrepare::promoteUniformOpToI32(BinaryOperator &I) const {
  assert(needsPromotionToI32(I.getType()) &&
         "I does not need promotion to i32");

  if (I.getOpcode() == Instruction::SDiv ||
      I.getOpcode() == Instruction::UDiv ||
      I.getOpcode() == Instruction::SRem ||
      I.getOpcode() == Instruction::URem)
    return false;

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Value *ExtOp0 = nullptr;
  Value *ExtOp1 = nullptr;
  Value *ExtRes = nullptr;
  Value *TruncRes = nullptr;

  if (isSigned(I)) {
    ExtOp0 = Builder.CreateSExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
  } else {
    ExtOp0 = Builder.CreateZExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
  }

  ExtRes = Builder.CreateBinOp(I.getOpcode(), ExtOp0, ExtOp1);
  if (Instruction *Inst = dyn_cast<Instruction>(ExtRes)) {
    if (promotedOpIsNSW(cast<Instruction>(I)))
      Inst->setHasNoSignedWrap();

    if (promotedOpIsNUW(cast<Instruction>(I)))
      Inst->setHasNoUnsignedWrap();

    if (const auto *ExactOp = dyn_cast<PossiblyExactOperator>(&I))
      Inst->setIsExact(ExactOp->isExact());
  }

  TruncRes = Builder.CreateTrunc(ExtRes, I.getType());

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

bool AMDGPUCodeGenPrepare::promoteUniformOpToI32(ICmpInst &I) const {
  assert(needsPromotionToI32(I.getOperand(0)->getType()) &&
         "I does not need promotion to i32");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getOperand(0)->getType());
  Value *ExtOp0 = nullptr;
  Value *ExtOp1 = nullptr;
  Value *NewICmp  = nullptr;

  if (I.isSigned()) {
    ExtOp0 = Builder.CreateSExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
  } else {
    ExtOp0 = Builder.CreateZExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
  }
  NewICmp = Builder.CreateICmp(I.getPredicate(), ExtOp0, ExtOp1);

  I.replaceAllUsesWith(NewICmp);
  I.eraseFromParent();

  return true;
}

bool AMDGPUCodeGenPrepare::promoteUniformOpToI32(SelectInst &I) const {
  assert(needsPromotionToI32(I.getType()) &&
         "I does not need promotion to i32");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Value *ExtOp1 = nullptr;
  Value *ExtOp2 = nullptr;
  Value *ExtRes = nullptr;
  Value *TruncRes = nullptr;

  if (isSigned(I)) {
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
    ExtOp2 = Builder.CreateSExt(I.getOperand(2), I32Ty);
  } else {
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
    ExtOp2 = Builder.CreateZExt(I.getOperand(2), I32Ty);
  }
  ExtRes = Builder.CreateSelect(I.getOperand(0), ExtOp1, ExtOp2);
  TruncRes = Builder.CreateTrunc(ExtRes, I.getType());

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

bool AMDGPUCodeGenPrepare::promoteUniformBitreverseToI32(
    IntrinsicInst &I) const {
  assert(I.getIntrinsicID() == Intrinsic::bitreverse &&
         "I must be bitreverse intrinsic");
  assert(needsPromotionToI32(I.getType()) &&
         "I does not need promotion to i32");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Function *I32 =
      Intrinsic::getDeclaration(Mod, Intrinsic::bitreverse, { I32Ty });
  Value *ExtOp = Builder.CreateZExt(I.getOperand(0), I32Ty);
  Value *ExtRes = Builder.CreateCall(I32, { ExtOp });
  Value *LShrOp =
      Builder.CreateLShr(ExtRes, 32 - getBaseElementBitWidth(I.getType()));
  Value *TruncRes =
      Builder.CreateTrunc(LShrOp, I.getType());

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

unsigned AMDGPUCodeGenPrepare::numBitsUnsigned(Value *Op,
                                               unsigned ScalarSize) const {
  KnownBits Known = computeKnownBits(Op, *DL, 0, AC);
  return ScalarSize - Known.countMinLeadingZeros();
}

unsigned AMDGPUCodeGenPrepare::numBitsSigned(Value *Op,
                                             unsigned ScalarSize) const {
  // In order for this to be a signed 24-bit value, bit 23, must
  // be a sign bit.
  return ScalarSize - ComputeNumSignBits(Op, *DL, 0, AC);
}

static void extractValues(IRBuilder<> &Builder,
                          SmallVectorImpl<Value *> &Values, Value *V) {
  auto *VT = dyn_cast<FixedVectorType>(V->getType());
  if (!VT) {
    Values.push_back(V);
    return;
  }

  for (int I = 0, E = VT->getNumElements(); I != E; ++I)
    Values.push_back(Builder.CreateExtractElement(V, I));
}

static Value *insertValues(IRBuilder<> &Builder,
                           Type *Ty,
                           SmallVectorImpl<Value *> &Values) {
  if (Values.size() == 1)
    return Values[0];

  Value *NewVal = UndefValue::get(Ty);
  for (int I = 0, E = Values.size(); I != E; ++I)
    NewVal = Builder.CreateInsertElement(NewVal, Values[I], I);

  return NewVal;
}

bool AMDGPUCodeGenPrepare::replaceMulWithMul24(BinaryOperator &I) const {
  if (I.getOpcode() != Instruction::Mul)
    return false;

  Type *Ty = I.getType();
  unsigned Size = Ty->getScalarSizeInBits();
  if (Size <= 16 && ST->has16BitInsts())
    return false;

  // Prefer scalar if this could be s_mul_i32
  if (DA->isUniform(&I))
    return false;

  Value *LHS = I.getOperand(0);
  Value *RHS = I.getOperand(1);
  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Intrinsic::ID IntrID = Intrinsic::not_intrinsic;

  unsigned LHSBits = 0, RHSBits = 0;

  if (ST->hasMulU24() && (LHSBits = numBitsUnsigned(LHS, Size)) <= 24 &&
      (RHSBits = numBitsUnsigned(RHS, Size)) <= 24) {
    // The mul24 instruction yields the low-order 32 bits. If the original
    // result and the destination is wider than 32 bits, the mul24 would
    // truncate the result.
    if (Size > 32 && LHSBits + RHSBits > 32)
      return false;

    IntrID = Intrinsic::amdgcn_mul_u24;
  } else if (ST->hasMulI24() &&
             (LHSBits = numBitsSigned(LHS, Size)) < 24 &&
             (RHSBits = numBitsSigned(RHS, Size)) < 24) {
    // The original result is positive if its destination is wider than 32 bits
    // and its highest set bit is at bit 31. Generating mul24 and sign-extending
    // it would yield a negative value.
    if (Size > 32 && LHSBits + RHSBits > 30)
      return false;

    IntrID = Intrinsic::amdgcn_mul_i24;
  } else
    return false;

  SmallVector<Value *, 4> LHSVals;
  SmallVector<Value *, 4> RHSVals;
  SmallVector<Value *, 4> ResultVals;
  extractValues(Builder, LHSVals, LHS);
  extractValues(Builder, RHSVals, RHS);


  IntegerType *I32Ty = Builder.getInt32Ty();
  FunctionCallee Intrin = Intrinsic::getDeclaration(Mod, IntrID);
  for (int I = 0, E = LHSVals.size(); I != E; ++I) {
    Value *LHS, *RHS;
    if (IntrID == Intrinsic::amdgcn_mul_u24) {
      LHS = Builder.CreateZExtOrTrunc(LHSVals[I], I32Ty);
      RHS = Builder.CreateZExtOrTrunc(RHSVals[I], I32Ty);
    } else {
      LHS = Builder.CreateSExtOrTrunc(LHSVals[I], I32Ty);
      RHS = Builder.CreateSExtOrTrunc(RHSVals[I], I32Ty);
    }

    Value *Result = Builder.CreateCall(Intrin, {LHS, RHS});

    if (IntrID == Intrinsic::amdgcn_mul_u24) {
      ResultVals.push_back(Builder.CreateZExtOrTrunc(Result,
                                                     LHSVals[I]->getType()));
    } else {
      ResultVals.push_back(Builder.CreateSExtOrTrunc(Result,
                                                     LHSVals[I]->getType()));
    }
  }

  Value *NewVal = insertValues(Builder, Ty, ResultVals);
  NewVal->takeName(&I);
  I.replaceAllUsesWith(NewVal);
  I.eraseFromParent();

  return true;
}

// Find a select instruction, which may have been casted. This is mostly to deal
// with cases where i16 selects were promoted here to i32.
static SelectInst *findSelectThroughCast(Value *V, CastInst *&Cast) {
  Cast = nullptr;
  if (SelectInst *Sel = dyn_cast<SelectInst>(V))
    return Sel;

  if ((Cast = dyn_cast<CastInst>(V))) {
    if (SelectInst *Sel = dyn_cast<SelectInst>(Cast->getOperand(0)))
      return Sel;
  }

  return nullptr;
}

bool AMDGPUCodeGenPrepare::foldBinOpIntoSelect(BinaryOperator &BO) const {
  // Don't do this unless the old select is going away. We want to eliminate the
  // binary operator, not replace a binop with a select.
  int SelOpNo = 0;

  CastInst *CastOp;

  // TODO: Should probably try to handle some cases with multiple
  // users. Duplicating the select may be profitable for division.
  SelectInst *Sel = findSelectThroughCast(BO.getOperand(0), CastOp);
  if (!Sel || !Sel->hasOneUse()) {
    SelOpNo = 1;
    Sel = findSelectThroughCast(BO.getOperand(1), CastOp);
  }

  if (!Sel || !Sel->hasOneUse())
    return false;

  Constant *CT = dyn_cast<Constant>(Sel->getTrueValue());
  Constant *CF = dyn_cast<Constant>(Sel->getFalseValue());
  Constant *CBO = dyn_cast<Constant>(BO.getOperand(SelOpNo ^ 1));
  if (!CBO || !CT || !CF)
    return false;

  if (CastOp) {
    if (!CastOp->hasOneUse())
      return false;
    CT = ConstantFoldCastOperand(CastOp->getOpcode(), CT, BO.getType(), *DL);
    CF = ConstantFoldCastOperand(CastOp->getOpcode(), CF, BO.getType(), *DL);
  }

  // TODO: Handle special 0/-1 cases DAG combine does, although we only really
  // need to handle divisions here.
  Constant *FoldedT = SelOpNo ?
    ConstantFoldBinaryOpOperands(BO.getOpcode(), CBO, CT, *DL) :
    ConstantFoldBinaryOpOperands(BO.getOpcode(), CT, CBO, *DL);
  if (isa<ConstantExpr>(FoldedT))
    return false;

  Constant *FoldedF = SelOpNo ?
    ConstantFoldBinaryOpOperands(BO.getOpcode(), CBO, CF, *DL) :
    ConstantFoldBinaryOpOperands(BO.getOpcode(), CF, CBO, *DL);
  if (isa<ConstantExpr>(FoldedF))
    return false;

  IRBuilder<> Builder(&BO);
  Builder.SetCurrentDebugLocation(BO.getDebugLoc());
  if (const FPMathOperator *FPOp = dyn_cast<const FPMathOperator>(&BO))
    Builder.setFastMathFlags(FPOp->getFastMathFlags());

  Value *NewSelect = Builder.CreateSelect(Sel->getCondition(),
                                          FoldedT, FoldedF);
  NewSelect->takeName(&BO);
  BO.replaceAllUsesWith(NewSelect);
  BO.eraseFromParent();
  if (CastOp)
    CastOp->eraseFromParent();
  Sel->eraseFromParent();
  return true;
}

// Optimize fdiv with rcp:
//
// 1/x -> rcp(x) when rcp is sufficiently accurate or inaccurate rcp is
//               allowed with unsafe-fp-math or afn.
//
// a/b -> a*rcp(b) when inaccurate rcp is allowed with unsafe-fp-math or afn.
static Value *optimizeWithRcp(Value *Num, Value *Den, bool AllowInaccurateRcp,
                              bool RcpIsAccurate, IRBuilder<> &Builder,
                              Module *Mod) {

  if (!AllowInaccurateRcp && !RcpIsAccurate)
    return nullptr;

  Type *Ty = Den->getType();
  if (const ConstantFP *CLHS = dyn_cast<ConstantFP>(Num)) {
    if (AllowInaccurateRcp || RcpIsAccurate) {
      if (CLHS->isExactlyValue(1.0)) {
        Function *Decl = Intrinsic::getDeclaration(
          Mod, Intrinsic::amdgcn_rcp, Ty);

        // v_rcp_f32 and v_rsq_f32 do not support denormals, and according to
        // the CI documentation has a worst case error of 1 ulp.
        // OpenCL requires <= 2.5 ulp for 1.0 / x, so it should always be OK to
        // use it as long as we aren't trying to use denormals.
        //
        // v_rcp_f16 and v_rsq_f16 DO support denormals.

        // NOTE: v_sqrt and v_rcp will be combined to v_rsq later. So we don't
        //       insert rsq intrinsic here.

        // 1.0 / x -> rcp(x)
        return Builder.CreateCall(Decl, { Den });
      }

       // Same as for 1.0, but expand the sign out of the constant.
      if (CLHS->isExactlyValue(-1.0)) {
        Function *Decl = Intrinsic::getDeclaration(
          Mod, Intrinsic::amdgcn_rcp, Ty);

         // -1.0 / x -> rcp (fneg x)
         Value *FNeg = Builder.CreateFNeg(Den);
         return Builder.CreateCall(Decl, { FNeg });
       }
    }
  }

  if (AllowInaccurateRcp) {
    Function *Decl = Intrinsic::getDeclaration(
      Mod, Intrinsic::amdgcn_rcp, Ty);

    // Turn into multiply by the reciprocal.
    // x / y -> x * (1.0 / y)
    Value *Recip = Builder.CreateCall(Decl, { Den });
    return Builder.CreateFMul(Num, Recip);
  }
  return nullptr;
}

// optimize with fdiv.fast:
//
// a/b -> fdiv.fast(a, b) when !fpmath >= 2.5ulp with denormals flushed.
//
// 1/x -> fdiv.fast(1,x)  when !fpmath >= 2.5ulp.
//
// NOTE: optimizeWithRcp should be tried first because rcp is the preference.
static Value *optimizeWithFDivFast(Value *Num, Value *Den, float ReqdAccuracy,
                                   bool HasDenormals, IRBuilder<> &Builder,
                                   Module *Mod) {
  // fdiv.fast can achieve 2.5 ULP accuracy.
  if (ReqdAccuracy < 2.5f)
    return nullptr;

  // Only have fdiv.fast for f32.
  Type *Ty = Den->getType();
  if (!Ty->isFloatTy())
    return nullptr;

  bool NumIsOne = false;
  if (const ConstantFP *CNum = dyn_cast<ConstantFP>(Num)) {
    if (CNum->isExactlyValue(+1.0) || CNum->isExactlyValue(-1.0))
      NumIsOne = true;
  }

  // fdiv does not support denormals. But 1.0/x is always fine to use it.
  if (HasDenormals && !NumIsOne)
    return nullptr;

  Function *Decl = Intrinsic::getDeclaration(Mod, Intrinsic::amdgcn_fdiv_fast);
  return Builder.CreateCall(Decl, { Num, Den });
}

// Optimizations is performed based on fpmath, fast math flags as well as
// denormals to optimize fdiv with either rcp or fdiv.fast.
//
// With rcp:
//   1/x -> rcp(x) when rcp is sufficiently accurate or inaccurate rcp is
//                 allowed with unsafe-fp-math or afn.
//
//   a/b -> a*rcp(b) when inaccurate rcp is allowed with unsafe-fp-math or afn.
//
// With fdiv.fast:
//   a/b -> fdiv.fast(a, b) when !fpmath >= 2.5ulp with denormals flushed.
//
//   1/x -> fdiv.fast(1,x)  when !fpmath >= 2.5ulp.
//
// NOTE: rcp is the preference in cases that both are legal.
bool AMDGPUCodeGenPrepare::visitFDiv(BinaryOperator &FDiv) {

  Type *Ty = FDiv.getType()->getScalarType();

  // The f64 rcp/rsq approximations are pretty inaccurate. We can do an
  // expansion around them in codegen.
  if (Ty->isDoubleTy())
    return false;

  // No intrinsic for fdiv16 if target does not support f16.
  if (Ty->isHalfTy() && !ST->has16BitInsts())
    return false;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&FDiv);
  const float ReqdAccuracy =  FPOp->getFPAccuracy();

  // Inaccurate rcp is allowed with unsafe-fp-math or afn.
  FastMathFlags FMF = FPOp->getFastMathFlags();
  const bool AllowInaccurateRcp = HasUnsafeFPMath || FMF.approxFunc();

  // rcp_f16 is accurate for !fpmath >= 1.0ulp.
  // rcp_f32 is accurate for !fpmath >= 1.0ulp and denormals are flushed.
  // rcp_f64 is never accurate.
  const bool RcpIsAccurate = (Ty->isHalfTy() && ReqdAccuracy >= 1.0f) ||
            (Ty->isFloatTy() && !HasFP32Denormals && ReqdAccuracy >= 1.0f);

  IRBuilder<> Builder(FDiv.getParent(), std::next(FDiv.getIterator()));
  Builder.setFastMathFlags(FMF);
  Builder.SetCurrentDebugLocation(FDiv.getDebugLoc());

  Value *Num = FDiv.getOperand(0);
  Value *Den = FDiv.getOperand(1);

  Value *NewFDiv = nullptr;
  if (auto *VT = dyn_cast<FixedVectorType>(FDiv.getType())) {
    NewFDiv = UndefValue::get(VT);

    // FIXME: Doesn't do the right thing for cases where the vector is partially
    // constant. This works when the scalarizer pass is run first.
    for (unsigned I = 0, E = VT->getNumElements(); I != E; ++I) {
      Value *NumEltI = Builder.CreateExtractElement(Num, I);
      Value *DenEltI = Builder.CreateExtractElement(Den, I);
      // Try rcp first.
      Value *NewElt = optimizeWithRcp(NumEltI, DenEltI, AllowInaccurateRcp,
                                      RcpIsAccurate, Builder, Mod);
      if (!NewElt) // Try fdiv.fast.
        NewElt = optimizeWithFDivFast(NumEltI, DenEltI, ReqdAccuracy,
                                      HasFP32Denormals, Builder, Mod);
      if (!NewElt) // Keep the original.
        NewElt = Builder.CreateFDiv(NumEltI, DenEltI);

      NewFDiv = Builder.CreateInsertElement(NewFDiv, NewElt, I);
    }
  } else { // Scalar FDiv.
    // Try rcp first.
    NewFDiv = optimizeWithRcp(Num, Den, AllowInaccurateRcp, RcpIsAccurate,
                              Builder, Mod);
    if (!NewFDiv) { // Try fdiv.fast.
      NewFDiv = optimizeWithFDivFast(Num, Den, ReqdAccuracy, HasFP32Denormals,
                                     Builder, Mod);
    }
  }

  if (NewFDiv) {
    FDiv.replaceAllUsesWith(NewFDiv);
    NewFDiv->takeName(&FDiv);
    FDiv.eraseFromParent();
  }

  return !!NewFDiv;
}

bool AMDGPUCodeGenPrepare::visitXor(BinaryOperator &I) {
  // Match the Xor instruction, its type and its operands
  IntrinsicInst *IntrinsicCall = dyn_cast<IntrinsicInst>(I.getOperand(0));
  ConstantInt *RHS = dyn_cast<ConstantInt>(I.getOperand(1));
  if (!RHS || !IntrinsicCall || RHS->getSExtValue() != -1)
    return visitBinaryOperator(I);

  // Check if the Call is an intrinsic instruction to amdgcn_class intrinsic
  // has only one use
  if (IntrinsicCall->getIntrinsicID() != Intrinsic::amdgcn_class ||
      !IntrinsicCall->hasOneUse())
    return visitBinaryOperator(I);

  // "Not" the second argument of the intrinsic call
  ConstantInt *Arg = dyn_cast<ConstantInt>(IntrinsicCall->getOperand(1));
  if (!Arg)
    return visitBinaryOperator(I);

  IntrinsicCall->setOperand(
      1, ConstantInt::get(Arg->getType(), Arg->getZExtValue() ^ 0x3ff));
  I.replaceAllUsesWith(IntrinsicCall);
  I.eraseFromParent();
  return true;
}

static bool hasUnsafeFPMath(const Function &F) {
  Attribute Attr = F.getFnAttribute("unsafe-fp-math");
  return Attr.getValueAsBool();
}

static std::pair<Value*, Value*> getMul64(IRBuilder<> &Builder,
                                          Value *LHS, Value *RHS) {
  Type *I32Ty = Builder.getInt32Ty();
  Type *I64Ty = Builder.getInt64Ty();

  Value *LHS_EXT64 = Builder.CreateZExt(LHS, I64Ty);
  Value *RHS_EXT64 = Builder.CreateZExt(RHS, I64Ty);
  Value *MUL64 = Builder.CreateMul(LHS_EXT64, RHS_EXT64);
  Value *Lo = Builder.CreateTrunc(MUL64, I32Ty);
  Value *Hi = Builder.CreateLShr(MUL64, Builder.getInt64(32));
  Hi = Builder.CreateTrunc(Hi, I32Ty);
  return std::make_pair(Lo, Hi);
}

static Value* getMulHu(IRBuilder<> &Builder, Value *LHS, Value *RHS) {
  return getMul64(Builder, LHS, RHS).second;
}

/// Figure out how many bits are really needed for this ddivision. \p AtLeast is
/// an optimization hint to bypass the second ComputeNumSignBits call if we the
/// first one is insufficient. Returns -1 on failure.
int AMDGPUCodeGenPrepare::getDivNumBits(BinaryOperator &I,
                                        Value *Num, Value *Den,
                                        unsigned AtLeast, bool IsSigned) const {
  const DataLayout &DL = Mod->getDataLayout();
  unsigned LHSSignBits = ComputeNumSignBits(Num, DL, 0, AC, &I);
  if (LHSSignBits < AtLeast)
    return -1;

  unsigned RHSSignBits = ComputeNumSignBits(Den, DL, 0, AC, &I);
  if (RHSSignBits < AtLeast)
    return -1;

  unsigned SignBits = std::min(LHSSignBits, RHSSignBits);
  unsigned DivBits = Num->getType()->getScalarSizeInBits() - SignBits;
  if (IsSigned)
    ++DivBits;
  return DivBits;
}

// The fractional part of a float is enough to accurately represent up to
// a 24-bit signed integer.
Value *AMDGPUCodeGenPrepare::expandDivRem24(IRBuilder<> &Builder,
                                            BinaryOperator &I,
                                            Value *Num, Value *Den,
                                            bool IsDiv, bool IsSigned) const {
  int DivBits = getDivNumBits(I, Num, Den, 9, IsSigned);
  if (DivBits == -1)
    return nullptr;
  return expandDivRem24Impl(Builder, I, Num, Den, DivBits, IsDiv, IsSigned);
}

Value *AMDGPUCodeGenPrepare::expandDivRem24Impl(IRBuilder<> &Builder,
                                                BinaryOperator &I,
                                                Value *Num, Value *Den,
                                                unsigned DivBits,
                                                bool IsDiv, bool IsSigned) const {
  Type *I32Ty = Builder.getInt32Ty();
  Num = Builder.CreateTrunc(Num, I32Ty);
  Den = Builder.CreateTrunc(Den, I32Ty);

  Type *F32Ty = Builder.getFloatTy();
  ConstantInt *One = Builder.getInt32(1);
  Value *JQ = One;

  if (IsSigned) {
    // char|short jq = ia ^ ib;
    JQ = Builder.CreateXor(Num, Den);

    // jq = jq >> (bitsize - 2)
    JQ = Builder.CreateAShr(JQ, Builder.getInt32(30));

    // jq = jq | 0x1
    JQ = Builder.CreateOr(JQ, One);
  }

  // int ia = (int)LHS;
  Value *IA = Num;

  // int ib, (int)RHS;
  Value *IB = Den;

  // float fa = (float)ia;
  Value *FA = IsSigned ? Builder.CreateSIToFP(IA, F32Ty)
                       : Builder.CreateUIToFP(IA, F32Ty);

  // float fb = (float)ib;
  Value *FB = IsSigned ? Builder.CreateSIToFP(IB,F32Ty)
                       : Builder.CreateUIToFP(IB,F32Ty);

  Function *RcpDecl = Intrinsic::getDeclaration(Mod, Intrinsic::amdgcn_rcp,
                                                Builder.getFloatTy());
  Value *RCP = Builder.CreateCall(RcpDecl, { FB });
  Value *FQM = Builder.CreateFMul(FA, RCP);

  // fq = trunc(fqm);
  CallInst *FQ = Builder.CreateUnaryIntrinsic(Intrinsic::trunc, FQM);
  FQ->copyFastMathFlags(Builder.getFastMathFlags());

  // float fqneg = -fq;
  Value *FQNeg = Builder.CreateFNeg(FQ);

  // float fr = mad(fqneg, fb, fa);
  auto FMAD = !ST->hasMadMacF32Insts()
                  ? Intrinsic::fma
                  : (Intrinsic::ID)Intrinsic::amdgcn_fmad_ftz;
  Value *FR = Builder.CreateIntrinsic(FMAD,
                                      {FQNeg->getType()}, {FQNeg, FB, FA}, FQ);

  // int iq = (int)fq;
  Value *IQ = IsSigned ? Builder.CreateFPToSI(FQ, I32Ty)
                       : Builder.CreateFPToUI(FQ, I32Ty);

  // fr = fabs(fr);
  FR = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, FR, FQ);

  // fb = fabs(fb);
  FB = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, FB, FQ);

  // int cv = fr >= fb;
  Value *CV = Builder.CreateFCmpOGE(FR, FB);

  // jq = (cv ? jq : 0);
  JQ = Builder.CreateSelect(CV, JQ, Builder.getInt32(0));

  // dst = iq + jq;
  Value *Div = Builder.CreateAdd(IQ, JQ);

  Value *Res = Div;
  if (!IsDiv) {
    // Rem needs compensation, it's easier to recompute it
    Value *Rem = Builder.CreateMul(Div, Den);
    Res = Builder.CreateSub(Num, Rem);
  }

  if (DivBits != 0 && DivBits < 32) {
    // Extend in register from the number of bits this divide really is.
    if (IsSigned) {
      int InRegBits = 32 - DivBits;

      Res = Builder.CreateShl(Res, InRegBits);
      Res = Builder.CreateAShr(Res, InRegBits);
    } else {
      ConstantInt *TruncMask
        = Builder.getInt32((UINT64_C(1) << DivBits) - 1);
      Res = Builder.CreateAnd(Res, TruncMask);
    }
  }

  return Res;
}

// Try to recognize special cases the DAG will emit special, better expansions
// than the general expansion we do here.

// TODO: It would be better to just directly handle those optimizations here.
bool AMDGPUCodeGenPrepare::divHasSpecialOptimization(
  BinaryOperator &I, Value *Num, Value *Den) const {
  if (Constant *C = dyn_cast<Constant>(Den)) {
    // Arbitrary constants get a better expansion as long as a wider mulhi is
    // legal.
    if (C->getType()->getScalarSizeInBits() <= 32)
      return true;

    // TODO: Sdiv check for not exact for some reason.

    // If there's no wider mulhi, there's only a better expansion for powers of
    // two.
    // TODO: Should really know for each vector element.
    if (isKnownToBeAPowerOfTwo(C, *DL, true, 0, AC, &I, DT))
      return true;

    return false;
  }

  if (BinaryOperator *BinOpDen = dyn_cast<BinaryOperator>(Den)) {
    // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
    if (BinOpDen->getOpcode() == Instruction::Shl &&
        isa<Constant>(BinOpDen->getOperand(0)) &&
        isKnownToBeAPowerOfTwo(BinOpDen->getOperand(0), *DL, true,
                               0, AC, &I, DT)) {
      return true;
    }
  }

  return false;
}

static Value *getSign32(Value *V, IRBuilder<> &Builder, const DataLayout *DL) {
  // Check whether the sign can be determined statically.
  KnownBits Known = computeKnownBits(V, *DL);
  if (Known.isNegative())
    return Constant::getAllOnesValue(V->getType());
  if (Known.isNonNegative())
    return Constant::getNullValue(V->getType());
  return Builder.CreateAShr(V, Builder.getInt32(31));
}

Value *AMDGPUCodeGenPrepare::expandDivRem32(IRBuilder<> &Builder,
                                            BinaryOperator &I, Value *X,
                                            Value *Y) const {
  Instruction::BinaryOps Opc = I.getOpcode();
  assert(Opc == Instruction::URem || Opc == Instruction::UDiv ||
         Opc == Instruction::SRem || Opc == Instruction::SDiv);

  FastMathFlags FMF;
  FMF.setFast();
  Builder.setFastMathFlags(FMF);

  if (divHasSpecialOptimization(I, X, Y))
    return nullptr;  // Keep it for later optimization.

  bool IsDiv = Opc == Instruction::UDiv || Opc == Instruction::SDiv;
  bool IsSigned = Opc == Instruction::SRem || Opc == Instruction::SDiv;

  Type *Ty = X->getType();
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  if (Ty->getScalarSizeInBits() < 32) {
    if (IsSigned) {
      X = Builder.CreateSExt(X, I32Ty);
      Y = Builder.CreateSExt(Y, I32Ty);
    } else {
      X = Builder.CreateZExt(X, I32Ty);
      Y = Builder.CreateZExt(Y, I32Ty);
    }
  }

  if (Value *Res = expandDivRem24(Builder, I, X, Y, IsDiv, IsSigned)) {
    return IsSigned ? Builder.CreateSExtOrTrunc(Res, Ty) :
                      Builder.CreateZExtOrTrunc(Res, Ty);
  }

  ConstantInt *Zero = Builder.getInt32(0);
  ConstantInt *One = Builder.getInt32(1);

  Value *Sign = nullptr;
  if (IsSigned) {
    Value *SignX = getSign32(X, Builder, DL);
    Value *SignY = getSign32(Y, Builder, DL);
    // Remainder sign is the same as LHS
    Sign = IsDiv ? Builder.CreateXor(SignX, SignY) : SignX;

    X = Builder.CreateAdd(X, SignX);
    Y = Builder.CreateAdd(Y, SignY);

    X = Builder.CreateXor(X, SignX);
    Y = Builder.CreateXor(Y, SignY);
  }

  // The algorithm here is based on ideas from "Software Integer Division", Tom
  // Rodeheffer, August 2008.
  //
  // unsigned udiv(unsigned x, unsigned y) {
  //   // Initial estimate of inv(y). The constant is less than 2^32 to ensure
  //   // that this is a lower bound on inv(y), even if some of the calculations
  //   // round up.
  //   unsigned z = (unsigned)((4294967296.0 - 512.0) * v_rcp_f32((float)y));
  //
  //   // One round of UNR (Unsigned integer Newton-Raphson) to improve z.
  //   // Empirically this is guaranteed to give a "two-y" lower bound on
  //   // inv(y).
  //   z += umulh(z, -y * z);
  //
  //   // Quotient/remainder estimate.
  //   unsigned q = umulh(x, z);
  //   unsigned r = x - q * y;
  //
  //   // Two rounds of quotient/remainder refinement.
  //   if (r >= y) {
  //     ++q;
  //     r -= y;
  //   }
  //   if (r >= y) {
  //     ++q;
  //     r -= y;
  //   }
  //
  //   return q;
  // }

  // Initial estimate of inv(y).
  Value *FloatY = Builder.CreateUIToFP(Y, F32Ty);
  Function *Rcp = Intrinsic::getDeclaration(Mod, Intrinsic::amdgcn_rcp, F32Ty);
  Value *RcpY = Builder.CreateCall(Rcp, {FloatY});
  Constant *Scale = ConstantFP::get(F32Ty, BitsToFloat(0x4F7FFFFE));
  Value *ScaledY = Builder.CreateFMul(RcpY, Scale);
  Value *Z = Builder.CreateFPToUI(ScaledY, I32Ty);

  // One round of UNR.
  Value *NegY = Builder.CreateSub(Zero, Y);
  Value *NegYZ = Builder.CreateMul(NegY, Z);
  Z = Builder.CreateAdd(Z, getMulHu(Builder, Z, NegYZ));

  // Quotient/remainder estimate.
  Value *Q = getMulHu(Builder, X, Z);
  Value *R = Builder.CreateSub(X, Builder.CreateMul(Q, Y));

  // First quotient/remainder refinement.
  Value *Cond = Builder.CreateICmpUGE(R, Y);
  if (IsDiv)
    Q = Builder.CreateSelect(Cond, Builder.CreateAdd(Q, One), Q);
  R = Builder.CreateSelect(Cond, Builder.CreateSub(R, Y), R);

  // Second quotient/remainder refinement.
  Cond = Builder.CreateICmpUGE(R, Y);
  Value *Res;
  if (IsDiv)
    Res = Builder.CreateSelect(Cond, Builder.CreateAdd(Q, One), Q);
  else
    Res = Builder.CreateSelect(Cond, Builder.CreateSub(R, Y), R);

  if (IsSigned) {
    Res = Builder.CreateXor(Res, Sign);
    Res = Builder.CreateSub(Res, Sign);
  }

  Res = Builder.CreateTrunc(Res, Ty);

  return Res;
}

Value *AMDGPUCodeGenPrepare::shrinkDivRem64(IRBuilder<> &Builder,
                                            BinaryOperator &I,
                                            Value *Num, Value *Den) const {
  if (!ExpandDiv64InIR && divHasSpecialOptimization(I, Num, Den))
    return nullptr;  // Keep it for later optimization.

  Instruction::BinaryOps Opc = I.getOpcode();

  bool IsDiv = Opc == Instruction::SDiv || Opc == Instruction::UDiv;
  bool IsSigned = Opc == Instruction::SDiv || Opc == Instruction::SRem;

  int NumDivBits = getDivNumBits(I, Num, Den, 32, IsSigned);
  if (NumDivBits == -1)
    return nullptr;

  Value *Narrowed = nullptr;
  if (NumDivBits <= 24) {
    Narrowed = expandDivRem24Impl(Builder, I, Num, Den, NumDivBits,
                                  IsDiv, IsSigned);
  } else if (NumDivBits <= 32) {
    Narrowed = expandDivRem32(Builder, I, Num, Den);
  }

  if (Narrowed) {
    return IsSigned ? Builder.CreateSExt(Narrowed, Num->getType()) :
                      Builder.CreateZExt(Narrowed, Num->getType());
  }

  return nullptr;
}

void AMDGPUCodeGenPrepare::expandDivRem64(BinaryOperator &I) const {
  Instruction::BinaryOps Opc = I.getOpcode();
  // Do the general expansion.
  if (Opc == Instruction::UDiv || Opc == Instruction::SDiv) {
    expandDivisionUpTo64Bits(&I);
    return;
  }

  if (Opc == Instruction::URem || Opc == Instruction::SRem) {
    expandRemainderUpTo64Bits(&I);
    return;
  }

  llvm_unreachable("not a division");
}

bool AMDGPUCodeGenPrepare::visitBinaryOperator(BinaryOperator &I) {
  if (foldBinOpIntoSelect(I))
    return true;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getType()) &&
      DA->isUniform(&I) && promoteUniformOpToI32(I))
    return true;

  if (UseMul24Intrin && replaceMulWithMul24(I))
    return true;

  bool Changed = false;
  Instruction::BinaryOps Opc = I.getOpcode();
  Type *Ty = I.getType();
  Value *NewDiv = nullptr;
  unsigned ScalarSize = Ty->getScalarSizeInBits();

  SmallVector<BinaryOperator *, 8> Div64ToExpand;

  if ((Opc == Instruction::URem || Opc == Instruction::UDiv ||
       Opc == Instruction::SRem || Opc == Instruction::SDiv) &&
      ScalarSize <= 64 &&
      !DisableIDivExpand) {
    Value *Num = I.getOperand(0);
    Value *Den = I.getOperand(1);
    IRBuilder<> Builder(&I);
    Builder.SetCurrentDebugLocation(I.getDebugLoc());

    if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
      NewDiv = UndefValue::get(VT);

      for (unsigned N = 0, E = VT->getNumElements(); N != E; ++N) {
        Value *NumEltN = Builder.CreateExtractElement(Num, N);
        Value *DenEltN = Builder.CreateExtractElement(Den, N);

        Value *NewElt;
        if (ScalarSize <= 32) {
          NewElt = expandDivRem32(Builder, I, NumEltN, DenEltN);
          if (!NewElt)
            NewElt = Builder.CreateBinOp(Opc, NumEltN, DenEltN);
        } else {
          // See if this 64-bit division can be shrunk to 32/24-bits before
          // producing the general expansion.
          NewElt = shrinkDivRem64(Builder, I, NumEltN, DenEltN);
          if (!NewElt) {
            // The general 64-bit expansion introduces control flow and doesn't
            // return the new value. Just insert a scalar copy and defer
            // expanding it.
            NewElt = Builder.CreateBinOp(Opc, NumEltN, DenEltN);
            Div64ToExpand.push_back(cast<BinaryOperator>(NewElt));
          }
        }

        NewDiv = Builder.CreateInsertElement(NewDiv, NewElt, N);
      }
    } else {
      if (ScalarSize <= 32)
        NewDiv = expandDivRem32(Builder, I, Num, Den);
      else {
        NewDiv = shrinkDivRem64(Builder, I, Num, Den);
        if (!NewDiv)
          Div64ToExpand.push_back(&I);
      }
    }

    if (NewDiv) {
      I.replaceAllUsesWith(NewDiv);
      I.eraseFromParent();
      Changed = true;
    }
  }

  if (ExpandDiv64InIR) {
    // TODO: We get much worse code in specially handled constant cases.
    for (BinaryOperator *Div : Div64ToExpand) {
      expandDivRem64(*Div);
      Changed = true;
    }
  }

  return Changed;
}

bool AMDGPUCodeGenPrepare::visitLoadInst(LoadInst &I) {
  if (!WidenLoads)
    return false;

  if ((I.getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS ||
       I.getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT) &&
      canWidenScalarExtLoad(I)) {
    IRBuilder<> Builder(&I);
    Builder.SetCurrentDebugLocation(I.getDebugLoc());

    Type *I32Ty = Builder.getInt32Ty();
    Type *PT = PointerType::get(I32Ty, I.getPointerAddressSpace());
    Value *BitCast= Builder.CreateBitCast(I.getPointerOperand(), PT);
    LoadInst *WidenLoad = Builder.CreateLoad(I32Ty, BitCast);
    WidenLoad->copyMetadata(I);

    // If we have range metadata, we need to convert the type, and not make
    // assumptions about the high bits.
    if (auto *Range = WidenLoad->getMetadata(LLVMContext::MD_range)) {
      ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Range->getOperand(0));

      if (Lower->isNullValue()) {
        WidenLoad->setMetadata(LLVMContext::MD_range, nullptr);
      } else {
        Metadata *LowAndHigh[] = {
          ConstantAsMetadata::get(ConstantInt::get(I32Ty, Lower->getValue().zext(32))),
          // Don't make assumptions about the high bits.
          ConstantAsMetadata::get(ConstantInt::get(I32Ty, 0))
        };

        WidenLoad->setMetadata(LLVMContext::MD_range,
                               MDNode::get(Mod->getContext(), LowAndHigh));
      }
    }

    int TySize = Mod->getDataLayout().getTypeSizeInBits(I.getType());
    Type *IntNTy = Builder.getIntNTy(TySize);
    Value *ValTrunc = Builder.CreateTrunc(WidenLoad, IntNTy);
    Value *ValOrig = Builder.CreateBitCast(ValTrunc, I.getType());
    I.replaceAllUsesWith(ValOrig);
    I.eraseFromParent();
    return true;
  }

  return false;
}

bool AMDGPUCodeGenPrepare::visitICmpInst(ICmpInst &I) {
  bool Changed = false;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getOperand(0)->getType()) &&
      DA->isUniform(&I))
    Changed |= promoteUniformOpToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::visitSelectInst(SelectInst &I) {
  bool Changed = false;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getType()) &&
      DA->isUniform(&I))
    Changed |= promoteUniformOpToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::visitIntrinsicInst(IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
  case Intrinsic::bitreverse:
    return visitBitreverseIntrinsicInst(I);
  default:
    return false;
  }
}

bool AMDGPUCodeGenPrepare::visitBitreverseIntrinsicInst(IntrinsicInst &I) {
  bool Changed = false;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getType()) &&
      DA->isUniform(&I))
    Changed |= promoteUniformBitreverseToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::doInitialization(Module &M) {
  Mod = &M;
  DL = &Mod->getDataLayout();
  return false;
}

bool AMDGPUCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  const AMDGPUTargetMachine &TM = TPC->getTM<AMDGPUTargetMachine>();
  ST = &TM.getSubtarget<GCNSubtarget>(F);
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DA = &getAnalysis<LegacyDivergenceAnalysis>();

  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTWP ? &DTWP->getDomTree() : nullptr;

  HasUnsafeFPMath = hasUnsafeFPMath(F);

  AMDGPU::SIModeRegisterDefaults Mode(F);
  HasFP32Denormals = Mode.allFP32Denormals();

  bool MadeChange = false;

  Function::iterator NextBB;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; FI = NextBB) {
    BasicBlock *BB = &*FI;
    NextBB = std::next(FI);

    BasicBlock::iterator Next;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; I = Next) {
      Next = std::next(I);

      MadeChange |= visit(*I);

      if (Next != E) { // Control flow changed
        BasicBlock *NextInstBB = Next->getParent();
        if (NextInstBB != BB) {
          BB = NextInstBB;
          E = BB->end();
          FE = F.end();
        }
      }
    }
  }

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(AMDGPUCodeGenPrepare, DEBUG_TYPE,
                      "AMDGPU IR optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_END(AMDGPUCodeGenPrepare, DEBUG_TYPE, "AMDGPU IR optimizations",
                    false, false)

char AMDGPUCodeGenPrepare::ID = 0;

FunctionPass *llvm::createAMDGPUCodeGenPreparePass() {
  return new AMDGPUCodeGenPrepare();
}
