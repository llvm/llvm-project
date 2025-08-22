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
#include "SIModeRegisterDefaults.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "amdgpu-codegenprepare"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

static cl::opt<bool> WidenLoads(
  "amdgpu-codegenprepare-widen-constant-loads",
  cl::desc("Widen sub-dword constant address space loads in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(false));

static cl::opt<bool>
    BreakLargePHIs("amdgpu-codegenprepare-break-large-phis",
                   cl::desc("Break large PHI nodes for DAGISel"),
                   cl::ReallyHidden, cl::init(true));

static cl::opt<bool>
    ForceBreakLargePHIs("amdgpu-codegenprepare-force-break-large-phis",
                        cl::desc("For testing purposes, always break large "
                                 "PHIs even if it isn't profitable."),
                        cl::ReallyHidden, cl::init(false));

static cl::opt<unsigned> BreakLargePHIsThreshold(
    "amdgpu-codegenprepare-break-large-phis-threshold",
    cl::desc("Minimum type size in bits for breaking large PHI nodes"),
    cl::ReallyHidden, cl::init(32));

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

// Disable processing of fdiv so we can better test the backend implementations.
static cl::opt<bool> DisableFDivExpand(
  "amdgpu-codegenprepare-disable-fdiv-expansion",
  cl::desc("Prevent expanding floating point division in AMDGPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(false));

class AMDGPUCodeGenPrepareImpl
    : public InstVisitor<AMDGPUCodeGenPrepareImpl, bool> {
public:
  Function &F;
  const GCNSubtarget &ST;
  const AMDGPUTargetMachine &TM;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  const DominatorTree *DT;
  const UniformityInfo &UA;
  const DataLayout &DL;
  const bool HasFP32DenormalFlush;
  bool FlowChanged = false;
  mutable Function *SqrtF32 = nullptr;
  mutable Function *LdexpF32 = nullptr;

  DenseMap<const PHINode *, bool> BreakPhiNodesCache;

  AMDGPUCodeGenPrepareImpl(Function &F, const AMDGPUTargetMachine &TM,
                           const TargetLibraryInfo *TLI, AssumptionCache *AC,
                           const DominatorTree *DT, const UniformityInfo &UA)
      : F(F), ST(TM.getSubtarget<GCNSubtarget>(F)), TM(TM), TLI(TLI), AC(AC),
        DT(DT), UA(UA), DL(F.getDataLayout()),
        HasFP32DenormalFlush(SIModeRegisterDefaults(F, ST).FP32Denormals ==
                             DenormalMode::getPreserveSign()) {}

  Function *getSqrtF32() const {
    if (SqrtF32)
      return SqrtF32;

    LLVMContext &Ctx = F.getContext();
    SqrtF32 = Intrinsic::getOrInsertDeclaration(
        F.getParent(), Intrinsic::amdgcn_sqrt, {Type::getFloatTy(Ctx)});
    return SqrtF32;
  }

  Function *getLdexpF32() const {
    if (LdexpF32)
      return LdexpF32;

    LLVMContext &Ctx = F.getContext();
    LdexpF32 = Intrinsic::getOrInsertDeclaration(
        F.getParent(), Intrinsic::ldexp,
        {Type::getFloatTy(Ctx), Type::getInt32Ty(Ctx)});
    return LdexpF32;
  }

  bool canBreakPHINode(const PHINode &I);

  /// \returns True if binary operation \p I is a signed binary operation, false
  /// otherwise.
  bool isSigned(const BinaryOperator &I) const;

  /// \returns True if the condition of 'select' operation \p I comes from a
  /// signed 'icmp' operation, false otherwise.
  bool isSigned(const SelectInst &I) const;

  /// Return true if \p T is a legal scalar floating point type.
  bool isLegalFloatingTy(const Type *T) const;

  /// Wrapper to pass all the arguments to computeKnownFPClass
  KnownFPClass computeKnownFPClass(const Value *V, FPClassTest Interested,
                                   const Instruction *CtxI) const {
    return llvm::computeKnownFPClass(V, DL, Interested, TLI, AC, CtxI, DT);
  }

  bool canIgnoreDenormalInput(const Value *V, const Instruction *CtxI) const {
    return HasFP32DenormalFlush ||
           computeKnownFPClass(V, fcSubnormal, CtxI).isKnownNeverSubnormal();
  }

  /// \returns The minimum number of bits needed to store the value of \Op as an
  /// unsigned integer. Truncating to this size and then zero-extending to
  /// the original will not change the value.
  unsigned numBitsUnsigned(Value *Op) const;

  /// \returns The minimum number of bits needed to store the value of \Op as a
  /// signed integer. Truncating to this size and then sign-extending to
  /// the original size will not change the value.
  unsigned numBitsSigned(Value *Op) const;

  /// Replace mul instructions with llvm.amdgcn.mul.u24 or llvm.amdgcn.mul.s24.
  /// SelectionDAG has an issue where an and asserting the bits are known
  bool replaceMulWithMul24(BinaryOperator &I) const;

  /// Perform same function as equivalently named function in DAGCombiner. Since
  /// we expand some divisions here, we need to perform this before obscuring.
  bool foldBinOpIntoSelect(BinaryOperator &I) const;

  bool divHasSpecialOptimization(BinaryOperator &I,
                                 Value *Num, Value *Den) const;
  unsigned getDivNumBits(BinaryOperator &I, Value *Num, Value *Den,
                         unsigned MaxDivBits, bool Signed) const;

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

  Value *matchFractPat(IntrinsicInst &I);
  Value *applyFractPat(IRBuilder<> &Builder, Value *FractArg);

  bool canOptimizeWithRsq(const FPMathOperator *SqrtOp, FastMathFlags DivFMF,
                          FastMathFlags SqrtFMF) const;

  Value *optimizeWithRsq(IRBuilder<> &Builder, Value *Num, Value *Den,
                         FastMathFlags DivFMF, FastMathFlags SqrtFMF,
                         const Instruction *CtxI) const;

  Value *optimizeWithRcp(IRBuilder<> &Builder, Value *Num, Value *Den,
                         FastMathFlags FMF, const Instruction *CtxI) const;
  Value *optimizeWithFDivFast(IRBuilder<> &Builder, Value *Num, Value *Den,
                              float ReqdAccuracy) const;

  Value *visitFDivElement(IRBuilder<> &Builder, Value *Num, Value *Den,
                          FastMathFlags DivFMF, FastMathFlags SqrtFMF,
                          Value *RsqOp, const Instruction *FDiv,
                          float ReqdAccuracy) const;

  std::pair<Value *, Value *> getFrexpResults(IRBuilder<> &Builder,
                                              Value *Src) const;

  Value *emitRcpIEEE1ULP(IRBuilder<> &Builder, Value *Src,
                         bool IsNegative) const;
  Value *emitFrexpDiv(IRBuilder<> &Builder, Value *LHS, Value *RHS,
                      FastMathFlags FMF) const;
  Value *emitSqrtIEEE2ULP(IRBuilder<> &Builder, Value *Src,
                          FastMathFlags FMF) const;

public:
  bool visitFDiv(BinaryOperator &I);

  bool visitInstruction(Instruction &I) { return false; }
  bool visitBinaryOperator(BinaryOperator &I);
  bool visitLoadInst(LoadInst &I);
  bool visitSelectInst(SelectInst &I);
  bool visitPHINode(PHINode &I);
  bool visitAddrSpaceCastInst(AddrSpaceCastInst &I);

  bool visitIntrinsicInst(IntrinsicInst &I);
  bool visitFMinLike(IntrinsicInst &I);
  bool visitSqrt(IntrinsicInst &I);
  bool run();
};

class AMDGPUCodeGenPrepare : public FunctionPass {
public:
  static char ID;
  AMDGPUCodeGenPrepare() : FunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();

    // FIXME: Division expansion needs to preserve the dominator tree.
    if (!ExpandDiv64InIR)
      AU.setPreservesAll();
  }
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return "AMDGPU IR optimizations"; }
};

} // end anonymous namespace

bool AMDGPUCodeGenPrepareImpl::run() {
  BreakPhiNodesCache.clear();
  bool MadeChange = false;

  Function::iterator NextBB;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; FI = NextBB) {
    BasicBlock *BB = &*FI;
    NextBB = std::next(FI);

    BasicBlock::iterator Next;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;
         I = Next) {
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

bool AMDGPUCodeGenPrepareImpl::isSigned(const BinaryOperator &I) const {
  return I.getOpcode() == Instruction::AShr ||
      I.getOpcode() == Instruction::SDiv || I.getOpcode() == Instruction::SRem;
}

bool AMDGPUCodeGenPrepareImpl::isSigned(const SelectInst &I) const {
  return isa<ICmpInst>(I.getOperand(0)) &&
         cast<ICmpInst>(I.getOperand(0))->isSigned();
}

bool AMDGPUCodeGenPrepareImpl::isLegalFloatingTy(const Type *Ty) const {
  return Ty->isFloatTy() || Ty->isDoubleTy() ||
         (Ty->isHalfTy() && ST.has16BitInsts());
}

bool AMDGPUCodeGenPrepareImpl::canWidenScalarExtLoad(LoadInst &I) const {
  Type *Ty = I.getType();
  int TySize = DL.getTypeSizeInBits(Ty);
  Align Alignment = DL.getValueOrABITypeAlignment(I.getAlign(), Ty);

  return I.isSimple() && TySize < 32 && Alignment >= 4 && UA.isUniform(&I);
}

unsigned AMDGPUCodeGenPrepareImpl::numBitsUnsigned(Value *Op) const {
  return computeKnownBits(Op, DL, AC).countMaxActiveBits();
}

unsigned AMDGPUCodeGenPrepareImpl::numBitsSigned(Value *Op) const {
  return ComputeMaxSignificantBits(Op, DL, AC);
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
  if (!Ty->isVectorTy()) {
    assert(Values.size() == 1);
    return Values[0];
  }

  Value *NewVal = PoisonValue::get(Ty);
  for (int I = 0, E = Values.size(); I != E; ++I)
    NewVal = Builder.CreateInsertElement(NewVal, Values[I], I);

  return NewVal;
}

bool AMDGPUCodeGenPrepareImpl::replaceMulWithMul24(BinaryOperator &I) const {
  if (I.getOpcode() != Instruction::Mul)
    return false;

  Type *Ty = I.getType();
  unsigned Size = Ty->getScalarSizeInBits();
  if (Size <= 16 && ST.has16BitInsts())
    return false;

  // Prefer scalar if this could be s_mul_i32
  if (UA.isUniform(&I))
    return false;

  Value *LHS = I.getOperand(0);
  Value *RHS = I.getOperand(1);
  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  unsigned LHSBits = 0, RHSBits = 0;
  bool IsSigned = false;

  if (ST.hasMulU24() && (LHSBits = numBitsUnsigned(LHS)) <= 24 &&
      (RHSBits = numBitsUnsigned(RHS)) <= 24) {
    IsSigned = false;

  } else if (ST.hasMulI24() && (LHSBits = numBitsSigned(LHS)) <= 24 &&
             (RHSBits = numBitsSigned(RHS)) <= 24) {
    IsSigned = true;

  } else
    return false;

  SmallVector<Value *, 4> LHSVals;
  SmallVector<Value *, 4> RHSVals;
  SmallVector<Value *, 4> ResultVals;
  extractValues(Builder, LHSVals, LHS);
  extractValues(Builder, RHSVals, RHS);

  IntegerType *I32Ty = Builder.getInt32Ty();
  IntegerType *IntrinTy = Size > 32 ? Builder.getInt64Ty() : I32Ty;
  Type *DstTy = LHSVals[0]->getType();

  for (int I = 0, E = LHSVals.size(); I != E; ++I) {
    Value *LHS = IsSigned ? Builder.CreateSExtOrTrunc(LHSVals[I], I32Ty)
                          : Builder.CreateZExtOrTrunc(LHSVals[I], I32Ty);
    Value *RHS = IsSigned ? Builder.CreateSExtOrTrunc(RHSVals[I], I32Ty)
                          : Builder.CreateZExtOrTrunc(RHSVals[I], I32Ty);
    Intrinsic::ID ID =
        IsSigned ? Intrinsic::amdgcn_mul_i24 : Intrinsic::amdgcn_mul_u24;
    Value *Result = Builder.CreateIntrinsic(ID, {IntrinTy}, {LHS, RHS});
    Result = IsSigned ? Builder.CreateSExtOrTrunc(Result, DstTy)
                      : Builder.CreateZExtOrTrunc(Result, DstTy);
    ResultVals.push_back(Result);
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

bool AMDGPUCodeGenPrepareImpl::foldBinOpIntoSelect(BinaryOperator &BO) const {
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
    CT = ConstantFoldCastOperand(CastOp->getOpcode(), CT, BO.getType(), DL);
    CF = ConstantFoldCastOperand(CastOp->getOpcode(), CF, BO.getType(), DL);
  }

  // TODO: Handle special 0/-1 cases DAG combine does, although we only really
  // need to handle divisions here.
  Constant *FoldedT =
      SelOpNo ? ConstantFoldBinaryOpOperands(BO.getOpcode(), CBO, CT, DL)
              : ConstantFoldBinaryOpOperands(BO.getOpcode(), CT, CBO, DL);
  if (!FoldedT || isa<ConstantExpr>(FoldedT))
    return false;

  Constant *FoldedF =
      SelOpNo ? ConstantFoldBinaryOpOperands(BO.getOpcode(), CBO, CF, DL)
              : ConstantFoldBinaryOpOperands(BO.getOpcode(), CF, CBO, DL);
  if (!FoldedF || isa<ConstantExpr>(FoldedF))
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

std::pair<Value *, Value *>
AMDGPUCodeGenPrepareImpl::getFrexpResults(IRBuilder<> &Builder,
                                          Value *Src) const {
  Type *Ty = Src->getType();
  Value *Frexp = Builder.CreateIntrinsic(Intrinsic::frexp,
                                         {Ty, Builder.getInt32Ty()}, Src);
  Value *FrexpMant = Builder.CreateExtractValue(Frexp, {0});

  // Bypass the bug workaround for the exponent result since it doesn't matter.
  // TODO: Does the bug workaround even really need to consider the exponent
  // result? It's unspecified by the spec.

  Value *FrexpExp =
      ST.hasFractBug()
          ? Builder.CreateIntrinsic(Intrinsic::amdgcn_frexp_exp,
                                    {Builder.getInt32Ty(), Ty}, Src)
          : Builder.CreateExtractValue(Frexp, {1});
  return {FrexpMant, FrexpExp};
}

/// Emit an expansion of 1.0 / Src good for 1ulp that supports denormals.
Value *AMDGPUCodeGenPrepareImpl::emitRcpIEEE1ULP(IRBuilder<> &Builder,
                                                 Value *Src,
                                                 bool IsNegative) const {
  // Same as for 1.0, but expand the sign out of the constant.
  // -1.0 / x -> rcp (fneg x)
  if (IsNegative)
    Src = Builder.CreateFNeg(Src);

  // The rcp instruction doesn't support denormals, so scale the input
  // out of the denormal range and convert at the end.
  //
  // Expand as 2^-n * (1.0 / (x * 2^n))

  // TODO: Skip scaling if input is known never denormal and the input
  // range won't underflow to denormal. The hard part is knowing the
  // result. We need a range check, the result could be denormal for
  // 0x1p+126 < den <= 0x1p+127.
  auto [FrexpMant, FrexpExp] = getFrexpResults(Builder, Src);
  Value *ScaleFactor = Builder.CreateNeg(FrexpExp);
  Value *Rcp = Builder.CreateUnaryIntrinsic(Intrinsic::amdgcn_rcp, FrexpMant);
  return Builder.CreateCall(getLdexpF32(), {Rcp, ScaleFactor});
}

/// Emit a 2ulp expansion for fdiv by using frexp for input scaling.
Value *AMDGPUCodeGenPrepareImpl::emitFrexpDiv(IRBuilder<> &Builder, Value *LHS,
                                              Value *RHS,
                                              FastMathFlags FMF) const {
  // If we have have to work around the fract/frexp bug, we're worse off than
  // using the fdiv.fast expansion. The full safe expansion is faster if we have
  // fast FMA.
  if (HasFP32DenormalFlush && ST.hasFractBug() && !ST.hasFastFMAF32() &&
      (!FMF.noNaNs() || !FMF.noInfs()))
    return nullptr;

  // We're scaling the LHS to avoid a denormal input, and scale the denominator
  // to avoid large values underflowing the result.
  auto [FrexpMantRHS, FrexpExpRHS] = getFrexpResults(Builder, RHS);

  Value *Rcp =
      Builder.CreateUnaryIntrinsic(Intrinsic::amdgcn_rcp, FrexpMantRHS);

  auto [FrexpMantLHS, FrexpExpLHS] = getFrexpResults(Builder, LHS);
  Value *Mul = Builder.CreateFMul(FrexpMantLHS, Rcp);

  // We multiplied by 2^N/2^M, so we need to multiply by 2^(N-M) to scale the
  // result.
  Value *ExpDiff = Builder.CreateSub(FrexpExpLHS, FrexpExpRHS);
  return Builder.CreateCall(getLdexpF32(), {Mul, ExpDiff});
}

/// Emit a sqrt that handles denormals and is accurate to 2ulp.
Value *AMDGPUCodeGenPrepareImpl::emitSqrtIEEE2ULP(IRBuilder<> &Builder,
                                                  Value *Src,
                                                  FastMathFlags FMF) const {
  Type *Ty = Src->getType();
  APFloat SmallestNormal =
      APFloat::getSmallestNormalized(Ty->getFltSemantics());
  Value *NeedScale =
      Builder.CreateFCmpOLT(Src, ConstantFP::get(Ty, SmallestNormal));

  ConstantInt *Zero = Builder.getInt32(0);
  Value *InputScaleFactor =
      Builder.CreateSelect(NeedScale, Builder.getInt32(32), Zero);

  Value *Scaled = Builder.CreateCall(getLdexpF32(), {Src, InputScaleFactor});

  Value *Sqrt = Builder.CreateCall(getSqrtF32(), Scaled);

  Value *OutputScaleFactor =
      Builder.CreateSelect(NeedScale, Builder.getInt32(-16), Zero);
  return Builder.CreateCall(getLdexpF32(), {Sqrt, OutputScaleFactor});
}

/// Emit an expansion of 1.0 / sqrt(Src) good for 1ulp that supports denormals.
static Value *emitRsqIEEE1ULP(IRBuilder<> &Builder, Value *Src,
                              bool IsNegative) {
  // bool need_scale = x < 0x1p-126f;
  // float input_scale = need_scale ? 0x1.0p+24f : 1.0f;
  // float output_scale = need_scale ? 0x1.0p+12f : 1.0f;
  // rsq(x * input_scale) * output_scale;

  Type *Ty = Src->getType();
  APFloat SmallestNormal =
      APFloat::getSmallestNormalized(Ty->getFltSemantics());
  Value *NeedScale =
      Builder.CreateFCmpOLT(Src, ConstantFP::get(Ty, SmallestNormal));
  Constant *One = ConstantFP::get(Ty, 1.0);
  Constant *InputScale = ConstantFP::get(Ty, 0x1.0p+24);
  Constant *OutputScale =
      ConstantFP::get(Ty, IsNegative ? -0x1.0p+12 : 0x1.0p+12);

  Value *InputScaleFactor = Builder.CreateSelect(NeedScale, InputScale, One);

  Value *ScaledInput = Builder.CreateFMul(Src, InputScaleFactor);
  Value *Rsq = Builder.CreateUnaryIntrinsic(Intrinsic::amdgcn_rsq, ScaledInput);
  Value *OutputScaleFactor = Builder.CreateSelect(
      NeedScale, OutputScale, IsNegative ? ConstantFP::get(Ty, -1.0) : One);

  return Builder.CreateFMul(Rsq, OutputScaleFactor);
}

bool AMDGPUCodeGenPrepareImpl::canOptimizeWithRsq(const FPMathOperator *SqrtOp,
                                                  FastMathFlags DivFMF,
                                                  FastMathFlags SqrtFMF) const {
  // The rsqrt contraction increases accuracy from ~2ulp to ~1ulp.
  if (!DivFMF.allowContract() || !SqrtFMF.allowContract())
    return false;

  // v_rsq_f32 gives 1ulp
  return SqrtFMF.approxFunc() || SqrtOp->getFPAccuracy() >= 1.0f;
}

Value *AMDGPUCodeGenPrepareImpl::optimizeWithRsq(
    IRBuilder<> &Builder, Value *Num, Value *Den, const FastMathFlags DivFMF,
    const FastMathFlags SqrtFMF, const Instruction *CtxI) const {
  // The rsqrt contraction increases accuracy from ~2ulp to ~1ulp.
  assert(DivFMF.allowContract() && SqrtFMF.allowContract());

  // rsq_f16 is accurate to 0.51 ulp.
  // rsq_f32 is accurate for !fpmath >= 1.0ulp and denormals are flushed.
  // rsq_f64 is never accurate.
  const ConstantFP *CLHS = dyn_cast<ConstantFP>(Num);
  if (!CLHS)
    return nullptr;

  assert(Den->getType()->isFloatTy());

  bool IsNegative = false;

  // TODO: Handle other numerator values with arcp.
  if (CLHS->isExactlyValue(1.0) || (IsNegative = CLHS->isExactlyValue(-1.0))) {
    // Add in the sqrt flags.
    IRBuilder<>::FastMathFlagGuard Guard(Builder);
    Builder.setFastMathFlags(DivFMF | SqrtFMF);

    if ((DivFMF.approxFunc() && SqrtFMF.approxFunc()) ||
        canIgnoreDenormalInput(Den, CtxI)) {
      Value *Result = Builder.CreateUnaryIntrinsic(Intrinsic::amdgcn_rsq, Den);
      // -1.0 / sqrt(x) -> fneg(rsq(x))
      return IsNegative ? Builder.CreateFNeg(Result) : Result;
    }

    return emitRsqIEEE1ULP(Builder, Den, IsNegative);
  }

  return nullptr;
}

// Optimize fdiv with rcp:
//
// 1/x -> rcp(x) when rcp is sufficiently accurate or inaccurate rcp is
//               allowed with afn.
//
// a/b -> a*rcp(b) when arcp is allowed, and we only need provide ULP 1.0
Value *
AMDGPUCodeGenPrepareImpl::optimizeWithRcp(IRBuilder<> &Builder, Value *Num,
                                          Value *Den, FastMathFlags FMF,
                                          const Instruction *CtxI) const {
  // rcp_f16 is accurate to 0.51 ulp.
  // rcp_f32 is accurate for !fpmath >= 1.0ulp and denormals are flushed.
  // rcp_f64 is never accurate.
  assert(Den->getType()->isFloatTy());

  if (const ConstantFP *CLHS = dyn_cast<ConstantFP>(Num)) {
    bool IsNegative = false;
    if (CLHS->isExactlyValue(1.0) ||
        (IsNegative = CLHS->isExactlyValue(-1.0))) {
      Value *Src = Den;

      if (HasFP32DenormalFlush || FMF.approxFunc()) {
        // -1.0 / x -> 1.0 / fneg(x)
        if (IsNegative)
          Src = Builder.CreateFNeg(Src);

        // v_rcp_f32 and v_rsq_f32 do not support denormals, and according to
        // the CI documentation has a worst case error of 1 ulp.
        // OpenCL requires <= 2.5 ulp for 1.0 / x, so it should always be OK
        // to use it as long as we aren't trying to use denormals.
        //
        // v_rcp_f16 and v_rsq_f16 DO support denormals.

        // NOTE: v_sqrt and v_rcp will be combined to v_rsq later. So we don't
        //       insert rsq intrinsic here.

        // 1.0 / x -> rcp(x)
        return Builder.CreateUnaryIntrinsic(Intrinsic::amdgcn_rcp, Src);
      }

      // TODO: If the input isn't denormal, and we know the input exponent isn't
      // big enough to introduce a denormal we can avoid the scaling.
      return emitRcpIEEE1ULP(Builder, Src, IsNegative);
    }
  }

  if (FMF.allowReciprocal()) {
    // x / y -> x * (1.0 / y)

    // TODO: Could avoid denormal scaling and use raw rcp if we knew the output
    // will never underflow.
    if (HasFP32DenormalFlush || FMF.approxFunc()) {
      Value *Recip = Builder.CreateUnaryIntrinsic(Intrinsic::amdgcn_rcp, Den);
      return Builder.CreateFMul(Num, Recip);
    }

    Value *Recip = emitRcpIEEE1ULP(Builder, Den, false);
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
Value *AMDGPUCodeGenPrepareImpl::optimizeWithFDivFast(
    IRBuilder<> &Builder, Value *Num, Value *Den, float ReqdAccuracy) const {
  // fdiv.fast can achieve 2.5 ULP accuracy.
  if (ReqdAccuracy < 2.5f)
    return nullptr;

  // Only have fdiv.fast for f32.
  assert(Den->getType()->isFloatTy());

  bool NumIsOne = false;
  if (const ConstantFP *CNum = dyn_cast<ConstantFP>(Num)) {
    if (CNum->isExactlyValue(+1.0) || CNum->isExactlyValue(-1.0))
      NumIsOne = true;
  }

  // fdiv does not support denormals. But 1.0/x is always fine to use it.
  //
  // TODO: This works for any value with a specific known exponent range, don't
  // just limit to constant 1.
  if (!HasFP32DenormalFlush && !NumIsOne)
    return nullptr;

  return Builder.CreateIntrinsic(Intrinsic::amdgcn_fdiv_fast, {Num, Den});
}

Value *AMDGPUCodeGenPrepareImpl::visitFDivElement(
    IRBuilder<> &Builder, Value *Num, Value *Den, FastMathFlags DivFMF,
    FastMathFlags SqrtFMF, Value *RsqOp, const Instruction *FDivInst,
    float ReqdDivAccuracy) const {
  if (RsqOp) {
    Value *Rsq =
        optimizeWithRsq(Builder, Num, RsqOp, DivFMF, SqrtFMF, FDivInst);
    if (Rsq)
      return Rsq;
  }

  Value *Rcp = optimizeWithRcp(Builder, Num, Den, DivFMF, FDivInst);
  if (Rcp)
    return Rcp;

  // In the basic case fdiv_fast has the same instruction count as the frexp div
  // expansion. Slightly prefer fdiv_fast since it ends in an fmul that can
  // potentially be fused into a user. Also, materialization of the constants
  // can be reused for multiple instances.
  Value *FDivFast = optimizeWithFDivFast(Builder, Num, Den, ReqdDivAccuracy);
  if (FDivFast)
    return FDivFast;

  return emitFrexpDiv(Builder, Num, Den, DivFMF);
}

// Optimizations is performed based on fpmath, fast math flags as well as
// denormals to optimize fdiv with either rcp or fdiv.fast.
//
// With rcp:
//   1/x -> rcp(x) when rcp is sufficiently accurate or inaccurate rcp is
//                 allowed with afn.
//
//   a/b -> a*rcp(b) when inaccurate rcp is allowed with afn.
//
// With fdiv.fast:
//   a/b -> fdiv.fast(a, b) when !fpmath >= 2.5ulp with denormals flushed.
//
//   1/x -> fdiv.fast(1,x)  when !fpmath >= 2.5ulp.
//
// NOTE: rcp is the preference in cases that both are legal.
bool AMDGPUCodeGenPrepareImpl::visitFDiv(BinaryOperator &FDiv) {
  if (DisableFDivExpand)
    return false;

  Type *Ty = FDiv.getType()->getScalarType();
  if (!Ty->isFloatTy())
    return false;

  // The f64 rcp/rsq approximations are pretty inaccurate. We can do an
  // expansion around them in codegen. f16 is good enough to always use.

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&FDiv);
  const FastMathFlags DivFMF = FPOp->getFastMathFlags();
  const float ReqdAccuracy = FPOp->getFPAccuracy();

  FastMathFlags SqrtFMF;

  Value *Num = FDiv.getOperand(0);
  Value *Den = FDiv.getOperand(1);

  Value *RsqOp = nullptr;
  auto *DenII = dyn_cast<IntrinsicInst>(Den);
  if (DenII && DenII->getIntrinsicID() == Intrinsic::sqrt &&
      DenII->hasOneUse()) {
    const auto *SqrtOp = cast<FPMathOperator>(DenII);
    SqrtFMF = SqrtOp->getFastMathFlags();
    if (canOptimizeWithRsq(SqrtOp, DivFMF, SqrtFMF))
      RsqOp = SqrtOp->getOperand(0);
  }

  // Inaccurate rcp is allowed with afn.
  //
  // Defer to codegen to handle this.
  //
  // TODO: Decide on an interpretation for interactions between afn + arcp +
  // !fpmath, and make it consistent between here and codegen. For now, defer
  // expansion of afn to codegen. The current interpretation is so aggressive we
  // don't need any pre-consideration here when we have better information. A
  // more conservative interpretation could use handling here.
  const bool AllowInaccurateRcp = DivFMF.approxFunc();
  if (!RsqOp && AllowInaccurateRcp)
    return false;

  // Defer the correct implementations to codegen.
  if (ReqdAccuracy < 1.0f)
    return false;

  IRBuilder<> Builder(FDiv.getParent(), std::next(FDiv.getIterator()));
  Builder.setFastMathFlags(DivFMF);
  Builder.SetCurrentDebugLocation(FDiv.getDebugLoc());

  SmallVector<Value *, 4> NumVals;
  SmallVector<Value *, 4> DenVals;
  SmallVector<Value *, 4> RsqDenVals;
  extractValues(Builder, NumVals, Num);
  extractValues(Builder, DenVals, Den);

  if (RsqOp)
    extractValues(Builder, RsqDenVals, RsqOp);

  SmallVector<Value *, 4> ResultVals(NumVals.size());
  for (int I = 0, E = NumVals.size(); I != E; ++I) {
    Value *NumElt = NumVals[I];
    Value *DenElt = DenVals[I];
    Value *RsqDenElt = RsqOp ? RsqDenVals[I] : nullptr;

    Value *NewElt =
        visitFDivElement(Builder, NumElt, DenElt, DivFMF, SqrtFMF, RsqDenElt,
                         cast<Instruction>(FPOp), ReqdAccuracy);
    if (!NewElt) {
      // Keep the original, but scalarized.

      // This has the unfortunate side effect of sometimes scalarizing when
      // we're not going to do anything.
      NewElt = Builder.CreateFDiv(NumElt, DenElt);
      if (auto *NewEltInst = dyn_cast<Instruction>(NewElt))
        NewEltInst->copyMetadata(FDiv);
    }

    ResultVals[I] = NewElt;
  }

  Value *NewVal = insertValues(Builder, FDiv.getType(), ResultVals);

  if (NewVal) {
    FDiv.replaceAllUsesWith(NewVal);
    NewVal->takeName(&FDiv);
    RecursivelyDeleteTriviallyDeadInstructions(&FDiv, TLI);
  }

  return true;
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
  return std::pair(Lo, Hi);
}

static Value* getMulHu(IRBuilder<> &Builder, Value *LHS, Value *RHS) {
  return getMul64(Builder, LHS, RHS).second;
}

/// Figure out how many bits are really needed for this division.
/// \p MaxDivBits is an optimization hint to bypass the second
/// ComputeNumSignBits/computeKnownBits call if the first one is
/// insufficient.
unsigned AMDGPUCodeGenPrepareImpl::getDivNumBits(BinaryOperator &I, Value *Num,
                                                 Value *Den,
                                                 unsigned MaxDivBits,
                                                 bool IsSigned) const {
  assert(Num->getType()->getScalarSizeInBits() ==
         Den->getType()->getScalarSizeInBits());
  unsigned SSBits = Num->getType()->getScalarSizeInBits();
  if (IsSigned) {
    unsigned RHSSignBits = ComputeNumSignBits(Den, DL, AC, &I);
    // A sign bit needs to be reserved for shrinking.
    unsigned DivBits = SSBits - RHSSignBits + 1;
    if (DivBits > MaxDivBits)
      return SSBits;

    unsigned LHSSignBits = ComputeNumSignBits(Num, DL, AC, &I);

    unsigned SignBits = std::min(LHSSignBits, RHSSignBits);
    DivBits = SSBits - SignBits + 1;
    return DivBits;
  }

  // All bits are used for unsigned division for Num or Den in range
  // (SignedMax, UnsignedMax].
  KnownBits Known = computeKnownBits(Den, DL, AC, &I);
  if (Known.isNegative() || !Known.isNonNegative())
    return SSBits;
  unsigned RHSSignBits = Known.countMinLeadingZeros();
  unsigned DivBits = SSBits - RHSSignBits;
  if (DivBits > MaxDivBits)
    return SSBits;

  Known = computeKnownBits(Num, DL, AC, &I);
  if (Known.isNegative() || !Known.isNonNegative())
    return SSBits;
  unsigned LHSSignBits = Known.countMinLeadingZeros();

  unsigned SignBits = std::min(LHSSignBits, RHSSignBits);
  DivBits = SSBits - SignBits;
  return DivBits;
}

// The fractional part of a float is enough to accurately represent up to
// a 24-bit signed integer.
Value *AMDGPUCodeGenPrepareImpl::expandDivRem24(IRBuilder<> &Builder,
                                                BinaryOperator &I, Value *Num,
                                                Value *Den, bool IsDiv,
                                                bool IsSigned) const {
  unsigned DivBits = getDivNumBits(I, Num, Den, 24, IsSigned);
  if (DivBits > 24)
    return nullptr;
  return expandDivRem24Impl(Builder, I, Num, Den, DivBits, IsDiv, IsSigned);
}

Value *AMDGPUCodeGenPrepareImpl::expandDivRem24Impl(
    IRBuilder<> &Builder, BinaryOperator &I, Value *Num, Value *Den,
    unsigned DivBits, bool IsDiv, bool IsSigned) const {
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

  Value *RCP = Builder.CreateIntrinsic(Intrinsic::amdgcn_rcp,
                                       Builder.getFloatTy(), {FB});
  Value *FQM = Builder.CreateFMul(FA, RCP);

  // fq = trunc(fqm);
  CallInst *FQ = Builder.CreateUnaryIntrinsic(Intrinsic::trunc, FQM);
  FQ->copyFastMathFlags(Builder.getFastMathFlags());

  // float fqneg = -fq;
  Value *FQNeg = Builder.CreateFNeg(FQ);

  // float fr = mad(fqneg, fb, fa);
  auto FMAD = !ST.hasMadMacF32Insts()
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
bool AMDGPUCodeGenPrepareImpl::divHasSpecialOptimization(BinaryOperator &I,
                                                         Value *Num,
                                                         Value *Den) const {
  if (Constant *C = dyn_cast<Constant>(Den)) {
    // Arbitrary constants get a better expansion as long as a wider mulhi is
    // legal.
    if (C->getType()->getScalarSizeInBits() <= 32)
      return true;

    // TODO: Sdiv check for not exact for some reason.

    // If there's no wider mulhi, there's only a better expansion for powers of
    // two.
    // TODO: Should really know for each vector element.
    if (isKnownToBeAPowerOfTwo(C, DL, true, AC, &I, DT))
      return true;

    return false;
  }

  if (BinaryOperator *BinOpDen = dyn_cast<BinaryOperator>(Den)) {
    // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
    if (BinOpDen->getOpcode() == Instruction::Shl &&
        isa<Constant>(BinOpDen->getOperand(0)) &&
        isKnownToBeAPowerOfTwo(BinOpDen->getOperand(0), DL, true, AC, &I, DT)) {
      return true;
    }
  }

  return false;
}

static Value *getSign32(Value *V, IRBuilder<> &Builder, const DataLayout DL) {
  // Check whether the sign can be determined statically.
  KnownBits Known = computeKnownBits(V, DL);
  if (Known.isNegative())
    return Constant::getAllOnesValue(V->getType());
  if (Known.isNonNegative())
    return Constant::getNullValue(V->getType());
  return Builder.CreateAShr(V, Builder.getInt32(31));
}

Value *AMDGPUCodeGenPrepareImpl::expandDivRem32(IRBuilder<> &Builder,
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

  if (Ty->getScalarSizeInBits() != 32) {
    if (IsSigned) {
      X = Builder.CreateSExtOrTrunc(X, I32Ty);
      Y = Builder.CreateSExtOrTrunc(Y, I32Ty);
    } else {
      X = Builder.CreateZExtOrTrunc(X, I32Ty);
      Y = Builder.CreateZExtOrTrunc(Y, I32Ty);
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
  Value *RcpY = Builder.CreateIntrinsic(Intrinsic::amdgcn_rcp, F32Ty, {FloatY});
  Constant *Scale = ConstantFP::get(F32Ty, llvm::bit_cast<float>(0x4F7FFFFE));
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
    Res = Builder.CreateSExtOrTrunc(Res, Ty);
  } else {
    Res = Builder.CreateZExtOrTrunc(Res, Ty);
  }
  return Res;
}

Value *AMDGPUCodeGenPrepareImpl::shrinkDivRem64(IRBuilder<> &Builder,
                                                BinaryOperator &I, Value *Num,
                                                Value *Den) const {
  if (!ExpandDiv64InIR && divHasSpecialOptimization(I, Num, Den))
    return nullptr;  // Keep it for later optimization.

  Instruction::BinaryOps Opc = I.getOpcode();

  bool IsDiv = Opc == Instruction::SDiv || Opc == Instruction::UDiv;
  bool IsSigned = Opc == Instruction::SDiv || Opc == Instruction::SRem;

  unsigned NumDivBits = getDivNumBits(I, Num, Den, 32, IsSigned);
  if (NumDivBits > 32)
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

void AMDGPUCodeGenPrepareImpl::expandDivRem64(BinaryOperator &I) const {
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

/*
This will cause non-byte load in consistency, for example:
```
    %load = load i1, ptr addrspace(4) %arg, align 4
    %zext = zext i1 %load to
    i64 %add = add i64 %zext
```
Instead of creating `s_and_b32 s0, s0, 1`,
it will create `s_and_b32 s0, s0, 0xff`.
We accept this change since the non-byte load assumes the upper bits
within the byte are all 0.
*/
static bool tryNarrowMathIfNoOverflow(Instruction *I,
                                      const SITargetLowering *TLI,
                                      const TargetTransformInfo &TTI,
                                      const DataLayout &DL) {
  unsigned Opc = I->getOpcode();
  Type *OldType = I->getType();

  if (Opc != Instruction::Add && Opc != Instruction::Mul)
    return false;

  unsigned OrigBit = OldType->getScalarSizeInBits();

  if (Opc != Instruction::Add && Opc != Instruction::Mul)
    llvm_unreachable("Unexpected opcode, only valid for Instruction::Add and "
                     "Instruction::Mul.");

  unsigned MaxBitsNeeded = computeKnownBits(I, DL).countMaxActiveBits();

  MaxBitsNeeded = std::max<unsigned>(bit_ceil(MaxBitsNeeded), 8);
  Type *NewType = DL.getSmallestLegalIntType(I->getContext(), MaxBitsNeeded);
  if (!NewType)
    return false;
  unsigned NewBit = NewType->getIntegerBitWidth();
  if (NewBit >= OrigBit)
    return false;
  NewType = I->getType()->getWithNewBitWidth(NewBit);

  // Old cost
  InstructionCost OldCost =
      TTI.getArithmeticInstrCost(Opc, OldType, TTI::TCK_RecipThroughput);
  // New cost of new op
  InstructionCost NewCost =
      TTI.getArithmeticInstrCost(Opc, NewType, TTI::TCK_RecipThroughput);
  // New cost of narrowing 2 operands (use trunc)
  int NumOfNonConstOps = 2;
  if (isa<Constant>(I->getOperand(0)) || isa<Constant>(I->getOperand(1))) {
    // Cannot be both constant, should be propagated
    NumOfNonConstOps = 1;
  }
  NewCost += NumOfNonConstOps * TTI.getCastInstrCost(Instruction::Trunc,
                                                     NewType, OldType,
                                                     TTI.getCastContextHint(I),
                                                     TTI::TCK_RecipThroughput);
  // New cost of zext narrowed result to original type
  NewCost +=
      TTI.getCastInstrCost(Instruction::ZExt, OldType, NewType,
                           TTI.getCastContextHint(I), TTI::TCK_RecipThroughput);
  if (NewCost >= OldCost)
    return false;

  IRBuilder<> Builder(I);
  Value *Trunc0 = Builder.CreateTrunc(I->getOperand(0), NewType);
  Value *Trunc1 = Builder.CreateTrunc(I->getOperand(1), NewType);
  Value *Arith =
      Builder.CreateBinOp((Instruction::BinaryOps)Opc, Trunc0, Trunc1);

  Value *Zext = Builder.CreateZExt(Arith, OldType);
  I->replaceAllUsesWith(Zext);
  I->eraseFromParent();
  return true;
}

bool AMDGPUCodeGenPrepareImpl::visitBinaryOperator(BinaryOperator &I) {
  if (foldBinOpIntoSelect(I))
    return true;

  if (UseMul24Intrin && replaceMulWithMul24(I))
    return true;
  if (tryNarrowMathIfNoOverflow(&I, ST.getTargetLowering(),
                                TM.getTargetTransformInfo(F), DL))
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
      NewDiv = PoisonValue::get(VT);

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
            // CreateBinOp does constant folding. If the operands are constant,
            // it will return a Constant instead of a BinaryOperator.
            if (auto *NewEltBO = dyn_cast<BinaryOperator>(NewElt))
              Div64ToExpand.push_back(NewEltBO);
          }
        }

        if (auto *NewEltI = dyn_cast<Instruction>(NewElt))
          NewEltI->copyIRFlags(&I);

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
      FlowChanged = true;
      Changed = true;
    }
  }

  return Changed;
}

bool AMDGPUCodeGenPrepareImpl::visitLoadInst(LoadInst &I) {
  if (!WidenLoads)
    return false;

  if ((I.getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS ||
       I.getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT) &&
      canWidenScalarExtLoad(I)) {
    IRBuilder<> Builder(&I);
    Builder.SetCurrentDebugLocation(I.getDebugLoc());

    Type *I32Ty = Builder.getInt32Ty();
    LoadInst *WidenLoad = Builder.CreateLoad(I32Ty, I.getPointerOperand());
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
                               MDNode::get(F.getContext(), LowAndHigh));
      }
    }

    int TySize = DL.getTypeSizeInBits(I.getType());
    Type *IntNTy = Builder.getIntNTy(TySize);
    Value *ValTrunc = Builder.CreateTrunc(WidenLoad, IntNTy);
    Value *ValOrig = Builder.CreateBitCast(ValTrunc, I.getType());
    I.replaceAllUsesWith(ValOrig);
    I.eraseFromParent();
    return true;
  }

  return false;
}

bool AMDGPUCodeGenPrepareImpl::visitSelectInst(SelectInst &I) {
  Value *Cond = I.getCondition();
  Value *TrueVal = I.getTrueValue();
  Value *FalseVal = I.getFalseValue();
  Value *CmpVal;
  CmpPredicate Pred;

  // Match fract pattern with nan check.
  if (!match(Cond, m_FCmp(Pred, m_Value(CmpVal), m_NonNaN())))
    return false;

  FPMathOperator *FPOp = dyn_cast<FPMathOperator>(&I);
  if (!FPOp)
    return false;

  IRBuilder<> Builder(&I);
  Builder.setFastMathFlags(FPOp->getFastMathFlags());

  auto *IITrue = dyn_cast<IntrinsicInst>(TrueVal);
  auto *IIFalse = dyn_cast<IntrinsicInst>(FalseVal);

  Value *Fract = nullptr;
  if (Pred == FCmpInst::FCMP_UNO && TrueVal == CmpVal && IIFalse &&
      CmpVal == matchFractPat(*IIFalse)) {
    // isnan(x) ? x : fract(x)
    Fract = applyFractPat(Builder, CmpVal);
  } else if (Pred == FCmpInst::FCMP_ORD && FalseVal == CmpVal && IITrue &&
             CmpVal == matchFractPat(*IITrue)) {
    // !isnan(x) ? fract(x) : x
    Fract = applyFractPat(Builder, CmpVal);
  } else
    return false;

  Fract->takeName(&I);
  I.replaceAllUsesWith(Fract);
  RecursivelyDeleteTriviallyDeadInstructions(&I, TLI);
  return true;
}

static bool areInSameBB(const Value *A, const Value *B) {
  const auto *IA = dyn_cast<Instruction>(A);
  const auto *IB = dyn_cast<Instruction>(B);
  return IA && IB && IA->getParent() == IB->getParent();
}

// Helper for breaking large PHIs that returns true when an extractelement on V
// is likely to be folded away by the DAG combiner.
static bool isInterestingPHIIncomingValue(const Value *V) {
  const auto *FVT = dyn_cast<FixedVectorType>(V->getType());
  if (!FVT)
    return false;

  const Value *CurVal = V;

  // Check for insertelements, keeping track of the elements covered.
  BitVector EltsCovered(FVT->getNumElements());
  while (const auto *IE = dyn_cast<InsertElementInst>(CurVal)) {
    const auto *Idx = dyn_cast<ConstantInt>(IE->getOperand(2));

    // Non constant index/out of bounds index -> folding is unlikely.
    // The latter is more of a sanity check because canonical IR should just
    // have replaced those with poison.
    if (!Idx || Idx->getZExtValue() >= FVT->getNumElements())
      return false;

    const auto *VecSrc = IE->getOperand(0);

    // If the vector source is another instruction, it must be in the same basic
    // block. Otherwise, the DAGCombiner won't see the whole thing and is
    // unlikely to be able to do anything interesting here.
    if (isa<Instruction>(VecSrc) && !areInSameBB(VecSrc, IE))
      return false;

    CurVal = VecSrc;
    EltsCovered.set(Idx->getZExtValue());

    // All elements covered.
    if (EltsCovered.all())
      return true;
  }

  // We either didn't find a single insertelement, or the insertelement chain
  // ended before all elements were covered. Check for other interesting values.

  // Constants are always interesting because we can just constant fold the
  // extractelements.
  if (isa<Constant>(CurVal))
    return true;

  // shufflevector is likely to be profitable if either operand is a constant,
  // or if either source is in the same block.
  // This is because shufflevector is most often lowered as a series of
  // insert/extract elements anyway.
  if (const auto *SV = dyn_cast<ShuffleVectorInst>(CurVal)) {
    return isa<Constant>(SV->getOperand(1)) ||
           areInSameBB(SV, SV->getOperand(0)) ||
           areInSameBB(SV, SV->getOperand(1));
  }

  return false;
}

static void collectPHINodes(const PHINode &I,
                            SmallPtrSet<const PHINode *, 8> &SeenPHIs) {
  const auto [It, Inserted] = SeenPHIs.insert(&I);
  if (!Inserted)
    return;

  for (const Value *Inc : I.incoming_values()) {
    if (const auto *PhiInc = dyn_cast<PHINode>(Inc))
      collectPHINodes(*PhiInc, SeenPHIs);
  }

  for (const User *U : I.users()) {
    if (const auto *PhiU = dyn_cast<PHINode>(U))
      collectPHINodes(*PhiU, SeenPHIs);
  }
}

bool AMDGPUCodeGenPrepareImpl::canBreakPHINode(const PHINode &I) {
  // Check in the cache first.
  if (const auto It = BreakPhiNodesCache.find(&I);
      It != BreakPhiNodesCache.end())
    return It->second;

  // We consider PHI nodes as part of "chains", so given a PHI node I, we
  // recursively consider all its users and incoming values that are also PHI
  // nodes. We then make a decision about all of those PHIs at once. Either they
  // all get broken up, or none of them do. That way, we avoid cases where a
  // single PHI is/is not broken and we end up reforming/exploding a vector
  // multiple times, or even worse, doing it in a loop.
  SmallPtrSet<const PHINode *, 8> WorkList;
  collectPHINodes(I, WorkList);

#ifndef NDEBUG
  // Check that none of the PHI nodes in the worklist are in the map. If some of
  // them are, it means we're not good enough at collecting related PHIs.
  for (const PHINode *WLP : WorkList) {
    assert(BreakPhiNodesCache.count(WLP) == 0);
  }
#endif

  // To consider a PHI profitable to break, we need to see some interesting
  // incoming values. At least 2/3rd (rounded up) of all PHIs in the worklist
  // must have one to consider all PHIs breakable.
  //
  // This threshold has been determined through performance testing.
  //
  // Note that the computation below is equivalent to
  //
  //    (unsigned)ceil((K / 3.0) * 2)
  //
  // It's simply written this way to avoid mixing integral/FP arithmetic.
  const auto Threshold = (alignTo(WorkList.size() * 2, 3) / 3);
  unsigned NumBreakablePHIs = 0;
  bool CanBreak = false;
  for (const PHINode *Cur : WorkList) {
    // Don't break PHIs that have no interesting incoming values. That is, where
    // there is no clear opportunity to fold the "extractelement" instructions
    // we would add.
    //
    // Note: IC does not run after this pass, so we're only interested in the
    // foldings that the DAG combiner can do.
    if (any_of(Cur->incoming_values(), isInterestingPHIIncomingValue)) {
      if (++NumBreakablePHIs >= Threshold) {
        CanBreak = true;
        break;
      }
    }
  }

  for (const PHINode *Cur : WorkList)
    BreakPhiNodesCache[Cur] = CanBreak;

  return CanBreak;
}

/// Helper class for "break large PHIs" (visitPHINode).
///
/// This represents a slice of a PHI's incoming value, which is made up of:
///   - The type of the slice (Ty)
///   - The index in the incoming value's vector where the slice starts (Idx)
///   - The number of elements in the slice (NumElts).
/// It also keeps track of the NewPHI node inserted for this particular slice.
///
/// Slice examples:
///   <4 x i64> -> Split into four i64 slices.
///     -> [i64, 0, 1], [i64, 1, 1], [i64, 2, 1], [i64, 3, 1]
///   <5 x i16> -> Split into 2 <2 x i16> slices + a i16 tail.
///     -> [<2 x i16>, 0, 2], [<2 x i16>, 2, 2], [i16, 4, 1]
class VectorSlice {
public:
  VectorSlice(Type *Ty, unsigned Idx, unsigned NumElts)
      : Ty(Ty), Idx(Idx), NumElts(NumElts) {}

  Type *Ty = nullptr;
  unsigned Idx = 0;
  unsigned NumElts = 0;
  PHINode *NewPHI = nullptr;

  /// Slice \p Inc according to the information contained within this slice.
  /// This is cached, so if called multiple times for the same \p BB & \p Inc
  /// pair, it returns the same Sliced value as well.
  ///
  /// Note this *intentionally* does not return the same value for, say,
  /// [%bb.0, %0] & [%bb.1, %0] as:
  ///   - It could cause issues with dominance (e.g. if bb.1 is seen first, then
  ///   the value in bb.1 may not be reachable from bb.0 if it's its
  ///   predecessor.)
  ///   - We also want to make our extract instructions as local as possible so
  ///   the DAG has better chances of folding them out. Duplicating them like
  ///   that is beneficial in that regard.
  ///
  /// This is both a minor optimization to avoid creating duplicate
  /// instructions, but also a requirement for correctness. It is not forbidden
  /// for a PHI node to have the same [BB, Val] pair multiple times. If we
  /// returned a new value each time, those previously identical pairs would all
  /// have different incoming values (from the same block) and it'd cause a "PHI
  /// node has multiple entries for the same basic block with different incoming
  /// values!" verifier error.
  Value *getSlicedVal(BasicBlock *BB, Value *Inc, StringRef NewValName) {
    Value *&Res = SlicedVals[{BB, Inc}];
    if (Res)
      return Res;

    IRBuilder<> B(BB->getTerminator());
    if (Instruction *IncInst = dyn_cast<Instruction>(Inc))
      B.SetCurrentDebugLocation(IncInst->getDebugLoc());

    if (NumElts > 1) {
      SmallVector<int, 4> Mask;
      for (unsigned K = Idx; K < (Idx + NumElts); ++K)
        Mask.push_back(K);
      Res = B.CreateShuffleVector(Inc, Mask, NewValName);
    } else
      Res = B.CreateExtractElement(Inc, Idx, NewValName);

    return Res;
  }

private:
  SmallDenseMap<std::pair<BasicBlock *, Value *>, Value *> SlicedVals;
};

bool AMDGPUCodeGenPrepareImpl::visitPHINode(PHINode &I) {
  // Break-up fixed-vector PHIs into smaller pieces.
  // Default threshold is 32, so it breaks up any vector that's >32 bits into
  // its elements, or into 32-bit pieces (for 8/16 bit elts).
  //
  // This is only helpful for DAGISel because it doesn't handle large PHIs as
  // well as GlobalISel. DAGISel lowers PHIs by using CopyToReg/CopyFromReg.
  // With large, odd-sized PHIs we may end up needing many `build_vector`
  // operations with most elements being "undef". This inhibits a lot of
  // optimization opportunities and can result in unreasonably high register
  // pressure and the inevitable stack spilling.
  if (!BreakLargePHIs || getCGPassBuilderOption().EnableGlobalISelOption)
    return false;

  FixedVectorType *FVT = dyn_cast<FixedVectorType>(I.getType());
  if (!FVT || FVT->getNumElements() == 1 ||
      DL.getTypeSizeInBits(FVT) <= BreakLargePHIsThreshold)
    return false;

  if (!ForceBreakLargePHIs && !canBreakPHINode(I))
    return false;

  std::vector<VectorSlice> Slices;

  Type *EltTy = FVT->getElementType();
  {
    unsigned Idx = 0;
    // For 8/16 bits type, don't scalarize fully but break it up into as many
    // 32-bit slices as we can, and scalarize the tail.
    const unsigned EltSize = DL.getTypeSizeInBits(EltTy);
    const unsigned NumElts = FVT->getNumElements();
    if (EltSize == 8 || EltSize == 16) {
      const unsigned SubVecSize = (32 / EltSize);
      Type *SubVecTy = FixedVectorType::get(EltTy, SubVecSize);
      for (unsigned End = alignDown(NumElts, SubVecSize); Idx < End;
           Idx += SubVecSize)
        Slices.emplace_back(SubVecTy, Idx, SubVecSize);
    }

    // Scalarize all remaining elements.
    for (; Idx < NumElts; ++Idx)
      Slices.emplace_back(EltTy, Idx, 1);
  }

  assert(Slices.size() > 1);

  // Create one PHI per vector piece. The "VectorSlice" class takes care of
  // creating the necessary instruction to extract the relevant slices of each
  // incoming value.
  IRBuilder<> B(I.getParent());
  B.SetCurrentDebugLocation(I.getDebugLoc());

  unsigned IncNameSuffix = 0;
  for (VectorSlice &S : Slices) {
    // We need to reset the build on each iteration, because getSlicedVal may
    // have inserted something into I's BB.
    B.SetInsertPoint(I.getParent()->getFirstNonPHIIt());
    S.NewPHI = B.CreatePHI(S.Ty, I.getNumIncomingValues());

    for (const auto &[Idx, BB] : enumerate(I.blocks())) {
      S.NewPHI->addIncoming(S.getSlicedVal(BB, I.getIncomingValue(Idx),
                                           "largephi.extractslice" +
                                               std::to_string(IncNameSuffix++)),
                            BB);
    }
  }

  // And replace this PHI with a vector of all the previous PHI values.
  Value *Vec = PoisonValue::get(FVT);
  unsigned NameSuffix = 0;
  for (VectorSlice &S : Slices) {
    const auto ValName = "largephi.insertslice" + std::to_string(NameSuffix++);
    if (S.NumElts > 1)
      Vec = B.CreateInsertVector(FVT, Vec, S.NewPHI, S.Idx, ValName);
    else
      Vec = B.CreateInsertElement(Vec, S.NewPHI, S.Idx, ValName);
  }

  I.replaceAllUsesWith(Vec);
  I.eraseFromParent();
  return true;
}

/// \param V  Value to check
/// \param DL DataLayout
/// \param TM TargetMachine (TODO: remove once DL contains nullptr values)
/// \param AS Target Address Space
/// \return true if \p V cannot be the null value of \p AS, false otherwise.
static bool isPtrKnownNeverNull(const Value *V, const DataLayout &DL,
                                const AMDGPUTargetMachine &TM, unsigned AS) {
  // Pointer cannot be null if it's a block address, GV or alloca.
  // NOTE: We don't support extern_weak, but if we did, we'd need to check for
  // it as the symbol could be null in such cases.
  if (isa<BlockAddress, GlobalValue, AllocaInst>(V))
    return true;

  // Check nonnull arguments.
  if (const auto *Arg = dyn_cast<Argument>(V); Arg && Arg->hasNonNullAttr())
    return true;

  // Check nonnull loads.
  if (const auto *Load = dyn_cast<LoadInst>(V);
      Load && Load->hasMetadata(LLVMContext::MD_nonnull))
    return true;

  // getUnderlyingObject may have looked through another addrspacecast, although
  // the optimizable situations most likely folded out by now.
  if (AS != cast<PointerType>(V->getType())->getAddressSpace())
    return false;

  // TODO: Calls that return nonnull?

  // For all other things, use KnownBits.
  // We either use 0 or all bits set to indicate null, so check whether the
  // value can be zero or all ones.
  //
  // TODO: Use ValueTracking's isKnownNeverNull if it becomes aware that some
  // address spaces have non-zero null values.
  auto SrcPtrKB = computeKnownBits(V, DL);
  const auto NullVal = TM.getNullPointerValue(AS);

  assert(SrcPtrKB.getBitWidth() == DL.getPointerSizeInBits(AS));
  assert((NullVal == 0 || NullVal == -1) &&
         "don't know how to check for this null value!");
  return NullVal ? !SrcPtrKB.getMaxValue().isAllOnes() : SrcPtrKB.isNonZero();
}

bool AMDGPUCodeGenPrepareImpl::visitAddrSpaceCastInst(AddrSpaceCastInst &I) {
  // Intrinsic doesn't support vectors, also it seems that it's often difficult
  // to prove that a vector cannot have any nulls in it so it's unclear if it's
  // worth supporting.
  if (I.getType()->isVectorTy())
    return false;

  // Check if this can be lowered to a amdgcn.addrspacecast.nonnull.
  // This is only worthwhile for casts from/to priv/local to flat.
  const unsigned SrcAS = I.getSrcAddressSpace();
  const unsigned DstAS = I.getDestAddressSpace();

  bool CanLower = false;
  if (SrcAS == AMDGPUAS::FLAT_ADDRESS)
    CanLower = (DstAS == AMDGPUAS::LOCAL_ADDRESS ||
                DstAS == AMDGPUAS::PRIVATE_ADDRESS);
  else if (DstAS == AMDGPUAS::FLAT_ADDRESS)
    CanLower = (SrcAS == AMDGPUAS::LOCAL_ADDRESS ||
                SrcAS == AMDGPUAS::PRIVATE_ADDRESS);
  if (!CanLower)
    return false;

  SmallVector<const Value *, 4> WorkList;
  getUnderlyingObjects(I.getOperand(0), WorkList);
  if (!all_of(WorkList, [&](const Value *V) {
        return isPtrKnownNeverNull(V, DL, TM, SrcAS);
      }))
    return false;

  IRBuilder<> B(&I);
  auto *Intrin = B.CreateIntrinsic(
      I.getType(), Intrinsic::amdgcn_addrspacecast_nonnull, {I.getOperand(0)});
  I.replaceAllUsesWith(Intrin);
  I.eraseFromParent();
  return true;
}

bool AMDGPUCodeGenPrepareImpl::visitIntrinsicInst(IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
  case Intrinsic::minnum:
  case Intrinsic::minimumnum:
  case Intrinsic::minimum:
    return visitFMinLike(I);
  case Intrinsic::sqrt:
    return visitSqrt(I);
  default:
    return false;
  }
}

/// Match non-nan fract pattern.
///   minnum(fsub(x, floor(x)), nextafter(1.0, -1.0))
///   minimumnum(fsub(x, floor(x)), nextafter(1.0, -1.0))
///   minimum(fsub(x, floor(x)), nextafter(1.0, -1.0))
///
/// If fract is a useful instruction for the subtarget. Does not account for the
/// nan handling; the instruction has a nan check on the input value.
Value *AMDGPUCodeGenPrepareImpl::matchFractPat(IntrinsicInst &I) {
  if (ST.hasFractBug())
    return nullptr;

  Intrinsic::ID IID = I.getIntrinsicID();

  // The value is only used in contexts where we know the input isn't a nan, so
  // any of the fmin variants are fine.
  if (IID != Intrinsic::minnum && IID != Intrinsic::minimum &&
      IID != Intrinsic::minimumnum)
    return nullptr;

  Type *Ty = I.getType();
  if (!isLegalFloatingTy(Ty->getScalarType()))
    return nullptr;

  Value *Arg0 = I.getArgOperand(0);
  Value *Arg1 = I.getArgOperand(1);

  const APFloat *C;
  if (!match(Arg1, m_APFloat(C)))
    return nullptr;

  APFloat One(1.0);
  bool LosesInfo;
  One.convert(C->getSemantics(), APFloat::rmNearestTiesToEven, &LosesInfo);

  // Match nextafter(1.0, -1)
  One.next(true);
  if (One != *C)
    return nullptr;

  Value *FloorSrc;
  if (match(Arg0, m_FSub(m_Value(FloorSrc),
                         m_Intrinsic<Intrinsic::floor>(m_Deferred(FloorSrc)))))
    return FloorSrc;
  return nullptr;
}

Value *AMDGPUCodeGenPrepareImpl::applyFractPat(IRBuilder<> &Builder,
                                               Value *FractArg) {
  SmallVector<Value *, 4> FractVals;
  extractValues(Builder, FractVals, FractArg);

  SmallVector<Value *, 4> ResultVals(FractVals.size());

  Type *Ty = FractArg->getType()->getScalarType();
  for (unsigned I = 0, E = FractVals.size(); I != E; ++I) {
    ResultVals[I] =
        Builder.CreateIntrinsic(Intrinsic::amdgcn_fract, {Ty}, {FractVals[I]});
  }

  return insertValues(Builder, FractArg->getType(), ResultVals);
}

bool AMDGPUCodeGenPrepareImpl::visitFMinLike(IntrinsicInst &I) {
  Value *FractArg = matchFractPat(I);
  if (!FractArg)
    return false;

  // Match pattern for fract intrinsic in contexts where the nan check has been
  // optimized out (and hope the knowledge the source can't be nan wasn't lost).
  if (!I.hasNoNaNs() && !isKnownNeverNaN(FractArg, SimplifyQuery(DL, TLI)))
    return false;

  IRBuilder<> Builder(&I);
  FastMathFlags FMF = I.getFastMathFlags();
  FMF.setNoNaNs();
  Builder.setFastMathFlags(FMF);

  Value *Fract = applyFractPat(Builder, FractArg);
  Fract->takeName(&I);
  I.replaceAllUsesWith(Fract);

  RecursivelyDeleteTriviallyDeadInstructions(&I, TLI);
  return true;
}

static bool isOneOrNegOne(const Value *Val) {
  const APFloat *C;
  return match(Val, m_APFloat(C)) && C->getExactLog2Abs() == 0;
}

// Expand llvm.sqrt.f32 calls with !fpmath metadata in a semi-fast way.
bool AMDGPUCodeGenPrepareImpl::visitSqrt(IntrinsicInst &Sqrt) {
  Type *Ty = Sqrt.getType()->getScalarType();
  if (!Ty->isFloatTy() && (!Ty->isHalfTy() || ST.has16BitInsts()))
    return false;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&Sqrt);
  FastMathFlags SqrtFMF = FPOp->getFastMathFlags();

  // We're trying to handle the fast-but-not-that-fast case only. The lowering
  // of fast llvm.sqrt will give the raw instruction anyway.
  if (SqrtFMF.approxFunc())
    return false;

  const float ReqdAccuracy = FPOp->getFPAccuracy();

  // Defer correctly rounded expansion to codegen.
  if (ReqdAccuracy < 1.0f)
    return false;

  // FIXME: This is an ugly hack for this pass using forward iteration instead
  // of reverse. If it worked like a normal combiner, the rsq would form before
  // we saw a sqrt call.
  auto *FDiv =
      dyn_cast_or_null<FPMathOperator>(Sqrt.getUniqueUndroppableUser());
  if (FDiv && FDiv->getOpcode() == Instruction::FDiv &&
      FDiv->getFPAccuracy() >= 1.0f &&
      canOptimizeWithRsq(FPOp, FDiv->getFastMathFlags(), SqrtFMF) &&
      // TODO: We should also handle the arcp case for the fdiv with non-1 value
      isOneOrNegOne(FDiv->getOperand(0)))
    return false;

  Value *SrcVal = Sqrt.getOperand(0);
  bool CanTreatAsDAZ = canIgnoreDenormalInput(SrcVal, &Sqrt);

  // The raw instruction is 1 ulp, but the correction for denormal handling
  // brings it to 2.
  if (!CanTreatAsDAZ && ReqdAccuracy < 2.0f)
    return false;

  IRBuilder<> Builder(&Sqrt);
  SmallVector<Value *, 4> SrcVals;
  extractValues(Builder, SrcVals, SrcVal);

  SmallVector<Value *, 4> ResultVals(SrcVals.size());
  for (int I = 0, E = SrcVals.size(); I != E; ++I) {
    if (CanTreatAsDAZ)
      ResultVals[I] = Builder.CreateCall(getSqrtF32(), SrcVals[I]);
    else
      ResultVals[I] = emitSqrtIEEE2ULP(Builder, SrcVals[I], SqrtFMF);
  }

  Value *NewSqrt = insertValues(Builder, Sqrt.getType(), ResultVals);
  NewSqrt->takeName(&Sqrt);
  Sqrt.replaceAllUsesWith(NewSqrt);
  Sqrt.eraseFromParent();
  return true;
}

bool AMDGPUCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  const AMDGPUTargetMachine &TM = TPC->getTM<AMDGPUTargetMachine>();
  const TargetLibraryInfo *TLI =
      &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  AssumptionCache *AC =
      &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  const DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  const UniformityInfo &UA =
      getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
  return AMDGPUCodeGenPrepareImpl(F, TM, TLI, AC, DT, UA).run();
}

PreservedAnalyses AMDGPUCodeGenPreparePass::run(Function &F,
                                                FunctionAnalysisManager &FAM) {
  const AMDGPUTargetMachine &ATM = static_cast<const AMDGPUTargetMachine &>(TM);
  const TargetLibraryInfo *TLI = &FAM.getResult<TargetLibraryAnalysis>(F);
  AssumptionCache *AC = &FAM.getResult<AssumptionAnalysis>(F);
  const DominatorTree *DT = FAM.getCachedResult<DominatorTreeAnalysis>(F);
  const UniformityInfo &UA = FAM.getResult<UniformityInfoAnalysis>(F);
  AMDGPUCodeGenPrepareImpl Impl(F, ATM, TLI, AC, DT, UA);
  if (!Impl.run())
    return PreservedAnalyses::all();
  PreservedAnalyses PA = PreservedAnalyses::none();
  if (!Impl.FlowChanged)
    PA.preserveSet<CFGAnalyses>();
  return PA;
}

INITIALIZE_PASS_BEGIN(AMDGPUCodeGenPrepare, DEBUG_TYPE,
                      "AMDGPU IR optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUCodeGenPrepare, DEBUG_TYPE, "AMDGPU IR optimizations",
                    false, false)

char AMDGPUCodeGenPrepare::ID = 0;

FunctionPass *llvm::createAMDGPUCodeGenPreparePass() {
  return new AMDGPUCodeGenPrepare();
}
