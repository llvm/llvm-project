//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. This analysis uses the services of the codegen
// to approximate the cost of any IR instruction when lowered to machine
// instructions. The cost results are unit-less and the cost number represents
// the throughput of the machine assuming that all loads hit the cache, all
// branches are predicted, etc. The cost numbers can be added in order to
// compare two or more transformation alternatives.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace PatternMatch;

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME

static cl::opt<bool> EnableReduxCost("costmodel-reduxcost", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Recognize reduction patterns."));

namespace {
  class CostModelAnalysis : public FunctionPass {

  public:
    static char ID; // Class identification, replacement for typeinfo
    CostModelAnalysis() : FunctionPass(ID), F(nullptr), TTI(nullptr) {
      initializeCostModelAnalysisPass(
        *PassRegistry::getPassRegistry());
    }

    /// Returns the expected cost of the instruction.
    /// Returns -1 if the cost is unknown.
    /// Note, this method does not cache the cost calculation and it
    /// can be expensive in some cases.
    unsigned getInstructionCost(const Instruction *I) const;

  private:
    void getAnalysisUsage(AnalysisUsage &AU) const override;
    bool runOnFunction(Function &F) override;
    void print(raw_ostream &OS, const Module*) const override;

    /// The function that we analyze.
    Function *F;
    /// Target information.
    const TargetTransformInfo *TTI;
  };
}  // End of anonymous namespace

// Register this pass.
char CostModelAnalysis::ID = 0;
static const char cm_name[] = "Cost Model Analysis";
INITIALIZE_PASS_BEGIN(CostModelAnalysis, CM_NAME, cm_name, false, true)
INITIALIZE_PASS_END  (CostModelAnalysis, CM_NAME, cm_name, false, true)

FunctionPass *llvm::createCostModelAnalysisPass() {
  return new CostModelAnalysis();
}

void
CostModelAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool
CostModelAnalysis::runOnFunction(Function &F) {
 this->F = &F;
 auto *TTIWP = getAnalysisIfAvailable<TargetTransformInfoWrapperPass>();
 TTI = TTIWP ? &TTIWP->getTTI(F) : nullptr;

 return false;
}

static bool isReverseVectorMask(ArrayRef<int> Mask) {
  for (unsigned i = 0, MaskSize = Mask.size(); i < MaskSize; ++i)
    if (Mask[i] >= 0 && Mask[i] != (int)(MaskSize - 1 - i))
      return false;
  return true;
}

static bool isSingleSourceVectorMask(ArrayRef<int> Mask) {
  bool Vec0 = false;
  bool Vec1 = false;
  for (unsigned i = 0, NumVecElts = Mask.size(); i < NumVecElts; ++i) {
    if (Mask[i] >= 0) {
      if ((unsigned)Mask[i] >= NumVecElts)
        Vec1 = true;
      else
        Vec0 = true;
    }
  }
  return !(Vec0 && Vec1);
}

static bool isZeroEltBroadcastVectorMask(ArrayRef<int> Mask) {
  for (unsigned i = 0; i < Mask.size(); ++i)
    if (Mask[i] > 0)
      return false;
  return true;
}

static bool isAlternateVectorMask(ArrayRef<int> Mask) {
  bool isAlternate = true;
  unsigned MaskSize = Mask.size();

  // Example: shufflevector A, B, <0,5,2,7>
  for (unsigned i = 0; i < MaskSize && isAlternate; ++i) {
    if (Mask[i] < 0)
      continue;
    isAlternate = Mask[i] == (int)((i & 1) ? MaskSize + i : i);
  }

  if (isAlternate)
    return true;

  isAlternate = true;
  // Example: shufflevector A, B, <4,1,6,3>
  for (unsigned i = 0; i < MaskSize && isAlternate; ++i) {
    if (Mask[i] < 0)
      continue;
    isAlternate = Mask[i] == (int)((i & 1) ? i : MaskSize + i);
  }

  return isAlternate;
}

static TargetTransformInfo::OperandValueKind getOperandInfo(Value *V) {
  TargetTransformInfo::OperandValueKind OpInfo =
      TargetTransformInfo::OK_AnyValue;

  // Check for a splat of a constant or for a non uniform vector of constants.
  if (isa<ConstantVector>(V) || isa<ConstantDataVector>(V)) {
    OpInfo = TargetTransformInfo::OK_NonUniformConstantValue;
    if (cast<Constant>(V)->getSplatValue() != nullptr)
      OpInfo = TargetTransformInfo::OK_UniformConstantValue;
  }

  // Check for a splat of a uniform value. This is not loop aware, so return
  // true only for the obviously uniform cases (argument, globalvalue)
  const Value *Splat = getSplatValue(V);
  if (Splat && (isa<Argument>(Splat) || isa<GlobalValue>(Splat)))
    OpInfo = TargetTransformInfo::OK_UniformValue;

  return OpInfo;
}

static bool matchPairwiseShuffleMask(ShuffleVectorInst *SI, bool IsLeft,
                                     unsigned Level) {
  // We don't need a shuffle if we just want to have element 0 in position 0 of
  // the vector.
  if (!SI && Level == 0 && IsLeft)
    return true;
  else if (!SI)
    return false;

  SmallVector<int, 32> Mask(SI->getType()->getVectorNumElements(), -1);

  // Build a mask of 0, 2, ... (left) or 1, 3, ... (right) depending on whether
  // we look at the left or right side.
  for (unsigned i = 0, e = (1 << Level), val = !IsLeft; i != e; ++i, val += 2)
    Mask[i] = val;

  SmallVector<int, 16> ActualMask = SI->getShuffleMask();
  return Mask == ActualMask;
}

namespace {
/// Kind of the reduction data.
enum ReductionKind {
  RK_None,           /// Not a reduction.
  RK_Arithmetic,     /// Binary reduction data.
  RK_MinMax,         /// Min/max reduction data.
  RK_UnsignedMinMax, /// Unsigned min/max reduction data.
};
/// Contains opcode + LHS/RHS parts of the reduction operations.
struct ReductionData {
  ReductionData() = delete;
  ReductionData(ReductionKind Kind, unsigned Opcode, Value *LHS, Value *RHS)
      : Opcode(Opcode), LHS(LHS), RHS(RHS), Kind(Kind) {
    assert(Kind != RK_None && "expected binary or min/max reduction only.");
  }
  unsigned Opcode = 0;
  Value *LHS = nullptr;
  Value *RHS = nullptr;
  ReductionKind Kind = RK_None;
  bool hasSameData(ReductionData &RD) const {
    return Kind == RD.Kind && Opcode == RD.Opcode;
  }
};
} // namespace

static Optional<ReductionData> getReductionData(Instruction *I) {
  Value *L, *R;
  if (m_BinOp(m_Value(L), m_Value(R)).match(I))
    return ReductionData(RK_Arithmetic, I->getOpcode(), L, R);
  if (auto *SI = dyn_cast<SelectInst>(I)) {
    if (m_SMin(m_Value(L), m_Value(R)).match(SI) ||
        m_SMax(m_Value(L), m_Value(R)).match(SI) ||
        m_OrdFMin(m_Value(L), m_Value(R)).match(SI) ||
        m_OrdFMax(m_Value(L), m_Value(R)).match(SI) ||
        m_UnordFMin(m_Value(L), m_Value(R)).match(SI) ||
        m_UnordFMax(m_Value(L), m_Value(R)).match(SI)) {
      auto *CI = cast<CmpInst>(SI->getCondition());
      return ReductionData(RK_MinMax, CI->getOpcode(), L, R);
    }
    if (m_UMin(m_Value(L), m_Value(R)).match(SI) ||
        m_UMax(m_Value(L), m_Value(R)).match(SI)) {
      auto *CI = cast<CmpInst>(SI->getCondition());
      return ReductionData(RK_UnsignedMinMax, CI->getOpcode(), L, R);
    }
  }
  return llvm::None;
}

static ReductionKind matchPairwiseReductionAtLevel(Instruction *I,
                                                   unsigned Level,
                                                   unsigned NumLevels) {
  // Match one level of pairwise operations.
  // %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  // %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  // %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  if (!I)
    return RK_None;

  assert(I->getType()->isVectorTy() && "Expecting a vector type");

  Optional<ReductionData> RD = getReductionData(I);
  if (!RD)
    return RK_None;

  ShuffleVectorInst *LS = dyn_cast<ShuffleVectorInst>(RD->LHS);
  if (!LS && Level)
    return RK_None;
  ShuffleVectorInst *RS = dyn_cast<ShuffleVectorInst>(RD->RHS);
  if (!RS && Level)
    return RK_None;

  // On level 0 we can omit one shufflevector instruction.
  if (!Level && !RS && !LS)
    return RK_None;

  // Shuffle inputs must match.
  Value *NextLevelOpL = LS ? LS->getOperand(0) : nullptr;
  Value *NextLevelOpR = RS ? RS->getOperand(0) : nullptr;
  Value *NextLevelOp = nullptr;
  if (NextLevelOpR && NextLevelOpL) {
    // If we have two shuffles their operands must match.
    if (NextLevelOpL != NextLevelOpR)
      return RK_None;

    NextLevelOp = NextLevelOpL;
  } else if (Level == 0 && (NextLevelOpR || NextLevelOpL)) {
    // On the first level we can omit the shufflevector <0, undef,...>. So the
    // input to the other shufflevector <1, undef> must match with one of the
    // inputs to the current binary operation.
    // Example:
    //  %NextLevelOpL = shufflevector %R, <1, undef ...>
    //  %BinOp        = fadd          %NextLevelOpL, %R
    if (NextLevelOpL && NextLevelOpL != RD->RHS)
      return RK_None;
    else if (NextLevelOpR && NextLevelOpR != RD->LHS)
      return RK_None;

    NextLevelOp = NextLevelOpL ? RD->RHS : RD->LHS;
  } else {
    return RK_None;
  }

  // Check that the next levels binary operation exists and matches with the
  // current one.
  if (Level + 1 != NumLevels) {
    Optional<ReductionData> NextLevelRD =
        getReductionData(cast<Instruction>(NextLevelOp));
    if (!NextLevelRD || !RD->hasSameData(*NextLevelRD))
      return RK_None;
  }

  // Shuffle mask for pairwise operation must match.
  if (matchPairwiseShuffleMask(LS, /*IsLeft=*/true, Level)) {
    if (!matchPairwiseShuffleMask(RS, /*IsLeft=*/false, Level))
      return RK_None;
  } else if (matchPairwiseShuffleMask(RS, /*IsLeft=*/true, Level)) {
    if (!matchPairwiseShuffleMask(LS, /*IsLeft=*/false, Level))
      return RK_None;
  } else {
    return RK_None;
  }

  if (++Level == NumLevels)
    return RD->Kind;

  // Match next level.
  return matchPairwiseReductionAtLevel(cast<Instruction>(NextLevelOp), Level,
                                       NumLevels);
}

static ReductionKind matchPairwiseReduction(const ExtractElementInst *ReduxRoot,
                                            unsigned &Opcode, Type *&Ty) {
  if (!EnableReduxCost)
    return RK_None;

  // Need to extract the first element.
  ConstantInt *CI = dyn_cast<ConstantInt>(ReduxRoot->getOperand(1));
  unsigned Idx = ~0u;
  if (CI)
    Idx = CI->getZExtValue();
  if (Idx != 0)
    return RK_None;

  auto *RdxStart = dyn_cast<Instruction>(ReduxRoot->getOperand(0));
  if (!RdxStart)
    return RK_None;
  Optional<ReductionData> RD = getReductionData(RdxStart);
  if (!RD)
    return RK_None;

  Type *VecTy = RdxStart->getType();
  unsigned NumVecElems = VecTy->getVectorNumElements();
  if (!isPowerOf2_32(NumVecElems))
    return RK_None;

  // We look for a sequence of shuffle,shuffle,add triples like the following
  // that builds a pairwise reduction tree.
  //
  //  (X0, X1, X2, X3)
  //   (X0 + X1, X2 + X3, undef, undef)
  //    ((X0 + X1) + (X2 + X3), undef, undef, undef)
  //
  // %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  // %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  // %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  // %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  // %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  // %bin.rdx8 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1
  // %r = extractelement <4 x float> %bin.rdx8, i32 0
  if (matchPairwiseReductionAtLevel(RdxStart, 0, Log2_32(NumVecElems)) ==
      RK_None)
    return RK_None;

  Opcode = RD->Opcode;
  Ty = VecTy;

  return RD->Kind;
}

static std::pair<Value *, ShuffleVectorInst *>
getShuffleAndOtherOprd(Value *L, Value *R) {
  ShuffleVectorInst *S = nullptr;

  if ((S = dyn_cast<ShuffleVectorInst>(L)))
    return std::make_pair(R, S);

  S = dyn_cast<ShuffleVectorInst>(R);
  return std::make_pair(L, S);
}

static ReductionKind
matchVectorSplittingReduction(const ExtractElementInst *ReduxRoot,
                              unsigned &Opcode, Type *&Ty) {
  if (!EnableReduxCost)
    return RK_None;

  // Need to extract the first element.
  ConstantInt *CI = dyn_cast<ConstantInt>(ReduxRoot->getOperand(1));
  unsigned Idx = ~0u;
  if (CI)
    Idx = CI->getZExtValue();
  if (Idx != 0)
    return RK_None;

  auto *RdxStart = dyn_cast<Instruction>(ReduxRoot->getOperand(0));
  if (!RdxStart)
    return RK_None;
  Optional<ReductionData> RD = getReductionData(RdxStart);
  if (!RD)
    return RK_None;

  Type *VecTy = ReduxRoot->getOperand(0)->getType();
  unsigned NumVecElems = VecTy->getVectorNumElements();
  if (!isPowerOf2_32(NumVecElems))
    return RK_None;

  // We look for a sequence of shuffles and adds like the following matching one
  // fadd, shuffle vector pair at a time.
  //
  // %rdx.shuf = shufflevector <4 x float> %rdx, <4 x float> undef,
  //                           <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  // %bin.rdx = fadd <4 x float> %rdx, %rdx.shuf
  // %rdx.shuf7 = shufflevector <4 x float> %bin.rdx, <4 x float> undef,
  //                          <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  // %bin.rdx8 = fadd <4 x float> %bin.rdx, %rdx.shuf7
  // %r = extractelement <4 x float> %bin.rdx8, i32 0

  unsigned MaskStart = 1;
  Instruction *RdxOp = RdxStart;
  SmallVector<int, 32> ShuffleMask(NumVecElems, 0);
  unsigned NumVecElemsRemain = NumVecElems;
  while (NumVecElemsRemain - 1) {
    // Check for the right reduction operation.
    if (!RdxOp)
      return RK_None;
    Optional<ReductionData> RDLevel = getReductionData(RdxOp);
    if (!RDLevel || !RDLevel->hasSameData(*RD))
      return RK_None;

    Value *NextRdxOp;
    ShuffleVectorInst *Shuffle;
    std::tie(NextRdxOp, Shuffle) =
        getShuffleAndOtherOprd(RDLevel->LHS, RDLevel->RHS);

    // Check the current reduction operation and the shuffle use the same value.
    if (Shuffle == nullptr)
      return RK_None;
    if (Shuffle->getOperand(0) != NextRdxOp)
      return RK_None;

    // Check that shuffle masks matches.
    for (unsigned j = 0; j != MaskStart; ++j)
      ShuffleMask[j] = MaskStart + j;
    // Fill the rest of the mask with -1 for undef.
    std::fill(&ShuffleMask[MaskStart], ShuffleMask.end(), -1);

    SmallVector<int, 16> Mask = Shuffle->getShuffleMask();
    if (ShuffleMask != Mask)
      return RK_None;

    RdxOp = dyn_cast<Instruction>(NextRdxOp);
    NumVecElemsRemain /= 2;
    MaskStart *= 2;
  }

  Opcode = RD->Opcode;
  Ty = VecTy;
  return RD->Kind;
}

unsigned CostModelAnalysis::getInstructionCost(const Instruction *I) const {
  if (!TTI)
    return -1;

  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
    return TTI->getUserCost(I);

  case Instruction::Ret:
  case Instruction::PHI:
  case Instruction::Br: {
    return TTI->getCFInstrCost(I->getOpcode());
  }
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    TargetTransformInfo::OperandValueKind Op1VK =
      getOperandInfo(I->getOperand(0));
    TargetTransformInfo::OperandValueKind Op2VK =
      getOperandInfo(I->getOperand(1));
    SmallVector<const Value*, 2> Operands(I->operand_values()); 
    return TTI->getArithmeticInstrCost(I->getOpcode(), I->getType(), Op1VK,
                                       Op2VK, TargetTransformInfo::OP_None, 
                                       TargetTransformInfo::OP_None, 
                                       Operands);
  }
  case Instruction::Select: {
    const SelectInst *SI = cast<SelectInst>(I);
    Type *CondTy = SI->getCondition()->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), I->getType(), CondTy, I);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), ValTy, I->getType(), I);
  }
  case Instruction::Store: {
    const StoreInst *SI = cast<StoreInst>(I);
    Type *ValTy = SI->getValueOperand()->getType();
    return TTI->getMemoryOpCost(I->getOpcode(), ValTy,
                                SI->getAlignment(),
                                SI->getPointerAddressSpace(), I);
  }
  case Instruction::Load: {
    const LoadInst *LI = cast<LoadInst>(I);
    return TTI->getMemoryOpCost(I->getOpcode(), I->getType(),
                                LI->getAlignment(),
                                LI->getPointerAddressSpace(), I);
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    return TTI->getCastInstrCost(I->getOpcode(), I->getType(), SrcTy, I);
  }
  case Instruction::ExtractElement: {
    const ExtractElementInst * EEI = cast<ExtractElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();

    // Try to match a reduction sequence (series of shufflevector and vector
    // adds followed by a extractelement).
    unsigned ReduxOpCode;
    Type *ReduxType;

    switch (matchVectorSplittingReduction(EEI, ReduxOpCode, ReduxType)) {
    case RK_Arithmetic:
      return TTI->getArithmeticReductionCost(ReduxOpCode, ReduxType,
                                             /*IsPairwiseForm=*/false);
    case RK_MinMax:
      return TTI->getMinMaxReductionCost(
          ReduxType, CmpInst::makeCmpResultType(ReduxType),
          /*IsPairwiseForm=*/false, /*IsUnsigned=*/false);
    case RK_UnsignedMinMax:
      return TTI->getMinMaxReductionCost(
          ReduxType, CmpInst::makeCmpResultType(ReduxType),
          /*IsPairwiseForm=*/false, /*IsUnsigned=*/true);
    case RK_None:
      break;
    }

    switch (matchPairwiseReduction(EEI, ReduxOpCode, ReduxType)) {
    case RK_Arithmetic:
      return TTI->getArithmeticReductionCost(ReduxOpCode, ReduxType,
                                             /*IsPairwiseForm=*/true);
    case RK_MinMax:
      return TTI->getMinMaxReductionCost(
          ReduxType, CmpInst::makeCmpResultType(ReduxType),
          /*IsPairwiseForm=*/true, /*IsUnsigned=*/false);
    case RK_UnsignedMinMax:
      return TTI->getMinMaxReductionCost(
          ReduxType, CmpInst::makeCmpResultType(ReduxType),
          /*IsPairwiseForm=*/true, /*IsUnsigned=*/true);
    case RK_None:
      break;
    }

    return TTI->getVectorInstrCost(I->getOpcode(),
                                   EEI->getOperand(0)->getType(), Idx);
  }
  case Instruction::InsertElement: {
    const InsertElementInst * IE = cast<InsertElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();
    return TTI->getVectorInstrCost(I->getOpcode(),
                                   IE->getType(), Idx);
  }
  case Instruction::ShuffleVector: {
    const ShuffleVectorInst *Shuffle = cast<ShuffleVectorInst>(I);
    Type *VecTypOp0 = Shuffle->getOperand(0)->getType();
    unsigned NumVecElems = VecTypOp0->getVectorNumElements();
    SmallVector<int, 16> Mask = Shuffle->getShuffleMask();

    if (NumVecElems == Mask.size()) {
      if (isReverseVectorMask(Mask))
        return TTI->getShuffleCost(TargetTransformInfo::SK_Reverse, VecTypOp0,
                                   0, nullptr);
      if (isAlternateVectorMask(Mask))
        return TTI->getShuffleCost(TargetTransformInfo::SK_Alternate,
                                   VecTypOp0, 0, nullptr);

      if (isZeroEltBroadcastVectorMask(Mask))
        return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast,
                                   VecTypOp0, 0, nullptr);

      if (isSingleSourceVectorMask(Mask))
        return TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                   VecTypOp0, 0, nullptr);

      return TTI->getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc,
                                 VecTypOp0, 0, nullptr);
    }

    return -1;
  }
  case Instruction::Call:
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      SmallVector<Value *, 4> Args(II->arg_operands());

      FastMathFlags FMF;
      if (auto *FPMO = dyn_cast<FPMathOperator>(II))
        FMF = FPMO->getFastMathFlags();

      return TTI->getIntrinsicInstrCost(II->getIntrinsicID(), II->getType(),
                                        Args, FMF);
    }
    return -1;
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

void CostModelAnalysis::print(raw_ostream &OS, const Module*) const {
  if (!F)
    return;

  for (BasicBlock &B : *F) {
    for (Instruction &Inst : B) {
      unsigned Cost = getInstructionCost(&Inst);
      if (Cost != (unsigned)-1)
        OS << "Cost Model: Found an estimated cost of " << Cost;
      else
        OS << "Cost Model: Unknown cost";

      OS << " for instruction: " << Inst << "\n";
    }
  }
}
