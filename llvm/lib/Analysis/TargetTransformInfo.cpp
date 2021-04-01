//===- llvm/Analysis/TargetTransformInfo.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <utility>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "tti"

static cl::opt<bool> EnableReduxCost("costmodel-reduxcost", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Recognize reduction patterns."));

namespace {
/// No-op implementation of the TTI interface using the utility base
/// classes.
///
/// This is used when no target specific information is available.
struct NoTTIImpl : TargetTransformInfoImplCRTPBase<NoTTIImpl> {
  explicit NoTTIImpl(const DataLayout &DL)
      : TargetTransformInfoImplCRTPBase<NoTTIImpl>(DL) {}
};
} // namespace

bool HardwareLoopInfo::canAnalyze(LoopInfo &LI) {
  // If the loop has irreducible control flow, it can not be converted to
  // Hardware loop.
  LoopBlocksRPO RPOT(L);
  RPOT.perform(&LI);
  if (containsIrreducibleCFG<const BasicBlock *>(RPOT, LI))
    return false;
  return true;
}

IntrinsicCostAttributes::IntrinsicCostAttributes(Intrinsic::ID Id,
                                                 const CallBase &CI,
                                                 unsigned ScalarizationCost)
    : II(dyn_cast<IntrinsicInst>(&CI)), RetTy(CI.getType()), IID(Id),
      ScalarizationCost(ScalarizationCost) {

  if (const auto *FPMO = dyn_cast<FPMathOperator>(&CI))
    FMF = FPMO->getFastMathFlags();

  Arguments.insert(Arguments.begin(), CI.arg_begin(), CI.arg_end());
  FunctionType *FTy = CI.getCalledFunction()->getFunctionType();
  ParamTys.insert(ParamTys.begin(), FTy->param_begin(), FTy->param_end());
}

IntrinsicCostAttributes::IntrinsicCostAttributes(Intrinsic::ID Id, Type *RTy,
                                                 ArrayRef<Type *> Tys,
                                                 FastMathFlags Flags,
                                                 const IntrinsicInst *I,
                                                 unsigned ScalarCost)
    : II(I), RetTy(RTy), IID(Id), FMF(Flags), ScalarizationCost(ScalarCost) {
  ParamTys.insert(ParamTys.begin(), Tys.begin(), Tys.end());
}

IntrinsicCostAttributes::IntrinsicCostAttributes(Intrinsic::ID Id, Type *Ty,
                                                 ArrayRef<const Value *> Args)
    : RetTy(Ty), IID(Id) {

  Arguments.insert(Arguments.begin(), Args.begin(), Args.end());
  ParamTys.reserve(Arguments.size());
  for (unsigned Idx = 0, Size = Arguments.size(); Idx != Size; ++Idx)
    ParamTys.push_back(Arguments[Idx]->getType());
}

IntrinsicCostAttributes::IntrinsicCostAttributes(Intrinsic::ID Id, Type *RTy,
                                                 ArrayRef<const Value *> Args,
                                                 ArrayRef<Type *> Tys,
                                                 FastMathFlags Flags,
                                                 const IntrinsicInst *I,
                                                 unsigned ScalarCost)
    : II(I), RetTy(RTy), IID(Id), FMF(Flags), ScalarizationCost(ScalarCost) {
  ParamTys.insert(ParamTys.begin(), Tys.begin(), Tys.end());
  Arguments.insert(Arguments.begin(), Args.begin(), Args.end());
}

bool HardwareLoopInfo::isHardwareLoopCandidate(ScalarEvolution &SE,
                                               LoopInfo &LI, DominatorTree &DT,
                                               bool ForceNestedLoop,
                                               bool ForceHardwareLoopPHI) {
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  for (BasicBlock *BB : ExitingBlocks) {
    // If we pass the updated counter back through a phi, we need to know
    // which latch the updated value will be coming from.
    if (!L->isLoopLatch(BB)) {
      if (ForceHardwareLoopPHI || CounterInReg)
        continue;
    }

    const SCEV *EC = SE.getExitCount(L, BB);
    if (isa<SCEVCouldNotCompute>(EC))
      continue;
    if (const SCEVConstant *ConstEC = dyn_cast<SCEVConstant>(EC)) {
      if (ConstEC->getValue()->isZero())
        continue;
    } else if (!SE.isLoopInvariant(EC, L))
      continue;

    if (SE.getTypeSizeInBits(EC->getType()) > CountType->getBitWidth())
      continue;

    // If this exiting block is contained in a nested loop, it is not eligible
    // for insertion of the branch-and-decrement since the inner loop would
    // end up messing up the value in the CTR.
    if (!IsNestingLegal && LI.getLoopFor(BB) != L && !ForceNestedLoop)
      continue;

    // We now have a loop-invariant count of loop iterations (which is not the
    // constant zero) for which we know that this loop will not exit via this
    // existing block.

    // We need to make sure that this block will run on every loop iteration.
    // For this to be true, we must dominate all blocks with backedges. Such
    // blocks are in-loop predecessors to the header block.
    bool NotAlways = false;
    for (BasicBlock *Pred : predecessors(L->getHeader())) {
      if (!L->contains(Pred))
        continue;

      if (!DT.dominates(BB, Pred)) {
        NotAlways = true;
        break;
      }
    }

    if (NotAlways)
      continue;

    // Make sure this blocks ends with a conditional branch.
    Instruction *TI = BB->getTerminator();
    if (!TI)
      continue;

    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional())
        continue;

      ExitBranch = BI;
    } else
      continue;

    // Note that this block may not be the loop latch block, even if the loop
    // has a latch block.
    ExitBlock = BB;
    TripCount = SE.getAddExpr(EC, SE.getOne(EC->getType()));

    if (!EC->getType()->isPointerTy() && EC->getType() != CountType)
      TripCount = SE.getZeroExtendExpr(TripCount, CountType);

    break;
  }

  if (!ExitBlock)
    return false;
  return true;
}

TargetTransformInfo::TargetTransformInfo(const DataLayout &DL)
    : TTIImpl(new Model<NoTTIImpl>(NoTTIImpl(DL))) {}

TargetTransformInfo::~TargetTransformInfo() {}

TargetTransformInfo::TargetTransformInfo(TargetTransformInfo &&Arg)
    : TTIImpl(std::move(Arg.TTIImpl)) {}

TargetTransformInfo &TargetTransformInfo::operator=(TargetTransformInfo &&RHS) {
  TTIImpl = std::move(RHS.TTIImpl);
  return *this;
}

unsigned TargetTransformInfo::getInliningThresholdMultiplier() const {
  return TTIImpl->getInliningThresholdMultiplier();
}

unsigned
TargetTransformInfo::adjustInliningThreshold(const CallBase *CB) const {
  return TTIImpl->adjustInliningThreshold(CB);
}

int TargetTransformInfo::getInlinerVectorBonusPercent() const {
  return TTIImpl->getInlinerVectorBonusPercent();
}

int TargetTransformInfo::getGEPCost(Type *PointeeType, const Value *Ptr,
                                    ArrayRef<const Value *> Operands,
                                    TTI::TargetCostKind CostKind) const {
  return TTIImpl->getGEPCost(PointeeType, Ptr, Operands, CostKind);
}

unsigned TargetTransformInfo::getEstimatedNumberOfCaseClusters(
    const SwitchInst &SI, unsigned &JTSize, ProfileSummaryInfo *PSI,
    BlockFrequencyInfo *BFI) const {
  return TTIImpl->getEstimatedNumberOfCaseClusters(SI, JTSize, PSI, BFI);
}

InstructionCost
TargetTransformInfo::getUserCost(const User *U,
                                 ArrayRef<const Value *> Operands,
                                 enum TargetCostKind CostKind) const {
  InstructionCost Cost = TTIImpl->getUserCost(U, Operands, CostKind);
  assert((CostKind == TTI::TCK_RecipThroughput || Cost >= 0) &&
         "TTI should not produce negative costs!");
  return Cost;
}

BranchProbability TargetTransformInfo::getPredictableBranchThreshold() const {
  return TTIImpl->getPredictableBranchThreshold();
}

bool TargetTransformInfo::hasBranchDivergence() const {
  return TTIImpl->hasBranchDivergence();
}

bool TargetTransformInfo::useGPUDivergenceAnalysis() const {
  return TTIImpl->useGPUDivergenceAnalysis();
}

bool TargetTransformInfo::isSourceOfDivergence(const Value *V) const {
  return TTIImpl->isSourceOfDivergence(V);
}

bool llvm::TargetTransformInfo::isAlwaysUniform(const Value *V) const {
  return TTIImpl->isAlwaysUniform(V);
}

unsigned TargetTransformInfo::getFlatAddressSpace() const {
  return TTIImpl->getFlatAddressSpace();
}

bool TargetTransformInfo::collectFlatAddressOperands(
    SmallVectorImpl<int> &OpIndexes, Intrinsic::ID IID) const {
  return TTIImpl->collectFlatAddressOperands(OpIndexes, IID);
}

bool TargetTransformInfo::isNoopAddrSpaceCast(unsigned FromAS,
                                              unsigned ToAS) const {
  return TTIImpl->isNoopAddrSpaceCast(FromAS, ToAS);
}

unsigned TargetTransformInfo::getAssumedAddrSpace(const Value *V) const {
  return TTIImpl->getAssumedAddrSpace(V);
}

Value *TargetTransformInfo::rewriteIntrinsicWithAddressSpace(
    IntrinsicInst *II, Value *OldV, Value *NewV) const {
  return TTIImpl->rewriteIntrinsicWithAddressSpace(II, OldV, NewV);
}

bool TargetTransformInfo::isLoweredToCall(const Function *F) const {
  return TTIImpl->isLoweredToCall(F);
}

bool TargetTransformInfo::isHardwareLoopProfitable(
    Loop *L, ScalarEvolution &SE, AssumptionCache &AC,
    TargetLibraryInfo *LibInfo, HardwareLoopInfo &HWLoopInfo) const {
  return TTIImpl->isHardwareLoopProfitable(L, SE, AC, LibInfo, HWLoopInfo);
}

bool TargetTransformInfo::preferPredicateOverEpilogue(
    Loop *L, LoopInfo *LI, ScalarEvolution &SE, AssumptionCache &AC,
    TargetLibraryInfo *TLI, DominatorTree *DT,
    const LoopAccessInfo *LAI) const {
  return TTIImpl->preferPredicateOverEpilogue(L, LI, SE, AC, TLI, DT, LAI);
}

bool TargetTransformInfo::emitGetActiveLaneMask() const {
  return TTIImpl->emitGetActiveLaneMask();
}

Optional<Instruction *>
TargetTransformInfo::instCombineIntrinsic(InstCombiner &IC,
                                          IntrinsicInst &II) const {
  return TTIImpl->instCombineIntrinsic(IC, II);
}

Optional<Value *> TargetTransformInfo::simplifyDemandedUseBitsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt DemandedMask, KnownBits &Known,
    bool &KnownBitsComputed) const {
  return TTIImpl->simplifyDemandedUseBitsIntrinsic(IC, II, DemandedMask, Known,
                                                   KnownBitsComputed);
}

Optional<Value *> TargetTransformInfo::simplifyDemandedVectorEltsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
    APInt &UndefElts2, APInt &UndefElts3,
    std::function<void(Instruction *, unsigned, APInt, APInt &)>
        SimplifyAndSetOp) const {
  return TTIImpl->simplifyDemandedVectorEltsIntrinsic(
      IC, II, DemandedElts, UndefElts, UndefElts2, UndefElts3,
      SimplifyAndSetOp);
}

void TargetTransformInfo::getUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, UnrollingPreferences &UP) const {
  return TTIImpl->getUnrollingPreferences(L, SE, UP);
}

void TargetTransformInfo::getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                                PeelingPreferences &PP) const {
  return TTIImpl->getPeelingPreferences(L, SE, PP);
}

bool TargetTransformInfo::isLegalAddImmediate(int64_t Imm) const {
  return TTIImpl->isLegalAddImmediate(Imm);
}

bool TargetTransformInfo::isLegalICmpImmediate(int64_t Imm) const {
  return TTIImpl->isLegalICmpImmediate(Imm);
}

bool TargetTransformInfo::isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                                int64_t BaseOffset,
                                                bool HasBaseReg, int64_t Scale,
                                                unsigned AddrSpace,
                                                Instruction *I) const {
  return TTIImpl->isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                        Scale, AddrSpace, I);
}

bool TargetTransformInfo::isLSRCostLess(LSRCost &C1, LSRCost &C2) const {
  return TTIImpl->isLSRCostLess(C1, C2);
}

bool TargetTransformInfo::isNumRegsMajorCostOfLSR() const {
  return TTIImpl->isNumRegsMajorCostOfLSR();
}

bool TargetTransformInfo::isProfitableLSRChainElement(Instruction *I) const {
  return TTIImpl->isProfitableLSRChainElement(I);
}

bool TargetTransformInfo::canMacroFuseCmp() const {
  return TTIImpl->canMacroFuseCmp();
}

bool TargetTransformInfo::canSaveCmp(Loop *L, BranchInst **BI,
                                     ScalarEvolution *SE, LoopInfo *LI,
                                     DominatorTree *DT, AssumptionCache *AC,
                                     TargetLibraryInfo *LibInfo) const {
  return TTIImpl->canSaveCmp(L, BI, SE, LI, DT, AC, LibInfo);
}

TTI::AddressingModeKind
TargetTransformInfo::getPreferredAddressingMode(const Loop *L,
                                                ScalarEvolution *SE) const {
  return TTIImpl->getPreferredAddressingMode(L, SE);
}

bool TargetTransformInfo::isLegalMaskedStore(Type *DataType,
                                             Align Alignment) const {
  return TTIImpl->isLegalMaskedStore(DataType, Alignment);
}

bool TargetTransformInfo::isLegalMaskedLoad(Type *DataType,
                                            Align Alignment) const {
  return TTIImpl->isLegalMaskedLoad(DataType, Alignment);
}

bool TargetTransformInfo::isLegalNTStore(Type *DataType,
                                         Align Alignment) const {
  return TTIImpl->isLegalNTStore(DataType, Alignment);
}

bool TargetTransformInfo::isLegalNTLoad(Type *DataType, Align Alignment) const {
  return TTIImpl->isLegalNTLoad(DataType, Alignment);
}

bool TargetTransformInfo::isLegalMaskedGather(Type *DataType,
                                              Align Alignment) const {
  return TTIImpl->isLegalMaskedGather(DataType, Alignment);
}

bool TargetTransformInfo::isLegalMaskedScatter(Type *DataType,
                                               Align Alignment) const {
  return TTIImpl->isLegalMaskedScatter(DataType, Alignment);
}

bool TargetTransformInfo::isLegalMaskedCompressStore(Type *DataType) const {
  return TTIImpl->isLegalMaskedCompressStore(DataType);
}

bool TargetTransformInfo::isLegalMaskedExpandLoad(Type *DataType) const {
  return TTIImpl->isLegalMaskedExpandLoad(DataType);
}

bool TargetTransformInfo::hasDivRemOp(Type *DataType, bool IsSigned) const {
  return TTIImpl->hasDivRemOp(DataType, IsSigned);
}

bool TargetTransformInfo::hasVolatileVariant(Instruction *I,
                                             unsigned AddrSpace) const {
  return TTIImpl->hasVolatileVariant(I, AddrSpace);
}

bool TargetTransformInfo::prefersVectorizedAddressing() const {
  return TTIImpl->prefersVectorizedAddressing();
}

int TargetTransformInfo::getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                              int64_t BaseOffset,
                                              bool HasBaseReg, int64_t Scale,
                                              unsigned AddrSpace) const {
  int Cost = TTIImpl->getScalingFactorCost(Ty, BaseGV, BaseOffset, HasBaseReg,
                                           Scale, AddrSpace);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

bool TargetTransformInfo::LSRWithInstrQueries() const {
  return TTIImpl->LSRWithInstrQueries();
}

bool TargetTransformInfo::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return TTIImpl->isTruncateFree(Ty1, Ty2);
}

bool TargetTransformInfo::isProfitableToHoist(Instruction *I) const {
  return TTIImpl->isProfitableToHoist(I);
}

bool TargetTransformInfo::useAA() const { return TTIImpl->useAA(); }

bool TargetTransformInfo::isTypeLegal(Type *Ty) const {
  return TTIImpl->isTypeLegal(Ty);
}

unsigned TargetTransformInfo::getRegUsageForType(Type *Ty) const {
  return TTIImpl->getRegUsageForType(Ty);
}

bool TargetTransformInfo::shouldBuildLookupTables() const {
  return TTIImpl->shouldBuildLookupTables();
}
bool TargetTransformInfo::shouldBuildLookupTablesForConstant(
    Constant *C) const {
  return TTIImpl->shouldBuildLookupTablesForConstant(C);
}

bool TargetTransformInfo::useColdCCForColdCall(Function &F) const {
  return TTIImpl->useColdCCForColdCall(F);
}

unsigned
TargetTransformInfo::getScalarizationOverhead(VectorType *Ty,
                                              const APInt &DemandedElts,
                                              bool Insert, bool Extract) const {
  return TTIImpl->getScalarizationOverhead(Ty, DemandedElts, Insert, Extract);
}

unsigned TargetTransformInfo::getOperandsScalarizationOverhead(
    ArrayRef<const Value *> Args, ArrayRef<Type *> Tys) const {
  return TTIImpl->getOperandsScalarizationOverhead(Args, Tys);
}

bool TargetTransformInfo::supportsEfficientVectorElementLoadStore() const {
  return TTIImpl->supportsEfficientVectorElementLoadStore();
}

bool TargetTransformInfo::enableAggressiveInterleaving(
    bool LoopHasReductions) const {
  return TTIImpl->enableAggressiveInterleaving(LoopHasReductions);
}

TargetTransformInfo::MemCmpExpansionOptions
TargetTransformInfo::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  return TTIImpl->enableMemCmpExpansion(OptSize, IsZeroCmp);
}

bool TargetTransformInfo::enableInterleavedAccessVectorization() const {
  return TTIImpl->enableInterleavedAccessVectorization();
}

bool TargetTransformInfo::enableMaskedInterleavedAccessVectorization() const {
  return TTIImpl->enableMaskedInterleavedAccessVectorization();
}

bool TargetTransformInfo::isFPVectorizationPotentiallyUnsafe() const {
  return TTIImpl->isFPVectorizationPotentiallyUnsafe();
}

bool TargetTransformInfo::allowsMisalignedMemoryAccesses(LLVMContext &Context,
                                                         unsigned BitWidth,
                                                         unsigned AddressSpace,
                                                         Align Alignment,
                                                         bool *Fast) const {
  return TTIImpl->allowsMisalignedMemoryAccesses(Context, BitWidth,
                                                 AddressSpace, Alignment, Fast);
}

TargetTransformInfo::PopcntSupportKind
TargetTransformInfo::getPopcntSupport(unsigned IntTyWidthInBit) const {
  return TTIImpl->getPopcntSupport(IntTyWidthInBit);
}

bool TargetTransformInfo::haveFastSqrt(Type *Ty) const {
  return TTIImpl->haveFastSqrt(Ty);
}

bool TargetTransformInfo::isFCmpOrdCheaperThanFCmpZero(Type *Ty) const {
  return TTIImpl->isFCmpOrdCheaperThanFCmpZero(Ty);
}

int TargetTransformInfo::getFPOpCost(Type *Ty) const {
  int Cost = TTIImpl->getFPOpCost(Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntImmCodeSizeCost(unsigned Opcode, unsigned Idx,
                                               const APInt &Imm,
                                               Type *Ty) const {
  int Cost = TTIImpl->getIntImmCodeSizeCost(Opcode, Idx, Imm, Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty,
                                       TTI::TargetCostKind CostKind) const {
  int Cost = TTIImpl->getIntImmCost(Imm, Ty, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                           const APInt &Imm, Type *Ty,
                                           TTI::TargetCostKind CostKind,
                                           Instruction *Inst) const {
  int Cost = TTIImpl->getIntImmCostInst(Opcode, Idx, Imm, Ty, CostKind, Inst);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int
TargetTransformInfo::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                         const APInt &Imm, Type *Ty,
                                         TTI::TargetCostKind CostKind) const {
  int Cost = TTIImpl->getIntImmCostIntrin(IID, Idx, Imm, Ty, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

unsigned TargetTransformInfo::getNumberOfRegisters(unsigned ClassID) const {
  return TTIImpl->getNumberOfRegisters(ClassID);
}

unsigned TargetTransformInfo::getRegisterClassForType(bool Vector,
                                                      Type *Ty) const {
  return TTIImpl->getRegisterClassForType(Vector, Ty);
}

const char *TargetTransformInfo::getRegisterClassName(unsigned ClassID) const {
  return TTIImpl->getRegisterClassName(ClassID);
}

TypeSize TargetTransformInfo::getRegisterBitWidth(
    TargetTransformInfo::RegisterKind K) const {
  return TTIImpl->getRegisterBitWidth(K);
}

unsigned TargetTransformInfo::getMinVectorRegisterBitWidth() const {
  return TTIImpl->getMinVectorRegisterBitWidth();
}

Optional<unsigned> TargetTransformInfo::getMaxVScale() const {
  return TTIImpl->getMaxVScale();
}

bool TargetTransformInfo::shouldMaximizeVectorBandwidth(bool OptSize) const {
  return TTIImpl->shouldMaximizeVectorBandwidth(OptSize);
}

ElementCount TargetTransformInfo::getMinimumVF(unsigned ElemWidth,
                                               bool IsScalable) const {
  return TTIImpl->getMinimumVF(ElemWidth, IsScalable);
}

unsigned TargetTransformInfo::getMaximumVF(unsigned ElemWidth,
                                           unsigned Opcode) const {
  return TTIImpl->getMaximumVF(ElemWidth, Opcode);
}

bool TargetTransformInfo::shouldConsiderAddressTypePromotion(
    const Instruction &I, bool &AllowPromotionWithoutCommonHeader) const {
  return TTIImpl->shouldConsiderAddressTypePromotion(
      I, AllowPromotionWithoutCommonHeader);
}

unsigned TargetTransformInfo::getCacheLineSize() const {
  return TTIImpl->getCacheLineSize();
}

llvm::Optional<unsigned>
TargetTransformInfo::getCacheSize(CacheLevel Level) const {
  return TTIImpl->getCacheSize(Level);
}

llvm::Optional<unsigned>
TargetTransformInfo::getCacheAssociativity(CacheLevel Level) const {
  return TTIImpl->getCacheAssociativity(Level);
}

unsigned TargetTransformInfo::getPrefetchDistance() const {
  return TTIImpl->getPrefetchDistance();
}

unsigned TargetTransformInfo::getMinPrefetchStride(
    unsigned NumMemAccesses, unsigned NumStridedMemAccesses,
    unsigned NumPrefetches, bool HasCall) const {
  return TTIImpl->getMinPrefetchStride(NumMemAccesses, NumStridedMemAccesses,
                                       NumPrefetches, HasCall);
}

unsigned TargetTransformInfo::getMaxPrefetchIterationsAhead() const {
  return TTIImpl->getMaxPrefetchIterationsAhead();
}

bool TargetTransformInfo::enableWritePrefetching() const {
  return TTIImpl->enableWritePrefetching();
}

unsigned TargetTransformInfo::getMaxInterleaveFactor(unsigned VF) const {
  return TTIImpl->getMaxInterleaveFactor(VF);
}

TargetTransformInfo::OperandValueKind
TargetTransformInfo::getOperandInfo(const Value *V,
                                    OperandValueProperties &OpProps) {
  OperandValueKind OpInfo = OK_AnyValue;
  OpProps = OP_None;

  if (const auto *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getValue().isPowerOf2())
      OpProps = OP_PowerOf2;
    return OK_UniformConstantValue;
  }

  // A broadcast shuffle creates a uniform value.
  // TODO: Add support for non-zero index broadcasts.
  // TODO: Add support for different source vector width.
  if (const auto *ShuffleInst = dyn_cast<ShuffleVectorInst>(V))
    if (ShuffleInst->isZeroEltSplat())
      OpInfo = OK_UniformValue;

  const Value *Splat = getSplatValue(V);

  // Check for a splat of a constant or for a non uniform vector of constants
  // and check if the constant(s) are all powers of two.
  if (isa<ConstantVector>(V) || isa<ConstantDataVector>(V)) {
    OpInfo = OK_NonUniformConstantValue;
    if (Splat) {
      OpInfo = OK_UniformConstantValue;
      if (auto *CI = dyn_cast<ConstantInt>(Splat))
        if (CI->getValue().isPowerOf2())
          OpProps = OP_PowerOf2;
    } else if (const auto *CDS = dyn_cast<ConstantDataSequential>(V)) {
      OpProps = OP_PowerOf2;
      for (unsigned I = 0, E = CDS->getNumElements(); I != E; ++I) {
        if (auto *CI = dyn_cast<ConstantInt>(CDS->getElementAsConstant(I)))
          if (CI->getValue().isPowerOf2())
            continue;
        OpProps = OP_None;
        break;
      }
    }
  }

  // Check for a splat of a uniform value. This is not loop aware, so return
  // true only for the obviously uniform cases (argument, globalvalue)
  if (Splat && (isa<Argument>(Splat) || isa<GlobalValue>(Splat)))
    OpInfo = OK_UniformValue;

  return OpInfo;
}

int TargetTransformInfo::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    OperandValueKind Opd1Info,
    OperandValueKind Opd2Info, OperandValueProperties Opd1PropInfo,
    OperandValueProperties Opd2PropInfo, ArrayRef<const Value *> Args,
    const Instruction *CxtI) const {
  int Cost = TTIImpl->getArithmeticInstrCost(
      Opcode, Ty, CostKind, Opd1Info, Opd2Info, Opd1PropInfo, Opd2PropInfo,
      Args, CxtI);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getShuffleCost(ShuffleKind Kind, VectorType *Ty,
                                        ArrayRef<int> Mask, int Index,
                                        VectorType *SubTp) const {
  int Cost = TTIImpl->getShuffleCost(Kind, Ty, Mask, Index, SubTp);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

TTI::CastContextHint
TargetTransformInfo::getCastContextHint(const Instruction *I) {
  if (!I)
    return CastContextHint::None;

  auto getLoadStoreKind = [](const Value *V, unsigned LdStOp, unsigned MaskedOp,
                             unsigned GatScatOp) {
    const Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      return CastContextHint::None;

    if (I->getOpcode() == LdStOp)
      return CastContextHint::Normal;

    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() == MaskedOp)
        return TTI::CastContextHint::Masked;
      if (II->getIntrinsicID() == GatScatOp)
        return TTI::CastContextHint::GatherScatter;
    }

    return TTI::CastContextHint::None;
  };

  switch (I->getOpcode()) {
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPExt:
    return getLoadStoreKind(I->getOperand(0), Instruction::Load,
                            Intrinsic::masked_load, Intrinsic::masked_gather);
  case Instruction::Trunc:
  case Instruction::FPTrunc:
    if (I->hasOneUse())
      return getLoadStoreKind(*I->user_begin(), Instruction::Store,
                              Intrinsic::masked_store,
                              Intrinsic::masked_scatter);
    break;
  default:
    return CastContextHint::None;
  }

  return TTI::CastContextHint::None;
}

int TargetTransformInfo::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                          CastContextHint CCH,
                                          TTI::TargetCostKind CostKind,
                                          const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  int Cost = TTIImpl->getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getExtractWithExtendCost(unsigned Opcode, Type *Dst,
                                                  VectorType *VecTy,
                                                  unsigned Index) const {
  int Cost = TTIImpl->getExtractWithExtendCost(Opcode, Dst, VecTy, Index);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCFInstrCost(unsigned Opcode,
                                        TTI::TargetCostKind CostKind) const {
  int Cost = TTIImpl->getCFInstrCost(Opcode, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                            Type *CondTy,
                                            CmpInst::Predicate VecPred,
                                            TTI::TargetCostKind CostKind,
                                            const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  int Cost =
      TTIImpl->getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getVectorInstrCost(unsigned Opcode, Type *Val,
                                            unsigned Index) const {
  int Cost = TTIImpl->getVectorInstrCost(Opcode, Val, Index);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getMemoryOpCost(unsigned Opcode, Type *Src,
                                         Align Alignment, unsigned AddressSpace,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  int Cost = TTIImpl->getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                      CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getMaskedMemoryOpCost(
    unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
    TTI::TargetCostKind CostKind) const {
  int Cost =
      TTIImpl->getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                     CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) const {
  int Cost = TTIImpl->getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                             Alignment, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) const {
  int Cost = TTIImpl->getInterleavedMemoryOpCost(
      Opcode, VecTy, Factor, Indices, Alignment, AddressSpace, CostKind,
      UseMaskForCond, UseMaskForGaps);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost
TargetTransformInfo::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                           TTI::TargetCostKind CostKind) const {
  InstructionCost Cost = TTIImpl->getIntrinsicInstrCost(ICA, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCallInstrCost(Function *F, Type *RetTy,
                                          ArrayRef<Type *> Tys,
                                          TTI::TargetCostKind CostKind) const {
  int Cost = TTIImpl->getCallInstrCost(F, RetTy, Tys, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

unsigned TargetTransformInfo::getNumberOfParts(Type *Tp) const {
  return TTIImpl->getNumberOfParts(Tp);
}

int TargetTransformInfo::getAddressComputationCost(Type *Tp,
                                                   ScalarEvolution *SE,
                                                   const SCEV *Ptr) const {
  int Cost = TTIImpl->getAddressComputationCost(Tp, SE, Ptr);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getMemcpyCost(const Instruction *I) const {
  int Cost = TTIImpl->getMemcpyCost(I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getArithmeticReductionCost(unsigned Opcode,
                                                    VectorType *Ty,
                                                    bool IsPairwiseForm,
                                                    TTI::TargetCostKind CostKind) const {
  int Cost = TTIImpl->getArithmeticReductionCost(Opcode, Ty, IsPairwiseForm,
                                                 CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getMinMaxReductionCost(
    VectorType *Ty, VectorType *CondTy, bool IsPairwiseForm, bool IsUnsigned,
    TTI::TargetCostKind CostKind) const {
  int Cost =
      TTIImpl->getMinMaxReductionCost(Ty, CondTy, IsPairwiseForm, IsUnsigned,
                                      CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getExtendedAddReductionCost(
    bool IsMLA, bool IsUnsigned, Type *ResTy, VectorType *Ty,
    TTI::TargetCostKind CostKind) const {
  return TTIImpl->getExtendedAddReductionCost(IsMLA, IsUnsigned, ResTy, Ty,
                                              CostKind);
}

unsigned
TargetTransformInfo::getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) const {
  return TTIImpl->getCostOfKeepingLiveOverCall(Tys);
}

bool TargetTransformInfo::getTgtMemIntrinsic(IntrinsicInst *Inst,
                                             MemIntrinsicInfo &Info) const {
  return TTIImpl->getTgtMemIntrinsic(Inst, Info);
}

unsigned TargetTransformInfo::getAtomicMemIntrinsicMaxElementSize() const {
  return TTIImpl->getAtomicMemIntrinsicMaxElementSize();
}

Value *TargetTransformInfo::getOrCreateResultFromMemIntrinsic(
    IntrinsicInst *Inst, Type *ExpectedType) const {
  return TTIImpl->getOrCreateResultFromMemIntrinsic(Inst, ExpectedType);
}

Type *TargetTransformInfo::getMemcpyLoopLoweringType(
    LLVMContext &Context, Value *Length, unsigned SrcAddrSpace,
    unsigned DestAddrSpace, unsigned SrcAlign, unsigned DestAlign) const {
  return TTIImpl->getMemcpyLoopLoweringType(Context, Length, SrcAddrSpace,
                                            DestAddrSpace, SrcAlign, DestAlign);
}

void TargetTransformInfo::getMemcpyLoopResidualLoweringType(
    SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
    unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
    unsigned SrcAlign, unsigned DestAlign) const {
  TTIImpl->getMemcpyLoopResidualLoweringType(OpsOut, Context, RemainingBytes,
                                             SrcAddrSpace, DestAddrSpace,
                                             SrcAlign, DestAlign);
}

bool TargetTransformInfo::areInlineCompatible(const Function *Caller,
                                              const Function *Callee) const {
  return TTIImpl->areInlineCompatible(Caller, Callee);
}

bool TargetTransformInfo::areFunctionArgsABICompatible(
    const Function *Caller, const Function *Callee,
    SmallPtrSetImpl<Argument *> &Args) const {
  return TTIImpl->areFunctionArgsABICompatible(Caller, Callee, Args);
}

bool TargetTransformInfo::isIndexedLoadLegal(MemIndexedMode Mode,
                                             Type *Ty) const {
  return TTIImpl->isIndexedLoadLegal(Mode, Ty);
}

bool TargetTransformInfo::isIndexedStoreLegal(MemIndexedMode Mode,
                                              Type *Ty) const {
  return TTIImpl->isIndexedStoreLegal(Mode, Ty);
}

unsigned TargetTransformInfo::getLoadStoreVecRegBitWidth(unsigned AS) const {
  return TTIImpl->getLoadStoreVecRegBitWidth(AS);
}

bool TargetTransformInfo::isLegalToVectorizeLoad(LoadInst *LI) const {
  return TTIImpl->isLegalToVectorizeLoad(LI);
}

bool TargetTransformInfo::isLegalToVectorizeStore(StoreInst *SI) const {
  return TTIImpl->isLegalToVectorizeStore(SI);
}

bool TargetTransformInfo::isLegalToVectorizeLoadChain(
    unsigned ChainSizeInBytes, Align Alignment, unsigned AddrSpace) const {
  return TTIImpl->isLegalToVectorizeLoadChain(ChainSizeInBytes, Alignment,
                                              AddrSpace);
}

bool TargetTransformInfo::isLegalToVectorizeStoreChain(
    unsigned ChainSizeInBytes, Align Alignment, unsigned AddrSpace) const {
  return TTIImpl->isLegalToVectorizeStoreChain(ChainSizeInBytes, Alignment,
                                               AddrSpace);
}

bool TargetTransformInfo::isLegalToVectorizeReduction(
    RecurrenceDescriptor RdxDesc, ElementCount VF) const {
  return TTIImpl->isLegalToVectorizeReduction(RdxDesc, VF);
}

unsigned TargetTransformInfo::getLoadVectorFactor(unsigned VF,
                                                  unsigned LoadSize,
                                                  unsigned ChainSizeInBytes,
                                                  VectorType *VecTy) const {
  return TTIImpl->getLoadVectorFactor(VF, LoadSize, ChainSizeInBytes, VecTy);
}

unsigned TargetTransformInfo::getStoreVectorFactor(unsigned VF,
                                                   unsigned StoreSize,
                                                   unsigned ChainSizeInBytes,
                                                   VectorType *VecTy) const {
  return TTIImpl->getStoreVectorFactor(VF, StoreSize, ChainSizeInBytes, VecTy);
}

bool TargetTransformInfo::preferInLoopReduction(unsigned Opcode, Type *Ty,
                                                ReductionFlags Flags) const {
  return TTIImpl->preferInLoopReduction(Opcode, Ty, Flags);
}

bool TargetTransformInfo::preferPredicatedReductionSelect(
    unsigned Opcode, Type *Ty, ReductionFlags Flags) const {
  return TTIImpl->preferPredicatedReductionSelect(Opcode, Ty, Flags);
}

bool TargetTransformInfo::shouldExpandReduction(const IntrinsicInst *II) const {
  return TTIImpl->shouldExpandReduction(II);
}

unsigned TargetTransformInfo::getGISelRematGlobalCost() const {
  return TTIImpl->getGISelRematGlobalCost();
}

bool TargetTransformInfo::supportsScalableVectors() const {
  return TTIImpl->supportsScalableVectors();
}

InstructionCost
TargetTransformInfo::getInstructionLatency(const Instruction *I) const {
  return TTIImpl->getInstructionLatency(I);
}

static bool matchPairwiseShuffleMask(ShuffleVectorInst *SI, bool IsLeft,
                                     unsigned Level) {
  // We don't need a shuffle if we just want to have element 0 in position 0 of
  // the vector.
  if (!SI && Level == 0 && IsLeft)
    return true;
  else if (!SI)
    return false;

  SmallVector<int, 32> Mask(
      cast<FixedVectorType>(SI->getType())->getNumElements(), -1);

  // Build a mask of 0, 2, ... (left) or 1, 3, ... (right) depending on whether
  // we look at the left or right side.
  for (unsigned i = 0, e = (1 << Level), val = !IsLeft; i != e; ++i, val += 2)
    Mask[i] = val;

  ArrayRef<int> ActualMask = SI->getShuffleMask();
  return Mask == ActualMask;
}

static Optional<TTI::ReductionData> getReductionData(Instruction *I) {
  Value *L, *R;
  if (m_BinOp(m_Value(L), m_Value(R)).match(I))
    return TTI::ReductionData(TTI::RK_Arithmetic, I->getOpcode(), L, R);
  if (auto *SI = dyn_cast<SelectInst>(I)) {
    if (m_SMin(m_Value(L), m_Value(R)).match(SI) ||
        m_SMax(m_Value(L), m_Value(R)).match(SI) ||
        m_OrdFMin(m_Value(L), m_Value(R)).match(SI) ||
        m_OrdFMax(m_Value(L), m_Value(R)).match(SI) ||
        m_UnordFMin(m_Value(L), m_Value(R)).match(SI) ||
        m_UnordFMax(m_Value(L), m_Value(R)).match(SI)) {
      auto *CI = cast<CmpInst>(SI->getCondition());
      return TTI::ReductionData(TTI::RK_MinMax, CI->getOpcode(), L, R);
    }
    if (m_UMin(m_Value(L), m_Value(R)).match(SI) ||
        m_UMax(m_Value(L), m_Value(R)).match(SI)) {
      auto *CI = cast<CmpInst>(SI->getCondition());
      return TTI::ReductionData(TTI::RK_UnsignedMinMax, CI->getOpcode(), L, R);
    }
  }
  return llvm::None;
}

static TTI::ReductionKind matchPairwiseReductionAtLevel(Instruction *I,
                                                        unsigned Level,
                                                        unsigned NumLevels) {
  // Match one level of pairwise operations.
  // %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  // %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  // %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  if (!I)
    return TTI::RK_None;

  assert(I->getType()->isVectorTy() && "Expecting a vector type");

  Optional<TTI::ReductionData> RD = getReductionData(I);
  if (!RD)
    return TTI::RK_None;

  ShuffleVectorInst *LS = dyn_cast<ShuffleVectorInst>(RD->LHS);
  if (!LS && Level)
    return TTI::RK_None;
  ShuffleVectorInst *RS = dyn_cast<ShuffleVectorInst>(RD->RHS);
  if (!RS && Level)
    return TTI::RK_None;

  // On level 0 we can omit one shufflevector instruction.
  if (!Level && !RS && !LS)
    return TTI::RK_None;

  // Shuffle inputs must match.
  Value *NextLevelOpL = LS ? LS->getOperand(0) : nullptr;
  Value *NextLevelOpR = RS ? RS->getOperand(0) : nullptr;
  Value *NextLevelOp = nullptr;
  if (NextLevelOpR && NextLevelOpL) {
    // If we have two shuffles their operands must match.
    if (NextLevelOpL != NextLevelOpR)
      return TTI::RK_None;

    NextLevelOp = NextLevelOpL;
  } else if (Level == 0 && (NextLevelOpR || NextLevelOpL)) {
    // On the first level we can omit the shufflevector <0, undef,...>. So the
    // input to the other shufflevector <1, undef> must match with one of the
    // inputs to the current binary operation.
    // Example:
    //  %NextLevelOpL = shufflevector %R, <1, undef ...>
    //  %BinOp        = fadd          %NextLevelOpL, %R
    if (NextLevelOpL && NextLevelOpL != RD->RHS)
      return TTI::RK_None;
    else if (NextLevelOpR && NextLevelOpR != RD->LHS)
      return TTI::RK_None;

    NextLevelOp = NextLevelOpL ? RD->RHS : RD->LHS;
  } else
    return TTI::RK_None;

  // Check that the next levels binary operation exists and matches with the
  // current one.
  if (Level + 1 != NumLevels) {
    if (!isa<Instruction>(NextLevelOp))
      return TTI::RK_None;
    Optional<TTI::ReductionData> NextLevelRD =
        getReductionData(cast<Instruction>(NextLevelOp));
    if (!NextLevelRD || !RD->hasSameData(*NextLevelRD))
      return TTI::RK_None;
  }

  // Shuffle mask for pairwise operation must match.
  if (matchPairwiseShuffleMask(LS, /*IsLeft=*/true, Level)) {
    if (!matchPairwiseShuffleMask(RS, /*IsLeft=*/false, Level))
      return TTI::RK_None;
  } else if (matchPairwiseShuffleMask(RS, /*IsLeft=*/true, Level)) {
    if (!matchPairwiseShuffleMask(LS, /*IsLeft=*/false, Level))
      return TTI::RK_None;
  } else {
    return TTI::RK_None;
  }

  if (++Level == NumLevels)
    return RD->Kind;

  // Match next level.
  return matchPairwiseReductionAtLevel(dyn_cast<Instruction>(NextLevelOp), Level,
                                       NumLevels);
}

TTI::ReductionKind TTI::matchPairwiseReduction(
  const ExtractElementInst *ReduxRoot, unsigned &Opcode, VectorType *&Ty) {
  if (!EnableReduxCost)
    return TTI::RK_None;

  // Need to extract the first element.
  ConstantInt *CI = dyn_cast<ConstantInt>(ReduxRoot->getOperand(1));
  unsigned Idx = ~0u;
  if (CI)
    Idx = CI->getZExtValue();
  if (Idx != 0)
    return TTI::RK_None;

  auto *RdxStart = dyn_cast<Instruction>(ReduxRoot->getOperand(0));
  if (!RdxStart)
    return TTI::RK_None;
  Optional<TTI::ReductionData> RD = getReductionData(RdxStart);
  if (!RD)
    return TTI::RK_None;

  auto *VecTy = cast<FixedVectorType>(RdxStart->getType());
  unsigned NumVecElems = VecTy->getNumElements();
  if (!isPowerOf2_32(NumVecElems))
    return TTI::RK_None;

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
      TTI::RK_None)
    return TTI::RK_None;

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

TTI::ReductionKind TTI::matchVectorSplittingReduction(
  const ExtractElementInst *ReduxRoot, unsigned &Opcode, VectorType *&Ty) {

  if (!EnableReduxCost)
    return TTI::RK_None;

  // Need to extract the first element.
  ConstantInt *CI = dyn_cast<ConstantInt>(ReduxRoot->getOperand(1));
  unsigned Idx = ~0u;
  if (CI)
    Idx = CI->getZExtValue();
  if (Idx != 0)
    return TTI::RK_None;

  auto *RdxStart = dyn_cast<Instruction>(ReduxRoot->getOperand(0));
  if (!RdxStart)
    return TTI::RK_None;
  Optional<TTI::ReductionData> RD = getReductionData(RdxStart);
  if (!RD)
    return TTI::RK_None;

  auto *VecTy = cast<FixedVectorType>(ReduxRoot->getOperand(0)->getType());
  unsigned NumVecElems = VecTy->getNumElements();
  if (!isPowerOf2_32(NumVecElems))
    return TTI::RK_None;

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
      return TTI::RK_None;
    Optional<TTI::ReductionData> RDLevel = getReductionData(RdxOp);
    if (!RDLevel || !RDLevel->hasSameData(*RD))
      return TTI::RK_None;

    Value *NextRdxOp;
    ShuffleVectorInst *Shuffle;
    std::tie(NextRdxOp, Shuffle) =
        getShuffleAndOtherOprd(RDLevel->LHS, RDLevel->RHS);

    // Check the current reduction operation and the shuffle use the same value.
    if (Shuffle == nullptr)
      return TTI::RK_None;
    if (Shuffle->getOperand(0) != NextRdxOp)
      return TTI::RK_None;

    // Check that shuffle masks matches.
    for (unsigned j = 0; j != MaskStart; ++j)
      ShuffleMask[j] = MaskStart + j;
    // Fill the rest of the mask with -1 for undef.
    std::fill(&ShuffleMask[MaskStart], ShuffleMask.end(), -1);

    ArrayRef<int> Mask = Shuffle->getShuffleMask();
    if (ShuffleMask != Mask)
      return TTI::RK_None;

    RdxOp = dyn_cast<Instruction>(NextRdxOp);
    NumVecElemsRemain /= 2;
    MaskStart *= 2;
  }

  Opcode = RD->Opcode;
  Ty = VecTy;
  return RD->Kind;
}

TTI::ReductionKind
TTI::matchVectorReduction(const ExtractElementInst *Root, unsigned &Opcode,
                          VectorType *&Ty, bool &IsPairwise) {
  TTI::ReductionKind RdxKind = matchVectorSplittingReduction(Root, Opcode, Ty);
  if (RdxKind != TTI::ReductionKind::RK_None) {
    IsPairwise = false;
    return RdxKind;
  }
  IsPairwise = true;
  return matchPairwiseReduction(Root, Opcode, Ty);
}

InstructionCost
TargetTransformInfo::getInstructionThroughput(const Instruction *I) const {
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
  case Instruction::Ret:
  case Instruction::PHI:
  case Instruction::Br:
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
  case Instruction::Xor:
  case Instruction::FNeg:
  case Instruction::Select:
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::Store:
  case Instruction::Load:
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
  case Instruction::AddrSpaceCast:
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
  case Instruction::ExtractValue:
  case Instruction::ShuffleVector:
  case Instruction::Call:
    return getUserCost(I, CostKind);
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

TargetTransformInfo::Concept::~Concept() {}

TargetIRAnalysis::TargetIRAnalysis() : TTICallback(&getDefaultTTI) {}

TargetIRAnalysis::TargetIRAnalysis(
    std::function<Result(const Function &)> TTICallback)
    : TTICallback(std::move(TTICallback)) {}

TargetIRAnalysis::Result TargetIRAnalysis::run(const Function &F,
                                               FunctionAnalysisManager &) {
  return TTICallback(F);
}

AnalysisKey TargetIRAnalysis::Key;

TargetIRAnalysis::Result TargetIRAnalysis::getDefaultTTI(const Function &F) {
  return Result(F.getParent()->getDataLayout());
}

// Register the basic pass.
INITIALIZE_PASS(TargetTransformInfoWrapperPass, "tti",
                "Target Transform Information", false, true)
char TargetTransformInfoWrapperPass::ID = 0;

void TargetTransformInfoWrapperPass::anchor() {}

TargetTransformInfoWrapperPass::TargetTransformInfoWrapperPass()
    : ImmutablePass(ID) {
  initializeTargetTransformInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

TargetTransformInfoWrapperPass::TargetTransformInfoWrapperPass(
    TargetIRAnalysis TIRA)
    : ImmutablePass(ID), TIRA(std::move(TIRA)) {
  initializeTargetTransformInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

TargetTransformInfo &TargetTransformInfoWrapperPass::getTTI(const Function &F) {
  FunctionAnalysisManager DummyFAM;
  TTI = TIRA.run(F, DummyFAM);
  return *TTI;
}

ImmutablePass *
llvm::createTargetTransformInfoWrapperPass(TargetIRAnalysis TIRA) {
  return new TargetTransformInfoWrapperPass(std::move(TIRA));
}
