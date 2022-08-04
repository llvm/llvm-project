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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include <utility>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "tti"

static cl::opt<bool> EnableReduxCost("costmodel-reduxcost", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Recognize reduction patterns."));

static cl::opt<unsigned> CacheLineSize(
    "cache-line-size", cl::init(0), cl::Hidden,
    cl::desc("Use this to override the target cache line size when "
             "specified by the user."));

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

IntrinsicCostAttributes::IntrinsicCostAttributes(
    Intrinsic::ID Id, const CallBase &CI, InstructionCost ScalarizationCost,
    bool TypeBasedOnly)
    : II(dyn_cast<IntrinsicInst>(&CI)), RetTy(CI.getType()), IID(Id),
      ScalarizationCost(ScalarizationCost) {

  if (const auto *FPMO = dyn_cast<FPMathOperator>(&CI))
    FMF = FPMO->getFastMathFlags();

  if (!TypeBasedOnly)
    Arguments.insert(Arguments.begin(), CI.arg_begin(), CI.arg_end());
  FunctionType *FTy = CI.getCalledFunction()->getFunctionType();
  ParamTys.insert(ParamTys.begin(), FTy->param_begin(), FTy->param_end());
}

IntrinsicCostAttributes::IntrinsicCostAttributes(Intrinsic::ID Id, Type *RTy,
                                                 ArrayRef<Type *> Tys,
                                                 FastMathFlags Flags,
                                                 const IntrinsicInst *I,
                                                 InstructionCost ScalarCost)
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
                                                 InstructionCost ScalarCost)
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
    ExitCount = EC;
    break;
  }

  if (!ExitBlock)
    return false;
  return true;
}

TargetTransformInfo::TargetTransformInfo(const DataLayout &DL)
    : TTIImpl(new Model<NoTTIImpl>(NoTTIImpl(DL))) {}

TargetTransformInfo::~TargetTransformInfo() = default;

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

InstructionCost
TargetTransformInfo::getGEPCost(Type *PointeeType, const Value *Ptr,
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

bool TargetTransformInfo::canHaveNonUndefGlobalInitializerInAddressSpace(
    unsigned AS) const {
  return TTIImpl->canHaveNonUndefGlobalInitializerInAddressSpace(AS);
}

unsigned TargetTransformInfo::getAssumedAddrSpace(const Value *V) const {
  return TTIImpl->getAssumedAddrSpace(V);
}

std::pair<const Value *, unsigned>
TargetTransformInfo::getPredicatedAddrSpace(const Value *V) const {
  return TTIImpl->getPredicatedAddrSpace(V);
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
    TargetLibraryInfo *TLI, DominatorTree *DT, LoopVectorizationLegality *LVL,
    InterleavedAccessInfo *IAI) const {
  return TTIImpl->preferPredicateOverEpilogue(L, LI, SE, AC, TLI, DT, LVL, IAI);
}

PredicationStyle TargetTransformInfo::emitGetActiveLaneMask() const {
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
    Loop *L, ScalarEvolution &SE, UnrollingPreferences &UP,
    OptimizationRemarkEmitter *ORE) const {
  return TTIImpl->getUnrollingPreferences(L, SE, UP, ORE);
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

bool TargetTransformInfo::isLSRCostLess(const LSRCost &C1,
                                        const LSRCost &C2) const {
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

bool TargetTransformInfo::isLegalBroadcastLoad(Type *ElementTy,
                                               ElementCount NumElements) const {
  return TTIImpl->isLegalBroadcastLoad(ElementTy, NumElements);
}

bool TargetTransformInfo::isLegalMaskedGather(Type *DataType,
                                              Align Alignment) const {
  return TTIImpl->isLegalMaskedGather(DataType, Alignment);
}

bool TargetTransformInfo::isLegalAltInstr(
    VectorType *VecTy, unsigned Opcode0, unsigned Opcode1,
    const SmallBitVector &OpcodeMask) const {
  return TTIImpl->isLegalAltInstr(VecTy, Opcode0, Opcode1, OpcodeMask);
}

bool TargetTransformInfo::isLegalMaskedScatter(Type *DataType,
                                               Align Alignment) const {
  return TTIImpl->isLegalMaskedScatter(DataType, Alignment);
}

bool TargetTransformInfo::forceScalarizeMaskedGather(VectorType *DataType,
                                                     Align Alignment) const {
  return TTIImpl->forceScalarizeMaskedGather(DataType, Alignment);
}

bool TargetTransformInfo::forceScalarizeMaskedScatter(VectorType *DataType,
                                                      Align Alignment) const {
  return TTIImpl->forceScalarizeMaskedScatter(DataType, Alignment);
}

bool TargetTransformInfo::isLegalMaskedCompressStore(Type *DataType) const {
  return TTIImpl->isLegalMaskedCompressStore(DataType);
}

bool TargetTransformInfo::isLegalMaskedExpandLoad(Type *DataType) const {
  return TTIImpl->isLegalMaskedExpandLoad(DataType);
}

bool TargetTransformInfo::enableOrderedReductions() const {
  return TTIImpl->enableOrderedReductions();
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

InstructionCost TargetTransformInfo::getScalingFactorCost(
    Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset, bool HasBaseReg,
    int64_t Scale, unsigned AddrSpace) const {
  InstructionCost Cost = TTIImpl->getScalingFactorCost(
      Ty, BaseGV, BaseOffset, HasBaseReg, Scale, AddrSpace);
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

bool TargetTransformInfo::shouldBuildRelLookupTables() const {
  return TTIImpl->shouldBuildRelLookupTables();
}

bool TargetTransformInfo::useColdCCForColdCall(Function &F) const {
  return TTIImpl->useColdCCForColdCall(F);
}

InstructionCost
TargetTransformInfo::getScalarizationOverhead(VectorType *Ty,
                                              const APInt &DemandedElts,
                                              bool Insert, bool Extract) const {
  return TTIImpl->getScalarizationOverhead(Ty, DemandedElts, Insert, Extract);
}

InstructionCost TargetTransformInfo::getOperandsScalarizationOverhead(
    ArrayRef<const Value *> Args, ArrayRef<Type *> Tys) const {
  return TTIImpl->getOperandsScalarizationOverhead(Args, Tys);
}

bool TargetTransformInfo::supportsEfficientVectorElementLoadStore() const {
  return TTIImpl->supportsEfficientVectorElementLoadStore();
}

bool TargetTransformInfo::supportsTailCalls() const {
  return TTIImpl->supportsTailCalls();
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

InstructionCost TargetTransformInfo::getFPOpCost(Type *Ty) const {
  InstructionCost Cost = TTIImpl->getFPOpCost(Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getIntImmCodeSizeCost(unsigned Opcode,
                                                           unsigned Idx,
                                                           const APInt &Imm,
                                                           Type *Ty) const {
  InstructionCost Cost = TTIImpl->getIntImmCodeSizeCost(Opcode, Idx, Imm, Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost
TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty,
                                   TTI::TargetCostKind CostKind) const {
  InstructionCost Cost = TTIImpl->getIntImmCost(Imm, Ty, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getIntImmCostInst(
    unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty,
    TTI::TargetCostKind CostKind, Instruction *Inst) const {
  InstructionCost Cost =
      TTIImpl->getIntImmCostInst(Opcode, Idx, Imm, Ty, CostKind, Inst);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost
TargetTransformInfo::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                         const APInt &Imm, Type *Ty,
                                         TTI::TargetCostKind CostKind) const {
  InstructionCost Cost =
      TTIImpl->getIntImmCostIntrin(IID, Idx, Imm, Ty, CostKind);
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

Optional<unsigned> TargetTransformInfo::getVScaleForTuning() const {
  return TTIImpl->getVScaleForTuning();
}

bool TargetTransformInfo::shouldMaximizeVectorBandwidth(
    TargetTransformInfo::RegisterKind K) const {
  return TTIImpl->shouldMaximizeVectorBandwidth(K);
}

ElementCount TargetTransformInfo::getMinimumVF(unsigned ElemWidth,
                                               bool IsScalable) const {
  return TTIImpl->getMinimumVF(ElemWidth, IsScalable);
}

unsigned TargetTransformInfo::getMaximumVF(unsigned ElemWidth,
                                           unsigned Opcode) const {
  return TTIImpl->getMaximumVF(ElemWidth, Opcode);
}

unsigned TargetTransformInfo::getStoreMinimumVF(unsigned VF, Type *ScalarMemTy,
                                                Type *ScalarValTy) const {
  return TTIImpl->getStoreMinimumVF(VF, ScalarMemTy, ScalarValTy);
}

bool TargetTransformInfo::shouldConsiderAddressTypePromotion(
    const Instruction &I, bool &AllowPromotionWithoutCommonHeader) const {
  return TTIImpl->shouldConsiderAddressTypePromotion(
      I, AllowPromotionWithoutCommonHeader);
}

unsigned TargetTransformInfo::getCacheLineSize() const {
  return CacheLineSize.getNumOccurrences() > 0 ? CacheLineSize
                                               : TTIImpl->getCacheLineSize();
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

bool TargetTransformInfo::shouldPrefetchAddressSpace(unsigned AS) const {
  return TTIImpl->shouldPrefetchAddressSpace(AS);
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

InstructionCost TargetTransformInfo::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    OperandValueKind Opd1Info, OperandValueKind Opd2Info,
    OperandValueProperties Opd1PropInfo, OperandValueProperties Opd2PropInfo,
    ArrayRef<const Value *> Args, const Instruction *CxtI) const {
  InstructionCost Cost =
      TTIImpl->getArithmeticInstrCost(Opcode, Ty, CostKind, Opd1Info, Opd2Info,
                                      Opd1PropInfo, Opd2PropInfo, Args, CxtI);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getShuffleCost(
    ShuffleKind Kind, VectorType *Ty, ArrayRef<int> Mask, int Index,
    VectorType *SubTp, ArrayRef<const Value *> Args) const {
  InstructionCost Cost =
      TTIImpl->getShuffleCost(Kind, Ty, Mask, Index, SubTp, Args);
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

InstructionCost TargetTransformInfo::getCastInstrCost(
    unsigned Opcode, Type *Dst, Type *Src, CastContextHint CCH,
    TTI::TargetCostKind CostKind, const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  InstructionCost Cost =
      TTIImpl->getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getExtractWithExtendCost(
    unsigned Opcode, Type *Dst, VectorType *VecTy, unsigned Index) const {
  InstructionCost Cost =
      TTIImpl->getExtractWithExtendCost(Opcode, Dst, VecTy, Index);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getCFInstrCost(
    unsigned Opcode, TTI::TargetCostKind CostKind, const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  InstructionCost Cost = TTIImpl->getCFInstrCost(Opcode, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getCmpSelInstrCost(
    unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
    TTI::TargetCostKind CostKind, const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  InstructionCost Cost =
      TTIImpl->getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getVectorInstrCost(unsigned Opcode,
                                                        Type *Val,
                                                        unsigned Index) const {
  // FIXME: Assert that Opcode is either InsertElement or ExtractElement.
  // This is mentioned in the interface description and respected by all
  // callers, but never asserted upon.
  InstructionCost Cost = TTIImpl->getVectorInstrCost(Opcode, Val, Index);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getVectorInstrCost(const Instruction &I,
                                                        Type *Val,
                                                        unsigned Index) const {
  // FIXME: Assert that Opcode is either InsertElement or ExtractElement.
  // This is mentioned in the interface description and respected by all
  // callers, but never asserted upon.
  InstructionCost Cost = TTIImpl->getVectorInstrCost(I, Val, Index);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getReplicationShuffleCost(
    Type *EltTy, int ReplicationFactor, int VF, const APInt &DemandedDstElts,
    TTI::TargetCostKind CostKind) {
  InstructionCost Cost = TTIImpl->getReplicationShuffleCost(
      EltTy, ReplicationFactor, VF, DemandedDstElts, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getMemoryOpCost(
    unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
    TTI::TargetCostKind CostKind, const Instruction *I) const {
  assert((I == nullptr || I->getOpcode() == Opcode) &&
         "Opcode should reflect passed instruction.");
  InstructionCost Cost = TTIImpl->getMemoryOpCost(Opcode, Src, Alignment,
                                                  AddressSpace, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getMaskedMemoryOpCost(
    unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
    TTI::TargetCostKind CostKind) const {
  InstructionCost Cost = TTIImpl->getMaskedMemoryOpCost(Opcode, Src, Alignment,
                                                        AddressSpace, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) const {
  InstructionCost Cost = TTIImpl->getGatherScatterOpCost(
      Opcode, DataTy, Ptr, VariableMask, Alignment, CostKind, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) const {
  InstructionCost Cost = TTIImpl->getInterleavedMemoryOpCost(
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

InstructionCost
TargetTransformInfo::getCallInstrCost(Function *F, Type *RetTy,
                                      ArrayRef<Type *> Tys,
                                      TTI::TargetCostKind CostKind) const {
  InstructionCost Cost = TTIImpl->getCallInstrCost(F, RetTy, Tys, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

unsigned TargetTransformInfo::getNumberOfParts(Type *Tp) const {
  return TTIImpl->getNumberOfParts(Tp);
}

InstructionCost
TargetTransformInfo::getAddressComputationCost(Type *Tp, ScalarEvolution *SE,
                                               const SCEV *Ptr) const {
  InstructionCost Cost = TTIImpl->getAddressComputationCost(Tp, SE, Ptr);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getMemcpyCost(const Instruction *I) const {
  InstructionCost Cost = TTIImpl->getMemcpyCost(I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getArithmeticReductionCost(
    unsigned Opcode, VectorType *Ty, Optional<FastMathFlags> FMF,
    TTI::TargetCostKind CostKind) const {
  InstructionCost Cost =
      TTIImpl->getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getMinMaxReductionCost(
    VectorType *Ty, VectorType *CondTy, bool IsUnsigned,
    TTI::TargetCostKind CostKind) const {
  InstructionCost Cost =
      TTIImpl->getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

InstructionCost TargetTransformInfo::getExtendedReductionCost(
    unsigned Opcode, bool IsUnsigned, Type *ResTy, VectorType *Ty,
    Optional<FastMathFlags> FMF, TTI::TargetCostKind CostKind) const {
  return TTIImpl->getExtendedReductionCost(Opcode, IsUnsigned, ResTy, Ty, FMF,
                                           CostKind);
}

InstructionCost TargetTransformInfo::getMulAccReductionCost(
    bool IsUnsigned, Type *ResTy, VectorType *Ty,
    TTI::TargetCostKind CostKind) const {
  return TTIImpl->getMulAccReductionCost(IsUnsigned, ResTy, Ty, CostKind);
}

InstructionCost
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
    unsigned DestAddrSpace, unsigned SrcAlign, unsigned DestAlign,
    Optional<uint32_t> AtomicElementSize) const {
  return TTIImpl->getMemcpyLoopLoweringType(Context, Length, SrcAddrSpace,
                                            DestAddrSpace, SrcAlign, DestAlign,
                                            AtomicElementSize);
}

void TargetTransformInfo::getMemcpyLoopResidualLoweringType(
    SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
    unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
    unsigned SrcAlign, unsigned DestAlign,
    Optional<uint32_t> AtomicCpySize) const {
  TTIImpl->getMemcpyLoopResidualLoweringType(
      OpsOut, Context, RemainingBytes, SrcAddrSpace, DestAddrSpace, SrcAlign,
      DestAlign, AtomicCpySize);
}

bool TargetTransformInfo::areInlineCompatible(const Function *Caller,
                                              const Function *Callee) const {
  return TTIImpl->areInlineCompatible(Caller, Callee);
}

bool TargetTransformInfo::areTypesABICompatible(
    const Function *Caller, const Function *Callee,
    const ArrayRef<Type *> &Types) const {
  return TTIImpl->areTypesABICompatible(Caller, Callee, Types);
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
    const RecurrenceDescriptor &RdxDesc, ElementCount VF) const {
  return TTIImpl->isLegalToVectorizeReduction(RdxDesc, VF);
}

bool TargetTransformInfo::isElementTypeLegalForScalableVector(Type *Ty) const {
  return TTIImpl->isElementTypeLegalForScalableVector(Ty);
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

TargetTransformInfo::VPLegalization
TargetTransformInfo::getVPLegalizationStrategy(const VPIntrinsic &VPI) const {
  return TTIImpl->getVPLegalizationStrategy(VPI);
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

bool TargetTransformInfo::enableScalableVectorization() const {
  return TTIImpl->enableScalableVectorization();
}

bool TargetTransformInfo::hasActiveVectorLength(unsigned Opcode, Type *DataType,
                                                Align Alignment) const {
  return TTIImpl->hasActiveVectorLength(Opcode, DataType, Alignment);
}

InstructionCost
TargetTransformInfo::getInstructionLatency(const Instruction *I) const {
  return TTIImpl->getInstructionLatency(I);
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
  case Instruction::Switch:
    return getUserCost(I, CostKind);
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

TargetTransformInfo::Concept::~Concept() = default;

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
