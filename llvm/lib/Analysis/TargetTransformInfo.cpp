//===- llvm/Analysis/TargetTransformInfo.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ErrorHandling.h"
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "tti"

namespace {
/// \brief No-op implementation of the TTI interface using the utility base
/// classes.
///
/// This is used when no target specific information is available.
struct NoTTIImpl : TargetTransformInfoImplCRTPBase<NoTTIImpl> {
  explicit NoTTIImpl(const DataLayout &DL)
      : TargetTransformInfoImplCRTPBase<NoTTIImpl>(DL) {}
};
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

int TargetTransformInfo::getOperationCost(unsigned Opcode, Type *Ty,
                                          Type *OpTy) const {
  int Cost = TTIImpl->getOperationCost(Opcode, Ty, OpTy);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCallCost(FunctionType *FTy, int NumArgs) const {
  int Cost = TTIImpl->getCallCost(FTy, NumArgs);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCallCost(const Function *F,
                                     ArrayRef<const Value *> Arguments) const {
  int Cost = TTIImpl->getCallCost(F, Arguments);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

unsigned TargetTransformInfo::getInliningThresholdMultiplier() const {
  return TTIImpl->getInliningThresholdMultiplier();
}

int TargetTransformInfo::getGEPCost(Type *PointeeType, const Value *Ptr,
                                    ArrayRef<const Value *> Operands) const {
  return TTIImpl->getGEPCost(PointeeType, Ptr, Operands);
}

int TargetTransformInfo::getIntrinsicCost(
    Intrinsic::ID IID, Type *RetTy, ArrayRef<const Value *> Arguments) const {
  int Cost = TTIImpl->getIntrinsicCost(IID, RetTy, Arguments);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

unsigned
TargetTransformInfo::getEstimatedNumberOfCaseClusters(const SwitchInst &SI,
                                                      unsigned &JTSize) const {
  return TTIImpl->getEstimatedNumberOfCaseClusters(SI, JTSize);
}

int TargetTransformInfo::getUserCost(const User *U) const {
  int Cost = TTIImpl->getUserCost(U);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

bool TargetTransformInfo::hasBranchDivergence() const {
  return TTIImpl->hasBranchDivergence();
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

bool TargetTransformInfo::isLoweredToCall(const Function *F) const {
  return TTIImpl->isLoweredToCall(F);
}

void TargetTransformInfo::getUnrollingPreferences(
    Loop *L, UnrollingPreferences &UP) const {
  return TTIImpl->getUnrollingPreferences(L, UP);
}

bool TargetTransformInfo::isLegalAddImmediate(int64_t Imm) const {
  return TTIImpl->isLegalAddImmediate(Imm);
}

bool TargetTransformInfo::isLegalICmpImmediate(int64_t Imm) const {
  return TTIImpl->isLegalICmpImmediate(Imm);
}

bool TargetTransformInfo::isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                                int64_t BaseOffset,
                                                bool HasBaseReg,
                                                int64_t Scale,
                                                unsigned AddrSpace) const {
  return TTIImpl->isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                        Scale, AddrSpace);
}

bool TargetTransformInfo::isLSRCostLess(LSRCost &C1, LSRCost &C2) const {
  return TTIImpl->isLSRCostLess(C1, C2);
}

bool TargetTransformInfo::isLegalMaskedStore(Type *DataType) const {
  return TTIImpl->isLegalMaskedStore(DataType);
}

bool TargetTransformInfo::isLegalMaskedLoad(Type *DataType) const {
  return TTIImpl->isLegalMaskedLoad(DataType);
}

bool TargetTransformInfo::isLegalMaskedGather(Type *DataType) const {
  return TTIImpl->isLegalMaskedGather(DataType);
}

bool TargetTransformInfo::isLegalMaskedScatter(Type *DataType) const {
  return TTIImpl->isLegalMaskedGather(DataType);
}

bool TargetTransformInfo::prefersVectorizedAddressing() const {
  return TTIImpl->prefersVectorizedAddressing();
}

int TargetTransformInfo::getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                              int64_t BaseOffset,
                                              bool HasBaseReg,
                                              int64_t Scale,
                                              unsigned AddrSpace) const {
  int Cost = TTIImpl->getScalingFactorCost(Ty, BaseGV, BaseOffset, HasBaseReg,
                                           Scale, AddrSpace);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

bool TargetTransformInfo::isFoldableMemAccessOffset(Instruction *I,
                                                    int64_t Offset) const {
  return TTIImpl->isFoldableMemAccessOffset(I, Offset);
}

bool TargetTransformInfo::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return TTIImpl->isTruncateFree(Ty1, Ty2);
}

bool TargetTransformInfo::isProfitableToHoist(Instruction *I) const {
  return TTIImpl->isProfitableToHoist(I);
}

bool TargetTransformInfo::isTypeLegal(Type *Ty) const {
  return TTIImpl->isTypeLegal(Ty);
}

unsigned TargetTransformInfo::getJumpBufAlignment() const {
  return TTIImpl->getJumpBufAlignment();
}

unsigned TargetTransformInfo::getJumpBufSize() const {
  return TTIImpl->getJumpBufSize();
}

bool TargetTransformInfo::shouldBuildLookupTables() const {
  return TTIImpl->shouldBuildLookupTables();
}
bool TargetTransformInfo::shouldBuildLookupTablesForConstant(Constant *C) const {
  return TTIImpl->shouldBuildLookupTablesForConstant(C);
}

unsigned TargetTransformInfo::
getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const {
  return TTIImpl->getScalarizationOverhead(Ty, Insert, Extract);
}

unsigned TargetTransformInfo::
getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                 unsigned VF) const {
  return TTIImpl->getOperandsScalarizationOverhead(Args, VF);
}

bool TargetTransformInfo::supportsEfficientVectorElementLoadStore() const {
  return TTIImpl->supportsEfficientVectorElementLoadStore();
}

bool TargetTransformInfo::enableAggressiveInterleaving(bool LoopHasReductions) const {
  return TTIImpl->enableAggressiveInterleaving(LoopHasReductions);
}

bool TargetTransformInfo::expandMemCmp(Instruction *I, unsigned &MaxLoadSize) const {
  return TTIImpl->expandMemCmp(I, MaxLoadSize);
}

bool TargetTransformInfo::enableInterleavedAccessVectorization() const {
  return TTIImpl->enableInterleavedAccessVectorization();
}

bool TargetTransformInfo::isFPVectorizationPotentiallyUnsafe() const {
  return TTIImpl->isFPVectorizationPotentiallyUnsafe();
}

bool TargetTransformInfo::allowsMisalignedMemoryAccesses(LLVMContext &Context,
                                                         unsigned BitWidth,
                                                         unsigned AddressSpace,
                                                         unsigned Alignment,
                                                         bool *Fast) const {
  return TTIImpl->allowsMisalignedMemoryAccesses(Context, BitWidth, AddressSpace,
                                                 Alignment, Fast);
}

TargetTransformInfo::PopcntSupportKind
TargetTransformInfo::getPopcntSupport(unsigned IntTyWidthInBit) const {
  return TTIImpl->getPopcntSupport(IntTyWidthInBit);
}

bool TargetTransformInfo::haveFastSqrt(Type *Ty) const {
  return TTIImpl->haveFastSqrt(Ty);
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

int TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty) const {
  int Cost = TTIImpl->getIntImmCost(Imm, Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntImmCost(unsigned Opcode, unsigned Idx,
                                       const APInt &Imm, Type *Ty) const {
  int Cost = TTIImpl->getIntImmCost(Opcode, Idx, Imm, Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                       const APInt &Imm, Type *Ty) const {
  int Cost = TTIImpl->getIntImmCost(IID, Idx, Imm, Ty);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

unsigned TargetTransformInfo::getNumberOfRegisters(bool Vector) const {
  return TTIImpl->getNumberOfRegisters(Vector);
}

unsigned TargetTransformInfo::getRegisterBitWidth(bool Vector) const {
  return TTIImpl->getRegisterBitWidth(Vector);
}

unsigned TargetTransformInfo::getMinVectorRegisterBitWidth() const {
  return TTIImpl->getMinVectorRegisterBitWidth();
}

bool TargetTransformInfo::shouldConsiderAddressTypePromotion(
    const Instruction &I, bool &AllowPromotionWithoutCommonHeader) const {
  return TTIImpl->shouldConsiderAddressTypePromotion(
      I, AllowPromotionWithoutCommonHeader);
}

unsigned TargetTransformInfo::getCacheLineSize() const {
  return TTIImpl->getCacheLineSize();
}

unsigned TargetTransformInfo::getPrefetchDistance() const {
  return TTIImpl->getPrefetchDistance();
}

unsigned TargetTransformInfo::getMinPrefetchStride() const {
  return TTIImpl->getMinPrefetchStride();
}

unsigned TargetTransformInfo::getMaxPrefetchIterationsAhead() const {
  return TTIImpl->getMaxPrefetchIterationsAhead();
}

unsigned TargetTransformInfo::getMaxInterleaveFactor(unsigned VF) const {
  return TTIImpl->getMaxInterleaveFactor(VF);
}

int TargetTransformInfo::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, OperandValueKind Opd1Info,
    OperandValueKind Opd2Info, OperandValueProperties Opd1PropInfo,
    OperandValueProperties Opd2PropInfo,
    ArrayRef<const Value *> Args) const {
  int Cost = TTIImpl->getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                             Opd1PropInfo, Opd2PropInfo, Args);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getShuffleCost(ShuffleKind Kind, Type *Ty, int Index,
                                        Type *SubTp) const {
  int Cost = TTIImpl->getShuffleCost(Kind, Ty, Index, SubTp);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCastInstrCost(unsigned Opcode, Type *Dst,
                                 Type *Src, const Instruction *I) const {
  assert ((I == nullptr || I->getOpcode() == Opcode) &&
          "Opcode should reflect passed instruction.");
  int Cost = TTIImpl->getCastInstrCost(Opcode, Dst, Src, I);
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

int TargetTransformInfo::getCFInstrCost(unsigned Opcode) const {
  int Cost = TTIImpl->getCFInstrCost(Opcode);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                 Type *CondTy, const Instruction *I) const {
  assert ((I == nullptr || I->getOpcode() == Opcode) &&
          "Opcode should reflect passed instruction.");
  int Cost = TTIImpl->getCmpSelInstrCost(Opcode, ValTy, CondTy, I);
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
                                         unsigned Alignment,
                                         unsigned AddressSpace,
                                         const Instruction *I) const {
  assert ((I == nullptr || I->getOpcode() == Opcode) &&
          "Opcode should reflect passed instruction.");
  int Cost = TTIImpl->getMemoryOpCost(Opcode, Src, Alignment, AddressSpace, I);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                               unsigned Alignment,
                                               unsigned AddressSpace) const {
  int Cost =
      TTIImpl->getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                                Value *Ptr, bool VariableMask,
                                                unsigned Alignment) const {
  int Cost = TTIImpl->getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                             Alignment);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    unsigned Alignment, unsigned AddressSpace) const {
  int Cost = TTIImpl->getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                                 Alignment, AddressSpace);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                    ArrayRef<Type *> Tys, FastMathFlags FMF,
                                    unsigned ScalarizationCostPassed) const {
  int Cost = TTIImpl->getIntrinsicInstrCost(ID, RetTy, Tys, FMF,
                                            ScalarizationCostPassed);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
           ArrayRef<Value *> Args, FastMathFlags FMF, unsigned VF) const {
  int Cost = TTIImpl->getIntrinsicInstrCost(ID, RetTy, Args, FMF, VF);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
}

int TargetTransformInfo::getCallInstrCost(Function *F, Type *RetTy,
                                          ArrayRef<Type *> Tys) const {
  int Cost = TTIImpl->getCallInstrCost(F, RetTy, Tys);
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

int TargetTransformInfo::getReductionCost(unsigned Opcode, Type *Ty,
                                          bool IsPairwiseForm) const {
  int Cost = TTIImpl->getReductionCost(Opcode, Ty, IsPairwiseForm);
  assert(Cost >= 0 && "TTI should not produce negative costs!");
  return Cost;
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

bool TargetTransformInfo::areInlineCompatible(const Function *Caller,
                                              const Function *Callee) const {
  return TTIImpl->areInlineCompatible(Caller, Callee);
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
    unsigned ChainSizeInBytes, unsigned Alignment, unsigned AddrSpace) const {
  return TTIImpl->isLegalToVectorizeLoadChain(ChainSizeInBytes, Alignment,
                                              AddrSpace);
}

bool TargetTransformInfo::isLegalToVectorizeStoreChain(
    unsigned ChainSizeInBytes, unsigned Alignment, unsigned AddrSpace) const {
  return TTIImpl->isLegalToVectorizeStoreChain(ChainSizeInBytes, Alignment,
                                               AddrSpace);
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

bool TargetTransformInfo::useReductionIntrinsic(unsigned Opcode,
                                                Type *Ty, ReductionFlags Flags) const {
  return TTIImpl->useReductionIntrinsic(Opcode, Ty, Flags);
}

bool TargetTransformInfo::shouldExpandReduction(const IntrinsicInst *II) const {
  return TTIImpl->shouldExpandReduction(II);
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
