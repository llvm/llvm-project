//===- TargetTransformInfoImpl.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides helpers for the implementation of
/// a TargetTransformInfo-conforming class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETTRANSFORMINFOIMPL_H
#define LLVM_ANALYSIS_TARGETTRANSFORMINFOIMPL_H

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include <utility>

namespace llvm {

class Function;

/// Base class for use as a mix-in that aids implementing
/// a TargetTransformInfo-compatible class.
class TargetTransformInfoImplBase {
protected:
  typedef TargetTransformInfo TTI;

  const DataLayout &DL;

  explicit TargetTransformInfoImplBase(const DataLayout &DL) : DL(DL) {}

public:
  // Provide value semantics. MSVC requires that we spell all of these out.
  TargetTransformInfoImplBase(const TargetTransformInfoImplBase &Arg) = default;
  TargetTransformInfoImplBase(TargetTransformInfoImplBase &&Arg) : DL(Arg.DL) {}

  const DataLayout &getDataLayout() const { return DL; }

  InstructionCost getGEPCost(Type *PointeeType, const Value *Ptr,
                             ArrayRef<const Value *> Operands,
                             TTI::TargetCostKind CostKind) const {
    // In the basic model, we just assume that all-constant GEPs will be folded
    // into their uses via addressing modes.
    for (unsigned Idx = 0, Size = Operands.size(); Idx != Size; ++Idx)
      if (!isa<Constant>(Operands[Idx]))
        return TTI::TCC_Basic;

    return TTI::TCC_Free;
  }

  unsigned getEstimatedNumberOfCaseClusters(const SwitchInst &SI,
                                            unsigned &JTSize,
                                            ProfileSummaryInfo *PSI,
                                            BlockFrequencyInfo *BFI) const {
    (void)PSI;
    (void)BFI;
    JTSize = 0;
    return SI.getNumCases();
  }

  unsigned getInliningThresholdMultiplier() const { return 1; }
  unsigned adjustInliningThreshold(const CallBase *CB) const { return 0; }

  int getInlinerVectorBonusPercent() const { return 150; }

  InstructionCost getMemcpyCost(const Instruction *I) const {
    return TTI::TCC_Expensive;
  }

  // Although this default value is arbitrary, it is not random. It is assumed
  // that a condition that evaluates the same way by a higher percentage than
  // this is best represented as control flow. Therefore, the default value N
  // should be set such that the win from N% correct executions is greater than
  // the loss from (100 - N)% mispredicted executions for the majority of
  //  intended targets.
  BranchProbability getPredictableBranchThreshold() const {
    return BranchProbability(99, 100);
  }

  bool hasBranchDivergence() const { return false; }

  bool useGPUDivergenceAnalysis() const { return false; }

  bool isSourceOfDivergence(const Value *V) const { return false; }

  bool isAlwaysUniform(const Value *V) const { return false; }

  unsigned getFlatAddressSpace() const { return -1; }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const {
    return false;
  }

  bool isNoopAddrSpaceCast(unsigned, unsigned) const { return false; }
  bool canHaveNonUndefGlobalInitializerInAddressSpace(unsigned AS) const {
    return AS == 0;
  };

  unsigned getAssumedAddrSpace(const Value *V) const { return -1; }

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const {
    return std::make_pair(nullptr, -1);
  }

  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const {
    return nullptr;
  }

  bool isLoweredToCall(const Function *F) const {
    assert(F && "A concrete function must be provided to this routine.");

    // FIXME: These should almost certainly not be handled here, and instead
    // handled with the help of TLI or the target itself. This was largely
    // ported from existing analysis heuristics here so that such refactorings
    // can take place in the future.

    if (F->isIntrinsic())
      return false;

    if (F->hasLocalLinkage() || !F->hasName())
      return true;

    StringRef Name = F->getName();

    // These will all likely lower to a single selection DAG node.
    if (Name == "copysign" || Name == "copysignf" || Name == "copysignl" ||
        Name == "fabs" || Name == "fabsf" || Name == "fabsl" || Name == "sin" ||
        Name == "fmin" || Name == "fminf" || Name == "fminl" ||
        Name == "fmax" || Name == "fmaxf" || Name == "fmaxl" ||
        Name == "sinf" || Name == "sinl" || Name == "cos" || Name == "cosf" ||
        Name == "cosl" || Name == "sqrt" || Name == "sqrtf" || Name == "sqrtl")
      return false;

    // These are all likely to be optimized into something smaller.
    if (Name == "pow" || Name == "powf" || Name == "powl" || Name == "exp2" ||
        Name == "exp2l" || Name == "exp2f" || Name == "floor" ||
        Name == "floorf" || Name == "ceil" || Name == "round" ||
        Name == "ffs" || Name == "ffsl" || Name == "abs" || Name == "labs" ||
        Name == "llabs")
      return false;

    return true;
  }

  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC, TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo) const {
    return false;
  }

  bool preferPredicateOverEpilogue(Loop *L, LoopInfo *LI, ScalarEvolution &SE,
                                   AssumptionCache &AC, TargetLibraryInfo *TLI,
                                   DominatorTree *DT,
                                   LoopVectorizationLegality *LVL,
                                   InterleavedAccessInfo *IAI) const {
    return false;
  }

  PredicationStyle emitGetActiveLaneMask() const {
    return PredicationStyle::None;
  }

  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const {
    return None;
  }

  Optional<Value *>
  simplifyDemandedUseBitsIntrinsic(InstCombiner &IC, IntrinsicInst &II,
                                   APInt DemandedMask, KnownBits &Known,
                                   bool &KnownBitsComputed) const {
    return None;
  }

  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const {
    return None;
  }

  void getUnrollingPreferences(Loop *, ScalarEvolution &,
                               TTI::UnrollingPreferences &,
                               OptimizationRemarkEmitter *) const {}

  void getPeelingPreferences(Loop *, ScalarEvolution &,
                             TTI::PeelingPreferences &) const {}

  bool isLegalAddImmediate(int64_t Imm) const { return false; }

  bool isLegalICmpImmediate(int64_t Imm) const { return false; }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale, unsigned AddrSpace,
                             Instruction *I = nullptr) const {
    // Guess that only reg and reg+reg addressing is allowed. This heuristic is
    // taken from the implementation of LSR.
    return !BaseGV && BaseOffset == 0 && (Scale == 0 || Scale == 1);
  }

  bool isLSRCostLess(const TTI::LSRCost &C1, const TTI::LSRCost &C2) const {
    return std::tie(C1.NumRegs, C1.AddRecCost, C1.NumIVMuls, C1.NumBaseAdds,
                    C1.ScaleCost, C1.ImmCost, C1.SetupCost) <
           std::tie(C2.NumRegs, C2.AddRecCost, C2.NumIVMuls, C2.NumBaseAdds,
                    C2.ScaleCost, C2.ImmCost, C2.SetupCost);
  }

  bool isNumRegsMajorCostOfLSR() const { return true; }

  bool isProfitableLSRChainElement(Instruction *I) const { return false; }

  bool canMacroFuseCmp() const { return false; }

  bool canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE, LoopInfo *LI,
                  DominatorTree *DT, AssumptionCache *AC,
                  TargetLibraryInfo *LibInfo) const {
    return false;
  }

  TTI::AddressingModeKind
    getPreferredAddressingMode(const Loop *L, ScalarEvolution *SE) const {
    return TTI::AMK_None;
  }

  bool isLegalMaskedStore(Type *DataType, Align Alignment) const {
    return false;
  }

  bool isLegalMaskedLoad(Type *DataType, Align Alignment) const {
    return false;
  }

  bool isLegalNTStore(Type *DataType, Align Alignment) const {
    // By default, assume nontemporal memory stores are available for stores
    // that are aligned and have a size that is a power of 2.
    unsigned DataSize = DL.getTypeStoreSize(DataType);
    return Alignment >= DataSize && isPowerOf2_32(DataSize);
  }

  bool isLegalNTLoad(Type *DataType, Align Alignment) const {
    // By default, assume nontemporal memory loads are available for loads that
    // are aligned and have a size that is a power of 2.
    unsigned DataSize = DL.getTypeStoreSize(DataType);
    return Alignment >= DataSize && isPowerOf2_32(DataSize);
  }

  bool isLegalBroadcastLoad(Type *ElementTy, ElementCount NumElements) const {
    return false;
  }

  bool isLegalMaskedScatter(Type *DataType, Align Alignment) const {
    return false;
  }

  bool isLegalMaskedGather(Type *DataType, Align Alignment) const {
    return false;
  }

  bool forceScalarizeMaskedGather(VectorType *DataType, Align Alignment) const {
    return false;
  }

  bool forceScalarizeMaskedScatter(VectorType *DataType,
                                   Align Alignment) const {
    return false;
  }

  bool isLegalMaskedCompressStore(Type *DataType) const { return false; }

  bool isLegalAltInstr(VectorType *VecTy, unsigned Opcode0, unsigned Opcode1,
                       const SmallBitVector &OpcodeMask) const {
    return false;
  }

  bool isLegalMaskedExpandLoad(Type *DataType) const { return false; }

  bool enableOrderedReductions() const { return false; }

  bool hasDivRemOp(Type *DataType, bool IsSigned) const { return false; }

  bool hasVolatileVariant(Instruction *I, unsigned AddrSpace) const {
    return false;
  }

  bool prefersVectorizedAddressing() const { return true; }

  InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                       int64_t BaseOffset, bool HasBaseReg,
                                       int64_t Scale,
                                       unsigned AddrSpace) const {
    // Guess that all legal addressing mode are free.
    if (isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg, Scale,
                              AddrSpace))
      return 0;
    return -1;
  }

  bool LSRWithInstrQueries() const { return false; }

  bool isTruncateFree(Type *Ty1, Type *Ty2) const { return false; }

  bool isProfitableToHoist(Instruction *I) const { return true; }

  bool useAA() const { return false; }

  bool isTypeLegal(Type *Ty) const { return false; }

  unsigned getRegUsageForType(Type *Ty) const { return 1; }

  bool shouldBuildLookupTables() const { return true; }

  bool shouldBuildLookupTablesForConstant(Constant *C) const { return true; }

  bool shouldBuildRelLookupTables() const { return false; }

  bool useColdCCForColdCall(Function &F) const { return false; }

  InstructionCost getScalarizationOverhead(VectorType *Ty,
                                           const APInt &DemandedElts,
                                           bool Insert, bool Extract) const {
    return 0;
  }

  InstructionCost getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                                   ArrayRef<Type *> Tys) const {
    return 0;
  }

  bool supportsEfficientVectorElementLoadStore() const { return false; }

  bool supportsTailCalls() const { return true; }

  bool enableAggressiveInterleaving(bool LoopHasReductions) const {
    return false;
  }

  TTI::MemCmpExpansionOptions enableMemCmpExpansion(bool OptSize,
                                                    bool IsZeroCmp) const {
    return {};
  }

  bool enableInterleavedAccessVectorization() const { return false; }

  bool enableMaskedInterleavedAccessVectorization() const { return false; }

  bool isFPVectorizationPotentiallyUnsafe() const { return false; }

  bool allowsMisalignedMemoryAccesses(LLVMContext &Context, unsigned BitWidth,
                                      unsigned AddressSpace, Align Alignment,
                                      bool *Fast) const {
    return false;
  }

  TTI::PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const {
    return TTI::PSK_Software;
  }

  bool haveFastSqrt(Type *Ty) const { return false; }

  bool isFCmpOrdCheaperThanFCmpZero(Type *Ty) const { return true; }

  InstructionCost getFPOpCost(Type *Ty) const {
    return TargetTransformInfo::TCC_Basic;
  }

  InstructionCost getIntImmCodeSizeCost(unsigned Opcode, unsigned Idx,
                                        const APInt &Imm, Type *Ty) const {
    return 0;
  }

  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) const {
    return TTI::TCC_Basic;
  }

  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) const {
    return TTI::TCC_Free;
  }

  InstructionCost getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                      const APInt &Imm, Type *Ty,
                                      TTI::TargetCostKind CostKind) const {
    return TTI::TCC_Free;
  }

  unsigned getNumberOfRegisters(unsigned ClassID) const { return 8; }

  unsigned getRegisterClassForType(bool Vector, Type *Ty = nullptr) const {
    return Vector ? 1 : 0;
  };

  const char *getRegisterClassName(unsigned ClassID) const {
    switch (ClassID) {
    default:
      return "Generic::Unknown Register Class";
    case 0:
      return "Generic::ScalarRC";
    case 1:
      return "Generic::VectorRC";
    }
  }

  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
    return TypeSize::getFixed(32);
  }

  unsigned getMinVectorRegisterBitWidth() const { return 128; }

  Optional<unsigned> getMaxVScale() const { return None; }
  Optional<unsigned> getVScaleForTuning() const { return None; }

  bool
  shouldMaximizeVectorBandwidth(TargetTransformInfo::RegisterKind K) const {
    return false;
  }

  ElementCount getMinimumVF(unsigned ElemWidth, bool IsScalable) const {
    return ElementCount::get(0, IsScalable);
  }

  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const { return 0; }
  unsigned getStoreMinimumVF(unsigned VF, Type *, Type *) const { return VF; }

  bool shouldConsiderAddressTypePromotion(
      const Instruction &I, bool &AllowPromotionWithoutCommonHeader) const {
    AllowPromotionWithoutCommonHeader = false;
    return false;
  }

  unsigned getCacheLineSize() const { return 0; }

  llvm::Optional<unsigned>
  getCacheSize(TargetTransformInfo::CacheLevel Level) const {
    switch (Level) {
    case TargetTransformInfo::CacheLevel::L1D:
      LLVM_FALLTHROUGH;
    case TargetTransformInfo::CacheLevel::L2D:
      return llvm::Optional<unsigned>();
    }
    llvm_unreachable("Unknown TargetTransformInfo::CacheLevel");
  }

  llvm::Optional<unsigned>
  getCacheAssociativity(TargetTransformInfo::CacheLevel Level) const {
    switch (Level) {
    case TargetTransformInfo::CacheLevel::L1D:
      LLVM_FALLTHROUGH;
    case TargetTransformInfo::CacheLevel::L2D:
      return llvm::Optional<unsigned>();
    }

    llvm_unreachable("Unknown TargetTransformInfo::CacheLevel");
  }

  unsigned getPrefetchDistance() const { return 0; }
  unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                unsigned NumStridedMemAccesses,
                                unsigned NumPrefetches, bool HasCall) const {
    return 1;
  }
  unsigned getMaxPrefetchIterationsAhead() const { return UINT_MAX; }
  bool enableWritePrefetching() const { return false; }
  bool shouldPrefetchAddressSpace(unsigned AS) const { return !AS; }

  unsigned getMaxInterleaveFactor(unsigned VF) const { return 1; }

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueKind Opd1Info, TTI::OperandValueKind Opd2Info,
      TTI::OperandValueProperties Opd1PropInfo,
      TTI::OperandValueProperties Opd2PropInfo, ArrayRef<const Value *> Args,
      const Instruction *CxtI = nullptr) const {
    // FIXME: A number of transformation tests seem to require these values
    // which seems a little odd for how arbitary there are.
    switch (Opcode) {
    default:
      break;
    case Instruction::FDiv:
    case Instruction::FRem:
    case Instruction::SDiv:
    case Instruction::SRem:
    case Instruction::UDiv:
    case Instruction::URem:
      // FIXME: Unlikely to be true for CodeSize.
      return TTI::TCC_Expensive;
    }
    return 1;
  }

  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Ty,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp,
                                 ArrayRef<const Value *> Args = None) const {
    return 1;
  }

  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   TTI::CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I) const {
    switch (Opcode) {
    default:
      break;
    case Instruction::IntToPtr: {
      unsigned SrcSize = Src->getScalarSizeInBits();
      if (DL.isLegalInteger(SrcSize) &&
          SrcSize <= DL.getPointerTypeSizeInBits(Dst))
        return 0;
      break;
    }
    case Instruction::PtrToInt: {
      unsigned DstSize = Dst->getScalarSizeInBits();
      if (DL.isLegalInteger(DstSize) &&
          DstSize >= DL.getPointerTypeSizeInBits(Src))
        return 0;
      break;
    }
    case Instruction::BitCast:
      if (Dst == Src || (Dst->isPointerTy() && Src->isPointerTy()))
        // Identity and pointer-to-pointer casts are free.
        return 0;
      break;
    case Instruction::Trunc: {
      // trunc to a native type is free (assuming the target has compare and
      // shift-right of the same width).
      TypeSize DstSize = DL.getTypeSizeInBits(Dst);
      if (!DstSize.isScalable() && DL.isLegalInteger(DstSize.getFixedSize()))
        return 0;
      break;
    }
    }
    return 1;
  }

  InstructionCost getExtractWithExtendCost(unsigned Opcode, Type *Dst,
                                           VectorType *VecTy,
                                           unsigned Index) const {
    return 1;
  }

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) const {
    // A phi would be free, unless we're costing the throughput because it
    // will require a register.
    if (Opcode == Instruction::PHI && CostKind != TTI::TCK_RecipThroughput)
      return 0;
    return 1;
  }

  InstructionCost getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                     CmpInst::Predicate VecPred,
                                     TTI::TargetCostKind CostKind,
                                     const Instruction *I) const {
    return 1;
  }

  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     unsigned Index) const {
    return 1;
  }

  InstructionCost getVectorInstrCost(const Instruction *I, Type *Val,
                                     unsigned Index) const {
    return 1;
  }

  unsigned getReplicationShuffleCost(Type *EltTy, int ReplicationFactor, int VF,
                                     const APInt &DemandedDstElts,
                                     TTI::TargetCostKind CostKind) {
    return 1;
  }

  InstructionCost getMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                  unsigned AddressSpace,
                                  TTI::TargetCostKind CostKind,
                                  const Instruction *I) const {
    return 1;
  }

  InstructionCost getVPMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                    unsigned AddressSpace,
                                    TTI::TargetCostKind CostKind,
                                    const Instruction *I) const {
    return 1;
  }

  InstructionCost getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                        Align Alignment, unsigned AddressSpace,
                                        TTI::TargetCostKind CostKind) const {
    return 1;
  }

  InstructionCost getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                         const Value *Ptr, bool VariableMask,
                                         Align Alignment,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I = nullptr) const {
    return 1;
  }

  unsigned getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond, bool UseMaskForGaps) const {
    return 1;
  }

  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind) const {
    switch (ICA.getID()) {
    default:
      break;
    case Intrinsic::annotation:
    case Intrinsic::assume:
    case Intrinsic::sideeffect:
    case Intrinsic::pseudoprobe:
    case Intrinsic::arithmetic_fence:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::dbg_label:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::launder_invariant_group:
    case Intrinsic::strip_invariant_group:
    case Intrinsic::is_constant:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::experimental_noalias_scope_decl:
    case Intrinsic::objectsize:
    case Intrinsic::ptr_annotation:
    case Intrinsic::var_annotation:
    case Intrinsic::experimental_gc_result:
    case Intrinsic::experimental_gc_relocate:
    case Intrinsic::coro_alloc:
    case Intrinsic::coro_begin:
    case Intrinsic::coro_free:
    case Intrinsic::coro_end:
    case Intrinsic::coro_frame:
    case Intrinsic::coro_size:
    case Intrinsic::coro_align:
    case Intrinsic::coro_suspend:
    case Intrinsic::coro_subfn_addr:
    case Intrinsic::threadlocal_address:
      // These intrinsics don't actually represent code after lowering.
      return 0;
    }
    return 1;
  }

  InstructionCost getCallInstrCost(Function *F, Type *RetTy,
                                   ArrayRef<Type *> Tys,
                                   TTI::TargetCostKind CostKind) const {
    return 1;
  }

  // Assume that we have a register of the right size for the type.
  unsigned getNumberOfParts(Type *Tp) const { return 1; }

  InstructionCost getAddressComputationCost(Type *Tp, ScalarEvolution *,
                                            const SCEV *) const {
    return 0;
  }

  InstructionCost getArithmeticReductionCost(unsigned, VectorType *,
                                             Optional<FastMathFlags> FMF,
                                             TTI::TargetCostKind) const {
    return 1;
  }

  InstructionCost getMinMaxReductionCost(VectorType *, VectorType *, bool,
                                         TTI::TargetCostKind) const {
    return 1;
  }

  InstructionCost getExtendedReductionCost(unsigned Opcode, bool IsUnsigned,
                                           Type *ResTy, VectorType *Ty,
                                           Optional<FastMathFlags> FMF,
                                           TTI::TargetCostKind CostKind) const {
    return 1;
  }

  InstructionCost getMulAccReductionCost(bool IsUnsigned, Type *ResTy,
                                         VectorType *Ty,
                                         TTI::TargetCostKind CostKind) const {
    return 1;
  }

  InstructionCost getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) const {
    return 0;
  }

  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) const {
    return false;
  }

  unsigned getAtomicMemIntrinsicMaxElementSize() const {
    // Note for overrides: You must ensure for all element unordered-atomic
    // memory intrinsics that all power-of-2 element sizes up to, and
    // including, the return value of this method have a corresponding
    // runtime lib call. These runtime lib call definitions can be found
    // in RuntimeLibcalls.h
    return 0;
  }

  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType) const {
    return nullptr;
  }

  Type *getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                  unsigned SrcAddrSpace, unsigned DestAddrSpace,
                                  unsigned SrcAlign, unsigned DestAlign,
                                  Optional<uint32_t> AtomicElementSize) const {
    return AtomicElementSize ? Type::getIntNTy(Context, *AtomicElementSize * 8)
                             : Type::getInt8Ty(Context);
  }

  void getMemcpyLoopResidualLoweringType(
      SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
      unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
      unsigned SrcAlign, unsigned DestAlign,
      Optional<uint32_t> AtomicCpySize) const {
    unsigned OpSizeInBytes = AtomicCpySize ? *AtomicCpySize : 1;
    Type *OpType = Type::getIntNTy(Context, OpSizeInBytes * 8);
    for (unsigned i = 0; i != RemainingBytes; i += OpSizeInBytes)
      OpsOut.push_back(OpType);
  }

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const {
    return (Caller->getFnAttribute("target-cpu") ==
            Callee->getFnAttribute("target-cpu")) &&
           (Caller->getFnAttribute("target-features") ==
            Callee->getFnAttribute("target-features"));
  }

  bool areTypesABICompatible(const Function *Caller, const Function *Callee,
                             const ArrayRef<Type *> &Types) const {
    return (Caller->getFnAttribute("target-cpu") ==
            Callee->getFnAttribute("target-cpu")) &&
           (Caller->getFnAttribute("target-features") ==
            Callee->getFnAttribute("target-features"));
  }

  bool isIndexedLoadLegal(TTI::MemIndexedMode Mode, Type *Ty,
                          const DataLayout &DL) const {
    return false;
  }

  bool isIndexedStoreLegal(TTI::MemIndexedMode Mode, Type *Ty,
                           const DataLayout &DL) const {
    return false;
  }

  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const { return 128; }

  bool isLegalToVectorizeLoad(LoadInst *LI) const { return true; }

  bool isLegalToVectorizeStore(StoreInst *SI) const { return true; }

  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const {
    return true;
  }

  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const {
    return true;
  }

  bool isLegalToVectorizeReduction(const RecurrenceDescriptor &RdxDesc,
                                   ElementCount VF) const {
    return true;
  }

  bool isElementTypeLegalForScalableVector(Type *Ty) const { return true; }

  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const {
    return VF;
  }

  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const {
    return VF;
  }

  bool preferInLoopReduction(unsigned Opcode, Type *Ty,
                             TTI::ReductionFlags Flags) const {
    return false;
  }

  bool preferPredicatedReductionSelect(unsigned Opcode, Type *Ty,
                                       TTI::ReductionFlags Flags) const {
    return false;
  }

  bool shouldExpandReduction(const IntrinsicInst *II) const { return true; }

  unsigned getGISelRematGlobalCost() const { return 1; }

  bool supportsScalableVectors() const { return false; }

  bool enableScalableVectorization() const { return false; }

  bool hasActiveVectorLength(unsigned Opcode, Type *DataType,
                             Align Alignment) const {
    return false;
  }

  TargetTransformInfo::VPLegalization
  getVPLegalizationStrategy(const VPIntrinsic &PI) const {
    return TargetTransformInfo::VPLegalization(
        /* EVLParamStrategy */ TargetTransformInfo::VPLegalization::Discard,
        /* OperatorStrategy */ TargetTransformInfo::VPLegalization::Convert);
  }

protected:
  // Obtain the minimum required size to hold the value (without the sign)
  // In case of a vector it returns the min required size for one element.
  unsigned minRequiredElementSize(const Value *Val, bool &isSigned) const {
    if (isa<ConstantDataVector>(Val) || isa<ConstantVector>(Val)) {
      const auto *VectorValue = cast<Constant>(Val);

      // In case of a vector need to pick the max between the min
      // required size for each element
      auto *VT = cast<FixedVectorType>(Val->getType());

      // Assume unsigned elements
      isSigned = false;

      // The max required size is the size of the vector element type
      unsigned MaxRequiredSize =
          VT->getElementType()->getPrimitiveSizeInBits().getFixedSize();

      unsigned MinRequiredSize = 0;
      for (unsigned i = 0, e = VT->getNumElements(); i < e; ++i) {
        if (auto *IntElement =
                dyn_cast<ConstantInt>(VectorValue->getAggregateElement(i))) {
          bool signedElement = IntElement->getValue().isNegative();
          // Get the element min required size.
          unsigned ElementMinRequiredSize =
              IntElement->getValue().getMinSignedBits() - 1;
          // In case one element is signed then all the vector is signed.
          isSigned |= signedElement;
          // Save the max required bit size between all the elements.
          MinRequiredSize = std::max(MinRequiredSize, ElementMinRequiredSize);
        } else {
          // not an int constant element
          return MaxRequiredSize;
        }
      }
      return MinRequiredSize;
    }

    if (const auto *CI = dyn_cast<ConstantInt>(Val)) {
      isSigned = CI->getValue().isNegative();
      return CI->getValue().getMinSignedBits() - 1;
    }

    if (const auto *Cast = dyn_cast<SExtInst>(Val)) {
      isSigned = true;
      return Cast->getSrcTy()->getScalarSizeInBits() - 1;
    }

    if (const auto *Cast = dyn_cast<ZExtInst>(Val)) {
      isSigned = false;
      return Cast->getSrcTy()->getScalarSizeInBits();
    }

    isSigned = false;
    return Val->getType()->getScalarSizeInBits();
  }

  bool isStridedAccess(const SCEV *Ptr) const {
    return Ptr && isa<SCEVAddRecExpr>(Ptr);
  }

  const SCEVConstant *getConstantStrideStep(ScalarEvolution *SE,
                                            const SCEV *Ptr) const {
    if (!isStridedAccess(Ptr))
      return nullptr;
    const SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ptr);
    return dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(*SE));
  }

  bool isConstantStridedAccessLessThan(ScalarEvolution *SE, const SCEV *Ptr,
                                       int64_t MergeDistance) const {
    const SCEVConstant *Step = getConstantStrideStep(SE, Ptr);
    if (!Step)
      return false;
    APInt StrideVal = Step->getAPInt();
    if (StrideVal.getBitWidth() > 64)
      return false;
    // FIXME: Need to take absolute value for negative stride case.
    return StrideVal.getSExtValue() < MergeDistance;
  }
};

/// CRTP base class for use as a mix-in that aids implementing
/// a TargetTransformInfo-compatible class.
template <typename T>
class TargetTransformInfoImplCRTPBase : public TargetTransformInfoImplBase {
private:
  typedef TargetTransformInfoImplBase BaseT;

protected:
  explicit TargetTransformInfoImplCRTPBase(const DataLayout &DL) : BaseT(DL) {}

public:
  using BaseT::getGEPCost;

  InstructionCost getGEPCost(Type *PointeeType, const Value *Ptr,
                             ArrayRef<const Value *> Operands,
                             TTI::TargetCostKind CostKind) {
    assert(PointeeType && Ptr && "can't get GEPCost of nullptr");
    assert(cast<PointerType>(Ptr->getType()->getScalarType())
               ->isOpaqueOrPointeeTypeMatches(PointeeType) &&
           "explicit pointee type doesn't match operand's pointee type");
    auto *BaseGV = dyn_cast<GlobalValue>(Ptr->stripPointerCasts());
    bool HasBaseReg = (BaseGV == nullptr);

    auto PtrSizeBits = DL.getPointerTypeSizeInBits(Ptr->getType());
    APInt BaseOffset(PtrSizeBits, 0);
    int64_t Scale = 0;

    auto GTI = gep_type_begin(PointeeType, Operands);
    Type *TargetType = nullptr;

    // Handle the case where the GEP instruction has a single operand,
    // the basis, therefore TargetType is a nullptr.
    if (Operands.empty())
      return !BaseGV ? TTI::TCC_Free : TTI::TCC_Basic;

    for (auto I = Operands.begin(); I != Operands.end(); ++I, ++GTI) {
      TargetType = GTI.getIndexedType();
      // We assume that the cost of Scalar GEP with constant index and the
      // cost of Vector GEP with splat constant index are the same.
      const ConstantInt *ConstIdx = dyn_cast<ConstantInt>(*I);
      if (!ConstIdx)
        if (auto Splat = getSplatValue(*I))
          ConstIdx = dyn_cast<ConstantInt>(Splat);
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        // For structures the index is always splat or scalar constant
        assert(ConstIdx && "Unexpected GEP index");
        uint64_t Field = ConstIdx->getZExtValue();
        BaseOffset += DL.getStructLayout(STy)->getElementOffset(Field);
      } else {
        // If this operand is a scalable type, bail out early.
        // TODO: handle scalable vectors
        if (isa<ScalableVectorType>(TargetType))
          return TTI::TCC_Basic;
        int64_t ElementSize =
            DL.getTypeAllocSize(GTI.getIndexedType()).getFixedSize();
        if (ConstIdx) {
          BaseOffset +=
              ConstIdx->getValue().sextOrTrunc(PtrSizeBits) * ElementSize;
        } else {
          // Needs scale register.
          if (Scale != 0)
            // No addressing mode takes two scale registers.
            return TTI::TCC_Basic;
          Scale = ElementSize;
        }
      }
    }

    if (static_cast<T *>(this)->isLegalAddressingMode(
            TargetType, const_cast<GlobalValue *>(BaseGV),
            BaseOffset.sextOrTrunc(64).getSExtValue(), HasBaseReg, Scale,
            Ptr->getType()->getPointerAddressSpace()))
      return TTI::TCC_Free;
    return TTI::TCC_Basic;
  }

  InstructionCost getUserCost(const User *U, ArrayRef<const Value *> Operands,
                              TTI::TargetCostKind CostKind) {
    using namespace llvm::PatternMatch;

    auto *TargetTTI = static_cast<T *>(this);
    // Handle non-intrinsic calls, invokes, and callbr.
    // FIXME: Unlikely to be true for anything but CodeSize.
    auto *CB = dyn_cast<CallBase>(U);
    if (CB && !isa<IntrinsicInst>(U)) {
      if (const Function *F = CB->getCalledFunction()) {
        if (!TargetTTI->isLoweredToCall(F))
          return TTI::TCC_Basic; // Give a basic cost if it will be lowered

        return TTI::TCC_Basic * (F->getFunctionType()->getNumParams() + 1);
      }
      // For indirect or other calls, scale cost by number of arguments.
      return TTI::TCC_Basic * (CB->arg_size() + 1);
    }

    Type *Ty = U->getType();
    unsigned Opcode = Operator::getOpcode(U);
    auto *I = dyn_cast<Instruction>(U);
    switch (Opcode) {
    default:
      break;
    case Instruction::Call: {
      assert(isa<IntrinsicInst>(U) && "Unexpected non-intrinsic call");
      auto *Intrinsic = cast<IntrinsicInst>(U);
      IntrinsicCostAttributes CostAttrs(Intrinsic->getIntrinsicID(), *CB);
      return TargetTTI->getIntrinsicInstrCost(CostAttrs, CostKind);
    }
    case Instruction::Br:
    case Instruction::Ret:
    case Instruction::PHI:
    case Instruction::Switch:
      return TargetTTI->getCFInstrCost(Opcode, CostKind, I);
    case Instruction::ExtractValue:
    case Instruction::Freeze:
      return TTI::TCC_Free;
    case Instruction::Alloca:
      if (cast<AllocaInst>(U)->isStaticAlloca())
        return TTI::TCC_Free;
      break;
    case Instruction::GetElementPtr: {
      const auto *GEP = cast<GEPOperator>(U);
      return TargetTTI->getGEPCost(GEP->getSourceElementType(),
                                   GEP->getPointerOperand(),
                                   Operands.drop_front(), CostKind);
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
    case Instruction::Xor:
    case Instruction::FNeg: {
      TTI::OperandValueProperties Op1VP = TTI::OP_None;
      TTI::OperandValueProperties Op2VP = TTI::OP_None;
      TTI::OperandValueKind Op1VK =
        TTI::getOperandInfo(U->getOperand(0), Op1VP);
      TTI::OperandValueKind Op2VK = Opcode != Instruction::FNeg ?
        TTI::getOperandInfo(U->getOperand(1), Op2VP) : TTI::OK_AnyValue;
      SmallVector<const Value *, 2> Operands(U->operand_values());
      return TargetTTI->getArithmeticInstrCost(Opcode, Ty, CostKind,
                                               Op1VK, Op2VK,
                                               Op1VP, Op2VP, Operands, I);
    }
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast:
    case Instruction::FPExt:
    case Instruction::SExt:
    case Instruction::ZExt:
    case Instruction::AddrSpaceCast: {
      Type *OpTy = U->getOperand(0)->getType();
      return TargetTTI->getCastInstrCost(
          Opcode, Ty, OpTy, TTI::getCastContextHint(I), CostKind, I);
    }
    case Instruction::Store: {
      auto *SI = cast<StoreInst>(U);
      Type *ValTy = U->getOperand(0)->getType();
      return TargetTTI->getMemoryOpCost(Opcode, ValTy, SI->getAlign(),
                                        SI->getPointerAddressSpace(),
                                        CostKind, I);
    }
    case Instruction::Load: {
      auto *LI = cast<LoadInst>(U);
      Type *LoadType = U->getType();
      // If there is a non-register sized type, the cost estimation may expand
      // it to be several instructions to load into multiple registers on the
      // target.  But, if the only use of the load is a trunc instruction to a
      // register sized type, the instruction selector can combine these
      // instructions to be a single load.  So, in this case, we use the
      // destination type of the trunc instruction rather than the load to
      // accurately estimate the cost of this load instruction.
      if (CostKind == TTI::TCK_CodeSize && LI->hasOneUse() &&
          !LoadType->isVectorTy()) {    
        if (const TruncInst *TI = dyn_cast<TruncInst>(*LI->user_begin()))
          LoadType = TI->getDestTy();
      }
      return TargetTTI->getMemoryOpCost(Opcode, LoadType, LI->getAlign(),
                                        LI->getPointerAddressSpace(),
                                        CostKind, I);
    }
    case Instruction::Select: {
      const Value *Op0, *Op1;
      if (match(U, m_LogicalAnd(m_Value(Op0), m_Value(Op1))) ||
          match(U, m_LogicalOr(m_Value(Op0), m_Value(Op1)))) {
        // select x, y, false --> x & y
        // select x, true, y --> x | y
        TTI::OperandValueProperties Op1VP = TTI::OP_None;
        TTI::OperandValueProperties Op2VP = TTI::OP_None;
        TTI::OperandValueKind Op1VK = TTI::getOperandInfo(Op0, Op1VP);
        TTI::OperandValueKind Op2VK = TTI::getOperandInfo(Op1, Op2VP);
        assert(Op0->getType()->getScalarSizeInBits() == 1 &&
               Op1->getType()->getScalarSizeInBits() == 1);

        SmallVector<const Value *, 2> Operands{Op0, Op1};
        return TargetTTI->getArithmeticInstrCost(
            match(U, m_LogicalOr()) ? Instruction::Or : Instruction::And, Ty,
            CostKind, Op1VK, Op2VK, Op1VP, Op2VP, Operands, I);
      }
      Type *CondTy = U->getOperand(0)->getType();
      return TargetTTI->getCmpSelInstrCost(Opcode, U->getType(), CondTy,
                                           CmpInst::BAD_ICMP_PREDICATE,
                                           CostKind, I);
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      Type *ValTy = U->getOperand(0)->getType();
      // TODO: Also handle ICmp/FCmp constant expressions.
      return TargetTTI->getCmpSelInstrCost(Opcode, ValTy, U->getType(),
                                           I ? cast<CmpInst>(I)->getPredicate()
                                             : CmpInst::BAD_ICMP_PREDICATE,
                                           CostKind, I);
    }
    case Instruction::InsertElement: {
      auto *IE = dyn_cast<InsertElementInst>(U);
      if (!IE)
        return TTI::TCC_Basic; // FIXME
      unsigned Idx = -1;
      if (auto *CI = dyn_cast<ConstantInt>(IE->getOperand(2)))
        if (CI->getValue().getActiveBits() <= 32)
          Idx = CI->getZExtValue();
      return TargetTTI->getVectorInstrCost(IE, Ty, Idx);
    }
    case Instruction::ShuffleVector: {
      auto *Shuffle = dyn_cast<ShuffleVectorInst>(U);
      if (!Shuffle)
        return TTI::TCC_Basic; // FIXME

      auto *VecTy = cast<VectorType>(U->getType());
      auto *VecSrcTy = cast<VectorType>(U->getOperand(0)->getType());
      int NumSubElts, SubIndex;

      if (Shuffle->changesLength()) {
        // Treat a 'subvector widening' as a free shuffle.
        if (Shuffle->increasesLength() && Shuffle->isIdentityWithPadding())
          return 0;

        if (Shuffle->isExtractSubvectorMask(SubIndex))
          return TargetTTI->getShuffleCost(TTI::SK_ExtractSubvector, VecSrcTy,
                                           Shuffle->getShuffleMask(), SubIndex,
                                           VecTy, Operands);

        if (Shuffle->isInsertSubvectorMask(NumSubElts, SubIndex))
          return TargetTTI->getShuffleCost(
              TTI::SK_InsertSubvector, VecTy, Shuffle->getShuffleMask(),
              SubIndex,
              FixedVectorType::get(VecTy->getScalarType(), NumSubElts),
              Operands);

        int ReplicationFactor, VF;
        if (Shuffle->isReplicationMask(ReplicationFactor, VF)) {
          APInt DemandedDstElts =
              APInt::getNullValue(Shuffle->getShuffleMask().size());
          for (auto I : enumerate(Shuffle->getShuffleMask())) {
            if (I.value() != UndefMaskElem)
              DemandedDstElts.setBit(I.index());
          }
          return TargetTTI->getReplicationShuffleCost(
              VecSrcTy->getElementType(), ReplicationFactor, VF,
              DemandedDstElts, CostKind);
        }

        return CostKind == TTI::TCK_RecipThroughput ? -1 : 1;
      }

      if (Shuffle->isIdentity())
        return 0;

      if (Shuffle->isReverse())
        return TargetTTI->getShuffleCost(TTI::SK_Reverse, VecTy,
                                         Shuffle->getShuffleMask(), 0, nullptr,
                                         Operands);

      if (Shuffle->isSelect())
        return TargetTTI->getShuffleCost(TTI::SK_Select, VecTy,
                                         Shuffle->getShuffleMask(), 0, nullptr,
                                         Operands);

      if (Shuffle->isTranspose())
        return TargetTTI->getShuffleCost(TTI::SK_Transpose, VecTy,
                                         Shuffle->getShuffleMask(), 0, nullptr,
                                         Operands);

      if (Shuffle->isZeroEltSplat())
        return TargetTTI->getShuffleCost(TTI::SK_Broadcast, VecTy,
                                         Shuffle->getShuffleMask(), 0, nullptr,
                                         Operands);

      if (Shuffle->isSingleSource())
        return TargetTTI->getShuffleCost(TTI::SK_PermuteSingleSrc, VecTy,
                                         Shuffle->getShuffleMask(), 0, nullptr,
                                         Operands);

      if (Shuffle->isInsertSubvectorMask(NumSubElts, SubIndex))
        return TargetTTI->getShuffleCost(
            TTI::SK_InsertSubvector, VecTy, Shuffle->getShuffleMask(), SubIndex,
            FixedVectorType::get(VecTy->getScalarType(), NumSubElts), Operands);

      return TargetTTI->getShuffleCost(TTI::SK_PermuteTwoSrc, VecTy,
                                       Shuffle->getShuffleMask(), 0, nullptr,
                                       Operands);
    }
    case Instruction::ExtractElement: {
      auto *EEI = dyn_cast<ExtractElementInst>(U);
      if (!EEI)
        return TTI::TCC_Basic; // FIXME
      unsigned Idx = -1;
      if (auto *CI = dyn_cast<ConstantInt>(EEI->getOperand(1)))
        if (CI->getValue().getActiveBits() <= 32)
          Idx = CI->getZExtValue();
      Type *DstTy = U->getOperand(0)->getType();
      return TargetTTI->getVectorInstrCost(EEI, DstTy, Idx);
    }
    }
    // By default, just classify everything as 'basic'.
    return TTI::TCC_Basic;
  }

  InstructionCost getInstructionLatency(const Instruction *I) {
    SmallVector<const Value *, 4> Operands(I->operand_values());
    if (getUserCost(I, Operands, TTI::TCK_Latency) == TTI::TCC_Free)
      return 0;

    if (isa<LoadInst>(I))
      return 4;

    Type *DstTy = I->getType();

    // Usually an intrinsic is a simple instruction.
    // A real function call is much slower.
    if (auto *CI = dyn_cast<CallInst>(I)) {
      const Function *F = CI->getCalledFunction();
      if (!F || static_cast<T *>(this)->isLoweredToCall(F))
        return 40;
      // Some intrinsics return a value and a flag, we use the value type
      // to decide its latency.
      if (StructType *StructTy = dyn_cast<StructType>(DstTy))
        DstTy = StructTy->getElementType(0);
      // Fall through to simple instructions.
    }

    if (VectorType *VectorTy = dyn_cast<VectorType>(DstTy))
      DstTy = VectorTy->getElementType();
    if (DstTy->isFloatingPointTy())
      return 3;

    return 1;
  }
};
} // namespace llvm

#endif
