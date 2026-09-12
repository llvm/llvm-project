//===-- NVPTXTargetTransformInfo.h - NVPTX specific TTI ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// NVPTX target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXTARGETTRANSFORMINFO_H

#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXUtilities.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <optional>

namespace llvm {

class NVPTXTTIImpl final : public BasicTTIImplBase<NVPTXTTIImpl> {
  typedef BasicTTIImplBase<NVPTXTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const NVPTXSubtarget *ST;
  const NVPTXTargetLowering *TLI;

  const NVPTXSubtarget *getST() const { return ST; };
  const NVPTXTargetLowering *getTLI() const { return TLI; };

  /// \returns true if the result of the value could potentially be
  /// different across threads in a warp.
  bool isSourceOfDivergence(const Value *V) const;

public:
  using BaseT::getVectorInstrCost;

  explicit NVPTXTTIImpl(const NVPTXTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl()),
        TLI(ST->getTargetLowering()) {}

  bool hasBranchDivergence(const Function *F = nullptr) const override {
    return true;
  }

  unsigned getFlatAddressSpace() const override {
    return AddressSpace::ADDRESS_SPACE_GENERIC;
  }

  bool
  canHaveNonUndefGlobalInitializerInAddressSpace(unsigned AS) const override {
    return AS != AddressSpace::ADDRESS_SPACE_SHARED &&
           AS != AddressSpace::ADDRESS_SPACE_LOCAL &&
           AS != AddressSpace::ADDRESS_SPACE_ENTRY_PARAM;
  }

  std::optional<Instruction *>
  instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const override;

  // Loads and stores can be vectorized if the alignment is at least as big as
  // the load/store we want to vectorize.
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const override {
    return Alignment >= ChainSizeInBytes;
  }
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const override {
    return isLegalToVectorizeLoadChain(ChainSizeInBytes, Alignment, AddrSpace);
  }

  // NVPTX has infinite registers of all kinds, but the actual machine doesn't.
  // We conservatively return 1 here which is just enough to enable the
  // vectorizers but disables heuristics based on the number of registers.
  // FIXME: Return a more reasonable number, while keeping an eye on
  // LoopVectorizer's unrolling heuristics.
  unsigned getNumberOfRegisters(unsigned ClassID) const override { return 1; }

  // The types  <2 x half> and  <2 x float> can be vectorized, so return 64 for
  // the vector register size.
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override {
    return TypeSize::getFixed(64);
  }
  unsigned getMinVectorRegisterBitWidth() const override { return 32; }

  bool shouldExpandReduction(const IntrinsicInst *II) const override {
    // Turn off ExpandReductions pass for NVPTX, which doesn't have advanced
    // swizzling operations. Our backend/Selection DAG can expand these
    // reductions with less movs.
    return false;
  }

  // We don't want to prevent inlining because of target-cpu and -features
  // attributes that were added to newer versions of LLVM/Clang: There are
  // no incompatible functions in PTX, ptxas will throw errors in such cases.
  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const override {
    return true;
  }

  // Increase the inlining cost threshold by a factor of 11, reflecting that
  // calls are particularly expensive in NVPTX.
  unsigned getInliningThresholdMultiplier() const override { return 11; }

  InstructionCost
  getInstructionCost(const User *U, ArrayRef<const Value *> Operands,
                     TTI::TargetCostKind CostKind) const override;

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      ArrayRef<const Value *> Args = {},
      const Instruction *CxtI = nullptr) const override;

  InstructionCost
  getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                   TTI::CastContextHint CCH, TTI::TargetCostKind CostKind,
                   const Instruction *I = nullptr) const override;

  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
                 ArrayRef<int> Mask = {},
                 TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
                 int Index = 0, VectorType *SubTp = nullptr,
                 ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override;

  InstructionCost
  getVectorInstrCost(unsigned Opcode, Type *Val, TTI::TargetCostKind CostKind,
                     unsigned Index = -1, const Value *Op0 = nullptr,
                     const Value *Op1 = nullptr,
                     TTI::VectorInstrContext VIC =
                         TTI::VectorInstrContext::None) const override;

  InstructionCost
  getScalarizationOverhead(VectorType *InTy, const APInt &DemandedElts,
                           bool Insert, bool Extract,
                           TTI::TargetCostKind CostKind,
                           bool ForPoisonSrc = true, ArrayRef<Value *> VL = {},
                           TTI::VectorInstrContext VIC =
                               TTI::VectorInstrContext::None) const override;

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;

  bool hasVolatileVariant(Instruction *I, unsigned AddrSpace) const override {
    // Volatile loads/stores are only supported for shared and global address
    // spaces, or for generic AS that maps to them.
    if (!(AddrSpace == llvm::ADDRESS_SPACE_GENERIC ||
          AddrSpace == llvm::ADDRESS_SPACE_GLOBAL ||
          AddrSpace == llvm::ADDRESS_SPACE_SHARED))
      return false;

    switch(I->getOpcode()){
    default:
      return false;
    case Instruction::Load:
    case Instruction::Store:
      return true;
    }
  }

  APInt getAddrSpaceCastPreservedPtrMask(unsigned SrcAS,
                                         unsigned DstAS) const override {
    if (SrcAS != llvm::ADDRESS_SPACE_GENERIC)
      return BaseT::getAddrSpaceCastPreservedPtrMask(SrcAS, DstAS);
    if (DstAS != llvm::ADDRESS_SPACE_GLOBAL &&
        DstAS != llvm::ADDRESS_SPACE_SHARED)
      return BaseT::getAddrSpaceCastPreservedPtrMask(SrcAS, DstAS);

    // Address change within 4K size does not change the original address space
    // and is safe to perform address cast form SrcAS to DstAS.
    APInt PtrMask(DL.getPointerSizeInBits(llvm::ADDRESS_SPACE_GENERIC), 0xfff);
    return PtrMask;
  }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const override;

  bool isLegalMaskedStore(Type *DataType, Align Alignment, unsigned AddrSpace,
                          TTI::MaskKind MaskKind) const override;

  bool isLegalMaskedLoad(Type *DataType, Align Alignment, unsigned AddrSpace,
                         TTI::MaskKind MaskKind) const override;

  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const override;

  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const override;
  unsigned getAssumedAddrSpace(const Value *V) const override;

  void collectKernelLaunchBounds(
      const Function &F,
      SmallVectorImpl<std::pair<StringRef, int64_t>> &LB) const override;

  bool shouldBuildRelLookupTables() const override {
    // Self-referential globals are not supported.
    return false;
  }

  InstructionCost getPartialReductionCost(
      unsigned Opcode, Type *InputTypeA, Type *InputTypeB, Type *AccumType,
      ElementCount VF, TTI::PartialReductionExtendKind OpAExtend,
      TTI::PartialReductionExtendKind OpBExtend, std::optional<unsigned> BinOp,
      TTI::TargetCostKind CostKind,
      std::optional<FastMathFlags> FMF) const override {
    return InstructionCost::getInvalid();
  }

  ValueUniformity getValueUniformity(const Value *V) const override;
};

} // end namespace llvm

#endif
