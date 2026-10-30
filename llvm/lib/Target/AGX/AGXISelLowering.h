//===-- AGXISelLowering.h - AGX DAG lowering interface ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces that AGX uses to lower LLVM code into a selection
// DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXISELLOWERING_H
#define LLVM_LIB_TARGET_AGX_AGXISELLOWERING_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class AGXSubtarget;

class AGXTargetLowering final : public TargetLowering {
  const AGXSubtarget &ST;

  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_FENCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_LOAD_SUB(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSETCC64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShift64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMUL64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerCTLZ_CTTZ64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFDIV32(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFSQRT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_TO_BF16(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFMinMaxNum(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerTrig(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerExpLog(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINT_TO_FP64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_TO_INT64(SDValue Op, SelectionDAG &DAG) const;

  SDValue performBRCONDCombine(SDNode *N, DAGCombinerInfo &DCI) const;

public:
  AGXTargetLowering(const TargetMachine &TM, const AGXSubtarget &ST);

  const AGXSubtarget &getSubtarget() const { return ST; }

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;
  MVT getScalarShiftAmountTy(const DataLayout &DL, EVT VT) const override;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  SDValue lowerVectorStore(SDValue Op, SelectionDAG &DAG) const;
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;
  bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AS, Align Alignment,
                                      MachineMemOperand::Flags Flags,
                                      unsigned *Fast) const override;

  // Ordered atomics are monotonic accesses between fences, so that fences
  // carry the measured barrier sequences (see lowerATOMIC_FENCE).
  bool shouldInsertFencesForAtomic(const Instruction *I) const override {
    return true;
  }
  Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                AtomicOrdering Ord) const override;
  Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                 AtomicOrdering Ord) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;
  void computeKnownBitsForTargetNode(const SDValue Op, KnownBits &Known,
                                     const APInt &DemandedElts,
                                     const SelectionDAG &DAG,
                                     unsigned Depth = 0) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;
  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context, const Type *RetTy) const override {
    return true;
  }
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override;
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override {
    return VT == MVT::f32 || VT == MVT::f16;
  }
  bool isTruncateFree(EVT SrcVT, EVT DstVT) const override;
  bool decomposeMulByConstant(LLVMContext &Context, EVT VT,
                              SDValue C) const override;
  bool isCheapToSpeculateCttz(Type *Ty) const override { return true; }
  bool isCheapToSpeculateCtlz(Type *Ty) const override { return true; }
  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override {
    return false;
  }
  bool mayBeEmittedAsTailCall(const CallInst *CI) const override {
    return false;
  }

  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(const AtomicRMWInst *RMW) const override;
  AtomicExpansionKind
  shouldExpandAtomicLoadInIR(LoadInst *LI) const override;
  AtomicExpansionKind
  shouldExpandAtomicStoreInIR(StoreInst *SI) const override;
  /// The first load of a compare-and-swap loop only seeds the loop, which the
  /// compare-and-swap checks, so it is a plain (single-copy atomic) load, as
  /// on AMDGPU: an atomic one would be a coherent, fenced load at system scope.
  bool shouldIssueAtomicLoadForAtomicEmulationLoop() const override {
    return false;
  }
  /// A system-scope (or volatile) 32-bit load or store of device memory,
  /// which is a coherent access rather than an atomic.
  static bool isSystemScopeDeviceAccess(const Instruction *I);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXISELLOWERING_H
