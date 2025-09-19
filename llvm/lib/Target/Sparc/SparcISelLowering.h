//===-- SparcISelLowering.h - Sparc DAG Lowering Interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Sparc uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_SPARCISELLOWERING_H
#define LLVM_LIB_TARGET_SPARC_SPARCISELLOWERING_H

#include "Sparc.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
  class SparcSubtarget;

  class SparcTargetLowering : public TargetLowering {
    const SparcSubtarget *Subtarget;
  public:
    SparcTargetLowering(const TargetMachine &TM, const SparcSubtarget &STI);
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    bool useSoftFloat() const override;

    bool softPromoteHalfType() const override { return true; }

    /// computeKnownBitsForTargetNode - Determine which of the bits specified
    /// in Mask are known to be either zero or one and return them in the
    /// KnownZero/KnownOne bitsets.
    void computeKnownBitsForTargetNode(const SDValue Op,
                                       KnownBits &Known,
                                       const APInt &DemandedElts,
                                       const SelectionDAG &DAG,
                                       unsigned Depth = 0) const override;

    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr &MI,
                                MachineBasicBlock *MBB) const override;

    ConstraintType getConstraintType(StringRef Constraint) const override;
    ConstraintWeight
    getSingleConstraintMatchWeight(AsmOperandInfo &info,
                                   const char *constraint) const override;
    void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const override;

    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 StringRef Constraint, MVT VT) const override;

    bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;
    MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
      return MVT::i32;
    }

    Register getRegisterByName(const char* RegName, LLT VT,
                               const MachineFunction &MF) const override;

    /// If a physical register, this returns the register that receives the
    /// exception address on entry to an EH pad.
    Register
    getExceptionPointerRegister(const Constant *PersonalityFn) const override {
      return SP::I0;
    }

    /// If a physical register, this returns the register that receives the
    /// exception typeid on entry to a landing pad.
    Register
    getExceptionSelectorRegister(const Constant *PersonalityFn) const override {
      return SP::I1;
    }

    /// Override to support customized stack guard loading.
    bool useLoadStackGuardNode(const Module &M) const override;

    /// getSetCCResultType - Return the ISD::SETCC ValueType
    EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                           EVT VT) const override;

    SDValue
    LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerFormalArguments_32(SDValue Chain, CallingConv::ID CallConv,
                                    bool isVarArg,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    const SDLoc &dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerFormalArguments_64(SDValue Chain, CallingConv::ID CallConv,
                                    bool isVarArg,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    const SDLoc &dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const;

    SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerCall_32(TargetLowering::CallLoweringInfo &CLI,
                         SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerCall_64(TargetLowering::CallLoweringInfo &CLI,
                         SmallVectorImpl<SDValue> &InVals) const;

    bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                        bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        LLVMContext &Context, const Type *RetTy) const override;

    SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        const SDLoc &dl, SelectionDAG &DAG) const override;
    SDValue LowerReturn_32(SDValue Chain, CallingConv::ID CallConv,
                           bool IsVarArg,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           const SmallVectorImpl<SDValue> &OutVals,
                           const SDLoc &DL, SelectionDAG &DAG) const;
    SDValue LowerReturn_64(SDValue Chain, CallingConv::ID CallConv,
                           bool IsVarArg,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           const SmallVectorImpl<SDValue> &OutVals,
                           const SDLoc &DL, SelectionDAG &DAG) const;

    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;

    SDValue withTargetFlags(SDValue Op, unsigned TF, SelectionDAG &DAG) const;
    SDValue makeHiLoPair(SDValue Op, unsigned HiTF, unsigned LoTF,
                         SelectionDAG &DAG) const;
    SDValue makeAddress(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerF128_LibCallArg(SDValue Chain, ArgListTy &Args, SDValue Arg,
                                 const SDLoc &DL, SelectionDAG &DAG) const;
    SDValue LowerF128Op(SDValue Op, SelectionDAG &DAG,
                        const char *LibFuncName,
                        unsigned numArgs) const;
    SDValue LowerF128Compare(SDValue LHS, SDValue RHS, unsigned &SPCC,
                             const SDLoc &DL, SelectionDAG &DAG) const;

    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;

    SDValue PerformBITCASTCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    SDValue bitcastConstantFPToInt(ConstantFPSDNode *C, const SDLoc &DL,
                                   SelectionDAG &DAG) const;

    SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

    bool IsEligibleForTailCallOptimization(CCState &CCInfo,
                                           CallLoweringInfo &CLI,
                                           MachineFunction &MF) const;

    bool ShouldShrinkFPConstant(EVT VT) const override {
      // Do not shrink FP constpool if VT == MVT::f128.
      // (ldd, call _Q_fdtoq) is more expensive than two ldds.
      return VT != MVT::f128;
    }

    bool isFNegFree(EVT VT) const override;

    bool isFPImmLegal(const APFloat &Imm, EVT VT,
                      bool ForCodeSize) const override;

    bool isCtlzFast() const override;

    bool isCheapToSpeculateCtlz(Type *Ty) const override {
      return isCtlzFast();
    }

    bool isCheapToSpeculateCttz(Type *Ty) const override;

    bool enableAggressiveFMAFusion(EVT VT) const override { return true; };

    bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                    EVT VT) const override;

    bool shouldInsertFencesForAtomic(const Instruction *I) const override {
      // FIXME: We insert fences for each atomics and generate
      // sub-optimal code for PSO/TSO. (Approximately nobody uses any
      // mode but TSO, which makes this even more silly)
      return true;
    }

    AtomicExpansionKind shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override;

    void ReplaceNodeResults(SDNode *N,
                            SmallVectorImpl<SDValue>& Results,
                            SelectionDAG &DAG) const override;

    MachineBasicBlock *expandSelectCC(MachineInstr &MI, MachineBasicBlock *BB,
                                      unsigned BROpcode) const;

    void AdjustInstrPostInstrSelection(MachineInstr &MI,
                                       SDNode *Node) const override;
  };
} // end namespace llvm

#endif // LLVM_LIB_TARGET_SPARC_SPARCISELLOWERING_H
