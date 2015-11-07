//===-- SparcISelLowering.h - Sparc DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  class SparcSubtarget;

  namespace SPISD {
    enum NodeType : unsigned {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,
      CMPICC,      // Compare two GPR operands, set icc+xcc.
      CMPFCC,      // Compare two FP operands, set fcc.
      BRICC,       // Branch to dest on icc condition
      BRXCC,       // Branch to dest on xcc condition (64-bit only).
      BRFCC,       // Branch to dest on fcc condition
      SELECT_ICC,  // Select between two values using the current ICC flags.
      SELECT_XCC,  // Select between two values using the current XCC flags.
      SELECT_FCC,  // Select between two values using the current FCC flags.

      Hi, Lo,      // Hi/Lo operations, typically on a global address.

      FTOI,        // FP to Int within a FP register.
      ITOF,        // Int to FP within a FP register.
      FTOX,        // FP to Int64 within a FP register.
      XTOF,        // Int64 to FP within a FP register.

      CALL,        // A call instruction.
      RET_FLAG,    // Return with a flag operand.
      GLOBAL_BASE_REG, // Global base reg for PIC.
      FLUSHW,      // FLUSH register windows to stack.

      TLS_ADD,     // For Thread Local Storage (TLS).
      TLS_LD,
      TLS_CALL
    };
  }

  class SparcTargetLowering : public TargetLowering {
    const SparcSubtarget *Subtarget;
  public:
    SparcTargetLowering(TargetMachine &TM, const SparcSubtarget &STI);
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    /// computeKnownBitsForTargetNode - Determine which of the bits specified
    /// in Mask are known to be either zero or one and return them in the
    /// KnownZero/KnownOne bitsets.
    void computeKnownBitsForTargetNode(const SDValue Op,
                                       APInt &KnownZero,
                                       APInt &KnownOne,
                                       const SelectionDAG &DAG,
                                       unsigned Depth = 0) const override;

    MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const override;

    const char *getTargetNodeName(unsigned Opcode) const override;

    ConstraintType getConstraintType(StringRef Constraint) const override;
    ConstraintWeight
    getSingleConstraintMatchWeight(AsmOperandInfo &info,
                                   const char *constraint) const override;
    void LowerAsmOperandForConstraint(SDValue Op,
                                      std::string &Constraint,
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const override;
    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 StringRef Constraint, MVT VT) const override;

    bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;
    MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
      return MVT::i32;
    }

    /// If a physical register, this returns the register that receives the
    /// exception address on entry to an EH pad.
    unsigned
    getExceptionPointerRegister(const Constant *PersonalityFn) const override {
      return SP::I0;
    }

    /// If a physical register, this returns the register that receives the
    /// exception typeid on entry to a landing pad.
    unsigned
    getExceptionSelectorRegister(const Constant *PersonalityFn) const override {
      return SP::I1;
    }

    /// getSetCCResultType - Return the ISD::SETCC ValueType
    EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                           EVT VT) const override;

    SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv,
                           bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           SDLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerFormalArguments_32(SDValue Chain,
                                    CallingConv::ID CallConv,
                                    bool isVarArg,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    SDLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerFormalArguments_64(SDValue Chain,
                                    CallingConv::ID CallConv,
                                    bool isVarArg,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    SDLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const;

    SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerCall_32(TargetLowering::CallLoweringInfo &CLI,
                         SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerCall_64(TargetLowering::CallLoweringInfo &CLI,
                         SmallVectorImpl<SDValue> &InVals) const;

    SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  SDLoc dl, SelectionDAG &DAG) const override;
    SDValue LowerReturn_32(SDValue Chain,
                           CallingConv::ID CallConv, bool IsVarArg,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           const SmallVectorImpl<SDValue> &OutVals,
                           SDLoc DL, SelectionDAG &DAG) const;
    SDValue LowerReturn_64(SDValue Chain,
                           CallingConv::ID CallConv, bool IsVarArg,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           const SmallVectorImpl<SDValue> &OutVals,
                           SDLoc DL, SelectionDAG &DAG) const;

    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;

    unsigned getSRetArgSize(SelectionDAG &DAG, SDValue Callee) const;
    SDValue withTargetFlags(SDValue Op, unsigned TF, SelectionDAG &DAG) const;
    SDValue makeHiLoPair(SDValue Op, unsigned HiTF, unsigned LoTF,
                         SelectionDAG &DAG) const;
    SDValue makeAddress(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerF128_LibCallArg(SDValue Chain, ArgListTy &Args,
                                 SDValue Arg, SDLoc DL,
                                 SelectionDAG &DAG) const;
    SDValue LowerF128Op(SDValue Op, SelectionDAG &DAG,
                        const char *LibFuncName,
                        unsigned numArgs) const;
    SDValue LowerF128Compare(SDValue LHS, SDValue RHS,
                             unsigned &SPCC,
                             SDLoc DL,
                             SelectionDAG &DAG) const;

    bool ShouldShrinkFPConstant(EVT VT) const override {
      // Do not shrink FP constpool if VT == MVT::f128.
      // (ldd, call _Q_fdtoq) is more expensive than two ldds.
      return VT != MVT::f128;
    }

    void ReplaceNodeResults(SDNode *N,
                            SmallVectorImpl<SDValue>& Results,
                            SelectionDAG &DAG) const override;

    MachineBasicBlock *expandSelectCC(MachineInstr *MI, MachineBasicBlock *BB,
                                      unsigned BROpcode) const;
    MachineBasicBlock *expandAtomicRMW(MachineInstr *MI,
                                       MachineBasicBlock *BB,
                                       unsigned Opcode,
                                       unsigned CondCode = 0) const;
  };
} // end namespace llvm

#endif    // SPARC_ISELLOWERING_H
