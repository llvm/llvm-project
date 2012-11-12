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

#ifndef SPARC_ISELLOWERING_H
#define SPARC_ISELLOWERING_H

#include "Sparc.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace SPISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,
      CMPICC,      // Compare two GPR operands, set icc.
      CMPFCC,      // Compare two FP operands, set fcc.
      BRICC,       // Branch to dest on icc condition
      BRFCC,       // Branch to dest on fcc condition
      SELECT_ICC,  // Select between two values using the current ICC flags.
      SELECT_FCC,  // Select between two values using the current FCC flags.

      Hi, Lo,      // Hi/Lo operations, typically on a global address.

      FTOI,        // FP to Int within a FP register.
      ITOF,        // Int to FP within a FP register.

      CALL,        // A call instruction.
      RET_FLAG,    // Return with a flag operand.
      GLOBAL_BASE_REG, // Global base reg for PIC
      FLUSHW       // FLUSH register windows to stack
    };
  }

  class SparcTargetLowering : public TargetLowering {
  public:
    SparcTargetLowering(TargetMachine &TM);
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// computeMaskedBitsForTargetNode - Determine which of the bits specified
    /// in Mask are known to be either zero or one and return them in the
    /// KnownZero/KnownOne bitsets.
    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                APInt &KnownZero,
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth = 0) const;

    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    virtual const char *getTargetNodeName(unsigned Opcode) const;

    ConstraintType getConstraintType(const std::string &Constraint) const;
    std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv,
                           bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;

    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;

    unsigned getSRetArgSize(SelectionDAG &DAG, SDValue Callee) const;
  };
} // end namespace llvm

#endif    // SPARC_ISELLOWERING_H
