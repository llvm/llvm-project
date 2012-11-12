//===-- MSP430ISelLowering.h - MSP430 DAG Lowering Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that MSP430 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MSP430_ISELLOWERING_H
#define LLVM_TARGET_MSP430_ISELLOWERING_H

#include "MSP430.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace MSP430ISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      /// Return with a flag operand. Operand 0 is the chain operand.
      RET_FLAG,

      /// Same as RET_FLAG, but used for returning from ISRs.
      RETI_FLAG,

      /// Y = R{R,L}A X, rotate right (left) arithmetically
      RRA, RLA,

      /// Y = RRC X, rotate right via carry
      RRC,

      /// CALL - These operations represent an abstract call
      /// instruction, which includes a bunch of information.
      CALL,

      /// Wrapper - A wrapper node for TargetConstantPool, TargetExternalSymbol,
      /// and TargetGlobalAddress.
      Wrapper,

      /// CMP - Compare instruction.
      CMP,

      /// SetCC - Operand 0 is condition code, and operand 1 is the flag
      /// operand produced by a CMP instruction.
      SETCC,

      /// MSP430 conditional branches. Operand 0 is the chain operand, operand 1
      /// is the block to branch if condition is true, operand 2 is the
      /// condition code, and operand 3 is the flag operand produced by a CMP
      /// instruction.
      BR_CC,

      /// SELECT_CC - Operand 0 and operand 1 are selection variable, operand 3
      /// is condition code and operand 4 is flag operand.
      SELECT_CC,

      /// SHL, SRA, SRL - Non-constant shifts.
      SHL, SRA, SRL
    };
  }

  class MSP430Subtarget;
  class MSP430TargetMachine;

  class MSP430TargetLowering : public TargetLowering {
  public:
    explicit MSP430TargetLowering(MSP430TargetMachine &TM);

    virtual MVT getShiftAmountTy(EVT LHSTy) const { return MVT::i8; }

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    SDValue LowerShifts(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSIGN_EXTEND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue getReturnAddressFrameIndex(SelectionDAG &DAG) const;

    TargetLowering::ConstraintType
    getConstraintType(const std::string &Constraint) const;
    std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const;

    /// isTruncateFree - Return true if it's free to truncate a value of type
    /// Ty1 to type Ty2. e.g. On msp430 it's free to truncate a i16 value in
    /// register R15W to i8 by referencing its sub-register R15B.
    virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const;
    virtual bool isTruncateFree(EVT VT1, EVT VT2) const;

    /// isZExtFree - Return true if any actual instruction that defines a value
    /// of type Ty1 implicit zero-extends the value to Ty2 in the result
    /// register. This does not necessarily include registers defined in unknown
    /// ways, such as incoming arguments, or copies from unknown virtual
    /// registers. Also, if isTruncateFree(Ty2, Ty1) is true, this does not
    /// necessarily apply to truncate instructions. e.g. on msp430, all
    /// instructions that define 8-bit values implicit zero-extend the result
    /// out to 16 bits.
    virtual bool isZExtFree(Type *Ty1, Type *Ty2) const;
    virtual bool isZExtFree(EVT VT1, EVT VT2) const;

    MachineBasicBlock* EmitInstrWithCustomInserter(MachineInstr *MI,
                                                   MachineBasicBlock *BB) const;
    MachineBasicBlock* EmitShiftInstr(MachineInstr *MI,
                                      MachineBasicBlock *BB) const;

  private:
    SDValue LowerCCCCallTo(SDValue Chain, SDValue Callee,
                           CallingConv::ID CallConv, bool isVarArg,
                           bool isTailCall,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           const SmallVectorImpl<SDValue> &OutVals,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    SDValue LowerCCCArguments(SDValue Chain,
                              CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::InputArg> &Ins,
                              DebugLoc dl,
                              SelectionDAG &DAG,
                              SmallVectorImpl<SDValue> &InVals) const;

    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
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

    virtual bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                            SDValue &Base,
                                            SDValue &Offset,
                                            ISD::MemIndexedMode &AM,
                                            SelectionDAG &DAG) const;

    const MSP430Subtarget &Subtarget;
    const DataLayout *TD;
  };
} // namespace llvm

#endif // LLVM_TARGET_MSP430_ISELLOWERING_H
