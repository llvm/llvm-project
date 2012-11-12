//===-- XCoreISelLowering.h - XCore DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that XCore uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREISELLOWERING_H
#define XCOREISELLOWERING_H

#include "XCore.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

  // Forward delcarations
  class XCoreSubtarget;
  class XCoreTargetMachine;

  namespace XCoreISD {
    enum NodeType {
      // Start the numbering where the builtin ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // Branch and link (call)
      BL,

      // pc relative address
      PCRelativeWrapper,

      // dp relative address
      DPRelativeWrapper,

      // cp relative address
      CPRelativeWrapper,

      // Store word to stack
      STWSP,

      // Corresponds to retsp instruction
      RETSP,

      // Corresponds to LADD instruction
      LADD,

      // Corresponds to LSUB instruction
      LSUB,

      // Corresponds to LMUL instruction
      LMUL,

      // Corresponds to MACCU instruction
      MACCU,

      // Corresponds to MACCS instruction
      MACCS,

      // Jumptable branch.
      BR_JT,

      // Jumptable branch using long branches for each entry.
      BR_JT32
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class XCoreTargetLowering : public TargetLowering
  {
  public:

    explicit XCoreTargetLowering(XCoreTargetMachine &TM);

    virtual unsigned getJumpTableEncoding() const;
    virtual MVT getShiftAmountTy(EVT LHSTy) const { return MVT::i32; }

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// ReplaceNodeResults - Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    virtual void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                                    SelectionDAG &DAG) const;

    /// getTargetNodeName - This method returns the name of a target specific
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    virtual bool isLegalAddressingMode(const AddrMode &AM,
                                       Type *Ty) const;

  private:
    const XCoreTargetMachine &TM;
    const XCoreSubtarget &Subtarget;

    // Lower Operand helpers
    SDValue LowerCCCArguments(SDValue Chain,
                              CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::InputArg> &Ins,
                              DebugLoc dl, SelectionDAG &DAG,
                              SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerCCCCallTo(SDValue Chain, SDValue Callee,
                           CallingConv::ID CallConv, bool isVarArg,
                           bool isTailCall,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           const SmallVectorImpl<SDValue> &OutVals,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;
    SDValue getReturnAddressFrameIndex(SelectionDAG &DAG) const;
    SDValue getGlobalAddressWrapper(SDValue GA, const GlobalValue *GV,
                                    SelectionDAG &DAG) const;

    // Lower Operand specifics
    SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBR_JT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerUMUL_LOHI(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSMUL_LOHI(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINIT_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADJUST_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;

    // Inline asm support
    std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint,
                                 EVT VT) const;

    // Expand specifics
    SDValue TryExpandADDWithMul(SDNode *Op, SelectionDAG &DAG) const;
    SDValue ExpandADDSUB(SDNode *Op, SelectionDAG &DAG) const;

    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                APInt &KnownZero,
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth = 0) const;

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

    virtual bool
      CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                     bool isVarArg,
                     const SmallVectorImpl<ISD::OutputArg> &ArgsFlags,
                     LLVMContext &Context) const;
  };
}

#endif // XCOREISELLOWERING_H
