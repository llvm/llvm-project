//===-- MipsISelLowering.h - Mips DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Mips uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef MipsISELLOWERING_H
#define MipsISELLOWERING_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include "Mips.h"
#include "MipsSubtarget.h"

namespace llvm {
  namespace MipsISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // Jump and link (call)
      JmpLink,

      // Get the Higher 16 bits from a 32-bit immediate
      // No relation with Mips Hi register
      Hi, 

      // Get the Lower 16 bits from a 32-bit immediate
      // No relation with Mips Lo register
      Lo, 

      // Handle gp_rel (small data/bss sections) relocation.
      GPRel,

      // Conditional Move
      CMov,

      // Select CC Pseudo Instruction
      SelectCC,

      // Floating Point Select CC Pseudo Instruction
      FPSelectCC,

      // Floating Point Branch Conditional
      FPBrcond,

      // Floating Point Compare
      FPCmp,

      // Return 
      Ret
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class MipsTargetLowering : public TargetLowering 
  {
    // FrameIndex for return slot.
    int ReturnAddrIndex;
  public:

    explicit MipsTargetLowering(MipsTargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);

    /// getTargetNodeName - This method returns the name of a target specific 
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - get the ISD::SETCC result ValueType
    MVT getSetCCResultType(MVT VT) const;

  private:
    // Subtarget Info
    const MipsSubtarget *Subtarget;

    // Lower Operand helpers
    SDNode *LowerCallResult(SDValue Chain, SDValue InFlag, CallSDNode *TheCall,
                            unsigned CallingConv, SelectionDAG &DAG);
    bool IsGlobalInSmallSection(GlobalValue *GV); 
    bool IsInSmallSection(unsigned Size); 

    // Lower Operand specifics
    SDValue LowerANDOR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCALL(SDValue Op, SelectionDAG &DAG);
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG);
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG);
    SDValue LowerRET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG);

    virtual MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                        MachineBasicBlock *MBB);

    // Inline asm support
    ConstraintType getConstraintType(const std::string &Constraint) const;

    std::pair<unsigned, const TargetRegisterClass*> 
              getRegForInlineAsmConstraint(const std::string &Constraint,
              MVT VT) const;

    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
              MVT VT) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;
  };
}

#endif // MipsISELLOWERING_H
