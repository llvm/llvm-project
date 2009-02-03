//===-- PIC16ISelLowering.h - PIC16 DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PIC16 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16ISELLOWERING_H
#define PIC16ISELLOWERING_H

#include "PIC16.h"
#include "PIC16Subtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace PIC16ISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      Lo,            // Low 8-bits of GlobalAddress.
      Hi,            // High 8-bits of GlobalAddress.
      PIC16Load,
      PIC16LdWF,
      PIC16Store,
      PIC16StWF,
      Banksel,
      MTLO,
      MTHI,
      BCF,
      LSLF,          // PIC16 Logical shift left
      LRLF,          // PIC16 Logical shift right
      RLF,           // Rotate left through carry
      RRF,           // Rotate right through carry
      CALL,          // PIC16 Call instruction 
      SUBCC,	     // Compare for equality or inequality.
      SELECT_ICC,    // Psuedo to be caught in schedular and expanded to brcond.
      BRCOND,        // Conditional branch.
      Dummy
    };

    // Keep track of different address spaces. 
    enum AddressSpace {
      RAM_SPACE = 0,   // RAM address space
      ROM_SPACE = 1    // ROM address space number is 1
    };
    enum PIC16Libcall {
      MUL_I8,
      SRA_I8,
      SLL_I8,
      SRL_I8,
      PIC16UnknownCall
    };
  }


  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class PIC16TargetLowering : public TargetLowering {
  public:
    explicit PIC16TargetLowering(PIC16TargetMachine &TM);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;
    /// getSetCCResultType - Return the ISD::SETCC ValueType
    virtual MVT getSetCCResultType(MVT ValType) const;
    SDValue LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerShift(SDValue Op, SelectionDAG &DAG);
    SDValue LowerADD(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSUB(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBinOp(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCALL(SDValue Op, SelectionDAG &DAG);
    SDValue LowerRET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCallReturn(SDValue Op, SDValue Chain, SDValue FrameAddress,
                            SDValue InFlag, SelectionDAG &DAG);
    SDValue LowerCallArguments(SDValue Op, SDValue Chain, SDValue FrameAddress,
                               SDValue InFlag, SelectionDAG &DAG);
    SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG);
    SDValue getPIC16Cmp(SDValue LHS, SDValue RHS, unsigned OrigCC, SDValue &CC,
                        SelectionDAG &DAG);
    virtual MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                        MachineBasicBlock *MBB);


    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);
    virtual void ReplaceNodeResults(SDNode *N,
                                    SmallVectorImpl<SDValue> &Results,
                                    SelectionDAG &DAG);
    virtual void LowerOperationWrapper(SDNode *N,
                                       SmallVectorImpl<SDValue> &Results,
                                       SelectionDAG &DAG);

    SDValue ExpandStore(SDNode *N, SelectionDAG &DAG);
    SDValue ExpandLoad(SDNode *N, SelectionDAG &DAG);
    //SDValue ExpandAdd(SDNode *N, SelectionDAG &DAG);
    SDValue ExpandGlobalAddress(SDNode *N, SelectionDAG &DAG);
    SDValue ExpandExternalSymbol(SDNode *N, SelectionDAG &DAG);
    SDValue ExpandFrameIndex(SDNode *N, SelectionDAG &DAG);

    SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const; 
    SDValue PerformPIC16LoadCombine(SDNode *N, DAGCombinerInfo &DCI) const; 

  private:
    // If the Node is a BUILD_PAIR representing representing an Address
    // then this function will return true
    bool isDirectAddress(const SDValue &Op);

    // If the Node is a DirectAddress in ROM_SPACE then this 
    // function will return true
    bool isRomAddress(const SDValue &Op);

    // To extract chain value from the SDValue Nodes
    // This function will help to maintain the chain extracting
    // code at one place. In case of any change in future it will
    // help maintain the code
    SDValue getChain(SDValue &Op);
    
    SDValue getOutFlag(SDValue &Op);


    // Extract the Lo and Hi component of Op. 
    void GetExpandedParts(SDValue Op, SelectionDAG &DAG, SDValue &Lo, 
                          SDValue &Hi); 


    // Load pointer can be a direct or indirect address. In PIC16 direct
    // addresses need Banksel and Indirect addresses need to be loaded to
    // FSR first. Handle address specific cases here.
    void LegalizeAddress(SDValue Ptr, SelectionDAG &DAG, SDValue &Chain, 
                         SDValue &NewPtr, unsigned &Offset);

    // FrameIndex should be broken down into ExternalSymbol and FrameOffset. 
    void LegalizeFrameIndex(SDValue Op, SelectionDAG &DAG, SDValue &ES, 
                            int &Offset);

    // We can not have both operands of a binary operation in W.
    // This function is used to put one operand on stack and generate a load.
    SDValue ConvertToMemOperand(SDValue Op, SelectionDAG &DAG); 

    // This function checks if we need to put an operand of an operation on
    // stack and generate a load or not.
    bool NeedToConvertToMemOp(SDValue Op, unsigned &MemOp); 

    /// Subtarget - Keep a pointer to the PIC16Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const PIC16Subtarget *Subtarget;


    // Extending the LIB Call framework of LLVM
    // To hold the names of PIC16Libcalls
    const char *PIC16LibcallNames[PIC16ISD::PIC16UnknownCall]; 

    // To set and retrieve the lib call names
    void setPIC16LibcallName(PIC16ISD::PIC16Libcall Call, const char *Name);
    const char *getPIC16LibcallName(PIC16ISD::PIC16Libcall Call);

    // Make PIC16 Libcall
    SDValue MakePIC16Libcall(PIC16ISD::PIC16Libcall Call, MVT RetVT, 
                             const SDValue *Ops, unsigned NumOps, bool isSigned,
                             SelectionDAG &DAG, DebugLoc dl);

    // Check if operation has a direct load operand.
    inline bool isDirectLoad(const SDValue Op);

    // Create the symbol and index for function frame
    void getCurrentFrameIndex(SelectionDAG &DAG, SDValue &ES, 
                              unsigned SlotSize, int &FI);

    SDValue getCurrentFrame(SelectionDAG &DAG);
  };
} // namespace llvm

#endif // PIC16ISELLOWERING_H
