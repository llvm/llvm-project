//===-- X86ISelLowering.h - X86 DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that X86 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef X86ISELLOWERING_H
#define X86ISELLOWERING_H

#include "X86Subtarget.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
  namespace X86ISD {
    // X86 Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+X86::INSTRUCTION_LIST_END,

      /// SHLD, SHRD - Double shift instructions. These correspond to
      /// X86::SHLDxx and X86::SHRDxx instructions.
      SHLD,
      SHRD,

      /// FAND - Bitwise logical AND of floating point values. This corresponds
      /// to X86::ANDPS or X86::ANDPD.
      FAND,

      /// FXOR - Bitwise logical XOR of floating point values. This corresponds
      /// to X86::XORPS or X86::XORPD.
      FXOR,

      /// FILD, FILD_FLAG - This instruction implements SINT_TO_FP with the
      /// integer source in memory and FP reg result.  This corresponds to the
      /// X86::FILD*m instructions. It has three inputs (token chain, address,
      /// and source type) and two outputs (FP value and token chain). FILD_FLAG
      /// also produces a flag).
      FILD,
      FILD_FLAG,

      /// FP_TO_INT*_IN_MEM - This instruction implements FP_TO_SINT with the
      /// integer destination in memory and a FP reg source.  This corresponds
      /// to the X86::FIST*m instructions and the rounding mode change stuff. It
      /// has two inputs (token chain and address) and two outputs (int value
      /// and token chain).
      FP_TO_INT16_IN_MEM,
      FP_TO_INT32_IN_MEM,
      FP_TO_INT64_IN_MEM,

      /// FLD - This instruction implements an extending load to FP stack slots.
      /// This corresponds to the X86::FLD32m / X86::FLD64m. It takes a chain
      /// operand, ptr to load from, and a ValueType node indicating the type
      /// to load to.
      FLD,

      /// FST - This instruction implements a truncating store to FP stack
      /// slots. This corresponds to the X86::FST32m / X86::FST64m. It takes a
      /// chain operand, value to store, address, and a ValueType to store it
      /// as.
      FST,

      /// FP_SET_RESULT - This corresponds to FpGETRESULT pseudo instrcuction
      /// which copies from ST(0) to the destination. It takes a chain and writes
      /// a RFP result and a chain.
      FP_GET_RESULT,

      /// FP_SET_RESULT - This corresponds to FpSETRESULT pseudo instrcuction
      /// which copies the source operand to ST(0). It takes a chain and writes
      /// a chain and a flag.
      FP_SET_RESULT,

      /// CALL/TAILCALL - These operations represent an abstract X86 call
      /// instruction, which includes a bunch of information.  In particular the
      /// operands of these node are:
      ///
      ///     #0 - The incoming token chain
      ///     #1 - The callee
      ///     #2 - The number of arg bytes the caller pushes on the stack.
      ///     #3 - The number of arg bytes the callee pops off the stack.
      ///     #4 - The value to pass in AL/AX/EAX (optional)
      ///     #5 - The value to pass in DL/DX/EDX (optional)
      ///
      /// The result values of these nodes are:
      ///
      ///     #0 - The outgoing token chain
      ///     #1 - The first register result value (optional)
      ///     #2 - The second register result value (optional)
      ///
      /// The CALL vs TAILCALL distinction boils down to whether the callee is
      /// known not to modify the caller's stack frame, as is standard with
      /// LLVM.
      CALL,
      TAILCALL,
      
      /// RDTSC_DAG - This operation implements the lowering for 
      /// readcyclecounter
      RDTSC_DAG,

      /// X86 compare and logical compare instructions.
      CMP, TEST, COMI, UCOMI,

      /// X86 SetCC. Operand 1 is condition code, and operand 2 is the flag
      /// operand produced by a CMP instruction.
      SETCC,

      /// X86 conditional moves. Operand 1 and operand 2 are the two values
      /// to select from (operand 1 is a R/W operand). Operand 3 is the
      /// condition code, and operand 4 is the flag operand produced by a CMP
      /// or TEST instruction. It also writes a flag result.
      CMOV,

      /// X86 conditional branches. Operand 1 is the chain operand, operand 2
      /// is the block to branch if condition is true, operand 3 is the
      /// condition code, and operand 4 is the flag operand produced by a CMP
      /// or TEST instruction.
      BRCOND,

      /// Return with a flag operand. Operand 1 is the chain operand, operand
      /// 2 is the number of bytes of stack to pop.
      RET_FLAG,

      /// REP_STOS - Repeat fill, corresponds to X86::REP_STOSx.
      REP_STOS,

      /// REP_MOVS - Repeat move, corresponds to X86::REP_MOVSx.
      REP_MOVS,

      /// LOAD_PACK Load a 128-bit packed float / double value. It has the same
      /// operands as a normal load.
      LOAD_PACK,

      /// LOAD_UA Load an unaligned 128-bit value. It has the same operands as
      /// a normal load.
      LOAD_UA,

      /// GlobalBaseReg - On Darwin, this node represents the result of the popl
      /// at function entry, used for PIC code.
      GlobalBaseReg,

      /// Wrapper - A wrapper node for TargetConstantPool,
      /// TargetExternalSymbol, and TargetGlobalAddress.
      Wrapper,

      /// S2VEC - X86 version of SCALAR_TO_VECTOR. The destination base does not
      /// have to match the operand type.
      S2VEC,

      /// PEXTRW - Extract a 16-bit value from a vector and zero extend it to
      /// i32, corresponds to X86::PEXTRW.
      PEXTRW,

      /// PINSRW - Insert the lower 16-bits of a 32-bit value to a vector,
      /// corresponds to X86::PINSRW.
      PINSRW,

      /// FMAX, FMIN - Floating point max and min.
      ///
      FMAX, FMIN
    };
  }

 /// Define some predicates that are used for node matching.
 namespace X86 {
   /// isPSHUFDMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to PSHUFD.
   bool isPSHUFDMask(SDNode *N);

   /// isPSHUFHWMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to PSHUFD.
   bool isPSHUFHWMask(SDNode *N);

   /// isPSHUFLWMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to PSHUFD.
   bool isPSHUFLWMask(SDNode *N);

   /// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to SHUFP*.
   bool isSHUFPMask(SDNode *N);

   /// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVHLPS.
   bool isMOVHLPSMask(SDNode *N);

   /// isMOVLPMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVLP{S|D}.
   bool isMOVLPMask(SDNode *N);

   /// isMOVHPMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVHP{S|D}
   /// as well as MOVLHPS.
   bool isMOVHPMask(SDNode *N);

   /// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to UNPCKL.
   bool isUNPCKLMask(SDNode *N, bool V2IsSplat = false);

   /// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to UNPCKH.
   bool isUNPCKHMask(SDNode *N, bool V2IsSplat = false);

   /// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
   /// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
   /// <0, 0, 1, 1>
   bool isUNPCKL_v_undef_Mask(SDNode *N);

   /// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVSS,
   /// MOVSD, and MOVD, i.e. setting the lowest element.
   bool isMOVLMask(SDNode *N);

   /// isMOVSHDUPMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVSHDUP.
   bool isMOVSHDUPMask(SDNode *N);

   /// isMOVSLDUPMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVSLDUP.
   bool isMOVSLDUPMask(SDNode *N);

   /// isSplatMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a splat of a single element.
   bool isSplatMask(SDNode *N);

   /// isSplatLoMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a splat of zero element.
   bool isSplatLoMask(SDNode *N);

   /// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
   /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUF* and SHUFP*
   /// instructions.
   unsigned getShuffleSHUFImmediate(SDNode *N);

   /// getShufflePSHUFHWImmediate - Return the appropriate immediate to shuffle
   /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUFHW
   /// instructions.
   unsigned getShufflePSHUFHWImmediate(SDNode *N);

   /// getShufflePSHUFKWImmediate - Return the appropriate immediate to shuffle
   /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUFLW
   /// instructions.
   unsigned getShufflePSHUFLWImmediate(SDNode *N);
 }

  //===--------------------------------------------------------------------===//
  //  X86TargetLowering - X86 Implementation of the TargetLowering interface
  class X86TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int RegSaveFrameIndex;            // X86-64 vararg func register save area.
    unsigned VarArgsGPOffset;         // X86-64 vararg func int reg offset.
    unsigned VarArgsFPOffset;         // X86-64 vararg func fp reg offset.
    int ReturnAddrIndex;              // FrameIndex for return slot.
    int BytesToPopOnReturn;           // Number of arg bytes ret should pop.
    int BytesCallerReserves;          // Number of arg bytes caller makes.
  public:
    X86TargetLowering(TargetMachine &TM);

    // Return the number of bytes that a function should pop when it returns (in
    // addition to the space used by the return address).
    //
    unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }

    // Return the number of bytes that the caller reserves for arguments passed
    // to this function.
    unsigned getBytesCallerReserves() const { return BytesCallerReserves; }
 
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    virtual SDOperand PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// computeMaskedBitsForTargetNode - Determine which of the bits specified 
    /// in Mask are known to be either zero or one and return them in the 
    /// KnownZero/KnownOne bitsets.
    virtual void computeMaskedBitsForTargetNode(const SDOperand Op,
                                                uint64_t Mask,
                                                uint64_t &KnownZero, 
                                                uint64_t &KnownOne,
                                                unsigned Depth = 0) const;
    
    SDOperand getReturnAddressFrameIndex(SelectionDAG &DAG);

    ConstraintType getConstraintType(char ConstraintLetter) const;
     
    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        MVT::ValueType VT) const;
    /// isOperandValidForConstraint - Return the specified operand (possibly
    /// modified) if the specified SDOperand is valid for the specified target
    /// constraint letter, otherwise return null.
    SDOperand isOperandValidForConstraint(SDOperand Op, char ConstraintLetter,
                                          SelectionDAG &DAG);
    
    /// getRegForInlineAsmConstraint - Given a physical register constraint
    /// (e.g. {edx}), return the register number and the register class for the
    /// register.  This should only be used for C_Register constraints.  On
    /// error, this returns a register number of 0.
    std::pair<unsigned, const TargetRegisterClass*> 
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   MVT::ValueType VT) const;
    
    /// isLegalAddressImmediate - Return true if the integer value or
    /// GlobalValue can be used as the offset of the target addressing mode.
    virtual bool isLegalAddressImmediate(int64_t V) const;
    virtual bool isLegalAddressImmediate(GlobalValue *GV) const;

    /// isShuffleMaskLegal - Targets can use this to indicate that they only
    /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
    /// By default, if a target supports the VECTOR_SHUFFLE node, all mask
    /// values are assumed to be legal.
    virtual bool isShuffleMaskLegal(SDOperand Mask, MVT::ValueType VT) const;

    /// isVectorClearMaskLegal - Similar to isShuffleMaskLegal. This is
    /// used by Targets can use this to indicate if there is a suitable
    /// VECTOR_SHUFFLE that can be used to replace a VAND with a constant
    /// pool entry.
    virtual bool isVectorClearMaskLegal(std::vector<SDOperand> &BVOps,
                                        MVT::ValueType EVT,
                                        SelectionDAG &DAG) const;
  private:
    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    /// X86StackPtr - X86 physical register used as stack ptr.
    unsigned X86StackPtr;

    /// X86ScalarSSE - Select between SSE2 or x87 floating point ops.
    bool X86ScalarSSE;

    // C Calling Convention implementation.
    SDOperand LowerCCCArguments(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCCCCallTo(SDOperand Op, SelectionDAG &DAG);

    // X86-64 C Calling Convention implementation.
    SDOperand LowerX86_64CCCArguments(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerX86_64CCCCallTo(SDOperand Op, SelectionDAG &DAG);

    // Fast Calling Convention implementation.
    SDOperand LowerFastCCArguments(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFastCCCallTo(SDOperand Op, SelectionDAG &DAG,
                                bool isFastCall);

    // StdCall Calling Convention implementation.
    SDOperand LowerStdCallCCArguments(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerStdCallCCCallTo(SDOperand Op, SelectionDAG &DAG);

    // FastCall Calling Convention implementation.
    SDOperand LowerFastCallCCArguments(SDOperand Op, SelectionDAG &DAG);

    SDOperand LowerBUILD_VECTOR(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerVECTOR_SHUFFLE(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerEXTRACT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerINSERT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSCALAR_TO_VECTOR(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerExternalSymbol(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerShift(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFABS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFNEG(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSETCC(SDOperand Op, SelectionDAG &DAG, SDOperand Chain);
    SDOperand LowerSELECT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerBRCOND(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerMEMSET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerMEMCPY(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerJumpTable(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerREADCYCLCECOUNTER(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerINTRINSIC_WO_CHAIN(SDOperand Op, SelectionDAG &DAG);
  };
}

// FASTCC_NUM_INT_ARGS_INREGS - This is the max number of integer arguments
// to pass in registers.  0 is none, 1 is is "use EAX", 2 is "use EAX and
// EDX".  Anything more is illegal.
//
// FIXME: The linscan register allocator currently has problem with
// coalescing.  At the time of this writing, whenever it decides to coalesce
// a physreg with a virtreg, this increases the size of the physreg's live
// range, and the live range cannot ever be reduced.  This causes problems if
// too many physregs are coaleced with virtregs, which can cause the register
// allocator to wedge itself.
//
// This code triggers this problem more often if we pass args in registers,
// so disable it until this is fixed.
//
#define FASTCC_NUM_INT_ARGS_INREGS 0

#endif    // X86ISELLOWERING_H
