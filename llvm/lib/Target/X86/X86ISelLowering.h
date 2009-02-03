//===-- X86ISelLowering.h - X86 DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "X86RegisterInfo.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {
  namespace X86ISD {
    // X86 Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      /// BSF - Bit scan forward.
      /// BSR - Bit scan reverse.
      BSF,
      BSR,

      /// SHLD, SHRD - Double shift instructions. These correspond to
      /// X86::SHLDxx and X86::SHRDxx instructions.
      SHLD,
      SHRD,

      /// FAND - Bitwise logical AND of floating point values. This corresponds
      /// to X86::ANDPS or X86::ANDPD.
      FAND,

      /// FOR - Bitwise logical OR of floating point values. This corresponds
      /// to X86::ORPS or X86::ORPD.
      FOR,

      /// FXOR - Bitwise logical XOR of floating point values. This corresponds
      /// to X86::XORPS or X86::XORPD.
      FXOR,

      /// FSRL - Bitwise logical right shift of floating point values. These
      /// corresponds to X86::PSRLDQ.
      FSRL,

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
      CMP, COMI, UCOMI,

      /// X86 bit-test instructions.
      BT,

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

      /// GlobalBaseReg - On Darwin, this node represents the result of the popl
      /// at function entry, used for PIC code.
      GlobalBaseReg,

      /// Wrapper - A wrapper node for TargetConstantPool,
      /// TargetExternalSymbol, and TargetGlobalAddress.
      Wrapper,

      /// WrapperRIP - Special wrapper used under X86-64 PIC mode for RIP
      /// relative displacements.
      WrapperRIP,

      /// PEXTRB - Extract an 8-bit value from a vector and zero extend it to
      /// i32, corresponds to X86::PEXTRB.
      PEXTRB,

      /// PEXTRW - Extract a 16-bit value from a vector and zero extend it to
      /// i32, corresponds to X86::PEXTRW.
      PEXTRW,

      /// INSERTPS - Insert any element of a 4 x float vector into any element
      /// of a destination 4 x floatvector.
      INSERTPS,

      /// PINSRB - Insert the lower 8-bits of a 32-bit value to a vector,
      /// corresponds to X86::PINSRB.
      PINSRB,

      /// PINSRW - Insert the lower 16-bits of a 32-bit value to a vector,
      /// corresponds to X86::PINSRW.
      PINSRW,

      /// FMAX, FMIN - Floating point max and min.
      ///
      FMAX, FMIN,

      /// FRSQRT, FRCP - Floating point reciprocal-sqrt and reciprocal
      /// approximation.  Note that these typically require refinement
      /// in order to obtain suitable precision.
      FRSQRT, FRCP,

      // TLSADDR, THREAD_POINTER - Thread Local Storage.
      TLSADDR, THREAD_POINTER,

      // EH_RETURN - Exception Handling helpers.
      EH_RETURN,
      
      /// TC_RETURN - Tail call return.
      ///   operand #0 chain
      ///   operand #1 callee (register or absolute)
      ///   operand #2 stack adjustment
      ///   operand #3 optional in flag
      TC_RETURN,

      // LCMPXCHG_DAG, LCMPXCHG8_DAG - Compare and swap.
      LCMPXCHG_DAG,
      LCMPXCHG8_DAG,

      // ATOMADD64_DAG, ATOMSUB64_DAG, ATOMOR64_DAG, ATOMAND64_DAG, 
      // ATOMXOR64_DAG, ATOMNAND64_DAG, ATOMSWAP64_DAG - 
      // Atomic 64-bit binary operations.
      ATOMADD64_DAG,
      ATOMSUB64_DAG,
      ATOMOR64_DAG,
      ATOMXOR64_DAG,
      ATOMAND64_DAG,
      ATOMNAND64_DAG,
      ATOMSWAP64_DAG,

      // FNSTCW16m - Store FP control world into i16 memory.
      FNSTCW16m,

      // VZEXT_MOVL - Vector move low and zero extend.
      VZEXT_MOVL,

      // VZEXT_LOAD - Load, scalar_to_vector, and zero extend.
      VZEXT_LOAD,

      // VSHL, VSRL - Vector logical left / right shift.
      VSHL, VSRL,
      
      // CMPPD, CMPPS - Vector double/float comparison.
      CMPPD, CMPPS,
      
      // PCMP* - Vector integer comparisons.
      PCMPEQB, PCMPEQW, PCMPEQD, PCMPEQQ,
      PCMPGTB, PCMPGTW, PCMPGTD, PCMPGTQ,

      // ADD, SUB, SMUL, UMUL - Arithmetic operations with overflow/carry
      // intrinsics.
      ADD, SUB, SMUL, UMUL
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

    /// isMOVHLPS_v_undef_Mask - Special case of isMOVHLPSMask for canonical form
    /// of vector_shuffle v, v, <2, 3, 2, 3>, i.e. vector_shuffle v, undef,
    /// <2, 3, 2, 3>
    bool isMOVHLPS_v_undef_Mask(SDNode *N);

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

    /// isUNPCKH_v_undef_Mask - Special case of isUNPCKHMask for canonical form
    /// of vector_shuffle v, v, <2, 6, 3, 7>, i.e. vector_shuffle v, undef,
    /// <2, 2, 3, 3>
    bool isUNPCKH_v_undef_Mask(SDNode *N);

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

    /// isMOVDDUPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVDDUP.
    bool isMOVDDUPMask(SDNode *N);

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
    int BytesToPopOnReturn;           // Number of arg bytes ret should pop.
    int BytesCallerReserves;          // Number of arg bytes caller makes.

  public:
    explicit X86TargetLowering(X86TargetMachine &TM);

    /// getPICJumpTableRelocaBase - Returns relocation base for the given PIC
    /// jumptable.
    SDValue getPICJumpTableRelocBase(SDValue Table,
                                       SelectionDAG &DAG) const;

    // Return the number of bytes that a function should pop when it returns (in
    // addition to the space used by the return address).
    //
    unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }

    // Return the number of bytes that the caller reserves for arguments passed
    // to this function.
    unsigned getBytesCallerReserves() const { return BytesCallerReserves; }
 
    /// getStackPtrReg - Return the stack pointer register we are using: either
    /// ESP or RSP.
    unsigned getStackPtrReg() const { return X86StackPtr; }

    /// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
    /// function arguments in the caller parameter area. For X86, aggregates
    /// that contains are placed at 16-byte boundaries while the rest are at
    /// 4-byte boundaries.
    virtual unsigned getByValTypeAlignment(const Type *Ty) const;

    /// getOptimalMemOpType - Returns the target specific optimal type for load
    /// and store operations as a result of memset, memcpy, and memmove
    /// lowering. It returns MVT::iAny if SelectionDAG should be responsible for
    /// determining it.
    virtual
    MVT getOptimalMemOpType(uint64_t Size, unsigned Align,
                            bool isSrcConst, bool isSrcStr) const;
    
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);

    /// ReplaceNodeResults - Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    virtual void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                                    SelectionDAG &DAG);

    
    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                        MachineBasicBlock *MBB);

 
    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - Return the ISD::SETCC ValueType
    virtual MVT getSetCCResultType(MVT VT) const;

    /// computeMaskedBitsForTargetNode - Determine which of the bits specified 
    /// in Mask are known to be either zero or one and return them in the 
    /// KnownZero/KnownOne bitsets.
    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                const APInt &Mask,
                                                APInt &KnownZero, 
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth = 0) const;

    virtual bool
    isGAPlusOffset(SDNode *N, GlobalValue* &GA, int64_t &Offset) const;
    
    SDValue getReturnAddressFrameIndex(SelectionDAG &DAG);

    ConstraintType getConstraintType(const std::string &Constraint) const;
     
    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        MVT VT) const;

    virtual const char *LowerXConstraint(MVT ConstraintVT) const;

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops. If hasMemory is
    /// true it means one of the asm constraint of the inline asm instruction
    /// being processed is 'm'.
    virtual void LowerAsmOperandForConstraint(SDValue Op,
                                              char ConstraintLetter,
                                              bool hasMemory,
                                              std::vector<SDValue> &Ops,
                                              SelectionDAG &DAG) const;
    
    /// getRegForInlineAsmConstraint - Given a physical register constraint
    /// (e.g. {edx}), return the register number and the register class for the
    /// register.  This should only be used for C_Register constraints.  On
    /// error, this returns a register number of 0.
    std::pair<unsigned, const TargetRegisterClass*> 
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   MVT VT) const;
    
    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    virtual bool isLegalAddressingMode(const AddrMode &AM, const Type *Ty)const;

    /// isTruncateFree - Return true if it's free to truncate a value of
    /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
    /// register EAX to i16 by referencing its sub-register AX.
    virtual bool isTruncateFree(const Type *Ty1, const Type *Ty2) const;
    virtual bool isTruncateFree(MVT VT1, MVT VT2) const;
  
    /// isShuffleMaskLegal - Targets can use this to indicate that they only
    /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
    /// By default, if a target supports the VECTOR_SHUFFLE node, all mask
    /// values are assumed to be legal.
    virtual bool isShuffleMaskLegal(SDValue Mask, MVT VT) const;

    /// isVectorClearMaskLegal - Similar to isShuffleMaskLegal. This is
    /// used by Targets can use this to indicate if there is a suitable
    /// VECTOR_SHUFFLE that can be used to replace a VAND with a constant
    /// pool entry.
    virtual bool isVectorClearMaskLegal(const std::vector<SDValue> &BVOps,
                                        MVT EVT, SelectionDAG &DAG) const;

    /// ShouldShrinkFPConstant - If true, then instruction selection should
    /// seek to shrink the FP constant of the specified type to a smaller type
    /// in order to save space and / or reduce runtime.
    virtual bool ShouldShrinkFPConstant(MVT VT) const {
      // Don't shrink FP constpool if SSE2 is available since cvtss2sd is more
      // expensive than a straight movsd. On the other hand, it's important to
      // shrink long double fp constant since fldt is very slow.
      return !X86ScalarSSEf64 || VT == MVT::f80;
    }
    
    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Target which want to do tail call
    /// optimization should implement this function.
    virtual bool IsEligibleForTailCallOptimization(CallSDNode *TheCall, 
                                                   SDValue Ret, 
                                                   SelectionDAG &DAG) const;

    virtual const X86Subtarget* getSubtarget() {
      return Subtarget;
    }

    /// isScalarFPTypeInSSEReg - Return true if the specified scalar FP type is
    /// computed in an SSE register, not on the X87 floating point stack.
    bool isScalarFPTypeInSSEReg(MVT VT) const {
      return (VT == MVT::f64 && X86ScalarSSEf64) || // f64 is when SSE2
      (VT == MVT::f32 && X86ScalarSSEf32);   // f32 is when SSE1
    }

    /// getWidenVectorType: given a vector type, returns the type to widen
    /// to (e.g., v7i8 to v8i8). If the vector type is legal, it returns itself.
    /// If there is no vector type that we want to widen to, returns MVT::Other
    /// When and were to widen is target dependent based on the cost of
    /// scalarizing vs using the wider vector type.
    virtual MVT getWidenVectorType(MVT VT) const;

    /// createFastISel - This method returns a target specific FastISel object,
    /// or null if the target does not support "fast" ISel.
    virtual FastISel *
    createFastISel(MachineFunction &mf,
                   MachineModuleInfo *mmi, DwarfWriter *dw,
                   DenseMap<const Value *, unsigned> &,
                   DenseMap<const BasicBlock *, MachineBasicBlock *> &,
                   DenseMap<const AllocaInst *, int> &
#ifndef NDEBUG
                   , SmallSet<Instruction*, 8> &
#endif
                   );
    
  private:
    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;
    const X86RegisterInfo *RegInfo;
    const TargetData *TD;

    /// X86StackPtr - X86 physical register used as stack ptr.
    unsigned X86StackPtr;
   
    /// X86ScalarSSEf32, X86ScalarSSEf64 - Select between SSE or x87 
    /// floating point ops.
    /// When SSE is available, use it for f32 operations.
    /// When SSE2 is available, use it for f64 operations.
    bool X86ScalarSSEf32;
    bool X86ScalarSSEf64;

    SDNode *LowerCallResult(SDValue Chain, SDValue InFlag, CallSDNode *TheCall,
                            unsigned CallingConv, SelectionDAG &DAG);

    SDValue LowerMemArgument(SDValue Op, SelectionDAG &DAG,
                               const CCValAssign &VA,  MachineFrameInfo *MFI,
                               unsigned CC, SDValue Root, unsigned i);

    SDValue LowerMemOpCallTo(CallSDNode *TheCall, SelectionDAG &DAG,
                               const SDValue &StackPtr,
                               const CCValAssign &VA, SDValue Chain,
                               SDValue Arg, ISD::ArgFlagsTy Flags);

    // Call lowering helpers.
    bool IsCalleePop(bool isVarArg, unsigned CallingConv);
    bool CallRequiresGOTPtrInReg(bool Is64Bit, bool IsTailCall);
    bool CallRequiresFnAddressInReg(bool Is64Bit, bool IsTailCall);
    SDValue EmitTailCallLoadRetAddr(SelectionDAG &DAG, SDValue &OutRetAddr,
                                SDValue Chain, bool IsTailCall, bool Is64Bit,
                                int FPDiff);

    CCAssignFn *CCAssignFnForNode(unsigned CallingConv) const;
    NameDecorationStyle NameDecorationForFORMAL_ARGUMENTS(SDValue Op);
    unsigned GetAlignedArgumentStackSize(unsigned StackSize, SelectionDAG &DAG);

    std::pair<SDValue,SDValue> FP_TO_SINTHelper(SDValue Op, 
                                                    SelectionDAG &DAG);
    
    SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG);
    SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerEXTRACT_VECTOR_ELT_SSE4(SDValue Op, SelectionDAG &DAG);
    SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerINSERT_VECTOR_ELT_SSE4(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalAddress(const GlobalValue *GV, int64_t Offset,
                               SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerExternalSymbol(SDValue Op, SelectionDAG &DAG);
    SDValue LowerShift(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG);
    SDValue LowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG);
    SDValue LowerUINT_TO_FP_i64(SDValue Op, SelectionDAG &DAG);
    SDValue LowerUINT_TO_FP_i32(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFABS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFNEG(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVSETCC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG);
    SDValue LowerMEMSET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCALL(SDValue Op, SelectionDAG &DAG);
    SDValue LowerRET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG);
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG);
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFRAME_TO_ARGS_OFFSET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerEH_RETURN(SDValue Op, SelectionDAG &DAG);
    SDValue LowerTRAMPOLINE(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFLT_ROUNDS_(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCTLZ(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCTTZ(SDValue Op, SelectionDAG &DAG);
    SDValue LowerMUL_V2I64(SDValue Op, SelectionDAG &DAG);
    SDValue LowerXALUO(SDValue Op, SelectionDAG &DAG);

    SDValue LowerCMP_SWAP(SDValue Op, SelectionDAG &DAG);
    SDValue LowerLOAD_SUB(SDValue Op, SelectionDAG &DAG);
    SDValue LowerREADCYCLECOUNTER(SDValue Op, SelectionDAG &DAG);

    void ReplaceATOMIC_BINARY_64(SDNode *N, SmallVectorImpl<SDValue> &Results,
                                 SelectionDAG &DAG, unsigned NewOp);

    SDValue EmitTargetCodeForMemset(SelectionDAG &DAG,
                                    SDValue Chain,
                                    SDValue Dst, SDValue Src,
                                    SDValue Size, unsigned Align,
                                    const Value *DstSV, uint64_t DstSVOff);
    SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG,
                                    SDValue Chain,
                                    SDValue Dst, SDValue Src,
                                    SDValue Size, unsigned Align,
                                    bool AlwaysInline,
                                    const Value *DstSV, uint64_t DstSVOff,
                                    const Value *SrcSV, uint64_t SrcSVOff);
    
    /// Utility function to emit atomic bitwise operations (and, or, xor).
    // It takes the bitwise instruction to expand, the associated machine basic
    // block, and the associated X86 opcodes for reg/reg and reg/imm.
    MachineBasicBlock *EmitAtomicBitwiseWithCustomInserter(
                                                    MachineInstr *BInstr,
                                                    MachineBasicBlock *BB,
                                                    unsigned regOpc,
                                                    unsigned immOpc,
                                                    unsigned loadOpc,
                                                    unsigned cxchgOpc,
                                                    unsigned copyOpc,
                                                    unsigned notOpc,
                                                    unsigned EAXreg,
                                                    TargetRegisterClass *RC,
                                                    bool invSrc = false);

    MachineBasicBlock *EmitAtomicBit6432WithCustomInserter(
                                                    MachineInstr *BInstr,
                                                    MachineBasicBlock *BB,
                                                    unsigned regOpcL,
                                                    unsigned regOpcH,
                                                    unsigned immOpcL,
                                                    unsigned immOpcH,
                                                    bool invSrc = false);
    
    /// Utility function to emit atomic min and max.  It takes the min/max
    // instruction to expand, the associated basic block, and the associated
    // cmov opcode for moving the min or max value.
    MachineBasicBlock *EmitAtomicMinMaxWithCustomInserter(MachineInstr *BInstr,
                                                          MachineBasicBlock *BB,
                                                          unsigned cmovOpc);
  };

  namespace X86 {
    FastISel *createFastISel(MachineFunction &mf,
                           MachineModuleInfo *mmi, DwarfWriter *dw,
                           DenseMap<const Value *, unsigned> &,
                           DenseMap<const BasicBlock *, MachineBasicBlock *> &,
                           DenseMap<const AllocaInst *, int> &
#ifndef NDEBUG
                           , SmallSet<Instruction*, 8> &
#endif
                           );
  }
}

#endif    // X86ISELLOWERING_H
