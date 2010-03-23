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
#include "llvm/Target/TargetOptions.h"
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

      /// CALL - These operations represent an abstract X86 call
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
      CALL,

      /// RDTSC_DAG - This operation implements the lowering for 
      /// readcyclecounter
      RDTSC_DAG,

      /// X86 compare and logical compare instructions.
      CMP, COMI, UCOMI,

      /// X86 bit-test instructions.
      BT,

      /// X86 SetCC. Operand 0 is condition code, and operand 1 is the flag
      /// operand produced by a CMP instruction.
      SETCC,

      // Same as SETCC except it's materialized with a sbb and the value is all
      // one's or all zero's.
      SETCC_CARRY,

      /// X86 conditional moves. Operand 0 and operand 1 are the two values
      /// to select from. Operand 2 is the condition code, and operand 3 is the
      /// flag operand produced by a CMP or TEST instruction. It also writes a
      /// flag result.
      CMOV,

      /// X86 conditional branches. Operand 0 is the chain operand, operand 1
      /// is the block to branch if condition is true, operand 2 is the
      /// condition code, and operand 3 is the flag operand produced by a CMP
      /// or TEST instruction.
      BRCOND,

      /// Return with a flag operand. Operand 0 is the chain operand, operand
      /// 1 is the number of bytes of stack to pop.
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

      /// MOVQ2DQ - Copies a 64-bit value from a vector to another vector.
      /// Can be used to move a vector value from a MMX register to a XMM
      /// register.
      MOVQ2DQ,

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
      PINSRW, MMX_PINSRW,

      /// PSHUFB - Shuffle 16 8-bit values within a vector.
      PSHUFB,

      /// FMAX, FMIN - Floating point max and min.
      ///
      FMAX, FMIN,

      /// FRSQRT, FRCP - Floating point reciprocal-sqrt and reciprocal
      /// approximation.  Note that these typically require refinement
      /// in order to obtain suitable precision.
      FRSQRT, FRCP,

      // TLSADDR - Thread Local Storage.
      TLSADDR,

      // SegmentBaseAddress - The address segment:0
      SegmentBaseAddress,

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

      // FNSTCW16m - Store FP control world into i16 memory.
      FNSTCW16m,

      // VZEXT_MOVL - Vector move low and zero extend.
      VZEXT_MOVL,

      // VZEXT_LOAD - Load, scalar_to_vector, and zero extend.
      VZEXT_LOAD,

      // VSHL, VSRL - Vector logical left / right shift.
      VSHL, VSRL,

      // CMPPD, CMPPS - Vector double/float comparison.
      // CMPPD, CMPPS - Vector double/float comparison.
      CMPPD, CMPPS,
      
      // PCMP* - Vector integer comparisons.
      PCMPEQB, PCMPEQW, PCMPEQD, PCMPEQQ,
      PCMPGTB, PCMPGTW, PCMPGTD, PCMPGTQ,

      // ADD, SUB, SMUL, UMUL, etc. - Arithmetic operations with FLAGS results.
      ADD, SUB, SMUL, UMUL,
      INC, DEC, OR, XOR, AND,

      // MUL_IMM - X86 specific multiply by immediate.
      MUL_IMM,
      
      // PTEST - Vector bitwise comparisons
      PTEST,

      // VASTART_SAVE_XMM_REGS - Save xmm argument registers to the stack,
      // according to %al. An operator is needed so that this can be expanded
      // with control flow.
      VASTART_SAVE_XMM_REGS,

      // MINGW_ALLOCA - MingW's __alloca call to do stack probing.
      MINGW_ALLOCA,

      // ATOMADD64_DAG, ATOMSUB64_DAG, ATOMOR64_DAG, ATOMAND64_DAG, 
      // ATOMXOR64_DAG, ATOMNAND64_DAG, ATOMSWAP64_DAG - 
      // Atomic 64-bit binary operations.
      ATOMADD64_DAG = ISD::FIRST_TARGET_MEMORY_OPCODE,
      ATOMSUB64_DAG,
      ATOMOR64_DAG,
      ATOMXOR64_DAG,
      ATOMAND64_DAG,
      ATOMNAND64_DAG,
      ATOMSWAP64_DAG

      // WARNING: Do not add anything in the end unless you want the node to
      // have memop! In fact, starting from ATOMADD64_DAG all opcodes will be
      // thought as target memory ops!
    };
  }

  /// Define some predicates that are used for node matching.
  namespace X86 {
    /// isPSHUFDMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PSHUFD.
    bool isPSHUFDMask(ShuffleVectorSDNode *N);

    /// isPSHUFHWMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PSHUFD.
    bool isPSHUFHWMask(ShuffleVectorSDNode *N);

    /// isPSHUFLWMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PSHUFD.
    bool isPSHUFLWMask(ShuffleVectorSDNode *N);

    /// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to SHUFP*.
    bool isSHUFPMask(ShuffleVectorSDNode *N);

    /// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVHLPS.
    bool isMOVHLPSMask(ShuffleVectorSDNode *N);

    /// isMOVHLPS_v_undef_Mask - Special case of isMOVHLPSMask for canonical form
    /// of vector_shuffle v, v, <2, 3, 2, 3>, i.e. vector_shuffle v, undef,
    /// <2, 3, 2, 3>
    bool isMOVHLPS_v_undef_Mask(ShuffleVectorSDNode *N);

    /// isMOVLPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for MOVLP{S|D}.
    bool isMOVLPMask(ShuffleVectorSDNode *N);

    /// isMOVHPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for MOVHP{S|D}.
    /// as well as MOVLHPS.
    bool isMOVLHPSMask(ShuffleVectorSDNode *N);

    /// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to UNPCKL.
    bool isUNPCKLMask(ShuffleVectorSDNode *N, bool V2IsSplat = false);

    /// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to UNPCKH.
    bool isUNPCKHMask(ShuffleVectorSDNode *N, bool V2IsSplat = false);

    /// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
    /// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
    /// <0, 0, 1, 1>
    bool isUNPCKL_v_undef_Mask(ShuffleVectorSDNode *N);

    /// isUNPCKH_v_undef_Mask - Special case of isUNPCKHMask for canonical form
    /// of vector_shuffle v, v, <2, 6, 3, 7>, i.e. vector_shuffle v, undef,
    /// <2, 2, 3, 3>
    bool isUNPCKH_v_undef_Mask(ShuffleVectorSDNode *N);

    /// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVSS,
    /// MOVSD, and MOVD, i.e. setting the lowest element.
    bool isMOVLMask(ShuffleVectorSDNode *N);

    /// isMOVSHDUPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVSHDUP.
    bool isMOVSHDUPMask(ShuffleVectorSDNode *N);

    /// isMOVSLDUPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVSLDUP.
    bool isMOVSLDUPMask(ShuffleVectorSDNode *N);

    /// isMOVDDUPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVDDUP.
    bool isMOVDDUPMask(ShuffleVectorSDNode *N);

    /// isPALIGNRMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PALIGNR.
    bool isPALIGNRMask(ShuffleVectorSDNode *N);

    /// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
    /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUF* and SHUFP*
    /// instructions.
    unsigned getShuffleSHUFImmediate(SDNode *N);

    /// getShufflePSHUFHWImmediate - Return the appropriate immediate to shuffle
    /// the specified VECTOR_SHUFFLE mask with PSHUFHW instruction.
    unsigned getShufflePSHUFHWImmediate(SDNode *N);

    /// getShufflePSHUFLWImmediate - Return the appropriate immediate to shuffle
    /// the specified VECTOR_SHUFFLE mask with PSHUFLW instruction.
    unsigned getShufflePSHUFLWImmediate(SDNode *N);

    /// getShufflePALIGNRImmediate - Return the appropriate immediate to shuffle
    /// the specified VECTOR_SHUFFLE mask with the PALIGNR instruction.
    unsigned getShufflePALIGNRImmediate(SDNode *N);

    /// isZeroNode - Returns true if Elt is a constant zero or a floating point
    /// constant +0.0.
    bool isZeroNode(SDValue Elt);

    /// isOffsetSuitableForCodeModel - Returns true of the given offset can be
    /// fit into displacement field of the instruction.
    bool isOffsetSuitableForCodeModel(int64_t Offset, CodeModel::Model M,
                                      bool hasSymbolicDisplacement = true);
  }

  //===--------------------------------------------------------------------===//
  //  X86TargetLowering - X86 Implementation of the TargetLowering interface
  class X86TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int RegSaveFrameIndex;            // X86-64 vararg func register save area.
    unsigned VarArgsGPOffset;         // X86-64 vararg func int reg offset.
    unsigned VarArgsFPOffset;         // X86-64 vararg func fp reg offset.
    int BytesToPopOnReturn;           // Number of arg bytes ret should pop.

  public:
    explicit X86TargetLowering(X86TargetMachine &TM);

    /// getPICBaseSymbol - Return the X86-32 PIC base.
    MCSymbol *getPICBaseSymbol(const MachineFunction *MF, MCContext &Ctx) const;
    
    virtual unsigned getJumpTableEncoding() const;

    virtual const MCExpr *
    LowerCustomJumpTableEntry(const MachineJumpTableInfo *MJTI,
                              const MachineBasicBlock *MBB, unsigned uid,
                              MCContext &Ctx) const;
    
    /// getPICJumpTableRelocaBase - Returns relocation base for the given PIC
    /// jumptable.
    virtual SDValue getPICJumpTableRelocBase(SDValue Table,
                                             SelectionDAG &DAG) const;
    virtual const MCExpr *
    getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                                 unsigned JTI, MCContext &Ctx) const;
    
    // Return the number of bytes that a function should pop when it returns (in
    // addition to the space used by the return address).
    //
    unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }

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
    /// lowering. It returns EVT::iAny if SelectionDAG should be responsible for
    /// determining it.
    virtual EVT getOptimalMemOpType(uint64_t Size, unsigned Align,
                                    bool isSrcConst, bool isSrcStr,
                                    SelectionDAG &DAG) const;

    /// allowsUnalignedMemoryAccesses - Returns true if the target allows
    /// unaligned memory accesses. of the specified type.
    virtual bool allowsUnalignedMemoryAccesses(EVT VT) const {
      return true;
    }

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
                                                         MachineBasicBlock *MBB,
                    DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) const;

 
    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - Return the ISD::SETCC ValueType
    virtual MVT::SimpleValueType getSetCCResultType(EVT VT) const;

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

    virtual bool ExpandInlineAsm(CallInst *CI) const;
    
    ConstraintType getConstraintType(const std::string &Constraint) const;
     
    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        EVT VT) const;

    virtual const char *LowerXConstraint(EVT ConstraintVT) const;

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
                                   EVT VT) const;
    
    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    virtual bool isLegalAddressingMode(const AddrMode &AM, const Type *Ty)const;

    /// isTruncateFree - Return true if it's free to truncate a value of
    /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
    /// register EAX to i16 by referencing its sub-register AX.
    virtual bool isTruncateFree(const Type *Ty1, const Type *Ty2) const;
    virtual bool isTruncateFree(EVT VT1, EVT VT2) const;

    /// isZExtFree - Return true if any actual instruction that defines a
    /// value of type Ty1 implicit zero-extends the value to Ty2 in the result
    /// register. This does not necessarily include registers defined in
    /// unknown ways, such as incoming arguments, or copies from unknown
    /// virtual registers. Also, if isTruncateFree(Ty2, Ty1) is true, this
    /// does not necessarily apply to truncate instructions. e.g. on x86-64,
    /// all instructions that define 32-bit values implicit zero-extend the
    /// result out to 64 bits.
    virtual bool isZExtFree(const Type *Ty1, const Type *Ty2) const;
    virtual bool isZExtFree(EVT VT1, EVT VT2) const;

    /// isNarrowingProfitable - Return true if it's profitable to narrow
    /// operations of type VT1 to VT2. e.g. on x86, it's profitable to narrow
    /// from i32 to i8 but not from i32 to i16.
    virtual bool isNarrowingProfitable(EVT VT1, EVT VT2) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    /// isShuffleMaskLegal - Targets can use this to indicate that they only
    /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
    /// By default, if a target supports the VECTOR_SHUFFLE node, all mask
    /// values are assumed to be legal.
    virtual bool isShuffleMaskLegal(const SmallVectorImpl<int> &Mask,
                                    EVT VT) const;

    /// isVectorClearMaskLegal - Similar to isShuffleMaskLegal. This is
    /// used by Targets can use this to indicate if there is a suitable
    /// VECTOR_SHUFFLE that can be used to replace a VAND with a constant
    /// pool entry.
    virtual bool isVectorClearMaskLegal(const SmallVectorImpl<int> &Mask,
                                        EVT VT) const;

    /// ShouldShrinkFPConstant - If true, then instruction selection should
    /// seek to shrink the FP constant of the specified type to a smaller type
    /// in order to save space and / or reduce runtime.
    virtual bool ShouldShrinkFPConstant(EVT VT) const {
      // Don't shrink FP constpool if SSE2 is available since cvtss2sd is more
      // expensive than a straight movsd. On the other hand, it's important to
      // shrink long double fp constant since fldt is very slow.
      return !X86ScalarSSEf64 || VT == MVT::f80;
    }
    
    virtual const X86Subtarget* getSubtarget() {
      return Subtarget;
    }

    /// isScalarFPTypeInSSEReg - Return true if the specified scalar FP type is
    /// computed in an SSE register, not on the X87 floating point stack.
    bool isScalarFPTypeInSSEReg(EVT VT) const {
      return (VT == MVT::f64 && X86ScalarSSEf64) || // f64 is when SSE2
      (VT == MVT::f32 && X86ScalarSSEf32);   // f32 is when SSE1
    }

    /// getWidenVectorType: given a vector type, returns the type to widen
    /// to (e.g., v7i8 to v8i8). If the vector type is legal, it returns itself.
    /// If there is no vector type that we want to widen to, returns EVT::Other
    /// When and were to widen is target dependent based on the cost of
    /// scalarizing vs using the wider vector type.
    virtual EVT getWidenVectorType(EVT VT) const;

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

    /// getFunctionAlignment - Return the Log2 alignment of this function.
    virtual unsigned getFunctionAlignment(const Function *F) const;

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

    /// LegalFPImmediates - A list of legal fp immediates.
    std::vector<APFloat> LegalFPImmediates;

    /// addLegalFPImmediate - Indicate that this x86 target can instruction
    /// select the specified FP immediate natively.
    void addLegalFPImmediate(const APFloat& Imm) {
      LegalFPImmediates.push_back(Imm);
    }

    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals);
    SDValue LowerMemArgument(SDValue Chain,
                             CallingConv::ID CallConv,
                             const SmallVectorImpl<ISD::InputArg> &ArgInfo,
                             DebugLoc dl, SelectionDAG &DAG,
                             const CCValAssign &VA,  MachineFrameInfo *MFI,
                              unsigned i);
    SDValue LowerMemOpCallTo(SDValue Chain, SDValue StackPtr, SDValue Arg,
                             DebugLoc dl, SelectionDAG &DAG,
                             const CCValAssign &VA,
                             ISD::ArgFlagsTy Flags);

    // Call lowering helpers.

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Targets which want to do tail call
    /// optimization should implement this function.
    bool IsEligibleForTailCallOptimization(SDValue Callee,
                                           CallingConv::ID CalleeCC,
                                           bool isVarArg,
                                           bool isCalleeStructRet,
                                           bool isCallerStructRet,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                           SelectionDAG& DAG) const;
    bool IsCalleePop(bool isVarArg, CallingConv::ID CallConv);
    SDValue EmitTailCallLoadRetAddr(SelectionDAG &DAG, SDValue &OutRetAddr,
                                SDValue Chain, bool IsTailCall, bool Is64Bit,
                                int FPDiff, DebugLoc dl);

    CCAssignFn *CCAssignFnForNode(CallingConv::ID CallConv) const;
    unsigned GetAlignedArgumentStackSize(unsigned StackSize, SelectionDAG &DAG);

    std::pair<SDValue,SDValue> FP_TO_INTHelper(SDValue Op, SelectionDAG &DAG,
                                               bool isSigned);

    SDValue LowerAsSplatVectorLoad(SDValue SrcOp, EVT VT, DebugLoc dl,
                                   SelectionDAG &DAG);
    SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG);
    SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerEXTRACT_VECTOR_ELT_SSE4(SDValue Op, SelectionDAG &DAG);
    SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerINSERT_VECTOR_ELT_SSE4(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalAddress(const GlobalValue *GV, DebugLoc dl,
                               int64_t Offset, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerExternalSymbol(SDValue Op, SelectionDAG &DAG);
    SDValue LowerShift(SDValue Op, SelectionDAG &DAG);
    SDValue BuildFILD(SDValue Op, EVT SrcVT, SDValue Chain, SDValue StackSlot,
                      SelectionDAG &DAG);
    SDValue LowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG);
    SDValue LowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG);
    SDValue LowerUINT_TO_FP_i64(SDValue Op, SelectionDAG &DAG);
    SDValue LowerUINT_TO_FP_i32(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFP_TO_UINT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFABS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFNEG(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerVSETCC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG);
    SDValue LowerMEMSET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG);
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG);
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

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals);
    virtual SDValue
      LowerCall(SDValue Chain, SDValue Callee,
                CallingConv::ID CallConv, bool isVarArg, bool &isTailCall,
                const SmallVectorImpl<ISD::OutputArg> &Outs,
                const SmallVectorImpl<ISD::InputArg> &Ins,
                DebugLoc dl, SelectionDAG &DAG,
                SmallVectorImpl<SDValue> &InVals);

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  DebugLoc dl, SelectionDAG &DAG);

    virtual bool
      CanLowerReturn(CallingConv::ID CallConv, bool isVarArg,
                     const SmallVectorImpl<EVT> &OutTys,
                     const SmallVectorImpl<ISD::ArgFlagsTy> &ArgsFlags,
                     SelectionDAG &DAG);

    void ReplaceATOMIC_BINARY_64(SDNode *N, SmallVectorImpl<SDValue> &Results,
                                 SelectionDAG &DAG, unsigned NewOp);

    SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                                    SDValue Chain,
                                    SDValue Dst, SDValue Src,
                                    SDValue Size, unsigned Align,
                                    const Value *DstSV, uint64_t DstSVOff);
    SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                    SDValue Chain,
                                    SDValue Dst, SDValue Src,
                                    SDValue Size, unsigned Align,
                                    bool AlwaysInline,
                                    const Value *DstSV, uint64_t DstSVOff,
                                    const Value *SrcSV, uint64_t SrcSVOff);
    
    /// Utility function to emit string processing sse4.2 instructions
    /// that return in xmm0.
    /// This takes the instruction to expand, the associated machine basic
    /// block, the number of args, and whether or not the second arg is
    /// in memory or not.
    MachineBasicBlock *EmitPCMP(MachineInstr *BInstr, MachineBasicBlock *BB,
				unsigned argNum, bool inMem) const;

    /// Utility function to emit atomic bitwise operations (and, or, xor).
    /// It takes the bitwise instruction to expand, the associated machine basic
    /// block, and the associated X86 opcodes for reg/reg and reg/imm.
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
                                                    bool invSrc = false) const;

    MachineBasicBlock *EmitAtomicBit6432WithCustomInserter(
                                                    MachineInstr *BInstr,
                                                    MachineBasicBlock *BB,
                                                    unsigned regOpcL,
                                                    unsigned regOpcH,
                                                    unsigned immOpcL,
                                                    unsigned immOpcH,
                                                    bool invSrc = false) const;
    
    /// Utility function to emit atomic min and max.  It takes the min/max
    /// instruction to expand, the associated basic block, and the associated
    /// cmov opcode for moving the min or max value.
    MachineBasicBlock *EmitAtomicMinMaxWithCustomInserter(MachineInstr *BInstr,
                                                          MachineBasicBlock *BB,
                                                        unsigned cmovOpc) const;

    /// Utility function to emit the xmm reg save portion of va_start.
    MachineBasicBlock *EmitVAStartSaveXMMRegsWithCustomInserter(
                                                   MachineInstr *BInstr,
                                                   MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredSelect(MachineInstr *I,
                                         MachineBasicBlock *BB,
                    DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) const;

    MachineBasicBlock *EmitLoweredMingwAlloca(MachineInstr *MI,
                                              MachineBasicBlock *BB,
                    DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) const;

    /// Emit nodes that will be selected as "test Op0,Op0", or something
    /// equivalent, for use with the given x86 condition code.
    SDValue EmitTest(SDValue Op0, unsigned X86CC, SelectionDAG &DAG);

    /// Emit nodes that will be selected as "cmp Op0,Op1", or something
    /// equivalent, for use with the given x86 condition code.
    SDValue EmitCmp(SDValue Op0, SDValue Op1, unsigned X86CC,
                    SelectionDAG &DAG);
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
