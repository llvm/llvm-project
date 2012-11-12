//===-- ARMISelLowering.h - ARM DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that ARM uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef ARMISELLOWERING_H
#define ARMISELLOWERING_H

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include <vector>

namespace llvm {
  class ARMConstantPoolValue;

  namespace ARMISD {
    // ARM Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builtin ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      Wrapper,      // Wrapper - A wrapper node for TargetConstantPool,
                    // TargetExternalSymbol, and TargetGlobalAddress.
      WrapperDYN,   // WrapperDYN - A wrapper node for TargetGlobalAddress in
                    // DYN mode.
      WrapperPIC,   // WrapperPIC - A wrapper node for TargetGlobalAddress in
                    // PIC mode.
      WrapperJT,    // WrapperJT - A wrapper node for TargetJumpTable

      // Add pseudo op to model memcpy for struct byval.
      COPY_STRUCT_BYVAL,

      CALL,         // Function call.
      CALL_PRED,    // Function call that's predicable.
      CALL_NOLINK,  // Function call with branch not branch-and-link.
      tCALL,        // Thumb function call.
      BRCOND,       // Conditional branch.
      BR_JT,        // Jumptable branch.
      BR2_JT,       // Jumptable branch (2 level - jumptable entry is a jump).
      RET_FLAG,     // Return with a flag operand.

      PIC_ADD,      // Add with a PC operand and a PIC label.

      CMP,          // ARM compare instructions.
      CMN,          // ARM CMN instructions.
      CMPZ,         // ARM compare that sets only Z flag.
      CMPFP,        // ARM VFP compare instruction, sets FPSCR.
      CMPFPw0,      // ARM VFP compare against zero instruction, sets FPSCR.
      FMSTAT,       // ARM fmstat instruction.

      CMOV,         // ARM conditional move instructions.

      BCC_i64,

      RBIT,         // ARM bitreverse instruction

      FTOSI,        // FP to sint within a FP register.
      FTOUI,        // FP to uint within a FP register.
      SITOF,        // sint to FP within a FP register.
      UITOF,        // uint to FP within a FP register.

      SRL_FLAG,     // V,Flag = srl_flag X -> srl X, 1 + save carry out.
      SRA_FLAG,     // V,Flag = sra_flag X -> sra X, 1 + save carry out.
      RRX,          // V = RRX X, Flag     -> srl X, 1 + shift in carry flag.

      ADDC,         // Add with carry
      ADDE,         // Add using carry
      SUBC,         // Sub with carry
      SUBE,         // Sub using carry

      VMOVRRD,      // double to two gprs.
      VMOVDRR,      // Two gprs to double.

      EH_SJLJ_SETJMP,         // SjLj exception handling setjmp.
      EH_SJLJ_LONGJMP,        // SjLj exception handling longjmp.

      TC_RETURN,    // Tail call return pseudo.

      THREAD_POINTER,

      DYN_ALLOC,    // Dynamic allocation on the stack.

      MEMBARRIER,   // Memory barrier (DMB)
      MEMBARRIER_MCR, // Memory barrier (MCR)

      PRELOAD,      // Preload

      VCEQ,         // Vector compare equal.
      VCEQZ,        // Vector compare equal to zero.
      VCGE,         // Vector compare greater than or equal.
      VCGEZ,        // Vector compare greater than or equal to zero.
      VCLEZ,        // Vector compare less than or equal to zero.
      VCGEU,        // Vector compare unsigned greater than or equal.
      VCGT,         // Vector compare greater than.
      VCGTZ,        // Vector compare greater than zero.
      VCLTZ,        // Vector compare less than zero.
      VCGTU,        // Vector compare unsigned greater than.
      VTST,         // Vector test bits.

      // Vector shift by immediate:
      VSHL,         // ...left
      VSHRs,        // ...right (signed)
      VSHRu,        // ...right (unsigned)
      VSHLLs,       // ...left long (signed)
      VSHLLu,       // ...left long (unsigned)
      VSHLLi,       // ...left long (with maximum shift count)
      VSHRN,        // ...right narrow

      // Vector rounding shift by immediate:
      VRSHRs,       // ...right (signed)
      VRSHRu,       // ...right (unsigned)
      VRSHRN,       // ...right narrow

      // Vector saturating shift by immediate:
      VQSHLs,       // ...left (signed)
      VQSHLu,       // ...left (unsigned)
      VQSHLsu,      // ...left (signed to unsigned)
      VQSHRNs,      // ...right narrow (signed)
      VQSHRNu,      // ...right narrow (unsigned)
      VQSHRNsu,     // ...right narrow (signed to unsigned)

      // Vector saturating rounding shift by immediate:
      VQRSHRNs,     // ...right narrow (signed)
      VQRSHRNu,     // ...right narrow (unsigned)
      VQRSHRNsu,    // ...right narrow (signed to unsigned)

      // Vector shift and insert:
      VSLI,         // ...left
      VSRI,         // ...right

      // Vector get lane (VMOV scalar to ARM core register)
      // (These are used for 8- and 16-bit element types only.)
      VGETLANEu,    // zero-extend vector extract element
      VGETLANEs,    // sign-extend vector extract element

      // Vector move immediate and move negated immediate:
      VMOVIMM,
      VMVNIMM,

      // Vector move f32 immediate:
      VMOVFPIMM,

      // Vector duplicate:
      VDUP,
      VDUPLANE,

      // Vector shuffles:
      VEXT,         // extract
      VREV64,       // reverse elements within 64-bit doublewords
      VREV32,       // reverse elements within 32-bit words
      VREV16,       // reverse elements within 16-bit halfwords
      VZIP,         // zip (interleave)
      VUZP,         // unzip (deinterleave)
      VTRN,         // transpose
      VTBL1,        // 1-register shuffle with mask
      VTBL2,        // 2-register shuffle with mask

      // Vector multiply long:
      VMULLs,       // ...signed
      VMULLu,       // ...unsigned

      UMLAL,        // 64bit Unsigned Accumulate Multiply
      SMLAL,        // 64bit Signed Accumulate Multiply

      // Operands of the standard BUILD_VECTOR node are not legalized, which
      // is fine if BUILD_VECTORs are always lowered to shuffles or other
      // operations, but for ARM some BUILD_VECTORs are legal as-is and their
      // operands need to be legalized.  Define an ARM-specific version of
      // BUILD_VECTOR for this purpose.
      BUILD_VECTOR,

      // Floating-point max and min:
      FMAX,
      FMIN,

      // Bit-field insert
      BFI,

      // Vector OR with immediate
      VORRIMM,
      // Vector AND with NOT of immediate
      VBICIMM,

      // Vector bitwise select
      VBSL,

      // Vector load N-element structure to all lanes:
      VLD2DUP = ISD::FIRST_TARGET_MEMORY_OPCODE,
      VLD3DUP,
      VLD4DUP,

      // NEON loads with post-increment base updates:
      VLD1_UPD,
      VLD2_UPD,
      VLD3_UPD,
      VLD4_UPD,
      VLD2LN_UPD,
      VLD3LN_UPD,
      VLD4LN_UPD,
      VLD2DUP_UPD,
      VLD3DUP_UPD,
      VLD4DUP_UPD,

      // NEON stores with post-increment base updates:
      VST1_UPD,
      VST2_UPD,
      VST3_UPD,
      VST4_UPD,
      VST2LN_UPD,
      VST3LN_UPD,
      VST4LN_UPD,

      // 64-bit atomic ops (value split into two registers)
      ATOMADD64_DAG,
      ATOMSUB64_DAG,
      ATOMOR64_DAG,
      ATOMXOR64_DAG,
      ATOMAND64_DAG,
      ATOMNAND64_DAG,
      ATOMSWAP64_DAG,
      ATOMCMPXCHG64_DAG
    };
  }

  /// Define some predicates that are used for node matching.
  namespace ARM {
    bool isBitFieldInvertedMask(unsigned v);
  }

  //===--------------------------------------------------------------------===//
  //  ARMTargetLowering - ARM Implementation of the TargetLowering interface

  class ARMTargetLowering : public TargetLowering {
  public:
    explicit ARMTargetLowering(TargetMachine &TM);

    virtual unsigned getJumpTableEncoding(void) const;

    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// ReplaceNodeResults - Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    virtual void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                                    SelectionDAG &DAG) const;

    virtual const char *getTargetNodeName(unsigned Opcode) const;

    virtual bool isSelectSupported(SelectSupportKind Kind) const {
      // ARM does not support scalar condition selects on vectors.
      return (Kind != ScalarCondVectorVal);
    }

    /// getSetCCResultType - Return the value type to use for ISD::SETCC.
    virtual EVT getSetCCResultType(EVT VT) const;

    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    virtual void
    AdjustInstrPostInstrSelection(MachineInstr *MI, SDNode *Node) const;

    SDValue PerformCMOVCombine(SDNode *N, SelectionDAG &DAG) const;
    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    bool isDesirableToTransformToIntegerOp(unsigned Opc, EVT VT) const;

    /// allowsUnalignedMemoryAccesses - Returns true if the target allows
    /// unaligned memory accesses. of the specified type.
    virtual bool allowsUnalignedMemoryAccesses(EVT VT) const;

    virtual EVT getOptimalMemOpType(uint64_t Size,
                                    unsigned DstAlign, unsigned SrcAlign,
                                    bool IsZeroVal,
                                    bool MemcpyStrSrc,
                                    MachineFunction &MF) const;

    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty)const;
    bool isLegalT2ScaledAddressingMode(const AddrMode &AM, EVT VT) const;

    /// isLegalICmpImmediate - Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can
    /// compare a register against the immediate without having to materialize
    /// the immediate into a register.
    virtual bool isLegalICmpImmediate(int64_t Imm) const;

    /// isLegalAddImmediate - Return true if the specified immediate is legal
    /// add immediate, that is the target has add instructions which can
    /// add a register and the immediate without having to materialize
    /// the immediate into a register.
    virtual bool isLegalAddImmediate(int64_t Imm) const;

    /// getPreIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if the node's address
    /// can be legally represented as pre-indexed load / store address.
    virtual bool getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                           SDValue &Offset,
                                           ISD::MemIndexedMode &AM,
                                           SelectionDAG &DAG) const;

    /// getPostIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if this node can be
    /// combined with a load / store to form a post-indexed load / store.
    virtual bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                            SDValue &Base, SDValue &Offset,
                                            ISD::MemIndexedMode &AM,
                                            SelectionDAG &DAG) const;

    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                APInt &KnownZero,
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth) const;


    virtual bool ExpandInlineAsm(CallInst *CI) const;

    ConstraintType getConstraintType(const std::string &Constraint) const;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

    std::pair<unsigned, const TargetRegisterClass*>
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   EVT VT) const;

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops. If hasMemory is
    /// true it means one of the asm constraint of the inline asm instruction
    /// being processed is 'm'.
    virtual void LowerAsmOperandForConstraint(SDValue Op,
                                              std::string &Constraint,
                                              std::vector<SDValue> &Ops,
                                              SelectionDAG &DAG) const;

    const ARMSubtarget* getSubtarget() const {
      return Subtarget;
    }

    /// getRegClassFor - Return the register class that should be used for the
    /// specified value type.
    virtual const TargetRegisterClass *getRegClassFor(EVT VT) const;

    /// getMaximalGlobalOffset - Returns the maximal possible offset which can
    /// be used for loads / stores from the global.
    virtual unsigned getMaximalGlobalOffset() const;

    /// createFastISel - This method returns a target specific FastISel object,
    /// or null if the target does not support "fast" ISel.
    virtual FastISel *createFastISel(FunctionLoweringInfo &funcInfo,
                                     const TargetLibraryInfo *libInfo) const;

    Sched::Preference getSchedulingPreference(SDNode *N) const;

    bool isShuffleMaskLegal(const SmallVectorImpl<int> &M, EVT VT) const;
    bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    virtual bool getTgtMemIntrinsic(IntrinsicInfo &Info,
                                    const CallInst &I,
                                    unsigned Intrinsic) const;
  protected:
    std::pair<const TargetRegisterClass*, uint8_t>
    findRepresentativeClass(EVT VT) const;

  private:
    /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
    /// make the right decision when generating code for different targets.
    const ARMSubtarget *Subtarget;

    const TargetRegisterInfo *RegInfo;

    const InstrItineraryData *Itins;

    /// ARMPCLabelIndex - Keep track of the number of ARM PC labels created.
    ///
    unsigned ARMPCLabelIndex;

    void addTypeForNEON(MVT VT, MVT PromotedLdStVT, MVT PromotedBitwiseVT);
    void addDRTypeForNEON(MVT VT);
    void addQRTypeForNEON(MVT VT);

    typedef SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPassVector;
    void PassF64ArgInRegs(DebugLoc dl, SelectionDAG &DAG,
                          SDValue Chain, SDValue &Arg,
                          RegsToPassVector &RegsToPass,
                          CCValAssign &VA, CCValAssign &NextVA,
                          SDValue &StackPtr,
                          SmallVector<SDValue, 8> &MemOpChains,
                          ISD::ArgFlagsTy Flags) const;
    SDValue GetF64FormalArgument(CCValAssign &VA, CCValAssign &NextVA,
                                 SDValue &Root, SelectionDAG &DAG,
                                 DebugLoc dl) const;

    CCAssignFn *CCAssignFnForNode(CallingConv::ID CC, bool Return,
                                  bool isVarArg) const;
    SDValue LowerMemOpCallTo(SDValue Chain, SDValue StackPtr, SDValue Arg,
                             DebugLoc dl, SelectionDAG &DAG,
                             const CCValAssign &VA,
                             ISD::ArgFlagsTy Flags) const;
    SDValue LowerEH_SJLJ_SETJMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_SJLJ_LONGJMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG,
                                    const ARMSubtarget *Subtarget) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddressDarwin(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddressELF(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
                                            SelectionDAG &DAG) const;
    SDValue LowerToTLSExecModels(GlobalAddressSDNode *GA,
                                 SelectionDAG &DAG,
                                 TLSModel::Model model) const;
    SDValue LowerGLOBAL_OFFSET_TABLE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBR_JT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerShiftRightParts(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFLT_ROUNDS_(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantFP(SDValue Op, SelectionDAG &DAG,
                            const ARMSubtarget *ST) const;
    SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG,
                              const ARMSubtarget *ST) const;

    SDValue ReconstructShuffle(SDValue Op, SelectionDAG &DAG) const;

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

    void VarArgStyleRegisters(CCState &CCInfo, SelectionDAG &DAG,
                              DebugLoc dl, SDValue &Chain,
                              const Value *OrigArg,
                              unsigned OffsetFromOrigArg,
                              unsigned ArgOffset,
                              bool ForceMutable = false)
      const;

    void computeRegArea(CCState &CCInfo, MachineFunction &MF,
                        unsigned &VARegSize, unsigned &VARegSaveSize) const;

    virtual SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const;

    /// HandleByVal - Target-specific cleanup for ByVal support.
    virtual void HandleByVal(CCState *, unsigned &, unsigned) const;

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Targets which want to do tail call
    /// optimization should implement this function.
    bool IsEligibleForTailCallOptimization(SDValue Callee,
                                           CallingConv::ID CalleeCC,
                                           bool isVarArg,
                                           bool isCalleeStructRet,
                                           bool isCallerStructRet,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                           SelectionDAG& DAG) const;
    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;

    virtual bool isUsedByReturnOnly(SDNode *N, SDValue &Chain) const;

    virtual bool mayBeEmittedAsTailCall(CallInst *CI) const;

    SDValue getARMCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                      SDValue &ARMcc, SelectionDAG &DAG, DebugLoc dl) const;
    SDValue getVFPCmp(SDValue LHS, SDValue RHS,
                      SelectionDAG &DAG, DebugLoc dl) const;
    SDValue duplicateCmp(SDValue Cmp, SelectionDAG &DAG) const;

    SDValue OptimizeVFPBrcond(SDValue Op, SelectionDAG &DAG) const;

    MachineBasicBlock *EmitAtomicCmpSwap(MachineInstr *MI,
                                         MachineBasicBlock *BB,
                                         unsigned Size) const;
    MachineBasicBlock *EmitAtomicBinary(MachineInstr *MI,
                                        MachineBasicBlock *BB,
                                        unsigned Size,
                                        unsigned BinOpcode) const;
    MachineBasicBlock *EmitAtomicBinary64(MachineInstr *MI,
                                          MachineBasicBlock *BB,
                                          unsigned Op1,
                                          unsigned Op2,
                                          bool NeedsCarry = false,
                                          bool IsCmpxchg = false) const;
    MachineBasicBlock * EmitAtomicBinaryMinMax(MachineInstr *MI,
                                               MachineBasicBlock *BB,
                                               unsigned Size,
                                               bool signExtend,
                                               ARMCC::CondCodes Cond) const;

    void SetupEntryBlockForSjLj(MachineInstr *MI,
                                MachineBasicBlock *MBB,
                                MachineBasicBlock *DispatchBB, int FI) const;

    MachineBasicBlock *EmitSjLjDispatchBlock(MachineInstr *MI,
                                             MachineBasicBlock *MBB) const;

    bool RemapAddSubWithFlags(MachineInstr *MI, MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitStructByval(MachineInstr *MI,
                                       MachineBasicBlock *MBB) const;
  };

  enum NEONModImmType {
    VMOVModImm,
    VMVNModImm,
    OtherModImm
  };


  namespace ARM {
    FastISel *createFastISel(FunctionLoweringInfo &funcInfo,
                             const TargetLibraryInfo *libInfo);
  }
}

#endif  // ARMISELLOWERING_H
