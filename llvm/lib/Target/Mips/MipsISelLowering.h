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

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace MipsISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // Jump and link (call)
      JmpLink,

      // Tail call
      TailCall,

      // Get the Higher 16 bits from a 32-bit immediate
      // No relation with Mips Hi register
      Hi,

      // Get the Lower 16 bits from a 32-bit immediate
      // No relation with Mips Lo register
      Lo,

      // Handle gp_rel (small data/bss sections) relocation.
      GPRel,

      // Thread Pointer
      ThreadPointer,

      // Floating Point Branch Conditional
      FPBrcond,

      // Floating Point Compare
      FPCmp,

      // Floating Point Conditional Moves
      CMovFP_T,
      CMovFP_F,

      // Floating Point Rounding
      FPRound,

      // Return
      Ret,

      // MAdd/Sub nodes
      MAdd,
      MAddu,
      MSub,
      MSubu,

      // DivRem(u)
      DivRem,
      DivRemU,

      BuildPairF64,
      ExtractElementF64,

      Wrapper,

      DynAlloc,

      Sync,

      Ext,
      Ins,

      // EXTR.W instrinsic nodes.
      EXTP,
      EXTPDP,
      EXTR_S_H,
      EXTR_W,
      EXTR_R_W,
      EXTR_RS_W,
      SHILO,
      MTHLIP,

      // DPA.W intrinsic nodes.
      MULSAQ_S_W_PH,
      MAQ_S_W_PHL,
      MAQ_S_W_PHR,
      MAQ_SA_W_PHL,
      MAQ_SA_W_PHR,
      DPAU_H_QBL,
      DPAU_H_QBR,
      DPSU_H_QBL,
      DPSU_H_QBR,
      DPAQ_S_W_PH,
      DPSQ_S_W_PH,
      DPAQ_SA_L_W,
      DPSQ_SA_L_W,
      DPA_W_PH,
      DPS_W_PH,
      DPAQX_S_W_PH,
      DPAQX_SA_W_PH,
      DPAX_W_PH,
      DPSX_W_PH,
      DPSQX_S_W_PH,
      DPSQX_SA_W_PH,
      MULSA_W_PH,

      MULT,
      MULTU,
      MADD_DSP,
      MADDU_DSP,
      MSUB_DSP,
      MSUBU_DSP,

      // Load/Store Left/Right nodes.
      LWL = ISD::FIRST_TARGET_MEMORY_OPCODE,
      LWR,
      SWL,
      SWR,
      LDL,
      LDR,
      SDL,
      SDR
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class MipsFunctionInfo;

  class MipsTargetLowering : public TargetLowering  {
  public:
    explicit MipsTargetLowering(MipsTargetMachine &TM);

    virtual MVT getShiftAmountTy(EVT LHSTy) const { return MVT::i32; }

    virtual bool allowsUnalignedMemoryAccesses (EVT VT) const;

    virtual void LowerOperationWrapper(SDNode *N,
                                       SmallVectorImpl<SDValue> &Results,
                                       SelectionDAG &DAG) const;

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

    /// getSetCCResultType - get the ISD::SETCC result ValueType
    EVT getSetCCResultType(EVT VT) const;

    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  private:

    /// ByValArgInfo - Byval argument information.
    struct ByValArgInfo {
      unsigned FirstIdx; // Index of the first register used.
      unsigned NumRegs;  // Number of registers used for this argument.
      unsigned Address;  // Offset of the stack area used to pass this argument.

      ByValArgInfo() : FirstIdx(0), NumRegs(0), Address(0) {}
    };

    /// MipsCC - This class provides methods used to analyze formal and call
    /// arguments and inquire about calling convention information.
    class MipsCC {
    public:
      MipsCC(CallingConv::ID CallConv, bool IsVarArg, bool IsO32,
             CCState &Info);

      void analyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs);
      void analyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins);
      void handleByValArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                          CCValAssign::LocInfo LocInfo,
                          ISD::ArgFlagsTy ArgFlags);

      const CCState &getCCInfo() const { return CCInfo; }

      /// hasByValArg - Returns true if function has byval arguments.
      bool hasByValArg() const { return !ByValArgs.empty(); }

      /// useRegsForByval - Returns true if the calling convention allows the
      /// use of registers to pass byval arguments.
      bool useRegsForByval() const { return UseRegsForByval; }

      /// regSize - Size (in number of bits) of integer registers.
      unsigned regSize() const { return RegSize; }

      /// numIntArgRegs - Number of integer registers available for calls.
      unsigned numIntArgRegs() const { return NumIntArgRegs; }

      /// reservedArgArea - The size of the area the caller reserves for
      /// register arguments. This is 16-byte if ABI is O32.
      unsigned reservedArgArea() const { return ReservedArgArea; }

      /// intArgRegs - Pointer to array of integer registers.
      const uint16_t *intArgRegs() const { return IntArgRegs; }

      typedef SmallVector<ByValArgInfo, 2>::const_iterator byval_iterator;
      byval_iterator byval_begin() const { return ByValArgs.begin(); }
      byval_iterator byval_end() const { return ByValArgs.end(); }

    private:
      void allocateRegs(ByValArgInfo &ByVal, unsigned ByValSize,
                        unsigned Align);

      CCState &CCInfo;
      bool UseRegsForByval;
      unsigned RegSize;
      unsigned NumIntArgRegs;
      unsigned ReservedArgArea;
      const uint16_t *IntArgRegs, *ShadowRegs;
      SmallVector<ByValArgInfo, 2> ByValArgs;
      llvm::CCAssignFn *FixedFn, *VarFn;
    };

    // Subtarget Info
    const MipsSubtarget *Subtarget;

    bool HasMips64, IsN64, IsO32;

    // Lower Operand helpers
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

    // Lower Operand specifics
    SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFABS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerMEMBARRIER(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerShiftRightParts(SDValue Op, SelectionDAG& DAG,
                                 bool IsSRA) const;
    SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADD(SDValue Op, SelectionDAG &DAG) const;

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization.
    bool IsEligibleForTailCallOptimization(const MipsCC &MipsCCInfo,
                                           unsigned NextStackOffset,
                                           const MipsFunctionInfo& FI) const;

    /// copyByValArg - Copy argument registers which were used to pass a byval
    /// argument to the stack. Create a stack frame object for the byval
    /// argument.
    void copyByValRegs(SDValue Chain, DebugLoc DL,
                       std::vector<SDValue> &OutChains, SelectionDAG &DAG,
                       const ISD::ArgFlagsTy &Flags,
                       SmallVectorImpl<SDValue> &InVals,
                       const Argument *FuncArg,
                       const MipsCC &CC, const ByValArgInfo &ByVal) const;

    /// passByValArg - Pass a byval argument in registers or on stack.
    void passByValArg(SDValue Chain, DebugLoc DL,
                      SmallVector<std::pair<unsigned, SDValue>, 16> &RegsToPass,
                      SmallVector<SDValue, 8> &MemOpChains, SDValue StackPtr,
                      MachineFrameInfo *MFI, SelectionDAG &DAG, SDValue Arg,
                      const MipsCC &CC, const ByValArgInfo &ByVal,
                      const ISD::ArgFlagsTy &Flags, bool isLittle) const;

    /// writeVarArgRegs - Write variable function arguments passed in registers
    /// to the stack. Also create a stack frame object for the first variable
    /// argument.
    void writeVarArgRegs(std::vector<SDValue> &OutChains, const MipsCC &CC,
                         SDValue Chain, DebugLoc DL, SelectionDAG &DAG) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    SDValue passArgOnStack(SDValue StackPtr, unsigned Offset, SDValue Chain,
                           SDValue Arg, DebugLoc DL, bool IsTailCall,
                           SelectionDAG &DAG) const;

    virtual SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual bool
      CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                     bool isVarArg,
                     const SmallVectorImpl<ISD::OutputArg> &Outs,
                     LLVMContext &Context) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;

    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    // Inline asm support
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

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    virtual EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                    unsigned SrcAlign, bool IsZeroVal,
                                    bool MemcpyStrSrc,
                                    MachineFunction &MF) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    virtual unsigned getJumpTableEncoding() const;

    MachineBasicBlock *EmitBPOSGE32(MachineInstr *MI,
                                    MachineBasicBlock *BB) const;
    MachineBasicBlock *EmitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                    unsigned Size, unsigned BinOpcode, bool Nand = false) const;
    MachineBasicBlock *EmitAtomicBinaryPartword(MachineInstr *MI,
                    MachineBasicBlock *BB, unsigned Size, unsigned BinOpcode,
                    bool Nand = false) const;
    MachineBasicBlock *EmitAtomicCmpSwap(MachineInstr *MI,
                                  MachineBasicBlock *BB, unsigned Size) const;
    MachineBasicBlock *EmitAtomicCmpSwapPartword(MachineInstr *MI,
                                  MachineBasicBlock *BB, unsigned Size) const;
  };
}

#endif // MipsISELLOWERING_H
