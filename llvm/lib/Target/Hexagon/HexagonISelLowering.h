//===-- HexagonISelLowering.h - Hexagon DAG Lowering Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Hexagon uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H

#include "Hexagon.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

// Return true when the given node fits in a positive half word.
bool isPositiveHalfWord(SDNode *N);

  namespace HexagonISD {
    enum NodeType : unsigned {
      OP_BEGIN = ISD::BUILTIN_OP_END,

      CONST32 = OP_BEGIN,
      CONST32_GP,  // For marking data present in GP.
      FCONST32,
      ALLOCA,
      ARGEXTEND,

      AT_GOT,      // Index in GOT.
      AT_PCREL,    // Offset relative to PC.

      CALLv3,      // A V3+ call instruction.
      CALLv3nr,    // A V3+ call instruction that doesn't return.
      CALLR,

      RET_FLAG,    // Return with a flag operand.
      BARRIER,     // Memory barrier.
      JT,          // Jump table.
      CP,          // Constant pool.

      POPCOUNT,
      COMBINE,
      PACKHL,
      VSPLATB,
      VSPLATH,
      SHUFFEB,
      SHUFFEH,
      SHUFFOB,
      SHUFFOH,
      VSXTBH,
      VSXTBW,
      VSRAW,
      VSRAH,
      VSRLW,
      VSRLH,
      VSHLW,
      VSHLH,
      VCMPBEQ,
      VCMPBGT,
      VCMPBGTU,
      VCMPHEQ,
      VCMPHGT,
      VCMPHGTU,
      VCMPWEQ,
      VCMPWGT,
      VCMPWGTU,

      INSERT,
      INSERTRP,
      EXTRACTU,
      EXTRACTURP,
      VCOMBINE,
      TC_RETURN,
      EH_RETURN,
      DCFETCH,

      OP_END
    };
  }

  class HexagonSubtarget;

  class HexagonTargetLowering : public TargetLowering {
    int VarArgsFrameOffset;   // Frame offset to start of varargs area.

    bool CanReturnSmallStruct(const Function* CalleeFn, unsigned& RetSize)
        const;
    void promoteLdStType(MVT VT, MVT PromotedLdStVT);
    const HexagonTargetMachine &HTM;
    const HexagonSubtarget &Subtarget;

  public:
    explicit HexagonTargetLowering(const TargetMachine &TM,
                                   const HexagonSubtarget &ST);

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Targets which want to do tail call
    /// optimization should implement this function.
    bool IsEligibleForTailCallOptimization(SDValue Callee,
        CallingConv::ID CalleeCC, bool isVarArg, bool isCalleeStructRet,
        bool isCallerStructRet, const SmallVectorImpl<ISD::OutputArg> &Outs,
        const SmallVectorImpl<SDValue> &OutVals,
        const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG& DAG) const;

    bool isTruncateFree(Type *Ty1, Type *Ty2) const override;
    bool isTruncateFree(EVT VT1, EVT VT2) const override;

    bool allowTruncateForTailCall(Type *Ty1, Type *Ty2) const override;

    // Should we expand the build vector with shuffles?
    bool shouldExpandBuildVectorWithShuffles(EVT VT,
        unsigned DefinedValues) const override;

    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
    const char *getTargetNodeName(unsigned Opcode) const override;
    SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEXTRACT_VECTOR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINSERT_VECTOR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINLINEASM(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerPREFETCH(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_LABEL(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const;
    SDValue
    LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerGLOBALADDRESS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
        SelectionDAG &DAG) const;
    SDValue LowerToTLSInitialExecModel(GlobalAddressSDNode *GA,
        SelectionDAG &DAG) const;
    SDValue LowerToTLSLocalExecModel(GlobalAddressSDNode *GA,
        SelectionDAG &DAG) const;
    SDValue GetDynamicTLSAddr(SelectionDAG &DAG, SDValue Chain,
        GlobalAddressSDNode *GA, SDValue *InFlag, EVT PtrVT,
        unsigned ReturnReg, unsigned char OperandFlags) const;
    SDValue LowerGLOBAL_OFFSET_TABLE(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
        SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            const SDLoc &dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals,
                            const SmallVectorImpl<SDValue> &OutVals,
                            SDValue Callee) const;

    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerCTPOP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        const SDLoc &dl, SelectionDAG &DAG) const override;

    bool mayBeEmittedAsTailCall(CallInst *CI) const override;
    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr &MI,
                                MachineBasicBlock *BB) const override;

    /// If a physical register, this returns the register that receives the
    /// exception address on entry to an EH pad.
    unsigned
    getExceptionPointerRegister(const Constant *PersonalityFn) const override {
      return Hexagon::R0;
    }

    /// If a physical register, this returns the register that receives the
    /// exception typeid on entry to a landing pad.
    unsigned
    getExceptionSelectorRegister(const Constant *PersonalityFn) const override {
      return Hexagon::R1;
    }

    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    EVT getSetCCResultType(const DataLayout &, LLVMContext &C,
                           EVT VT) const override {
      if (!VT.isVector())
        return MVT::i1;
      else
        return EVT::getVectorVT(C, MVT::i1, VT.getVectorNumElements());
    }

    bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                    SDValue &Base, SDValue &Offset,
                                    ISD::MemIndexedMode &AM,
                                    SelectionDAG &DAG) const override;

    ConstraintType getConstraintType(StringRef Constraint) const override;

    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 StringRef Constraint, MVT VT) const override;

    unsigned
    getInlineAsmMemConstraint(StringRef ConstraintCode) const override {
      if (ConstraintCode == "o")
        return InlineAsm::Constraint_o;
      return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
    }

    // Intrinsics
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    /// The type may be VoidTy, in which case only return true if the addressing
    /// mode is legal for a load/store of any legal type.
    /// TODO: Handle pre/postinc as well.
    bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM,
                               Type *Ty, unsigned AS) const override;
    /// Return true if folding a constant offset with the given GlobalAddress
    /// is legal.  It is frequently not legal in PIC relocation models.
    bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

    bool isFPImmLegal(const APFloat &Imm, EVT VT) const override;

    /// isLegalICmpImmediate - Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can
    /// compare a register against the immediate without having to materialize
    /// the immediate into a register.
    bool isLegalICmpImmediate(int64_t Imm) const override;

    bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AddrSpace,
        unsigned Align, bool *Fast) const override;

    /// Returns relocation base for the given PIC jumptable.
    SDValue getPICJumpTableRelocBase(SDValue Table, SelectionDAG &DAG)
                                     const override;

    // Handling of atomic RMW instructions.
    Value *emitLoadLinked(IRBuilder<> &Builder, Value *Addr,
        AtomicOrdering Ord) const override;
    Value *emitStoreConditional(IRBuilder<> &Builder, Value *Val,
        Value *Addr, AtomicOrdering Ord) const override;
    AtomicExpansionKind shouldExpandAtomicLoadInIR(LoadInst *LI) const override;
    bool shouldExpandAtomicStoreInIR(StoreInst *SI) const override;
    bool shouldExpandAtomicCmpXchgInIR(AtomicCmpXchgInst *AI) const override;

    AtomicExpansionKind
    shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override {
      return AtomicExpansionKind::LLSC;
    }

  protected:
    std::pair<const TargetRegisterClass*, uint8_t>
    findRepresentativeClass(const TargetRegisterInfo *TRI, MVT VT)
        const override;
  };
} // end namespace llvm

#endif    // Hexagon_ISELLOWERING_H
