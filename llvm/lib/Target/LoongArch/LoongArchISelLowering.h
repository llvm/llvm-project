//=- LoongArchISelLowering.h - LoongArch DAG Lowering Interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that LoongArch uses to lower LLVM code into
// a selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H

#include "LoongArch.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class LoongArchSubtarget;
namespace LoongArchISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // TODO: add more LoongArchISDs
  CALL,
  CALL_MEDIUM,
  CALL_LARGE,
  RET,
  TAIL,
  TAIL_MEDIUM,
  TAIL_LARGE,

  // Select
  SELECT_CC,

  // Branch
  BR_CC,
  BRCOND,

  // 32-bit shifts, directly matching the semantics of the named LoongArch
  // instructions.
  SLL_W,
  SRA_W,
  SRL_W,

  ROTL_W,
  ROTR_W,

  // unsigned 32-bit integer division
  DIV_W,
  MOD_W,
  DIV_WU,
  MOD_WU,

  // FPR<->GPR transfer operations
  MOVGR2FR_W,
  MOVGR2FR_W_LA64,
  MOVGR2FR_D,
  MOVGR2FR_D_LO_HI,
  MOVFR2GR_S_LA64,
  MOVFCSR2GR,
  MOVGR2FCSR,

  FTINT,

  // Build and split F64 pair
  BUILD_PAIR_F64,
  SPLIT_PAIR_F64,

  // Bit counting operations
  CLZ_W,
  CTZ_W,

  BSTRINS,
  BSTRPICK,

  // Byte-swapping and bit-reversal
  REVB_2H,
  REVB_2W,
  BITREV_4B,
  BITREV_8B,
  BITREV_W,

  // Intrinsic operations start ============================================
  BREAK,
  CACOP_D,
  CACOP_W,
  DBAR,
  IBAR,
  SYSCALL,

  // CRC check operations
  CRC_W_B_W,
  CRC_W_H_W,
  CRC_W_W_W,
  CRC_W_D_W,
  CRCC_W_B_W,
  CRCC_W_H_W,
  CRCC_W_W_W,
  CRCC_W_D_W,

  CSRRD,

  // Write new value to CSR and return old value.
  // Operand 0: A chain pointer.
  // Operand 1: The new value to write.
  // Operand 2: The address of the required CSR.
  // Result 0: The old value of the CSR.
  // Result 1: The new chain pointer.
  CSRWR,

  // Similar to CSRWR but with a write mask.
  // Operand 0: A chain pointer.
  // Operand 1: The new value to write.
  // Operand 2: The write mask.
  // Operand 3: The address of the required CSR.
  // Result 0: The old value of the CSR.
  // Result 1: The new chain pointer.
  CSRXCHG,

  // IOCSR access operations
  IOCSRRD_B,
  IOCSRRD_W,
  IOCSRRD_H,
  IOCSRRD_D,
  IOCSRWR_B,
  IOCSRWR_H,
  IOCSRWR_W,
  IOCSRWR_D,

  // Read CPU configuration information operation
  CPUCFG,

  // Vector Shuffle
  VREPLVE,
  VSHUF,
  VPICKEV,
  VPICKOD,
  VPACKEV,
  VPACKOD,
  VILVL,
  VILVH,
  VSHUF4I,
  VREPLVEI,
  VREPLGR2VR,
  XVPERMI,
  XVPERM,

  // Extended vector element extraction
  VPICK_SEXT_ELT,
  VPICK_ZEXT_ELT,

  // Vector comparisons
  VALL_ZERO,
  VANY_ZERO,
  VALL_NONZERO,
  VANY_NONZERO,

  // Floating point approximate reciprocal operation
  FRECIPE,
  FRSQRTE,

  // Vector logicial left / right shift by immediate
  VSLLI,
  VSRLI,

  // Vector byte logicial left / right shift
  VBSLL,
  VBSRL,

  // Scalar load broadcast to vector
  VLDREPL,

  // Vector mask set by condition
  VMSKLTZ,
  VMSKGEZ,
  VMSKEQZ,
  VMSKNEZ,
  XVMSKLTZ,
  XVMSKGEZ,
  XVMSKEQZ,
  XVMSKNEZ,

  // Vector Horizontal Addition with Widening‌
  VHADDW

  // Intrinsic operations end =============================================
};
} // end namespace LoongArchISD

class LoongArchTargetLowering : public TargetLowering {
  const LoongArchSubtarget &Subtarget;

public:
  explicit LoongArchTargetLowering(const TargetMachine &TM,
                                   const LoongArchSubtarget &STI);

  const LoongArchSubtarget &getSubtarget() const { return Subtarget; }

  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

  // Provide custom lowering hooks for some operations.
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  // This method returns the name of a target specific DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  // Lower incoming arguments, copy physregs into vregs.
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;
  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context, const Type *RetTy) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;
  bool isCheapToSpeculateCttz(Type *Ty) const override;
  bool isCheapToSpeculateCtlz(Type *Ty) const override;
  bool hasAndNot(SDValue Y) const override;
  TargetLowering::AtomicExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override;
  void emitExpandAtomicRMW(AtomicRMWInst *AI) const override;

  Value *emitMaskedAtomicRMWIntrinsic(IRBuilderBase &Builder, AtomicRMWInst *AI,
                                      Value *AlignedAddr, Value *Incr,
                                      Value *Mask, Value *ShiftAmt,
                                      AtomicOrdering Ord) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;
  TargetLowering::AtomicExpansionKind
  shouldExpandAtomicCmpXchgInIR(AtomicCmpXchgInst *CI) const override;
  Value *emitMaskedAtomicCmpXchgIntrinsic(IRBuilderBase &Builder,
                                          AtomicCmpXchgInst *CI,
                                          Value *AlignedAddr, Value *CmpVal,
                                          Value *NewVal, Value *Mask,
                                          AtomicOrdering Ord) const override;

  bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallInst &I,
                          MachineFunction &MF,
                          unsigned Intrinsic) const override;

  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override;

  Register
  getExceptionPointerRegister(const Constant *PersonalityFn) const override;

  Register
  getExceptionSelectorRegister(const Constant *PersonalityFn) const override;

  bool isFsqrtCheap(SDValue Operand, SelectionDAG &DAG) const override {
    return true;
  }

  SDValue getSqrtEstimate(SDValue Operand, SelectionDAG &DAG, int Enabled,
                          int &RefinementSteps, bool &UseOneConstNR,
                          bool Reciprocal) const override;

  SDValue getRecipEstimate(SDValue Operand, SelectionDAG &DAG, int Enabled,
                           int &RefinementSteps) const override;

  ISD::NodeType getExtendForAtomicOps() const override {
    return ISD::SIGN_EXTEND;
  }

  ISD::NodeType getExtendForAtomicCmpSwapArg() const override;

  Register getRegisterByName(const char *RegName, LLT VT,
                             const MachineFunction &MF) const override;
  bool mayBeEmittedAsTailCall(const CallInst *CI) const override;

  bool decomposeMulByConstant(LLVMContext &Context, EVT VT,
                              SDValue C) const override;

  bool isUsedByReturnOnly(SDNode *N, SDValue &Chain) const override;

  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;

  bool isLegalICmpImmediate(int64_t Imm) const override;
  bool isLegalAddImmediate(int64_t Imm) const override;
  bool isZExtFree(SDValue Val, EVT VT2) const override;
  bool isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const override;
  bool signExtendConstant(const ConstantInt *CI) const override;

  bool hasAndNotCompare(SDValue Y) const override;

  bool convertSelectOfConstantsToMath(EVT VT) const override { return true; }

  bool allowsMisalignedMemoryAccesses(
      EVT VT, unsigned AddrSpace = 0, Align Alignment = Align(1),
      MachineMemOperand::Flags Flags = MachineMemOperand::MONone,
      unsigned *Fast = nullptr) const override;

  bool isShuffleMaskLegal(ArrayRef<int> Mask, EVT VT) const override {
    if (!VT.isSimple())
      return false;

    // Not for i1 vectors
    if (VT.getSimpleVT().getScalarType() == MVT::i1)
      return false;

    return isTypeLegal(VT.getSimpleVT());
  }
  bool shouldConsiderGEPOffsetSplit() const override { return true; }
  bool shouldSignExtendTypeInLibCall(Type *Ty, bool IsSigned) const override;
  bool shouldExtendTypeInLibCall(EVT Type) const override;

  bool shouldAlignPointerArgs(CallInst *CI, unsigned &MinSize,
                              Align &PrefAlign) const override;

  bool isFPImmVLDILegal(const APFloat &Imm, EVT VT) const;
  LegalizeTypeAction getPreferredVectorAction(MVT VT) const override;

  bool SimplifyDemandedBitsForTargetNode(SDValue Op, const APInt &DemandedBits,
                                         const APInt &DemandedElts,
                                         KnownBits &Known,
                                         TargetLoweringOpt &TLO,
                                         unsigned Depth) const override;

private:
  /// Target-specific function used to lower LoongArch calling conventions.
  typedef bool LoongArchCCAssignFn(const DataLayout &DL, LoongArchABI::ABI ABI,
                                   unsigned ValNo, MVT ValVT,
                                   CCValAssign::LocInfo LocInfo,
                                   ISD::ArgFlagsTy ArgFlags, CCState &State,
                                   bool IsRet, Type *OrigTy);

  void analyzeInputArgs(MachineFunction &MF, CCState &CCInfo,
                        const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
                        LoongArchCCAssignFn Fn) const;
  void analyzeOutputArgs(MachineFunction &MF, CCState &CCInfo,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         bool IsRet, CallLoweringInfo *CLI,
                         LoongArchCCAssignFn Fn) const;

  template <class NodeTy>
  SDValue getAddr(NodeTy *N, SelectionDAG &DAG, CodeModel::Model M,
                  bool IsLocal = true) const;
  SDValue getStaticTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                           unsigned Opc, bool UseGOT, bool Large = false) const;
  SDValue getDynamicTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                            unsigned Opc, bool Large = false) const;
  SDValue getTLSDescAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                         unsigned Opc, bool Large = false) const;
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShiftRightParts(SDValue Op, SelectionDAG &DAG, bool IsSRA) const;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;
  SDValue lowerATOMIC_FENCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEH_DWARF_CFA(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBITCAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerWRITE_REGISTER(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBITREVERSE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerPREFETCH(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_TO_FP16(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP16_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_TO_BF16(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBF16_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECREDUCE_ADD(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECREDUCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerConstantFP(SDValue Op, SelectionDAG &DAG) const;

  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;

  bool shouldInsertFencesForAtomic(const Instruction *I) const override;

  ConstraintType getConstraintType(StringRef Constraint) const override;

  InlineAsm::ConstraintCode
  getInlineAsmMemConstraint(StringRef ConstraintCode) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  bool isEligibleForTailCallOptimization(
      CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
      const SmallVectorImpl<CCValAssign> &ArgLocs) const;

  bool softPromoteHalfType() const override { return true; }

  bool
  splitValueIntoRegisterParts(SelectionDAG &DAG, const SDLoc &DL, SDValue Val,
                              SDValue *Parts, unsigned NumParts, MVT PartVT,
                              std::optional<CallingConv::ID> CC) const override;

  SDValue
  joinRegisterPartsIntoValue(SelectionDAG &DAG, const SDLoc &DL,
                             const SDValue *Parts, unsigned NumParts,
                             MVT PartVT, EVT ValueVT,
                             std::optional<CallingConv::ID> CC) const override;

  /// Return the register type for a given MVT, ensuring vectors are treated
  /// as a series of gpr sized integers.
  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override;

  /// Return the number of registers for a given MVT, ensuring vectors are
  /// treated as a series of gpr sized integers.
  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H
