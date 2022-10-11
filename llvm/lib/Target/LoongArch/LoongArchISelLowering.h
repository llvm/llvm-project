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
struct LoongArchRegisterInfo;
namespace LoongArchISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // TODO: add more LoongArchISDs
  CALL,
  RET,
  // 32-bit shifts, directly matching the semantics of the named LoongArch
  // instructions.
  SLL_W,
  SRA_W,
  SRL_W,

  ROTL_W,
  ROTR_W,

  // FPR<->GPR transfer operations
  MOVGR2FR_W_LA64,
  MOVFR2GR_S_LA64,

  FTINT,

  // Bit counting operations
  CLZ_W,
  CTZ_W,

  BSTRINS,
  BSTRPICK,

  // Byte-swapping and bit-reversal
  REVB_2H,
  REVB_2W,
  BITREV_4B,
  BITREV_W,
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
                      LLVMContext &Context) const override;
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

  Value *emitMaskedAtomicRMWIntrinsic(IRBuilderBase &Builder, AtomicRMWInst *AI,
                                      Value *AlignedAddr, Value *Incr,
                                      Value *Mask, Value *ShiftAmt,
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

private:
  /// Target-specific function used to lower LoongArch calling conventions.
  typedef bool LoongArchCCAssignFn(const DataLayout &DL, LoongArchABI::ABI ABI,
                                   unsigned ValNo, MVT ValVT,
                                   CCValAssign::LocInfo LocInfo,
                                   ISD::ArgFlagsTy ArgFlags, CCState &State,
                                   bool IsFixed, bool IsReg, Type *OrigTy);

  void analyzeInputArgs(MachineFunction &MF, CCState &CCInfo,
                        const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
                        LoongArchCCAssignFn Fn) const;
  void analyzeOutputArgs(MachineFunction &MF, CCState &CCInfo,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         bool IsRet, CallLoweringInfo *CLI,
                         LoongArchCCAssignFn Fn) const;

  template <class NodeTy>
  SDValue getAddr(NodeTy *N, SelectionDAG &DAG, bool IsLocal = true) const;
  SDValue getStaticTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                           unsigned Opc) const;
  SDValue getDynamicTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                            unsigned Opc) const;
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShiftRightParts(SDValue Op, SelectionDAG &DAG, bool IsSRA) const;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;
  SDValue lowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEH_DWARF_CFA(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBITCAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;

  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;

  bool shouldInsertFencesForAtomic(const Instruction *I) const override;

  ConstraintType getConstraintType(StringRef Constraint) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H
