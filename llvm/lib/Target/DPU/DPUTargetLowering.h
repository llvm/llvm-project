//===-- DPUTargetLowering.h - DPU DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that DPU uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUTARGETLOWERING_H
#define LLVM_LIB_TARGET_DPU_DPUTARGETLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class DPUSubtarget;

class DPUTargetLowering : public TargetLowering {
public:
  explicit DPUTargetLowering(const TargetMachine &TM, DPUSubtarget &STI);

  // To simplify log analysis: put an explicit name on DPU specific nodes.
  const char *getTargetNodeName(unsigned Opcode) const override;

  // Lower incoming arguments, copy physregs into vregs
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  // Lower return values
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &dl,
                      SelectionDAG &DAG) const override;

  // Lower a call into CALLSEQ_START - DPUISD::CALL - CALLSEQ_END chain
  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  // Lower the result values of a call, copying them out of physregs into vregs
  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL,
                          SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals) const;

  // Lowers specific operations.
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  bool isLegalICmpImmediate(int64_t) const override;

  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;
  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

  bool isSelectSupported(SelectSupportKind kind) const override {
    // Whatever the kind is, we want to very strongly discourage using select...
    // It's VERY expensive in most of the cases, so the compiler should try
    // another way, which will certainly produce more efficient code.
    return false;
  }

  SDValue LowerSetCc(SDValue Op, SelectionDAG &DAG) const;

  bool isCheapToSpeculateCtlz() const override { return true; }

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

private:
  CodeGenOpt::Level optLevel;

  // -----------------------------------------------------------------------------
  // Custom lowering.
  // Promotion of several internal structures, such as globals, using the
  // wrapper instruction.
  // -----------------------------------------------------------------------------
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerBrCc(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerStore(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerLoad(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerMultiplication(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerIntrinsic(SDValue Op, SelectionDAG &DAG,
                         int IntrinsicType) const;

  SDValue LowerUnsupported(SDValue Op, SelectionDAG &DAG, StringRef Message) const;

  // Analysis and mute for 64 bits emulation.
  // Shifts: third parameter stands for Left (0), Right (1) or Arithmetic (2)
  SDValue LowerShift(SDValue Op, SelectionDAG &DAG, int LRA) const;

  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
};
} // namespace llvm
#endif
