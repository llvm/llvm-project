//===-- VEISelLowering.h - VE DAG Lowering Interface ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that VE uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_VEISELLOWERING_H
#define LLVM_LIB_TARGET_VE_VEISELLOWERING_H

#include "VE.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class VESubtarget;

namespace VEISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  CALL,                   // A call instruction.
  EH_SJLJ_LONGJMP,        // SjLj exception handling longjmp.
  EH_SJLJ_SETJMP,         // SjLj exception handling setjmp.
  EH_SJLJ_SETUP_DISPATCH, // SjLj exception handling setup_dispatch.
  GETFUNPLT,              // Load function address through %plt insturction.
  GETTLSADDR,             // Load address for TLS access.
  GETSTACKTOP,            // Retrieve address of stack top (first address of
                          // locals and temporaries).
  GLOBAL_BASE_REG,        // Global base reg for PIC.
  Hi,                     // Hi/Lo operations, typically on a global address.
  Lo,                     // Hi/Lo operations, typically on a global address.
  MEMBARRIER,             // Compiler barrier only; generate a no-op.
  RET_FLAG,               // Return with a flag operand.
  TS1AM,                  // A TS1AM instruction used for 1/2 bytes swap.
  VEC_BROADCAST,          // A vector broadcast instruction.
                          //   0: scalar value, 1: VL

// VVP_* nodes.
#define ADD_VVP_OP(VVP_NAME, ...) VVP_NAME,
#include "VVPNodes.def"
};
}

class VETargetLowering : public TargetLowering {
  const VESubtarget *Subtarget;

  void initRegisterClasses();
  void initSPUActions();
  void initVPUActions();

public:
  VETargetLowering(const TargetMachine &TM, const VESubtarget &STI);

  const char *getTargetNodeName(unsigned Opcode) const override;
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
    return MVT::i32;
  }

  Register getRegisterByName(const char *RegName, LLT VT,
                             const MachineFunction &MF) const override;

  /// getSetCCResultType - Return the ISD::SETCC ValueType
  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &ArgsFlags,
                      LLVMContext &Context) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &dl,
                      SelectionDAG &DAG) const override;

  /// Helper functions for atomic operations.
  bool shouldInsertFencesForAtomic(const Instruction *I) const override {
    // VE uses release consistency, so need fence for each atomics.
    return true;
  }
  Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                AtomicOrdering Ord) const override;
  Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                 AtomicOrdering Ord) const override;
  TargetLoweringBase::AtomicExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override;

  /// Custom Lower {
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  unsigned getJumpTableEncoding() const override;
  const MCExpr *LowerCustomJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                          const MachineBasicBlock *MBB,
                                          unsigned Uid,
                                          MCContext &Ctx) const override;
  SDValue getPICJumpTableRelocBase(SDValue Table,
                                   SelectionDAG &DAG) const override;
  // VE doesn't need getPICJumpTableRelocBaseExpr since it is used for only
  // EK_LabelDifference32.

  SDValue lowerATOMIC_FENCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_SWAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEH_SJLJ_LONGJMP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEH_SJLJ_SETJMP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEH_SJLJ_SETUP_DISPATCH(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerToTLSGeneralDynamicModel(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVAARG(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  /// } Custom Lower

  /// Replace the results of node with an illegal result
  /// type with new values built out of custom code.
  ///
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  /// Custom Inserter {
  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *MBB) const override;
  MachineBasicBlock *emitEHSjLjLongJmp(MachineInstr &MI,
                                       MachineBasicBlock *MBB) const;
  MachineBasicBlock *emitEHSjLjSetJmp(MachineInstr &MI,
                                      MachineBasicBlock *MBB) const;
  MachineBasicBlock *emitSjLjDispatchBlock(MachineInstr &MI,
                                           MachineBasicBlock *BB) const;

  void setupEntryBlockForSjLj(MachineInstr &MI, MachineBasicBlock *MBB,
                              MachineBasicBlock *DispatchBB, int FI,
                              int Offset) const;
  // Setup basic block address.
  Register prepareMBB(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                      MachineBasicBlock *TargetBB, const DebugLoc &DL) const;
  // Prepare function/variable address.
  Register prepareSymbol(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                         StringRef Symbol, const DebugLoc &DL, bool IsLocal,
                         bool IsCall) const;
  /// } Custom Inserter

  /// VVP Lowering {
  SDValue lowerToVVP(SDValue Op, SelectionDAG &DAG) const;
  /// } VVPLowering

  /// Custom DAGCombine {
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  SDValue combineTRUNCATE(SDNode *N, DAGCombinerInfo &DCI) const;
  /// } Custom DAGCombine

  SDValue withTargetFlags(SDValue Op, unsigned TF, SelectionDAG &DAG) const;
  SDValue makeHiLoPair(SDValue Op, unsigned HiTF, unsigned LoTF,
                       SelectionDAG &DAG) const;
  SDValue makeAddress(SDValue Op, SelectionDAG &DAG) const;

  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;
  /// Returns true if the target allows unaligned memory accesses of the
  /// specified type.
  bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AS, Align A,
                                      MachineMemOperand::Flags Flags,
                                      bool *Fast) const override;

  /// Inline Assembly {

  ConstraintType getConstraintType(StringRef Constraint) const override;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  /// } Inline Assembly

  /// Target Optimization {

  // Return lower limit for number of blocks in a jump table.
  unsigned getMinimumJumpTableEntries() const override;

  // SX-Aurora VE's s/udiv is 5-9 times slower than multiply.
  bool isIntDivCheap(EVT, AttributeList) const override { return false; }
  // VE doesn't have rem.
  bool hasStandaloneRem(EVT) const override { return false; }
  // VE LDZ instruction returns 64 if the input is zero.
  bool isCheapToSpeculateCtlz() const override { return true; }
  // VE LDZ instruction is fast.
  bool isCtlzFast() const override { return true; }
  // VE has NND instruction.
  bool hasAndNot(SDValue Y) const override;

  /// } Target Optimization
};
} // namespace llvm

#endif // VE_ISELLOWERING_H
