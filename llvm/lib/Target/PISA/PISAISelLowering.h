//===-- PISAISelLowering.h - PISA DAG Lowering Interface ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAISELLOWERING_H
#define LLVM_LIB_TARGET_PISA_PISAISELLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class PISASubtarget;

class PISATargetLowering : public TargetLowering {
public:
  explicit PISATargetLowering(const TargetMachine &TM,
                              const PISASubtarget &STI);

  bool isCheapToSpeculateCttz(Type *Ty) const override;

  bool isCheapToSpeculateCtlz(Type *Ty) const override;

  bool isReassocProfitable(MachineRegisterInfo &MRI, Register N0,
                           Register N1) const override;

  // Stop IRTranslator breaking up FMA instrs to preserve types information.
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT) const override {
    return true;
  }
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  LLT) const override {
    return true;
  }
  bool isFMAFasterThanFMulAndFAdd(const Function &F, Type *) const override {
    return true;
  }

  // This is to prevent sexts of non-i64 vector indices which are generated
  // within general IRTranslator hence type generation for it is omitted.
  unsigned getVectorIdxWidth(const DataLayout &DL) const override { return 32; }

  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;
  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override;
  void getTgtMemIntrinsic(SmallVectorImpl<IntrinsicInfo> &Infos,
                          const CallBase &I, MachineFunction &MF,
                          unsigned Intrinsic) const override;
  LLT getOptimalMemOpLLT(const MemOp &Op,
                         const AttributeList &FuncAttributes) const override;

  bool useFTZ(const MachineFunction &MF) const;

  bool areJTsAllowed(const Function *Fn) const override { return false; }
  bool isShuffleMaskLegal(ArrayRef<int> /*Mask*/, EVT /*VT*/) const override {
    return false;
  }
  ConstraintType getConstraintType(StringRef Constraint) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  MachineMemOperand::Flags
  getTargetMMOFlags(const Instruction &I) const override;

  bool isFMADLegal(const SelectionDAG &DAG, const SDNode *N) const override {
    return false;
  }
  bool isFMADLegal(const MachineInstr &MI, const LLT Ty) const override {
    return false;
  }

  void computeKnownBitsForTargetInstr(GISelValueTracking &Analysis, Register R,
                                      KnownBits &Known,
                                      const APInt &DemandedElts,
                                      const MachineRegisterInfo &MRI,
                                      unsigned Depth = 0) const override;

  // --- Atomic legalization: driven by AtomicExpandPass ---
  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(const AtomicRMWInst *RMW) const override;
  AtomicExpansionKind
  shouldExpandAtomicCmpXchgInIR(const AtomicCmpXchgInst *CI) const override;
  AtomicExpansionKind shouldExpandAtomicLoadInIR(LoadInst *LI) const override;
  AtomicExpansionKind shouldExpandAtomicStoreInIR(StoreInst *SI) const override;

  bool shouldInsertFencesForAtomic(const Instruction *I) const override;

  // Seed the CAS emulation loop with a plain load. Private/shared/global paths
  // add the synchronization they need outside the loop, so the initial load
  // does not need to be a redundant relaxed atomic load.
  bool shouldIssueAtomicLoadForAtomicEmulationLoop() const override {
    return false;
  }

  Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                AtomicOrdering Ord) const override;
  Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                 AtomicOrdering Ord) const override;
  void emitExpandAtomicRMW(AtomicRMWInst *AI) const override;
  void emitExpandAtomicStore(StoreInst *SI) const override;
  void emitExpandAtomicLoad(LoadInst *LI) const override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAISELLOWERING_H
