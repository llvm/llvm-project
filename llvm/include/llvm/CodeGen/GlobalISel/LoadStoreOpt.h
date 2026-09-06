//== llvm/CodeGen/GlobalISel/LoadStoreOpt.h - LoadStoreOpt -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This is an optimization pass for GlobalISel generic memory operations.
/// Specifically, it focuses on merging stores and loads to consecutive
/// addresses.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LOADSTOREOPT_H
#define LLVM_CODEGEN_GLOBALISEL_LOADSTOREOPT_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
// Forward declarations.
class AnalysisUsage;
class GStore;
class LegalizerInfo;
class MachineBasicBlock;
class MachineInstr;
class TargetLowering;
struct LegalityQuery;
class MachineRegisterInfo;
namespace GISelAddressing {
/// Helper struct to store a base, index and offset that forms an address
class BaseIndexOffset {
private:
  Register BaseReg;
  Register IndexReg;
  std::optional<int64_t> Offset;

public:
  BaseIndexOffset() = default;
  Register getBase() { return BaseReg; }
  Register getBase() const { return BaseReg; }
  Register getIndex() { return IndexReg; }
  Register getIndex() const { return IndexReg; }
  void setBase(Register NewBase) { BaseReg = NewBase; }
  void setIndex(Register NewIndex) { IndexReg = NewIndex; }
  void setOffset(std::optional<int64_t> NewOff) { Offset = NewOff; }
  bool hasValidOffset() const { return Offset.has_value(); }
  int64_t getOffset() const { return *Offset; }
};

/// Returns a BaseIndexOffset which describes the pointer in \p Ptr.
LLVM_ABI BaseIndexOffset getPointerInfo(Register Ptr, MachineRegisterInfo &MRI);

/// Compute whether or not a memory access at \p MI1 aliases with an access at
/// \p MI2 \returns true if either alias/no-alias is known. Sets \p IsAlias
/// accordingly.
LLVM_ABI bool aliasIsKnownForLoadStore(const MachineInstr &MI1,
                                       const MachineInstr &MI2, bool &IsAlias,
                                       MachineRegisterInfo &MRI);

/// Returns true if the instruction \p MI may alias \p Other.
/// This function uses multiple strategies to detect aliasing, whereas
/// aliasIsKnownForLoadStore just looks at the addresses of load/stores and is
/// tries to reason about base/index/offsets.
LLVM_ABI bool instMayAlias(const MachineInstr &MI, const MachineInstr &Other,
                           MachineRegisterInfo &MRI, AliasAnalysis *AA);
} // namespace GISelAddressing

class LLVM_ABI LoadStoreOptLegacy : public MachineFunctionPass {
public:
  static char ID;

  LoadStoreOptLegacy();

  StringRef getPassName() const override { return "LoadStoreOpt"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
};

class LoadStoreOptPass : public RequiredPassInfoMixin<LoadStoreOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // End namespace llvm.

#endif
