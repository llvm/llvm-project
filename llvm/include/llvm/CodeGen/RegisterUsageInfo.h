//==- RegisterUsageInfo.h - Register Usage Informartion Storage --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass is required to take advantage of the interprocedural register
/// allocation infrastructure.
///
/// This pass is simple immutable pass which keeps RegMasks (calculated based on
/// actual register allocation) for functions in a module and provides simple
/// API to query this information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERUSAGEINFO_H
#define LLVM_CODEGEN_REGISTERUSAGEINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include <cstdint>
#include <vector>

namespace llvm {

class Function;
class TargetMachine;

class PhysicalRegisterUsageInfo {
public:
  /// Set TargetMachine which is used to print analysis.
  void setTargetMachine(const TargetMachine &TM);

  bool doInitialization(Module &M);

  bool doFinalization(Module &M);

  /// To store RegMask for given Function *.
  void storeUpdateRegUsageInfo(const Function &FP,
                               ArrayRef<uint32_t> RegMask);

  /// To query stored RegMask for given Function *, it will returns ane empty
  /// array if function is not known.
  ArrayRef<uint32_t> getRegUsageInfo(const Function &FP);

  void print(raw_ostream &OS, const Module *M = nullptr) const;

  bool invalidate(Module &M, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &Inv);

private:
  /// A Dense map from Function * to RegMask.
  /// In RegMask 0 means register used (clobbered) by function.
  /// and 1 means content of register will be preserved around function call.
  DenseMap<const Function *, std::vector<uint32_t>> RegMasks;

  const TargetMachine *TM = nullptr;
};

class PhysicalRegisterUsageInfoWrapperLegacy : public ImmutablePass {
  std::unique_ptr<PhysicalRegisterUsageInfo> PRUI;

public:
  static char ID;
  PhysicalRegisterUsageInfoWrapperLegacy() : ImmutablePass(ID) {
    initializePhysicalRegisterUsageInfoWrapperLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  PhysicalRegisterUsageInfo &getPRUI() { return *PRUI; }
  const PhysicalRegisterUsageInfo &getPRUI() const { return *PRUI; }

  bool doInitialization(Module &M) override {
    PRUI.reset(new PhysicalRegisterUsageInfo());
    return PRUI->doInitialization(M);
  }

  bool doFinalization(Module &M) override { return PRUI->doFinalization(M); }

  void print(raw_ostream &OS, const Module *M = nullptr) const override {
    PRUI->print(OS, M);
  }
};

class PhysicalRegisterUsageAnalysis
    : public AnalysisInfoMixin<PhysicalRegisterUsageAnalysis> {
  friend AnalysisInfoMixin<PhysicalRegisterUsageAnalysis>;
  static AnalysisKey Key;

public:
  using Result = PhysicalRegisterUsageInfo;

  PhysicalRegisterUsageInfo run(Module &M, ModuleAnalysisManager &);
};

class PhysicalRegisterUsageInfoPrinterPass
    : public PassInfoMixin<PhysicalRegisterUsageInfoPrinterPass> {
  raw_ostream &OS;

public:
  explicit PhysicalRegisterUsageInfoPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_REGISTERUSAGEINFO_H
