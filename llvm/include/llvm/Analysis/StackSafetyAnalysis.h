//===- StackSafetyAnalysis.h - Stack memory safety analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stack Safety Analysis detects allocas and arguments with safe access.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_STACKSAFETYANALYSIS_H
#define LLVM_ANALYSIS_STACKSAFETYANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AllocaInst;

/// Interface to access stack safety analysis results for single function.
class StackSafetyInfo {
public:
  struct InfoTy;

private:
  std::unique_ptr<InfoTy> Info;

public:
  StackSafetyInfo(InfoTy Info);
  StackSafetyInfo(StackSafetyInfo &&);
  StackSafetyInfo &operator=(StackSafetyInfo &&);
  ~StackSafetyInfo();

  const InfoTy &getInfo() const { return *Info; }

  // TODO: Add useful for client methods.
  void print(raw_ostream &O, const GlobalValue &F) const;
};

class StackSafetyGlobalInfo {
public:
  using GVToSSI = std::map<const GlobalValue *, StackSafetyInfo>;

private:
  GVToSSI SSGI;

public:
  StackSafetyGlobalInfo() = default;
  StackSafetyGlobalInfo(GVToSSI SSGI) : SSGI(std::move(SSGI)) {}

  bool setMetadata(Module &M) const;
  void print(raw_ostream &O) const;
  void dump() const;
};

/// StackSafetyInfo wrapper for the new pass manager.
class StackSafetyAnalysis : public AnalysisInfoMixin<StackSafetyAnalysis> {
  friend AnalysisInfoMixin<StackSafetyAnalysis>;
  static AnalysisKey Key;

public:
  using Result = StackSafetyInfo;
  StackSafetyInfo run(Function &F, FunctionAnalysisManager &AM);
};

/// Printer pass for the \c StackSafetyAnalysis results.
class StackSafetyPrinterPass : public PassInfoMixin<StackSafetyPrinterPass> {
  raw_ostream &OS;

public:
  explicit StackSafetyPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// StackSafetyInfo wrapper for the legacy pass manager
class StackSafetyInfoWrapperPass : public FunctionPass {
  Optional<StackSafetyInfo> SSI;
  const Function *F = nullptr;

public:
  static char ID;
  StackSafetyInfoWrapperPass();

  const StackSafetyInfo &getResult() const { return *SSI; }

  void print(raw_ostream &O, const Module *M) const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
};

/// This pass performs the global (interprocedural) stack safety analysis (new
/// pass manager).
class StackSafetyGlobalAnalysis
    : public AnalysisInfoMixin<StackSafetyGlobalAnalysis> {
  friend AnalysisInfoMixin<StackSafetyGlobalAnalysis>;
  static AnalysisKey Key;

public:
  using Result = StackSafetyGlobalInfo;
  Result run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for the \c StackSafetyGlobalAnalysis results.
class StackSafetyGlobalPrinterPass
    : public PassInfoMixin<StackSafetyGlobalPrinterPass> {
  raw_ostream &OS;

public:
  explicit StackSafetyGlobalPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class StackSafetyGlobalAnnotatorPass
    : public PassInfoMixin<StackSafetyGlobalAnnotatorPass> {

public:
  explicit StackSafetyGlobalAnnotatorPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// This pass performs the global (interprocedural) stack safety analysis
/// (legacy pass manager).
class StackSafetyGlobalInfoWrapperPass : public ModulePass {
  StackSafetyGlobalInfo SSGI;

public:
  static char ID;

  StackSafetyGlobalInfoWrapperPass();

  const StackSafetyGlobalInfo &getResult() const { return SSGI; }

  void print(raw_ostream &O, const Module *M) const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnModule(Module &M) override;
};

ModulePass *createStackSafetyGlobalInfoWrapperPass();

} // end namespace llvm

#endif // LLVM_ANALYSIS_STACKSAFETYANALYSIS_H
