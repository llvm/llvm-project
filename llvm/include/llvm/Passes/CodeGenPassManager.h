//===- CodeGenPassManager.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// PassManager wrappers for building CodeGen pipeline.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_CODEGENPASSMANAGER_H
#define LLVM_PASSES_CODEGENPASSMANAGER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include <list>
#include <variant>

namespace llvm {

class LLVM_ABI CodeGenMachineFunctionPassManager
    : protected MachineFunctionPassManager {
  friend class CodeGenModulePassManager;
  friend class CodeGenFunctionPassManager;

public:
  template <typename PassT> void addPass(PassT &&Pass) {
    static_assert(!std::is_same_v<PassT, MachineFunctionPassManager>,
                  "Use CodeGenMachineFunctionPassManager instead!");
    MachineFunctionPassManager::addPass(std::forward<PassT>(Pass));
    PassList.push_back(std::move(Passes.back()));
    Passes.pop_back();
  }

  void addPass(CodeGenMachineFunctionPassManager &&CGPM) {
    for (auto &P : CGPM.PassList)
      PassList.push_back(std::move(P));
    CGPM.PassList.clear();
  }

  bool isEmpty() const { return PassList.empty(); }

private:
  MachineFunctionPassManager getMachineFunctionPassManager() {
    CodeGenMachineFunctionPassManager MFPM;
    for (auto &P : PassList)
      MFPM.Passes.push_back(std::move(P));
    PassList.clear();
    return MFPM;
  }

  void eraseIf(function_ref<bool(StringRef)> Pred) {
    for (auto I = PassList.begin(), E = PassList.end(); I != E;) {
      if (Pred((*I)->name()))
        I = PassList.erase(I);
      else
        ++I;
    }
  }

  void handleInsert(
      function_ref<void(StringRef, CodeGenMachineFunctionPassManager &)> F) {
    for (auto I = PassList.begin(), E = PassList.end(); I != E; ++I) {
      CodeGenMachineFunctionPassManager CGFPM;
      F((*I)->name(), CGFPM);
      if (CGFPM.isEmpty())
        continue;
      for (auto J = CGFPM.PassList.rbegin(), End = CGFPM.PassList.rend();
           J != End; ++J)
        PassList.insert(std::next(I), std::move(*J));
    }
  }

private:
  std::list<PassConceptT::unique_ptr> PassList;
};

class LLVM_ABI CodeGenLoopPassManager : public LoopPassManager {
  friend class CodeGenFunctionPassManager;

public:
  template <typename PassT> void addPass(PassT &&Pass) {
    static_assert(!std::is_same_v<PassT, LoopPassManager>,
                  "Use CodeGenLoopPassManager instead!");
    LoopPassManager::addPass(std::forward<PassT>(Pass));
    bool IsLoopNest = IsLoopNestPass.back();
    IsLoopNestPass.pop_back();
    if (IsLoopNest) {
      PassList.push_back(std::move(LoopNestPasses.back()));
      LoopNestPasses.pop_back();
    } else {
      PassList.push_back(std::move(LoopPasses.back()));
      LoopPasses.pop_back();
    }
  }

  void addPass(CodeGenLoopPassManager &&CGPM) {
    for (auto &P : CGPM.PassList)
      PassList.push_back(std::move(P));
    CGPM.PassList.clear();
  }

  bool isEmpty() const { return PassList.empty(); }

private:
  LoopPassManager getLoopPassManager();

  void eraseIf(function_ref<bool(StringRef)> Pred);

  bool isSimilarWith(const CodeGenLoopPassManager &CGLPM) const {
    return UseMemorySSA == CGLPM.UseMemorySSA;
  }

private:
  bool UseMemorySSA = false;
  std::list<std::variant<LoopPassConceptT::unique_ptr,
                         LoopNestPassConceptT::unique_ptr>>
      PassList;
};

class LLVM_ABI CodeGenFunctionPassManager : protected FunctionPassManager {
  friend class CodeGenModulePassManager;

public:
  template <typename PassT> void addPass(PassT &&Pass) {
    static_assert(!std::is_same_v<PassT, FunctionPassManager>,
                  "Use CodeGenFunctionPassManager instead!");
    static_assert(!std::is_same_v<PassT, FunctionToMachineFunctionPassAdaptor>,
                  "Use addCodeGenMachineFunctionPassManager instead!");
    static_assert(!std::is_same_v<PassT, FunctionToLoopPassAdaptor>,
                  "Use addCodeGenLoopPassManager instead!");
    FunctionPassManager::addPass(std::forward<PassT>(Pass));
    PassList.push_back(std::move(Passes.back()));
    Passes.pop_back();
  }

  void addPass(CodeGenFunctionPassManager &&CGPM) {
    for (auto &P : CGPM.PassList)
      PassList.push_back(std::move(P));
    CGPM.PassList.clear();
  }

  void addCodeGenLoopPassManager(CodeGenLoopPassManager &&CGLPM,
                                 bool UseMemorySSA = false) {
    CGLPM.UseMemorySSA = UseMemorySSA;
    PassList.push_back(std::move(CGLPM));
  }

  void addCodeGenMachineFunctionPassManager(
      CodeGenMachineFunctionPassManager &&CGMFPM) {
    PassList.push_back(std::move(CGMFPM));
  }

  bool isEmpty() const { return PassList.empty(); }

private:
  FunctionPassManager getFunctionPassManager();

  void eraseIf(function_ref<bool(StringRef)>);

  void combineSimilarPassManagers();

  bool isSimilarWith(const CodeGenFunctionPassManager &CGFPM) const {
    return EagerlyInvalidate == CGFPM.EagerlyInvalidate &&
           RequireCGSCCOrder == CGFPM.RequireCGSCCOrder;
  }

private:
  bool EagerlyInvalidate = false;
  bool RequireCGSCCOrder = false;
  std::list<std::variant<PassConceptT::unique_ptr, CodeGenLoopPassManager,
                         CodeGenMachineFunctionPassManager>>
      PassList;
};

class LLVM_ABI CodeGenModulePassManager : public ModulePassManager {
  template <typename DerivedT, typename TargetMachineT>
  friend class CodeGenPassBuilder;

public:
  template <typename PassT> void addPass(PassT &&Pass) {
    static_assert(!std::is_same_v<PassT, ModulePassManager>,
                  "Use CodeGenModulePassManager instead!");
    static_assert(!std::is_same_v<PassT, ModuleToFunctionPassAdaptor>,
                  "Use addCodeGenFunctionPassManager instead!");
    static_assert(!std::is_same_v<PassT, ModuleToPostOrderCGSCCPassAdaptor>,
                  "Use addCodeGenFunctionPassManagerInCGSCCOrder instead!");
    ModulePassManager::addPass(std::forward<PassT>(Pass));
    PassList.push_back(std::move(Passes.back()));
    Passes.pop_back();
  }

  void addPass(CodeGenModulePassManager &&CGPM) {
    for (auto &P : CGPM.PassList)
      PassList.push_back(std::move(P));
    CGPM.PassList.clear();
  }

  void addCodeGenFunctionPassManager(CodeGenFunctionPassManager &&CGFPM,
                                     bool EagerlyInvalidate = false) {
    CGFPM.EagerlyInvalidate = EagerlyInvalidate;
    PassList.push_back(std::move(CGFPM));
  }

  void addCodeGenFunctionPassManagerInCGSCCOrder(
      CodeGenFunctionPassManager &&CGFPM) {
    CGFPM.RequireCGSCCOrder = true;
    PassList.push_back(std::move(CGFPM));
  }

  bool isEmpty() const { return PassList.empty(); }

private:
  ModulePassManager getModulePassManager();

  void eraseIf(function_ref<bool(StringRef)> Pred);

  /**
   * @brief Merge pass managers which have the same type and options.
   */
  void combineSimilarPassManagers();

  void handleInsert(
      function_ref<void(StringRef, CodeGenMachineFunctionPassManager &)> F);

private:
  std::list<std::variant<PassConceptT::unique_ptr, CodeGenFunctionPassManager>>
      PassList;
};

} // namespace llvm

#endif // LLVM_PASSES_CODEGENPASSMANAGER_H
