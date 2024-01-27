//===- PassManager.h --- Pass management for CodeGen ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines the pass manager interface for codegen. The codegen
// pipeline consists of only machine function passes. There is no container
// relationship between IR module/function and machine function in terms of pass
// manager organization. So there is no need for adaptor classes (for example
// ModuleToMachineFunctionAdaptor). Since invalidation could only happen among
// machine function passes, there is no proxy classes to handle cross-IR-unit
// invalidation. IR analysis results are provided for machine function passes by
// their respective analysis managers such as ModuleAnalysisManager and
// FunctionAnalysisManager.
//
// TODO: Add MachineFunctionProperties support.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPASSMANAGER_H
#define LLVM_CODEGEN_MACHINEPASSMANAGER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Error.h"

#include <map>

namespace llvm {
class Module;
class Function;
class MachineFunction;

extern template class AnalysisManager<MachineFunction>;

/// A CRTP mix-in that provides informational APIs needed for machine passes.
///
/// This provides some boilerplate for types that are machine passes. It
/// automatically mixes in \c PassInfoMixin.
template <typename DerivedT>
struct MachinePassInfoMixin : public PassInfoMixin<DerivedT> {
  // TODO: Add MachineFunctionProperties support.
};

/// An AnalysisManager<MachineFunction> that also exposes IR analysis results.
class MachineFunctionAnalysisManager : public AnalysisManager<MachineFunction> {
public:
  using Base = AnalysisManager<MachineFunction>;

  MachineFunctionAnalysisManager() : FAM(nullptr), MAM(nullptr) {}
  MachineFunctionAnalysisManager(FunctionAnalysisManager &FAM,
                                 ModuleAnalysisManager &MAM)
      : FAM(&FAM), MAM(&MAM) {}
  MachineFunctionAnalysisManager(MachineFunctionAnalysisManager &&) = default;
  MachineFunctionAnalysisManager &
  operator=(MachineFunctionAnalysisManager &&) = default;

  /// Get the result of an analysis pass for a Function.
  ///
  /// Runs the analysis if a cached result is not available.
  template <typename PassT> typename PassT::Result &getResult(Function &F) {
    return FAM->getResult<PassT>(F);
  }

  /// Get the cached result of an analysis pass for a Function.
  ///
  /// This method never runs the analysis.
  ///
  /// \returns null if there is no cached result.
  template <typename PassT>
  typename PassT::Result *getCachedResult(Function &F) {
    return FAM->getCachedResult<PassT>(F);
  }

  /// Get the result of an analysis pass for a Module.
  ///
  /// Runs the analysis if a cached result is not available.
  template <typename PassT> typename PassT::Result &getResult(Module &M) {
    return MAM->getResult<PassT>(M);
  }

  /// Get the cached result of an analysis pass for a Module.
  ///
  /// This method never runs the analysis.
  ///
  /// \returns null if there is no cached result.
  template <typename PassT> typename PassT::Result *getCachedResult(Module &M) {
    return MAM->getCachedResult<PassT>(M);
  }

  /// Get the result of an analysis pass for a MachineFunction.
  ///
  /// Runs the analysis if a cached result is not available.
  using Base::getResult;

  /// Get the cached result of an analysis pass for a MachineFunction.
  ///
  /// This method never runs the analysis.
  ///
  /// returns null if there is no cached result.
  using Base::getCachedResult;

  // FIXME: Add LoopAnalysisManager or CGSCCAnalysisManager if needed.
  FunctionAnalysisManager *FAM;
  ModuleAnalysisManager *MAM;
};

extern template class PassManager<MachineFunction>;

/// MachineFunctionPassManager adds/removes below features to/from the base
/// PassManager template instantiation.
///
/// - Support passes that implement doInitialization/doFinalization. This is for
///   machine function passes to work on module level constructs. One such pass
///   is AsmPrinter.
///
/// - Support machine module pass which runs over the module (for example,
///   MachineOutliner). A machine module pass needs to define the method:
///
///   ```Error run(Module &, MachineFunctionAnalysisManager &)```
///
///   FIXME: machine module passes still need to define the usual machine
///          function pass interface, namely,
///          `PreservedAnalyses run(MachineFunction &,
///                                 MachineFunctionAnalysisManager &)`
///          But this interface wouldn't be executed. It is just a placeholder
///          to satisfy the pass manager type-erased inteface. This
///          special-casing of machine module pass is due to its limited use
///          cases and the unnecessary complexity it may bring to the machine
///          pass manager.
///
/// - The base class `run` method is replaced by an alternative `run` method.
///   See details below.
///
/// - Support codegening in the SCC order. Users include interprocedural
///   register allocation (IPRA).
class MachineFunctionPassManager
    : public PassManager<MachineFunction, MachineFunctionAnalysisManager> {
  using Base = PassManager<MachineFunction, MachineFunctionAnalysisManager>;

public:
  MachineFunctionPassManager(bool RequireCodeGenSCCOrder = false,
                             bool VerifyMachineFunction = false)
      : RequireCodeGenSCCOrder(RequireCodeGenSCCOrder),
        VerifyMachineFunction(VerifyMachineFunction) {}
  MachineFunctionPassManager(MachineFunctionPassManager &&) = default;
  MachineFunctionPassManager &
  operator=(MachineFunctionPassManager &&) = default;

  /// Run machine passes for a Module.
  ///
  /// The intended use is to start the codegen pipeline for a Module. The base
  /// class's `run` method is deliberately hidden by this due to the observation
  /// that we don't yet have the use cases of compositing two instances of
  /// machine pass managers, or compositing machine pass managers with other
  /// types of pass managers.
  Error run(Module &M, MachineFunctionAnalysisManager &MFAM);

  template <typename PassT> void addPass(PassT &&Pass) {
    Base::addPass(std::forward<PassT>(Pass));
    PassConceptT *P = Passes.back().get();
    addDoInitialization<PassT>(P);
    addDoFinalization<PassT>(P);

    // Add machine module pass.
    addRunOnModule<PassT>(P);
  }

private:
  template <typename PassT>
  using has_init_t = decltype(std::declval<PassT &>().doInitialization(
      std::declval<Module &>(),
      std::declval<MachineFunctionAnalysisManager &>()));

  template <typename PassT>
  std::enable_if_t<!is_detected<has_init_t, PassT>::value>
  addDoInitialization(PassConceptT *Pass) {}

  template <typename PassT>
  std::enable_if_t<is_detected<has_init_t, PassT>::value>
  addDoInitialization(PassConceptT *Pass) {
    using PassModelT =
        detail::PassModel<MachineFunction, PassT, PreservedAnalyses,
                          MachineFunctionAnalysisManager>;
    auto *P = static_cast<PassModelT *>(Pass);
    InitializationFuncs.emplace_back(
        [=](Module &M, MachineFunctionAnalysisManager &MFAM) {
          return P->Pass.doInitialization(M, MFAM);
        });
  }

  template <typename PassT>
  using has_fini_t = decltype(std::declval<PassT &>().doFinalization(
      std::declval<Module &>(),
      std::declval<MachineFunctionAnalysisManager &>()));

  template <typename PassT>
  std::enable_if_t<!is_detected<has_fini_t, PassT>::value>
  addDoFinalization(PassConceptT *Pass) {}

  template <typename PassT>
  std::enable_if_t<is_detected<has_fini_t, PassT>::value>
  addDoFinalization(PassConceptT *Pass) {
    using PassModelT =
        detail::PassModel<MachineFunction, PassT, PreservedAnalyses,
                          MachineFunctionAnalysisManager>;
    auto *P = static_cast<PassModelT *>(Pass);
    FinalizationFuncs.emplace_back(
        [=](Module &M, MachineFunctionAnalysisManager &MFAM) {
          return P->Pass.doFinalization(M, MFAM);
        });
  }

  template <typename PassT>
  using is_machine_module_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Module &>(),
      std::declval<MachineFunctionAnalysisManager &>()));

  template <typename PassT>
  using is_machine_function_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<MachineFunction &>(),
      std::declval<MachineFunctionAnalysisManager &>()));

  template <typename PassT>
  std::enable_if_t<!is_detected<is_machine_module_pass_t, PassT>::value>
  addRunOnModule(PassConceptT *Pass) {}

  template <typename PassT>
  std::enable_if_t<is_detected<is_machine_module_pass_t, PassT>::value>
  addRunOnModule(PassConceptT *Pass) {
    static_assert(is_detected<is_machine_function_pass_t, PassT>::value,
                  "machine module pass needs to define machine function pass "
                  "api. sorry.");

    using PassModelT =
        detail::PassModel<MachineFunction, PassT, PreservedAnalyses,
                          MachineFunctionAnalysisManager>;
    auto *P = static_cast<PassModelT *>(Pass);
    MachineModulePasses.emplace(
        Passes.size() - 1,
        [=](Module &M, MachineFunctionAnalysisManager &MFAM) {
          return P->Pass.run(M, MFAM);
        });
  }

  using FuncTy = Error(Module &, MachineFunctionAnalysisManager &);
  SmallVector<llvm::unique_function<FuncTy>, 4> InitializationFuncs;
  SmallVector<llvm::unique_function<FuncTy>, 4> FinalizationFuncs;

  using PassIndex = decltype(Passes)::size_type;
  std::map<PassIndex, llvm::unique_function<FuncTy>> MachineModulePasses;

  // Run codegen in the SCC order.
  bool RequireCodeGenSCCOrder;

  bool VerifyMachineFunction;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEPASSMANAGER_H
