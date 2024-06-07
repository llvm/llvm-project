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
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPASSMANAGER_H
#define LLVM_CODEGEN_MACHINEPASSMANAGER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Support/Error.h"

namespace llvm {
class Module;
class Function;
class MachineFunction;

extern template class AnalysisManager<MachineFunction>;
using MachineFunctionAnalysisManager = AnalysisManager<MachineFunction>;

namespace detail {

template <typename PassT>
struct MachinePassModel
    : PassModel<MachineFunction, PassT, MachineFunctionAnalysisManager> {
  explicit MachinePassModel(PassT &&Pass)
      : PassModel<MachineFunction, PassT, MachineFunctionAnalysisManager>(
            std::move(Pass)) {}

  friend void swap(MachinePassModel &LHS, MachinePassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }

  MachinePassModel &operator=(MachinePassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  MachinePassModel &operator=(const MachinePassModel &) = delete;
  PreservedAnalyses run(MachineFunction &IR,
                        MachineFunctionAnalysisManager &AM) override {
#ifndef NDEBUG
    if constexpr (is_detected<has_get_required_properties_t, PassT>::value) {
      auto &MFProps = IR.getProperties();
      auto RequiredProperties = this->Pass.getRequiredProperties();
      if (!MFProps.verifyRequiredProperties(RequiredProperties)) {
        errs() << "MachineFunctionProperties required by " << PassT::name()
               << " pass are not met by function " << IR.getName() << ".\n"
               << "Required properties: ";
        RequiredProperties.print(errs());
        errs() << "\nCurrent properties: ";
        MFProps.print(errs());
        errs() << '\n';
        report_fatal_error("MachineFunctionProperties check failed");
      }
    }
#endif

    auto PA = this->Pass.run(IR, AM);

    if constexpr (is_detected<has_get_set_properties_t, PassT>::value)
      IR.getProperties().set(this->Pass.getSetProperties());
    if constexpr (is_detected<has_get_cleared_properties_t, PassT>::value)
      IR.getProperties().reset(this->Pass.getClearedProperties());
    return PA;
  }

private:
  template <typename T>
  using has_get_required_properties_t =
      decltype(std::declval<T &>().getRequiredProperties());

  template <typename T>
  using has_get_set_properties_t =
      decltype(std::declval<T &>().getSetProperties());

  template <typename T>
  using has_get_cleared_properties_t =
      decltype(std::declval<T &>().getClearedProperties());
};
} // namespace detail

using MachineFunctionAnalysisManagerModuleProxy =
    InnerAnalysisManagerProxy<MachineFunctionAnalysisManager, Module>;

template <>
bool MachineFunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &Inv);
extern template class InnerAnalysisManagerProxy<MachineFunctionAnalysisManager,
                                                Module>;
using MachineFunctionAnalysisManagerFunctionProxy =
    InnerAnalysisManagerProxy<MachineFunctionAnalysisManager, Function>;

template <>
bool MachineFunctionAnalysisManagerFunctionProxy::Result::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &Inv);
extern template class InnerAnalysisManagerProxy<MachineFunctionAnalysisManager,
                                                Function>;

extern template class OuterAnalysisManagerProxy<ModuleAnalysisManager,
                                                MachineFunction>;
/// Provide the \c ModuleAnalysisManager to \c Function proxy.
using ModuleAnalysisManagerMachineFunctionProxy =
    OuterAnalysisManagerProxy<ModuleAnalysisManager, MachineFunction>;

class FunctionAnalysisManagerMachineFunctionProxy
    : public AnalysisInfoMixin<FunctionAnalysisManagerMachineFunctionProxy> {
public:
  class Result {
  public:
    explicit Result(FunctionAnalysisManager &FAM) : FAM(&FAM) {}

    Result(Result &&Arg) : FAM(std::move(Arg.FAM)) {
      // We have to null out the analysis manager in the moved-from state
      // because we are taking ownership of the responsibilty to clear the
      // analysis state.
      Arg.FAM = nullptr;
    }

    Result &operator=(Result &&RHS) {
      FAM = RHS.FAM;
      // We have to null out the analysis manager in the moved-from state
      // because we are taking ownership of the responsibilty to clear the
      // analysis state.
      RHS.FAM = nullptr;
      return *this;
    }

    /// Accessor for the analysis manager.
    FunctionAnalysisManager &getManager() { return *FAM; }

    /// Handler for invalidation of the outer IR unit, \c IRUnitT.
    ///
    /// If the proxy analysis itself is not preserved, we assume that the set of
    /// inner IR objects contained in IRUnit may have changed.  In this case,
    /// we have to call \c clear() on the inner analysis manager, as it may now
    /// have stale pointers to its inner IR objects.
    ///
    /// Regardless of whether the proxy analysis is marked as preserved, all of
    /// the analyses in the inner analysis manager are potentially invalidated
    /// based on the set of preserved analyses.
    bool invalidate(MachineFunction &IR, const PreservedAnalyses &PA,
                    MachineFunctionAnalysisManager::Invalidator &Inv);

  private:
    FunctionAnalysisManager *FAM;
  };

  explicit FunctionAnalysisManagerMachineFunctionProxy(
      FunctionAnalysisManager &FAM)
      : FAM(&FAM) {}

  /// Run the analysis pass and create our proxy result object.
  ///
  /// This doesn't do any interesting work; it is primarily used to insert our
  /// proxy result object into the outer analysis cache so that we can proxy
  /// invalidation to the inner analysis manager.
  Result run(MachineFunction &, MachineFunctionAnalysisManager &) {
    return Result(*FAM);
  }

  static AnalysisKey Key;

private:
  FunctionAnalysisManager *FAM;
};

class FunctionToMachineFunctionPassAdaptor
    : public PassInfoMixin<FunctionToMachineFunctionPassAdaptor> {
public:
  using PassConceptT =
      detail::PassConcept<MachineFunction, MachineFunctionAnalysisManager>;

  explicit FunctionToMachineFunctionPassAdaptor(
      std::unique_ptr<PassConceptT> Pass)
      : Pass(std::move(Pass)) {}

  /// Runs the function pass across every function in the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

  static bool isRequired() { return true; }

private:
  std::unique_ptr<PassConceptT> Pass;
};

template <typename MachineFunctionPassT>
FunctionToMachineFunctionPassAdaptor
createFunctionToMachineFunctionPassAdaptor(MachineFunctionPassT &&Pass) {
  using PassModelT = detail::PassModel<MachineFunction, MachineFunctionPassT,
                                       MachineFunctionAnalysisManager>;
  // Do not use make_unique, it causes too many template instantiations,
  // causing terrible compile times.
  return FunctionToMachineFunctionPassAdaptor(
      std::unique_ptr<FunctionToMachineFunctionPassAdaptor::PassConceptT>(
          new PassModelT(std::forward<MachineFunctionPassT>(Pass))));
}

template <>
template <typename PassT>
void PassManager<MachineFunction>::addPass(PassT &&Pass) {
  using MachinePassModelT = detail::MachinePassModel<PassT>;
  // Do not use make_unique or emplace_back, they cause too many template
  // instantiations, causing terrible compile times.
  if constexpr (std::is_same_v<PassT, PassManager<MachineFunction>>) {
    for (auto &P : Pass.Passes)
      Passes.push_back(std::move(P));
  } else {
    Passes.push_back(std::unique_ptr<MachinePassModelT>(
        new MachinePassModelT(std::forward<PassT>(Pass))));
  }
}

template <>
PreservedAnalyses
PassManager<MachineFunction>::run(MachineFunction &,
                                  AnalysisManager<MachineFunction> &);
extern template class PassManager<MachineFunction>;

/// Convenience typedef for a pass manager over functions.
using MachineFunctionPassManager = PassManager<MachineFunction>;

/// Returns the minimum set of Analyses that all machine function passes must
/// preserve.
PreservedAnalyses getMachineFunctionPassPreservedAnalyses();

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEPASSMANAGER_H
