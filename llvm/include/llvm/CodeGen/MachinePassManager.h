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
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"

namespace llvm {
class Module;
class Function;
class MachineFunction;

extern template class AnalysisManager<MachineFunction>;
using MachineFunctionAnalysisManager = AnalysisManager<MachineFunction>;

/// An RAII based helper class to modify MachineFunctionProperties when running
/// pass. Define a MFPropsModifier in PassT::run to set
/// MachineFunctionProperties properly.
template <typename PassT> class MFPropsModifier {
public:
  MFPropsModifier(const PassT &P_, MachineFunction &MF_) : P(P_), MF(MF_) {
    auto &MFProps = MF.getProperties();
#ifndef NDEBUG
    if constexpr (has_get_required_properties_v<PassT>) {
      auto &MFProps = MF.getProperties();
      auto RequiredProperties = P.getRequiredProperties();
      if (!MFProps.verifyRequiredProperties(RequiredProperties)) {
        errs() << "MachineFunctionProperties required by " << PassT::name()
               << " pass are not met by function " << MF.getName() << ".\n"
               << "Required properties: ";
        RequiredProperties.print(errs());
        errs() << "\nCurrent properties: ";
        MFProps.print(errs());
        errs() << '\n';
        report_fatal_error("MachineFunctionProperties check failed");
      }
    }
#endif // NDEBUG
    if constexpr (has_get_cleared_properties_v<PassT>)
      MFProps.reset(P.getClearedProperties());
  }

  ~MFPropsModifier() {
    if constexpr (has_get_set_properties_v<PassT>) {
      auto &MFProps = MF.getProperties();
      MFProps.set(P.getSetProperties());
    }
  }

private:
  const PassT &P;
  MachineFunction &MF;

  template <typename T>
  using has_get_required_properties_t =
      decltype(std::declval<T &>().getRequiredProperties());

  template <typename T>
  using has_get_set_properties_t =
      decltype(std::declval<T &>().getSetProperties());

  template <typename T>
  using has_get_cleared_properties_t =
      decltype(std::declval<T &>().getClearedProperties());

  template <typename T>
  static constexpr bool has_get_required_properties_v =
      is_detected<has_get_required_properties_t, T>::value;

  template <typename T>
  static constexpr bool has_get_set_properties_v =
      is_detected<has_get_set_properties_t, T>::value;

  template <typename T>
  static constexpr bool has_get_cleared_properties_v =
      is_detected<has_get_cleared_properties_t, T>::value;
};

// Additional deduction guide to suppress warning.
template <typename PassT>
MFPropsModifier(PassT &P, MachineFunction &MF) -> MFPropsModifier<PassT>;

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
PreservedAnalyses
PassManager<MachineFunction>::run(MachineFunction &,
                                  AnalysisManager<MachineFunction> &);
extern template class PassManager<MachineFunction>;

/// Convenience typedef for a pass manager over functions.
using MachineFunctionPassManager = PassManager<MachineFunction>;

/// Returns the minimum set of Analyses that all machine function passes must
/// preserve.
PreservedAnalyses getMachineFunctionPassPreservedAnalyses();

/// For migrating to new pass manager
/// Provides a common interface to fetch analyses instead of doing it twice in
/// the *LegacyPass::runOnMachineFunction and NPM Pass::run.
///
/// NPM analyses must have the LegacyWrapper type to indicate which legacy
/// analysis to run. Legacy wrapper analyses must have `getResult()` method.
/// This can be added on a needs-to basis.
///
/// Outer analyses passes(Module or Function) can also be requested through
/// `getAnalysis` or `getCachedAnalysis`.
class MFAnalysisGetter {
private:
  Pass *LegacyPass = nullptr;
  MachineFunctionAnalysisManager *MFAM = nullptr;

  template <typename T>
  using type_of_run =
      typename function_traits<decltype(&T::run)>::template arg_t<0>;

  template <typename T>
  static constexpr bool IsFunctionAnalysis =
      std::is_same_v<Function &, type_of_run<T>>;

  template <typename T>
  static constexpr bool IsModuleAnalysis =
      std::is_same_v<Module &, type_of_run<T>>;

public:
  MFAnalysisGetter(Pass *LegacyPass) : LegacyPass(LegacyPass) {}
  MFAnalysisGetter(MachineFunctionAnalysisManager *MFAM) : MFAM(MFAM) {}

  /// Outer analyses requested from NPM will be cached results and can be null
  template <typename AnalysisT>
  typename AnalysisT::Result *getAnalysis(MachineFunction &MF) {
    if (MFAM) {
      // need a proxy to get the result for outer analyses
      // this can return null
      if constexpr (IsModuleAnalysis<AnalysisT>)
        return MFAM->getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
            .getCachedResult<AnalysisT>(*MF.getFunction().getParent());
      else if constexpr (IsFunctionAnalysis<AnalysisT>) {
        return &MFAM->getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
                    .getManager()
                    .getResult<AnalysisT>(MF.getFunction());
      }
      return &MFAM->getResult<AnalysisT>(MF);
    }
    return &LegacyPass->getAnalysis<typename AnalysisT::LegacyWrapper>()
                .getResult();
  }

  template <typename AnalysisT>
  typename AnalysisT::Result *getCachedAnalysis(MachineFunction &MF) {
    if (MFAM) {
      if constexpr (IsFunctionAnalysis<AnalysisT>) {
        return MFAM->getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
            .getManager()
            .getCachedResult<AnalysisT>(MF.getFunction());
      } else if constexpr (IsModuleAnalysis<AnalysisT>)
        return MFAM->getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
            .getCachedResult<AnalysisT>(*MF.getFunction().getParent());

      return &MFAM->getCachedResult<AnalysisT>(MF);
    }

    if (auto *P =
            LegacyPass->getAnalysisIfAvailable<AnalysisT::LegacyWrapper>())
      return &P->getResult();
    return nullptr;
  }

  /// This is not intended to be used to invoke getAnalysis()
  Pass *getLegacyPass() const { return LegacyPass; }
  MachineFunctionAnalysisManager *getMFAM() const { return MFAM; }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEPASSMANAGER_H
