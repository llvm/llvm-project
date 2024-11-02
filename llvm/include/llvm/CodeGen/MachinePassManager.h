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

/// A CRTP mix-in that provides informational APIs needed for machine passes.
///
/// This provides some boilerplate for types that are machine passes. It
/// automatically mixes in \c PassInfoMixin.
template <typename DerivedT>
struct MachinePassInfoMixin : public PassInfoMixin<DerivedT> {
  // TODO: Add MachineFunctionProperties support.
};

namespace detail {
struct MachinePassConcept
    : PassConcept<MachineFunction, MachineFunctionAnalysisManager> {
  virtual MachineFunctionProperties getRequiredProperties() const = 0;
  virtual MachineFunctionProperties getSetProperties() const = 0;
  virtual MachineFunctionProperties getClearedProperties() const = 0;
};

template <typename PassT> struct MachinePassModel : MachinePassConcept {
  explicit MachinePassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  MachinePassModel(const MachinePassModel &Arg) : Pass(Arg.Pass) {}
  MachinePassModel(MachinePassModel &&Arg) : Pass(std::move(Arg.Pass)) {}

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
    return Pass.run(IR, AM);
  }

  void printPipeline(
      raw_ostream &OS,
      function_ref<StringRef(StringRef)> MapClassName2PassName) override {
    Pass.printPipeline(OS, MapClassName2PassName);
  }

  StringRef name() const override { return PassT::name(); }

  template <typename T>
  using has_required_t = decltype(std::declval<T &>().isRequired());
  template <typename T>
  static std::enable_if_t<is_detected<has_required_t, T>::value, bool>
  passIsRequiredImpl() {
    return T::isRequired();
  }
  template <typename T>
  static std::enable_if_t<!is_detected<has_required_t, T>::value, bool>
  passIsRequiredImpl() {
    return false;
  }
  bool isRequired() const override { return passIsRequiredImpl<PassT>(); }

  template <typename T>
  using has_get_required_properties_t =
      decltype(std::declval<T &>().getRequiredProperties());
  template <typename T>
  static std::enable_if_t<is_detected<has_get_required_properties_t, T>::value,
                          MachineFunctionProperties>
  getRequiredPropertiesImpl() {
    return PassT::getRequiredProperties();
  }
  template <typename T>
  static std::enable_if_t<!is_detected<has_get_required_properties_t, T>::value,
                          MachineFunctionProperties>
  getRequiredPropertiesImpl() {
    return MachineFunctionProperties();
  }
  MachineFunctionProperties getRequiredProperties() const override {
    return getRequiredPropertiesImpl<PassT>();
  }

  template <typename T>
  using has_get_set_properties_t =
      decltype(std::declval<T &>().getSetProperties());
  template <typename T>
  static std::enable_if_t<is_detected<has_get_set_properties_t, T>::value,
                          MachineFunctionProperties>
  getSetPropertiesImpl() {
    return PassT::getSetProperties();
  }
  template <typename T>
  static std::enable_if_t<!is_detected<has_get_set_properties_t, T>::value,
                          MachineFunctionProperties>
  getSetPropertiesImpl() {
    return MachineFunctionProperties();
  }
  MachineFunctionProperties getSetProperties() const override {
    return getSetPropertiesImpl<PassT>();
  }

  template <typename T>
  using has_get_cleared_properties_t =
      decltype(std::declval<T &>().getClearedProperties());
  template <typename T>
  static std::enable_if_t<is_detected<has_get_cleared_properties_t, T>::value,
                          MachineFunctionProperties>
  getClearedPropertiesImpl() {
    return PassT::getClearedProperties();
  }
  template <typename T>
  static std::enable_if_t<!is_detected<has_get_cleared_properties_t, T>::value,
                          MachineFunctionProperties>
  getClearedPropertiesImpl() {
    return MachineFunctionProperties();
  }
  MachineFunctionProperties getClearedProperties() const override {
    return getClearedPropertiesImpl<PassT>();
  }

  PassT Pass;
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

    ~Result() {
      // FAM is cleared in a moved from state where there is nothing to do.
      if (!FAM)
        return;

      // Clear out the analysis manager if we're being destroyed -- it means we
      // didn't even see an invalidate call when we got invalidated.
      FAM->clear();
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

class ModuleToMachineFunctionPassAdaptor
    : public PassInfoMixin<ModuleToMachineFunctionPassAdaptor> {
  using MachinePassConcept = detail::MachinePassConcept;

public:
  explicit ModuleToMachineFunctionPassAdaptor(
      std::unique_ptr<MachinePassConcept> Pass)
      : Pass(std::move(Pass)) {}

  /// Runs the function pass across every function in the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

  static bool isRequired() { return true; }

private:
  std::unique_ptr<MachinePassConcept> Pass;
};

template <typename MachineFunctionPassT>
ModuleToMachineFunctionPassAdaptor
createModuleToMachineFunctionPassAdaptor(MachineFunctionPassT &&Pass) {
  using PassModelT = detail::MachinePassModel<MachineFunctionPassT>;
  // Do not use make_unique, it causes too many template instantiations,
  // causing terrible compile times.
  return ModuleToMachineFunctionPassAdaptor(
      std::unique_ptr<detail::MachinePassConcept>(
          new PassModelT(std::forward<MachineFunctionPassT>(Pass))));
}

template <>
PreservedAnalyses
PassManager<MachineFunction>::run(MachineFunction &,
                                  AnalysisManager<MachineFunction> &);
extern template class PassManager<MachineFunction>;

/// Convenience typedef for a pass manager over functions.
using MachineFunctionPassManager = PassManager<MachineFunction>;

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEPASSMANAGER_H
