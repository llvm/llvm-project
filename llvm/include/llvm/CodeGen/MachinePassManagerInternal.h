//===- MachinePassManagerInternal.h --------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides internal APIs and implementation details used by the
/// pass management interfaces exposed in MachinePassManager.h. Most of them are
/// copied from PassManagerInternal.h.
/// See also PassManagerInternal.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPASSMANAGERINTERNAL_H
#define LLVM_CODEGEN_MACHINEPASSMANAGERINTERNAL_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class MachineFunctionAnalysisManager;

using MachinePassConcept =
    detail::PassConcept<MachineFunction, MachineFunctionAnalysisManager>;

namespace detail {

/// Template for the abstract base class used to dispatch
/// polymorphically over pass objects. See also \c PassConcept.
template <>
struct PassConcept<MachineFunction, MachineFunctionAnalysisManager> {
  /// Generic PassConcept interface.
  virtual ~PassConcept() = default;
  PassConcept(MachineFunctionProperties RequiredProperties,
              MachineFunctionProperties SetProperties,
              MachineFunctionProperties ClearedProperties)
      : RequiredProperties(RequiredProperties), SetProperties(SetProperties),
        ClearedProperties(ClearedProperties) {}
  virtual PreservedAnalyses run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) = 0;

  virtual void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName) = 0;
  virtual StringRef name() const = 0;
  virtual bool isRequired() const = 0;

  /// MachineFunction Properties.
  MachineFunctionProperties RequiredProperties;
  MachineFunctionProperties SetProperties;
  MachineFunctionProperties ClearedProperties;
};

template <typename PassT>
struct PassModel<MachineFunction, PassT, PreservedAnalyses,
                 MachineFunctionAnalysisManager> : public MachinePassConcept {
  // Generic interface
  explicit PassModel(PassT Pass)
      : MachinePassConcept(PassT::getRequiredProperties(),
                           PassT::getSetProperties(),
                           PassT::getClearedProperties()),
        Pass(std::move(Pass)) {}

  friend void swap(PassModel &LHS, PassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }

  PassModel &operator=(PassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) override {
    return Pass.run(MF, MFAM);
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

  PassT Pass;
};

template <typename PassT>
using MachinePassModel = PassModel<MachineFunction, PassT, PreservedAnalyses,
                                   MachineFunctionAnalysisManager>;

} // namespace detail

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINEPASSMANAGERINTERNAL_H
