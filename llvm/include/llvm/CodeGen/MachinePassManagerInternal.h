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
struct PassConcept<MachineFunction, MachineFunctionAnalysisManager>
    : public PassConceptBase<MachineFunction, MachineFunctionAnalysisManager> {
  /// MachineFunction Properties.
  PassConcept(MachineFunctionProperties RequiredProperties,
              MachineFunctionProperties SetProperties,
              MachineFunctionProperties ClearedProperties)
      : RequiredProperties(RequiredProperties), SetProperties(SetProperties),
        ClearedProperties(ClearedProperties) {}

  MachineFunctionProperties RequiredProperties;
  MachineFunctionProperties SetProperties;
  MachineFunctionProperties ClearedProperties;
};

template <typename IRUnitT, typename PassT, typename PreservedAnalysesT,
          typename AnalysisManagerT, typename... ExtraArgTs>
template <typename MachineFunctionT, typename>
PassModel<IRUnitT, PassT, PreservedAnalysesT, AnalysisManagerT, ExtraArgTs...>::
    PassModel(PassT Pass, MachineFunctionProperties RequiredProperties,
              MachineFunctionProperties SetProperties,
              MachineFunctionProperties ClearedProperties)
    : PassConcept<MachineFunction, MachineFunctionAnalysisManager>(
          RequiredProperties, SetProperties, ClearedProperties),
      Pass(std::move(Pass)) {}

template <typename PassT>
using MachinePassModel = PassModel<MachineFunction, PassT, PreservedAnalyses,
                                   MachineFunctionAnalysisManager>;

} // namespace detail

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINEPASSMANAGERINTERNAL_H
