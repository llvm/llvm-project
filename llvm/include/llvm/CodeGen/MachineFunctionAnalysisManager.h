//===- llvm/CodeGen/MachineFunctionAnalysisManager.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typedef for MachineFunctionAnalysisManager as an explicit instantiation of
// AnalysisManager<MachineFunction>.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_MACHINEFUNCTIONANALYSISMANAGER
#define LLVM_CODEGEN_MACHINEFUNCTIONANALYSISMANAGER

#include "llvm/IR/PassManager.h"

namespace llvm {

class MachineFunction;

extern template class AnalysisManager<MachineFunction>;
using MachineFunctionAnalysisManager = AnalysisManager<MachineFunction>;

} // namespace llvm

#endif
