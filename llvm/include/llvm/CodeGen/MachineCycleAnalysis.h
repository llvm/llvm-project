//===- MachineCycleAnalysis.h - Cycle Info for Machine IR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineCycleInfo class, which is a thin wrapper over
// the Machine IR instance of GenericCycleInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECYCLEANALYSIS_H
#define LLVM_CODEGEN_MACHINECYCLEANALYSIS_H

#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineSSAContext.h"

namespace llvm {

extern template class GenericCycleInfo<MachineSSAContext>;
extern template class GenericCycle<MachineSSAContext>;

using MachineCycleInfo = GenericCycleInfo<MachineSSAContext>;
using MachineCycle = MachineCycleInfo::CycleT;

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINECYCLEANALYSIS_H
