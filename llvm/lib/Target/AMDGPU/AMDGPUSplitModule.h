//===- AMDGPUSplitModule.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AMDGPUSPLITMODULE_H
#define LLVM_TARGET_AMDGPUSPLITMODULE_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include <memory>

namespace llvm {

class Module;
class AMDGPUTargetMachine;

/// Splits the module M into N linkable partitions. The function ModuleCallback
/// is called N times passing each individual partition as the MPart argument.
void splitAMDGPUModule(
    const AMDGPUTargetMachine &TM, Module &M, unsigned N,
    function_ref<void(std::unique_ptr<Module> MPart)> ModuleCallback);

} // end namespace llvm

#endif // LLVM_TARGET_AMDGPUSPLITMODULE_H
