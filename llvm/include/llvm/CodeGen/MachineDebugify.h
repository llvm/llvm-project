//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attaches synthetic debug info to the MachineFunction for a Function. To be
// used both by the legacy and the new pass manager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDEBUGIFY_H_
#define LLVM_CODEGEN_MACHINEDEBUGIFY_H_

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

LLVM_ABI bool applyDebugifyMetadataToMachineFunction(
    DIBuilder &DIB, Function &F,
    llvm::function_ref<MachineFunction *(Function &)> GetMF);

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINEDEBUGIFY_H_
