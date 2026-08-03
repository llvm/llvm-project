//===-- PISAKernelByValArgsLowering.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers byval arguments. The byval semantics cannot work with the
// PISA calling convention, so the pass replaces byval with byref as follows:
//
// 1. The pass creates a new function with the same signature as the original
//    function, but with byref arguments instead of byval arguments. The new
//    byref arguments are pointers with constant address space.
// 2. The pass creates alloca instructions for each byval argument of the
//    original function and replaces the byval argument uses with the alloca.
// 3. The pass creates a memcpy intrinsic to copy the new byref argument to the
//    alloca.
// 4. The pass replaces old function uses with the new function and removes the
//    old function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAKERNELBYVALARGSLOWERING_H
#define LLVM_LIB_TARGET_PISA_PISAKERNELBYVALARGSLOWERING_H

#include <llvm/IR/PassManager.h>

namespace llvm {
namespace PISA {
class KernelByValArgsLoweringPass
    : public OptionalPassInfoMixin<KernelByValArgsLoweringPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAKERNELBYVALARGSLOWERING_H
