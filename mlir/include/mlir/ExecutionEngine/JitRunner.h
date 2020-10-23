//===- JitRunner.h - MLIR CPU Execution Driver Library ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a library that provides a shared implementation for command line
// utilities that execute an MLIR file on the CPU by translating MLIR to LLVM
// IR before JIT-compiling and executing the latter.
//
// The translation can be customized by providing an MLIR to MLIR
// transformation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_JITRUNNER_H_
#define MLIR_SUPPORT_JITRUNNER_H_

#include "mlir/IR/Module.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"

namespace mlir {

using TranslationCallback = llvm::function_ref<std::unique_ptr<llvm::Module>(
    ModuleOp, llvm::LLVMContext &)>;

class ModuleOp;
struct LogicalResult;

// Entry point for all CPU runners. Expects the common argc/argv arguments for
// standard C++ main functions, `mlirTransformer` and `llvmModuleBuilder`.
/// `mlirTransformer` is applied after parsing the input into MLIR IR and before
/// passing the MLIR module to the ExecutionEngine.
/// `llvmModuleBuilder` is a custom function that is passed to ExecutionEngine.
/// It processes MLIR module and creates LLVM IR module.
int JitRunnerMain(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer,
    TranslationCallback llvmModuleBuilder = nullptr);

} // namespace mlir

#endif // MLIR_SUPPORT_JITRUNNER_H_
