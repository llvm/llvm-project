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

#ifndef MLIR_EXECUTIONENGINE_JITRUNNER_H
#define MLIR_EXECUTIONENGINE_JITRUNNER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
class Module;
class LLVMContext;

namespace orc {
class MangleAndInterner;
} // namespace orc
} // namespace llvm

namespace mlir {

class DialectRegistry;
class Operation;
struct LogicalResult;

struct JitRunnerConfig {
  /// MLIR transformer applied after parsing the input into MLIR IR and before
  /// passing the MLIR IR to the ExecutionEngine.
  llvm::function_ref<LogicalResult(mlir::Operation *)> mlirTransformer =
      nullptr;

  /// A custom function that is passed to ExecutionEngine. It processes MLIR and
  /// creates an LLVM IR module.
  llvm::function_ref<std::unique_ptr<llvm::Module>(Operation *,
                                                   llvm::LLVMContext &)>
      llvmModuleBuilder = nullptr;

  /// A callback to register symbols with ExecutionEngine at runtime.
  llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
      runtimesymbolMap = nullptr;
};

/// Entry point for all CPU runners. Expects the common argc/argv arguments for
/// standard C++ main functions. The supplied dialect registry is expected to
/// contain any registers that appear in the input IR, they will be loaded
/// on-demand by the parser.
int JitRunnerMain(int argc, char **argv, const DialectRegistry &registry,
                  JitRunnerConfig config = {});

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_JITRUNNER_H
