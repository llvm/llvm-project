//===- mlir-runner.cpp - MLIR CPU Execution Driver ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by  translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

// TODO: Consider removing this linking functionality from the SPIR-V CPU Runner
//       flow in favour of a more proper host/device split like other runners.
//       https://github.com/llvm/llvm-project/issues/115348
llvm::cl::opt<bool> LinkNestedModules(
    "link-nested-modules",
    llvm::cl::desc("Link two nested MLIR modules into a single LLVM IR module. "
                   "Useful if both the host and device code can be run on the "
                   "same CPU, as in SPIR-V CPU Runner tests."));

/// A utility function that builds llvm::Module from two nested MLIR modules.
///
/// module @main {
///   module @kernel {
///     // Some ops
///   }
///   // Some other ops
/// }
///
/// Each of these two modules is translated to LLVM IR module, then they are
/// linked together and returned.
static std::unique_ptr<llvm::Module>
convertMLIRModule(Operation *op, llvm::LLVMContext &context) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("op must be a 'builtin.module"), nullptr;

  std::unique_ptr<llvm::Module> kernelModule;
  if (LinkNestedModules) {
    // Verify that there is only one nested module.
    auto modules = module.getOps<ModuleOp>();
    if (!llvm::hasSingleElement(modules)) {
      module.emitError("The module must contain exactly one nested module");
      return nullptr;
    }

    // Translate nested module and erase it.
    ModuleOp nested = *modules.begin();
    kernelModule = translateModuleToLLVMIR(nested, context);
    nested.erase();
  }

  std::unique_ptr<llvm::Module> mainModule =
      translateModuleToLLVMIR(module, context);

  if (LinkNestedModules)
    llvm::Linker::linkModules(*mainModule, std::move(kernelModule));

  return mainModule;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.llvmModuleBuilder = convertMLIRModule;
  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
