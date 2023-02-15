//===-- Passes.td - LLVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LLVMLegalizeForExport : Pass<"llvm-legalize-for-export"> {
  let summary = "Legalize LLVM dialect to be convertible to LLVM IR";
  let constructor = "::mlir::LLVM::createLegalizeForExportPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LLVMRequestCWrappers
    : Pass<"llvm-request-c-wrappers", "::mlir::func::FuncOp"> {
  let summary = "Request C wrapper emission for all functions";
  let description = [{
    Annotate every builtin function in the module with the LLVM dialect
    attribute that instructs the conversion to LLVM to emit the C wrapper for
    the function. This pass is expected to be applied immediately before the
    conversion of builtin functions to LLVM to avoid the attribute being
    dropped by other passes.
  }];
  let constructor = "::mlir::LLVM::createRequestCWrappersPass()";
}

def NVVMOptimizeForTarget : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
  let constructor = "::mlir::NVVM::createOptimizeForTargetPass()";
}

def DIScopeForLLVMFuncOp : Pass<"ensure-debug-info-scope-on-llvm-func", "LLVM::LLVMFuncOp"> {
  let summary = "Materialize LLVM debug info subprogram attribute on every LLVMFuncOp";
  let description = [{
    Having a debug info subprogram attribute on a function is required for
    emitting line tables from MLIR FileLocCol locations.

    This is not intended to be a proper replacement for frontends to emit
    complete debug informations, however it is a convenient way to get line
    tables for debugging purposes. This allow to step trough in a debugger
    line-by-line or get a backtrace with line numbers.
  }];

  let constructor = "mlir::LLVM::createDIScopeForLLVMFuncOpPass()";
}

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
