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

def LLVMAddComdats : Pass<"llvm-add-comdats", "::mlir::ModuleOp"> {
  let summary = "Add comdats to linkonce and linkonce_odr functions";
  let description = [{
    Add an any COMDAT to every linkonce and linkonce_odr function.
    This is necessary on Windows to link these functions as the system
    linker won't link weak symbols without a COMDAT. It also provides better
    behavior than standard weak symbols on ELF-based platforms.
    This pass will still add COMDATs on platforms that do not support them,
    for example macOS, so should only be run when the target platform supports
    COMDATs.
  }];
}

def LLVMLegalizeForExportPass : Pass<"llvm-legalize-for-export"> {
  let summary = "Legalize LLVM dialect to be convertible to LLVM IR";
  let description = [{
    Creates a pass that legalizes the LLVM dialect operations so that they can
    be translated to LLVM IR.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LLVMRequestCWrappersPass
    : Pass<"llvm-request-c-wrappers", "::mlir::func::FuncOp"> {
  let summary = "Request C wrapper emission for all functions";
  let description = [{
    Annotate every builtin function in the module with the LLVM dialect
    attribute that instructs the conversion to LLVM to emit the C wrapper for
    the function. This pass is expected to be applied immediately before the
    conversion of builtin functions to LLVM to avoid the attribute being
    dropped by other passes.
  }];
}

def NVVMOptimizeForTargetPass : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
}

def DIScopeForLLVMFuncOpPass : Pass<"ensure-debug-info-scope-on-llvm-func", "::mlir::ModuleOp"> {
  let summary = "Materialize LLVM debug info subprogram attribute on every LLVMFuncOp";
  let description = [{
    Having a debug info subprogram attribute on a function is required for
    emitting line tables from MLIR FileLocCol locations.

    This is not intended to be a proper replacement for frontends to emit
    complete debug informations, however it is a convenient way to get line
    tables for debugging purposes. This allow to step trough in a debugger
    line-by-line or get a backtrace with line numbers.
  }];

  let options = [
    Option<"emissionKind", "emission-kind", "mlir::LLVM::DIEmissionKind",
    /*default=*/"mlir::LLVM::DIEmissionKind::LineTablesOnly", "Emission kind to generate debug info.",
    [{::llvm::cl::values(
	     clEnumValN(::mlir::LLVM::DIEmissionKind::None, "None", "None"),
	     clEnumValN(::mlir::LLVM::DIEmissionKind::Full, "Full", "Full"),
	     clEnumValN(::mlir::LLVM::DIEmissionKind::LineTablesOnly, "LineTablesOnly", "LineTablesOnly (default)"),
	     clEnumValN(::mlir::LLVM::DIEmissionKind::DebugDirectivesOnly, "DebugDirectivesOnly", "DebugDirectivesOnly")
	   )}]>,
  ];
}

def PrepareForOMPOffloadPrivatizationPass : Pass<"omp-offload-privatization-prepare", "::mlir::LLVM::LLVMFuncOp"> {
    let summary = "Prepare OpenMP maps for privatization for deferred target tasks";
    let description = [{
      When generating LLVMIR for privatized variables in an OpenMP offloading directive (eg. omp::TargetOp)
      that creates a deferred target task (when the nowait clause is used), we need to copy the privatized
      variable out of the stack of the generating task and into the heap so that the deferred target task
      can still access it. However, if such a privatized variable is also mapped, typically the case for
      allocatables, then the corresponding `omp::MapInfoOp` needs to be fixed up to map the new heap-allocated
      variable and not the original variable.
    }];
  let dependentDialects = ["LLVM::LLVMDialect", "mlir::omp::OpenMPDialect"];
}
#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
