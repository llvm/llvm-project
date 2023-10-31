//===- FunctionCallUtils.h - Utilities for C function calls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions to call common simple C functions in
// LLVMIR (e.g. among others to support printing and debugging).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_
#define MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <optional>

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Operation;
class Type;
class ValueRange;

namespace LLVM {
class LLVMFuncOp;

/// Helper functions to lookup or create the declaration for commonly used
/// external C function calls. The list of functions provided here must be
/// implemented separately (e.g. as part of a support runtime library or as part
/// of the libc).
LLVM::LLVMFuncOp lookupOrCreatePrintI64Fn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintU64Fn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintF16Fn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintBF16Fn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintF32Fn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintF64Fn(ModuleOp moduleOp);
/// Declares a function to print a C-string.
/// If a custom runtime function is defined via `runtimeFunctionName`, it must
/// have the signature void(char const*). The default function is `printString`.
LLVM::LLVMFuncOp
lookupOrCreatePrintStringFn(ModuleOp moduleOp, bool opaquePointers,
                            std::optional<StringRef> runtimeFunctionName = {});
LLVM::LLVMFuncOp lookupOrCreatePrintOpenFn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintCloseFn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintCommaFn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintNewlineFn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateMallocFn(ModuleOp moduleOp, Type indexType,
                                        bool opaquePointers);
LLVM::LLVMFuncOp lookupOrCreateAlignedAllocFn(ModuleOp moduleOp, Type indexType,
                                              bool opaquePointers = true);
LLVM::LLVMFuncOp lookupOrCreateFreeFn(ModuleOp moduleOp,
                                      bool opaquePointers = true);
LLVM::LLVMFuncOp lookupOrCreateGenericAllocFn(ModuleOp moduleOp, Type indexType,
                                              bool opaquePointers);
LLVM::LLVMFuncOp lookupOrCreateGenericAlignedAllocFn(ModuleOp moduleOp,
                                                     Type indexType,
                                                     bool opaquePointers);
LLVM::LLVMFuncOp lookupOrCreateGenericFreeFn(ModuleOp moduleOp,
                                             bool opaquePointers);
LLVM::LLVMFuncOp lookupOrCreateMemRefCopyFn(ModuleOp moduleOp, Type indexType,
                                            Type unrankedDescriptorType);

/// Create a FuncOp with signature `resultType`(`paramTypes`)` and name `name`.
LLVM::LLVMFuncOp lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                                  ArrayRef<Type> paramTypes = {},
                                  Type resultType = {}, bool isVarArg = false);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_
