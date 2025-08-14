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

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Operation;
class Type;
class ValueRange;
class SymbolTableCollection;

namespace LLVM {
class LLVMFuncOp;

/// Helper functions to look up or create the declaration for commonly used
/// external C function calls. The list of functions provided here must be
/// implemented separately (e.g. as part of a support runtime library or as part
/// of the libc).
/// Failure if an unexpected version of function is found.
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintI64Fn(OpBuilder &b, Operation *moduleOp,
                         SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintU64Fn(OpBuilder &b, Operation *moduleOp,
                         SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintF16Fn(OpBuilder &b, Operation *moduleOp,
                         SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintBF16Fn(OpBuilder &b, Operation *moduleOp,
                          SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintF32Fn(OpBuilder &b, Operation *moduleOp,
                         SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintF64Fn(OpBuilder &b, Operation *moduleOp,
                         SymbolTableCollection *symbolTables = nullptr);
/// Declares a function to print a C-string.
/// If a custom runtime function is defined via `runtimeFunctionName`, it must
/// have the signature void(char const*). The default function is `printString`.
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintStringFn(OpBuilder &b, Operation *moduleOp,
                            std::optional<StringRef> runtimeFunctionName = {},
                            SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintOpenFn(OpBuilder &b, Operation *moduleOp,
                          SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintCloseFn(OpBuilder &b, Operation *moduleOp,
                           SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintCommaFn(OpBuilder &b, Operation *moduleOp,
                           SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreatePrintNewlineFn(OpBuilder &b, Operation *moduleOp,
                             SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateMallocFn(OpBuilder &b, Operation *moduleOp, Type indexType,
                       SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateAlignedAllocFn(OpBuilder &b, Operation *moduleOp, Type indexType,
                             SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateFreeFn(OpBuilder &b, Operation *moduleOp,
                     SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateGenericAllocFn(OpBuilder &b, Operation *moduleOp, Type indexType,
                             SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp> lookupOrCreateGenericAlignedAllocFn(
    OpBuilder &b, Operation *moduleOp, Type indexType,
    SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateGenericFreeFn(OpBuilder &b, Operation *moduleOp,
                            SymbolTableCollection *symbolTables = nullptr);
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateMemRefCopyFn(OpBuilder &b, Operation *moduleOp, Type indexType,
                           Type unrankedDescriptorType,
                           SymbolTableCollection *symbolTables = nullptr);

/// Create a FuncOp with signature `resultType`(`paramTypes`)` and name `name`.
/// Return a failure if the FuncOp found has unexpected signature.
FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateFn(OpBuilder &b, Operation *moduleOp, StringRef name,
                 ArrayRef<Type> paramTypes = {}, Type resultType = {},
                 bool isVarArg = false, bool isReserved = false,
                 SymbolTableCollection *symbolTables = nullptr);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_
