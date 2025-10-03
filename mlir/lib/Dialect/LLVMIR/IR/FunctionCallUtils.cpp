//===- FunctionCallUtils.cpp - Utilities for C function calls -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions to call common simple C functions in
// LLVMIR (e.g. amon others to support printing and debugging).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::LLVM;

/// Helper functions to lookup or create the declaration for commonly used
/// external C function calls. The list of functions provided here must be
/// implemented separately (e.g. as  part of a support runtime library or as
/// part of the libc).
static constexpr llvm::StringRef kPrintI64 = "printI64";
static constexpr llvm::StringRef kPrintU64 = "printU64";
static constexpr llvm::StringRef kPrintF16 = "printF16";
static constexpr llvm::StringRef kPrintBF16 = "printBF16";
static constexpr llvm::StringRef kPrintF32 = "printF32";
static constexpr llvm::StringRef kPrintF64 = "printF64";
static constexpr llvm::StringRef kPrintString = "printString";
static constexpr llvm::StringRef kPrintOpen = "printOpen";
static constexpr llvm::StringRef kPrintClose = "printClose";
static constexpr llvm::StringRef kPrintComma = "printComma";
static constexpr llvm::StringRef kPrintNewline = "printNewline";
static constexpr llvm::StringRef kMalloc = "malloc";
static constexpr llvm::StringRef kAlignedAlloc = "aligned_alloc";
static constexpr llvm::StringRef kFree = "free";
static constexpr llvm::StringRef kGenericAlloc = "_mlir_memref_to_llvm_alloc";
static constexpr llvm::StringRef kGenericAlignedAlloc =
    "_mlir_memref_to_llvm_aligned_alloc";
static constexpr llvm::StringRef kGenericFree = "_mlir_memref_to_llvm_free";
static constexpr llvm::StringRef kMemRefCopy = "memrefCopy";

namespace {
/// Search for an LLVMFuncOp with a given name within an operation with the
/// SymbolTable trait. An optional collection of cached symbol tables can be
/// given to avoid a linear scan of the symbol table operation.
LLVM::LLVMFuncOp lookupFuncOp(StringRef name, Operation *symbolTableOp,
                              SymbolTableCollection *symbolTables = nullptr) {
  if (symbolTables) {
    return symbolTables->lookupSymbolIn<LLVM::LLVMFuncOp>(
        symbolTableOp, StringAttr::get(symbolTableOp->getContext(), name));
  }

  return llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTableOp, name));
}
} // namespace

/// Generic print function lookupOrCreate helper.
FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreateFn(OpBuilder &b, Operation *moduleOp, StringRef name,
                             ArrayRef<Type> paramTypes, Type resultType,
                             bool isVarArg, bool isReserved,
                             SymbolTableCollection *symbolTables) {
  assert(moduleOp->hasTrait<OpTrait::SymbolTable>() &&
         "expected SymbolTable operation");
  auto func = lookupFuncOp(name, moduleOp, symbolTables);
  auto funcT = LLVMFunctionType::get(resultType, paramTypes, isVarArg);
  // Assert the signature of the found function is same as expected
  if (func) {
    if (funcT != func.getFunctionType()) {
      if (isReserved) {
        func.emitError("redefinition of reserved function '")
            << name << "' of different type " << func.getFunctionType()
            << " is prohibited";
      } else {
        func.emitError("redefinition of function '")
            << name << "' of different type " << funcT << " is prohibited";
      }
      return failure();
    }
    return func;
  }

  OpBuilder::InsertionGuard g(b);
  assert(!moduleOp->getRegion(0).empty() && "expected non-empty region");
  b.setInsertionPointToStart(&moduleOp->getRegion(0).front());
  auto funcOp = LLVM::LLVMFuncOp::create(
      b, moduleOp->getLoc(), name,
      LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg));

  if (symbolTables) {
    SymbolTable &symbolTable = symbolTables->getSymbolTable(moduleOp);
    symbolTable.insert(funcOp, moduleOp->getRegion(0).front().begin());
  }

  return funcOp;
}

static FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateReservedFn(OpBuilder &b, Operation *moduleOp, StringRef name,
                         ArrayRef<Type> paramTypes, Type resultType,
                         SymbolTableCollection *symbolTables) {
  return lookupOrCreateFn(b, moduleOp, name, paramTypes, resultType,
                          /*isVarArg=*/false, /*isReserved=*/true,
                          symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintI64Fn(OpBuilder &b, Operation *moduleOp,
                                     SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintI64, IntegerType::get(moduleOp->getContext(), 64),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintU64Fn(OpBuilder &b, Operation *moduleOp,
                                     SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintU64, IntegerType::get(moduleOp->getContext(), 64),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintF16Fn(OpBuilder &b, Operation *moduleOp,
                                     SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintF16,
      IntegerType::get(moduleOp->getContext(), 16), // bits!
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintBF16Fn(OpBuilder &b, Operation *moduleOp,
                                      SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintBF16,
      IntegerType::get(moduleOp->getContext(), 16), // bits!
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintF32Fn(OpBuilder &b, Operation *moduleOp,
                                     SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintF32, Float32Type::get(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintF64Fn(OpBuilder &b, Operation *moduleOp,
                                     SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintF64, Float64Type::get(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

static LLVM::LLVMPointerType getCharPtr(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

static LLVM::LLVMPointerType getVoidPtr(MLIRContext *context) {
  // A char pointer and void ptr are the same in LLVM IR.
  return getCharPtr(context);
}

FailureOr<LLVM::LLVMFuncOp> mlir::LLVM::lookupOrCreatePrintStringFn(
    OpBuilder &b, Operation *moduleOp,
    std::optional<StringRef> runtimeFunctionName,
    SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, runtimeFunctionName.value_or(kPrintString),
      getCharPtr(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintOpenFn(OpBuilder &b, Operation *moduleOp,
                                      SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintOpen, {},
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintCloseFn(OpBuilder &b, Operation *moduleOp,
                                       SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintClose, {},
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintCommaFn(OpBuilder &b, Operation *moduleOp,
                                       SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintComma, {},
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreatePrintNewlineFn(OpBuilder &b, Operation *moduleOp,
                                         SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kPrintNewline, {},
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreateMallocFn(OpBuilder &b, Operation *moduleOp,
                                   Type indexType,
                                   SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(b, moduleOp, kMalloc, indexType,
                                  getVoidPtr(moduleOp->getContext()),
                                  symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreateAlignedAllocFn(OpBuilder &b, Operation *moduleOp,
                                         Type indexType,
                                         SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kAlignedAlloc, {indexType, indexType},
      getVoidPtr(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreateFreeFn(OpBuilder &b, Operation *moduleOp,
                                 SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kFree, getVoidPtr(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreateGenericAllocFn(OpBuilder &b, Operation *moduleOp,
                                         Type indexType,
                                         SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(b, moduleOp, kGenericAlloc, indexType,
                                  getVoidPtr(moduleOp->getContext()),
                                  symbolTables);
}

FailureOr<LLVM::LLVMFuncOp> mlir::LLVM::lookupOrCreateGenericAlignedAllocFn(
    OpBuilder &b, Operation *moduleOp, Type indexType,
    SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kGenericAlignedAlloc, {indexType, indexType},
      getVoidPtr(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp>
mlir::LLVM::lookupOrCreateGenericFreeFn(OpBuilder &b, Operation *moduleOp,
                                        SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kGenericFree, getVoidPtr(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}

FailureOr<LLVM::LLVMFuncOp> mlir::LLVM::lookupOrCreateMemRefCopyFn(
    OpBuilder &b, Operation *moduleOp, Type indexType,
    Type unrankedDescriptorType, SymbolTableCollection *symbolTables) {
  return lookupOrCreateReservedFn(
      b, moduleOp, kMemRefCopy,
      ArrayRef<Type>{indexType, unrankedDescriptorType, unrankedDescriptorType},
      LLVM::LLVMVoidType::get(moduleOp->getContext()), symbolTables);
}
