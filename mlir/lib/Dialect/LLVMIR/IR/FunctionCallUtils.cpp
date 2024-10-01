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

/// Generic print function lookupOrCreate helper.
LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateFn(Operation *moduleOp,
                                              StringRef name,
                                              ArrayRef<Type> paramTypes,
                                              Type resultType, bool isVarArg) {
  assert(moduleOp->hasTrait<OpTrait::SymbolTable>() &&
         "expected SymbolTable operation");
  auto func = llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, name));
  if (func)
    return func;
  OpBuilder b(moduleOp->getRegion(0));
  return b.create<LLVM::LLVMFuncOp>(
      moduleOp->getLoc(), name,
      LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintI64Fn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintI64,
                          IntegerType::get(moduleOp->getContext(), 64),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintU64Fn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintU64,
                          IntegerType::get(moduleOp->getContext(), 64),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintF16Fn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintF16,
                          IntegerType::get(moduleOp->getContext(), 16), // bits!
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintBF16Fn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintBF16,
                          IntegerType::get(moduleOp->getContext(), 16), // bits!
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintF32Fn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintF32,
                          Float32Type::get(moduleOp->getContext()),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintF64Fn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintF64,
                          Float64Type::get(moduleOp->getContext()),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

static LLVM::LLVMPointerType getCharPtr(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

static LLVM::LLVMPointerType getVoidPtr(MLIRContext *context) {
  // A char pointer and void ptr are the same in LLVM IR.
  return getCharPtr(context);
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintStringFn(
    Operation *moduleOp, std::optional<StringRef> runtimeFunctionName) {
  return lookupOrCreateFn(moduleOp, runtimeFunctionName.value_or(kPrintString),
                          getCharPtr(moduleOp->getContext()),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintOpenFn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintOpen, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintCloseFn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintClose, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintCommaFn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintComma, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintNewlineFn(Operation *moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintNewline, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateMallocFn(Operation *moduleOp,
                                                    Type indexType) {
  return LLVM::lookupOrCreateFn(moduleOp, kMalloc, indexType,
                                getVoidPtr(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateAlignedAllocFn(Operation *moduleOp,
                                                          Type indexType) {
  return LLVM::lookupOrCreateFn(moduleOp, kAlignedAlloc, {indexType, indexType},
                                getVoidPtr(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateFreeFn(Operation *moduleOp) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kFree, getVoidPtr(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateGenericAllocFn(Operation *moduleOp,
                                                          Type indexType) {
  return LLVM::lookupOrCreateFn(moduleOp, kGenericAlloc, indexType,
                                getVoidPtr(moduleOp->getContext()));
}

LLVM::LLVMFuncOp
mlir::LLVM::lookupOrCreateGenericAlignedAllocFn(Operation *moduleOp,
                                                Type indexType) {
  return LLVM::lookupOrCreateFn(moduleOp, kGenericAlignedAlloc,
                                {indexType, indexType},
                                getVoidPtr(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateGenericFreeFn(Operation *moduleOp) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kGenericFree, getVoidPtr(moduleOp->getContext()),
      LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp
mlir::LLVM::lookupOrCreateMemRefCopyFn(Operation *moduleOp, Type indexType,
                                       Type unrankedDescriptorType) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kMemRefCopy,
      ArrayRef<Type>{indexType, unrankedDescriptorType, unrankedDescriptorType},
      LLVM::LLVMVoidType::get(moduleOp->getContext()));
}
