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
static constexpr llvm::StringRef kPrintF32 = "printF32";
static constexpr llvm::StringRef kPrintF64 = "printF64";
static constexpr llvm::StringRef kPrintStr = "puts";
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
LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                                              ArrayRef<Type> paramTypes,
                                              Type resultType) {
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (func)
    return func;
  OpBuilder b(moduleOp.getBodyRegion());
  return b.create<LLVM::LLVMFuncOp>(
      moduleOp->getLoc(), name,
      LLVM::LLVMFunctionType::get(resultType, paramTypes));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintI64Fn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintI64,
                          IntegerType::get(moduleOp->getContext(), 64),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintU64Fn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintU64,
                          IntegerType::get(moduleOp->getContext(), 64),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintF32Fn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintF32,
                          Float32Type::get(moduleOp->getContext()),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintF64Fn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintF64,
                          Float64Type::get(moduleOp->getContext()),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

static LLVM::LLVMPointerType getCharPtr(MLIRContext *context,
                                        bool opaquePointers) {
  if (opaquePointers)
    return LLVM::LLVMPointerType::get(context);

  return LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
}

static LLVM::LLVMPointerType getVoidPtr(MLIRContext *context,
                                        bool opaquePointers) {
  // A char pointer and void ptr are the same in LLVM IR.
  return getCharPtr(context, opaquePointers);
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintStrFn(ModuleOp moduleOp,
                                                      bool opaquePointers) {
  return lookupOrCreateFn(moduleOp, kPrintStr,
                          getCharPtr(moduleOp->getContext(), opaquePointers),
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintOpenFn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintOpen, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintCloseFn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintClose, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintCommaFn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintComma, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreatePrintNewlineFn(ModuleOp moduleOp) {
  return lookupOrCreateFn(moduleOp, kPrintNewline, {},
                          LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateMallocFn(ModuleOp moduleOp,
                                                    Type indexType,
                                                    bool opaquePointers) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kMalloc, indexType,
      getVoidPtr(moduleOp->getContext(), opaquePointers));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateAlignedAllocFn(ModuleOp moduleOp,
                                                          Type indexType,
                                                          bool opaquePointers) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kAlignedAlloc, {indexType, indexType},
      getVoidPtr(moduleOp->getContext(), opaquePointers));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateFreeFn(ModuleOp moduleOp,
                                                  bool opaquePointers) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kFree, getVoidPtr(moduleOp->getContext(), opaquePointers),
      LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateGenericAllocFn(ModuleOp moduleOp,
                                                          Type indexType,
                                                          bool opaquePointers) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kGenericAlloc, indexType,
      getVoidPtr(moduleOp->getContext(), opaquePointers));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateGenericAlignedAllocFn(
    ModuleOp moduleOp, Type indexType, bool opaquePointers) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kGenericAlignedAlloc, {indexType, indexType},
      getVoidPtr(moduleOp->getContext(), opaquePointers));
}

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateGenericFreeFn(ModuleOp moduleOp,
                                                         bool opaquePointers) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kGenericFree,
      getVoidPtr(moduleOp->getContext(), opaquePointers),
      LLVM::LLVMVoidType::get(moduleOp->getContext()));
}

LLVM::LLVMFuncOp
mlir::LLVM::lookupOrCreateMemRefCopyFn(ModuleOp moduleOp, Type indexType,
                                       Type unrankedDescriptorType) {
  return LLVM::lookupOrCreateFn(
      moduleOp, kMemRefCopy,
      ArrayRef<Type>{indexType, unrankedDescriptorType, unrankedDescriptorType},
      LLVM::LLVMVoidType::get(moduleOp->getContext()));
}
