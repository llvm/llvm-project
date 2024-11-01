//===- Support.cpp - Helpers for C interface to MLIR API ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Support.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ThreadPool.h"

#include <cstring>

MlirStringRef mlirStringRefCreateFromCString(const char *str) {
  return mlirStringRefCreate(str, strlen(str));
}

bool mlirStringRefEqual(MlirStringRef string, MlirStringRef other) {
  return llvm::StringRef(string.data, string.length) ==
         llvm::StringRef(other.data, other.length);
}

//===----------------------------------------------------------------------===//
// LLVM ThreadPool API.
//===----------------------------------------------------------------------===//
MlirLlvmThreadPool mlirLlvmThreadPoolCreate() {
  return wrap(new llvm::ThreadPool());
}

void mlirLlvmThreadPoolDestroy(MlirLlvmThreadPool threadPool) {
  delete unwrap(threadPool);
}

//===----------------------------------------------------------------------===//
// TypeID API.
//===----------------------------------------------------------------------===//
MlirTypeID mlirTypeIDCreate(const void *ptr) {
  assert(reinterpret_cast<uintptr_t>(ptr) % 8 == 0 &&
         "ptr must be 8 byte aligned");
  // This is essentially a no-op that returns back `ptr`, but by going through
  // the `TypeID` functions we can get compiler errors in case the `TypeID`
  // api/representation changes
  return wrap(mlir::TypeID::getFromOpaquePointer(ptr));
}

bool mlirTypeIDEqual(MlirTypeID typeID1, MlirTypeID typeID2) {
  return unwrap(typeID1) == unwrap(typeID2);
}

size_t mlirTypeIDHashValue(MlirTypeID typeID) {
  return hash_value(unwrap(typeID));
}

//===----------------------------------------------------------------------===//
// TypeIDAllocator API.
//===----------------------------------------------------------------------===//

MlirTypeIDAllocator mlirTypeIDAllocatorCreate() {
  return wrap(new mlir::TypeIDAllocator());
}

void mlirTypeIDAllocatorDestroy(MlirTypeIDAllocator allocator) {
  delete unwrap(allocator);
}

MlirTypeID mlirTypeIDAllocatorAllocateTypeID(MlirTypeIDAllocator allocator) {
  return wrap(unwrap(allocator)->allocate());
}
