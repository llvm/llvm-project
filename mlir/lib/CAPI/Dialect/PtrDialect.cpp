//===- PtrDialect.cpp - C interface for the Ptr dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/PtrDialect.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ptr-dialect-capi"

using namespace mlir;
using namespace ptr;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ptr, ptr, mlir::ptr::PtrDialect)

bool mlirPtrTypeIsAPtrType(MlirType type) {
  return llvm::isa<ptr::PtrType>(unwrap(type));
}

MlirType mlirPtrGetPtrType(MlirAttribute memorySpace) {
  MemorySpaceAttrInterface memorySpaceAttr =
      dyn_cast<MemorySpaceAttrInterface>(unwrap(memorySpace));
  if (!memorySpaceAttr) {
    LLVM_DEBUG(llvm::dbgs()
               << "expected memory-space to be MemorySpaceAttrInterface");
    return {nullptr};
  }
  return wrap(ptr::PtrType::get(memorySpaceAttr));
}
