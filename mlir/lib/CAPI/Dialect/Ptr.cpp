//===- Ptr.cpp - C Interface for Ptr dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Ptr.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ptr, ptr, mlir::ptr::PtrDialect)
