//===- XeVM.cpp - C Interface for XeVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/XeVM.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(XeVM, xevm, mlir::xevm::XeVMDialect)
