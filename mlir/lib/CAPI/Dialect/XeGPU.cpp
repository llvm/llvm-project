//===- XeGPU.cpp - C Interface for XeGPU dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/XeGPU.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(XeGPU, xegpu, mlir::xegpu::XeGPUDialect)
