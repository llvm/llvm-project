//===- Bufferization.cpp - C Interface for Bufferization dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Bufferization.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir::bufferization;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Bufferization, bufferization,
                                      mlir::bufferization::BufferizationDialect)
