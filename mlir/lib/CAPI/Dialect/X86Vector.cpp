//===- X86Vector.cpp - C Interface for X86Vector dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/X86Vector.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(X86Vector, x86vector,
                                      mlir::x86vector::X86VectorDialect)
