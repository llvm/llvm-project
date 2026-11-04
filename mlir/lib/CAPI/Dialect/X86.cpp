//===- X86.cpp - C Interface for X86 dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/X86.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/X86/X86Dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(X86, x86, mlir::x86::X86Dialect)
