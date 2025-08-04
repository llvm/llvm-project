//===- Presburger.h - C API Utils for Presburger library --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// Presburger library. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_PRESBURGER_H
#define MLIR_CAPI_PRESBURGER_H

#include "mlir-c/Presburger.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/CAPI/Wrap.h"

DEFINE_C_API_PTR_METHODS(MlirPresburgerIntegerRelation,
                         mlir::presburger::IntegerRelation)

#endif /* MLIR_CAPI_PRESBURGER_H */