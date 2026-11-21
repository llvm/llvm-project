//===- IRMapping.h - C API wrap/unwrap for IRMapping -------------*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_IRMAPPING_H
#define MLIR_CAPI_IRMAPPING_H

#include "mlir-c/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/IRMapping.h"

DEFINE_C_API_PTR_METHODS(MlirIRMapping, mlir::IRMapping)

#endif // MLIR_CAPI_IRMAPPING_H
