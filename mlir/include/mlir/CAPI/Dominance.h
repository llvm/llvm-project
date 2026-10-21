//===- Dominance.h - C API wrap/unwrap for Dominance -------------*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_DOMINANCE_H
#define MLIR_CAPI_DOMINANCE_H

#include "mlir-c/Dominance.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Dominance.h"

DEFINE_C_API_PTR_METHODS(MlirDominanceInfo, mlir::DominanceInfo)
DEFINE_C_API_PTR_METHODS(MlirPostDominanceInfo, mlir::PostDominanceInfo)

#endif // MLIR_CAPI_DOMINANCE_H
