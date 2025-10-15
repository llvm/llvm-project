//===- OpenMPOpsAttributes.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPOPSATTRIBUTES_H_
#define MLIR_DIALECT_OPENMP_OPENMPOPSATTRIBUTES_H_

#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.h.inc"

#endif // MLIR_DIALECT_OPENMP_OPENMPOPSATTRIBUTES_H_
