//===- PassDetail.h - Reducer Pass class details ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_PASSDETAIL_H
#define MLIR_REDUCER_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "mlir/Reducer/Passes.h.inc"

} // end namespace mlir

#endif // MLIR_REDUCER_PASSDETAIL_H
