//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_PASSDETAIL_H_
#define TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_CLASSES
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#endif // TRANSFORMS_PASSDETAIL_H_
