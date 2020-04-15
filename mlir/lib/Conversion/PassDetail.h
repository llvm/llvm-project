//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H_
#define CONVERSION_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace gpu {
class GPUModuleOp;
} // end namespace gpu

#define GEN_PASS_CLASSES
#include "mlir/Conversion/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_PASSDETAIL_H_
