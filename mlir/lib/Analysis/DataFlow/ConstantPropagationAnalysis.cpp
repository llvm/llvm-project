//===- ConstantPropagationAnalysis.cpp - Constant propagation analysis ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// ConstantValue
//===----------------------------------------------------------------------===//

void ConstantValue::print(raw_ostream &os) const {
  if (constant)
    return constant.print(os);
  os << "<NO VALUE>";
}
