//===- OpenACCSupport.cpp - OpenACCSupport Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenACCSupport analysis interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"

namespace mlir {
namespace acc {

std::string OpenACCSupport::getVariableName(Value v) {
  if (impl)
    return impl->getVariableName(v);
  return acc::getVariableName(v);
}

} // namespace acc
} // namespace mlir
