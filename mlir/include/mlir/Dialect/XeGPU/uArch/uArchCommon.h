//===- uArchCommon.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Common functionality related to uArch instances.
//
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_UARCHCOMMON_H
#define MLIR_DIALECT_XEGPU_UARCH_UARCHCOMMON_H

#include "IntelGpuXe2.h"
#include "IntelGpuXe3.h"

namespace mlir {
namespace xegpu {
namespace uArch {

inline const uArch *getUArch(llvm::StringRef archName) {
  if (archName.equals_insensitive("pvc"))
    return PVCuArch::getInstance();
  if (archName.equals_insensitive("bmg"))
    return BMGuArch::getInstance();
  if (archName.equals_insensitive("cri"))
    return CRIuArch::getInstance();
  return nullptr;
}

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_UARCHCOMMON_H
