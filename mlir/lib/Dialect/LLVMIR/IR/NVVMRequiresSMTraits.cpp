//===--- NVVMRequiresSMTraits.cpp - NVVM Requires SM Traits -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op traits for the NVVM Dialect in MLIR
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.cpp.inc"

using namespace mlir;

llvm::NVPTX::GPUKind NVVM::parseTargetArch(StringRef chip) {
  using namespace llvm::NVPTX;
  GPUKind arch = parseArch(chip);
  switch (arch) {
  case GK_SM_101:
    return GK_SM_110;
  case GK_SM_101f:
    return GK_SM_110f;
  case GK_SM_101a:
    return GK_SM_110a;
  default:
    return arch;
  }
}
