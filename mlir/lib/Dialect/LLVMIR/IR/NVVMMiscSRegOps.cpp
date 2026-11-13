//===- NVVMMiscSRegOps.cpp - NVVM dialect special registers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `MiscSReg` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMMiscSRegOps.h"

#include "IR/NVVMOps.h"

using namespace mlir;
using namespace mlir::NVVM;

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMiscSRegOps.cpp.inc"
