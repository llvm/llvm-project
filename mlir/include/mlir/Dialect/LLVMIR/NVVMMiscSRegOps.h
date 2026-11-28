//===- NVVMMiscSRegOps.h - NVVM dialect special registers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `MiscSReg` of the NVVM dialect:
// the reads of all other special registers: lane masks, shared memory sizes,
// clocks, timers and environment registers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMMISCSREGOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMMISCSREGOPS_H_

#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMiscSRegOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMMISCSREGOPS_H_
