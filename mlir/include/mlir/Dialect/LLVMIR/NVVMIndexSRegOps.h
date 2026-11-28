//===- NVVMIndexSRegOps.h - NVVM dialect index registers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `IndexSReg` of the NVVM dialect:
// the reads of the special registers that hold the indices and dimensions of
// threads, blocks, clusters, lanes, warps, SMs and grids.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMINDEXSREGOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMINDEXSREGOPS_H_

#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMIndexSRegOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMINDEXSREGOPS_H_
