//===- NVVMAsyncCopyOps.h - NVVM dialect asynchronous copy ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `AsyncCopy` of the NVVM dialect:
// the asynchronous copies, i.e. `cp.async`, the bulk and tensor (TMA) copies,
// their completion mechanisms and the tensor map updates.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMASYNCCOPYOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMASYNCCOPYOPS_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMAsyncCopyOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMASYNCCOPYOPS_H_
