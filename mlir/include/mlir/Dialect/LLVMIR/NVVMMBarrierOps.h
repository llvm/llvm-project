//===- NVVMMBarrierOps.h - NVVM dialect mbarrier ops ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `MBarrier` of the NVVM dialect:
// the mbarrier ops, including `cp.async.mbarrier.arrive`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMMBARRIEROPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMMBARRIEROPS_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMBarrierOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMMBARRIEROPS_H_
