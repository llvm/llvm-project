//===- NVVMDataMovementOps.h - NVVM dialect data movement ops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `DataMovement` of the NVVM
// dialect: the data movement instructions of PTX except `cvt`, i.e. the
// asynchronous and bulk copies, prefetches, stores, shuffles, byte
// permutations, sparse compressions, address mapping and tensor map updates.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMDATAMOVEMENTOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMDATAMOVEMENTOPS_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMDataMovementOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMDATAMOVEMENTOPS_H_
