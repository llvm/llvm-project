//===- NVVMWgmmaOps.h - NVVM dialect wgmma ops ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `Wgmma` of the NVVM dialect: the
// asynchronous warpgroup-level matrix multiply-accumulate ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMWGMMAOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMWGMMAOPS_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMWgmmaOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMWGMMAOPS_H_
