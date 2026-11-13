//===- NVVMMiscOps.h - NVVM dialect miscellaneous ops -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `Misc` of the NVVM dialect: the
// ops that fit no other partition.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMMISCOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMMISCOPS_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMiscOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMMISCOPS_H_
