//===- NVVMMmaOps.h - NVVM dialect MMA ops ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `Mma` of the NVVM dialect: the
// matrix multiply-accumulate ops, including wmma, wgmma and the matrix loads
// and stores.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMMMAOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMMMAOPS_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMmaOps.h.inc"

namespace mlir {
namespace NVVM {
/// Return the element type and number of elements associated with a wmma matrix
/// of given chracteristics. This matches the logic in IntrinsicsNVVM.td
/// WMMA_REGS structure.
std::pair<mlir::Type, unsigned> inferMMAType(mlir::NVVM::MMATypes type,
                                             mlir::NVVM::MMAFrag frag, int nRow,
                                             int nCol,
                                             mlir::MLIRContext *context);
} // namespace NVVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_NVVMMMAOPS_H_
