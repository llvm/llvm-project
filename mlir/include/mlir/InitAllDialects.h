//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITALLDIALECTS_H_
#define MLIR_INITALLDIALECTS_H_

#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SDBM/SDBMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
inline void registerAllDialects() {
  static bool init_once = []() {
    registerDialect<acc::OpenACCDialect>();
    registerDialect<AffineDialect>();
    registerDialect<avx512::AVX512Dialect>();
    registerDialect<gpu::GPUDialect>();
    registerDialect<LLVM::LLVMAVX512Dialect>();
    registerDialect<LLVM::LLVMDialect>();
    registerDialect<linalg::LinalgDialect>();
    registerDialect<scf::SCFDialect>();
    registerDialect<omp::OpenMPDialect>();
    registerDialect<quant::QuantizationDialect>();
    registerDialect<spirv::SPIRVDialect>();
    registerDialect<StandardOpsDialect>();
    registerDialect<vector::VectorDialect>();
    registerDialect<NVVM::NVVMDialect>();
    registerDialect<ROCDL::ROCDLDialect>();
    registerDialect<SDBMDialect>();
    registerDialect<shape::ShapeDialect>();
    return true;
  }();
  (void)init_once;
}
} // namespace mlir

#endif // MLIR_INITALLDIALECTS_H_
