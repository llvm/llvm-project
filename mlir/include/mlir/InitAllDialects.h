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
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SDBM/SDBMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<acc::OpenACCDialect,
                  AffineDialect,
                  arm_neon::ArmNeonDialect,
                  async::AsyncDialect,
                  avx512::AVX512Dialect,
                  gpu::GPUDialect,
                  LLVM::LLVMAVX512Dialect,
                  LLVM::LLVMDialect,
                  LLVM::LLVMArmNeonDialect,
                  LLVM::LLVMArmSVEDialect,
                  linalg::LinalgDialect,
                  scf::SCFDialect,
                  omp::OpenMPDialect,
                  pdl::PDLDialect,
                  pdl_interp::PDLInterpDialect,
                  quant::QuantizationDialect,
                  spirv::SPIRVDialect,
                  StandardOpsDialect,
                  arm_sve::ArmSVEDialect,
                  vector::VectorDialect,
                  NVVM::NVVMDialect,
                  ROCDL::ROCDLDialect,
                  SDBMDialect,
                  shape::ShapeDialect,
                  tensor::TensorDialect,
                  tosa::TosaDialect>();
  // clang-format on
}

} // namespace mlir

#endif // MLIR_INITALLDIALECTS_H_
