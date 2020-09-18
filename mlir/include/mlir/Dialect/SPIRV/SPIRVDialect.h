//===- SPIRVDialect.h - MLIR SPIR-V dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_
#define MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace spirv {

enum class Decoration : uint32_t;

} // end namespace spirv
} // end namespace mlir

#include "mlir/Dialect/SPIRV/SPIRVOpsDialect.h.inc"

#endif // MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_
