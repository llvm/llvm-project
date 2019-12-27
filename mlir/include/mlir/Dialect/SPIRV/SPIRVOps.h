//===- SPIRVOps.h - MLIR SPIR-V operations ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVOPS_H_
#define MLIR_DIALECT_SPIRV_SPIRVOPS_H_

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Function.h"

namespace mlir {
class OpBuilder;

namespace spirv {

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/SPIRVAvailability.h.inc"

// TablenGen'erated operation declarations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVOps.h.inc"

// TableGen'erated helper functions.
//
// Get the name used in the Op to refer to an enum value of the given
// `EnumClass`.
// template <typename EnumClass> StringRef attributeName();
//
// Get the function that can be used to symbolize an enum value.
// template <typename EnumClass>
// Optional<EnumClass> (*)(StringRef) symbolizeEnum();
#include "mlir/Dialect/SPIRV/SPIRVOpUtils.inc"

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVOPS_H_
