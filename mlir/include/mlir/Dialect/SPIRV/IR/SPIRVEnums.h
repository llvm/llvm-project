//===- SPIRVEnums.h - MLIR SPIR-V Enums -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the C/C++ enums from SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_SPIRVENUMS_H_
#define MLIR_DIALECT_SPIRV_IR_SPIRVENUMS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"

// Forward declare enum classes related to op availability. Their definitions
// are in the TableGen'erated SPIRVEnums.h.inc and can be referenced by other
// declarations in SPIRVEnums.h.inc.
namespace mlir {
namespace spirv {
enum class Version : uint32_t;
enum class Extension;
enum class Capability : uint32_t;
} // namespace spirv
} // namespace mlir

// Pull in all enum type definitions and utility function declarations
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h.inc"

// Pull in all enum type availability query function declarations
#include "mlir/Dialect/SPIRV/IR/SPIRVEnumAvailability.h.inc"

namespace mlir {
namespace spirv {
/// Returns the implied extensions for the given version. These extensions are
/// incorporated into the current version so they are implicitly declared when
/// targeting the given version.
ArrayRef<Extension> getImpliedExtensions(Version version);

/// Returns the directly implied capabilities for the given capability. These
/// capabilities are implicitly declared by the given capability.
ArrayRef<Capability> getDirectImpliedCapabilities(Capability cap);
/// Returns the recursively implied capabilities for the given capability. These
/// capabilities are implicitly declared by the given capability. Compared to
/// the above function, this function collects implied capabilities recursively:
/// if an implicitly declared capability implicitly declares a third one, the
/// third one will also be returned.
SmallVector<Capability, 0> getRecursiveImpliedCapabilities(Capability cap);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_IR_SPIRVENUMS_H_
