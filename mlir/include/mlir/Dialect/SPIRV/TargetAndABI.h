//===- TargetAndABI.h - SPIR-V target and ABI utilities  --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for SPIR-V target and shader interface ABI.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_TARGETANDABI_H
#define MLIR_DIALECT_SPIRV_TARGETANDABI_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class OpBuilder;
class Operation;
class Value;

// Pull in SPIR-V attribute definitions.
#include "mlir/Dialect/SPIRV/TargetAndABI.h.inc"

namespace spirv {
enum class StorageClass : uint32_t;

/// Returns the attribute name for specifying argument ABI information.
StringRef getInterfaceVarABIAttrName();

/// Gets the InterfaceVarABIAttr given its fields.
InterfaceVarABIAttr getInterfaceVarABIAttr(unsigned descriptorSet,
                                           unsigned binding,
                                           StorageClass storageClass,
                                           MLIRContext *context);

/// Returns the attribute name for specifying entry point information.
StringRef getEntryPointABIAttrName();

/// Gets the EntryPointABIAttr given its fields.
EntryPointABIAttr getEntryPointABIAttr(ArrayRef<int32_t> localSize,
                                       MLIRContext *context);

/// Returns a default resource limits attribute that uses numbers from
/// "Table 46. Required Limits" of the Vulkan spec.
ResourceLimitsAttr getDefaultResourceLimits(MLIRContext *context);

/// Returns the attribute name for specifying SPIR-V target environment.
StringRef getTargetEnvAttrName();

/// Returns the default target environment: SPIR-V 1.0 with Shader capability
/// and no extra extensions.
TargetEnvAttr getDefaultTargetEnv(MLIRContext *context);

/// Queries the target environment from the given `op` or returns the default
/// target environment (SPIR-V 1.0 with Shader capability and no extra
/// extensions) if not provided.
TargetEnvAttr lookupTargetEnvOrDefault(Operation *op);

/// Queries the local workgroup size from entry point ABI on the nearest
/// function-like op containing the given `op`. Returns null attribute if not
/// found.
DenseIntElementsAttr lookupLocalWorkGroupSize(Operation *op);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TARGETANDABI_H
