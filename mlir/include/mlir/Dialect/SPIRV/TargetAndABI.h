//===- TargetAndABI.h - SPIR-V target and ABI utilities  --------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
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

/// Attribute name for specifying argument ABI information.
StringRef getInterfaceVarABIAttrName();

/// Get the InterfaceVarABIAttr given its fields.
InterfaceVarABIAttr getInterfaceVarABIAttr(unsigned descriptorSet,
                                           unsigned binding,
                                           StorageClass storageClass,
                                           MLIRContext *context);

/// Attribute name for specifying entry point information.
StringRef getEntryPointABIAttrName();

/// Get the EntryPointABIAttr given its fields.
EntryPointABIAttr getEntryPointABIAttr(ArrayRef<int32_t> localSize,
                                       MLIRContext *context);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TARGETANDABI_H
