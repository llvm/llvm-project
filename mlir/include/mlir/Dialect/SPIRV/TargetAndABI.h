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
enum class Capability : uint32_t;
enum class Extension;
enum class StorageClass : uint32_t;
enum class Version : uint32_t;

namespace detail {
struct TargetEnvAttributeStorage;
} // namespace detail

/// SPIR-V dialect-specific attribute kinds.
// TODO(antiagainst): move to a more suitable place if we have more attributes.
namespace AttrKind {
enum Kind {
  TargetEnv = Attribute::FIRST_SPIRV_ATTR,
};
} // namespace AttrKind

/// An attribute that specifies the target version, allowed extensions and
/// capabilities, and resource limits. These information describles a SPIR-V
/// target environment.
class TargetEnvAttr
    : public Attribute::AttrBase<TargetEnvAttr, Attribute,
                                 detail::TargetEnvAttributeStorage> {
public:
  using Base::Base;

  /// Gets a TargetEnvAttr instance.
  static TargetEnvAttr get(IntegerAttr version, ArrayAttr extensions,
                           ArrayAttr capabilities, DictionaryAttr limits);

  /// Returns the attribute kind's name (without the 'spv.' prefix).
  static StringRef getKindName();

  /// Returns the target version.
  Version getVersion();

  struct ext_iterator final
      : public llvm::mapped_iterator<ArrayAttr::iterator,
                                     Extension (*)(Attribute)> {
    explicit ext_iterator(ArrayAttr::iterator it);
  };
  using ext_range = llvm::iterator_range<ext_iterator>;

  /// Returns the target extensions.
  ext_range getExtensions();
  /// Returns the target extensions as a string array attribute.
  ArrayAttr getExtensionsAttr();

  struct cap_iterator final
      : public llvm::mapped_iterator<ArrayAttr::iterator,
                                     Capability (*)(Attribute)> {
    explicit cap_iterator(ArrayAttr::iterator it);
  };
  using cap_range = llvm::iterator_range<cap_iterator>;

  /// Returns the target capabilities.
  cap_range getCapabilities();
  /// Returns the target capabilities as an integer array attribute.
  ArrayAttr getCapabilitiesAttr();

  /// Returns the target resource limits.
  DictionaryAttr getResourceLimits();

  static bool kindof(unsigned kind) { return kind == AttrKind::TargetEnv; }

  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    IntegerAttr version,
                                                    ArrayAttr extensions,
                                                    ArrayAttr capabilities,
                                                    DictionaryAttr limits);
};

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

/// Queries the entry point ABI on the nearest function-like op containing the
/// given `op`. Returns null attribute if not found.
EntryPointABIAttr lookupEntryPointABI(Operation *op);

/// Queries the local workgroup size from entry point ABI on the nearest
/// function-like op containing the given `op`. Returns null attribute if not
/// found.
DenseIntElementsAttr lookupLocalWorkGroupSize(Operation *op);

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

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TARGETANDABI_H
