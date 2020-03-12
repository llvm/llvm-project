//===- SPIRVAttributes.h - SPIR-V attribute declarations  -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares SPIR-V dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVATTRIBUTES_H
#define MLIR_DIALECT_SPIRV_SPIRVATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
// Pull in SPIR-V attribute definitions for target and ABI.
#include "mlir/Dialect/SPIRV/TargetAndABI.h.inc"

namespace spirv {
enum class Capability : uint32_t;
enum class Extension;
enum class Version : uint32_t;

namespace detail {
struct TargetEnvAttributeStorage;
struct VerCapExtAttributeStorage;
} // namespace detail

/// SPIR-V dialect-specific attribute kinds.
namespace AttrKind {
enum Kind {
  TargetEnv = Attribute::FIRST_SPIRV_ATTR, /// Target environment
  VerCapExt, /// (version, extension, capability) triple
};
} // namespace AttrKind

/// An attribute that specifies the SPIR-V (version, capabilities, extensions)
/// triple.
class VerCapExtAttr
    : public Attribute::AttrBase<VerCapExtAttr, Attribute,
                                 detail::VerCapExtAttributeStorage> {
public:
  using Base::Base;

  /// Gets a VerCapExtAttr instance.
  static VerCapExtAttr get(Version version, ArrayRef<Capability> capabilities,
                           ArrayRef<Extension> extensions,
                           MLIRContext *context);
  static VerCapExtAttr get(IntegerAttr version, ArrayAttr capabilities,
                           ArrayAttr extensions);

  /// Returns the attribute kind's name (without the 'spv.' prefix).
  static StringRef getKindName();

  /// Returns the version.
  Version getVersion();

  struct ext_iterator final
      : public llvm::mapped_iterator<ArrayAttr::iterator,
                                     Extension (*)(Attribute)> {
    explicit ext_iterator(ArrayAttr::iterator it);
  };
  using ext_range = llvm::iterator_range<ext_iterator>;

  /// Returns the extensions.
  ext_range getExtensions();
  /// Returns the extensions as a string array attribute.
  ArrayAttr getExtensionsAttr();

  struct cap_iterator final
      : public llvm::mapped_iterator<ArrayAttr::iterator,
                                     Capability (*)(Attribute)> {
    explicit cap_iterator(ArrayAttr::iterator it);
  };
  using cap_range = llvm::iterator_range<cap_iterator>;

  /// Returns the capabilities.
  cap_range getCapabilities();
  /// Returns the capabilities as an integer array attribute.
  ArrayAttr getCapabilitiesAttr();

  static bool kindof(unsigned kind) { return kind == AttrKind::VerCapExt; }

  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    IntegerAttr version,
                                                    ArrayAttr capabilities,
                                                    ArrayAttr extensions);
};

/// An attribute that specifies the target version, allowed extensions and
/// capabilities, and resource limits. These information describles a SPIR-V
/// target environment.
class TargetEnvAttr
    : public Attribute::AttrBase<TargetEnvAttr, Attribute,
                                 detail::TargetEnvAttributeStorage> {
public:
  using Base::Base;

  /// Gets a TargetEnvAttr instance.
  static TargetEnvAttr get(VerCapExtAttr triple, DictionaryAttr limits);

  /// Returns the attribute kind's name (without the 'spv.' prefix).
  static StringRef getKindName();

  /// Returns the (version, capabilities, extensions) triple attribute.
  VerCapExtAttr getTripleAttr();

  /// Returns the target version.
  Version getVersion();

  /// Returns the target extensions.
  VerCapExtAttr::ext_range getExtensions();
  /// Returns the target extensions as a string array attribute.
  ArrayAttr getExtensionsAttr();

  /// Returns the target capabilities.
  VerCapExtAttr::cap_range getCapabilities();
  /// Returns the target capabilities as an integer array attribute.
  ArrayAttr getCapabilitiesAttr();

  /// Returns the target resource limits.
  ResourceLimitsAttr getResourceLimits();

  static bool kindof(unsigned kind) { return kind == AttrKind::TargetEnv; }

  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    VerCapExtAttr triple,
                                                    DictionaryAttr limits);
};
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVATTRIBUTES_H
