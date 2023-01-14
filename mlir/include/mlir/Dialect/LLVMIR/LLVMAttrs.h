//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
#define MLIR_DIALECT_LLVMIR_LLVMATTRS_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpImplementation.h"
#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"

namespace mlir {
namespace LLVM {
class LoopOptionsAttrBuilder;

/// This class represents the base attribute for all debug info attributes.
class DINodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  // Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info scope.
class DIScopeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info type.
class DITypeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

// Inline the LLVM generated Linkage enum and utility.
// This is only necessary to isolate the "enum generated code" from the
// attribute definition itself.
// TODO: this shouldn't be needed after we unify the attribute generation, i.e.
// --gen-attr-* and --gen-attrdef-*.
using cconv::CConv;
using linkage::Linkage;
} // namespace LLVM
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc"

namespace mlir {
namespace LLVM {

/// Builder class for LoopOptionsAttr. This helper class allows to progressively
/// build a LoopOptionsAttr one option at a time, and pay the price of attribute
/// creation once all the options are in place.
class LoopOptionsAttrBuilder {
public:
  /// Construct a empty builder.
  LoopOptionsAttrBuilder() = default;

  /// Construct a builder with an initial list of options from an existing
  /// LoopOptionsAttr.
  LoopOptionsAttrBuilder(LoopOptionsAttr attr);

  /// Set the `disable_licm` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setDisableLICM(Optional<bool> value);

  /// Set the `interleave_count` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setInterleaveCount(Optional<uint64_t> count);

  /// Set the `disable_unroll` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setDisableUnroll(Optional<bool> value);

  /// Set the `disable_pipeline` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setDisablePipeline(Optional<bool> value);

  /// Set the `pipeline_initiation_interval` option to the provided value.
  /// If no value is provided the option is deleted.
  LoopOptionsAttrBuilder &
  setPipelineInitiationInterval(Optional<uint64_t> count);

  /// Returns true if any option has been set.
  bool empty() { return options.empty(); }

private:
  template <typename T>
  LoopOptionsAttrBuilder &setOption(LoopOptionCase tag, Optional<T> value);

  friend class LoopOptionsAttr;
  SmallVector<LoopOptionsAttr::OptionValuePair> options;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
