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

/// This class represents a LLVM attribute that describes a local debug info
/// scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;

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

namespace detail {
class DistinctSequenceAttrStorage;
} // namespace detail

/// This class is a helper attribute to generate a sequence of unique
/// identifiers that can be used to model distinct metadata nodes. The
/// attribute has a scope that limits the validity of the generated sequence to
/// a function since generating unique identifiers at the module level could
/// lead to non determinism due to the parallel processing of functions. A
/// mutable state can be incremented to generate the next unique identifier.
///
/// Example:
/// ```
/// #distinct_sequence = #llvm.distinct_sequence<scope = @foo, state = 2>
/// #access_group = #llvm.access_group<id = 0, elem_of = #sequence>
/// #access_group1 = #llvm.access_group<id = 1, elem_of = #sequence>
///
/// llvm.func @foo(%arg0: !llvm.ptr) {
///   %0 = llvm.load %arg0 {access_groups = [#access_group, #access_group1]}
/// }
/// ```
class DistinctSequenceAttr
    : public Attribute::AttrBase<DistinctSequenceAttr, Attribute,
                                 detail::DistinctSequenceAttrStorage,
                                 AttributeTrait::IsMutable> {
public:
  // Inherit Base constructors.
  using Base::Base;

  /// Returns a distinct sequences attribute for the given scope.
  static DistinctSequenceAttr get(SymbolRefAttr scope);

  /// Returns the keyword used when printing and parsing the attribute.
  static constexpr StringLiteral getMnemonic() { return {"distinct_sequence"}; }

  /// Returns the symbol that limits the scope of the sequence.
  SymbolRefAttr getScope() const;

  /// Returns the next identifier without incrementing the mutable state.
  int64_t getState() const;

  /// Returns the next identifier and increments the mutable state.
  int64_t getNextID();

  /// Parses an instance of this attribute.
  static Attribute parse(AsmParser &parser, Type type);

  /// Prints this attribute.
  void print(AsmPrinter &os) const;
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
/// Verifies the access groups attached to the given operation.
LogicalResult verifyAccessGroups(Operation *op,
                                 ArrayRef<AccessGroupAttr> accessGroups);
} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
