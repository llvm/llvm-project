//===- DLTI.h - Data Layout and Target Info MLIR Dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the dialect containing the objects pertaining to target information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_DLTI_H
#define MLIR_DIALECT_DLTI_DLTI_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace impl {
class DataLayoutEntryStorage;
class DataLayoutSpecStorage;
} // namespace impl

//===----------------------------------------------------------------------===//
// DataLayoutEntryAttr
//===----------------------------------------------------------------------===//

/// A data layout entry attribute is a key-value pair where the key is a type or
/// an identifier and the value is another attribute. These entries form a data
/// layout specification.
class DataLayoutEntryAttr
    : public Attribute::AttrBase<DataLayoutEntryAttr, Attribute,
                                 impl::DataLayoutEntryStorage,
                                 DataLayoutEntryInterface::Trait> {
public:
  using Base::Base;

  /// The keyword used for this attribute in custom syntax.
  constexpr const static llvm::StringLiteral kAttrKeyword = "dl_entry";

  /// Returns the entry with the given key and value.
  static DataLayoutEntryAttr get(StringAttr key, Attribute value);
  static DataLayoutEntryAttr get(Type key, Attribute value);

  /// Returns the key of this entry.
  DataLayoutEntryKey getKey() const;

  /// Returns the value of this entry.
  Attribute getValue() const;

  /// Parses an instance of this attribute.
  static DataLayoutEntryAttr parse(AsmParser &parser);

  /// Prints this attribute.
  void print(AsmPrinter &os) const;

  static constexpr StringLiteral name = "builtin.data_layout_entry";
};

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//

/// A data layout specification is a list of entries that specify (partial) data
/// layout information. It is expected to be attached to operations that serve
/// as scopes for data layout requests.
class DataLayoutSpecAttr
    : public Attribute::AttrBase<DataLayoutSpecAttr, Attribute,
                                 impl::DataLayoutSpecStorage,
                                 DataLayoutSpecInterface::Trait> {
public:
  using Base::Base;

  /// The keyword used for this attribute in custom syntax.
  constexpr const static StringLiteral kAttrKeyword = "dl_spec";

  /// Returns the specification containing the given list of keys.
  static DataLayoutSpecAttr get(MLIRContext *ctx,
                                ArrayRef<DataLayoutEntryInterface> entries);

  /// Returns the specification containing the given list of keys. If the list
  /// contains duplicate keys or is otherwise invalid, reports errors using the
  /// given callback and returns null.
  static DataLayoutSpecAttr
  getChecked(function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
             ArrayRef<DataLayoutEntryInterface> entries);

  /// Checks that the given list of entries does not contain duplicate keys.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<DataLayoutEntryInterface> entries);

  /// Combines this specification with `specs`, enclosing specifications listed
  /// from outermost to innermost. This overwrites the older entries with the
  /// same key as the newer entries if the entries are compatible. Returns null
  /// if the specifications are not compatible.
  DataLayoutSpecAttr combineWith(ArrayRef<DataLayoutSpecInterface> specs) const;

  /// Returns the list of entries.
  DataLayoutEntryListRef getEntries() const;

  /// Returns the alloca memory space identifier.
  StringAttr getAllocaMemorySpaceIdentifier(MLIRContext *context) const;

  /// Returns the stack alignment identifier.
  StringAttr getStackAlignmentIdentifier(MLIRContext *context) const;

  /// Parses an instance of this attribute.
  static DataLayoutSpecAttr parse(AsmParser &parser);

  /// Prints this attribute.
  void print(AsmPrinter &os) const;

  static constexpr StringLiteral name = "builtin.data_layout_spec";
};

} // namespace mlir

#include "mlir/Dialect/DLTI/DLTIDialect.h.inc"

#endif // MLIR_DIALECT_DLTI_DLTI_H
