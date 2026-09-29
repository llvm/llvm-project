//===- EnumInfo.h - EnumInfo wrapper class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EnumInfo wrapper to simplify using a TableGen Record defining an Enum
// via EnumInfo and its `EnumCase`s.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ENUMINFO_H_
#define MLIR_TABLEGEN_ENUMINFO_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Attribute.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // namespace llvm

namespace mlir::tblgen {

// Wrapper class providing around enum cases defined in TableGen.
class EnumCase {
public:
  explicit EnumCase(const llvm::Record *record);
  explicit EnumCase(const llvm::DefInit *init);

  // Returns the symbol of this enum attribute case.
  StringRef getSymbol() const;

  // Returns the textual representation of this enum attribute case.
  StringRef getStr() const;

  // Returns the value of this enum attribute case.
  int64_t getValue() const;

  // Returns the TableGen definition this EnumAttrCase was constructed from.
  const llvm::Record &getDef() const;

protected:
  // The TableGen definition of this constraint.
  const llvm::Record *def;
};

// Wrapper class providing helper methods for accessing enums defined
// in TableGen using EnumInfo. Some methods are only applicable when
// the enum is also an attribute, or only when it is a bit enum.
class EnumInfo {
public:
  explicit EnumInfo(const llvm::Record *record);
  explicit EnumInfo(const llvm::Record &record);
  explicit EnumInfo(const llvm::DefInit *init);

  // Returns true if the given EnumInfo is a subclass of the named TableGen
  // class.
  bool isSubClassOf(StringRef className) const;

  // Returns true if this enum is an EnumAttrInfo, thus making it define an
  // attribute.
  bool isEnumAttr() const;

  // Create the `Attribute` wrapper around this EnumInfo if it is defining an
  // attribute.
  std::optional<Attribute> asEnumAttr() const;

  // Returns true if this is a bit enum.
  bool isBitEnum() const;

  // Returns the enum class name.
  StringRef getEnumClassName() const;

  // Returns the C++ namespaces this enum class should be placed in.
  StringRef getCppNamespace() const;

  // Returns the summary of the enum.
  StringRef getSummary() const;

  // Returns the description of the enum.
  StringRef getDescription() const;

  // Returns the bitwidth of the enum.
  int64_t getBitwidth() const;

  // Returns the underlying type.
  StringRef getUnderlyingType() const;

  // Returns the name of the utility function that converts a value of the
  // underlying type to the corresponding symbol.
  StringRef getUnderlyingToSymbolFnName() const;

  // Returns the name of the utility function that converts a string to the
  // corresponding symbol.
  StringRef getStringToSymbolFnName() const;

  // Returns the name of the utility function that converts a symbol to the
  // corresponding string.
  StringRef getSymbolToStringFnName() const;

  // Returns the return type of the utility function that converts a symbol to
  // the corresponding string.
  StringRef getSymbolToStringFnRetType() const;

  // Returns the name of the utilit function that returns the max enum value
  // used within the enum class.
  StringRef getMaxEnumValFnName() const;

  // Returns all allowed cases for this enum attribute.
  std::vector<EnumCase> getAllCases() const;

  // Only applicable for enum attributes.

  bool genSpecializedAttr() const;
  const llvm::Record *getBaseAttrClass() const;
  StringRef getSpecializedAttrClassName() const;

  // Only applicable for bit enums.

  bool printBitEnumPrimaryGroups() const;
  bool printBitEnumQuoted() const;

  // Returns the TableGen definition this EnumAttrCase was constructed from.
  const llvm::Record &getDef() const;

protected:
  // The TableGen definition of this constraint.
  const llvm::Record *def;
};

} // namespace mlir::tblgen

#endif
