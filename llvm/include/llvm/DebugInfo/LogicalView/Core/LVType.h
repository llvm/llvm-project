//===-- LVType.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVType class, which is used to describe a debug
// information type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVTYPE_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVTYPE_H

#include "llvm/DebugInfo/LogicalView/Core/LVElement.h"

namespace llvm {
namespace logicalview {

enum class LVTypeKind {
  IsBase,
  IsConst,
  IsEnumerator,
  IsImport,
  IsImportDeclaration,
  IsImportModule,
  IsPointer,
  IsPointerMember,
  IsReference,
  IsRestrict,
  IsRvalueReference,
  IsSubrange,
  IsTemplateParam,
  IsTemplateTemplateParam,
  IsTemplateTypeParam,
  IsTemplateValueParam,
  IsTypedef,
  IsUnaligned,
  IsUnspecified,
  IsVolatile,
  IsModifier, // CodeView - LF_MODIFIER
  LastEntry
};
using LVTypeKindSelection = std::set<LVTypeKind>;

// Class to represent a DWARF Type.
class LVType : public LVElement {
  enum class Property { IsSubrangeCount, LastEntry };

  // Typed bitvector with kinds and properties for this type.
  LVProperties<LVTypeKind> Kinds;
  LVProperties<Property> Properties;

public:
  LVType() : LVElement(LVSubclassID::LV_TYPE) { setIsType(); }
  LVType(const LVType &) = delete;
  LVType &operator=(const LVType &) = delete;
  virtual ~LVType() = default;

  static bool classof(const LVElement *Element) {
    return Element->getSubclassID() == LVSubclassID::LV_TYPE;
  }

  KIND(LVTypeKind, IsBase);
  KIND(LVTypeKind, IsConst);
  KIND(LVTypeKind, IsEnumerator);
  KIND(LVTypeKind, IsImport);
  KIND_1(LVTypeKind, IsImportDeclaration, IsImport);
  KIND_1(LVTypeKind, IsImportModule, IsImport);
  KIND(LVTypeKind, IsPointer);
  KIND(LVTypeKind, IsPointerMember);
  KIND(LVTypeKind, IsReference);
  KIND(LVTypeKind, IsRestrict);
  KIND(LVTypeKind, IsRvalueReference);
  KIND(LVTypeKind, IsSubrange);
  KIND(LVTypeKind, IsTemplateParam);
  KIND_1(LVTypeKind, IsTemplateTemplateParam, IsTemplateParam);
  KIND_1(LVTypeKind, IsTemplateTypeParam, IsTemplateParam);
  KIND_1(LVTypeKind, IsTemplateValueParam, IsTemplateParam);
  KIND(LVTypeKind, IsTypedef);
  KIND(LVTypeKind, IsUnaligned);
  KIND(LVTypeKind, IsUnspecified);
  KIND(LVTypeKind, IsVolatile);
  KIND(LVTypeKind, IsModifier);

  PROPERTY(Property, IsSubrangeCount);

  const char *kind() const override;

  // Follow a chain of references given by DW_AT_abstract_origin and/or
  // DW_AT_specification and update the type name.
  StringRef resolveReferencesChain();

  bool isBase() const override { return getIsBase(); }
  bool isTemplateParam() const override { return getIsTemplateParam(); }

  // Encode the specific template argument.
  virtual void encodeTemplateArgument(std::string &Name) const {}

  // Return the underlying type for a type definition.
  virtual LVElement *getUnderlyingType() { return nullptr; }
  virtual void setUnderlyingType(LVElement *Element) {}

  void resolveName() override;
  void resolveReferences() override;

  void print(raw_ostream &OS, bool Full = true) const override;
  void printExtra(raw_ostream &OS, bool Full = true) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const override { print(dbgs()); }
#endif
};

// Class to represent DW_TAG_typedef_type.
class LVTypeDefinition final : public LVType {
public:
  LVTypeDefinition() : LVType() {
    setIsTypedef();
    setIncludeInPrint();
  }
  LVTypeDefinition(const LVTypeDefinition &) = delete;
  LVTypeDefinition &operator=(const LVTypeDefinition &) = delete;
  ~LVTypeDefinition() = default;

  // Return the underlying type for a type definition.
  LVElement *getUnderlyingType() override;
  void setUnderlyingType(LVElement *Element) override { setType(Element); }

  void resolveExtra() override;

  void printExtra(raw_ostream &OS, bool Full = true) const override;
};

// Class to represent a DW_TAG_enumerator.
class LVTypeEnumerator final : public LVType {
  // Index in the String pool representing any initial value.
  size_t ValueIndex = 0;

public:
  LVTypeEnumerator() : LVType() {
    setIsEnumerator();
    setIncludeInPrint();
  }
  LVTypeEnumerator(const LVTypeEnumerator &) = delete;
  LVTypeEnumerator &operator=(const LVTypeEnumerator &) = delete;
  ~LVTypeEnumerator() = default;

  // Process the values for a DW_TAG_enumerator.
  std::string getValue() const override {
    return std::string(getStringPool().getString(ValueIndex));
  }
  void setValue(StringRef Value) override {
    ValueIndex = getStringPool().getIndex(Value);
  }
  size_t getValueIndex() const override { return ValueIndex; }

  void printExtra(raw_ostream &OS, bool Full = true) const override;
};

// Class to represent DW_TAG_imported_module / DW_TAG_imported_declaration.
class LVTypeImport final : public LVType {
public:
  LVTypeImport() : LVType() { setIncludeInPrint(); }
  LVTypeImport(const LVTypeImport &) = delete;
  LVTypeImport &operator=(const LVTypeImport &) = delete;
  ~LVTypeImport() = default;

  void printExtra(raw_ostream &OS, bool Full = true) const override;
};

// Class to represent a DWARF Template parameter holder (type or param).
class LVTypeParam final : public LVType {
  // Index in the String pool representing any initial value.
  size_t ValueIndex = 0;

public:
  LVTypeParam();
  LVTypeParam(const LVTypeParam &) = delete;
  LVTypeParam &operator=(const LVTypeParam &) = delete;
  ~LVTypeParam() = default;

  // Template parameter value.
  std::string getValue() const override {
    return std::string(getStringPool().getString(ValueIndex));
  }
  void setValue(StringRef Value) override {
    ValueIndex = getStringPool().getIndex(Value);
  }
  size_t getValueIndex() const override { return ValueIndex; }

  // Encode the specific template argument.
  void encodeTemplateArgument(std::string &Name) const override;

  void printExtra(raw_ostream &OS, bool Full = true) const override;
};

// Class to represent a DW_TAG_subrange_type.
class LVTypeSubrange final : public LVType {
  // Values describing the subrange bounds.
  int64_t LowerBound = 0; // DW_AT_lower_bound or DW_AT_count value.
  int64_t UpperBound = 0; // DW_AT_upper_bound value.

public:
  LVTypeSubrange() : LVType() {
    setIsSubrange();
    setIncludeInPrint();
  }
  LVTypeSubrange(const LVTypeSubrange &) = delete;
  LVTypeSubrange &operator=(const LVTypeSubrange &) = delete;
  ~LVTypeSubrange() = default;

  int64_t getCount() const override {
    return getIsSubrangeCount() ? LowerBound : 0;
  }
  void setCount(int64_t Value) override {
    LowerBound = Value;
    setIsSubrangeCount();
  }

  int64_t getLowerBound() const override { return LowerBound; }
  void setLowerBound(int64_t Value) override { LowerBound = Value; }

  int64_t getUpperBound() const override { return UpperBound; }
  void setUpperBound(int64_t Value) override { UpperBound = Value; }

  std::pair<unsigned, unsigned> getBounds() const override {
    return {LowerBound, UpperBound};
  }
  void setBounds(unsigned Lower, unsigned Upper) override {
    LowerBound = Lower;
    UpperBound = Upper;
  }

  void resolveExtra() override;

  void printExtra(raw_ostream &OS, bool Full = true) const override;
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVTYPE_H
