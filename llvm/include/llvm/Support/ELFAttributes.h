//===-- ELFAttributes.h - ELF Attributes ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ELFATTRIBUTES_H
#define LLVM_SUPPORT_ELFATTRIBUTES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

// Tag to string: ELF compact build attribute section
struct TagNameItem {
  unsigned attr;
  StringRef tagName;
};

using TagNameMap = ArrayRef<TagNameItem>;

// Build Attribute storage for ELF extended attribute section
struct BuildAttributeItem {
  enum Types : uint8_t {
    NumericAttribute = 0,
    TextAttribute,
  } Type;
  unsigned Tag;
  unsigned IntValue;
  std::string StringValue;
  BuildAttributeItem(Types Ty, unsigned Tg, unsigned IV, std::string SV)
      : Type(Ty), Tag(Tg), IntValue(IV), StringValue(std::move(SV)) {}
};
struct BuildAttributeSubSection {
  std::string Name;
  unsigned IsOptional;
  unsigned ParameterType;
  SmallVector<BuildAttributeItem, 64> Content;
};

// Tag to string: ELF extended build attribute section
struct SubsectionAndTagToTagName {
  StringRef SubsectionName;
  unsigned Tag;
  StringRef TagName;
};

namespace ELFAttrs {

enum AttrType : unsigned { File = 1, Section = 2, Symbol = 3 };

LLVM_ABI StringRef attrTypeAsString(unsigned attr, TagNameMap tagNameMap,
                                    bool hasTagPrefix = true);
LLVM_ABI std::optional<unsigned> attrTypeFromString(StringRef tag,
                                                    TagNameMap tagNameMap);

// Magic numbers for ELF attributes.
enum AttrMagic { Format_Version = 0x41 };

} // namespace ELFAttrs
} // namespace llvm
#endif
