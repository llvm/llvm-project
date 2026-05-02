//===- ELF AttributeParser.h - ELF Attribute Parser -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ELFCOMPACTATTRPARSER_H
#define LLVM_SUPPORT_ELFCOMPACTATTRPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/ELFAttributeParser.h"
#include "llvm/Support/ELFAttributes.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <unordered_map>

namespace llvm {
class StringRef;
class ScopedPrinter;

class LLVM_ABI ELFCompactAttrParser : public ELFAttributeParser {
  StringRef vendor;
  std::unordered_map<unsigned, unsigned> attributes;
  std::unordered_map<unsigned, StringRef> attributesStr;

  virtual Error handler(uint64_t tag, bool &handled) = 0;

protected:
  ScopedPrinter *sw;
  TagNameMap tagToStringMap;
  DataExtractor de{ArrayRef<uint8_t>{}, true, 0};
  DataExtractor::Cursor cursor{0};

  void printAttribute(unsigned tag, unsigned value, StringRef valueDesc);

  Error parseStringAttribute(const char *name, unsigned tag,
                             ArrayRef<const char *> strings);
  Error parseAttributeList(uint32_t length);
  void parseIndexList(SmallVectorImpl<uint8_t> &indexList);
  Error parseSubsection(uint32_t length);

  void setAttributeString(unsigned tag, StringRef value) {
    attributesStr.emplace(tag, value);
  }

public:
  ~ELFCompactAttrParser() override { static_cast<void>(!cursor.takeError()); }
  Error integerAttribute(unsigned tag);
  Error stringAttribute(unsigned tag);

  ELFCompactAttrParser(ScopedPrinter *sw, TagNameMap tagNameMap,
                       StringRef vendor)
      : vendor(vendor), sw(sw), tagToStringMap(tagNameMap) {}
  ELFCompactAttrParser(TagNameMap tagNameMap, StringRef vendor)
      : vendor(vendor), sw(nullptr), tagToStringMap(tagNameMap) {}

  Error parse(ArrayRef<uint8_t> section, llvm::endianness endian) override;

  std::optional<unsigned> getAttributeValue(unsigned tag) const override {
    auto I = attributes.find(tag);
    if (I == attributes.end())
      return std::nullopt;
    return I->second;
  }
  std::optional<unsigned>
  getAttributeValue(StringRef buildAttributeSubsectionName,
                    unsigned tag) const override {
    assert("" == buildAttributeSubsectionName &&
           "buildAttributeSubsectionName must be an empty string");
    return getAttributeValue(tag);
  }
  std::optional<StringRef> getAttributeString(unsigned tag) const override {
    auto I = attributesStr.find(tag);
    if (I == attributesStr.end())
      return std::nullopt;
    return I->second;
  }
  std::optional<StringRef>
  getAttributeString(StringRef buildAttributeSubsectionName,
                     unsigned tag) const override {
    assert("" == buildAttributeSubsectionName &&
           "buildAttributeSubsectionName must be an empty string");
    return getAttributeString(tag);
  }
};

} // namespace llvm
#endif // LLVM_SUPPORT_ELFCOMPACTATTRPARSER_H
