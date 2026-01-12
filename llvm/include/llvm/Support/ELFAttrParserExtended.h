//===- ELF AttributeParser.h - ELF Attribute Parser -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ELFEXTENDEDATTRPARSER_H
#define LLVM_SUPPORT_ELFEXTENDEDATTRPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/ELFAttributeParser.h"
#include "llvm/Support/ELFAttributes.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <vector>

namespace llvm {
class StringRef;
class ScopedPrinter;

class LLVM_ABI ELFExtendedAttrParser : public ELFAttributeParser {
protected:
  ScopedPrinter *Sw;
  DataExtractor De{ArrayRef<uint8_t>{}, true, 0};
  DataExtractor::Cursor Cursor{0};

  // Data structure for holding Extended ELF Build Attribute subsection
  SmallVector<BuildAttributeSubSection, 8> SubSectionVec;
  // Maps SubsectionName + Tag to tags names. Required for printing comments.
  const std::vector<SubsectionAndTagToTagName> TagsNamesMap;
  StringRef getTagName(const StringRef &BuildAttrSubsectionName,
                       const unsigned Tag);

public:
  ~ELFExtendedAttrParser() override { static_cast<void>(!Cursor.takeError()); }
  Error parse(ArrayRef<uint8_t> Section, llvm::endianness Endian) override;

  std::optional<unsigned> getAttributeValue(unsigned Tag) const override;
  std::optional<unsigned> getAttributeValue(StringRef BuildAttrSubsectionName,
                                            unsigned Tag) const override;
  std::optional<StringRef> getAttributeString(unsigned Tag) const override;
  std::optional<StringRef> getAttributeString(StringRef BuildAttrSubsectionName,
                                              unsigned Tag) const override;

  ELFExtendedAttrParser(
      ScopedPrinter *Sw,
      const std::vector<SubsectionAndTagToTagName> TagsNamesMap)
      : Sw(Sw), TagsNamesMap(TagsNamesMap) {}
  ELFExtendedAttrParser(
      const std::vector<SubsectionAndTagToTagName> TagsNamesMap)
      : Sw(nullptr), TagsNamesMap(TagsNamesMap) {}
};
} // namespace llvm
#endif // LLVM_SUPPORT_ELFEXTENDEDATTRPARSER_H
