//===-ELFAttrParserExtended.cpp-ELF Extended Attribute Information Printer-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#include "llvm/Support/ELFAttrParserExtended.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AArch64BuildAttributes.h"
#include "llvm/Support/ELFAttributes.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;
using namespace ELFAttrs;

std::optional<unsigned>
ELFExtendedAttrParser::getAttributeValue(unsigned Tag) const {
  assert(
      0 &&
      "use getAttributeValue overloaded version accepting Stringref, unsigned");
  return std::nullopt;
}

std::optional<unsigned>
ELFExtendedAttrParser::getAttributeValue(StringRef BuildAttrSubsectionName,
                                         unsigned Tag) const {
  for (const auto &SubSection : SubSectionVec) {
    if (BuildAttrSubsectionName == SubSection.Name)
      for (const auto &BAItem : SubSection.Content) {
        if (Tag == BAItem.Tag)
          return std::optional<unsigned>(BAItem.IntValue);
      }
  }
  return std::nullopt;
}

std::optional<StringRef>
ELFExtendedAttrParser::getAttributeString(unsigned Tag) const {
  assert(
      0 &&
      "use getAttributeValue overloaded version accepting Stringref, unsigned");
  return std::nullopt;
}

std::optional<StringRef>
ELFExtendedAttrParser::getAttributeString(StringRef BuildAttrSubsectionName,
                                          unsigned Tag) const {
  for (const auto &SubSection : SubSectionVec) {
    if (BuildAttrSubsectionName == SubSection.Name)
      for (const auto &BAItem : SubSection.Content) {
        if (Tag == BAItem.Tag)
          return std::optional<StringRef>(BAItem.StringValue);
      }
  }
  return std::nullopt;
}

StringRef
ELFExtendedAttrParser::getTagName(const StringRef &BuildAttrSubsectionName,
                                  const unsigned Tag) {
  for (const auto &Entry : TagsNamesMap) {
    if (BuildAttrSubsectionName == Entry.SubsectionName)
      if (Tag == Entry.Tag)
        return Entry.TagName;
  }
  return "";
}

Error ELFExtendedAttrParser::parse(ArrayRef<uint8_t> Section,
                                   llvm::endianness Endian) {

  unsigned SectionNumber = 0;
  De = DataExtractor(Section, Endian == llvm::endianness::little, 0);

  // Early returns have specific errors. Consume the Error in Cursor.
  struct ClearCursorError {
    DataExtractor::Cursor &Cursor;
    ~ClearCursorError() { consumeError(Cursor.takeError()); }
  } Clear{Cursor};

  /*
      ELF Extended Build Attributes Layout:
      <format-version: ‘A’> --> Currently, there is only one version: 'A' (0x41)
      [ <uint32: subsection-length> <NTBS: vendor-name> <bytes: vendor-data> ]
        --> subsection-length: Offset from the start of this subsection to the
     start of the next one.
        --> vendor-name: Null-terminated byte string.
        --> vendor-data expands to:
          [ <uint8: optional> <uint8: parameter type> <attribute>* ]
            --> optional: 0 = required, 1 = optional.
            --> parameter type: 0 = ULEB128, 1 = NTBS.
            --> attribute: <tag, value>* pair. Tag is ULEB128, value is of
     <parameter type>.
  */

  // Get format-version
  uint8_t FormatVersion = De.getU8(Cursor);
  if (ELFAttrs::Format_Version != FormatVersion)
    return createStringError(errc::invalid_argument,
                             "unrecognized format-version: 0x" +
                                 utohexstr(FormatVersion));

  while (!De.eof(Cursor)) {
    uint32_t ExtBASubsectionLength = De.getU32(Cursor);
    // Minimal valid Extended Build Attributes subsection header size is at
    // least 8: length(4) name(at least a single char + null) optionality(1) and
    // type(1)
    if (ExtBASubsectionLength < 8)
      return createStringError(
          errc::invalid_argument,
          "invalid Extended Build Attributes subsection size at offset: " +
              utohexstr(Cursor.tell() - 4));

    StringRef VendorName = De.getCStrRef(Cursor);
    uint8_t IsOptional = De.getU8(Cursor);
    StringRef IsOptionalStr = IsOptional ? "optional" : "required";
    uint8_t Type = De.getU8(Cursor);
    StringRef TypeStr = Type ? "ntbs" : "uleb128";

    BuildAttributeSubSection BASubSection;
    BASubSection.Name = VendorName;
    BASubSection.IsOptional = IsOptional;
    BASubSection.ParameterType = Type;

    if (Sw) {
      Sw->startLine() << "Section " << ++SectionNumber << " {\n";
      Sw->indent();
      Sw->printNumber("SectionLength", ExtBASubsectionLength);
      Sw->startLine() << "VendorName" << ": " << VendorName
                      << " Optionality: " << IsOptionalStr
                      << " Type: " << TypeStr << "\n";
      Sw->startLine() << "Attributes {\n";
      Sw->indent();
    }

    // Offset in Section
    uint64_t OffsetInSection = Cursor.tell();
    // Size: 4 bytes, Vendor Name: VendorName.size() + 1 (null termination),
    // optionality: 1, size: 1
    uint32_t BytesAllButAttributes = 4 + (VendorName.size() + 1) + 1 + 1;
    while (Cursor.tell() <
           (OffsetInSection + ExtBASubsectionLength - BytesAllButAttributes)) {

      uint64_t Tag = De.getULEB128(Cursor);

      StringRef TagName = getTagName(VendorName, Tag);

      uint64_t ValueInt = 0;
      std::string ValueStr = "";
      if (Type) { // type==1 --> ntbs
        ValueStr = De.getCStrRef(Cursor);
        if (Sw)
          Sw->printString("" != TagName ? TagName : utostr(Tag), ValueStr);
      } else { // type==0 --> uleb128
        ValueInt = De.getULEB128(Cursor);
        if (Sw)
          Sw->printNumber("" != TagName ? TagName : utostr(Tag), ValueInt);
      }

      // populate data structure
      BuildAttributeItem BAItem(static_cast<BuildAttributeItem::Types>(Type),
                                Tag, ValueInt, ValueStr);
      BASubSection.Content.push_back(BAItem);
    }
    if (Sw) {
      // Close 'Attributes'
      Sw->unindent();
      Sw->startLine() << "}\n";
      // Close 'Section'
      Sw->unindent();
      Sw->startLine() << "}\n";
    }

    // populate data structure
    SubSectionVec.push_back(BASubSection);
  }

  return Cursor.takeError();
}
