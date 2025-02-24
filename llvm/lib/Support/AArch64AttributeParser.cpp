//===-AArch64AttributeParser.cpp-AArch64 Attribute Information Printer-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#include "llvm/Support/AArch64AttributeParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AArch64BuildAttributes.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

Error AArch64AttributeParser::parse(ArrayRef<uint8_t> Section,
                                    llvm::endianness Endian) {

  unsigned SectionNumber = 0;
  de = DataExtractor(Section, Endian == llvm::endianness::little, 0);

  // Early returns have specific errors. Consume the Error in cursor.
  struct ClearCursorError {
    DataExtractor::Cursor &Cursor;
    ~ClearCursorError() { consumeError(Cursor.takeError()); }
  } Clear{cursor};

  /*
    AArch64 build attributes layout:
    <format-version: ‘A’> --> There is only one version, 'A' (0x41)
    [ <uint32: subsection-length> <NTBS: vendor-name> <bytes: vendor-data> ]
      --> subsection-length: the offset from the start of this subsection to the
    start of the next one.
      --> vendor-name: NUL-terminated byte string.
      --> vendor-data expands to:
        [ <uint8: optional> <uint8: parameter type> <attribute>*]
          --> optional: 0- required, 1- optional
          --> type: 0- ULEB128, 1- NTBS
          --> attribute: <tag, value>* pair. Tag is ULEB128, value is <parameter
    type> type.
  */

  // Get format-version
  uint8_t FormatVersion = de.getU8(cursor);
  if (ELFAttrs::Format_Version != FormatVersion)
    return createStringError(errc::invalid_argument,
                             "unrecognized format-version: 0x" +
                                 utohexstr(FormatVersion));

  while (!de.eof(cursor)) {
    uint32_t BASubsectionLength = de.getU32(cursor);
    // Minimal valid BA subsection header size is at least 8: length(4) name(at
    // least a single char + null) optionality(1) and type(1)
    if (BASubsectionLength < 8)
      return createStringError(
          errc::invalid_argument,
          "invalid AArch64 build attribute subsection size at offset: " +
              utohexstr(cursor.tell() - 4));

    StringRef VendorName = de.getCStrRef(cursor);
    // The layout of a private subsection (--> vendor name does not starts with
    // 'aeabi') is unknown, skip)
    if (!VendorName.starts_with("aeabi")) {
      sw->startLine()
          << "** Skipping private AArch64 build attributes subsection: "
          << VendorName << "\n";
      // Offset in Section
      uint64_t OffsetInSection = cursor.tell();
      // Size: 4 bytes, Vendor Name: VendorName.size() + 1 (null termination)
      uint32_t BytesForLengthName = 4 + (VendorName.size() + 1);
      cursor.seek(OffsetInSection + BASubsectionLength - BytesForLengthName);
      continue;
    }
    // All public subsections names must be known
    if (VendorName.starts_with("aeabi")) {
      if (!("aeabi_feature_and_bits" == VendorName ||
            "aeabi_pauthabi" == VendorName)) {
        return createStringError(
            errc::invalid_argument,
            "unknown public AArch64 build attribute subsection name at "
            "offset: " +
                utohexstr(cursor.tell() - (VendorName.size() + 1)));
      }
    }

    uint8_t IsOptional = de.getU8(cursor);
    StringRef IsOptionalStr = IsOptional ? "optional" : "required";
    uint8_t Type = de.getU8(cursor);
    StringRef TypeStr = Type ? "ntbs" : "uleb128";

    if (sw) {
      sw->startLine() << "Section " << ++SectionNumber << " {\n";
      sw->indent();
      sw->printNumber("SectionLength", BASubsectionLength);
      sw->startLine() << "VendorName" << ": " << VendorName
                      << " Optionality: " << IsOptionalStr
                      << " Type: " << TypeStr << "\n";
      sw->startLine() << "Attributes {\n";
      sw->indent();
    }

    // Offset in Section
    uint64_t OffsetInSection = cursor.tell();
    // Size: 4 bytes, Vendor Name: VendorName.size() + 1 (null termination),
    // optionality: 1, size: 1
    uint32_t BytesAllButAttributes = 4 + (VendorName.size() + 1) + 1 + 1;
    while (cursor.tell() <
           (OffsetInSection + BASubsectionLength - BytesAllButAttributes)) {

      uint64_t Tag = de.getULEB128(cursor);
      std::string Str = utostr(Tag);
      StringRef TagStr(Str);
      if ("aeabi_feature_and_bits" == VendorName) {
        StringRef TagAsString =
            AArch64BuildAttributes::getFeatureAndBitsTagsStr(Tag);
        if ("" != TagAsString)
          TagStr = TagAsString;
      }
      if ("aeabi_pauthabi" == VendorName) {
        StringRef TagAsString = AArch64BuildAttributes::getPauthABITagsStr(Tag);
        if ("" != TagAsString)
          TagStr = TagAsString;
      }

      if (Type) { // type==1 --> ntbs
        StringRef Value = de.getCStrRef(cursor);
        if (sw)
          sw->printString(TagStr, Value);
      } else { // type==0 --> uleb128
        uint64_t Value = de.getULEB128(cursor);
        if (sw)
          sw->printNumber(TagStr, Value);
      }
    }
    if (sw) {
      // Close 'Attributes'
      sw->unindent();
      sw->startLine() << "}\n";
      // Close 'Section'
      sw->unindent();
      sw->startLine() << "}\n";
    }
  }

  return cursor.takeError();
}
