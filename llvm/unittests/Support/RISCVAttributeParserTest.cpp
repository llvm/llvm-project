//===----- unittests/RISCVAttributeParserTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/RISCVAttributeParser.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/ELFAttributes.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

struct RISCVAttributeSection {
  unsigned Tag;
  unsigned Value;

  RISCVAttributeSection(unsigned tag, unsigned value)
      : Tag(tag), Value(value) {}

  void write(raw_ostream &OS) {
    OS.flush();
    // length = length + "riscv\0" + TagFile + ByteSize + Tag + Value;
    // length = 17 bytes

    OS << 'A' << (uint8_t)17 << (uint8_t)0 << (uint8_t)0 << (uint8_t)0;
    OS << "riscv" << '\0';
    OS << (uint8_t)1 << (uint8_t)7 << (uint8_t)0 << (uint8_t)0 << (uint8_t)0;
    OS << (uint8_t)Tag << (uint8_t)Value;
  }
};

static bool testAttribute(unsigned Tag, unsigned Value, unsigned ExpectedTag,
                          unsigned ExpectedValue) {
  std::string buffer;
  raw_string_ostream OS(buffer);
  RISCVAttributeSection Section(Tag, Value);
  Section.write(OS);
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(OS.str().c_str()),
                          OS.str().size());

  RISCVAttributeParser Parser;
  cantFail(Parser.parse(Bytes, llvm::endianness::little));

  std::optional<unsigned> Attr = Parser.getAttributeValue(ExpectedTag);
  return Attr && *Attr == ExpectedValue;
}

static bool testTagString(unsigned Tag, const char *name) {
  return ELFAttrs::attrTypeAsString(Tag, RISCVAttrs::getRISCVAttributeTags())
             .str() == name;
}

TEST(StackAlign, testAttribute) {
  EXPECT_TRUE(testTagString(4, "Tag_stack_align"));
  EXPECT_TRUE(testAttribute(4, 4, RISCVAttrs::STACK_ALIGN, 4));
  EXPECT_TRUE(testAttribute(4, 16, RISCVAttrs::STACK_ALIGN, 16));
}

TEST(UnalignedAccess, testAttribute) {
  EXPECT_TRUE(testTagString(6, "Tag_unaligned_access"));
  EXPECT_TRUE(testAttribute(6, 0, RISCVAttrs::UNALIGNED_ACCESS,
                            RISCVAttrs::NOT_ALLOWED));
  EXPECT_TRUE(
      testAttribute(6, 1, RISCVAttrs::UNALIGNED_ACCESS, RISCVAttrs::ALLOWED));
}
