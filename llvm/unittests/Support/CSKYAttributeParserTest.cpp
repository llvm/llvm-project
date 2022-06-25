//===----- unittests/CSKYAttributeParserTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/CSKYAttributeParser.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/ELFAttributes.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

struct CSKYAttributeSection {
  unsigned Tag;
  struct {
    unsigned IntValue;
    const char *StringValue;
  } Value;

  CSKYAttributeSection(unsigned tag, unsigned value) : Tag(tag) {
    Value.IntValue = value;
  }

  CSKYAttributeSection(unsigned tag, const char *value) : Tag(tag) {
    Value.StringValue = value;
  }

  void writeInt(raw_ostream &OS) {
    OS.flush();
    // Format version.
    OS << 'A'
       // uint32_t = VendorHeaderSize + TagHeaderSize + ContentsSize.
       << (uint8_t)16 << (uint8_t)0 << (uint8_t)0
       << (uint8_t)0
       // CurrentVendor.
       << "csky"
       << '\0'
       // ELFAttrs::File.
       << (uint8_t)1
       // uint32_t = TagHeaderSize + ContentsSize.
       << (uint8_t)6 << (uint8_t)0 << (uint8_t)0
       << (uint8_t)0
       // Tag
       << (uint8_t)Tag
       // IntValue
       << (uint8_t)Value.IntValue;
  }

  void writeString(raw_ostream &OS) {
    OS.flush();
    // Format version.
    OS << 'A'
       // uint32_t = VendorHeaderSize + TagHeaderSize + ContentsSize.
       << (uint8_t)(16 + strlen(Value.StringValue)) << (uint8_t)0 << (uint8_t)0
       << (uint8_t)0
       // CurrentVendor.
       << "csky"
       << '\0'
       // ELFAttrs::File.
       << (uint8_t)1
       // uint32_t = TagHeaderSize + ContentsSize.
       << (uint8_t)(6 + strlen(Value.StringValue)) << (uint8_t)0 << (uint8_t)0
       << (uint8_t)0
       // Tag
       << (uint8_t)Tag
       // StringValue
       << Value.StringValue << '\0';
  }
};

static bool testAttributeInt(unsigned Tag, unsigned Value, unsigned ExpectedTag,
                             unsigned ExpectedValue) {
  std::string buffer;
  raw_string_ostream OS(buffer);
  CSKYAttributeSection Section(Tag, Value);
  Section.writeInt(OS);
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(OS.str().c_str()),
                          OS.str().size());

  CSKYAttributeParser Parser;
  cantFail(Parser.parse(Bytes, support::little));

  Optional<unsigned> Attr = Parser.getAttributeValue(ExpectedTag);
  return Attr && *Attr == ExpectedValue;
}

static bool testAttributeString(unsigned Tag, const char *Value,
                                unsigned ExpectedTag,
                                const char *ExpectedValue) {
  std::string buffer;
  raw_string_ostream OS(buffer);
  CSKYAttributeSection Section(Tag, Value);
  Section.writeString(OS);
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(OS.str().c_str()),
                          OS.str().size());

  CSKYAttributeParser Parser;
  cantFail(Parser.parse(Bytes, support::little));

  Optional<StringRef> Attr = Parser.getAttributeString(ExpectedTag);
  return Attr && *Attr == ExpectedValue;
}

static void testParseError(unsigned Tag, unsigned Value, const char *msg) {
  std::string buffer;
  raw_string_ostream OS(buffer);
  CSKYAttributeSection Section(Tag, Value);
  Section.writeInt(OS);
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(OS.str().c_str()),
                          OS.str().size());

  CSKYAttributeParser Parser;
  Error e = Parser.parse(Bytes, support::little);
  EXPECT_STREQ(toString(std::move(e)).c_str(), msg);
}

static bool testTagString(unsigned Tag, const char *name) {
  return ELFAttrs::attrTypeAsString(Tag, CSKYAttrs::getCSKYAttributeTags())
             .str() == name;
}

TEST(ArchName, testAttribute) {
  EXPECT_TRUE(testTagString(4, "Tag_CSKY_ARCH_NAME"));
  EXPECT_TRUE(
      testAttributeString(4, "ck860", CSKYAttrs::CSKY_ARCH_NAME, "ck860"));
  EXPECT_FALSE(
      testAttributeString(4, "ck86", CSKYAttrs::CSKY_ARCH_NAME, "ck60"));
}

TEST(CPUName, testAttribute) {
  EXPECT_TRUE(testTagString(5, "Tag_CSKY_CPU_NAME"));
  EXPECT_TRUE(
      testAttributeString(5, "ck860fv", CSKYAttrs::CSKY_CPU_NAME, "ck860fv"));
  EXPECT_FALSE(
      testAttributeString(5, "ck860", CSKYAttrs::CSKY_CPU_NAME, "ck860fv"));
}

TEST(DSPVersion, testAttribute) {
  EXPECT_TRUE(testTagString(8, "Tag_CSKY_DSP_VERSION"));
  EXPECT_TRUE(testAttributeInt(8, 1, CSKYAttrs::CSKY_DSP_VERSION,
                               CSKYAttrs::DSP_VERSION_EXTENSION));
  EXPECT_TRUE(testAttributeInt(8, 2, CSKYAttrs::CSKY_DSP_VERSION,
                               CSKYAttrs::DSP_VERSION_2));
  EXPECT_FALSE(testAttributeInt(8, 0, CSKYAttrs::CSKY_DSP_VERSION,
                                CSKYAttrs::DSP_VERSION_EXTENSION));
  testParseError(8, 3, "unknown Tag_CSKY_DSP_VERSION value: 3");
}

TEST(VDSPVersion, testAttribute) {
  EXPECT_TRUE(testTagString(9, "Tag_CSKY_VDSP_VERSION"));
  EXPECT_TRUE(testAttributeInt(9, 1, CSKYAttrs::CSKY_VDSP_VERSION,
                               CSKYAttrs::VDSP_VERSION_1));
  EXPECT_TRUE(testAttributeInt(9, 2, CSKYAttrs::CSKY_VDSP_VERSION,
                               CSKYAttrs::VDSP_VERSION_2));
  EXPECT_FALSE(testAttributeInt(9, 0, CSKYAttrs::CSKY_VDSP_VERSION,
                                CSKYAttrs::VDSP_VERSION_2));
  testParseError(9, 3, "unknown Tag_CSKY_VDSP_VERSION value: 3");
}

TEST(FPUVersion, testAttribute) {
  EXPECT_TRUE(testTagString(16, "Tag_CSKY_FPU_VERSION"));
  EXPECT_TRUE(testAttributeInt(16, 1, CSKYAttrs::CSKY_FPU_VERSION,
                               CSKYAttrs::FPU_VERSION_1));
  EXPECT_TRUE(testAttributeInt(16, 2, CSKYAttrs::CSKY_FPU_VERSION,
                               CSKYAttrs::FPU_VERSION_2));
  EXPECT_TRUE(testAttributeInt(16, 3, CSKYAttrs::CSKY_FPU_VERSION,
                               CSKYAttrs::FPU_VERSION_3));
  EXPECT_FALSE(testAttributeInt(16, 0, CSKYAttrs::CSKY_FPU_VERSION,
                                CSKYAttrs::FPU_VERSION_3));
  testParseError(16, 4, "unknown Tag_CSKY_FPU_VERSION value: 4");
}

TEST(FPUABI, testAttribute) {
  EXPECT_TRUE(testTagString(17, "Tag_CSKY_FPU_ABI"));
  EXPECT_TRUE(testAttributeInt(17, 1, CSKYAttrs::CSKY_FPU_ABI,
                               CSKYAttrs::FPU_ABI_SOFT));
  EXPECT_TRUE(testAttributeInt(17, 2, CSKYAttrs::CSKY_FPU_ABI,
                               CSKYAttrs::FPU_ABI_SOFTFP));
  EXPECT_TRUE(testAttributeInt(17, 3, CSKYAttrs::CSKY_FPU_ABI,
                               CSKYAttrs::FPU_ABI_HARD));
  EXPECT_FALSE(testAttributeInt(17, 0, CSKYAttrs::CSKY_FPU_ABI,
                                CSKYAttrs::FPU_ABI_HARD));
  testParseError(17, 4, "unknown Tag_CSKY_FPU_ABI value: 4");
}

TEST(FPURounding, testAttribute) {
  EXPECT_TRUE(testTagString(18, "Tag_CSKY_FPU_ROUNDING"));
  EXPECT_TRUE(
      testAttributeInt(18, 0, CSKYAttrs::CSKY_FPU_ROUNDING, CSKYAttrs::NONE));
  EXPECT_TRUE(
      testAttributeInt(18, 1, CSKYAttrs::CSKY_FPU_ROUNDING, CSKYAttrs::NEEDED));
  testParseError(18, 2, "unknown Tag_CSKY_FPU_ROUNDING value: 2");
}

TEST(FPUDenormal, testAttribute) {
  EXPECT_TRUE(testTagString(19, "Tag_CSKY_FPU_DENORMAL"));
  EXPECT_TRUE(
      testAttributeInt(19, 0, CSKYAttrs::CSKY_FPU_DENORMAL, CSKYAttrs::NONE));
  EXPECT_TRUE(
      testAttributeInt(19, 1, CSKYAttrs::CSKY_FPU_DENORMAL, CSKYAttrs::NEEDED));
  testParseError(19, 2, "unknown Tag_CSKY_FPU_DENORMAL value: 2");
}

TEST(FPUException, testAttribute) {
  EXPECT_TRUE(testTagString(20, "Tag_CSKY_FPU_EXCEPTION"));
  EXPECT_TRUE(
      testAttributeInt(20, 0, CSKYAttrs::CSKY_FPU_EXCEPTION, CSKYAttrs::NONE));
  EXPECT_TRUE(testAttributeInt(20, 1, CSKYAttrs::CSKY_FPU_EXCEPTION,
                               CSKYAttrs::NEEDED));
  testParseError(20, 2, "unknown Tag_CSKY_FPU_EXCEPTION value: 2");
}

TEST(FPUNumberModule, testAttribute) {
  EXPECT_TRUE(testTagString(21, "Tag_CSKY_FPU_NUMBER_MODULE"));
  EXPECT_TRUE(testAttributeString(
      21, "IEEE 754", CSKYAttrs::CSKY_FPU_NUMBER_MODULE, "IEEE 754"));
  EXPECT_FALSE(testAttributeString(
      21, "IEEE 755", CSKYAttrs::CSKY_FPU_NUMBER_MODULE, "IEEE 754"));
}

TEST(FPUHardFP, testAttribute) {
  EXPECT_TRUE(testTagString(22, "Tag_CSKY_FPU_HARDFP"));
  EXPECT_TRUE(testAttributeInt(22, 1, CSKYAttrs::CSKY_FPU_HARDFP,
                               CSKYAttrs::FPU_HARDFP_HALF));
  EXPECT_TRUE(testAttributeInt(22, 2, CSKYAttrs::CSKY_FPU_HARDFP,
                               CSKYAttrs::FPU_HARDFP_SINGLE));
  EXPECT_TRUE(testAttributeInt(22, 4, CSKYAttrs::CSKY_FPU_HARDFP,
                               CSKYAttrs::FPU_HARDFP_DOUBLE));
  EXPECT_FALSE(testAttributeInt(22, 3, CSKYAttrs::CSKY_FPU_HARDFP,
                                CSKYAttrs::FPU_HARDFP_DOUBLE));
  testParseError(22, 0, "unknown Tag_CSKY_FPU_HARDFP value: 0");
  testParseError(22, 8, "unknown Tag_CSKY_FPU_HARDFP value: 8");
}
