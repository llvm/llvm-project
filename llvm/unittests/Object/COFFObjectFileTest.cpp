//===- COFFObjectFileTest.cpp - Tests for COFFObjectFile ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(COFFObjectFileTest, CHPERangeEntry) {
  chpe_range_entry range;

  range.StartOffset = 0x1000;
  EXPECT_EQ(range.getStart(), 0x1000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Arm64);

  range.StartOffset = 0x2000 | chpe_range_type::Arm64EC;
  EXPECT_EQ(range.getStart(), 0x2000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Arm64EC);

  range.StartOffset = 0x3000 | chpe_range_type::Amd64;
  EXPECT_EQ(range.getStart(), 0x3000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Amd64);
}

namespace {
struct TestSection {
  bool Comdat;
  const char *Selection; // For a COMDAT section, the selection kind.
  int Associated;        // For an associative section, the associated one.
};

// A COFF object with one section, and its definition symbol, per entry.
std::string makeCOFFYAML(ArrayRef<TestSection> Sections) {
  std::string Secs, Syms;
  for (auto [I, S] : enumerate(Sections)) {
    std::string Name = ".text$" + std::to_string(I + 1);
    Secs += "  - Name: " + Name + "\n    Characteristics: [ " +
            (S.Comdat ? "IMAGE_SCN_LNK_COMDAT, " : "") +
            "IMAGE_SCN_MEM_READ ]\n    SectionData: C3\n";
    Syms += "  - Name: " + Name +
            "\n    Value: 0\n    SectionNumber: " + std::to_string(I + 1) +
            "\n    SimpleType: IMAGE_SYM_TYPE_NULL\n"
            "    ComplexType: IMAGE_SYM_DTYPE_NULL\n"
            "    StorageClass: IMAGE_SYM_CLASS_STATIC\n"
            "    SectionDefinition:\n      Length: 1\n"
            "      NumberOfRelocations: 0\n      NumberOfLinenumbers: 0\n"
            "      CheckSum: 0\n      Number: " +
            std::to_string(S.Associated) + "\n";
    if (S.Comdat)
      Syms += std::string("      Selection: ") + S.Selection + "\n";
  }
  return "--- !COFF\nheader:\n  Machine: IMAGE_FILE_MACHINE_AMD64\n"
         "  Characteristics: [ ]\nsections:\n" +
         Secs + "symbols:\n" + Syms + "...\n";
}
} // namespace

TEST(COFFObjectFileTest, ComdatSections) {
  const char *Assoc = "IMAGE_COMDAT_SELECT_ASSOCIATIVE";
  std::string YAML = makeCOFFYAML({
      {false, nullptr, 0},                           // 1: not COMDAT
      {true, Assoc, 3},                              // 2: before its leader
      {true, "IMAGE_COMDAT_SELECT_ANY", 0},          // 3
      {true, Assoc, 2},                              // 4: chain via 2
      {true, Assoc, 1},                              // 5: non-COMDAT root
      {true, Assoc, 6},                              // 6: cycle
      {true, Assoc, 99},                             // 7: no such section
      {true, "IMAGE_COMDAT_SELECT_NODUPLICATES", 0}, // 8
      {true, Assoc, 10},                             // 9: cycle via 10
      {true, Assoc, 9},                              // 10
  });
  SmallString<0> Storage;
  std::unique_ptr<ObjectFile> Obj =
      yaml::yaml2ObjectFile(Storage, YAML, [](const Twine &) {});
  ASSERT_TRUE(Obj);
  auto C = COFFComdatSections::create(*cast<COFFObjectFile>(Obj.get()));
  ASSERT_THAT_EXPECTED(C, Succeeded());

  EXPECT_EQ(C->getSelection(1), std::nullopt);
  EXPECT_EQ(C->getSelection(2), COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE);
  EXPECT_EQ(C->getSelection(3), COFF::IMAGE_COMDAT_SELECT_ANY);
  EXPECT_EQ(C->getSelection(8), COFF::IMAGE_COMDAT_SELECT_NODUPLICATES);
  EXPECT_EQ(C->getSelection(0), std::nullopt);
  EXPECT_EQ(C->getSelection(11), std::nullopt);

  EXPECT_EQ(C->getAssociativeRoot(2), 3);
  EXPECT_EQ(C->getAssociativeRoot(4), 3);
  EXPECT_EQ(C->getAssociativeRoot(5), 1);
  EXPECT_EQ(C->getAssociativeRoot(6), std::nullopt);
  EXPECT_EQ(C->getAssociativeRoot(7), std::nullopt);
  EXPECT_EQ(C->getAssociativeRoot(3), std::nullopt);
  EXPECT_EQ(C->getAssociativeRoot(1), std::nullopt);
  EXPECT_EQ(C->getAssociativeRoot(9), std::nullopt);
  EXPECT_EQ(C->getAssociativeRoot(11), std::nullopt);
}
