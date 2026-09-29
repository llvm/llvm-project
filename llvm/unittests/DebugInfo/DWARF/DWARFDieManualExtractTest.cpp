//===-llvm/unittest/DebugInfo/DWARFDieManualExtractTest.cpp---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "DwarfUtils.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dwarf;
using namespace utils;

namespace {

TEST(DWARFDie, manualExtractDump) {
  typedef uint32_t AddrType;
  uint16_t Version = 4;
  Triple Triple = getDefaultTargetTripleForAddrSize(sizeof(AddrType));
  if (!isConfigurationSupported(Triple))
    GTEST_SKIP();

  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &DGCU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = DGCU.getUnitDIE();

  CUDie.addAttribute(DW_AT_name, DW_FORM_strp, "/tmp/main.c");
  CUDie.addAttribute(DW_AT_language, DW_FORM_data2, DW_LANG_C);

  dwarfgen::DIE SubprogramDie = CUDie.addChild(DW_TAG_subprogram);
  SubprogramDie.addAttribute(DW_AT_name, DW_FORM_strp, "main");
  SubprogramDie.addAttribute(DW_AT_low_pc, DW_FORM_addr, 0x1000U);
  SubprogramDie.addAttribute(DW_AT_high_pc, DW_FORM_addr, 0x2000U);

  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> Ctx = DWARFContext::create(**Obj);

  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  // Manually extracting DWARF DIE.
  uint64_t DIEOffset = CU->getOffset() + CU->getHeaderSize();
  uint64_t NextCUOffset = CU->getNextUnitOffset();
  DWARFDebugInfoEntry DieInfo;
  DWARFDataExtractor DebugInfoData = CU->getDebugInfoExtractor();
  ASSERT_TRUE(DieInfo.extractFast(*CU, &DIEOffset, DebugInfoData, NextCUOffset,
                                  UINT32_MAX));
  DWARFDie Die(CU, &DieInfo);
  ASSERT_TRUE(Die.isValid());
  ASSERT_TRUE(Die.hasChildren());
  // Since we have extracted manually DieArray is empty.
  // Dump function should respect the default flags and print just current DIE,
  // and not explore children.
  SmallString<512> Output;
  raw_svector_ostream OS(Output);
  Die.dump(OS);
  constexpr size_t NumOfLines = 3;
  SmallVector<StringRef, NumOfLines> Strings;
  SmallVector<StringRef, NumOfLines> ValidStrings = {
      "0x0000000b: DW_TAG_compile_unit",
      "              DW_AT_name	(\"/tmp/main.c\")",
      "              DW_AT_language	(DW_LANG_C)"};
  Output.str().split(Strings, '\n', -1, false);
  ASSERT_EQ(Strings.size(), NumOfLines);
  for (size_t I = 0; I < NumOfLines; ++I)
    EXPECT_EQ(ValidStrings[I], Strings[I]);
}

} // end anonymous namespace
