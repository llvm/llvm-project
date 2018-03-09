//===- DWARFDebugLineTest.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "DwarfUtils.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf;
using namespace object;
using namespace utils;

namespace {

struct DebugLineGenerator {
  bool init() {
    Triple T = getHostTripleForAddrSize(8);
    if (!isConfigurationSupported(T))
      return false;
    auto ExpectedGenerator = dwarfgen::Generator::create(T, 4);
    if (ExpectedGenerator)
      Generator.reset(ExpectedGenerator->release());
    return true;
  }

  std::unique_ptr<DWARFContext> createContext() {
    if (!Generator)
      return nullptr;
    StringRef FileBytes = Generator->generate();
    MemoryBufferRef FileBuffer(FileBytes, "dwarf");
    auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
    if (Obj)
      return DWARFContext::create(**Obj);
    return nullptr;
  }

  std::unique_ptr<dwarfgen::Generator> Generator;
};

TEST(DWARFDebugLine, GetLineTableAtInvalidOffset) {
  DebugLineGenerator LineGen;
  if (!LineGen.init())
    return;

  DWARFDebugLine Line;
  std::unique_ptr<DWARFContext> Context = LineGen.createContext();
  ASSERT_TRUE(Context != nullptr);
  const DWARFObject &Obj = Context->getDWARFObj();
  DWARFDataExtractor LineData(Obj, Obj.getLineSection(), true, 8);

  EXPECT_EQ(Line.getOrParseLineTable(LineData, 0, *Context, nullptr), nullptr);
}

} // end anonymous namespace
