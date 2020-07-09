//===-- TestLineEntry.cpp -------------------------------------------------===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include <iostream>

#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;
using namespace lldb;

class LineEntryTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileMachO, SymbolFileDWARF,
                TypeSystemClang>
      subsystem;

public:
  void SetUp() override;

protected:
  llvm::Expected<LineEntry> GetLineEntryForLine(uint32_t line);
  llvm::Optional<TestFile> m_file;
  ModuleSP m_module_sp;
};

void LineEntryTest::SetUp() {
  auto ExpectedFile = TestFile::fromYamlFile("inlined-functions.yaml");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  m_file.emplace(std::move(*ExpectedFile));
  m_module_sp = std::make_shared<Module>(m_file->moduleSpec());
}

llvm::Expected<LineEntry> LineEntryTest::GetLineEntryForLine(uint32_t line) {
  bool check_inlines = true;
  bool exact = true;
  SymbolContextList sc_comp_units;
  SymbolContextList sc_line_entries;
  FileSpec file_spec("inlined-functions.cpp");
  m_module_sp->ResolveSymbolContextsForFileSpec(file_spec, line, check_inlines,
                                                lldb::eSymbolContextCompUnit,
                                                sc_comp_units);
  if (sc_comp_units.GetSize() == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No comp unit found on the test object.");
  sc_comp_units[0].comp_unit->ResolveSymbolContext(
      file_spec, line, check_inlines, exact, eSymbolContextLineEntry,
      sc_line_entries);
  if (sc_line_entries.GetSize() == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No line entry found on the test object.");
  return sc_line_entries[0].line_entry;
}

TEST_F(LineEntryTest, GetSameLineContiguousAddressRangeNoInlines) {
  auto line_entry = GetLineEntryForLine(18);
  ASSERT_THAT_EXPECTED(line_entry, llvm::Succeeded());
  bool include_inlined_functions = false;
  auto range =
      line_entry->GetSameLineContiguousAddressRange(include_inlined_functions);
  ASSERT_EQ(range.GetByteSize(), (uint64_t)0x24);
}

TEST_F(LineEntryTest, GetSameLineContiguousAddressRangeOneInline) {
  auto line_entry = GetLineEntryForLine(18);
  ASSERT_THAT_EXPECTED(line_entry, llvm::Succeeded());
  bool include_inlined_functions = true;
  auto range =
      line_entry->GetSameLineContiguousAddressRange(include_inlined_functions);
  ASSERT_EQ(range.GetByteSize(), (uint64_t)0x49);
}

TEST_F(LineEntryTest, GetSameLineContiguousAddressRangeNestedInline) {
  auto line_entry = GetLineEntryForLine(12);
  ASSERT_THAT_EXPECTED(line_entry, llvm::Succeeded());
  bool include_inlined_functions = true;
  auto range =
      line_entry->GetSameLineContiguousAddressRange(include_inlined_functions);
  ASSERT_EQ(range.GetByteSize(), (uint64_t)0x33);
}

/*
# inlined-functions.cpp
inline __attribute__((always_inline)) int sum2(int a, int b) {
    int result = a + b;
    return result;
}

int sum3(int a, int b, int c) {
    int result = a + b + c;
    return result;
}

inline __attribute__((always_inline)) int sum4(int a, int b, int c, int d) {
    int result = sum2(a, b) + sum2(c, d);
    result += 0;
    return result;
}

int main(int argc, char** argv) {
    sum3(3, 4, 5) + sum2(1, 2);
    int sum = sum4(1, 2, 3, 4);
    sum2(5, 6);
    return 0;
}

// g++ -c inlined-functions.cpp -o inlined-functions.o -g -Wno-unused-value
// obj2yaml inlined-functions.o > inlined-functions.yaml

# Dump of source line per address:
# inlined-functions.cpp is src.cpp for space considerations.
0x20: src.cpp:17
0x21: src.cpp:17
0x26: src.cpp:17
0x27: src.cpp:17
0x29: src.cpp:17
0x2e: src.cpp:17
0x2f: src.cpp:17
0x31: src.cpp:17
0x36: src.cpp:18
0x37: src.cpp:18
0x39: src.cpp:18
0x3e: src.cpp:18
0x3f: src.cpp:18
0x41: src.cpp:18
0x46: src.cpp:18
0x47: src.cpp:18
0x49: src.cpp:18
0x4e: src.cpp:18
0x4f: src.cpp:18
0x51: src.cpp:18
0x56: src.cpp:18
0x57: src.cpp:18
0x59: src.cpp:18
0x5e: src.cpp:18 -> sum2@src.cpp:2
0x5f: src.cpp:18 -> sum2@src.cpp:2
0x61: src.cpp:18 -> sum2@src.cpp:2
0x66: src.cpp:18 -> sum2@src.cpp:2
0x67: src.cpp:18 -> sum2@src.cpp:2
0x69: src.cpp:18 -> sum2@src.cpp:2
0x6e: src.cpp:18 -> sum2@src.cpp:2
0x6f: src.cpp:18 -> sum2@src.cpp:2
0x71: src.cpp:18 -> sum2@src.cpp:2
0x76: src.cpp:18 -> sum2@src.cpp:2
0x77: src.cpp:18 -> sum2@src.cpp:2
0x79: src.cpp:18 -> sum2@src.cpp:2
0x7e: src.cpp:18 -> sum2@src.cpp:2
0x7f: src.cpp:19 -> sum4@src.cpp:12
0x81: src.cpp:19 -> sum4@src.cpp:12
0x86: src.cpp:19 -> sum4@src.cpp:12
0x87: src.cpp:19 -> sum4@src.cpp:12
0x89: src.cpp:19 -> sum4@src.cpp:12
0x8e: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:2
0x8f: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:2
0x91: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:2
0x96: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:3
0x97: src.cpp:19 -> sum4@src.cpp:12
0x99: src.cpp:19 -> sum4@src.cpp:12
0x9e: src.cpp:19 -> sum4@src.cpp:12
0x9f: src.cpp:19 -> sum4@src.cpp:12
0xa1: src.cpp:19 -> sum4@src.cpp:12
0xa6: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:2
0xa7: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:2
0xa9: src.cpp:19 -> sum4@src.cpp:12 -> sum2@src.cpp:2
0xae: src.cpp:19 -> sum4@src.cpp:12
0xaf: src.cpp:19 -> sum4@src.cpp:12
0xb1: src.cpp:19 -> sum4@src.cpp:12
0xb6: src.cpp:19 -> sum4@src.cpp:13
0xb7: src.cpp:19 -> sum4@src.cpp:13
0xb9: src.cpp:19 -> sum4@src.cpp:14
0xbe: src.cpp:19
0xbf: src.cpp:19
0xc1: src.cpp:19
0xc6: src.cpp:19
0xc7: src.cpp:19
0xc9: src.cpp:19
0xce: src.cpp:20 -> sum2@src.cpp:2
0xcf: src.cpp:20 -> sum2@src.cpp:2
0xd1: src.cpp:20 -> sum2@src.cpp:2
0xd6: src.cpp:21
0xd7: src.cpp:21
0xd9: src.cpp:21
0xde: src.cpp:21
*/
