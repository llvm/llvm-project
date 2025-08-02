//===-- TestLineEntry.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include <iostream>
#include <optional>

#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_linux;
using namespace lldb_private::plugin::dwarf;
using namespace testing;

namespace {

constexpr llvm::StringLiteral k_source_file("inlined-functions.cpp");

class LineEntryTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileMachO, PlatformLinux,
                SymbolFileDWARF, TypeSystemClang>
      subsystem;

public:
  void SetUp() override;
  void TearDown() override;

protected:
  llvm::Expected<SymbolContextList>
  GetLineEntriesForLine(uint32_t line, std::optional<uint16_t> column);
  void CheckNoCallback();
  void CheckCallbackWithArgs(const ModuleSP &module_sp,
                             const FileSpec &resolved_file_spec,
                             const ModuleSP &expected_module_sp);
  FileSpec GetTestSourceFile();
  std::optional<TestFile> m_file;
  ModuleSP m_module_sp;
  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
  TargetSP m_target_sp;
};

void LineEntryTest::SetUp() {
  auto ExpectedFile = TestFile::fromYamlFile("inlined-functions.yaml");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  m_file.emplace(std::move(*ExpectedFile));
  m_module_sp = std::make_shared<Module>(m_file->moduleSpec());

  // Create Debugger.
  m_debugger_sp = Debugger::CreateInstance();
  EXPECT_TRUE(m_debugger_sp);

  // Create Platform.
  ArchSpec host_arch("i386-pc-linux");
  m_platform_sp =
      platform_linux::PlatformLinux::CreateInstance(true, &host_arch);
  Platform::SetHostPlatform(m_platform_sp);
  EXPECT_TRUE(m_platform_sp);

  // Create Target.
  m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", host_arch,
                                              eLoadDependentsNo, m_platform_sp,
                                              m_target_sp);
  EXPECT_TRUE(m_target_sp);

  m_target_sp->SetExecutableModule(m_module_sp);
}

void LineEntryTest::TearDown() {
  if (m_module_sp) {
    ModuleList::RemoveSharedModule(m_module_sp);
  }
}

void LineEntryTest::CheckNoCallback() {
  EXPECT_FALSE(m_platform_sp->GetResolveSourceFileCallback());
}

void LineEntryTest::CheckCallbackWithArgs(const ModuleSP &module_sp,
                                          const FileSpec &resolved_file_spec,
                                          const ModuleSP &expected_module_sp) {
  EXPECT_TRUE(module_sp == expected_module_sp);
  EXPECT_FALSE(resolved_file_spec);
}

FileSpec LineEntryTest::GetTestSourceFile() {
  const auto *info = UnitTest::GetInstance()->current_test_info();
  FileSpec test_file = HostInfo::GetProcessTempDir();
  test_file.AppendPathComponent(std::string(info->test_case_name()) + "-" +
                                info->name());
  test_file.AppendPathComponent(k_source_file);

  std::error_code ec =
      llvm::sys::fs::create_directory(test_file.GetDirectory().GetCString());
  EXPECT_FALSE(ec);

  ec = llvm::sys::fs::copy_file(GetInputFilePath(k_source_file),
                                test_file.GetPath().c_str());
  EXPECT_FALSE(ec);
  return test_file;
}

  // TODO: Handle SourceLocationSpec column information
llvm::Expected<SymbolContextList> LineEntryTest::GetLineEntriesForLine(
    uint32_t line, std::optional<uint16_t> column = std::nullopt) {
  SymbolContextList sc_comp_units;
  SymbolContextList sc_line_entries;
  FileSpec file_spec("inlined-functions.cpp");
  m_module_sp->ResolveSymbolContextsForFileSpec(
      file_spec, line, /*check_inlines=*/true, lldb::eSymbolContextCompUnit,
      sc_comp_units);
  if (sc_comp_units.GetSize() == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No comp unit found on the test object.");

  SourceLocationSpec location_spec(file_spec, line, column,
                                   /*check_inlines=*/true,
                                   /*exact_match=*/true);

  sc_comp_units[0].comp_unit->ResolveSymbolContext(
      location_spec, eSymbolContextLineEntry, sc_line_entries);
  if (sc_line_entries.GetSize() == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No line entry found on the test object.");
  return sc_line_entries;
}

} // end namespace

// This tests if we can get all line entries that match the passed line, if
// no column is specified.
TEST_F(LineEntryTest, GetAllExactLineMatchesWithoutColumn) {
  auto sc_line_entries = GetLineEntriesForLine(12);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  ASSERT_EQ(sc_line_entries->NumLineEntriesWithLine(12), 6u);
}

// This tests if we can get exact line and column matches.
TEST_F(LineEntryTest, GetAllExactLineColumnMatches) {
  auto sc_line_entries = GetLineEntriesForLine(12, 39);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  ASSERT_EQ(sc_line_entries->NumLineEntriesWithLine(12), 1u);
  auto line_entry = sc_line_entries.get()[0].line_entry;
  ASSERT_EQ(line_entry.column, 39);
}

TEST_F(LineEntryTest, GetSameLineContiguousAddressRangeNoInlines) {
  auto sc_line_entries = GetLineEntriesForLine(18);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;
  bool include_inlined_functions = false;
  auto range =
      line_entry.GetSameLineContiguousAddressRange(include_inlined_functions);
  ASSERT_EQ(range.GetByteSize(), (uint64_t)0x24);
}

TEST_F(LineEntryTest, GetSameLineContiguousAddressRangeOneInline) {
  auto sc_line_entries = GetLineEntriesForLine(18);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;
  bool include_inlined_functions = true;
  auto range =
      line_entry.GetSameLineContiguousAddressRange(include_inlined_functions);
  ASSERT_EQ(range.GetByteSize(), (uint64_t)0x49);
}

TEST_F(LineEntryTest, GetSameLineContiguousAddressRangeNestedInline) {
  auto sc_line_entries = GetLineEntriesForLine(12);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;
  bool include_inlined_functions = true;
  auto range =
      line_entry.GetSameLineContiguousAddressRange(include_inlined_functions);
  ASSERT_EQ(range.GetByteSize(), (uint64_t)0x33);
}

TEST_F(LineEntryTest, ResolveSourceFileCallbackNotSetTest) {
  // Resolve source file callback is not set. ApplyFileMappings should succeed
  // to return the original file spec from line entry
  auto sc_line_entries = GetLineEntriesForLine(12);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;

  const lldb::SupportFileSP original_file_sp = line_entry.file_sp;
  CheckNoCallback();

  line_entry.ApplyFileMappings(m_target_sp, m_module_sp);
  ASSERT_EQ(line_entry.file_sp, original_file_sp);
}

TEST_F(LineEntryTest, ResolveSourceFileCallbackSetFailTest) {
  // Resolve source file callback fails for some reason.
  // CallResolveSourceFileCallbackIfSet should fail and ApplyFileMappings should
  // return the original file spec.
  auto sc_line_entries = GetLineEntriesForLine(12);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;

  const lldb::SupportFileSP original_file_sp = line_entry.file_sp;
  m_platform_sp->SetResolveSourceFileCallback(
      [this](const lldb::ModuleSP &module_sp,
             const FileSpec &original_file_spec, FileSpec &resolved_file_spec) {
        CheckCallbackWithArgs(module_sp, resolved_file_spec, m_module_sp);
        return Status::FromErrorString(
            "The resolve source file callback failed");
      });

  line_entry.ApplyFileMappings(m_target_sp, m_module_sp);
  ASSERT_EQ(line_entry.file_sp, original_file_sp);
}

TEST_F(LineEntryTest, ResolveSourceFileCallbackSetNonExistentPathTest) {
  // Resolve source file callback succeeds but returns a non-existent path.
  // CallResolveSourceFileCallbackIfSet should succeed and ApplyFileMappings
  // should return the original file spec.
  auto sc_line_entries = GetLineEntriesForLine(12);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;

  const lldb::SupportFileSP original_file_sp = line_entry.file_sp;
  FileSpec callback_file_spec;
  const FileSpec expected_callback_file_spec =
      FileSpec("/this path does not exist");

  m_platform_sp->SetResolveSourceFileCallback(
      [this, &expected_callback_file_spec, &callback_file_spec](
          const lldb::ModuleSP &module_sp, const FileSpec &original_file_spec,
          FileSpec &resolved_file_spec) {
        CheckCallbackWithArgs(module_sp, resolved_file_spec, m_module_sp);
        resolved_file_spec.SetPath(expected_callback_file_spec.GetPath());
        callback_file_spec.SetPath(resolved_file_spec.GetPath());
        return Status();
      });

  line_entry.ApplyFileMappings(m_target_sp, m_module_sp);

  ASSERT_EQ(callback_file_spec, expected_callback_file_spec);
  ASSERT_EQ(line_entry.file_sp, original_file_sp);
}

TEST_F(LineEntryTest, ResolveSourceFileCallbackSetSuccessTest) {
  // Resolve source file callback is set. CallResolveSourceFileCallbackIfSet
  // should succeed and ApplyFileMappings should return the new file spec from
  // the callback.
  auto sc_line_entries = GetLineEntriesForLine(12);
  ASSERT_THAT_EXPECTED(sc_line_entries, llvm::Succeeded());
  auto line_entry = sc_line_entries.get()[0].line_entry;

  const FileSpec expected_callback_file_spec = GetTestSourceFile();

  m_platform_sp->SetResolveSourceFileCallback(
      [this, &expected_callback_file_spec](const lldb::ModuleSP &module_sp,
                                           const FileSpec &original_file_spec,
                                           FileSpec &resolved_file_spec) {
        CheckCallbackWithArgs(module_sp, resolved_file_spec, m_module_sp);
        resolved_file_spec.SetPath(expected_callback_file_spec.GetPath());
        return Status();
      });

  line_entry.ApplyFileMappings(m_target_sp, m_module_sp);
  ASSERT_EQ(line_entry.file_sp->GetSpecOnly(), expected_callback_file_spec);
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
