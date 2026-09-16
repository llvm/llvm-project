//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/FormatterSection.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatterBytecode.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/LEB128.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;

// Helpers for building bytecode formatter records, embedded into a binary and
// then read by LoadFormattersForModule.

template <typename T>
static void AppendBytes(std::vector<uint8_t> &bytes, T data) {
  bytes.insert(bytes.end(), data.begin(), data.end());
}

static void AppendULEB(std::vector<uint8_t> &bytes, uint64_t value) {
  uint8_t buf[10];
  unsigned len = llvm::encodeULEB128(value, buf);
  AppendBytes(bytes, llvm::ArrayRef(buf, len));
}

/// Append a bytecode formatter record to a section.
static void AppendRecord(std::vector<uint8_t> &section, uint64_t version,
                         llvm::StringRef type_name,
                         llvm::ArrayRef<uint8_t> entry,
                         std::optional<uint64_t> record_size_override = {}) {
  std::vector<uint8_t> body;
  AppendULEB(body, type_name.size());
  AppendBytes(body, type_name);
  AppendBytes(body, entry);

  AppendULEB(section, version);
  AppendULEB(section, record_size_override.value_or(body.size()));
  AppendBytes(section, llvm::ArrayRef<uint8_t>(body));
}

/// Build a minimal ELF binary with a single named section with the given
/// contents.
static std::string BuildBinaryYaml(llvm::StringRef section_name,
                                   llvm::ArrayRef<uint8_t> content) {
  return ("--- !ELF\n"
          "FileHeader:\n"
          "  Class:           ELFCLASS64\n"
          "  Data:            ELFDATA2LSB\n"
          "  Type:            ET_DYN\n"
          "  Machine:         EM_X86_64\n"
          "Sections:\n"
          "  - Name:            " +
          section_name.str() +
          "\n"
          "    Type:            SHT_PROGBITS\n"
          "    Flags:           [ ]\n"
          "    Address:         0x2010\n"
          "    AddressAlign:    0x10\n"
          "    Content:         " +
          llvm::toHex(content) +
          "\n"
          "    Size:            " +
          std::to_string(content.size()) +
          "\n"
          "...\n");
}

namespace {

struct MockProcess : Process {
  MockProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}

  llvm::StringRef GetPluginName() override { return "mock process"; }

  bool CanDebug(TargetSP target, bool plugin_specified_by_name) override {
    return false;
  };

  Status DoDestroy() override { return {}; }

  void RefreshStateAfterStop() override {}

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  };

  size_t DoReadMemory(const ProcessAddress &process_addr, void *buf,
                      size_t size, Status &error) override {
    return 0;
  }
};

class FormatterSectionTest : public ::testing::Test {
public:
  void SetUp() override {
    // The "default" category lives in a process-wide FormatManager, so start
    // each test from a clean slate regardless of what earlier tests in this
    // binary registered.
    TypeCategoryImplSP category;
    DataVisualization::Categories::GetCategory(ConstString("default"),
                                               category);
    if (category)
      category->Clear();

    ArchSpec arch("x86_64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(m_debugger_sp);
    m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                                eLoadDependentsNo,
                                                m_platform_sp, m_target_sp);
    ASSERT_TRUE(m_target_sp);
    ASSERT_TRUE(m_target_sp->GetArchitecture().IsValid());
    ASSERT_TRUE(m_platform_sp);
    m_listener_sp = Listener::MakeListener("dummy");
    m_process_sp = std::make_shared<MockProcess>(m_target_sp, m_listener_sp);
    ASSERT_TRUE(m_process_sp);
    m_exe_ctx = ExecutionContext(m_process_sp);
  }

  ExecutionContext m_exe_ctx;
  TypeSystemClang *m_type_system;
  lldb::TargetSP m_target_sp;

private:
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF,
                platform_linux::PlatformLinux, SymbolFileSymtab>
      m_subsystems;

  lldb::DebuggerSP m_debugger_sp;
  lldb::PlatformSP m_platform_sp;
  lldb::ListenerSP m_listener_sp;
  lldb::ProcessSP m_process_sp;
};

} // namespace

/// Test that multiple formatters can be loaded
TEST_F(FormatterSectionTest, LoadFormattersForModule) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .lldbformatters
    Type:            SHT_PROGBITS
    Flags:           [ ]
    Address:         0x2010
    AddressAlign:    0x10
    # Two summaries for "Point" and "Rect" that return "AAAAA" and "BBBBB" respectively
    Content:         011205506F696E74000009012205414141414113000000000111045265637400000901220542424242421300000000
    Size:            256
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);

  ASSERT_EQ(category->GetCount(), 2u);

  TypeSummaryImplSP point_summary_sp =
      category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
          "Point", lldb::eFormatterMatchExact));
  ASSERT_TRUE(point_summary_sp != nullptr);

  TypeSummaryImplSP rect_summary_sp =
      category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
          "Rect", lldb::eFormatterMatchExact));
  ASSERT_TRUE(rect_summary_sp != nullptr);

  std::string dest;
  Scalar val;
  ValueObjectSP valobj = ValueObjectConstResult::CreateValueObjectFromScalar(
      ExecutionContext(m_target_sp.get(), false), val, CompilerType(), "mock");
  ASSERT_TRUE(
      point_summary_sp->FormatObject(valobj.get(), dest, TypeSummaryOptions()));
  ASSERT_EQ(dest, "AAAAA");
  dest.clear();
  ASSERT_TRUE(
      rect_summary_sp->FormatObject(valobj.get(), dest, TypeSummaryOptions()));
  ASSERT_EQ(dest, "BBBBB");
}

/// Test an invalid leading version number can't be decoded.
TEST_F(FormatterSectionTest, MalformedULEBAtStart) {
  //  A lone continuation byte (high bit set) is not a complete ULEB128 value.
  std::vector<uint8_t> section = {0x80};

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}

/// A record whose version isn't 1 is unsupported and should be skipped.
TEST_F(FormatterSectionTest, SkipsRecordWithUnsupportedVersion) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(FormatterBytecode::Signatures::sig_summary);
  AppendULEB(entry, /*bytecode_size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0xAA, 0xBB}));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/2, "Bogus", entry);
  AppendRecord(section, /*version=*/1, "Good", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
                "Bogus", lldb::eFormatterMatchExact)),
            nullptr);
  EXPECT_NE(category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
                "Good", lldb::eFormatterMatchExact)),
            nullptr);
}

/// Test mismatch of decalred type name size and actual length of type name.
TEST_F(FormatterSectionTest, TypeNameSizeExceedsLengthOfTypeName) {
  std::vector<uint8_t> body;
  // Declare a type name of incorrect length (name: "Foo", length: 10).
  AppendULEB(body, /*type_size=*/10);
  AppendBytes(body, llvm::StringRef("Foo"));

  std::vector<uint8_t> section;
  AppendULEB(section, /*version=*/1);
  AppendULEB(section, /*record_size=*/body.size());
  AppendBytes(section, llvm::ArrayRef<uint8_t>(body));

  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(FormatterBytecode::Signatures::sig_summary);
  AppendULEB(entry, /*bytecode_size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0xAA, 0xBB}));
  AppendRecord(section, /*version=*/1, "Good", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 1u);
  EXPECT_NE(category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
                "Good", lldb::eFormatterMatchExact)),
            nullptr);
}

// Test that a record does not extend past the section it is within.
TEST_F(FormatterSectionTest, RecordSizeExceedsRemainingSectionIsRejected) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(FormatterBytecode::Signatures::sig_summary);
  AppendULEB(entry, /*bytecode_size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0xAA, 0xBB}));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Good", entry);
  AppendRecord(section, /*version=*/1, "Oversized", entry,
               /*record_size_override=*/1000000);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 1u);
  EXPECT_NE(category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
                "Good", lldb::eFormatterMatchExact)),
            nullptr);
  EXPECT_EQ(category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
                "Oversized", lldb::eFormatterMatchExact)),
            nullptr);
}

// Test that an unrecognized signature skips the current formatter entry.
TEST_F(FormatterSectionTest, UnsupportedSignatureSkipsEntry) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  // Invalid signature.
  entry.push_back(0xFF);
  AppendULEB(entry, /*size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0x11, 0x22}));
  entry.push_back(FormatterBytecode::Signatures::sig_summary);
  AppendULEB(entry, /*size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0xAA, 0xBB}));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Widget", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_NE(category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
                "Widget", lldb::eFormatterMatchExact)),
            nullptr);
}

/// Test a signature body being declared with too large a size.
TEST_F(FormatterSectionTest, TruncatedBytecodeSizeAbortsEntryParsing) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(FormatterBytecode::Signatures::sig_init);
  // Declared bytecode size is larger than the 0 bytes of the entry.
  AppendULEB(entry, /*size=*/500);

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Broken", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
  EXPECT_EQ(
      category->GetSyntheticForType(std::make_shared<TypeNameSpecifierImpl>(
          "Broken", lldb::eFormatterMatchExact)),
      nullptr);
}

/// Test that an entry which has flags but neither summary or synthetic
/// signature (valid framing, but empty) must not register a formatter either.
TEST_F(FormatterSectionTest, EmptyEntryRegistersNothing) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Empty", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}

/// Test that an embedded type summary with an empty summary string is dropped
/// instead of being registered.
TEST_F(FormatterSectionTest, EmptySummaryStringIsNotRegistered) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*summary_size=*/0);

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Empty", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbsummaries", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadTypeSummariesForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}

/// Test that a declared summary size larger than the bytes actually available
/// in the entry must fail cleanly instead of reading out of bounds, and the
/// summary must not be registered.
TEST_F(FormatterSectionTest, SummarySizeExceedsAvailableBytes) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*summary_size=*/50);
  AppendBytes(entry, llvm::StringRef("short"));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Oops", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbsummaries", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadTypeSummariesForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}
