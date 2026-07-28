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
#include "llvm/Support/LEB128.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;

namespace {

// --- Helpers for hand-assembling the embedded formatter/summary record
// format read by FormatterSection.cpp, so malformed inputs can be expressed
// as typed fields instead of raw hex blobs. ---

void AppendULEB(std::vector<uint8_t> &bytes, uint64_t value) {
  uint8_t buf[10];
  unsigned len = llvm::encodeULEB128(value, buf);
  bytes.insert(bytes.end(), buf, buf + len);
}

void AppendBytes(std::vector<uint8_t> &bytes, llvm::StringRef data) {
  bytes.insert(bytes.end(), data.begin(), data.end());
}

void AppendBytes(std::vector<uint8_t> &bytes, llvm::ArrayRef<uint8_t> data) {
  bytes.insert(bytes.end(), data.begin(), data.end());
}

// Appends one length-framed record: [version][record_size][type_size]
// [type_name][entry]. `record_size` is declared honestly as the size of
// [type_size][type_name][entry] unless a test overrides it to exercise a
// mismatched/corrupt size.
void AppendRecord(std::vector<uint8_t> &section, uint64_t version,
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

std::string ToHex(llvm::ArrayRef<uint8_t> bytes) {
  static const char digits[] = "0123456789ABCDEF";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    hex.push_back(digits[b >> 4]);
    hex.push_back(digits[b & 0xF]);
  }
  return hex;
}

// Builds a minimal ELF with a single section named `section_name` whose
// contents are exactly `content` (no implicit padding).
std::string BuildSectionYaml(llvm::StringRef section_name,
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
          ToHex(content) +
          "\n"
          "    Size:            " +
          std::to_string(content.size()) +
          "\n"
          "...\n");
}

} // namespace

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

  size_t DoReadMemory(addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
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

/// A lone continuation byte (high bit set) is not a complete ULEB128 value,
/// so even the leading version number can't be decoded. This must not read
/// out of bounds or crash.
TEST_F(FormatterSectionTest, MalformedULEBAtStart) {
  std::vector<uint8_t> section = {0x80};

  auto ExpectedFile =
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}

/// A record whose version isn't 1 is unsupported and should be skipped over
/// (using its honestly-declared record_size) without disturbing a
/// well-formed record that follows it.
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
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
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

/// The record declares a type name of length 10, but the record itself
/// (honestly sized by the outer record_size field) only has room for 3
/// bytes of name and nothing else. The type name read must fail cleanly
/// instead of reading past the record's bounds, and a well-formed record
/// that follows must still be reached.
TEST_F(FormatterSectionTest, InnerTypeSizeExceedsRecordBounds) {
  std::vector<uint8_t> body;
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
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
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

/// A record_size far larger than the number of bytes actually remaining in
/// the section is an internally-inconsistent (corrupt/truncated) record: its
/// own declared size can't be trusted to locate the next record, so it must
/// be rejected rather than silently parsed from whatever bytes happen to be
/// left. A well-formed record preceding it is unaffected.
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
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
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

/// An unrecognized signature byte (with otherwise well-formed size/bytecode
/// framing) is logged and skipped without preventing a later, valid
/// signature in the same entry from being picked up.
TEST_F(FormatterSectionTest, UnsupportedSignatureByteIsSkippedWithinEntry) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(0xFF);
  AppendULEB(entry, /*size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0x11, 0x22}));
  entry.push_back(FormatterBytecode::Signatures::sig_summary);
  AppendULEB(entry, /*size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0xAA, 0xBB}));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Widget", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
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

/// The declared bytecode size (500) is far larger than the 0 bytes that
/// actually remain in the entry, so reading it fails cleanly rather than
/// reading out of bounds. Since no summary and no synthetic method was
/// successfully parsed, nothing should be registered for the type.
TEST_F(FormatterSectionTest, TruncatedBytecodeSizeAbortsEntryParsing) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(FormatterBytecode::Signatures::sig_init);
  AppendULEB(entry, /*size=*/500);

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Broken", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
  EXPECT_EQ(category->GetSyntheticForType(std::make_shared<TypeNameSpecifierImpl>(
                "Broken", lldb::eFormatterMatchExact)),
            nullptr);
}

/// An entry that has flags but no summary or synthetic-method sub-entries
/// at all (valid framing, just empty) must not register a formatter either.
TEST_F(FormatterSectionTest, EmptyEntryRegistersNothing) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Empty", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildSectionYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}

/// An embedded type summary with an empty summary string is dropped instead
/// of being registered.
TEST_F(FormatterSectionTest, EmptySummaryStringIsNotRegistered) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*summary_size=*/0);

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Empty", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildSectionYaml(".lldbsummaries", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadTypeSummariesForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}

/// A declared summary_size larger than the bytes actually available in the
/// entry must fail cleanly instead of reading out of bounds, and the
/// summary must not be registered.
TEST_F(FormatterSectionTest, SummarySizeExceedsAvailableBytes) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*summary_size=*/50);
  AppendBytes(entry, llvm::StringRef("short"));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "Oops", entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildSectionYaml(".lldbsummaries", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadTypeSummariesForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);
  EXPECT_EQ(category->GetCount(), 0u);
}
