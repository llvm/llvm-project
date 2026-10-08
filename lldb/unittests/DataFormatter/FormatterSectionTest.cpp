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
#include "lldb/DataFormatters/TypeSynthetic.h"
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

/// Append a formatter method, its signature followed by its bytecode, to a
/// record's entry.
static void AppendMethod(std::vector<uint8_t> &entry,
                         FormatterBytecode::Signatures sig,
                         std::vector<uint8_t> code) {
  entry.push_back(sig);
  AppendULEB(entry, code.size());
  AppendBytes(entry, code);
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

/// A record whose version isn't 1 or 2 is unsupported and should be skipped.
TEST_F(FormatterSectionTest, SkipsRecordWithUnsupportedVersion) {
  std::vector<uint8_t> entry;
  AppendULEB(entry, /*flags=*/0);
  entry.push_back(FormatterBytecode::Signatures::sig_summary);
  AppendULEB(entry, /*bytecode_size=*/2);
  AppendBytes(entry, llvm::ArrayRef<uint8_t>({0xAA, 0xBB}));

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/3, "Bogus", entry);
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

/// Selectors return Integer in version 2 records, and UInt in version 1.
TEST_F(FormatterSectionTest, Version2SelectorsReturnInteger) {
  using namespace FormatterBytecode;
  // Adding a literal of the other integer type would be a type error.
  std::vector<uint8_t> v1;
  AppendULEB(v1, /*flags=*/0);
  AppendMethod(v1, sig_summary,
               {op_drop, op_lit_string, 5, 'h', 'e', 'l', 'l', 'o',
                op_lit_selector, sel_strlen, op_call, op_lit_uint, 0, op_plus});
  std::vector<uint8_t> v2;
  AppendULEB(v2, /*flags=*/0);
  AppendMethod(v2, sig_summary,
               {op_drop, op_lit_string, 5, 'h', 'e', 'l', 'l', 'o',
                op_lit_selector, sel_strlen, op_call, op_lit_integer, 0,
                op_plus});

  std::vector<uint8_t> section;
  AppendRecord(section, /*version=*/1, "V1", v1);
  AppendRecord(section, /*version=*/2, "V2", v2);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);

  Scalar val;
  ValueObjectSP valobj = ValueObjectConstResult::CreateValueObjectFromScalar(
      ExecutionContext(m_target_sp.get(), false), val, CompilerType(), "mock");
  for (const char *type_name : {"V1", "V2"}) {
    SCOPED_TRACE(type_name);
    TypeSummaryImplSP summary_sp =
        category->GetSummaryForType(std::make_shared<TypeNameSpecifierImpl>(
            type_name, lldb::eFormatterMatchExact));
    ASSERT_TRUE(summary_sp != nullptr);
    std::string dest;
    EXPECT_TRUE(
        summary_sp->FormatObject(valobj.get(), dest, TypeSummaryOptions()))
        << dest;
    EXPECT_EQ(dest, "5");
  }
}

/// Under version 2, the runtime passes the same `self` Dictionary to every
/// method, which modify it in place.
TEST_F(FormatterSectionTest, LoadsVersion2SyntheticChildren) {
  using namespace FormatterBytecode;
  // @init: (self Object -> ), sets self["n"] = 5.
  std::vector<uint8_t> init = {op_drop, op_lit_string, 1, 'n', op_lit_integer,
                               5,       op_dict_set};
  // @update: (self -> Integer), sets self["n"] = 7, replies "reuse".
  std::vector<uint8_t> update = {
      op_lit_string, 1, 'n', op_lit_integer, 7, op_dict_set, op_lit_integer, 1};
  // @update: (self -> ), with the required reply missing.
  std::vector<uint8_t> update_no_reply = {op_drop};
  // @get_num_children: (self -> Integer), returns self["n"].
  std::vector<uint8_t> num_children = {op_lit_string, 1, 'n', op_dict_get};
  // @get_child_at_index: (self Integer -> ), sets self["n"] = index + 0,
  // which fails unless the index is an Integer.
  std::vector<uint8_t> child_at_index = {
      op_lit_integer, 0, op_plus, op_lit_string, 1, 'n', op_swap, op_dict_set};
  // @get_child_index: (self String -> Integer), returns 2.
  std::vector<uint8_t> child_index = {op_drop, op_drop, op_lit_integer, 2};

  std::vector<uint8_t> widget;
  AppendULEB(widget, /*flags=*/0);
  AppendMethod(widget, sig_init, init);
  AppendMethod(widget, sig_get_num_children, num_children);

  std::vector<uint8_t> gadget;
  AppendULEB(gadget, /*flags=*/0);
  AppendMethod(gadget, sig_init, init);
  AppendMethod(gadget, sig_update, update);
  AppendMethod(gadget, sig_get_num_children, num_children);

  std::vector<uint8_t> gizmo;
  AppendULEB(gizmo, /*flags=*/0);
  AppendMethod(gizmo, sig_update, update_no_reply);

  std::vector<uint8_t> indexed;
  AppendULEB(indexed, /*flags=*/0);
  AppendMethod(indexed, sig_get_num_children, num_children);
  AppendMethod(indexed, sig_get_child_at_index, child_at_index);
  AppendMethod(indexed, sig_get_child_index, child_index);

  // Zero children, and a child index of zero.
  std::vector<uint8_t> empty;
  AppendULEB(empty, /*flags=*/0);
  AppendMethod(empty, sig_get_num_children, {op_drop, op_lit_integer, 0});
  AppendMethod(empty, sig_get_child_index, {op_drop, op_drop, op_lit_int, 0});

  // Without @init, self["valobj"] is the Object.
  std::vector<uint8_t> implicit_init;
  AppendULEB(implicit_init, /*flags=*/0);
  AppendMethod(implicit_init, sig_get_num_children,
               {op_lit_string, 6, 'v', 'a', 'l', 'o', 'b', 'j', op_dict_get,
                op_is_null});

  std::vector<uint8_t> section;
  for (auto [name, entry] :
       {std::pair{"Widget", &widget}, std::pair{"Gadget", &gadget},
        std::pair{"Gizmo", &gizmo}, std::pair{"Indexed", &indexed},
        std::pair{"Empty", &empty}, std::pair{"ImplicitInit", &implicit_init}})
    AppendRecord(section, /*version=*/2, name, *entry);

  auto ExpectedFile =
      TestFile::fromYaml(BuildBinaryYaml(".lldbformatters", section));
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  LoadFormattersForModule(module_sp);

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);
  ASSERT_TRUE(category != nullptr);

  Scalar val;
  ValueObjectSP valobj = ValueObjectConstResult::CreateValueObjectFromScalar(
      ExecutionContext(m_target_sp.get(), false), val, CompilerType(), "mock");
  auto GetFrontEnd = [&](const char *type_name) {
    SyntheticChildrenSP synthetic_sp =
        category->GetSyntheticForType(std::make_shared<TypeNameSpecifierImpl>(
            type_name, lldb::eFormatterMatchExact));
    return synthetic_sp ? synthetic_sp->GetFrontEnd(*valobj) : nullptr;
  };

  // @init's modification of self is visible to @get_num_children.
  auto widget_fe = GetFrontEnd("Widget");
  ASSERT_TRUE(widget_fe != nullptr);
  EXPECT_EQ(widget_fe->Update(), lldb::ChildCacheState::eReuse);
  EXPECT_THAT_EXPECTED(widget_fe->CalculateNumChildren(), llvm::HasValue(5u));
  // Without @get_child_index, looking up a child by name is an error.
  EXPECT_THAT_EXPECTED(widget_fe->GetIndexOfChildWithName(ConstString("x")),
                       llvm::Failed());

  // @update's modification of self is visible to @get_num_children.
  auto gadget_fe = GetFrontEnd("Gadget");
  ASSERT_TRUE(gadget_fe != nullptr);
  EXPECT_EQ(gadget_fe->Update(), lldb::ChildCacheState::eReuse);
  EXPECT_THAT_EXPECTED(gadget_fe->CalculateNumChildren(), llvm::HasValue(7u));

  // Without a reply from @update, the children must be refetched.
  auto gizmo_fe = GetFrontEnd("Gizmo");
  ASSERT_TRUE(gizmo_fe != nullptr);
  EXPECT_EQ(gizmo_fe->Update(), lldb::ChildCacheState::eRefetch);

  // @get_child_at_index is passed an Integer index.
  auto indexed_fe = GetFrontEnd("Indexed");
  ASSERT_TRUE(indexed_fe != nullptr);
  indexed_fe->Update();
  indexed_fe->GetChildAtIndex(3);
  EXPECT_THAT_EXPECTED(indexed_fe->CalculateNumChildren(), llvm::HasValue(3u));
  EXPECT_THAT_EXPECTED(indexed_fe->GetIndexOfChildWithName(ConstString("x")),
                       llvm::HasValue(2u));

  // Zero is a valid number of children, and a valid child index.
  auto empty_fe = GetFrontEnd("Empty");
  ASSERT_TRUE(empty_fe != nullptr);
  empty_fe->Update();
  EXPECT_THAT_EXPECTED(empty_fe->CalculateNumChildren(), llvm::HasValue(0u));
  EXPECT_THAT_EXPECTED(empty_fe->GetIndexOfChildWithName(ConstString("x")),
                       llvm::HasValue(0u));

  // Without @init, self["valobj"] is set to the (non-null) Object.
  auto implicit_init_fe = GetFrontEnd("ImplicitInit");
  ASSERT_TRUE(implicit_init_fe != nullptr);
  implicit_init_fe->Update();
  EXPECT_THAT_EXPECTED(implicit_init_fe->CalculateNumChildren(),
                       llvm::HasValue(0u));
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
