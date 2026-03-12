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
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

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
  ValueObjectSP valobj = ValueObjectConstResult::CreateValueObjectFromBool(
      m_target_sp, false, "mock");
  ASSERT_TRUE(
      point_summary_sp->FormatObject(valobj.get(), dest, TypeSummaryOptions()));
  ASSERT_EQ(dest, "AAAAA");
  dest.clear();
  ASSERT_TRUE(
      rect_summary_sp->FormatObject(valobj.get(), dest, TypeSummaryOptions()));
  ASSERT_EQ(dest, "BBBBB");
}
