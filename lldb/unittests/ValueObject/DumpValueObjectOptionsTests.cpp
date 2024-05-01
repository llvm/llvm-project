//===-- DumpValueObjectOptionsTests.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

struct MockProcess : Process {
  MockProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}

  llvm::StringRef GetPluginName() override { return "mock process"; }

  bool CanDebug(lldb::TargetSP target, bool plugin_specified_by_name) override {
    return false;
  };

  Status DoDestroy() override { return {}; }

  void RefreshStateAfterStop() override {}

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  };

  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    // No need to read memory in these tests.
    return size;
  }
};

class ValueObjectMockProcessTest : public ::testing::Test {
public:
  void SetUp() override {
    ArchSpec arch("i386-pc-linux");
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

    m_holder = std::make_unique<clang_utils::TypeSystemClangHolder>("test");
    m_type_system = m_holder->GetAST();
  }

  ExecutionContext m_exe_ctx;
  TypeSystemClang *m_type_system;

private:
  SubsystemRAII<FileSystem, HostInfo, platform_linux::PlatformLinux,
                ScriptInterpreterNone>
      m_subsystems;

  std::unique_ptr<clang_utils::TypeSystemClangHolder> m_holder;
  lldb::DebuggerSP m_debugger_sp;
  lldb::TargetSP m_target_sp;
  lldb::PlatformSP m_platform_sp;
  lldb::ListenerSP m_listener_sp;
  lldb::ProcessSP m_process_sp;
};

TEST_F(ValueObjectMockProcessTest, Enum) {
  CompilerType uint_type = m_type_system->GetBuiltinTypeForEncodingAndBitSize(
      lldb::eEncodingUint, 32);
  CompilerType enum_type = m_type_system->CreateEnumerationType(
      "test_enum", m_type_system->GetTranslationUnitDecl(),
      OptionalClangModuleID(), Declaration(), uint_type, false);

  m_type_system->StartTagDeclarationDefinition(enum_type);
  Declaration decl;
  // Each value sets one bit in the enum, to make this a "bitfield like enum".
  m_type_system->AddEnumerationValueToEnumerationType(enum_type, decl, "test_2",
                                                      2, 32);
  m_type_system->AddEnumerationValueToEnumerationType(enum_type, decl, "test_4",
                                                      4, 32);
  m_type_system->CompleteTagDeclarationDefinition(enum_type);

  std::vector<std::tuple<uint32_t, DumpValueObjectOptions, const char *>> enums{
      {0, {}, "(test_enum) test_var =\n"},
      {1, {}, "(test_enum) test_var = 0x1\n"},
      {2, {}, "(test_enum) test_var = test_2\n"},
      {4, {}, "(test_enum) test_var = test_4\n"},
      {6, {}, "(test_enum) test_var = test_2 | test_4\n"},
      {7, {}, "(test_enum) test_var = test_2 | test_4 | 0x1\n"},
      {8, {}, "(test_enum) test_var = 0x8\n"},
      {1, DumpValueObjectOptions().SetHideRootName(true), "(test_enum) 0x1\n"},
      {1, DumpValueObjectOptions().SetHideRootType(true), "test_var = 0x1\n"},
      {1, DumpValueObjectOptions().SetHideRootName(true).SetHideRootType(true),
       "0x1\n"},
      {1, DumpValueObjectOptions().SetHideName(true), "(test_enum) 0x1\n"},
      {1, DumpValueObjectOptions().SetHideValue(true),
       "(test_enum) test_var =\n"},
      {1, DumpValueObjectOptions().SetHideName(true).SetHideValue(true),
       "(test_enum) \n"},
  };

  StreamString strm;
  ExecutionContextScope *exe_scope = m_exe_ctx.GetBestExecutionContextScope();
  ConstString var_name("test_var");
  ByteOrder endian = endian::InlHostByteOrder();
  for (auto [value, options, expected] : enums) {
    DataExtractor data_extractor{&value, sizeof(value), endian, 4};
    ValueObjectConstResult::Create(exe_scope, enum_type, var_name,
                                   data_extractor)
        ->Dump(strm, options);
    ASSERT_STREQ(strm.GetString().str().c_str(), expected);
    strm.Clear();
  }
}
