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

  CompilerType
  MakeEnumType(const std::vector<std::pair<const char *, int>> enumerators) {
    CompilerType uint_type = m_type_system->GetBuiltinTypeForEncodingAndBitSize(
        lldb::eEncodingUint, 32);
    CompilerType enum_type = m_type_system->CreateEnumerationType(
        "TestEnum", m_type_system->GetTranslationUnitDecl(),
        OptionalClangModuleID(), Declaration(), uint_type, false);

    m_type_system->StartTagDeclarationDefinition(enum_type);
    Declaration decl;
    for (auto [name, value] : enumerators)
      m_type_system->AddEnumerationValueToEnumerationType(enum_type, decl, name,
                                                          value, 32);
    m_type_system->CompleteTagDeclarationDefinition(enum_type);

    return enum_type;
  }

  void TestDumpValueObject(
      CompilerType enum_type,
      const std::vector<
          std::tuple<uint32_t, DumpValueObjectOptions, const char *>> &tests) {
    StreamString strm;
    ConstString var_name("test_var");
    ByteOrder endian = endian::InlHostByteOrder();
    ExecutionContextScope *exe_scope = m_exe_ctx.GetBestExecutionContextScope();
    for (auto [value, options, expected] : tests) {
      DataExtractor data_extractor{&value, sizeof(value), endian, 4};
      ValueObjectConstResult::Create(exe_scope, enum_type, var_name,
                                     data_extractor)
          ->Dump(strm, options);
      ASSERT_STREQ(strm.GetString().str().c_str(), expected);
      strm.Clear();
    }
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
  // This is not a bitfield-like enum, so values are printed as decimal by
  // default. Also we only show the enumerator name if the value is an
  // exact match.
  TestDumpValueObject(
      MakeEnumType({{"test_2", 2}, {"test_3", 3}}),
      {{0, {}, "(TestEnum) test_var = 0\n"},
       {1, {}, "(TestEnum) test_var = 1\n"},
       {2, {}, "(TestEnum) test_var = test_2\n"},
       {3, {}, "(TestEnum) test_var = test_3\n"},
       {4, {}, "(TestEnum) test_var = 4\n"},
       {5, {}, "(TestEnum) test_var = 5\n"},
       {1, DumpValueObjectOptions().SetHideRootName(true), "(TestEnum) 1\n"},
       {1, DumpValueObjectOptions().SetHideRootType(true), "test_var = 1\n"},
       {1, DumpValueObjectOptions().SetHideRootName(true).SetHideRootType(true),
        "1\n"},
       {1, DumpValueObjectOptions().SetHideName(true), "(TestEnum) 1\n"},
       {1, DumpValueObjectOptions().SetHideValue(true),
        "(TestEnum) test_var =\n"},
       {1, DumpValueObjectOptions().SetHideName(true).SetHideValue(true),
        "(TestEnum) \n"}});
}

TEST_F(ValueObjectMockProcessTest, BitFieldLikeEnum) {
  // These enumerators set individual bits in the value, as if it were a flag
  // set. lldb treats this as a "bitfield like enum". This means we show values
  // as hex, a value of 0 shows nothing, and values with no exact enumerator are
  // shown as combinations of the other values.
  TestDumpValueObject(
      MakeEnumType({{"test_2", 2}, {"test_4", 4}}),
      {
          {0, {}, "(TestEnum) test_var =\n"},
          {1, {}, "(TestEnum) test_var = 0x1\n"},
          {2, {}, "(TestEnum) test_var = test_2\n"},
          {4, {}, "(TestEnum) test_var = test_4\n"},
          {6, {}, "(TestEnum) test_var = test_2 | test_4\n"},
          {7, {}, "(TestEnum) test_var = test_2 | test_4 | 0x1\n"},
          {8, {}, "(TestEnum) test_var = 0x8\n"},
          {1, DumpValueObjectOptions().SetHideRootName(true),
           "(TestEnum) 0x1\n"},
          {1, DumpValueObjectOptions().SetHideRootType(true),
           "test_var = 0x1\n"},
          {1,
           DumpValueObjectOptions().SetHideRootName(true).SetHideRootType(true),
           "0x1\n"},
          {1, DumpValueObjectOptions().SetHideName(true), "(TestEnum) 0x1\n"},
          {1, DumpValueObjectOptions().SetHideValue(true),
           "(TestEnum) test_var =\n"},
          {1, DumpValueObjectOptions().SetHideName(true).SetHideValue(true),
           "(TestEnum) \n"},
      });
}
