//===-- StackFrameListTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Process/Utility/HistoryThread.h"
#include "Plugins/Process/Utility/RegisterContextDummy.h"
#include "Plugins/Process/Utility/RegisterContextHistory.h"
#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"
#include <mutex>

using namespace lldb;
using namespace lldb_private;

namespace {

class DummyProcess : public Process {
public:
  DummyProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(const ProcessAddress &, void *, size_t,
                      Status &) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
};

class StackFrameListTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, TypeSystemClang, ObjectFileELF,
                plugin::dwarf::SymbolFileDWARF, platform_linux::PlatformLinux,
                ScriptInterpreterNone>
      subsystems;

public:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    ArchSpec arch("x86_64-pc-linux");
    PlatformSP platform_sp =
        platform_linux::PlatformLinux::CreateInstance(true, &arch);
    ASSERT_TRUE(platform_sp);
    Platform::SetHostPlatform(platform_sp);

    DebuggerSP debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(debugger_sp);
    PlatformSP target_platform_sp;
    ASSERT_TRUE(debugger_sp->GetTargetList()
                    .CreateTarget(*debugger_sp, "", arch, eLoadDependentsNo,
                                  target_platform_sp, m_target_sp)
                    .Success());
    ASSERT_TRUE(m_target_sp);

    llvm::Expected<TestFile> file =
        TestFile::fromYamlFile("inlined-function.yaml");
    ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
    m_file.emplace(std::move(*file));
    ModuleSP module_sp = std::make_shared<Module>(m_file->moduleSpec());
    ASSERT_TRUE(module_sp);
    m_target_sp->GetImages().Append(module_sp);
    bool changed = false;
    ASSERT_TRUE(module_sp->SetLoadAddress(*m_target_sp, 0,
                                          /*value_is_offset=*/true, changed));
  }

protected:
  std::optional<TestFile> m_file;
  TargetSP m_target_sp;
};

// Frames synthesized for an inlined scope share the concrete frame's PC, they
// should also share its "behaves like frame zero" behavior.
TEST_F(StackFrameListTest, InlineFramesInheritZerothFrameSymbolication) {
  // outer() inlines inner() starting at this offset into .text; see the
  // generation recipe in Inputs/inlined-function.yaml.
  const addr_t pc = 0xd;

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp =
      std::make_shared<DummyProcess>(m_target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  // HistoryPCType::Returns makes frame zero behave like the zeroth frame.
  ThreadSP thread_sp = std::make_shared<HistoryThread>(
      *process_sp, /*tid=*/0x1234, std::vector<addr_t>{pc});

  // An inline chain requires more than the concrete frame.
  ASSERT_GT(thread_sp->GetStackFrameCount(), 1u)
      << "no inline frames synthesized at pc " << pc;

  // The YAML object file creates a backtrace like this:
  //   frame #0: 0x000000000000000d inner() at inl.cpp:2:13 [inlined]
  //   frame #1: 0x000000000000000d outer() at inl.cpp:6:27

  for (uint32_t i = 0; i < thread_sp->GetStackFrameCount(); ++i) {
    StackFrameSP frame_sp = thread_sp->GetStackFrameAtIndex(i);
    ASSERT_TRUE(frame_sp);
    if (frame_sp->GetFrameCodeAddress().GetFileAddress() != pc)
      continue;
    EXPECT_EQ(frame_sp->GetFrameCodeAddressForSymbolication().GetFileAddress(),
              frame_sp->GetFrameCodeAddress().GetFileAddress())
        << "frame " << i << " symbolicates at an adjusted address";
  }
}
} // namespace

TEST_F(StackFrameListTest, SyntheticPCContextsOwnLogicalDescriptions) {
  auto process = std::make_shared<DummyProcess>(
      m_target_sp, Listener::MakeListener("synthetic-registers"));
  auto thread = std::make_shared<HistoryThread>(
      *process, 0x1234, std::vector<addr_t>{0x12345678});
  RegisterContextHistory history(*thread, 0, 8, 0x12345678);
  auto description = history.GetDescription();
  ASSERT_TRUE(description);
  const auto *pc =
      description->FindByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  ASSERT_NE(pc, nullptr);
  EXPECT_EQ(pc->name, "pc");
  EXPECT_EQ(pc->byte_size, 8u);
  ASSERT_EQ(description->GetGroups().size(), 1u);
  EXPECT_EQ(description->GetGroups()[0].registers,
            (std::vector<RegisterID>{pc->id}));
  EXPECT_EQ(history.GetBankSize(0), 0u);
  EXPECT_EQ(history.GetRegisterCount(), 1u);
  EXPECT_EQ(history.GetRegisterAtIndex(1), nullptr);
  RegisterValue value;
  ASSERT_TRUE(!llvm::errorToBool(
      history
          .ReadRegister((history.GetRegisterAtIndex(0)
                             ? (history.GetRegisterAtIndex(0))->id
                             : LLDB_INVALID_REGNUM))
          .moveInto(value)));
  EXPECT_EQ(value.GetAsUInt64(), 0x12345678u);

  RegisterContextDummy dummy(*thread, 0, 4);
  ASSERT_TRUE(dummy.GetDescription());
  EXPECT_EQ(dummy.GetDescription()->Find("pc")->byte_size, 4u);
  EXPECT_EQ(dummy.GetBankSize(0), 0u);
  ASSERT_TRUE(!llvm::errorToBool(
      dummy
          .ReadRegister((dummy.GetRegisterAtIndex(0)
                             ? (dummy.GetRegisterAtIndex(0))->id
                             : LLDB_INVALID_REGNUM))
          .moveInto(value)));
  EXPECT_EQ(value.GetAsUInt32(), UINT32_MAX);
}
