//===-- ThreadElfCoreTest.cpp ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Plugins/Process/elf-core/ThreadElfCore.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Process/elf-core/RegisterContextELFCore_arm64.h"
#include "Plugins/Process/elf-core/RegisterContextELFCore_loongarch64.h"
#include "Plugins/Process/elf-core/RegisterContextELFCore_riscv32.h"
#include "Plugins/Process/elf-core/RegisterContextELFCore_riscv64.h"
#include "Plugins/Process/elf-core/RegisterContextELFCore_x86.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_64.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Listener.h"
#include "llvm/Support/Endian.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <memory>
#include <mutex>
#include <sys/resource.h>
#include <unistd.h>

#ifndef HAVE_GETTID
#include <sys/syscall.h>
pid_t gettid() { return ((pid_t)syscall(SYS_gettid)); }
#endif


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

namespace {

struct ElfCoreTest : public testing::Test {
  static void SetUpTestCase() {
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  }
  static void TearDownTestCase() {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

struct DummyProcess : public Process {
  DummyProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {
    SetID(getpid());
  }

  bool CanDebug(lldb::TargetSP target, bool plugin_specified_by_name) override {
    return true;
  }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(const ProcessAddress &process_addr, void *buf,
                      size_t size, Status &error) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
};

struct DummyThread : public Thread {
  using Thread::Thread;

  ~DummyThread() override { DestroyThread(); }

  void RefreshStateAfterStop() override {}

  lldb::ThreadRegisterContextSP GetRegisterContext() override { return nullptr; }

  lldb::ThreadRegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override {
    return nullptr;
  }

  bool CalculateStopInfo() override { return false; }
};

lldb::TargetSP CreateTarget(lldb::DebuggerSP &debugger_sp, ArchSpec &arch) {
  lldb::PlatformSP platform_sp;
  lldb::TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  return target_sp;
}

lldb::ThreadSP CreateThread(lldb::ProcessSP &process_sp) {
  lldb::ThreadSP thread_sp =
      std::make_shared<DummyThread>(*process_sp.get(), gettid());
  if (thread_sp == nullptr) {
    return nullptr;
  }
  process_sp->GetThreadList().AddThread(thread_sp);

  return thread_sp;
}

} // namespace

TEST_F(ElfCoreTest, PopulatePrpsInfoTest) {
  ArchSpec arch{HostInfo::GetTargetTriple()};
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  lldb::TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  auto prpsinfo_opt = ELFLinuxPrPsInfo::Populate(process_sp);
  ASSERT_TRUE(prpsinfo_opt.has_value());
  ASSERT_EQ(prpsinfo_opt->pr_pid, getpid());
  ASSERT_EQ(prpsinfo_opt->pr_state, 0);
  ASSERT_EQ(prpsinfo_opt->pr_sname, 'R');
  ASSERT_EQ(prpsinfo_opt->pr_zomb, 0);
  int priority = getpriority(PRIO_PROCESS, getpid());
  if (priority == -1)
    ASSERT_EQ(errno, 0);
  ASSERT_EQ(prpsinfo_opt->pr_nice, priority);
  ASSERT_EQ(prpsinfo_opt->pr_flag, 0UL);
  ASSERT_EQ(prpsinfo_opt->pr_uid, getuid());
  ASSERT_EQ(prpsinfo_opt->pr_gid, getgid());
  ASSERT_EQ(prpsinfo_opt->pr_pid, getpid());
  ASSERT_EQ(prpsinfo_opt->pr_ppid, getppid());
  ASSERT_EQ(prpsinfo_opt->pr_pgrp, getpgrp());
  ASSERT_EQ(prpsinfo_opt->pr_sid, getsid(getpid()));
  ASSERT_EQ(std::string{prpsinfo_opt->pr_fname}, "ProcessElfCoreT");
  ASSERT_TRUE(std::string{prpsinfo_opt->pr_psargs}.empty());
  lldb_private::ProcessInstanceInfo info;
  ASSERT_TRUE(process_sp->GetProcessInfo(info));
  const char *args[] = {"a.out", "--foo=bar", "--baz=boo", nullptr};
  info.SetArguments(args, true);
  prpsinfo_opt =
      ELFLinuxPrPsInfo::Populate(info, lldb::StateType::eStateStopped);
  ASSERT_TRUE(prpsinfo_opt.has_value());
  ASSERT_EQ(prpsinfo_opt->pr_pid, getpid());
  ASSERT_EQ(prpsinfo_opt->pr_state, 3);
  ASSERT_EQ(prpsinfo_opt->pr_sname, 'T');
  ASSERT_EQ(std::string{prpsinfo_opt->pr_fname}, "a.out");
  ASSERT_FALSE(std::string{prpsinfo_opt->pr_psargs}.empty());
  ASSERT_EQ(std::string{prpsinfo_opt->pr_psargs}, "a.out --foo=bar --baz=boo");
}

TEST_F(ElfCoreTest, PopulatePrStatusTest) {
  ArchSpec arch{HostInfo::GetTargetTriple()};
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  lldb::TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  lldb::ThreadSP thread_sp = CreateThread(process_sp);
  ASSERT_TRUE(thread_sp);
  auto prstatus_opt = ELFLinuxPrStatus::Populate(thread_sp);
  ASSERT_TRUE(prstatus_opt.has_value());
  ASSERT_EQ(prstatus_opt->si_signo, 0);
  ASSERT_EQ(prstatus_opt->si_code, 0);
  ASSERT_EQ(prstatus_opt->si_errno, 0);
  ASSERT_EQ(prstatus_opt->pr_cursig, 0);
  ASSERT_EQ(prstatus_opt->pr_sigpend, 0UL);
  ASSERT_EQ(prstatus_opt->pr_sighold, 0UL);
  ASSERT_EQ(prstatus_opt->pr_pid, static_cast<uint32_t>(gettid()));
  ASSERT_EQ(prstatus_opt->pr_ppid, static_cast<uint32_t>(getppid()));
  ASSERT_EQ(prstatus_opt->pr_pgrp, static_cast<uint32_t>(getpgrp()));
  ASSERT_EQ(prstatus_opt->pr_sid, static_cast<uint32_t>(getsid(gettid())));
}

TEST_F(ElfCoreTest, NoStopReasonWhenNoPrStatus) {
  ArchSpec arch{HostInfo::GetTargetTriple()};
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  lldb::TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  lldb::ThreadSP thread_sp = CreateThread(process_sp);
  ASSERT_TRUE(thread_sp);
  ASSERT_FALSE(thread_sp->ThreadStoppedForAReason());
}

TEST_F(ElfCoreTest, RegisterLayoutsPreserveX86GroupsAndUnavailableBanks) {
  ArchSpec arch("x86_64-pc-linux");
  auto debugger = Debugger::CreateInstance();
  ASSERT_TRUE(debugger);
  auto target = CreateTarget(debugger, arch);
  ASSERT_TRUE(target);
  lldb::ProcessSP process = std::make_shared<DummyProcess>(
      target, Listener::MakeListener("register-model"));
  auto thread = CreateThread(process);
  ASSERT_TRUE(thread);

  auto model = CreateLinuxX86_64RegisterLayout(arch);
  auto description = model.GetDescription();
  std::vector<uint8_t> gp(model.FindBank(x86_layout::GP)->byte_size, 0);
  const auto *rax = description->Find("rax");
  ASSERT_NE(rax, nullptr);
  ASSERT_THAT_ERROR(model.Write(rax->id,
                                RegisterValue(uint64_t(0x123456789abcdef0)),
                                {{x86_layout::GP, gp}}),
                    llvm::Succeeded());
  DataExtractor general(gp.data(), gp.size(), lldb::eByteOrderLittle, 8);
  RegisterContextELFCore_x86 context(*thread, std::move(model), general, {});

  // The context publishes its own presentation with unavailable groups hidden.
  // Logical register identities and definitions remain those of the layout.
  ASSERT_EQ(context.GetDescription()->GetRegisters().size(),
            description->GetRegisters().size());
  for (const auto &reg : description->GetRegisters()) {
    const auto *actual = context.GetDescription()->Find(reg.id);
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(actual->name, reg.name);
    EXPECT_EQ(actual->byte_size, reg.byte_size);
    EXPECT_EQ(actual->numbers, reg.numbers);
  }
  EXPECT_EQ(context.GetRegisterGroupCount(), 2u);
  EXPECT_EQ(context.GetRegisterGroup(2), nullptr);
  EXPECT_EQ(context.GetUserRegisterCount(),
            description->GetGroups()[0].registers.size() +
                description->GetGroups()[1].registers.size());
  // Use the model's Linux numbering, including FS/GS base registers.
  const auto *base = context.GetRegisterByName("fs_base");
  ASSERT_NE(base, nullptr);
  EXPECT_EQ(base->numbers[lldb::eRegisterKindLLDB],
            description->Find("fs_base")->numbers[lldb::eRegisterKindLLDB]);
  RegisterValue value;
  ASSERT_TRUE(!llvm::errorToBool(
      context
          .ReadRegister((context.GetRegisterByName("eax")
                             ? (context.GetRegisterByName("eax"))->id
                             : LLDB_INVALID_REGNUM))
          .moveInto(value)));
  EXPECT_EQ(value.GetAsUInt32(), 0x9abcdef0u);
  EXPECT_FALSE(!llvm::errorToBool(
      context
          .ReadRegister((context.GetRegisterByName("xmm0")
                             ? (context.GetRegisterByName("xmm0"))->id
                             : LLDB_INVALID_REGNUM))
          .moveInto(value)));
  EXPECT_FALSE(!llvm::errorToBool(
      context.WriteRegister((context.GetRegisterByName("rax")
                                 ? (context.GetRegisterByName("rax"))->id
                                 : LLDB_INVALID_REGNUM),
                            value)));

  // An FXSAVE note restores the historical AVX group exposure. Values whose
  // additional bank was not saved remain unavailable to the shared codec.
  std::vector<uint8_t> fp(sizeof(FXSAVE), 0);
  CoreNote note;
  note.info.n_type = llvm::ELF::NT_FPREGSET;
  note.data = DataExtractor(fp.data(), fp.size(), lldb::eByteOrderLittle, 8);
  RegisterContextELFCore_x86 with_fp(
      *thread, CreateLinuxX86_64RegisterLayout(arch), general, {note});
  EXPECT_EQ(with_fp.GetRegisterGroupCount(), 3u);
  EXPECT_STREQ(with_fp.GetRegisterGroup(2)->short_name.c_str(), "avx");
  EXPECT_TRUE(!llvm::errorToBool(
      with_fp
          .ReadRegister((with_fp.GetRegisterByName("xmm0")
                             ? (with_fp.GetRegisterByName("xmm0"))->id
                             : LLDB_INVALID_REGNUM))
          .moveInto(value)));
  EXPECT_FALSE(!llvm::errorToBool(
      with_fp
          .ReadRegister((with_fp.GetRegisterByName("ymm0")
                             ? (with_fp.GetRegisterByName("ymm0"))->id
                             : LLDB_INVALID_REGNUM))
          .moveInto(value)));
}

TEST_F(ElfCoreTest, RejectsUnsupportedRegisterOperatingSystems) {
  ArchSpec arch{HostInfo::GetTargetTriple()};
  auto debugger = Debugger::CreateInstance();
  ASSERT_TRUE(debugger);
  auto target = CreateTarget(debugger, arch);
  ASSERT_TRUE(target);
  lldb::ProcessSP process = std::make_shared<DummyProcess>(
      target, Listener::MakeListener("unsupported-core-registers"));
  auto thread = CreateThread(process);
  ASSERT_TRUE(thread);
  DataExtractor general;
  EXPECT_THAT_EXPECTED(
      RegisterContextELFCore_arm64::Create(
          *thread, ArchSpec("aarch64-unknown-unknown"), general, {}),
      llvm::Failed());
  EXPECT_THAT_EXPECTED(
      RegisterContextELFCore_loongarch64::Create(
          *thread, ArchSpec("loongarch64-unknown-freebsd"), general, {}),
      llvm::Failed());
  EXPECT_THAT_EXPECTED(
      RegisterContextELFCore_riscv64::Create(
          *thread, ArchSpec("riscv64-unknown-netbsd"), general, {}),
      llvm::Failed());
  EXPECT_THAT_EXPECTED(
      RegisterContextELFCore_riscv32::Create(
          *thread, ArchSpec("riscv32-unknown-linux"), general, {}),
      llvm::Failed());
}

TEST_F(ElfCoreTest, BSDArm64CoreNotesUseTheirOwnFPControlOrder) {
  struct TestCase {
    const char *triple;
    uint32_t note_type;
    uint32_t fpcr;
    uint32_t fpsr;
  };
  for (const auto &test : {TestCase{"aarch64-unknown-netbsd",
                                    NETBSD::AARCH64::NT_FPREGS, 0x1234, 0x5678},
                           TestCase{"aarch64-unknown-openbsd",
                                    OPENBSD::NT_FPREGS, 0x5678, 0x1234}}) {
    SCOPED_TRACE(test.triple);
    ArchSpec arch(test.triple);
    auto debugger = Debugger::CreateInstance();
    ASSERT_TRUE(debugger);
    auto target = CreateTarget(debugger, arch);
    ASSERT_TRUE(target);
    lldb::ProcessSP process = std::make_shared<DummyProcess>(
        target, Listener::MakeListener("bsd-arm64-registers"));
    auto thread = CreateThread(process);
    ASSERT_TRUE(thread);
    std::vector<uint8_t> gp(280, 0), fp(528, 0);
    llvm::support::endian::write32le(fp.data() + 512, 0x1234);
    llvm::support::endian::write32le(fp.data() + 516, 0x5678);
    DataExtractor general(gp.data(), gp.size(), lldb::eByteOrderLittle, 8);
    CoreNote note{};
    note.info.n_type = test.note_type;
    note.data = DataExtractor(fp.data(), fp.size(), lldb::eByteOrderLittle, 8);
    auto result =
        RegisterContextELFCore_arm64::Create(*thread, arch, general, {note});
    ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
    auto &context = *result;
    EXPECT_EQ(context->GetBankSize(0), gp.size());
    EXPECT_EQ(context->GetBankSize(1), fp.size());
    const auto *fpcr = context->GetRegisterByName("fpcr");
    const auto *fpsr = context->GetRegisterByName("fpsr");
    ASSERT_NE(fpcr, nullptr);
    ASSERT_NE(fpsr, nullptr);
    auto control = context->ReadRegister(fpcr->id);
    auto status = context->ReadRegister(fpsr->id);
    ASSERT_THAT_EXPECTED(control, llvm::Succeeded());
    ASSERT_THAT_EXPECTED(status, llvm::Succeeded());
    EXPECT_EQ(control->GetAsUInt64(), test.fpcr);
    EXPECT_EQ(status->GetAsUInt64(), test.fpsr);
  }
}

TEST_F(ElfCoreTest, LinuxRISCV64DoesNotExposeTruncatedFPNotes) {
  ArchSpec arch("riscv64-unknown-linux");
  auto debugger = Debugger::CreateInstance();
  ASSERT_TRUE(debugger);
  auto target = CreateTarget(debugger, arch);
  ASSERT_TRUE(target);
  lldb::ProcessSP process = std::make_shared<DummyProcess>(
      target, Listener::MakeListener("riscv64-core-no-fp"));
  auto thread = CreateThread(process);
  ASSERT_TRUE(thread);
  std::vector<uint8_t> gp(256, 0), fp(264, 0);
  DataExtractor general(gp.data(), gp.size(), lldb::eByteOrderLittle, 8);
  for (size_t size : {0u, 8u, 263u}) {
    CoreNote note{};
    note.info.n_type = llvm::ELF::NT_FPREGSET;
    note.data = DataExtractor(fp.data(), size, lldb::eByteOrderLittle, 8);
    auto result =
        RegisterContextELFCore_riscv64::Create(*thread, arch, general, {note});
    ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
    auto &context = *result;
    EXPECT_EQ(context->GetDescription()->FindGroup("fpr"), nullptr);
    EXPECT_THAT_EXPECTED(context->ReadRegister(reg_ids::fpr_f1),
                         llvm::Failed());
  }
}
