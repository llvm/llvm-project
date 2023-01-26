//===-- PlatformSiginfoTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <initializer_list>
#include <optional>
#include <tuple>

#include "Plugins/Platform/FreeBSD/PlatformFreeBSD.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Platform/NetBSD/PlatformNetBSD.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/ArchSpec.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::repro;

namespace {
class PlatformSiginfoTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, TypeSystemClang> subsystems;
  PlatformSP platform_sp;
  DebuggerSP debugger_sp;
  TargetSP target_sp;

public:
  CompilerType siginfo_type;

  void SetUp() override {
    platform_freebsd::PlatformFreeBSD::Initialize();
    platform_linux::PlatformLinux::Initialize();
    platform_netbsd::PlatformNetBSD::Initialize();
  }

  void TearDown() override {
    platform_netbsd::PlatformNetBSD::Terminate();
    platform_linux::PlatformLinux::Terminate();
    platform_freebsd::PlatformFreeBSD::Terminate();
  }

  typedef std::tuple<const char *, uint64_t, uint64_t> field_tuple;

  void ExpectField(const CompilerType &siginfo_type, field_tuple field) {
    const char *path;
    uint64_t offset, size;
    std::tie(path, offset, size) = field;

    SCOPED_TRACE(path);
    CompilerType field_type = siginfo_type;
    uint64_t total_offset = 0;
    for (auto field_name : llvm::split(path, '.')) {
      uint64_t bit_offset;
      ASSERT_NE(field_type.GetIndexOfFieldWithName(field_name.str().c_str(),
                                                   &field_type, &bit_offset),
                UINT32_MAX);
      total_offset += bit_offset;
    }

    EXPECT_EQ(total_offset, offset * 8);
    EXPECT_EQ(field_type.GetByteSize(nullptr), std::optional<uint64_t>(size));
  }

  void ExpectFields(const CompilerType &container,
                    std::initializer_list<field_tuple> fields) {
    for (auto x : fields)
      ExpectField(container, x);
  }

  void InitializeSiginfo(const std::string &triple) {
    ArchSpec arch(triple);

    switch (arch.GetTriple().getOS()) {
    case llvm::Triple::FreeBSD:
      platform_sp =
          platform_freebsd::PlatformFreeBSD::CreateInstance(true, &arch);
      break;
    case llvm::Triple::Linux:
      platform_sp = platform_linux::PlatformLinux::CreateInstance(true, &arch);
      break;
    case llvm::Triple::NetBSD:
      platform_sp =
          platform_netbsd::PlatformNetBSD::CreateInstance(true, &arch);
      break;
    default:
      llvm_unreachable("unknown ostype in triple");
    }
    Platform::SetHostPlatform(platform_sp);

    debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(debugger_sp);

    debugger_sp->GetTargetList().CreateTarget(
        *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
    ASSERT_TRUE(target_sp);

    siginfo_type = platform_sp->GetSiginfoType(arch.GetTriple());
  }
};

} // namespace

TEST_F(PlatformSiginfoTest, TestLinux_64bit) {
  for (std::string arch : {"x86_64", "aarch64", "powerpc64le"}) {
    SCOPED_TRACE(arch);
    InitializeSiginfo(arch + "-pc-linux-gnu");
    ASSERT_TRUE(siginfo_type);

    ExpectFields(siginfo_type,
                 {
                     {"si_signo", 0, 4},
                     {"si_errno", 4, 4},
                     {"si_code", 8, 4},
                     {"_sifields._kill.si_pid", 16, 4},
                     {"_sifields._kill.si_uid", 20, 4},
                     {"_sifields._timer.si_tid", 16, 4},
                     {"_sifields._timer.si_overrun", 20, 4},
                     {"_sifields._timer.si_sigval", 24, 8},
                     {"_sifields._rt.si_pid", 16, 4},
                     {"_sifields._rt.si_uid", 20, 4},
                     {"_sifields._rt.si_sigval", 24, 8},
                     {"_sifields._sigchld.si_pid", 16, 4},
                     {"_sifields._sigchld.si_uid", 20, 4},
                     {"_sifields._sigchld.si_status", 24, 4},
                     {"_sifields._sigchld.si_utime", 32, 8},
                     {"_sifields._sigchld.si_stime", 40, 8},
                     {"_sifields._sigfault.si_addr", 16, 8},
                     {"_sifields._sigfault.si_addr_lsb", 24, 2},
                     {"_sifields._sigfault._bounds._addr_bnd._lower", 32, 8},
                     {"_sifields._sigfault._bounds._addr_bnd._upper", 40, 8},
                     {"_sifields._sigfault._bounds._pkey", 32, 4},
                     {"_sifields._sigpoll.si_band", 16, 8},
                     {"_sifields._sigpoll.si_fd", 24, 4},
                     {"_sifields._sigsys._call_addr", 16, 8},
                     {"_sifields._sigsys._syscall", 24, 4},
                     {"_sifields._sigsys._arch", 28, 4},
                 });
  }
}

TEST_F(PlatformSiginfoTest, TestLinux_32bit) {
  for (std::string arch : {"i386", "armv7"}) {
    SCOPED_TRACE(arch);
    InitializeSiginfo(arch + "-pc-linux");
    ASSERT_TRUE(siginfo_type);

    ExpectFields(siginfo_type,
                 {
                     {"si_signo", 0, 4},
                     {"si_errno", 4, 4},
                     {"si_code", 8, 4},
                     {"_sifields._kill.si_pid", 12, 4},
                     {"_sifields._kill.si_uid", 16, 4},
                     {"_sifields._timer.si_tid", 12, 4},
                     {"_sifields._timer.si_overrun", 16, 4},
                     {"_sifields._timer.si_sigval", 20, 4},
                     {"_sifields._rt.si_pid", 12, 4},
                     {"_sifields._rt.si_uid", 16, 4},
                     {"_sifields._rt.si_sigval", 20, 4},
                     {"_sifields._sigchld.si_pid", 12, 4},
                     {"_sifields._sigchld.si_uid", 16, 4},
                     {"_sifields._sigchld.si_status", 20, 4},
                     {"_sifields._sigchld.si_utime", 24, 4},
                     {"_sifields._sigchld.si_stime", 28, 4},
                     {"_sifields._sigfault.si_addr", 12, 4},
                     {"_sifields._sigfault.si_addr_lsb", 16, 2},
                     {"_sifields._sigfault._bounds._addr_bnd._lower", 20, 4},
                     {"_sifields._sigfault._bounds._addr_bnd._upper", 24, 4},
                     {"_sifields._sigfault._bounds._pkey", 20, 4},
                     {"_sifields._sigpoll.si_band", 12, 4},
                     {"_sifields._sigpoll.si_fd", 16, 4},
                     {"_sifields._sigsys._call_addr", 12, 4},
                     {"_sifields._sigsys._syscall", 16, 4},
                     {"_sifields._sigsys._arch", 20, 4},
                 });
  }
}

TEST_F(PlatformSiginfoTest, TestFreeBSD_64bit) {
  for (std::string arch : {"x86_64", "aarch64"}) {
    SCOPED_TRACE(arch);
    InitializeSiginfo("x86_64-unknown-freebsd13.0");
    ASSERT_TRUE(siginfo_type);

    ExpectFields(siginfo_type, {
                                   {"si_signo", 0, 4},
                                   {"si_errno", 4, 4},
                                   {"si_code", 8, 4},
                                   {"si_pid", 12, 4},
                                   {"si_uid", 16, 4},
                                   {"si_status", 20, 4},
                                   {"si_addr", 24, 8},
                                   {"si_value", 32, 8},
                                   {"_reason._fault._trapno", 40, 4},
                                   {"_reason._timer._timerid", 40, 4},
                                   {"_reason._timer._overrun", 44, 4},
                                   {"_reason._mesgq._mqd", 40, 4},
                                   {"_reason._poll._band", 40, 8},
                               });
  }
}

TEST_F(PlatformSiginfoTest, TestFreeBSD_32bit) {
  for (std::string arch : {"i386"}) {
    SCOPED_TRACE(arch);
    InitializeSiginfo(arch + "-unknown-freebsd13.0");
    ASSERT_TRUE(siginfo_type);

    ExpectFields(siginfo_type, {
                                   {"si_signo", 0, 4},
                                   {"si_errno", 4, 4},
                                   {"si_code", 8, 4},
                                   {"si_pid", 12, 4},
                                   {"si_uid", 16, 4},
                                   {"si_status", 20, 4},
                                   {"si_addr", 24, 4},
                                   {"si_value", 28, 4},
                                   {"_reason._fault._trapno", 32, 4},
                                   {"_reason._timer._timerid", 32, 4},
                                   {"_reason._timer._overrun", 36, 4},
                                   {"_reason._mesgq._mqd", 32, 4},
                                   {"_reason._poll._band", 32, 4},
                               });
  }
}

TEST_F(PlatformSiginfoTest, TestNetBSD_64bit) {
  for (std::string arch : {"x86_64"}) {
    SCOPED_TRACE(arch);
    InitializeSiginfo(arch + "-unknown-netbsd9.0");
    ASSERT_TRUE(siginfo_type);

    ExpectFields(
        siginfo_type,
        {
            {"_info._signo", 0, 4},
            {"_info._code", 4, 4},
            {"_info._errno", 8, 4},
            {"_info._reason._rt._pid", 16, 4},
            {"_info._reason._rt._uid", 20, 4},
            {"_info._reason._rt._value", 24, 8},
            {"_info._reason._child._pid", 16, 4},
            {"_info._reason._child._uid", 20, 4},
            {"_info._reason._child._status", 24, 4},
            {"_info._reason._child._utime", 28, 4},
            {"_info._reason._child._stime", 32, 4},
            {"_info._reason._fault._addr", 16, 8},
            {"_info._reason._fault._trap", 24, 4},
            {"_info._reason._fault._trap2", 28, 4},
            {"_info._reason._fault._trap3", 32, 4},
            {"_info._reason._poll._band", 16, 8},
            {"_info._reason._poll._fd", 24, 4},
            {"_info._reason._syscall._sysnum", 16, 4},
            {"_info._reason._syscall._retval", 20, 8},
            {"_info._reason._syscall._error", 28, 4},
            {"_info._reason._syscall._args", 32, 64},
            {"_info._reason._ptrace_state._pe_report_event", 16, 4},
            {"_info._reason._ptrace_state._option._pe_other_pid", 20, 4},
            {"_info._reason._ptrace_state._option._pe_lwp", 20, 4},
        });
  }
}

TEST_F(PlatformSiginfoTest, TestNetBSD_32bit) {
  for (std::string arch : {"i386"}) {
    SCOPED_TRACE(arch);
    InitializeSiginfo(arch + "-unknown-netbsd9.0");
    ASSERT_TRUE(siginfo_type);

    ExpectFields(
        siginfo_type,
        {
            {"_info._signo", 0, 4},
            {"_info._code", 4, 4},
            {"_info._errno", 8, 4},
            {"_info._reason._rt._pid", 12, 4},
            {"_info._reason._rt._uid", 16, 4},
            {"_info._reason._rt._value", 20, 4},
            {"_info._reason._child._pid", 12, 4},
            {"_info._reason._child._uid", 16, 4},
            {"_info._reason._child._status", 20, 4},
            {"_info._reason._child._utime", 24, 4},
            {"_info._reason._child._stime", 28, 4},
            {"_info._reason._fault._addr", 12, 4},
            {"_info._reason._fault._trap", 16, 4},
            {"_info._reason._fault._trap2", 20, 4},
            {"_info._reason._fault._trap3", 24, 4},
            {"_info._reason._poll._band", 12, 4},
            {"_info._reason._poll._fd", 16, 4},
            {"_info._reason._syscall._sysnum", 12, 4},
            {"_info._reason._syscall._retval", 16, 8},
            {"_info._reason._syscall._error", 24, 4},
            {"_info._reason._syscall._args", 28, 64},
            {"_info._reason._ptrace_state._pe_report_event", 12, 4},
            {"_info._reason._ptrace_state._option._pe_other_pid", 16, 4},
            {"_info._reason._ptrace_state._option._pe_lwp", 16, 4},
        });
  }
}
