//===-- PlatformAIX.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformAIX.h"
#include "lldb/Host/Config.h"
#include <cstdio>
#if LLDB_ENABLE_POSIX
#include <sys/utsname.h>
#endif
#include "Utility/PPC64_DWARF_Registers.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

// Use defined constants from AIX mman.h for use when targeting remote aix
// systems even when host has different values.

// For remotely cross debugging aix
constexpr int MapVariable = 0x0;
constexpr int MapPrivate = 0x2;
constexpr int MapAnonymous = 0x10;
#if defined(_AIX)
#include <sys/mman.h>
static_assert(MapVariable == MAP_VARIABLE);
static_assert(MapPrivate == MAP_PRIVATE);
static_assert(MapAnonymous == MAP_ANONYMOUS);
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_aix;

LLDB_PLUGIN_DEFINE(PlatformAIX)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformAIX::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch=({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create = force || (arch && arch->IsValid() &&
                          arch->GetTriple().getOS() == llvm::Triple::AIX);
  LLDB_LOG(log, "create = {0}", create);
  if (create) {
    return PlatformSP(new PlatformAIX(false));
  }
  return PlatformSP();
}

llvm::StringRef PlatformAIX::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local AIX user platform plug-in.";
  return "Remote AIX user platform plug-in.";
}

void PlatformAIX::Initialize() {
  PlatformPOSIX::Initialize();

  if (g_initialize_count++ == 0) {
#ifdef _AIX
    PlatformSP default_platform_sp(new PlatformAIX(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformAIX::GetPluginNameStatic(false),
        PlatformAIX::GetPluginDescriptionStatic(false),
        PlatformAIX::CreateInstance, nullptr);
  }
}

void PlatformAIX::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformAIX::CreateInstance);

  PlatformPOSIX::Terminate();
}

PlatformAIX::PlatformAIX(bool is_host) : PlatformPOSIX(is_host) {
  if (is_host) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    m_supported_architectures.push_back(hostArch);
  } else {
    m_supported_architectures =
        CreateArchList({llvm::Triple::ppc64}, llvm::Triple::AIX);
  }
}

std::vector<ArchSpec>
PlatformAIX::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetSupportedArchitectures(process_host_arch);
  return m_supported_architectures;
}

void PlatformAIX::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#if LLDB_ENABLE_POSIX
  // Display local kernel information only when we are running in host mode.
  // Otherwise, we would end up printing non-AIX information (when running on
  // Mac OS for example).
  if (IsHost()) {
    struct utsname un;

    if (uname(&un))
      return;

    strm.Printf("    Kernel: %s\n", un.sysname);
    strm.Printf("   Release: %s\n", un.release);
    strm.Printf("   Version: %s\n", un.version);
  }
#endif
}

void PlatformAIX::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP PlatformAIX::GetTrapHandlerUnwindPlan(const ArchSpec &arch,
                                                         ConstString name) {
  return {};
}

// AIX signal trampolines live at fixed low-memory addresses in the kernel
// 0x4ac0 to 0x5000
static constexpr lldb::addr_t kAIXSigTrampolineStart = 0x4ac0;
static constexpr lldb::addr_t kAIXSigTrampolineEnd   = 0x5000;

bool PlatformAIX::IsTrapHandlerAddress(lldb::addr_t pc) const {
  bool result = pc >= kAIXSigTrampolineStart && pc < kAIXSigTrampolineEnd;
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log,
           "IsTrapHandlerAddress(pc=0x{0:x}): range=[0x{1:x}, 0x{2:x}) "
           "result={3}",
           pc, kAIXSigTrampolineStart, kAIXSigTrampolineEnd, result);
  return result;
}

// Build the unwind plan for sig_epilog64.
//
// At the point sig_epilog64 executes, r14 holds a pointer to ucontext64_t
//
// ucontext64_t struct 
//   +0   int __sc_onstack                 (4 bytes + 4 pad = 8)
//   +8   sigset64_t uc_sigmask            (uint64_t[4] = 32 bytes)
//   +40  int __sc_uerror                  (4 bytes + 4 pad = 8)
//   +48  mcontext64_t uc_mcontext64       (__context64)
//
// struct __context64
//   +0   gpr[32]   (32 x 8 bytes)
//   +256 msr
//   +264 iar
//   +272 lr
//   +280 ctr
//   +288 cr    (4 bytes)
//   +292 xer   (4 bytes)
//
// Offsets from r14 (CFA):
//   GPR[n] = CFA + 48 + n*8
//   PC     = CFA + 48 + 264 = CFA + 312
//   LR     = CFA + 48 + 272 = CFA + 320
//   CTR    = CFA + 48 + 280 = CFA + 328
//   CR     = CFA + 48 + 288 = CFA + 336
//   XER    = CFA + 48 + 292 = CFA + 340
static lldb::UnwindPlanSP GetPPC64AIXUnwindPlan64() {
  using namespace ppc64_dwarf;

  UnwindPlan::Row row;
  // CFA = r14 + 0  -  ucontext64_t
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_r14_ppc64, 0);

  // Restore gprs 
  for (int n = 0; n < 32; ++n)
    row.SetRegisterLocationToAtCFAPlusOffset(dwarf_r0_ppc64 + n,
                                             48 + n * 8, false);
  
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_pc_ppc64,  312, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_lr_ppc64,  320, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_ctr_ppc64, 328, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_cr_ppc64,  336, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_xer_ppc64, 340, false);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("AIX ppc64 64-bit signal handler unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolYes);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  plan_sp->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  return plan_sp;
}

// Build the unwind plan for sig_epilog32.
//
// r14 holds a pointer to ucontext32_t
//
// ucontext32_t
//   +0   int __sc_onstack                 (4 bytes)
//   +4   sigset32_t uc_sigmask            (2 x uint = 8 bytes)
//   +12  int __sc_uerror                  (4 bytes)
//   +16  mcontext32_t uc_mcontext         (mstsave32)
//
// struct mstsave32
//   +0   prev, kjmpbuf, stackfix          (3 x __ptr32 = 12)
//   +12  intpri, backt, rsvd[2]           (4)
//   +16  curid                            (4)
//   +20  excp_type                        (4)
//   +24  iar                              (4)
//   +28  msr                              (4)
//   +32  cr                               (4)
//   +36  lr                               (4)
//   ...
//   +140 adspace32_t as (4 + 16*4 = 68 bytes)
//   +208 gpr[32]  (32 x 4 bytes), gpr[n] at +208 + n*4
//
// Offsets from r14 (CFA):
//   GPR[n] = CFA + 16 + 208 + n*4 = CFA + 224 + n*4
//   PC     = CFA + 16 + 24  = CFA + 40
//   LR     = CFA + 16 + 36  = CFA + 52
//   CTR    = CFA + 16 + 40  = CFA + 56
//   CR     = CFA + 16 + 32  = CFA + 48
//   XER    = CFA + 16 + 44  = CFA + 60
static lldb::UnwindPlanSP GetPPC64AIXUnwindPlan32() {
  using namespace ppc64_dwarf;

  UnwindPlan::Row row;
  // CFA = r14 + 0 - ucontext32_t
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_r14_ppc64, 0);

  // Restore GPRs
  for (int n = 0; n < 32; ++n)
    row.SetRegisterLocationToAtCFAPlusOffset(dwarf_r0_ppc64 + n,
                                             224 + n * 4, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_pc_ppc64,  40, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_lr_ppc64,  52, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_ctr_ppc64, 56, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_cr_ppc64,  48, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_xer_ppc64, 60, false);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("AIX ppc64 32-bit signal handler unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolYes);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  plan_sp->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  return plan_sp;
}

lldb::UnwindPlanSP PlatformAIX::GetTrapHandlerUnwindPlan(const ArchSpec &arch,
                                                         lldb::addr_t pc) {
  if (arch.GetTriple().getArch() == llvm::Triple::ppc64)
      return GetPPC64AIXUnwindPlan64();
  else if (arch.GetTriple().getArch() == llvm::Triple::ppc)
      return GetPPC64AIXUnwindPlan32();
  return {};

}

MmapArgList PlatformAIX::GetMmapArgumentList(const ArchSpec &arch, addr_t addr,
                                             addr_t length, unsigned prot,
                                             unsigned flags, addr_t fd,
                                             addr_t offset) {
  unsigned flags_platform = MapVariable;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MapPrivate;
  if (flags & eMmapFlagsAnon)
    flags_platform |= MapAnonymous;

  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}

CompilerType PlatformAIX::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}
