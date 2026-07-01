//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformFreeBSDKernel.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/TargetParser/Triple.h"

#include <set>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_freebsdkernel;

LLDB_PLUGIN_DEFINE(PlatformFreeBSDKernel)

static uint32_t g_initialize_count = 0;

PlatformFreeBSDKernel::PlatformFreeBSDKernel() : Platform(/*is_host=*/false) {
  const llvm::Triple::ArchType arches[] = {
      llvm::Triple::arm,     // arm32 (legacy)
      llvm::Triple::aarch64, // arm64
      llvm::Triple::ppc64le, // powerpc64le
      llvm::Triple::riscv64, // riscv64
      llvm::Triple::x86,     // i386 (legacy)
      llvm::Triple::x86_64,  // amd64
  };

  for (auto arch : arches) {
    ArchSpec spec;
    spec.SetTriple(llvm::Triple(llvm::Triple::getArchTypeName(arch), "unknown",
                                "freebsd"));
    m_supported_architectures.push_back(spec);
  }
}

void PlatformFreeBSDKernel::Initialize() {
  Platform::Initialize();
  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance,
                                  nullptr);
  }
}

void PlatformFreeBSDKernel::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(CreateInstance);
  }
  Platform::Terminate();
}

lldb::PlatformSP PlatformFreeBSDKernel::CreateInstance(bool force,
                                                       const ArchSpec *arch) {
  // PlatformFreeBSDKernel is never auto-selected. ProcessFreeBSDKernelCore sets
  // this platform explicitly (force=true).
  if (!force)
    return nullptr;
  return std::make_shared<PlatformFreeBSDKernel>();
}

void PlatformFreeBSDKernel::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);
  strm.Printf("  Kernel Mode: yes\n");
}

std::vector<ArchSpec> PlatformFreeBSDKernel::GetSupportedArchitectures(
    const ArchSpec &process_host_arch) {
  return m_supported_architectures;
}

lldb::UnwindPlanSP
PlatformFreeBSDKernel::GetTrapHandlerUnwindPlan(const ArchSpec &arch,
                                                ConstString name) {
  switch (arch.GetMachine()) {
  case llvm::Triple::aarch64:
    return GetTrapframeUnwindPlan_arm64(name);
  case llvm::Triple::ppc64le:
    return GetTrapframeUnwindPlan_ppc64le(name);
  case llvm::Triple::riscv64:
    return GetTrapframeUnwindPlan_riscv64(name);
  case llvm::Triple::x86_64:
    return GetTrapframeUnwindPlan_x86_64(name);

  // UnwindPlan is not implemented for the archs below as they are not
  // expressible as a static UnwindPlan.
  case llvm::Triple::arm:
    // SP/LR offsets depend on saved PSR mode bits (runtime memory read
    // required.
  case llvm::Triple::x86:
    // Trapframe base depends on stub identity, kernel version opcode probe, and
    // CPL of the interrupted context.
  default:
    return {};
  }
}

void PlatformFreeBSDKernel::CalculateTrapHandlerSymbolNames() {
  // Intentionally empty. All trap handler names are populated in
  // PopulateTrapHandlerNames() once the target architecture is known.
  // This override exists only to suppress the default implementation.
}

void PlatformFreeBSDKernel::PopulateTrapHandlerNames(Target &target) {
  ModuleSP kernel_module = target.GetExecutableModule();
  if (!kernel_module)
    return;

  const llvm::Triple::ArchType arch =
      target.GetArchitecture().GetTriple().getArch();

  // List trapframe sniffers from
  // https://cgit.freebsd.org/ports/tree/devel/gdb/files/kgdb/<arch>-kern.c
  switch (arch) {
  case llvm::Triple::aarch64:
    // From aarch64_fbsd_trapframe_sniffer in kgdb.
    m_trap_handlers.push_back(ConstString("handle_el1h_sync"));
    m_trap_handlers.push_back(ConstString("handle_el1h_irq"));
    m_trap_handlers.push_back(ConstString("handle_el0_sync"));
    m_trap_handlers.push_back(ConstString("handle_el0_irq"));
    m_trap_handlers.push_back(ConstString("handle_el0_error"));
    m_trap_handlers.push_back(ConstString("fork_trampoline"));
    return;

  case llvm::Triple::arm:
    // From arm_fbsd_trapframe_sniffer in kgdb.
    m_trap_handlers.push_back(ConstString("data_abort_entry"));
    m_trap_handlers.push_back(ConstString("prefetch_abort_entry"));
    m_trap_handlers.push_back(ConstString("undefined_entry"));
    m_trap_handlers.push_back(ConstString("exception_exit"));
    m_trap_handlers.push_back(ConstString("irq_entry"));
    m_trap_handlers.push_back(ConstString("swi_entry"));
    m_trap_handlers.push_back(ConstString("swi_exit"));
    return;

  case llvm::Triple::ppc64le:
    // From ppcfbsd_trapframe_sniffer in kgdb.
    m_trap_handlers.push_back(ConstString("trapagain"));
    m_trap_handlers.push_back(ConstString("trapexit"));
    m_trap_handlers.push_back(ConstString("dbtrap"));
    break;

  case llvm::Triple::riscv64:
    // From riscv_fbsd_trapframe_sniffer in kgdb.
    m_trap_handlers.push_back(ConstString("cpu_exception_handler_user"));
    m_trap_handlers.push_back(ConstString("cpu_exception_handler_supervisor"));
    return;

  case llvm::Triple::x86:
    // Fixed names from i386fbsd_trapframe_sniffer in kgdb.
    m_trap_handlers.push_back(ConstString("calltrap"));
    m_trap_handlers.push_back(ConstString("fork_trampoline"));
    break;

  case llvm::Triple::x86_64:
    // Fixed names from amd64fbsd_trapframe_sniffer in kgdb.
    m_trap_handlers.push_back(ConstString("calltrap"));
    m_trap_handlers.push_back(ConstString("fast_syscall_common"));
    m_trap_handlers.push_back(ConstString("fork_trampoline"));
    m_trap_handlers.push_back(ConstString("mchk_calltrap"));
    m_trap_handlers.push_back(ConstString("nmi_calltrap"));
    break;

  default:
    assert(false && "Unexpected architecture for PlatformFreeBSDKernel plugin");
    return;
  }

  // x86 / x86_64: scan symtab for IDTVEC stubs matching kgdb heuristic:
  //   name[0] == 'X' && name[1] != '_'
  Symtab *symtab = kernel_module->GetSymtab();
  if (!symtab)
    return;

  RegularExpression x_stub_regex("^X[^_]");
  std::vector<uint32_t> indexes;
  symtab->AppendSymbolIndexesMatchingRegExAndType(x_stub_regex, eSymbolTypeCode,
                                                  indexes);
  for (uint32_t idx : indexes)
    m_trap_handlers.push_back(symtab->SymbolAtIndex(idx)->GetName());
}

lldb::UnwindPlanSP PlatformFreeBSDKernel::BuildTrapframeUnwindPlan(
    llvm::StringRef source_name, uint32_t cfa_dwarf_reg, int32_t cfa_offset,
    llvm::ArrayRef<std::pair<uint32_t, int32_t>> regs) {
  auto plan = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  UnwindPlan::Row row;

  // CFA = <cfa_reg> + cfa_offset.  For most trapframes this is SP+0,
  // meaning RSP/SP at stub entry already points at the struct base.
  row.GetCFAValue().SetIsRegisterPlusOffset(cfa_dwarf_reg, cfa_offset);

  for (auto [dwarf_reg, offset] : regs)
    row.SetRegisterLocationToAtCFAPlusOffset(dwarf_reg, offset,
                                             /*can_replace=*/true);

  plan->AppendRow(row);
  plan->SetSourceName(source_name.data());
  plan->SetSourcedFromCompiler(eLazyBoolNo);
  // Valid at all PCs within the stub: the trapframe is fully populated
  // before any C function is called, so the plan applies everywhere.
  plan->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  // Signals to RegisterContextUnwind that the frame *above* this one (i.e. the
  // interrupted context) has its registers available from this saved trapframe.
  plan->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  return plan;
}
