//===-- ProcessEZH.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessEZH.h"
#include "EZHRegisters.h"
#include "ThreadEZH.h"
#include "lldb/Core/Module.h"
#include "lldb/Utility/State.h"

#include <thread>
#include <chrono>

#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Breakpoint/BreakpointSite.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/OptionValueUInt64.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Core/Value.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/RegisterValue.h"

namespace {
class ProcessEZHProperties : public lldb_private::Properties {
public:
  ProcessEZHProperties() : lldb_private::Properties() {
    m_collection_sp = std::make_shared<lldb_private::OptionValueProperties>("ezh-remote");
    m_collection_sp->Initialize(lldb_private::PropertyCollectionDefinition{llvm::ArrayRef<lldb_private::PropertyDefinition>(g_properties), "plugin.process.ezh-remote"});
  }
  static llvm::StringRef GetSettingName() { return "ezh-remote"; }

  lldb::addr_t GetBaseAddress() const {
    const uint32_t idx = 0;
    lldb_private::OptionValueUInt64 *value = m_collection_sp->GetPropertyAtIndexAsOptionValueUInt64(idx);
    if (value)
      return value->GetCurrentValue();
    return 0x40027000;
  }

private:
  static const lldb_private::PropertyDefinition g_properties[1];
};

const lldb_private::PropertyDefinition ProcessEZHProperties::g_properties[1] = {
    {"base-address",
     lldb_private::OptionValue::eTypeUInt64,
     true,
     0x40027000,
     nullptr,
     {},
     "The physical base address of the EZH hardware."}};

static ProcessEZHProperties &GetGlobalEZHProperties() {
  static ProcessEZHProperties g_settings;
  return g_settings;
}
} // namespace

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

LLDB_PLUGIN_DEFINE_ADV(ProcessEZH, ProcessEZH)

ProcessEZH::ProcessEZH(TargetSP target_sp, ListenerSP listener_sp)
    : ProcessGDBRemote(target_sp, listener_sp) {
  for (int i = 0; i < 16; ++i) {
    m_debug_sw_bp_addrs[i] = LLDB_INVALID_ADDRESS;
    m_active_sw_breakpoints[i] = LLDB_INVALID_ADDRESS;
  }
  m_polling_thread = std::thread(&ProcessEZH::PollingThread, this);
}

ProcessEZH::~ProcessEZH() {
  m_destroy_polling_thread = true;
  m_polling_cv.notify_all();
  if (m_polling_thread.joinable()) {
    m_polling_thread.join();
  }
}

lldb::addr_t ProcessEZH::GetBaseAddress() const {
  return GetGlobalEZHProperties().GetBaseAddress();
}

Status ProcessEZH::DoConnectRemote(llvm::StringRef remote_url) {
  Status error = ProcessGDBRemote::DoConnectRemote(remote_url);
  if (error.Success()) {
    if (m_async_thread.IsJoinable()) {
      m_async_broadcaster.BroadcastEvent(eBroadcastBitAsyncThreadShouldExit);
      m_async_thread.Join(nullptr);
      m_async_thread.Reset();
    }

    // Force state to stopped so LLDB command interpreter gets control instantly!
    SetPrivateState(eStateStopped);
  }
  return error;
}

void ProcessEZH::WillPublicStop() {
  // Do absolutely nothing! Bypass base class GDB remote thread queries to eliminate OpenOCD deprecation warnings.
}

void ProcessEZH::DidAttach(lldb_private::ArchSpec &process_arch) {
  // Do absolutely nothing! Bypass base class DidLaunchOrAttach to prevent sending qProcessInfo, qSymbol, or structured data queries!
}



llvm::Error ProcessEZH::UpdateBreakpointSites(
    const BreakpointSiteToActionMap &site_to_action) {
  return lldb_private::Process::UpdateBreakpointSites(site_to_action);
}

Status ProcessEZH::EnableBreakpointSite(BreakpointSite *bp_site) {
  if (!GetTarget().GetExecutableModule())
    return Status::FromErrorString(
        "Cannot enable breakpoints on EZH co-processor without an active ELF symbols file loaded.");

  if (!bp_site)
    return Status::FromErrorString("Invalid breakpoint site.");

  addr_t addr = bp_site->GetLoadAddress();

  if (IsBreakpointSiteEnabled(*bp_site))
    return Status();

  // Find if already enabled.
  int slot = -1;
  for (int i = 0; i < 16; ++i)
    if (m_active_sw_breakpoints[i] == addr) {
      slot = i;
      break;
    }

  // If not already enabled, find an empty slot.
  if (slot == -1)
    for (int i = 0; i < 16; ++i)
      if (m_active_sw_breakpoints[i] == LLDB_INVALID_ADDRESS) {
        slot = i;
        break;
      }

  if (slot == -1)
    return Status::FromErrorString("EZH co-processor only supports up to 16 active software breakpoints concurrently.");

  // 1. Resolve slot-specific debug_software_breakpoint_N address.
  lldb::addr_t debug_sw_bp_addr = GetDebugSoftwareBreakpointAddr(slot);
  if (debug_sw_bp_addr == LLDB_INVALID_ADDRESS)
    return Status::FromErrorStringWithFormat("Failed to resolve 'debug_software_breakpoint_%d' exception vector for software breakpoint. Please make sure crt0 object symbols are linked.", slot);

  // 2. Read original 4-byte instruction at addr from target RAM.
  uint8_t original_bytes[4];
  Status error;
  size_t bytes_read = DoReadMemoryDirect(addr, original_bytes, 4, error);
  if (bytes_read != 4 || error.Fail())
    return Status::FromErrorStringWithFormat("Failed to read target instruction for software breakpoint backup: %s", error.AsCString());

  // 3. Backup the original instruction bytes in bp_site (so LLDB can restore it on hit/disable).
  memcpy(bp_site->GetSavedOpcodeBytes(), original_bytes, 4);
  bp_site->SetType(BreakpointSite::eSoftware);

  // 4. Construct software breakpoint instruction: e_goto &debug_software_breakpoint_K.
  // Target Address is word-addressable in 21-bit: (TargetAddr >> 2) & 0x1FFFFF
  uint32_t target_word = (static_cast<uint32_t>(debug_sw_bp_addr) >> 2) & 0x1FFFFF;
  uint32_t sw_bp_op = (target_word << 11) | 0x215;

  // 5. Write software breakpoint instruction word directly to target memory (RAM patching).
  size_t bytes_written = DoWriteMemory(addr, &sw_bp_op, 4, error);
  if (bytes_written != 4 || error.Fail())
    return Status::FromErrorStringWithFormat("Failed to write software breakpoint trap opcode to target RAM: %s", error.AsCString());

  m_active_sw_breakpoints[slot] = addr;
  SetBreakpointSiteEnabled(*bp_site, true);
  return Status();
}

Status ProcessEZH::DisableBreakpointSite(BreakpointSite *bp_site) {
  if (!bp_site)
    return Status::FromErrorString("Invalid breakpoint site.");

  addr_t addr = bp_site->GetLoadAddress();

  if (!IsBreakpointSiteEnabled(*bp_site))
    return Status();

  // 1. Retrieve saved original instruction bytes from bp_site backup.
  const uint8_t *original_bytes = bp_site->GetSavedOpcodeBytes();
  if (!original_bytes)
    return Status::FromErrorString("No backup bytes available to restore original instruction.");

  // 2. Overwrite software breakpoint instruction in RAM with original bytes (RAM restore).
  Status error;
  size_t bytes_written = DoWriteMemory(addr, original_bytes, 4, error);
  if (bytes_written != 4 || error.Fail())
    return Status::FromErrorStringWithFormat("Failed to restore target instruction from backup: %s", error.AsCString());

  SetBreakpointSiteEnabled(*bp_site, false);
  return Status();
}

void ProcessEZH::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "GDB Remote EZH coprocessor debugging plugin",
                                CreateInstance, DebuggerInitialize);
}

void ProcessEZH::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForProcessPlugin(
          debugger, ProcessEZHProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForProcessPlugin(
        debugger, GetGlobalEZHProperties().GetValueProperties(),
        "Properties for the EZH remote process plugin.", is_global_setting);
  }
}

void ProcessEZH::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ProcessSP ProcessEZH::CreateInstance(TargetSP target_sp, ListenerSP listener_sp,
                                     const FileSpec *crash_file_path,
                                     bool can_connect) {
  ProcessSP process_sp;
  if (crash_file_path == nullptr) {
    if (target_sp->GetArchitecture().GetTriple().getArchName().starts_with("ezh")) {
      process_sp = std::make_shared<ProcessEZH>(target_sp, listener_sp);
    }
  }
  return process_sp;
}

std::shared_ptr<lldb_private::process_gdb_remote::ThreadGDBRemote>
ProcessEZH::CreateThread(lldb::tid_t tid) {
  return std::make_shared<ThreadEZH>(*this, tid);
}



ArchSpec ProcessEZH::GetSystemArchitecture() {
  return GetTarget().GetArchitecture();
}

Status ProcessEZH::DoResume(lldb::RunDirection direction) {
  if (!GetTarget().GetExecutableModule()) {
    return Status::FromErrorString(
        "Cannot resume or single-step EZH co-processor without an active ELF symbols file loaded ('target create <elf>').");
  }
  m_memory_cache.Clear();

  ThreadSP thread_sp = m_thread_list.GetThreadAtIndex(0);
  if (!thread_sp)
    return Status::FromErrorString("No active thread.");

  RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
  if (reg_ctx_sp)
    reg_ctx_sp->InvalidateAllRegisters();

  StateType resume_state = thread_sp->GetTemporaryResumeState();
  m_is_stepping = (resume_state == eStateStepping);
  m_halt_requested = false;
  Status error;

  // 0.1 Resolve and read EZH's RAM handshake variable debug_frame over JTAG
  addr_t debug_frame_addr_init = LLDB_INVALID_ADDRESS;
  VariableList variable_list_init;
  GetTarget().GetImages().FindGlobalVariables(ConstString("debug_frame"), 1, variable_list_init);
  if (variable_list_init.GetSize() > 0) {
    VariableSP var_sp_init = variable_list_init.GetVariableAtIndex(0);
    if (var_sp_init) {
      ExecutionContext exe_ctx(GetTarget());
      auto value_or_error = var_sp_init->LocationExpressionList().Evaluate(&exe_ctx, nullptr, LLDB_INVALID_ADDRESS, nullptr, nullptr);
      if (value_or_error) {
        Value val = value_or_error.get();
        debug_frame_addr_init = val.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
      }
    }
  }

  uint32_t sp_val_init = 0;
  if (debug_frame_addr_init != LLDB_INVALID_ADDRESS)
    DoReadMemory(debug_frame_addr_init, &sp_val_init, 4, error);

  // 0.23 If virtually halted, force commit virtualized PC back to stack RAM to replace any JTAG markers!
  // IMPORTANT: Do this BEFORE clearing deferred breakpoints, because virtualization relies on m_active_sw_breakpoints!
  if (sp_val_init != 0 && reg_ctx_sp) {
    const RegisterInfo *pc_reg_info = reg_ctx_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t pc_val = reg_ctx_sp->ReadRegisterAsUnsigned(pc_reg_info, 0);
    reg_ctx_sp->WriteRegister(pc_reg_info, RegisterValue(pc_val));
  }

  // Now it is safe to clean up any deferred cleared software breakpoints!
  // Their slots will become free for the single-step logic to reuse.
  for (int i = 0; i < 16; ++i) {
    addr_t addr = m_active_sw_breakpoints[i];
    if (addr != LLDB_INVALID_ADDRESS && addr != m_step_bp_addr) {
      BreakpointSiteSP bp_site_sp = GetBreakpointSiteList().FindByAddress(addr);
      if (!bp_site_sp || !IsBreakpointSiteEnabled(*bp_site_sp))
        m_active_sw_breakpoints[i] = LLDB_INVALID_ADDRESS;
    }
  }

  // Invalidate registers again to force re-reading PC from target RAM (where we just wrote the virtual PC)
  if (reg_ctx_sp)
    reg_ctx_sp->InvalidateAllRegisters();

  // 0.2 Clear EZH JTAG Interrupt 7 pending status to prevent recursive loops on resume.
  // Read current PENDTRAP, and only clear our pending request bit 7 (AND-NOT ~ (1<<7))
  uint32_t trap_val_clear = 0;
  error = ReadEZHRegister(EZHB_PENDTRAP_OFFSET, trap_val_clear);
  if (error.Success()) {
    trap_val_clear &= ~(1 << 7);
    error = WriteEZHRegister(EZHB_PENDTRAP_OFFSET, trap_val_clear);
  }
  if (error.Fail())
    return error;

  // 0.3 Prevent stepping actively executing targets that have no stack frame!
  if (resume_state == eStateStepping && sp_val_init == 0) {
    uint32_t ctrl_val_check = 0;
    ReadEZHRegister(EZHB_CTRL_OFFSET, ctrl_val_check);
    if ((ctrl_val_check & 1) != 0) {
      if (thread_sp)
        thread_sp->DiscardThreadPlansUpToPlan(thread_sp->GetCurrentPlan());
      return Status::FromErrorString("Cannot single-step an actively executing target. Halt it first.");
    }
  }

  if (resume_state == eStateStepping) {
    // 0.5 Check if EZH is physically stopped/reset (Start = 0).
    uint32_t ctrl_val = 0;
    error = ReadEZHRegister(EZHB_CTRL_OFFSET, ctrl_val);
    if (error.Success() && ((ctrl_val & 1) == 0)) {
      if (thread_sp)
        thread_sp->DiscardThreadPlansUpToPlan(thread_sp->GetCurrentPlan());
      return Status::FromErrorString("Cannot step, stepi, next or finish a non running core (Start = 0) because the stack pointer (SP) has not been initialized yet. Please set a breakpoint after stack initialization and use 'continue' to start the core.");
    }

    // 1. Read active PC natively from register context
    RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
    uint32_t pc_val = 0;
    if (reg_ctx_sp)
      pc_val = reg_ctx_sp->ReadRegisterAsUnsigned(
          reg_ctx_sp->GetRegisterInfo(eRegisterKindGeneric,
                                      LLDB_REGNUM_GENERIC_PC),
          0);

    // 2. Decode EZH instruction at current PC to predict next PC (branch target prediction!)
    uint32_t next_pc = pc_val + 4; // Default sequential fallback
    uint32_t inst_val = 0;
    uint32_t rn_wb = 99;
    uint32_t rn_val = 0;
    DoReadMemory(pc_val, &inst_val, 4, error);
    
    if (error.Success()) {
      if ((inst_val & 0x3) == 0x3) {
        // e_gosub (CALL) -> Unconditional 30-bit absolute address branch!
        next_pc = inst_val & 0xFFFFFFFC;
      } else if ((inst_val & 0x1F) == 0x15) {
        // e_goto / e_gotol / e_goto_reg -> Conditional branches!
        uint32_t cond = (inst_val >> 5) & 0xF;
        bool is_reg = (((inst_val >> 9) & 1) == 0);

        // Read ALU flags register directly from the active JTAG stack frame in RAM!
        uint32_t flags = 0;
        if (sp_val_init != 0)
          DoReadMemory(sp_val_init + EZH_FRAME_OFFSET_FLAGS, &flags, 4, error);

        // Evaluate the branch condition: cond == 0 is unconditional (EU), otherwise test bit 'cond' in flags!
        bool condition_met = (cond == 0) || ((flags & (1 << cond)) != 0);


        
        if (condition_met) {
          if (is_reg) {
            // Register Branch -> target is in register raddr (Inst{17-14})
            uint32_t raddr = (inst_val >> 14) & 0xF;
            if (reg_ctx_sp)
              next_pc = reg_ctx_sp->ReadRegisterAsUnsigned(raddr, pc_val + 4);
          } else {
            // Immediate Branch -> target is 21-bit absolute word offset (Inst{31-11}) from 0x24000000
            uint32_t addr = (inst_val >> 11) & 0x1FFFFF;
            next_pc = 0x24000000 + (addr << 2);
          }

      }
      } else if ((inst_val & 0x1F) == 0x01 && ((inst_val >> 10) & 0xF) == 0xD) {
        // e_ldr pc, ... (including e_popd pc!) -> Load to PC return!
        // Predict next PC by reading the return address from the stack frame in RAM!
        rn_wb = (inst_val >> 14) & 0xF; // Base register Rn (should be SP)
        rn_val = 0;
        if (reg_ctx_sp)
          rn_val = reg_ctx_sp->ReadRegisterAsUnsigned(rn_wb, 0);

        // EZH LDR_POST Rn, Offset: Address read is simply the current Rn value!
        // (Post-increment reads from Rn, and then increments Rn! So the load address is exactly Rn!)
        uint32_t load_addr = rn_val;
        uint32_t ret_addr = 0;
        DoReadMemory(load_addr, &ret_addr, 4, error);
        if (error.Success()) {
          next_pc = ret_addr;
        }
      } else if ((inst_val & 0x0000FFFF) == 0x0000f400) {
        // e_mov pc, Rs -> Register move return!
        // Rs is in Inst{17-14}
        uint32_t rs = (inst_val >> 14) & 0xF;
        if (reg_ctx_sp)
          next_pc = reg_ctx_sp->ReadRegisterAsUnsigned(rs, pc_val + 4);
      }
    }

    // 3. Set temporary Software Breakpoint (RAM patch) at next PC to prevent prefetch PC corruption hazards!
    fprintf(stderr, "[ProcessEZH::DoResume] SINGLE_STEP: PC=0x%08x, inst=0x%08x -> Predicted next_pc=0x%08x (sp_init=0x%08x, rn_wb=%u, rn_val=0x%08x)\n",
            pc_val, inst_val, next_pc, sp_val_init, rn_wb, rn_val);

    // Clean up any stale step breakpoint first
    if (m_step_bp_addr != LLDB_INVALID_ADDRESS) {
      DoWriteMemory(m_step_bp_addr, &m_step_bp_original_op, 4, error);
      if (m_step_bp_slot != -1)
        m_active_sw_breakpoints[m_step_bp_slot] = LLDB_INVALID_ADDRESS;
      m_step_bp_addr = LLDB_INVALID_ADDRESS;
      m_step_bp_original_op = 0;
      m_step_bp_slot = -1;
    }

    // Find an empty slot for the step breakpoint.
    int slot = -1;
    for (int i = 0; i < 16; ++i)
      if (m_active_sw_breakpoints[i] == LLDB_INVALID_ADDRESS) {
        slot = i;
        break;
      }

    if (slot == -1)
      return Status::FromErrorString("Cannot single-step EZH target: All 16 software breakpoint slots are currently full. Please disable a breakpoint first.");

    lldb::addr_t debug_sw_bp_addr = GetDebugSoftwareBreakpointAddr(slot);
    if (debug_sw_bp_addr == LLDB_INVALID_ADDRESS)
      return Status::FromErrorStringWithFormat("Failed to resolve debug_software_breakpoint_%d address for software single-step.", slot);

    // Read original instruction at next_pc.
    uint32_t original_op = 0;
    size_t bytes_read = DoReadMemoryDirect(next_pc, &original_op, 4, error);
    if (bytes_read != 4 || error.Fail())
      return Status::FromErrorStringWithFormat("Failed to read original instruction at single-step target 0x%08llx: %s", (unsigned long long)next_pc, error.AsCString());

    // Backup original state.
    m_step_bp_addr = next_pc;
    m_step_bp_original_op = original_op;
    m_step_bp_slot = slot;
    m_active_sw_breakpoints[slot] = next_pc;

    // Construct software breakpoint instruction: e_goto &debug_software_breakpoint_K.
    uint32_t target_word = (static_cast<uint32_t>(debug_sw_bp_addr) >> 2) & 0x1FFFFF;
    uint32_t sw_bp_op = (target_word << 11) | 0x215;

    // Patch RAM.
    size_t bytes_written = DoWriteMemory(next_pc, &sw_bp_op, 4, error);
    if (bytes_written != 4 || error.Fail()) {
      // Rollback backup state on write error.
      m_active_sw_breakpoints[slot] = LLDB_INVALID_ADDRESS;
      m_step_bp_addr = LLDB_INVALID_ADDRESS;
      m_step_bp_original_op = 0;
      m_step_bp_slot = -1;
      return Status::FromErrorStringWithFormat("Failed to patch RAM at single-step target 0x%08llx: %s", (unsigned long long)next_pc, error.AsCString());
    }

    // 4. Unhalt EZH co-processor: set debug_frame in RAM to 0 over JTAG!
    uint32_t zero = 0;
    if (debug_frame_addr_init != LLDB_INVALID_ADDRESS) {
      WriteMemory(debug_frame_addr_init, &zero, 4, error);
      if (error.Fail())
        return error;
      // Give the remote JTAG adapter 10ms to process the unhalt cleanly!
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // No artificial sleep needed; let PollingThread verify physical halt instantly.

  } else if (resume_state == eStateRunning) {
    // 1. Check the physical EZH ignition state first!
    uint32_t ctrl_val = 0;
    error = ReadEZHRegister(EZHB_CTRL_OFFSET, ctrl_val);
    if (error.Fail())
      return error;
    bool ignition_set = ((ctrl_val & 1) != 0);

    if (!ignition_set) {
      // EZH is physically stopped/reset (Start = 0). Ignite it!

      // 1. Invalidate EZH's cached JTAG RAM address to guarantee dynamic self-refresh on new compiles!
      m_debug_frame_addr = LLDB_INVALID_ADDRESS;

      // 2. Clear EZH's RAM JTAG handshake variable debug_frame over JTAG first!
      uint32_t zero = 0;
      if (debug_frame_addr_init != LLDB_INVALID_ADDRESS)
        WriteMemory(debug_frame_addr_init, &zero, 4, error);

      // 2. Clear JTAG Trap 7 request and arm JTAG Trap Enable 7 in hardware.
      error = WriteEZHRegister(EZHB_PENDTRAP_OFFSET, 0x00800000);
      if (error.Fail())
        return error;

      // 3. Automatically resolve EZH's entry point address from the ELF headers!
      auto entry_point_address = GetTarget().GetEntryPointAddress();
      if (entry_point_address) {
        addr_t entry_point = entry_point_address->GetLoadAddress(&GetTarget());
        if (entry_point == LLDB_INVALID_ADDRESS)
          entry_point = entry_point_address->GetFileAddress(); // Robust static File Address fallback!
        if (entry_point != LLDB_INVALID_ADDRESS) {
          error = WriteEZHRegister(EZHB_BOOT_OFFSET, static_cast<uint32_t>(entry_point));
          if (error.Fail())
            return error;
        }
      } else {
        llvm::consumeError(entry_point_address.takeError());
      }

      // 4. Ignite EZH!
      uint32_t ctrl_ignite = 0xC0DE0000 | (ctrl_val & 0xFFFF) | 1;
      error = WriteEZHRegister(EZHB_CTRL_OFFSET, ctrl_ignite);
      if (error.Fail())
        return error;
      m_is_stepping = false;
    } else {
      // EZH is physically ignited (Start = 1). Only unhalt if we have a valid stack frame!
      if (sp_val_init != 0) {
        // EZH is in Virtually Halted state. Unhalt EZH: set debug_frame in RAM to 0 over JTAG!
        uint32_t zero = 0;
        if (debug_frame_addr_init != LLDB_INVALID_ADDRESS) {
          WriteMemory(debug_frame_addr_init, &zero, 4, error);
          if (error.Fail())
            return error;
        }
      }
    }
  }

  // Transition state to eStateRunning so polling thread runs and LLDB waits correctly
  SetPrivateState(eStateRunning);

  // Notify polling thread to start polling
  m_polling_cv.notify_one();

  return Status();
}

bool ProcessEZH::DoUpdateThreadList(ThreadList &old_thread_list,
                                    ThreadList &new_thread_list) {
  // Preserve EZH's real JTAG thread context if it exists to keep stop info and register contexts intact!
  // Pass false to GetThreadAtIndex to prevent infinite recursive thread list update checking!
  ThreadSP thread_sp = old_thread_list.GetThreadAtIndex(0, false);
  if (!thread_sp) {
    thread_sp = std::make_shared<ThreadEZH>(*this, 1);
  } else {
    // Forcefully invalidate register cache on stop refresh to guarantee fresh JTAG reads!
    RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
    if (reg_ctx_sp)
      reg_ctx_sp->InvalidateAllRegisters();
  }
  new_thread_list.AddThread(thread_sp);
  return true;
}

Status ProcessEZH::DoHalt(bool &caused_stop) {
  if (m_is_stepping)
    // If EZH is actively single-stepping, do NOT inject Trap 7. Let the hardware hit the breakpoint naturally!
    return Status();
  m_halt_requested = true;
  caused_stop = false;

  // 1. Read current PENDTRAP to preserve other active traps!
  uint32_t trap_val = 0;
  Status error;
  error = ReadEZHRegister(EZHB_PENDTRAP_OFFSET, trap_val);
  if (error.Fail())
    return error;

  // 2. Enable pending trap 7 (bit 23) and set request (bit 7) safely via OR
  trap_val |= (1 << 23) | (1 << 7);

  error = WriteEZHRegister(EZHB_PENDTRAP_OFFSET, trap_val);
  if (error.Success()) {
    caused_stop = true;
    return Status();
  }

  return Status::FromErrorStringWithFormat(
      "Failed to write EZH PENDTRAP register over JTAG: %s", error.AsCString());
}

void ProcessEZH::RefreshStateAfterStop() {
  Status error;
  // 0. Clean up temporary Software Single-Step breakpoint if it was active.
  if (m_step_bp_addr != LLDB_INVALID_ADDRESS) {
    DoWriteMemory(m_step_bp_addr, &m_step_bp_original_op, 4, error);

    // Force commit the virtual step PC back to stack RAM to replace the JTAG marker immediately!
    // This allows us to free the step BP slot right now, instead of waiting for DoResume!
    lldb::addr_t debug_frame_addr = GetDebugFrameAddr();
    if (debug_frame_addr != LLDB_INVALID_ADDRESS) {
      uint32_t sp_val = 0;
      DoReadMemory(debug_frame_addr, &sp_val, 4, error);
      if (sp_val != 0) {
        uint32_t pc_val = static_cast<uint32_t>(m_step_bp_addr);
        // Write virtual PC to stack frame (sp - 16)
        DoWriteMemory(sp_val - 16, &pc_val, 4, error);
      }
    }

    // Now we can safely free the slot immediately.
    if (m_step_bp_slot != -1)
      m_active_sw_breakpoints[m_step_bp_slot] = LLDB_INVALID_ADDRESS;

    m_step_bp_addr = LLDB_INVALID_ADDRESS;
    m_step_bp_original_op = 0;
    m_step_bp_slot = -1;
  }

  // 0.5. Ensure EZH physical halt has fully completed and stabilized in RAM!
  lldb::addr_t debug_frame_addr = GetDebugFrameAddr();
  if (debug_frame_addr != LLDB_INVALID_ADDRESS) {
    uint32_t sp_val = 0;
    Status error;
    m_memory_cache.Clear();
    DoReadMemory(debug_frame_addr, &sp_val, 4, error);
    if (sp_val == 0)
      for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        m_memory_cache.Clear();
        DoReadMemory(debug_frame_addr, &sp_val, 4, error);
        if (sp_val != 0)
          break;
      }
  }

  // Refresh thread list directly without invoking base class GDB remote thread queries
  m_thread_ids.clear();
  m_thread_pcs.clear();
  UpdateThreadListIfNeeded();
  if (m_last_stop_packet) {
    SetThreadStopInfo(*m_last_stop_packet);
    m_last_stop_packet.reset();
  }
  m_thread_list_real.RefreshStateAfterStop();

  ThreadSP thread_sp = m_thread_list.GetThreadAtIndex(0);
  if (thread_sp) {
    RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
    if (reg_ctx_sp)
      reg_ctx_sp->InvalidateAllRegisters();

    StateType temp_state = thread_sp->GetTemporaryResumeState();
    if (temp_state == eStateStepping) {
      thread_sp->SetStopInfo(StopInfo::CreateStopReasonToTrace(*thread_sp));
    } else {
      addr_t pc_val = 0;
      if (reg_ctx_sp)
        pc_val = reg_ctx_sp->ReadRegisterAsUnsigned(
            13u, 0ULL); // PC is 13 (ra is 16)
      BreakpointSiteSP bp_site_sp = GetBreakpointSiteList().FindByAddress(pc_val);
      if (bp_site_sp)
        thread_sp->SetStopInfo(StopInfo::CreateStopReasonWithBreakpointSiteID(*thread_sp, bp_site_sp->GetID()));
      else
        thread_sp->SetStopInfo(StopInfo::CreateStopReasonToTrace(*thread_sp));
    }
  }
}

void ProcessEZH::ModulesDidLoad(lldb_private::ModuleList &module_list) {
  // Call base class to process the newly loaded modules first!
  ProcessGDBRemote::ModulesDidLoad(module_list);

  // Automatically resolve ELF entrypoint address and load it into hardware BOOTADR register!
  auto entry_point_address = GetTarget().GetEntryPointAddress();
  if (entry_point_address) {
    addr_t entry_point = entry_point_address->GetLoadAddress(&GetTarget());
    if (entry_point == LLDB_INVALID_ADDRESS)
      entry_point = entry_point_address->GetFileAddress(); // Robust static File Address fallback!
    if (entry_point != LLDB_INVALID_ADDRESS)
      WriteEZHRegister(EZHB_BOOT_OFFSET, (uint32_t)entry_point);
  } else {
    llvm::consumeError(entry_point_address.takeError());
  }
}

void ProcessEZH::PollingThread() {
  lldb_private::Status error;

  while (!m_destroy_polling_thread) {
    std::unique_lock<std::mutex> lock(m_polling_mutex);
    m_polling_cv.wait(lock, [this]() {
      return GetPrivateState() == lldb::eStateRunning || m_destroy_polling_thread;
    });

    if (m_destroy_polling_thread)
      break;

    lock.unlock();

    uint32_t sp_val = 0;
    lldb::addr_t exc_signal_addr = LLDB_INVALID_ADDRESS;
    bool first_poll = true;
    int poll_count = 0;
    
    while (GetPrivateState() == lldb::eStateRunning && !m_destroy_polling_thread) {
      poll_count++;
      if (first_poll) {
        if (m_is_stepping)
          std::this_thread::sleep_for(std::chrono::milliseconds(
              5)); // Initial 5ms wait for single-stepping to complete over JTAG
      } else {
        if (m_is_stepping)
          std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 5ms poll for single-stepping
        else
          std::this_thread::sleep_for(std::chrono::milliseconds(
              500)); // Standard 500ms poll for continuous execution
      }
      first_poll = false;

      if (m_destroy_polling_thread)
        break;

      // Re-read debug_frame immediately after waking up from sleep to ensure sp_val reflects the latest JTAG state
      if (m_debug_frame_addr != LLDB_INVALID_ADDRESS && !m_destroy_polling_thread) {
        m_memory_cache.Clear();
        DoReadMemory(m_debug_frame_addr, &sp_val, 4, error);
      }

      if (m_debug_frame_addr == LLDB_INVALID_ADDRESS) {
        VariableList variable_list;
        GetTarget().GetImages().FindGlobalVariables(ConstString("debug_frame"), 1, variable_list);
        if (variable_list.GetSize() > 0) {
          VariableSP var_sp = variable_list.GetVariableAtIndex(0);
          if (var_sp) {
            ExecutionContext exe_ctx(GetTarget());
            auto value_or_error = var_sp->LocationExpressionList().Evaluate(&exe_ctx, nullptr, LLDB_INVALID_ADDRESS, nullptr, nullptr);
            if (value_or_error) {
              Value val = value_or_error.get();
              m_debug_frame_addr = val.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
            }
          }
        }
      }

      if (exc_signal_addr == LLDB_INVALID_ADDRESS) {
        VariableList variable_list_exc;
        GetTarget().GetImages().FindGlobalVariables(ConstString("exc_signal"), 1, variable_list_exc);
        if (variable_list_exc.GetSize() > 0) {
          VariableSP var_sp_exc = variable_list_exc.GetVariableAtIndex(0);
          if (var_sp_exc) {
            ExecutionContext exe_ctx(GetTarget());
            auto value_or_error = var_sp_exc->LocationExpressionList().Evaluate(&exe_ctx, nullptr, LLDB_INVALID_ADDRESS, nullptr, nullptr);
            if (value_or_error) {
              Value val = value_or_error.get();
              exc_signal_addr = val.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
            }
          }
        }
      }

      size_t bytes_read = 0;
      uint32_t exc_val = 0;
      if (m_debug_frame_addr != LLDB_INVALID_ADDRESS && !m_destroy_polling_thread) {
        m_memory_cache.Clear();
        bytes_read = DoReadMemory(m_debug_frame_addr, &sp_val, 4, error);
      }
      if (exc_signal_addr != LLDB_INVALID_ADDRESS && !m_destroy_polling_thread) {
        Status exc_error;
        m_memory_cache.Clear();
        DoReadMemory(exc_signal_addr, &exc_val, 4, exc_error);
      }

      if (bytes_read == 4 && error.Success() && sp_val != 0) {
        m_halt_requested = false;
        SetPrivateState(lldb::eStateStopped);
        break;
      }
    }
  }
}

Status ProcessEZH::WriteEZHRegister(addr_t offset, uint32_t value) {
  Status error;
  WriteMemory(GetBaseAddress() + offset, &value, 4, error);
  return error;
}

Status ProcessEZH::ReadEZHRegister(addr_t offset, uint32_t &value) {
  Status error;
  m_memory_cache.Clear();
  ReadMemory(GetBaseAddress() + offset, &value, 4, error);
  return error;
}

lldb::addr_t ProcessEZH::GetDebugFrameAddr() {
  if (m_debug_frame_addr == LLDB_INVALID_ADDRESS) {
    VariableList variable_list;
    GetTarget().GetImages().FindGlobalVariables(ConstString("debug_frame"), 1, variable_list);
    if (variable_list.GetSize() > 0) {
      VariableSP var_sp = variable_list.GetVariableAtIndex(0);
      if (var_sp) {
        ExecutionContext exe_ctx(GetTarget());
        auto value_or_error = var_sp->LocationExpressionList().Evaluate(&exe_ctx, nullptr, LLDB_INVALID_ADDRESS, nullptr, nullptr);
        if (value_or_error) {
          Value val = value_or_error.get();
          m_debug_frame_addr = val.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
        }
      }
    }
  }
  return m_debug_frame_addr;
}

lldb::addr_t ProcessEZH::GetDebugSoftwareBreakpointAddr(uint32_t slot) {
  if (slot >= 16)
    return LLDB_INVALID_ADDRESS;

  if (m_debug_sw_bp_addrs[slot] == LLDB_INVALID_ADDRESS) {
    char sym_name[64];
    snprintf(sym_name, sizeof(sym_name), "debug_software_breakpoint_%u", slot);
    SymbolContextList sc_list;
    GetTarget().GetImages().FindSymbolsWithNameAndType(ConstString(sym_name), eSymbolTypeAny, sc_list);
    if (sc_list.GetSize() > 0) {
      SymbolContext sc;
      if (sc_list.GetContextAtIndex(0, sc))
        if (sc.symbol)
          m_debug_sw_bp_addrs[slot] = sc.symbol->GetAddress().GetLoadAddress(&GetTarget());
    }
  }
  return m_debug_sw_bp_addrs[slot];
}

Status ProcessEZH::DoDetach(bool keep_stopped) {
  // Commit virtualized PC back to physical stack RAM before detaching so target is left in clean state.
  ThreadSP thread_sp = m_thread_list.GetThreadAtIndex(0);
  if (thread_sp) {
    RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
    if (reg_ctx_sp) {
      const RegisterInfo *pc_reg_info = reg_ctx_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
      uint32_t pc_val = reg_ctx_sp->ReadRegisterAsUnsigned(pc_reg_info, 0);
      if (pc_val != 0 &&
          pc_val != static_cast<uint32_t>(LLDB_INVALID_ADDRESS))
        reg_ctx_sp->WriteRegister(pc_reg_info, RegisterValue(pc_val));
    }
  }
  return ProcessGDBRemote::DoDetach(keep_stopped);
}

Status ProcessEZH::DoDestroy() {
  // Commit virtualized PC back to physical stack RAM before detaching so target is left in clean state.
  ThreadSP thread_sp = m_thread_list.GetThreadAtIndex(0);
  if (thread_sp) {
    RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
    if (reg_ctx_sp) {
      const RegisterInfo *pc_reg_info = reg_ctx_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
      uint32_t pc_val = reg_ctx_sp->ReadRegisterAsUnsigned(pc_reg_info, 0);
      if (pc_val != 0 &&
          pc_val != static_cast<uint32_t>(LLDB_INVALID_ADDRESS))
        reg_ctx_sp->WriteRegister(pc_reg_info, RegisterValue(pc_val));
    }
  }
  return ProcessGDBRemote::DoDestroy();
}


