//===-- NativeRegisterContextWindows_WoW64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__x86_64__) || defined(_M_X64)

#include <set>
#include "lldb/Register/Arch/lldb-x86-register-enums.h"

#include "NativeRegisterContextWindows_WoW64.h"

#include "NativeThreadWindows.h"
#include "ProcessWindowsLog.h"
#include "RegisterLayoutWindows_x86.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/ADT/STLExtras.h"

namespace reg_ids = lldb_private::registers::x86::id;
using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE sizeof(::WOW64_CONTEXT)

namespace {
static const uint32_t g_gpr_regnums_WoW64[] = {
    reg_ids::eax,       reg_ids::ebx,    reg_ids::ecx, reg_ids::edx,
    reg_ids::edi,       reg_ids::esi,    reg_ids::ebp, reg_ids::esp,
    reg_ids::eip,       reg_ids::eflags, reg_ids::cs,  reg_ids::fs,
    reg_ids::gs,        reg_ids::ss,     reg_ids::ds,  reg_ids::es,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag.
};

struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_WoW64[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_WoW64,
                             g_gpr_regnums_WoW64 +
                                 std::size(g_gpr_regnums_WoW64) - 1)}};
enum { k_num_register_sets = 1 };

static const DWORD kWoW64ContextFlags =
    WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER | WOW64_CONTEXT_SEGMENTS;

} // namespace

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  // i686 32bit instruction set.
  assert((target_arch.GetAddressByteSize() == 4 &&
          HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
         "Register setting path assumes this is a 64-bit host");
  return CreateWindowsX86RegisterLayout(target_arch, X86RegisterMode::I386);
}

static Status
GetWoW64ThreadContextHelper(lldb::thread_t thread_handle,
                            PWOW64_CONTEXT context_ptr,
                            const DWORD control_flag = kWoW64ContextFlags) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  memset(context_ptr, 0, sizeof(::WOW64_CONTEXT));
  context_ptr->ContextFlags = control_flag;
  if (!::Wow64GetThreadContext(thread_handle, context_ptr)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} Wow64GetThreadContext failed with error {1}",
             __FUNCTION__, error);
    return error;
  }
  return Status();
}

static Status SetWoW64ThreadContextHelper(lldb::thread_t thread_handle,
                                          PWOW64_CONTEXT context_ptr) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  if (!::Wow64SetThreadContext(thread_handle, context_ptr)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} Wow64SetThreadContext failed with error {1}",
             __FUNCTION__, error);
    return error;
  }
  return Status();
}

NativeRegisterContextWindows_WoW64::NativeRegisterContextWindows_WoW64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)) {
  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (set_index >= k_num_register_sets)
      return nullptr;
    return &g_reg_sets_WoW64[set_index];
  };
  auto previous = GetDescription();
  std::vector<RegisterGroup> groups;
  std::set<RegisterID> visible;
  for (uint32_t index = 0; index < k_num_register_sets; ++index) {
    const auto *input = select_group(index);
    if (!input)
      continue;
    RegisterGroup group{input->name, input->short_name, {}};
    for (uint32_t number : input->registers) {
      const auto *reg = previous->FindByNumber(lldb::eRegisterKindLLDB, number);
      assert(reg && "native presentation names an unknown register");
      if (reg) {
        group.registers.push_back(reg->id);
        visible.insert(reg->id);
      }
    }
    groups.push_back(std::move(group));
  }
  auto description = llvm::cantFail(RegisterDescription::Create(
      {previous->GetRegisters().begin(), previous->GetRegisters().end()},
      std::move(groups),
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()}));
  std::vector<RegisterID> order;
  for (RegisterID id : previous->GetUserRegisters())
    if (visible.count(id))
      order.push_back(id);
  auto presentation = llvm::cantFail(description.WithUserRegisters(order));
  SetRegisterLayout(llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      GetLayout())));
}

bool NativeRegisterContextWindows_WoW64::IsGPR(uint32_t reg_index) const {
  return (reg_index >= reg_ids::k_first_gpr &&
          reg_index < reg_ids::k_first_alias);
}

bool NativeRegisterContextWindows_WoW64::IsDR(uint32_t reg_index) const {
  return (reg_index >= reg_ids::dr0 && reg_index <= reg_ids::dr7);
}

Status NativeRegisterContextWindows_WoW64::GPRRead(const uint32_t reg,
                                                   RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  Status error = GetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::eax:
    reg_value.SetUInt32(tls_context.Eax);
    break;
  case reg_ids::ebx:
    reg_value.SetUInt32(tls_context.Ebx);
    break;
  case reg_ids::ecx:
    reg_value.SetUInt32(tls_context.Ecx);
    break;
  case reg_ids::edx:
    reg_value.SetUInt32(tls_context.Edx);
    break;
  case reg_ids::edi:
    reg_value.SetUInt32(tls_context.Edi);
    break;
  case reg_ids::esi:
    reg_value.SetUInt32(tls_context.Esi);
    break;
  case reg_ids::ebp:
    reg_value.SetUInt32(tls_context.Ebp);
    break;
  case reg_ids::esp:
    reg_value.SetUInt32(tls_context.Esp);
    break;
  case reg_ids::eip:
    reg_value.SetUInt32(tls_context.Eip);
    break;
  case reg_ids::eflags:
    reg_value.SetUInt32(tls_context.EFlags);
    break;
  case reg_ids::cs:
    reg_value.SetUInt32(tls_context.SegCs);
    break;
  case reg_ids::fs:
    reg_value.SetUInt32(tls_context.SegFs);
    break;
  case reg_ids::gs:
    reg_value.SetUInt32(tls_context.SegGs);
    break;
  case reg_ids::ss:
    reg_value.SetUInt32(tls_context.SegSs);
    break;
  case reg_ids::ds:
    reg_value.SetUInt32(tls_context.SegDs);
    break;
  case reg_ids::es:
    reg_value.SetUInt32(tls_context.SegEs);
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_WoW64::GPRWrite(const uint32_t reg,
                                             const RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  auto thread_handle = GetThreadHandle();
  Status error = GetWoW64ThreadContextHelper(thread_handle, &tls_context);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::eax:
    tls_context.Eax = reg_value.GetAsUInt32();
    break;
  case reg_ids::ebx:
    tls_context.Ebx = reg_value.GetAsUInt32();
    break;
  case reg_ids::ecx:
    tls_context.Ecx = reg_value.GetAsUInt32();
    break;
  case reg_ids::edx:
    tls_context.Edx = reg_value.GetAsUInt32();
    break;
  case reg_ids::edi:
    tls_context.Edi = reg_value.GetAsUInt32();
    break;
  case reg_ids::esi:
    tls_context.Esi = reg_value.GetAsUInt32();
    break;
  case reg_ids::ebp:
    tls_context.Ebp = reg_value.GetAsUInt32();
    break;
  case reg_ids::esp:
    tls_context.Esp = reg_value.GetAsUInt32();
    break;
  case reg_ids::eip:
    tls_context.Eip = reg_value.GetAsUInt32();
    break;
  case reg_ids::eflags:
    tls_context.EFlags = reg_value.GetAsUInt32();
    break;
  case reg_ids::cs:
    tls_context.SegCs = reg_value.GetAsUInt32();
    break;
  case reg_ids::fs:
    tls_context.SegFs = reg_value.GetAsUInt32();
    break;
  case reg_ids::gs:
    tls_context.SegGs = reg_value.GetAsUInt32();
    break;
  case reg_ids::ss:
    tls_context.SegSs = reg_value.GetAsUInt32();
    break;
  case reg_ids::ds:
    tls_context.SegDs = reg_value.GetAsUInt32();
    break;
  case reg_ids::es:
    tls_context.SegEs = reg_value.GetAsUInt32();
    break;
  }

  return SetWoW64ThreadContextHelper(thread_handle, &tls_context);
}

Status NativeRegisterContextWindows_WoW64::DRRead(const uint32_t reg,
                                                  RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  DWORD context_flag = CONTEXT_DEBUG_REGISTERS;
  Status error = GetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context,
                                             context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::dr0:
    reg_value.SetUInt32(tls_context.Dr0);
    break;
  case reg_ids::dr1:
    reg_value.SetUInt32(tls_context.Dr1);
    break;
  case reg_ids::dr2:
    reg_value.SetUInt32(tls_context.Dr2);
    break;
  case reg_ids::dr3:
    reg_value.SetUInt32(tls_context.Dr3);
    break;
  case reg_ids::dr4:
    return Status::FromErrorString("register DR4 is obsolete");
  case reg_ids::dr5:
    return Status::FromErrorString("register DR5 is obsolete");
  case reg_ids::dr6:
    reg_value.SetUInt32(tls_context.Dr6);
    break;
  case reg_ids::dr7:
    reg_value.SetUInt32(tls_context.Dr7);
    break;
  }

  return {};
}

Status
NativeRegisterContextWindows_WoW64::DRWrite(const uint32_t reg,
                                            const RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  DWORD context_flag = CONTEXT_DEBUG_REGISTERS;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetWoW64ThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::dr0:
    tls_context.Dr0 = reg_value.GetAsUInt32();
    break;
  case reg_ids::dr1:
    tls_context.Dr1 = reg_value.GetAsUInt32();
    break;
  case reg_ids::dr2:
    tls_context.Dr2 = reg_value.GetAsUInt32();
    break;
  case reg_ids::dr3:
    tls_context.Dr3 = reg_value.GetAsUInt32();
    break;
  case reg_ids::dr4:
    return Status::FromErrorString("register DR4 is obsolete");
  case reg_ids::dr5:
    return Status::FromErrorString("register DR5 is obsolete");
  case reg_ids::dr6:
    tls_context.Dr6 = reg_value.GetAsUInt32();
    break;
  case reg_ids::dr7:
    tls_context.Dr7 = reg_value.GetAsUInt32();
    break;
  }

  return SetWoW64ThreadContextHelper(thread_handle, &tls_context);
}

llvm::Expected<RegisterValue>
NativeRegisterContextWindows_WoW64::ReadRegister(RegisterID id) {
  RegisterValue reg_value;
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  Status error;
  if (!descriptor) {
    error = Status::FromErrorString("unknown register");
    return error.ToError();
  }

  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb "
        "register, cannot read directly",
        descriptor->name.c_str());
    return error.ToError();
  }

  if (IsGPR(reg)) {
    if (Status status = GPRRead(reg, reg_value); status.Fail())
      return status.ToError();
    return reg_value;
  }

  if (IsDR(reg)) {
    if (Status status = DRRead(reg, reg_value); status.Fail())
      return status.ToError();
    return reg_value;
  }

  return Status::FromErrorString("unimplemented").ToError();
}

llvm::Error NativeRegisterContextWindows_WoW64::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  Status error;

  if (!descriptor) {
    error = Status::FromErrorString("unknown register");
    return error.ToError();
  }

  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly written.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb "
        "register, cannot write directly",
        descriptor->name.c_str());
    return error.ToError();
  }

  if (IsGPR(reg))
    return GPRWrite(reg, reg_value).ToError();

  if (IsDR(reg))
    return DRWrite(reg, reg_value).ToError();

  return Status::FromErrorString("unimplemented").ToError();
}

Status NativeRegisterContextWindows_WoW64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  const size_t data_size = REG_CONTEXT_SIZE;
  data_sp = std::make_shared<DataBufferHeap>(data_size, 0);
  ::WOW64_CONTEXT tls_context;
  Status error = GetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context);
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &tls_context, data_size);
  return error;
}

Status NativeRegisterContextWindows_WoW64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;
  const size_t data_size = REG_CONTEXT_SIZE;
  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextWindows_WoW64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != data_size) {
    error = Status::FromErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        data_size, data_sp->GetByteSize());
    return error;
  }

  ::WOW64_CONTEXT tls_context;
  memcpy(&tls_context, data_sp->GetBytes(), data_size);
  return SetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context);
}

Status NativeRegisterContextWindows_WoW64::IsWatchpointHit(uint32_t wp_index,
                                                           bool &is_hit) {
  is_hit = false;

  if (wp_index >= NumSupportedHardwareWatchpoints())
    return Status::FromErrorString("watchpoint index out of range");

  RegisterValue reg_value;
  Status error = DRRead(reg_ids::dr6, reg_value);
  if (error.Fail())
    return error;

  is_hit = reg_value.GetAsUInt32() & (1 << wp_index);

  return {};
}

Status NativeRegisterContextWindows_WoW64::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  wp_index = LLDB_INVALID_INDEX32;

  for (uint32_t i = 0; i < NumSupportedHardwareWatchpoints(); i++) {
    bool is_hit;
    Status error = IsWatchpointHit(i, is_hit);
    if (error.Fail())
      return error;

    if (is_hit) {
      wp_index = i;
      return {};
    }
  }

  return {};
}

Status NativeRegisterContextWindows_WoW64::IsWatchpointVacant(uint32_t wp_index,
                                                              bool &is_vacant) {
  is_vacant = false;

  if (wp_index >= NumSupportedHardwareWatchpoints())
    return Status::FromErrorString("Watchpoint index out of range");

  RegisterValue reg_value;
  Status error = DRRead(reg_ids::dr7, reg_value);
  if (error.Fail())
    return error;

  is_vacant = !(reg_value.GetAsUInt32() & (1 << (2 * wp_index)));

  return error;
}

bool NativeRegisterContextWindows_WoW64::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  if (wp_index >= NumSupportedHardwareWatchpoints())
    return false;

  // for watchpoints 0, 1, 2, or 3, respectively, clear bits 0, 1, 2, or 3 of
  // the debug status register (DR6)

  RegisterValue reg_value;
  Status error = DRRead(reg_ids::dr6, reg_value);
  if (error.Fail())
    return false;

  uint32_t bit_mask = 1 << wp_index;
  uint32_t status_bits = reg_value.GetAsUInt32() & ~bit_mask;
  error = DRWrite(reg_ids::dr6, RegisterValue(status_bits));
  if (error.Fail())
    return false;

  // for watchpoints 0, 1, 2, or 3, respectively, clear bits {0-1,16-19},
  // {2-3,20-23}, {4-5,24-27}, or {6-7,28-31} of the debug control register
  // (DR7)

  error = DRRead(reg_ids::dr7, reg_value);
  if (error.Fail())
    return false;

  bit_mask = (0x3 << (2 * wp_index)) | (0xF << (16 + 4 * wp_index));
  uint32_t control_bits = reg_value.GetAsUInt32() & ~bit_mask;
  return DRWrite(reg_ids::dr7, RegisterValue(control_bits)).Success();
}

Status NativeRegisterContextWindows_WoW64::ClearAllHardwareWatchpoints() {
  RegisterValue reg_value;

  // clear bits {0-4} of the debug status register (DR6)

  Status error = DRRead(reg_ids::dr6, reg_value);
  if (error.Fail())
    return error;

  uint32_t status_bits = reg_value.GetAsUInt32() & ~0xF;
  error = DRWrite(reg_ids::dr6, RegisterValue(status_bits));
  if (error.Fail())
    return error;

  // clear bits {0-7,16-31} of the debug control register (DR7)

  error = DRRead(reg_ids::dr7, reg_value);
  if (error.Fail())
    return error;

  uint32_t control_bits = reg_value.GetAsUInt32() & ~0xFFFF00FF;
  return DRWrite(reg_ids::dr7, RegisterValue(control_bits));
}

uint32_t NativeRegisterContextWindows_WoW64::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  switch (size) {
  case 1:
  case 2:
  case 4:
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  if (watch_flags == 0x2)
    watch_flags = 0x3;

  if (watch_flags != 0x1 && watch_flags != 0x3)
    return LLDB_INVALID_INDEX32;

  for (uint32_t wp_index = 0; wp_index < NumSupportedHardwareWatchpoints();
       ++wp_index) {
    bool is_vacant;
    if (IsWatchpointVacant(wp_index, is_vacant).Fail())
      return LLDB_INVALID_INDEX32;

    if (is_vacant) {
      if (!ClearHardwareWatchpoint(wp_index))
        return LLDB_INVALID_INDEX32;

      if (ApplyHardwareBreakpoint(wp_index, addr, size, watch_flags).Fail())
        return LLDB_INVALID_INDEX32;

      return wp_index;
    }
  }
  return LLDB_INVALID_INDEX32;
}

Status NativeRegisterContextWindows_WoW64::ApplyHardwareBreakpoint(
    uint32_t wp_index, lldb::addr_t addr, size_t size, uint32_t flags) {
  RegisterValue reg_value;
  auto error = DRRead(reg_ids::dr7, reg_value);
  if (error.Fail())
    return error;

  // for watchpoints 0, 1, 2, or 3, respectively, set bits 1, 3, 5, or 7
  uint32_t enable_bit = 1 << (2 * wp_index);

  // set bits 16-17, 20-21, 24-25, or 28-29
  // with 0b01 for write, and 0b11 for read/write
  uint32_t rw_bits = flags << (16 + 4 * wp_index);

  // set bits 18-19, 22-23, 26-27, or 30-31
  // with 0b00, 0b01, 0b10, or 0b11
  // for 1, 2, 8 (if supported), or 4 bytes, respectively
  uint32_t size_bits = (size == 8 ? 0x2 : size - 1) << (18 + 4 * wp_index);

  uint32_t bit_mask = (0x3 << (2 * wp_index)) | (0xF << (16 + 4 * wp_index));

  uint32_t control_bits = reg_value.GetAsUInt32() & ~bit_mask;
  control_bits |= enable_bit | rw_bits | size_bits;

  error = DRWrite(reg_ids::dr7, RegisterValue(control_bits));
  if (error.Fail())
    return error;

  error = DRWrite(reg_ids::dr0 + wp_index, RegisterValue(addr));
  if (error.Fail())
    return error;

  return {};
}

lldb::addr_t
NativeRegisterContextWindows_WoW64::GetWatchpointAddress(uint32_t wp_index) {
  if (wp_index >= NumSupportedHardwareWatchpoints())
    return LLDB_INVALID_ADDRESS;

  RegisterValue reg_value;
  if (DRRead(reg_ids::dr0 + wp_index, reg_value).Fail())
    return LLDB_INVALID_ADDRESS;

  return reg_value.GetAsUInt32();
}

uint32_t NativeRegisterContextWindows_WoW64::NumSupportedHardwareWatchpoints() {
  return 4;
}

#endif // defined(__x86_64__) || defined(_M_X64)
