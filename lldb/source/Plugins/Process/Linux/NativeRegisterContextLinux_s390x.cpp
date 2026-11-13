//===-- NativeRegisterContextLinux_s390x.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__s390x__) && defined(__linux__)

#include <set>
#include "lldb/Register/Arch/lldb-s390x-register-enums.h"

#include "NativeRegisterContextLinux_s390x.h"
#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_s390x.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include <sys/ptrace.h>
#include <sys/uio.h>

namespace reg_ids = lldb_private::registers::s390x::id;
using namespace lldb_private;
using namespace lldb_private::process_linux;

// Private namespace.

namespace {
// s390x 64-bit general purpose registers.
static const uint32_t g_gpr_regnums_s390x[] = {
    reg_ids::r0,        reg_ids::r1,    reg_ids::r2,    reg_ids::r3,
    reg_ids::r4,        reg_ids::r5,    reg_ids::r6,    reg_ids::r7,
    reg_ids::r8,        reg_ids::r9,    reg_ids::r10,   reg_ids::r11,
    reg_ids::r12,       reg_ids::r13,   reg_ids::r14,   reg_ids::r15,
    reg_ids::acr0,      reg_ids::acr1,  reg_ids::acr2,  reg_ids::acr3,
    reg_ids::acr4,      reg_ids::acr5,  reg_ids::acr6,  reg_ids::acr7,
    reg_ids::acr8,      reg_ids::acr9,  reg_ids::acr10, reg_ids::acr11,
    reg_ids::acr12,     reg_ids::acr13, reg_ids::acr14, reg_ids::acr15,
    reg_ids::pswm,      reg_ids::pswa,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_s390x) / sizeof(g_gpr_regnums_s390x[0])) -
                      1 ==
                  reg_ids::k_num_gpr_registers,
              "g_gpr_regnums_s390x has wrong number of register infos");

// s390x 64-bit floating point registers.
static const uint32_t g_fpu_regnums_s390x[] = {
    reg_ids::f0,        reg_ids::f1,  reg_ids::f2,  reg_ids::f3,  reg_ids::f4,
    reg_ids::f5,        reg_ids::f6,  reg_ids::f7,  reg_ids::f8,  reg_ids::f9,
    reg_ids::f10,       reg_ids::f11, reg_ids::f12, reg_ids::f13, reg_ids::f14,
    reg_ids::f15,       reg_ids::fpc,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_s390x) / sizeof(g_fpu_regnums_s390x[0])) -
                      1 ==
                  reg_ids::k_num_fpr_registers,
              "g_fpu_regnums_s390x has wrong number of register infos");

// s390x Linux operating-system information.
static const uint32_t g_linux_regnums_s390x[] = {
    reg_ids::orig_r2, reg_ids::last_break, reg_ids::system_call,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_linux_regnums_s390x) /
               sizeof(g_linux_regnums_s390x[0])) -
                      1 ==
                  reg_ids::k_num_linux_registers,
              "g_linux_regnums_s390x has wrong number of register infos");

// Number of register sets provided by this context.
enum { k_num_register_sets = 3 };

// Register sets for s390x 64-bit.
struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_s390x[k_num_register_sets] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_s390x,
                             g_gpr_regnums_s390x +
                                 reg_ids::k_num_gpr_registers)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpu_regnums_s390x,
                             g_fpu_regnums_s390x +
                                 reg_ids::k_num_fpr_registers)},
    {"Linux Operating System Data", "linux",
     std::vector<RegisterID>(g_linux_regnums_s390x,
                             g_linux_regnums_s390x +
                                 reg_ids::k_num_linux_registers)}};
} // namespace

#define REG_CONTEXT_SIZE (sizeof(s390_regs) + sizeof(s390_fp_regs) + 4)

// Required ptrace defines.

#define NT_S390_LAST_BREAK 0x306  /* s390 breaking event address */
#define NT_S390_SYSTEM_CALL 0x307 /* s390 system call restart data */

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  return std::make_unique<NativeRegisterContextLinux_s390x>(target_arch,
                                                             native_thread);
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return HostInfo::GetArchitecture();
}

// NativeRegisterContextLinux_s390x members.

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
         "Register setting path assumes this is a 64-bit host");
  return CreateLinuxS390XRegisterLayout(target_arch);
}

NativeRegisterContextLinux_s390x::NativeRegisterContextLinux_s390x(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)),
      NativeRegisterContextLinux(native_thread) {
  // Set up data about ranges of valid registers.
  switch (target_arch.GetMachine()) {
  case llvm::Triple::systemz:
    m_reg_info.num_registers = reg_ids::k_num_registers;
    m_reg_info.num_gpr_registers = reg_ids::k_num_gpr_registers;
    m_reg_info.num_fpr_registers = reg_ids::k_num_fpr_registers;
    m_reg_info.last_gpr = reg_ids::k_last_gpr;
    m_reg_info.first_fpr = reg_ids::k_first_fpr;
    m_reg_info.last_fpr = reg_ids::k_last_fpr;
    break;
  default:
    assert(false && "Unhandled target architecture.");
    break;
  }

  // Clear out the watchpoint state.
  m_watchpoint_addr = LLDB_INVALID_ADDRESS;

  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (!IsRegisterSetAvailable(set_index))
      return nullptr;

    switch (GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::systemz:
      return &g_reg_sets_s390x[set_index];
    default:
      assert(false && "Unhandled target architecture.");
      return nullptr;
    }

    return nullptr;
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

bool NativeRegisterContextLinux_s390x::IsRegisterSetAvailable(
    uint32_t set_index) const {
  return set_index < k_num_register_sets;
}

bool NativeRegisterContextLinux_s390x::IsGPR(uint32_t reg_index) const {
  // GPRs come first.  "orig_r2" counts as GPR since it is part of the GPR
  // register area.
  return reg_index <= m_reg_info.last_gpr || reg_index == reg_ids::orig_r2;
}

bool NativeRegisterContextLinux_s390x::IsFPR(uint32_t reg_index) const {
  return (m_reg_info.first_fpr <= reg_index &&
          reg_index <= m_reg_info.last_fpr);
}

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_s390x::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, false);
  if (!required)
    return required.takeError();
  uint64_t last_break = 0;
  uint32_t system_call = 0;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = ReadGPR();
      break;
    case 1:
      status = ReadFPR();
      break;
    case 2:
      status = DoReadRegisterSet(NT_S390_LAST_BREAK, &last_break,
                                 sizeof(last_break));
      break;
    case 3:
      status = DoReadRegisterSet(NT_S390_SYSTEM_CALL, &system_call,
                                 sizeof(system_call));
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{0, {reinterpret_cast<const uint8_t *>(&m_regs), sizeof(m_regs)}},
       {1, {reinterpret_cast<const uint8_t *>(&m_fp_regs), sizeof(m_fp_regs)}},
       {2,
        {reinterpret_cast<const uint8_t *>(&last_break), sizeof(last_break)}},
       {3,
        {reinterpret_cast<const uint8_t *>(&system_call),
         sizeof(system_call)}}});
  return value;
}

llvm::Error NativeRegisterContextLinux_s390x::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, true);
  if (!required)
    return required.takeError();
  uint64_t last_break = 0;
  uint32_t system_call = 0;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = ReadGPR();
      break;
    case 1:
      status = ReadFPR();
      break;
    case 2:
      return Status::FromErrorString("The last break address is read-only")
          .ToError();
    case 3:
      status = DoReadRegisterSet(NT_S390_SYSTEM_CALL, &system_call,
                                 sizeof(system_call));
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (auto error = WriteBankRegister(
          descriptor->id, reg_value,
          {{0, {reinterpret_cast<uint8_t *>(&m_regs), sizeof(m_regs)}},
           {1, {reinterpret_cast<uint8_t *>(&m_fp_regs), sizeof(m_fp_regs)}},
           {2, {reinterpret_cast<uint8_t *>(&last_break), sizeof(last_break)}},
           {3,
            {reinterpret_cast<uint8_t *>(&system_call), sizeof(system_call)}}}))
    return error;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = WriteGPR();
      break;
    case 1:
      status = WriteFPR();
      break;
    case 3:
      status = DoWriteRegisterSet(NT_S390_SYSTEM_CALL, &system_call,
                                  sizeof(system_call));
      break;
    default:
      llvm_unreachable("validated writable bank");
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextLinux_s390x::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  uint8_t *dst = data_sp->GetBytes();
  error = ReadGPR();
  if (error.Fail())
    return error;
  memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();

  error = ReadFPR();
  if (error.Fail())
    return error;
  memcpy(dst, GetFPRBuffer(), GetFPRSize());
  dst += GetFPRSize();

  // Ignore errors if the regset is unsupported (happens on older kernels).
  DoReadRegisterSet(NT_S390_SYSTEM_CALL, dst, 4);
  dst += 4;

  // To enable inferior function calls while the process is stopped in an
  // interrupted system call, we need to clear the system call flag. It will be
  // restored to its original value by WriteAllRegisterValues. Again we ignore
  // error if the regset is unsupported.
  uint32_t system_call = 0;
  DoWriteRegisterSet(NT_S390_SYSTEM_CALL, &system_call, 4);

  return error;
}

Status NativeRegisterContextLinux_s390x::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_s390x::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_s390x::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_s390x::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }

  memcpy(GetGPRBuffer(), src, GetGPRSize());
  src += GetGPRSize();
  error = WriteGPR();
  if (error.Fail())
    return error;

  memcpy(GetFPRBuffer(), src, GetFPRSize());
  src += GetFPRSize();
  error = WriteFPR();
  if (error.Fail())
    return error;

  // Ignore errors if the regset is unsupported (happens on older kernels).
  DoWriteRegisterSet(NT_S390_SYSTEM_CALL, src, 4);
  src += 4;

  return error;
}

Status NativeRegisterContextLinux_s390x::DoReadRegisterValue(
    uint32_t offset, const char *reg_name, uint32_t size,
    RegisterValue &value) {
  return Status::FromErrorString("DoReadRegisterValue unsupported");
}

Status NativeRegisterContextLinux_s390x::DoWriteRegisterValue(
    uint32_t offset, const char *reg_name, const RegisterValue &value) {
  return Status::FromErrorString("DoWriteRegisterValue unsupported");
}

Status NativeRegisterContextLinux_s390x::PeekUserArea(uint32_t offset,
                                                      void *buf,
                                                      size_t buf_size) {
  ptrace_area parea;
  parea.len = buf_size;
  parea.process_addr = (addr_t)buf;
  parea.kernel_addr = offset;

  return NativeProcessLinux::PtraceWrapper(PTRACE_PEEKUSR_AREA,
                                           m_thread.GetID(), &parea);
}

Status NativeRegisterContextLinux_s390x::PokeUserArea(uint32_t offset,
                                                      const void *buf,
                                                      size_t buf_size) {
  ptrace_area parea;
  parea.len = buf_size;
  parea.process_addr = (addr_t)buf;
  parea.kernel_addr = offset;

  return NativeProcessLinux::PtraceWrapper(PTRACE_POKEUSR_AREA,
                                           m_thread.GetID(), &parea);
}

Status NativeRegisterContextLinux_s390x::ReadGPR() {
  return PeekUserArea(offsetof(user_regs_struct, psw), GetGPRBuffer(),
                      GetGPRSize());
}

Status NativeRegisterContextLinux_s390x::WriteGPR() {
  return PokeUserArea(offsetof(user_regs_struct, psw), GetGPRBuffer(),
                      GetGPRSize());
}

Status NativeRegisterContextLinux_s390x::ReadFPR() {
  return PeekUserArea(offsetof(user_regs_struct, fp_regs), GetGPRBuffer(),
                      GetGPRSize());
}

Status NativeRegisterContextLinux_s390x::WriteFPR() {
  return PokeUserArea(offsetof(user_regs_struct, fp_regs), GetGPRBuffer(),
                      GetGPRSize());
}

Status NativeRegisterContextLinux_s390x::DoReadRegisterSet(uint32_t regset,
                                                           void *buf,
                                                           size_t buf_size) {
  struct iovec iov;
  iov.iov_base = buf;
  iov.iov_len = buf_size;

  return ReadRegisterSet(&iov, buf_size, regset);
}

Status NativeRegisterContextLinux_s390x::DoWriteRegisterSet(uint32_t regset,
                                                            const void *buf,
                                                            size_t buf_size) {
  struct iovec iov;
  iov.iov_base = const_cast<void *>(buf);
  iov.iov_len = buf_size;

  return WriteRegisterSet(&iov, buf_size, regset);
}

Status NativeRegisterContextLinux_s390x::IsWatchpointHit(uint32_t wp_index,
                                                         bool &is_hit) {
  per_lowcore_bits per_lowcore;

  if (wp_index >= NumSupportedHardwareWatchpoints())
    return Status::FromErrorString("Watchpoint index out of range");

  if (m_watchpoint_addr == LLDB_INVALID_ADDRESS) {
    is_hit = false;
    return Status();
  }

  Status error = PeekUserArea(offsetof(user_regs_struct, per_info.lowcore),
                              &per_lowcore, sizeof(per_lowcore));
  if (error.Fail()) {
    is_hit = false;
    return error;
  }

  is_hit = (per_lowcore.perc_storage_alteration == 1 &&
            per_lowcore.perc_store_real_address == 0);

  if (is_hit) {
    // Do not report this watchpoint again.
    memset(&per_lowcore, 0, sizeof(per_lowcore));
    PokeUserArea(offsetof(user_regs_struct, per_info.lowcore), &per_lowcore,
                 sizeof(per_lowcore));
  }

  return Status();
}

Status NativeRegisterContextLinux_s390x::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  uint32_t num_hw_wps = NumSupportedHardwareWatchpoints();
  for (wp_index = 0; wp_index < num_hw_wps; ++wp_index) {
    bool is_hit;
    Status error = IsWatchpointHit(wp_index, is_hit);
    if (error.Fail()) {
      wp_index = LLDB_INVALID_INDEX32;
      return error;
    } else if (is_hit) {
      return error;
    }
  }
  wp_index = LLDB_INVALID_INDEX32;
  return Status();
}

Status NativeRegisterContextLinux_s390x::IsWatchpointVacant(uint32_t wp_index,
                                                            bool &is_vacant) {
  if (wp_index >= NumSupportedHardwareWatchpoints())
    return Status::FromErrorString("Watchpoint index out of range");

  is_vacant = m_watchpoint_addr == LLDB_INVALID_ADDRESS;

  return Status();
}

bool NativeRegisterContextLinux_s390x::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  per_struct per_info;

  if (wp_index >= NumSupportedHardwareWatchpoints())
    return false;

  Status error = PeekUserArea(offsetof(user_regs_struct, per_info), &per_info,
                              sizeof(per_info));
  if (error.Fail())
    return false;

  per_info.control_regs.bits.em_storage_alteration = 0;
  per_info.control_regs.bits.storage_alt_space_ctl = 0;
  per_info.starting_addr = 0;
  per_info.ending_addr = 0;

  error = PokeUserArea(offsetof(user_regs_struct, per_info), &per_info,
                       sizeof(per_info));
  if (error.Fail())
    return false;

  m_watchpoint_addr = LLDB_INVALID_ADDRESS;
  return true;
}

Status NativeRegisterContextLinux_s390x::ClearAllHardwareWatchpoints() {
  if (ClearHardwareWatchpoint(0))
    return Status();
  return Status::FromErrorString("Clearing all hardware watchpoints failed.");
}

uint32_t NativeRegisterContextLinux_s390x::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  per_struct per_info;

  if (watch_flags != 0x1)
    return LLDB_INVALID_INDEX32;

  if (m_watchpoint_addr != LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_INDEX32;

  Status error = PeekUserArea(offsetof(user_regs_struct, per_info), &per_info,
                              sizeof(per_info));
  if (error.Fail())
    return LLDB_INVALID_INDEX32;

  per_info.control_regs.bits.em_storage_alteration = 1;
  per_info.control_regs.bits.storage_alt_space_ctl = 1;
  per_info.starting_addr = addr;
  per_info.ending_addr = addr + size - 1;

  error = PokeUserArea(offsetof(user_regs_struct, per_info), &per_info,
                       sizeof(per_info));
  if (error.Fail())
    return LLDB_INVALID_INDEX32;

  m_watchpoint_addr = addr;
  return 0;
}

lldb::addr_t
NativeRegisterContextLinux_s390x::GetWatchpointAddress(uint32_t wp_index) {
  if (wp_index >= NumSupportedHardwareWatchpoints())
    return LLDB_INVALID_ADDRESS;
  return m_watchpoint_addr;
}

uint32_t NativeRegisterContextLinux_s390x::NumSupportedHardwareWatchpoints() {
  return 1;
}

#endif // defined(__s390x__) && defined(__linux__)
