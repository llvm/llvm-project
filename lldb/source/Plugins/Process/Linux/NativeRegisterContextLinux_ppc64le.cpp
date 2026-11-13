//===-- NativeRegisterContextLinux_ppc64le.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This implementation is related to the OpenPOWER ABI for Power Architecture
// 64-bit ELF V2 ABI

#if defined(__powerpc64__)

#include <set>
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"

#include "NativeRegisterContextLinux_ppc64le.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Register/Common/RegisterInfo_ppc64le.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules
#include <asm/ptrace.h>
#include <elf.h>
#include <sys/socket.h>

#define REG_CONTEXT_SIZE                                                       \
  (GetGPRSize() + GetFPRSize() + sizeof(m_vmx_ppc64le) + sizeof(m_vsx_ppc64le))

namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::linux_ppc64le;
using namespace lldb_private::process_linux;

static const uint32_t g_gpr_regnums_ppc64le[] = {
    reg_ids::gpr_r0,    reg_ids::gpr_r1,     reg_ids::gpr_r2,
    reg_ids::gpr_r3,    reg_ids::gpr_r4,     reg_ids::gpr_r5,
    reg_ids::gpr_r6,    reg_ids::gpr_r7,     reg_ids::gpr_r8,
    reg_ids::gpr_r9,    reg_ids::gpr_r10,    reg_ids::gpr_r11,
    reg_ids::gpr_r12,   reg_ids::gpr_r13,    reg_ids::gpr_r14,
    reg_ids::gpr_r15,   reg_ids::gpr_r16,    reg_ids::gpr_r17,
    reg_ids::gpr_r18,   reg_ids::gpr_r19,    reg_ids::gpr_r20,
    reg_ids::gpr_r21,   reg_ids::gpr_r22,    reg_ids::gpr_r23,
    reg_ids::gpr_r24,   reg_ids::gpr_r25,    reg_ids::gpr_r26,
    reg_ids::gpr_r27,   reg_ids::gpr_r28,    reg_ids::gpr_r29,
    reg_ids::gpr_r30,   reg_ids::gpr_r31,    reg_ids::gpr_pc,
    reg_ids::gpr_msr,   reg_ids::gpr_origr3, reg_ids::gpr_ctr,
    reg_ids::gpr_lr,    reg_ids::gpr_xer,    reg_ids::gpr_cr,
    reg_ids::gpr_softe, reg_ids::gpr_trap,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_fpr_regnums_ppc64le[] = {
    reg_ids::fpr_f0,    reg_ids::fpr_f1,  reg_ids::fpr_f2,  reg_ids::fpr_f3,
    reg_ids::fpr_f4,    reg_ids::fpr_f5,  reg_ids::fpr_f6,  reg_ids::fpr_f7,
    reg_ids::fpr_f8,    reg_ids::fpr_f9,  reg_ids::fpr_f10, reg_ids::fpr_f11,
    reg_ids::fpr_f12,   reg_ids::fpr_f13, reg_ids::fpr_f14, reg_ids::fpr_f15,
    reg_ids::fpr_f16,   reg_ids::fpr_f17, reg_ids::fpr_f18, reg_ids::fpr_f19,
    reg_ids::fpr_f20,   reg_ids::fpr_f21, reg_ids::fpr_f22, reg_ids::fpr_f23,
    reg_ids::fpr_f24,   reg_ids::fpr_f25, reg_ids::fpr_f26, reg_ids::fpr_f27,
    reg_ids::fpr_f28,   reg_ids::fpr_f29, reg_ids::fpr_f30, reg_ids::fpr_f31,
    reg_ids::fpr_fpscr,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_vmx_regnums_ppc64le[] = {
    reg_ids::vmx_vr0,    reg_ids::vmx_vr1,  reg_ids::vmx_vr2,
    reg_ids::vmx_vr3,    reg_ids::vmx_vr4,  reg_ids::vmx_vr5,
    reg_ids::vmx_vr6,    reg_ids::vmx_vr7,  reg_ids::vmx_vr8,
    reg_ids::vmx_vr9,    reg_ids::vmx_vr10, reg_ids::vmx_vr11,
    reg_ids::vmx_vr12,   reg_ids::vmx_vr13, reg_ids::vmx_vr14,
    reg_ids::vmx_vr15,   reg_ids::vmx_vr16, reg_ids::vmx_vr17,
    reg_ids::vmx_vr18,   reg_ids::vmx_vr19, reg_ids::vmx_vr20,
    reg_ids::vmx_vr21,   reg_ids::vmx_vr22, reg_ids::vmx_vr23,
    reg_ids::vmx_vr24,   reg_ids::vmx_vr25, reg_ids::vmx_vr26,
    reg_ids::vmx_vr27,   reg_ids::vmx_vr28, reg_ids::vmx_vr29,
    reg_ids::vmx_vr30,   reg_ids::vmx_vr31, reg_ids::vmx_vscr,
    reg_ids::vmx_vrsave,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_vsx_regnums_ppc64le[] = {
    reg_ids::vsx_vs0,   reg_ids::vsx_vs1,  reg_ids::vsx_vs2,  reg_ids::vsx_vs3,
    reg_ids::vsx_vs4,   reg_ids::vsx_vs5,  reg_ids::vsx_vs6,  reg_ids::vsx_vs7,
    reg_ids::vsx_vs8,   reg_ids::vsx_vs9,  reg_ids::vsx_vs10, reg_ids::vsx_vs11,
    reg_ids::vsx_vs12,  reg_ids::vsx_vs13, reg_ids::vsx_vs14, reg_ids::vsx_vs15,
    reg_ids::vsx_vs16,  reg_ids::vsx_vs17, reg_ids::vsx_vs18, reg_ids::vsx_vs19,
    reg_ids::vsx_vs20,  reg_ids::vsx_vs21, reg_ids::vsx_vs22, reg_ids::vsx_vs23,
    reg_ids::vsx_vs24,  reg_ids::vsx_vs25, reg_ids::vsx_vs26, reg_ids::vsx_vs27,
    reg_ids::vsx_vs28,  reg_ids::vsx_vs29, reg_ids::vsx_vs30, reg_ids::vsx_vs31,
    reg_ids::vsx_vs32,  reg_ids::vsx_vs33, reg_ids::vsx_vs34, reg_ids::vsx_vs35,
    reg_ids::vsx_vs36,  reg_ids::vsx_vs37, reg_ids::vsx_vs38, reg_ids::vsx_vs39,
    reg_ids::vsx_vs40,  reg_ids::vsx_vs41, reg_ids::vsx_vs42, reg_ids::vsx_vs43,
    reg_ids::vsx_vs44,  reg_ids::vsx_vs45, reg_ids::vsx_vs46, reg_ids::vsx_vs47,
    reg_ids::vsx_vs48,  reg_ids::vsx_vs49, reg_ids::vsx_vs50, reg_ids::vsx_vs51,
    reg_ids::vsx_vs52,  reg_ids::vsx_vs53, reg_ids::vsx_vs54, reg_ids::vsx_vs55,
    reg_ids::vsx_vs56,  reg_ids::vsx_vs57, reg_ids::vsx_vs58, reg_ids::vsx_vs59,
    reg_ids::vsx_vs60,  reg_ids::vsx_vs61, reg_ids::vsx_vs62, reg_ids::vsx_vs63,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

// Number of register sets provided by this context.
static constexpr int k_num_register_sets = 4;

struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_ppc64le[k_num_register_sets] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_ppc64le,
                             g_gpr_regnums_ppc64le +
                                 reg_ids::k_num_gpr_registers)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpr_regnums_ppc64le,
                             g_fpr_regnums_ppc64le +
                                 reg_ids::k_num_fpr_registers)},
    {"AltiVec/VMX Registers", "vmx",
     std::vector<RegisterID>(g_vmx_regnums_ppc64le,
                             g_vmx_regnums_ppc64le +
                                 reg_ids::k_num_vmx_registers)},
    {"VSX Registers", "vsx",
     std::vector<RegisterID>(g_vsx_regnums_ppc64le,
                             g_vsx_regnums_ppc64le +
                                 reg_ids::k_num_vsx_registers)}};

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::ppc64le:
    return std::make_unique<NativeRegisterContextLinux_ppc64le>(target_arch,
                                                                 native_thread);
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return HostInfo::GetArchitecture();
}

NativeRegisterContextLinux_ppc64le::NativeRegisterContextLinux_ppc64le(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(
          native_thread, target_arch,
          CreateLinuxPPC64LERegisterLayout(target_arch).release()),
      NativeRegisterContextLinux(native_thread) {
  if (target_arch.GetMachine() != llvm::Triple::ppc64le) {
    llvm_unreachable("Unhandled target architecture.");
  }

  ::memset(&m_gpr_ppc64le, 0, sizeof(m_gpr_ppc64le));
  ::memset(&m_fpr_ppc64le, 0, sizeof(m_fpr_ppc64le));
  ::memset(&m_vmx_ppc64le, 0, sizeof(m_vmx_ppc64le));
  ::memset(&m_vsx_ppc64le, 0, sizeof(m_vsx_ppc64le));

  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (set_index < k_num_register_sets)
      return &g_reg_sets_ppc64le[set_index];

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

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_ppc64le::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, false);
  if (!required)
    return required.takeError();
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
      status = ReadVMX();
      break;
    case 3:
      status = ReadVSX();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id, {{0,
                        {reinterpret_cast<const uint8_t *>(&m_gpr_ppc64le),
                         sizeof(m_gpr_ppc64le)}},
                       {1,
                        {reinterpret_cast<const uint8_t *>(&m_fpr_ppc64le),
                         sizeof(m_fpr_ppc64le)}},
                       {2,
                        {reinterpret_cast<const uint8_t *>(&m_vmx_ppc64le),
                         sizeof(m_vmx_ppc64le)}},
                       {3,
                        {reinterpret_cast<const uint8_t *>(&m_vsx_ppc64le),
                         sizeof(m_vsx_ppc64le)}}});
  return value;
}

llvm::Error NativeRegisterContextLinux_ppc64le::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, true);
  if (!required)
    return required.takeError();
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
      status = ReadVMX();
      break;
    case 3:
      status = ReadVSX();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (auto error =
          WriteBankRegister(descriptor->id, reg_value,
                            {{0,
                              {reinterpret_cast<uint8_t *>(&m_gpr_ppc64le),
                               sizeof(m_gpr_ppc64le)}},
                             {1,
                              {reinterpret_cast<uint8_t *>(&m_fpr_ppc64le),
                               sizeof(m_fpr_ppc64le)}},
                             {2,
                              {reinterpret_cast<uint8_t *>(&m_vmx_ppc64le),
                               sizeof(m_vmx_ppc64le)}},
                             {3,
                              {reinterpret_cast<uint8_t *>(&m_vsx_ppc64le),
                               sizeof(m_vsx_ppc64le)}}}))
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
    case 2:
      status = WriteVMX();
      break;
    case 3:
      status = WriteVSX();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextLinux_ppc64le::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  error = ReadVMX();
  if (error.Fail())
    return error;

  error = ReadVSX();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_gpr_ppc64le, GetGPRSize());
  dst += GetGPRSize();
  ::memcpy(dst, &m_fpr_ppc64le, GetFPRSize());
  dst += GetFPRSize();
  ::memcpy(dst, &m_vmx_ppc64le, sizeof(m_vmx_ppc64le));
  dst += sizeof(m_vmx_ppc64le);
  ::memcpy(dst, &m_vsx_ppc64le, sizeof(m_vsx_ppc64le));

  return error;
}

Status NativeRegisterContextLinux_ppc64le::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_ppc64le::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_ppc64le::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_ppc64le::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }

  ::memcpy(&m_gpr_ppc64le, src, GetGPRSize());
  error = WriteGPR();

  if (error.Fail())
    return error;

  src += GetGPRSize();
  ::memcpy(&m_fpr_ppc64le, src, GetFPRSize());

  error = WriteFPR();
  if (error.Fail())
    return error;

  src += GetFPRSize();
  ::memcpy(&m_vmx_ppc64le, src, sizeof(m_vmx_ppc64le));

  error = WriteVMX();
  if (error.Fail())
    return error;

  src += sizeof(m_vmx_ppc64le);
  ::memcpy(&m_vsx_ppc64le, src, sizeof(m_vsx_ppc64le));
  error = WriteVSX();

  return error;
}

bool NativeRegisterContextLinux_ppc64le::IsGPR(unsigned reg) const {
  return reg <= reg_ids::k_last_gpr; // GPR's come first.
}

bool NativeRegisterContextLinux_ppc64le::IsFPR(unsigned reg) const {
  return (reg_ids::k_first_fpr <= reg && reg <= reg_ids::k_last_fpr);
}

Status NativeRegisterContextLinux_ppc64le::ReadVMX() {
  int regset = NT_PPC_VMX;
  return NativeProcessLinux::PtraceWrapper(PTRACE_GETVRREGS, m_thread.GetID(),
                                           &regset, &m_vmx_ppc64le,
                                           sizeof(m_vmx_ppc64le));
}

Status NativeRegisterContextLinux_ppc64le::WriteVMX() {
  int regset = NT_PPC_VMX;
  return NativeProcessLinux::PtraceWrapper(PTRACE_SETVRREGS, m_thread.GetID(),
                                           &regset, &m_vmx_ppc64le,
                                           sizeof(m_vmx_ppc64le));
}

Status NativeRegisterContextLinux_ppc64le::ReadVSX() {
  int regset = NT_PPC_VSX;
  return NativeProcessLinux::PtraceWrapper(PTRACE_GETVSRREGS, m_thread.GetID(),
                                           &regset, &m_vsx_ppc64le,
                                           sizeof(m_vsx_ppc64le));
}

Status NativeRegisterContextLinux_ppc64le::WriteVSX() {
  int regset = NT_PPC_VSX;
  return NativeProcessLinux::PtraceWrapper(PTRACE_SETVSRREGS, m_thread.GetID(),
                                           &regset, &m_vsx_ppc64le,
                                           sizeof(m_vsx_ppc64le));
}

bool NativeRegisterContextLinux_ppc64le::IsVMX(unsigned reg) {
  return (reg >= reg_ids::k_first_vmx) && (reg <= reg_ids::k_last_vmx);
}

bool NativeRegisterContextLinux_ppc64le::IsVSX(unsigned reg) {
  return (reg >= reg_ids::k_first_vsx) && (reg <= reg_ids::k_last_vsx);
}

uint32_t NativeRegisterContextLinux_ppc64le::NumSupportedHardwareWatchpoints() {
  Log *log = GetLog(POSIXLog::Watchpoints);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return 0;

  LLDB_LOG(log, "{0}", m_max_hwp_supported);
  return m_max_hwp_supported;
}

uint32_t NativeRegisterContextLinux_ppc64le::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "addr: {0:x}, size: {1:x} watch_flags: {2:x}", addr, size,
           watch_flags);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return LLDB_INVALID_INDEX32;

  uint32_t control_value = 0, wp_index = 0;
  lldb::addr_t real_addr = addr;
  uint32_t rw_mode = 0;

  // Check if we are setting watchpoint other than read/write/access Update
  // watchpoint flag to match ppc64le write-read bit configuration.
  switch (watch_flags) {
  case eWatchpointKindWrite:
    rw_mode = PPC_BREAKPOINT_TRIGGER_WRITE;
    watch_flags = 2;
    break;
  case eWatchpointKindRead:
    rw_mode = PPC_BREAKPOINT_TRIGGER_READ;
    watch_flags = 1;
    break;
  case (eWatchpointKindRead | eWatchpointKindWrite):
    rw_mode = PPC_BREAKPOINT_TRIGGER_RW;
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  // Check if size has a valid hardware watchpoint length.
  if (size != 1 && size != 2 && size != 4 && size != 8)
    return LLDB_INVALID_INDEX32;

  // Check 8-byte alignment for hardware watchpoint target address. Below is a
  // hack to recalculate address and size in order to make sure we can watch
  // non 8-byte aligned addresses as well.
  if (addr & 0x07) {

    addr_t begin = llvm::alignDown(addr, 8);
    addr_t end = llvm::alignTo(addr + size, 8);
    size = llvm::PowerOf2Ceil(end - begin);

    addr = addr & (~0x07);
  }

  // Setup control value
  control_value = watch_flags << 3;
  control_value |= ((1 << size) - 1) << 5;
  control_value |= (2 << 1) | 1;

  // Iterate over stored watchpoints and find a free wp_index
  wp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    if ((m_hwp_regs[i].control & 1) == 0) {
      wp_index = i; // Mark last free slot
    } else if (m_hwp_regs[i].address == addr) {
      return LLDB_INVALID_INDEX32; // We do not support duplicate watchpoints.
    }
  }

  if (wp_index == LLDB_INVALID_INDEX32)
    return LLDB_INVALID_INDEX32;

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].real_addr = real_addr;
  m_hwp_regs[wp_index].address = addr;
  m_hwp_regs[wp_index].control = control_value;
  m_hwp_regs[wp_index].mode = rw_mode;

  // PTRACE call to set corresponding watchpoint register.
  error = WriteHardwareDebugRegs();

  if (error.Fail()) {
    m_hwp_regs[wp_index].address = 0;
    m_hwp_regs[wp_index].control &= llvm::maskTrailingZeros<uint32_t>(1);

    return LLDB_INVALID_INDEX32;
  }

  return wp_index;
}

bool NativeRegisterContextLinux_ppc64le::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return false;

  if (wp_index >= m_max_hwp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hwp_regs[wp_index].address;
  uint32_t tempControl = m_hwp_regs[wp_index].control;
  long *tempSlot = reinterpret_cast<long *>(m_hwp_regs[wp_index].slot);

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].control &= llvm::maskTrailingZeros<uint32_t>(1);
  m_hwp_regs[wp_index].address = 0;
  m_hwp_regs[wp_index].slot = 0;
  m_hwp_regs[wp_index].mode = 0;

  // Ptrace call to update hardware debug registers
  error = NativeProcessLinux::PtraceWrapper(PPC_PTRACE_DELHWDEBUG,
                                            m_thread.GetID(), 0, tempSlot);

  if (error.Fail()) {
    m_hwp_regs[wp_index].control = tempControl;
    m_hwp_regs[wp_index].address = tempAddr;
    m_hwp_regs[wp_index].slot = reinterpret_cast<long>(tempSlot);

    return false;
  }

  return true;
}

uint32_t
NativeRegisterContextLinux_ppc64le::GetWatchpointSize(uint32_t wp_index) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  unsigned control = (m_hwp_regs[wp_index].control >> 5) & 0xff;
  if (llvm::isPowerOf2_32(control + 1)) {
    return llvm::popcount(control);
  }

  return 0;
}

bool NativeRegisterContextLinux_ppc64le::WatchpointIsEnabled(
    uint32_t wp_index) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  return !!((m_hwp_regs[wp_index].control & 0x1) == 0x1);
}

Status NativeRegisterContextLinux_ppc64le::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}, trap_addr: {1:x}", wp_index, trap_addr);

  uint32_t watch_size;
  lldb::addr_t watch_addr;

  for (wp_index = 0; wp_index < m_max_hwp_supported; ++wp_index) {
    watch_size = GetWatchpointSize(wp_index);
    watch_addr = m_hwp_regs[wp_index].address;

    if (WatchpointIsEnabled(wp_index) && trap_addr >= watch_addr &&
        trap_addr <= watch_addr + watch_size) {
      m_hwp_regs[wp_index].hit_addr = trap_addr;
      return Status();
    }
  }

  wp_index = LLDB_INVALID_INDEX32;
  return Status();
}

lldb::addr_t
NativeRegisterContextLinux_ppc64le::GetWatchpointAddress(uint32_t wp_index) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].real_addr;
  else
    return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
NativeRegisterContextLinux_ppc64le::GetWatchpointHitAddress(uint32_t wp_index) {
  Log *log = GetLog(POSIXLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].hit_addr;

  return LLDB_INVALID_ADDRESS;
}

Status NativeRegisterContextLinux_ppc64le::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info) {
    return Status();
  }

  ::pid_t tid = m_thread.GetID();

  struct ppc_debug_info hwdebug_info;
  Status error;

  error = NativeProcessLinux::PtraceWrapper(
      PPC_PTRACE_GETHWDBGINFO, tid, 0, &hwdebug_info, sizeof(hwdebug_info));

  if (error.Fail())
    return error;

  m_max_hwp_supported = hwdebug_info.num_data_bps;
  m_max_hbp_supported = hwdebug_info.num_instruction_bps;
  m_refresh_hwdebug_info = false;

  return error;
}

Status NativeRegisterContextLinux_ppc64le::WriteHardwareDebugRegs() {
  struct ppc_hw_breakpoint reg_state;
  Status error;
  long ret;

  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    reg_state.addr = m_hwp_regs[i].address;
    reg_state.trigger_type = m_hwp_regs[i].mode;
    reg_state.version = 1;
    reg_state.addr_mode = PPC_BREAKPOINT_MODE_EXACT;
    reg_state.condition_mode = PPC_BREAKPOINT_CONDITION_NONE;
    reg_state.addr2 = 0;
    reg_state.condition_value = 0;

    error = NativeProcessLinux::PtraceWrapper(PPC_PTRACE_SETHWDEBUG,
                                              m_thread.GetID(), 0, &reg_state,
                                              sizeof(reg_state), &ret);

    if (error.Fail())
      return error;

    m_hwp_regs[i].slot = ret;
  }

  return error;
}

#endif // defined(__powerpc64__)
