//===-- NativeRegisterContextLinux_loongarch64.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__loongarch__) && __loongarch_grlen == 64

#include "NativeRegisterContextLinux_loongarch64.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules.
#include <elf.h>
#include <sys/ptrace.h>
#include <sys/uio.h>

#ifndef PTRACE_GETREGSET
#define PTRACE_GETREGSET 0x4204
#endif

#ifndef PTRACE_SETREGSET
#define PTRACE_SETREGSET 0x4205
#endif

// LoongArch SIMD eXtension registers
#ifndef NT_LOONGARCH_LSX
#define NT_LOONGARCH_LSX 0xa02
#endif

// LoongArch Advanced SIMD eXtension registers
#ifndef NT_LOONGARCH_LASX
#define NT_LOONGARCH_LASX 0xa03
#endif

// LoongArch hardware breakpoint registers
#ifndef NT_LOONGARCH_HW_BREAK
#define NT_LOONGARCH_HW_BREAK 0xa05
#endif

// LoongArch hardware watchpoint registers
#ifndef NT_LOONGARCH_HW_WATCH
#define NT_LOONGARCH_HW_WATCH 0xa06
#endif

#define REG_CONTEXT_SIZE                                                       \
  (GetGPRSize() + GetFPRSize() + sizeof(m_lsx) + sizeof(m_lasx))

// ptrace has a struct type user_watch_state, which was replaced by
// user_watch_state_v2 when more watchpoints were added, so this file
// may be built on systems with one or both in the system headers.
// The type below has the same layout as user_watch_state_v2 but will
// not clash with that name if it exists. We can use the v2 layout even
// on old kernels as we will only see 8 watchpoints and the kernel will
// truncate any extra data we send to it.
struct loongarch_user_watch_state {
  uint64_t dbg_info;
  struct {
    uint64_t addr;
    uint64_t mask;
    uint32_t ctrl;
    uint32_t pad;
  } dbg_regs[14];
};


namespace reg_ids = lldb_private::registers::loongarch::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::loongarch64: {
    auto register_model = CreateLinuxLoongArch64RegisterLayout(target_arch);
    return std::make_unique<NativeRegisterContextLinux_loongarch64>(
        target_arch, native_thread, std::move(register_model));
  }
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return HostInfo::GetArchitecture();
}

NativeRegisterContextLinux_loongarch64::NativeRegisterContextLinux_loongarch64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    RegisterLayout register_model)
    : NativeRegisterContext(native_thread, target_arch,
                            std::move(register_model)),
      NativeRegisterContextLinux(native_thread) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr, 0, sizeof(m_gpr));
  ::memset(&m_lsx, 0, sizeof(m_lsx));
  ::memset(&m_lasx, 0, sizeof(m_lasx));

  // Refer to:
  // https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html#control-and-status-registers-related-to-watchpoints
  // 14 is just a maximum value, query hardware for actual watchpoint count.
  m_max_hwp_supported = 14;
  m_max_hbp_supported = 14;
  m_refresh_hwdebug_info = true;

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;
}

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_loongarch64::ReadRegister(RegisterID id) {
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
      status = ReadLSX();
      break;
    case 3:
      status = ReadLASX();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{0, {reinterpret_cast<const uint8_t *>(GetGPRBuffer()), GetGPRSize()}},
       {1, {reinterpret_cast<const uint8_t *>(GetFPRBuffer()), GetFPRSize()}},
       {2, {reinterpret_cast<const uint8_t *>(&m_lsx), sizeof(m_lsx)}},
       {3, {reinterpret_cast<const uint8_t *>(&m_lasx), sizeof(m_lasx)}}});
  return value;
}

llvm::Error NativeRegisterContextLinux_loongarch64::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  if (descriptor->numbers[lldb::eRegisterKindLLDB] == reg_ids::gpr_r0)
    return llvm::Error::success();
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
      status = ReadLSX();
      break;
    case 3:
      status = ReadLASX();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (auto error = WriteBankRegister(
          descriptor->id, reg_value,
          {{0, {reinterpret_cast<uint8_t *>(GetGPRBuffer()), GetGPRSize()}},
           {1, {reinterpret_cast<uint8_t *>(GetFPRBuffer()), GetFPRSize()}},
           {2, {reinterpret_cast<uint8_t *>(&m_lsx), sizeof(m_lsx)}},
           {3, {reinterpret_cast<uint8_t *>(&m_lasx), sizeof(m_lasx)}}}))
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
      status = WriteLSX();
      break;
    case 3:
      status = WriteLASX();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextLinux_loongarch64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));

  error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  error = ReadLSX();
  if (error.Fail())
    return error;

  error = ReadLASX();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();
  ::memcpy(dst, GetFPRBuffer(), GetFPRSize());
  dst += GetFPRSize();
  ::memcpy(dst, &m_lsx, sizeof(m_lsx));
  dst += sizeof(m_lsx);
  ::memcpy(dst, &m_lasx, sizeof(m_lasx));

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_loongarch64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_loongarch64::%s data_sp contained "
        "mismatched data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_loongarch64::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }
  ::memcpy(GetGPRBuffer(), src, GetBankSize(0));

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetBankSize(0);
  ::memcpy(GetFPRBuffer(), src, GetFPRSize());
  m_fpu_is_valid = true;
  error = WriteFPR();
  if (error.Fail())
    return error;

  // Currently, we assume that LoongArch always support LASX.
  // TODO: check whether LSX/LASX exists.
  src += GetFPRSize();
  ::memcpy(&m_lsx, src, sizeof(m_lsx));
  m_lsx_is_valid = true;
  error = WriteLSX();
  if (error.Fail())
    return error;

  src += sizeof(m_lsx);
  ::memcpy(&m_lasx, src, sizeof(m_lasx));
  m_lasx_is_valid = true;
  error = WriteLASX();
  if (error.Fail())
    return error;

  return error;
}

bool NativeRegisterContextLinux_loongarch64::IsGPR(unsigned reg) const {
  const auto *value =
      GetDescription()->FindByNumber(lldb::eRegisterKindLLDB, reg);
  return value && GetDescription()->IsInGroup(value->id, "gpr");
}

bool NativeRegisterContextLinux_loongarch64::IsFPR(unsigned reg) const {
  const auto *value =
      GetDescription()->FindByNumber(lldb::eRegisterKindLLDB, reg);
  return value && GetDescription()->IsInGroup(value->id, "fpr");
}

bool NativeRegisterContextLinux_loongarch64::IsLSX(unsigned reg) const {
  const auto *value =
      GetDescription()->FindByNumber(lldb::eRegisterKindLLDB, reg);
  return value && GetDescription()->IsInGroup(value->id, "lsx");
}

bool NativeRegisterContextLinux_loongarch64::IsLASX(unsigned reg) const {
  const auto *value =
      GetDescription()->FindByNumber(lldb::eRegisterKindLLDB, reg);
  return value && GetDescription()->IsInGroup(value->id, "lasx");
}

Status NativeRegisterContextLinux_loongarch64::ReadGPR() {
  Status error;

  if (m_gpr_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  error = ReadRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);

  if (error.Success())
    m_gpr_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteGPR() {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  m_gpr_is_valid = false;

  return WriteRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);
}

Status NativeRegisterContextLinux_loongarch64::ReadFPR() {
  Status error;

  if (m_fpu_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  error = ReadRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);

  if (error.Success())
    m_fpu_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteFPR() {
  Status error = ReadFPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);
}

Status NativeRegisterContextLinux_loongarch64::ReadLSX() {
  Status error;

  if (m_lsx_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lsx;
  ioVec.iov_len = sizeof(m_lsx);

  error = ReadRegisterSet(&ioVec, sizeof(m_lsx), NT_LOONGARCH_LSX);

  if (error.Success())
    m_lsx_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteLSX() {
  Status error = ReadLSX();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lsx;
  ioVec.iov_len = sizeof(m_lsx);

  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;

  return WriteRegisterSet(&ioVec, sizeof(m_lsx), NT_LOONGARCH_LSX);
}

Status NativeRegisterContextLinux_loongarch64::ReadLASX() {
  Status error;

  if (m_lasx_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lasx;
  ioVec.iov_len = sizeof(m_lasx);

  error = ReadRegisterSet(&ioVec, sizeof(m_lasx), NT_LOONGARCH_LASX);

  if (error.Success())
    m_lasx_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteLASX() {
  Status error = ReadLASX();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lasx;
  ioVec.iov_len = sizeof(m_lasx);

  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;

  return WriteRegisterSet(&ioVec, sizeof(m_lasx), NT_LOONGARCH_LASX);
}

void NativeRegisterContextLinux_loongarch64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;
}

llvm::Error NativeRegisterContextLinux_loongarch64::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info)
    return llvm::Error::success();

  ::pid_t tid = m_thread.GetID();

  int regset = NT_LOONGARCH_HW_WATCH;
  struct iovec ioVec;
  struct loongarch_user_watch_state dreg_state;
  Status error;

  ioVec.iov_base = &dreg_state;
  ioVec.iov_len = sizeof(dreg_state);
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);
  if (error.Fail())
    return error.ToError();

  m_max_hwp_supported = dreg_state.dbg_info & 0x3f;

  regset = NT_LOONGARCH_HW_BREAK;
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);
  if (error.Fail())
    return error.ToError();

  m_max_hbp_supported = dreg_state.dbg_info & 0x3f;

  m_refresh_hwdebug_info = false;

  return llvm::Error::success();
}

llvm::Error NativeRegisterContextLinux_loongarch64::WriteHardwareDebugRegs(
    DREGType hwbType) {
  struct iovec ioVec;
  struct loongarch_user_watch_state dreg_state;
  int regset;

  memset(&dreg_state, 0, sizeof(dreg_state));
  ioVec.iov_base = &dreg_state;

  switch (hwbType) {
  case eDREGTypeWATCH:
    regset = NT_LOONGARCH_HW_WATCH;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hwp_supported);

    for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hwp_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hwp_regs[i].control;
    }
    break;
  case eDREGTypeBREAK:
    regset = NT_LOONGARCH_HW_BREAK;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hbp_supported);

    for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hbp_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hbp_regs[i].control;
    }
    break;
  }

  return NativeProcessLinux::PtraceWrapper(PTRACE_SETREGSET, m_thread.GetID(),
                                           &regset, &ioVec, ioVec.iov_len)
      .ToError();
}
#endif // defined(__loongarch__) && __loongarch_grlen == 64
