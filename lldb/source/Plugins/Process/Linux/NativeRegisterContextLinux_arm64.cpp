//===-- NativeRegisterContextLinux_arm64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm64__) || defined(__aarch64__)

#include "NativeRegisterContextLinux_arm.h"
#include "NativeRegisterContextLinux_arm64.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/linux/Ptrace.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/MemoryTagManagerAArch64MTE.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules
#include <sys/uio.h>
// NT_PRSTATUS and NT_FPREGSET definition
#include <elf.h>
#include <optional>

#ifndef NT_ARM_SVE
#define NT_ARM_SVE 0x405 /* ARM Scalable Vector Extension */
#endif

#ifndef NT_ARM_PAC_MASK
#define NT_ARM_PAC_MASK 0x406 /* Pointer authentication code masks */
#endif

#ifndef NT_ARM_TAGGED_ADDR_CTRL
#define NT_ARM_TAGGED_ADDR_CTRL 0x409 /* Tagged address control register */
#endif

#define HWCAP_PACA (1 << 30)
#define HWCAP2_MTE (1 << 18)

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::arm:
    return std::make_unique<NativeRegisterContextLinux_arm>(target_arch,
                                                            native_thread);
  case llvm::Triple::aarch64: {
    // Configure register sets supported by this AArch64 target.
    // Read SVE header to check for SVE support.
    struct sve::user_sve_header sve_header;
    struct iovec ioVec;
    ioVec.iov_base = &sve_header;
    ioVec.iov_len = sizeof(sve_header);
    unsigned int regset = NT_ARM_SVE;

    Flags opt_regsets;
    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, sizeof(sve_header))
            .Success())
      opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskSVE);

    NativeProcessLinux &process = native_thread.GetProcess();

    std::optional<uint64_t> auxv_at_hwcap =
        process.GetAuxValue(AuxVector::AUXV_AT_HWCAP);
    if (auxv_at_hwcap && (*auxv_at_hwcap & HWCAP_PACA))
      opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskPAuth);

    std::optional<uint64_t> auxv_at_hwcap2 =
        process.GetAuxValue(AuxVector::AUXV_AT_HWCAP2);
    if (auxv_at_hwcap2 && (*auxv_at_hwcap2 & HWCAP2_MTE))
      opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskMTE);

    opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskTLS);

    auto register_info_up =
        std::make_unique<RegisterInfoPOSIX_arm64>(target_arch, opt_regsets);
    return std::make_unique<NativeRegisterContextLinux_arm64>(
        target_arch, native_thread, std::move(register_info_up));
  }
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return DetermineArchitectureViaGPR(
      tid, RegisterInfoPOSIX_arm64::GetGPRSizeStatic());
}

NativeRegisterContextLinux_arm64::NativeRegisterContextLinux_arm64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    std::unique_ptr<RegisterInfoPOSIX_arm64> register_info_up)
    : NativeRegisterContextRegisterInfo(native_thread,
                                        register_info_up.release()),
      NativeRegisterContextLinux(native_thread) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr_arm64, 0, sizeof(m_gpr_arm64));
  ::memset(&m_hwp_regs, 0, sizeof(m_hwp_regs));
  ::memset(&m_hbp_regs, 0, sizeof(m_hbp_regs));
  ::memset(&m_sve_header, 0, sizeof(m_sve_header));
  ::memset(&m_pac_mask, 0, sizeof(m_pac_mask));

  m_mte_ctrl_reg = 0;
  m_tls_tpidr_reg = 0;

  // 16 is just a maximum value, query hardware for actual watchpoint count
  m_max_hwp_supported = 16;
  m_max_hbp_supported = 16;

  m_refresh_hwdebug_info = true;

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_sve_buffer_is_valid = false;
  m_sve_header_is_valid = false;
  m_pac_mask_is_valid = false;
  m_mte_ctrl_is_valid = false;
  m_tls_tpidr_is_valid = false;

  if (GetRegisterInfo().IsSVEEnabled())
    m_sve_state = SVEState::Unknown;
  else
    m_sve_state = SVEState::Disabled;
}

RegisterInfoPOSIX_arm64 &
NativeRegisterContextLinux_arm64::GetRegisterInfo() const {
  return static_cast<RegisterInfoPOSIX_arm64 &>(*m_register_info_interface_up);
}

uint32_t NativeRegisterContextLinux_arm64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *
NativeRegisterContextLinux_arm64::GetRegisterSet(uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

uint32_t NativeRegisterContextLinux_arm64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

Status
NativeRegisterContextLinux_arm64::ReadRegister(const RegisterInfo *reg_info,
                                               RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status("no lldb regnum for %s", reg_info && reg_info->name
                                               ? reg_info->name
                                               : "<unknown register>");

  uint8_t *src;
  uint32_t offset = LLDB_INVALID_INDEX32;
  uint64_t sve_vg;
  std::vector<uint8_t> sve_reg_non_live;

  if (IsGPR(reg)) {
    error = ReadGPR();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset;
    assert(offset < GetGPRSize());
    src = (uint8_t *)GetGPRBuffer() + offset;

  } else if (IsFPR(reg)) {
    if (m_sve_state == SVEState::Disabled) {
      // SVE is disabled take legacy route for FPU register access
      error = ReadFPR();
      if (error.Fail())
        return error;

      offset = CalculateFprOffset(reg_info);
      assert(offset < GetFPRSize());
      src = (uint8_t *)GetFPRBuffer() + offset;
    } else {
      // SVE enabled, we will read and cache SVE ptrace data
      error = ReadAllSVE();
      if (error.Fail())
        return error;

      // FPSR and FPCR will be located right after Z registers in
      // SVEState::FPSIMD while in SVEState::Full they will be located at the
      // end of register data after an alignment correction based on currently
      // selected vector length.
      uint32_t sve_reg_num = LLDB_INVALID_REGNUM;
      if (reg == GetRegisterInfo().GetRegNumFPSR()) {
        sve_reg_num = reg;
        if (m_sve_state == SVEState::Full)
          offset = sve::PTraceFPSROffset(sve::vq_from_vl(m_sve_header.vl));
        else if (m_sve_state == SVEState::FPSIMD)
          offset = sve::ptrace_fpsimd_offset + (32 * 16);
      } else if (reg == GetRegisterInfo().GetRegNumFPCR()) {
        sve_reg_num = reg;
        if (m_sve_state == SVEState::Full)
          offset = sve::PTraceFPCROffset(sve::vq_from_vl(m_sve_header.vl));
        else if (m_sve_state == SVEState::FPSIMD)
          offset = sve::ptrace_fpsimd_offset + (32 * 16) + 4;
      } else {
        // Extract SVE Z register value register number for this reg_info
        if (reg_info->value_regs &&
            reg_info->value_regs[0] != LLDB_INVALID_REGNUM)
          sve_reg_num = reg_info->value_regs[0];
        offset = CalculateSVEOffset(GetRegisterInfoAtIndex(sve_reg_num));
      }

      assert(offset < GetSVEBufferSize());
      src = (uint8_t *)GetSVEBuffer() + offset;
    }
  } else if (IsTLS(reg)) {
    error = ReadTLSTPIDR();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset - GetRegisterInfo().GetTLSOffset();
    assert(offset < GetTLSTPIDRSize());
    src = (uint8_t *)GetTLSTPIDR() + offset;
  } else if (IsSVE(reg)) {
    if (m_sve_state == SVEState::Disabled || m_sve_state == SVEState::Unknown)
      return Status("SVE disabled or not supported");

    if (GetRegisterInfo().IsSVERegVG(reg)) {
      sve_vg = GetSVERegVG();
      src = (uint8_t *)&sve_vg;
    } else {
      // SVE enabled, we will read and cache SVE ptrace data
      error = ReadAllSVE();
      if (error.Fail())
        return error;

      if (m_sve_state == SVEState::FPSIMD) {
        // In FPSIMD state SVE payload mirrors legacy fpsimd struct and so
        // just copy 16 bytes of v register to the start of z register. All
        // other SVE register will be set to zero.
        sve_reg_non_live.resize(reg_info->byte_size, 0);
        src = sve_reg_non_live.data();

        if (GetRegisterInfo().IsSVEZReg(reg)) {
          offset = CalculateSVEOffset(reg_info);
          assert(offset < GetSVEBufferSize());
          ::memcpy(sve_reg_non_live.data(), (uint8_t *)GetSVEBuffer() + offset,
                   16);
        }
      } else {
        offset = CalculateSVEOffset(reg_info);
        assert(offset < GetSVEBufferSize());
        src = (uint8_t *)GetSVEBuffer() + offset;
      }
    }
  } else if (IsPAuth(reg)) {
    error = ReadPAuthMask();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset - GetRegisterInfo().GetPAuthOffset();
    assert(offset < GetPACMaskSize());
    src = (uint8_t *)GetPACMask() + offset;
  } else if (IsMTE(reg)) {
    error = ReadMTEControl();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset - GetRegisterInfo().GetMTEOffset();
    assert(offset < GetMTEControlSize());
    src = (uint8_t *)GetMTEControl() + offset;
  } else
    return Status("failed - register wasn't recognized to be a GPR or an FPR, "
                  "write strategy unknown");

  reg_value.SetFromMemoryData(*reg_info, src, reg_info->byte_size,
                              eByteOrderLittle, error);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Status error;

  if (!reg_info)
    return Status("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status("no lldb regnum for %s", reg_info && reg_info->name
                                               ? reg_info->name
                                               : "<unknown register>");

  uint8_t *dst;
  uint32_t offset = LLDB_INVALID_INDEX32;
  std::vector<uint8_t> sve_reg_non_live;

  if (IsGPR(reg)) {
    error = ReadGPR();
    if (error.Fail())
      return error;

    assert(reg_info->byte_offset < GetGPRSize());
    dst = (uint8_t *)GetGPRBuffer() + reg_info->byte_offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteGPR();
  } else if (IsFPR(reg)) {
    if (m_sve_state == SVEState::Disabled) {
      // SVE is disabled take legacy route for FPU register access
      error = ReadFPR();
      if (error.Fail())
        return error;

      offset = CalculateFprOffset(reg_info);
      assert(offset < GetFPRSize());
      dst = (uint8_t *)GetFPRBuffer() + offset;
      ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

      return WriteFPR();
    } else {
      // SVE enabled, we will read and cache SVE ptrace data
      error = ReadAllSVE();
      if (error.Fail())
        return error;

      // FPSR and FPCR will be located right after Z registers in
      // SVEState::FPSIMD while in SVEState::Full they will be located at the
      // end of register data after an alignment correction based on currently
      // selected vector length.
      uint32_t sve_reg_num = LLDB_INVALID_REGNUM;
      if (reg == GetRegisterInfo().GetRegNumFPSR()) {
        sve_reg_num = reg;
        if (m_sve_state == SVEState::Full)
          offset = sve::PTraceFPSROffset(sve::vq_from_vl(m_sve_header.vl));
        else if (m_sve_state == SVEState::FPSIMD)
          offset = sve::ptrace_fpsimd_offset + (32 * 16);
      } else if (reg == GetRegisterInfo().GetRegNumFPCR()) {
        sve_reg_num = reg;
        if (m_sve_state == SVEState::Full)
          offset = sve::PTraceFPCROffset(sve::vq_from_vl(m_sve_header.vl));
        else if (m_sve_state == SVEState::FPSIMD)
          offset = sve::ptrace_fpsimd_offset + (32 * 16) + 4;
      } else {
        // Extract SVE Z register value register number for this reg_info
        if (reg_info->value_regs &&
            reg_info->value_regs[0] != LLDB_INVALID_REGNUM)
          sve_reg_num = reg_info->value_regs[0];
        offset = CalculateSVEOffset(GetRegisterInfoAtIndex(sve_reg_num));
      }

      assert(offset < GetSVEBufferSize());
      dst = (uint8_t *)GetSVEBuffer() + offset;
      ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);
      return WriteAllSVE();
    }
  } else if (IsSVE(reg)) {
    if (m_sve_state == SVEState::Disabled || m_sve_state == SVEState::Unknown)
      return Status("SVE disabled or not supported");
    else {
      // Target has SVE enabled, we will read and cache SVE ptrace data
      error = ReadAllSVE();
      if (error.Fail())
        return error;

      if (GetRegisterInfo().IsSVERegVG(reg)) {
        uint64_t vg_value = reg_value.GetAsUInt64();

        if (sve::vl_valid(vg_value * 8)) {
          if (m_sve_header_is_valid && vg_value == GetSVERegVG())
            return error;

          SetSVERegVG(vg_value);

          error = WriteSVEHeader();
          if (error.Success())
            ConfigureRegisterContext();

          if (m_sve_header_is_valid && vg_value == GetSVERegVG())
            return error;
        }

        return Status("SVE vector length update failed.");
      }

      // If target supports SVE but currently in FPSIMD mode.
      if (m_sve_state == SVEState::FPSIMD) {
        // Here we will check if writing this SVE register enables
        // SVEState::Full
        bool set_sve_state_full = false;
        const uint8_t *reg_bytes = (const uint8_t *)reg_value.GetBytes();
        if (GetRegisterInfo().IsSVEZReg(reg)) {
          for (uint32_t i = 16; i < reg_info->byte_size; i++) {
            if (reg_bytes[i]) {
              set_sve_state_full = true;
              break;
            }
          }
        } else if (GetRegisterInfo().IsSVEPReg(reg) ||
                   reg == GetRegisterInfo().GetRegNumSVEFFR()) {
          for (uint32_t i = 0; i < reg_info->byte_size; i++) {
            if (reg_bytes[i]) {
              set_sve_state_full = true;
              break;
            }
          }
        }

        if (!set_sve_state_full && GetRegisterInfo().IsSVEZReg(reg)) {
          // We are writing a Z register which is zero beyond 16 bytes so copy
          // first 16 bytes only as SVE payload mirrors legacy fpsimd structure
          offset = CalculateSVEOffset(reg_info);
          assert(offset < GetSVEBufferSize());
          dst = (uint8_t *)GetSVEBuffer() + offset;
          ::memcpy(dst, reg_value.GetBytes(), 16);

          return WriteAllSVE();
        } else
          return Status("SVE state change operation not supported");
      } else {
        offset = CalculateSVEOffset(reg_info);
        assert(offset < GetSVEBufferSize());
        dst = (uint8_t *)GetSVEBuffer() + offset;
        ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);
        return WriteAllSVE();
      }
    }
  } else if (IsMTE(reg)) {
    error = ReadMTEControl();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset - GetRegisterInfo().GetMTEOffset();
    assert(offset < GetMTEControlSize());
    dst = (uint8_t *)GetMTEControl() + offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteMTEControl();
  } else if (IsTLS(reg)) {
    error = ReadTLSTPIDR();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset - GetRegisterInfo().GetTLSOffset();
    assert(offset < GetTLSTPIDRSize());
    dst = (uint8_t *)GetTLSTPIDR() + offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteTLSTPIDR();
  }

  return Status("Failed to write register value");
}

Status NativeRegisterContextLinux_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  // AArch64 register data must contain GPRs, either FPR or SVE registers
  // and optional MTE register. Pointer Authentication (PAC) registers are
  // read-only and will be skiped.

  // In order to create register data checkpoint we first read all register
  // values if not done already and calculate total size of register set data.
  // We store all register values in data_sp by copying full PTrace data that
  // corresponds to register sets enabled by current register context.

  Status error;
  uint32_t reg_data_byte_size = GetGPRBufferSize();
  error = ReadGPR();
  if (error.Fail())
    return error;

  // If SVE is enabled we need not copy FPR separately.
  if (GetRegisterInfo().IsSVEEnabled()) {
    reg_data_byte_size += GetSVEBufferSize();
    error = ReadAllSVE();
  } else {
    reg_data_byte_size += GetFPRSize();
    error = ReadFPR();
  }
  if (error.Fail())
    return error;

  if (GetRegisterInfo().IsMTEEnabled()) {
    reg_data_byte_size += GetMTEControlSize();
    error = ReadMTEControl();
    if (error.Fail())
      return error;
  }

  // tpidr is always present but there will be more in future.
  reg_data_byte_size += GetTLSTPIDRSize();
  error = ReadTLSTPIDR();
  if (error.Fail())
    return error;

  data_sp.reset(new DataBufferHeap(reg_data_byte_size, 0));
  uint8_t *dst = data_sp->GetBytes();

  ::memcpy(dst, GetGPRBuffer(), GetGPRBufferSize());
  dst += GetGPRBufferSize();

  if (GetRegisterInfo().IsSVEEnabled()) {
    ::memcpy(dst, GetSVEBuffer(), GetSVEBufferSize());
    dst += GetSVEBufferSize();
  } else {
    ::memcpy(dst, GetFPRBuffer(), GetFPRSize());
    dst += GetFPRSize();
  }

  if (GetRegisterInfo().IsMTEEnabled()) {
    ::memcpy(dst, GetMTEControl(), GetMTEControlSize());
    dst += GetMTEControlSize();
  }

  ::memcpy(dst, GetTLSTPIDR(), GetTLSTPIDRSize());

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  // AArch64 register data must contain GPRs, either FPR or SVE registers
  // and optional MTE register. Pointer Authentication (PAC) registers are
  // read-only and will be skiped.

  // We store all register values in data_sp by copying full PTrace data that
  // corresponds to register sets enabled by current register context. In order
  // to restore from register data checkpoint we will first restore GPRs, based
  // on size of remaining register data either SVE or FPRs should be restored
  // next. SVE is not enabled if we have register data size less than or equal
  // to size of GPR + FPR + MTE.

  Status error;
  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_arm64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error.SetErrorStringWithFormat("NativeRegisterContextLinux_arm64::%s "
                                   "DataBuffer::GetBytes() returned a null "
                                   "pointer",
                                   __FUNCTION__);
    return error;
  }

  uint64_t reg_data_min_size = GetGPRBufferSize() + GetFPRSize();
  if (data_sp->GetByteSize() < reg_data_min_size) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_arm64::%s data_sp contained insufficient "
        "register data bytes, expected at least %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, reg_data_min_size, data_sp->GetByteSize());
    return error;
  }

  // Register data starts with GPRs
  ::memcpy(GetGPRBuffer(), src, GetGPRBufferSize());
  m_gpr_is_valid = true;

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetGPRBufferSize();

  // Verify if register data may contain SVE register values.
  bool contains_sve_reg_data =
      (data_sp->GetByteSize() > (reg_data_min_size + GetSVEHeaderSize()));

  if (contains_sve_reg_data) {
    // We have SVE register data first write SVE header.
    ::memcpy(GetSVEHeader(), src, GetSVEHeaderSize());
    if (!sve::vl_valid(m_sve_header.vl)) {
      m_sve_header_is_valid = false;
      error.SetErrorStringWithFormat("NativeRegisterContextLinux_arm64::%s "
                                     "Invalid SVE header in data_sp",
                                     __FUNCTION__);
      return error;
    }
    m_sve_header_is_valid = true;
    error = WriteSVEHeader();
    if (error.Fail())
      return error;

    // SVE header has been written configure SVE vector length if needed.
    ConfigureRegisterContext();

    // Make sure data_sp contains sufficient data to write all SVE registers.
    reg_data_min_size = GetGPRBufferSize() + GetSVEBufferSize();
    if (data_sp->GetByteSize() < reg_data_min_size) {
      error.SetErrorStringWithFormat(
          "NativeRegisterContextLinux_arm64::%s data_sp contained insufficient "
          "register data bytes, expected %" PRIu64 ", actual %" PRIu64,
          __FUNCTION__, reg_data_min_size, data_sp->GetByteSize());
      return error;
    }

    ::memcpy(GetSVEBuffer(), src, GetSVEBufferSize());
    m_sve_buffer_is_valid = true;
    error = WriteAllSVE();
    src += GetSVEBufferSize();
  } else {
    ::memcpy(GetFPRBuffer(), src, GetFPRSize());
    m_fpu_is_valid = true;
    error = WriteFPR();
    src += GetFPRSize();
  }

  if (error.Fail())
    return error;

  if (GetRegisterInfo().IsMTEEnabled() &&
      data_sp->GetByteSize() > reg_data_min_size) {
    ::memcpy(GetMTEControl(), src, GetMTEControlSize());
    m_mte_ctrl_is_valid = true;
    error = WriteMTEControl();
    if (error.Fail())
      return error;
    src += GetMTEControlSize();
  }

  // There is always a TLS set. It changes size based on system properties, it's
  // not something an expression can change.
  ::memcpy(GetTLSTPIDR(), src, GetTLSTPIDRSize());
  m_tls_tpidr_is_valid = true;
  error = WriteTLSTPIDR();

  return error;
}

bool NativeRegisterContextLinux_arm64::IsGPR(unsigned reg) const {
  if (GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
      RegisterInfoPOSIX_arm64::GPRegSet)
    return true;
  return false;
}

bool NativeRegisterContextLinux_arm64::IsFPR(unsigned reg) const {
  if (GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
      RegisterInfoPOSIX_arm64::FPRegSet)
    return true;
  return false;
}

bool NativeRegisterContextLinux_arm64::IsSVE(unsigned reg) const {
  return GetRegisterInfo().IsSVEReg(reg);
}

bool NativeRegisterContextLinux_arm64::IsPAuth(unsigned reg) const {
  return GetRegisterInfo().IsPAuthReg(reg);
}

bool NativeRegisterContextLinux_arm64::IsMTE(unsigned reg) const {
  return GetRegisterInfo().IsMTEReg(reg);
}

bool NativeRegisterContextLinux_arm64::IsTLS(unsigned reg) const {
  return GetRegisterInfo().IsTLSReg(reg);
}

llvm::Error NativeRegisterContextLinux_arm64::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info) {
    return llvm::Error::success();
  }

  ::pid_t tid = m_thread.GetID();

  int regset = NT_ARM_HW_WATCH;
  struct iovec ioVec;
  struct user_hwdebug_state dreg_state;
  Status error;

  ioVec.iov_base = &dreg_state;
  ioVec.iov_len = sizeof(dreg_state);
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);

  if (error.Fail())
    return error.ToError();

  m_max_hwp_supported = dreg_state.dbg_info & 0xff;

  regset = NT_ARM_HW_BREAK;
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);

  if (error.Fail())
    return error.ToError();

  m_max_hbp_supported = dreg_state.dbg_info & 0xff;
  m_refresh_hwdebug_info = false;

  return llvm::Error::success();
}

llvm::Error
NativeRegisterContextLinux_arm64::WriteHardwareDebugRegs(DREGType hwbType) {
  struct iovec ioVec;
  struct user_hwdebug_state dreg_state;
  int regset;

  memset(&dreg_state, 0, sizeof(dreg_state));
  ioVec.iov_base = &dreg_state;

  switch (hwbType) {
  case eDREGTypeWATCH:
    regset = NT_ARM_HW_WATCH;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) + sizeof(dreg_state.pad) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hwp_supported);

    for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hwp_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hwp_regs[i].control;
    }
    break;
  case eDREGTypeBREAK:
    regset = NT_ARM_HW_BREAK;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) + sizeof(dreg_state.pad) +
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

Status NativeRegisterContextLinux_arm64::ReadGPR() {
  Status error;

  if (m_gpr_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRBufferSize();

  error = ReadRegisterSet(&ioVec, GetGPRBufferSize(), NT_PRSTATUS);

  if (error.Success())
    m_gpr_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteGPR() {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRBufferSize();

  m_gpr_is_valid = false;

  return WriteRegisterSet(&ioVec, GetGPRBufferSize(), NT_PRSTATUS);
}

Status NativeRegisterContextLinux_arm64::ReadFPR() {
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

Status NativeRegisterContextLinux_arm64::WriteFPR() {
  Status error = ReadFPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  m_fpu_is_valid = false;

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);
}

void NativeRegisterContextLinux_arm64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_sve_buffer_is_valid = false;
  m_sve_header_is_valid = false;
  m_pac_mask_is_valid = false;
  m_mte_ctrl_is_valid = false;
  m_tls_tpidr_is_valid = false;

  // Update SVE registers in case there is change in configuration.
  ConfigureRegisterContext();
}

Status NativeRegisterContextLinux_arm64::ReadSVEHeader() {
  Status error;

  if (m_sve_header_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSVEHeader();
  ioVec.iov_len = GetSVEHeaderSize();

  error = ReadRegisterSet(&ioVec, GetSVEHeaderSize(), NT_ARM_SVE);

  if (error.Success())
    m_sve_header_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::ReadPAuthMask() {
  Status error;

  if (m_pac_mask_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetPACMask();
  ioVec.iov_len = GetPACMaskSize();

  error = ReadRegisterSet(&ioVec, GetPACMaskSize(), NT_ARM_PAC_MASK);

  if (error.Success())
    m_pac_mask_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteSVEHeader() {
  Status error;

  error = ReadSVEHeader();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSVEHeader();
  ioVec.iov_len = GetSVEHeaderSize();

  m_sve_buffer_is_valid = false;
  m_sve_header_is_valid = false;
  m_fpu_is_valid = false;

  return WriteRegisterSet(&ioVec, GetSVEHeaderSize(), NT_ARM_SVE);
}

Status NativeRegisterContextLinux_arm64::ReadAllSVE() {
  Status error;

  if (m_sve_buffer_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSVEBuffer();
  ioVec.iov_len = GetSVEBufferSize();

  error = ReadRegisterSet(&ioVec, GetSVEBufferSize(), NT_ARM_SVE);

  if (error.Success())
    m_sve_buffer_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteAllSVE() {
  Status error;

  error = ReadAllSVE();
  if (error.Fail())
    return error;

  struct iovec ioVec;

  ioVec.iov_base = GetSVEBuffer();
  ioVec.iov_len = GetSVEBufferSize();

  m_sve_buffer_is_valid = false;
  m_sve_header_is_valid = false;
  m_fpu_is_valid = false;

  return WriteRegisterSet(&ioVec, GetSVEBufferSize(), NT_ARM_SVE);
}

Status NativeRegisterContextLinux_arm64::ReadMTEControl() {
  Status error;

  if (m_mte_ctrl_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetMTEControl();
  ioVec.iov_len = GetMTEControlSize();

  error = ReadRegisterSet(&ioVec, GetMTEControlSize(), NT_ARM_TAGGED_ADDR_CTRL);

  if (error.Success())
    m_mte_ctrl_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteMTEControl() {
  Status error;

  error = ReadMTEControl();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetMTEControl();
  ioVec.iov_len = GetMTEControlSize();

  m_mte_ctrl_is_valid = false;

  return WriteRegisterSet(&ioVec, GetMTEControlSize(), NT_ARM_TAGGED_ADDR_CTRL);
}

Status NativeRegisterContextLinux_arm64::ReadTLSTPIDR() {
  Status error;

  if (m_tls_tpidr_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetTLSTPIDR();
  ioVec.iov_len = GetTLSTPIDRSize();

  error = ReadRegisterSet(&ioVec, GetTLSTPIDRSize(), NT_ARM_TLS);

  if (error.Success())
    m_tls_tpidr_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteTLSTPIDR() {
  Status error;

  error = ReadTLSTPIDR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetTLSTPIDR();
  ioVec.iov_len = GetTLSTPIDRSize();

  m_tls_tpidr_is_valid = false;

  return WriteRegisterSet(&ioVec, GetTLSTPIDRSize(), NT_ARM_TLS);
}

void NativeRegisterContextLinux_arm64::ConfigureRegisterContext() {
  // ConfigureRegisterContext gets called from InvalidateAllRegisters
  // on every stop and configures SVE vector length.
  // If m_sve_state is set to SVEState::Disabled on first stop, code below will
  // be deemed non operational for the lifetime of current process.
  if (!m_sve_header_is_valid && m_sve_state != SVEState::Disabled) {
    Status error = ReadSVEHeader();
    if (error.Success()) {
      // If SVE is enabled thread can switch between SVEState::FPSIMD and
      // SVEState::Full on every stop.
      if ((m_sve_header.flags & sve::ptrace_regs_mask) ==
          sve::ptrace_regs_fpsimd)
        m_sve_state = SVEState::FPSIMD;
      else if ((m_sve_header.flags & sve::ptrace_regs_mask) ==
               sve::ptrace_regs_sve)
        m_sve_state = SVEState::Full;

      // On every stop we configure SVE vector length by calling
      // ConfigureVectorLength regardless of current SVEState of this thread.
      uint32_t vq = RegisterInfoPOSIX_arm64::eVectorQuadwordAArch64SVE;
      if (sve::vl_valid(m_sve_header.vl))
        vq = sve::vq_from_vl(m_sve_header.vl);

      GetRegisterInfo().ConfigureVectorLength(vq);
      m_sve_ptrace_payload.resize(sve::PTraceSize(vq, sve::ptrace_regs_sve));
    }
  }
}

uint32_t NativeRegisterContextLinux_arm64::CalculateFprOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize();
}

uint32_t NativeRegisterContextLinux_arm64::CalculateSVEOffset(
    const RegisterInfo *reg_info) const {
  // Start of Z0 data is after GPRs plus 8 bytes of vg register
  uint32_t sve_reg_offset = LLDB_INVALID_INDEX32;
  if (m_sve_state == SVEState::FPSIMD) {
    const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
    sve_reg_offset = sve::ptrace_fpsimd_offset +
                     (reg - GetRegisterInfo().GetRegNumSVEZ0()) * 16;
  } else if (m_sve_state == SVEState::Full) {
    uint32_t sve_z0_offset = GetGPRSize() + 16;
    sve_reg_offset =
        sve::SigRegsOffset() + reg_info->byte_offset - sve_z0_offset;
  }
  return sve_reg_offset;
}

std::vector<uint32_t> NativeRegisterContextLinux_arm64::GetExpeditedRegisters(
    ExpeditedRegs expType) const {
  std::vector<uint32_t> expedited_reg_nums =
      NativeRegisterContext::GetExpeditedRegisters(expType);
  if (m_sve_state == SVEState::FPSIMD || m_sve_state == SVEState::Full)
    expedited_reg_nums.push_back(GetRegisterInfo().GetRegNumSVEVG());

  return expedited_reg_nums;
}

llvm::Expected<NativeRegisterContextLinux::MemoryTaggingDetails>
NativeRegisterContextLinux_arm64::GetMemoryTaggingDetails(int32_t type) {
  if (type == MemoryTagManagerAArch64MTE::eMTE_allocation) {
    return MemoryTaggingDetails{std::make_unique<MemoryTagManagerAArch64MTE>(),
                                PTRACE_PEEKMTETAGS, PTRACE_POKEMTETAGS};
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Unknown AArch64 memory tag type %d", type);
}

lldb::addr_t NativeRegisterContextLinux_arm64::FixWatchpointHitAddress(
    lldb::addr_t hit_addr) {
  // Linux configures user-space virtual addresses with top byte ignored.
  // We set default value of mask such that top byte is masked out.
  lldb::addr_t mask = ~((1ULL << 56) - 1);

  // Try to read pointer authentication data_mask register and calculate a
  // consolidated data address mask after ignoring the top byte.
  if (ReadPAuthMask().Success())
    mask |= m_pac_mask.data_mask;

  return hit_addr & ~mask;
  ;
}

#endif // defined (__arm64__) || defined (__aarch64__)
