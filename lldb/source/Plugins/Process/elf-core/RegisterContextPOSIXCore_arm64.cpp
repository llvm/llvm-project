//===-- RegisterContextPOSIXCore_arm64.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_arm64.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"

#include <memory>

using namespace lldb_private;

std::unique_ptr<RegisterContextCorePOSIX_arm64>
RegisterContextCorePOSIX_arm64::Create(Thread &thread, const ArchSpec &arch,
                                       const DataExtractor &gpregset,
                                       llvm::ArrayRef<CoreNote> notes) {
  Flags opt_regsets = RegisterInfoPOSIX_arm64::eRegsetMaskDefault;

  DataExtractor sve_data = getRegset(notes, arch.GetTriple(), AARCH64_SVE_Desc);
  if (sve_data.GetByteSize() > sizeof(sve::user_sve_header))
    opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskSVE);

  // Pointer Authentication register set data is based on struct
  // user_pac_mask declared in ptrace.h. See reference implementation
  // in Linux kernel source at arch/arm64/include/uapi/asm/ptrace.h.
  DataExtractor pac_data = getRegset(notes, arch.GetTriple(), AARCH64_PAC_Desc);
  if (pac_data.GetByteSize() >= sizeof(uint64_t) * 2)
    opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskPAuth);

  DataExtractor tls_data = getRegset(notes, arch.GetTriple(), AARCH64_TLS_Desc);
  // A valid note will always contain at least one register, "tpidr". It may
  // expand in future.
  if (tls_data.GetByteSize() >= sizeof(uint64_t))
    opt_regsets.Set(RegisterInfoPOSIX_arm64::eRegsetMaskTLS);

  auto register_info_up =
      std::make_unique<RegisterInfoPOSIX_arm64>(arch, opt_regsets);
  return std::unique_ptr<RegisterContextCorePOSIX_arm64>(
      new RegisterContextCorePOSIX_arm64(thread, std::move(register_info_up),
                                         gpregset, notes));
}

RegisterContextCorePOSIX_arm64::RegisterContextCorePOSIX_arm64(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_arm64> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_arm64(thread, std::move(register_info)) {
  m_gpr_data.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                      gpregset.GetByteSize()));
  m_gpr_data.SetByteOrder(gpregset.GetByteOrder());

  const llvm::Triple &target_triple =
      m_register_info_up->GetTargetArchitecture().GetTriple();
  m_fpr_data = getRegset(notes, target_triple, FPR_Desc);

  if (m_register_info_up->IsSVEEnabled())
    m_sve_data = getRegset(notes, target_triple, AARCH64_SVE_Desc);

  if (m_register_info_up->IsPAuthEnabled())
    m_pac_data = getRegset(notes, target_triple, AARCH64_PAC_Desc);

  if (m_register_info_up->IsTLSEnabled())
    m_tls_data = getRegset(notes, target_triple, AARCH64_TLS_Desc);

  ConfigureRegisterContext();
}

RegisterContextCorePOSIX_arm64::~RegisterContextCorePOSIX_arm64() = default;

bool RegisterContextCorePOSIX_arm64::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_arm64::ReadFPR() { return false; }

bool RegisterContextCorePOSIX_arm64::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_arm64::WriteFPR() {
  assert(0);
  return false;
}

const uint8_t *RegisterContextCorePOSIX_arm64::GetSVEBuffer(uint64_t offset) {
  return m_sve_data.GetDataStart() + offset;
}

void RegisterContextCorePOSIX_arm64::ConfigureRegisterContext() {
  if (m_sve_data.GetByteSize() > sizeof(sve::user_sve_header)) {
    uint64_t sve_header_field_offset = 8;
    m_sve_vector_length = m_sve_data.GetU16(&sve_header_field_offset);
    sve_header_field_offset = 12;
    uint16_t sve_header_flags_field =
        m_sve_data.GetU16(&sve_header_field_offset);
    if ((sve_header_flags_field & sve::ptrace_regs_mask) ==
        sve::ptrace_regs_fpsimd)
      m_sve_state = SVEState::FPSIMD;
    else if ((sve_header_flags_field & sve::ptrace_regs_mask) ==
             sve::ptrace_regs_sve)
      m_sve_state = SVEState::Full;

    if (!sve::vl_valid(m_sve_vector_length)) {
      m_sve_state = SVEState::Disabled;
      m_sve_vector_length = 0;
    }
  } else
    m_sve_state = SVEState::Disabled;

  if (m_sve_state != SVEState::Disabled)
    m_register_info_up->ConfigureVectorLengthSVE(
        sve::vq_from_vl(m_sve_vector_length));
}

uint32_t RegisterContextCorePOSIX_arm64::CalculateSVEOffset(
    const RegisterInfo *reg_info) {
  // Start of Z0 data is after GPRs plus 8 bytes of vg register
  uint32_t sve_reg_offset = LLDB_INVALID_INDEX32;
  if (m_sve_state == SVEState::FPSIMD) {
    const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
    sve_reg_offset = sve::ptrace_fpsimd_offset + (reg - GetRegNumSVEZ0()) * 16;
  } else if (m_sve_state == SVEState::Full) {
    uint32_t sve_z0_offset = GetGPRSize() + 16;
    sve_reg_offset =
        sve::SigRegsOffset() + reg_info->byte_offset - sve_z0_offset;
  }

  return sve_reg_offset;
}

bool RegisterContextCorePOSIX_arm64::ReadRegister(const RegisterInfo *reg_info,
                                                  RegisterValue &value) {
  Status error;
  lldb::offset_t offset;

  offset = reg_info->byte_offset;
  if (offset + reg_info->byte_size <= GetGPRSize()) {
    uint64_t v = m_gpr_data.GetMaxU64(&offset, reg_info->byte_size);
    if (offset == reg_info->byte_offset + reg_info->byte_size) {
      value = v;
      return true;
    }
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM)
    return false;

  if (IsFPR(reg)) {
    if (m_sve_state == SVEState::Disabled) {
      // SVE is disabled take legacy route for FPU register access
      offset -= GetGPRSize();
      if (offset < m_fpr_data.GetByteSize()) {
        value.SetFromMemoryData(*reg_info, m_fpr_data.GetDataStart() + offset,
                                reg_info->byte_size, lldb::eByteOrderLittle,
                                error);
        return error.Success();
      }
    } else {
      // FPSR and FPCR will be located right after Z registers in
      // SVEState::FPSIMD while in SVEState::Full they will be located at the
      // end of register data after an alignment correction based on currently
      // selected vector length.
      uint32_t sve_reg_num = LLDB_INVALID_REGNUM;
      if (reg == GetRegNumFPSR()) {
        sve_reg_num = reg;
        if (m_sve_state == SVEState::Full)
          offset = sve::PTraceFPSROffset(sve::vq_from_vl(m_sve_vector_length));
        else if (m_sve_state == SVEState::FPSIMD)
          offset = sve::ptrace_fpsimd_offset + (32 * 16);
      } else if (reg == GetRegNumFPCR()) {
        sve_reg_num = reg;
        if (m_sve_state == SVEState::Full)
          offset = sve::PTraceFPCROffset(sve::vq_from_vl(m_sve_vector_length));
        else if (m_sve_state == SVEState::FPSIMD)
          offset = sve::ptrace_fpsimd_offset + (32 * 16) + 4;
      } else {
        // Extract SVE Z register value register number for this reg_info
        if (reg_info->value_regs &&
            reg_info->value_regs[0] != LLDB_INVALID_REGNUM)
          sve_reg_num = reg_info->value_regs[0];
        offset = CalculateSVEOffset(GetRegisterInfoAtIndex(sve_reg_num));
      }

      assert(sve_reg_num != LLDB_INVALID_REGNUM);
      assert(offset < m_sve_data.GetByteSize());
      value.SetFromMemoryData(*reg_info, GetSVEBuffer(offset),
                              reg_info->byte_size, lldb::eByteOrderLittle,
                              error);
    }
  } else if (IsSVE(reg)) {
    if (IsSVEVG(reg)) {
      value = GetSVERegVG();
      return true;
    }

    switch (m_sve_state) {
    case SVEState::FPSIMD: {
      // In FPSIMD state SVE payload mirrors legacy fpsimd struct and so just
      // copy 16 bytes of v register to the start of z register. All other
      // SVE register will be set to zero.
      uint64_t byte_size = 1;
      uint8_t zeros = 0;
      const uint8_t *src = &zeros;
      if (IsSVEZ(reg)) {
        byte_size = 16;
        offset = CalculateSVEOffset(reg_info);
        assert(offset < m_sve_data.GetByteSize());
        src = GetSVEBuffer(offset);
      }
      value.SetFromMemoryData(*reg_info, src, byte_size, lldb::eByteOrderLittle,
                              error);
    } break;
    case SVEState::Full:
      offset = CalculateSVEOffset(reg_info);
      assert(offset < m_sve_data.GetByteSize());
      value.SetFromMemoryData(*reg_info, GetSVEBuffer(offset),
                              reg_info->byte_size, lldb::eByteOrderLittle,
                              error);
      break;
    case SVEState::Disabled:
    default:
      return false;
    }
  } else if (IsPAuth(reg)) {
    offset = reg_info->byte_offset - m_register_info_up->GetPAuthOffset();
    assert(offset < m_pac_data.GetByteSize());
    value.SetFromMemoryData(*reg_info, m_pac_data.GetDataStart() + offset,
                            reg_info->byte_size, lldb::eByteOrderLittle, error);
  } else if (IsTLS(reg)) {
    offset = reg_info->byte_offset - m_register_info_up->GetTLSOffset();
    assert(offset < m_tls_data.GetByteSize());
    value.SetFromMemoryData(*reg_info, m_tls_data.GetDataStart() + offset,
                            reg_info->byte_size, lldb::eByteOrderLittle, error);
  } else
    return false;

  return error.Success();
}

bool RegisterContextCorePOSIX_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_arm64::WriteRegister(const RegisterInfo *reg_info,
                                                   const RegisterValue &value) {
  return false;
}

bool RegisterContextCorePOSIX_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_arm64::HardwareSingleStep(bool enable) {
  return false;
}
