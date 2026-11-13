//===-- NativeRegisterContextLinux_arm64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm64__) || defined(__aarch64__)

#include "NativeRegisterContextLinux_arm64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/NativeRegisterContextLinux_arm.h"
#include "Plugins/Process/Linux/NativeRegisterContextLinux_arm64dbreg.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/AuxVector.h"
#include "Plugins/Process/Utility/MemoryTagManagerAArch64MTE.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "llvm/BinaryFormat/ELF.h"

#include <algorithm>
#include <optional>

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules.
#include <sys/ptrace.h>
#include <sys/uio.h>

#ifndef HWCAP_PACA
#define HWCAP_PACA (1 << 30)
#endif

#ifndef HWCAP_GCS
#define HWCAP_GCS (1UL << 32)
#endif

#ifndef HWCAP2_MTE
#define HWCAP2_MTE (1 << 18)
#endif

#ifndef HWCAP2_FPMR
#define HWCAP2_FPMR (1UL << 48)
#endif

#ifndef HWCAP2_POE
#define HWCAP2_POE (1ULL << 63)
#endif

#ifndef PTRACE_GETREGSET
#define PTRACE_GETREGSET 0x4204
#endif

#ifndef PTRACE_PEEKMTETAGS
#define PTRACE_PEEKMTETAGS 33
#endif

#ifndef PTRACE_POKEMTETAGS
#define PTRACE_POKEMTETAGS 34
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

NativeRegisterContextLinux_arm64::RegisterSetType
NativeRegisterContextLinux_arm64::GetInvalidationMask(
    const RegisterSetType set) const {
  switch (set) {
  case RegisterSetType::FPMR:
  case RegisterSetType::GPR:
  case RegisterSetType::GCS:
  case RegisterSetType::MTE:
  case RegisterSetType::PAC:
  case RegisterSetType::POE:
  case RegisterSetType::TLS:
    return set;
  case RegisterSetType::SVE_HEADER:
  case RegisterSetType::SVE:
  case RegisterSetType::FPR:
    return RegisterSetType::SVE_HEADER | RegisterSetType::SVE |
           // SVE registers overlap FP registers in hardware.
           RegisterSetType::FPR;
  case RegisterSetType::ZA_HEADER:
  case RegisterSetType::ZA:
  case RegisterSetType::ZT:
    // In the Linux ptrace ABI, writes that enable ZA or ZT result in
    // both ZA and ZT being enabled.
    return RegisterSetType::ZA_HEADER | RegisterSetType::ZA |
           RegisterSetType::ZT;
  }
}

unsigned int NativeRegisterContextLinux_arm64::GetPtraceSet(
    NativeRegisterContextLinux_arm64::RegisterSetType set) const {
  switch (set) {
  case RegisterSetType::GPR:
    return llvm::ELF::NT_PRSTATUS;
  case RegisterSetType::FPR:
    return llvm::ELF::NT_FPREGSET;
  case RegisterSetType::SVE:
  case RegisterSetType::SVE_HEADER:
    switch (m_sve_state) {
    case SVEState::Streaming:
    case SVEState::StreamingFPSIMD:
      return llvm::ELF::NT_ARM_SSVE;
    default:
      return llvm::ELF::NT_ARM_SVE;
    }
  case RegisterSetType::PAC:
    return llvm::ELF::NT_ARM_PAC_MASK;
  case RegisterSetType::MTE:
    return llvm::ELF::NT_ARM_TAGGED_ADDR_CTRL;
  case RegisterSetType::TLS:
    return llvm::ELF::NT_ARM_TLS;
  case RegisterSetType::ZA:
  case RegisterSetType::ZA_HEADER:
    return llvm::ELF::NT_ARM_ZA;
  case RegisterSetType::ZT:
    return llvm::ELF::NT_ARM_ZT;
  case RegisterSetType::FPMR:
    return llvm::ELF::NT_ARM_FPMR;
  case RegisterSetType::GCS:
    return llvm::ELF::NT_ARM_GCS;
  case RegisterSetType::POE:
    return llvm::ELF::NT_ARM_POE;
  }

  llvm_unreachable("No ptrace set for this RegisterType.");
}

size_t NativeRegisterContextLinux_arm64::GetSetSize(
    NativeRegisterContextLinux_arm64::RegisterSetType set) const {
  switch (set) {
  case RegisterSetType::GPR:
    // Returns sizeof arm64 GPR ptrace buffer, which is different
    // from GetGPRSize which returns sizeof linux_arm64::GPR.
    return sizeof(m_gpr_arm64);
  case RegisterSetType::FPR:
    return sizeof(m_fpr);
  case RegisterSetType::SVE:
    return m_sve_ptrace_payload.size();
  case RegisterSetType::SVE_HEADER:
    return sizeof(m_sve_header);
  case RegisterSetType::PAC:
    return sizeof(m_pac_mask);
  case RegisterSetType::MTE:
    return sizeof(m_mte_ctrl_reg);
  case RegisterSetType::TLS:
    return m_tls_size;
  case RegisterSetType::ZA:
    return m_za_ptrace_payload.size();
  case RegisterSetType::ZA_HEADER:
    return sizeof(m_za_header);
  case RegisterSetType::ZT:
    return m_zt_reg.size();
  case RegisterSetType::FPMR:
    return sizeof(m_fpmr_reg);
  case RegisterSetType::GCS:
    return sizeof(m_gcs_regs);
  case RegisterSetType::POE:
    return sizeof(m_poe_regs);
  }

  llvm_unreachable("No set size for this RegisterType.");
}

void *NativeRegisterContextLinux_arm64::GetSetBuffer(
    NativeRegisterContextLinux_arm64::RegisterSetType set) {
  switch (set) {
  case RegisterSetType::GPR:
    return &m_gpr_arm64;
  case RegisterSetType::FPR:
    return &m_fpr;
  case RegisterSetType::SVE:
    return m_sve_ptrace_payload.data();
  case RegisterSetType::SVE_HEADER:
    return &m_sve_header;
  case RegisterSetType::PAC:
    return &m_pac_mask;
  case RegisterSetType::MTE:
    return &m_mte_ctrl_reg;
  case RegisterSetType::TLS:
    return &m_tls_regs;
  case RegisterSetType::ZA:
    return m_za_ptrace_payload.data();
  case RegisterSetType::ZA_HEADER:
    return &m_za_header;
  case RegisterSetType::ZT:
    return m_zt_reg.data();
  case RegisterSetType::FPMR:
    return &m_fpmr_reg;
  case RegisterSetType::GCS:
    return &m_gcs_regs;
  case RegisterSetType::POE:
    return &m_poe_regs;
  }

  llvm_unreachable("No set buffer for this RegisterType.");
}

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
    unsigned int regset = llvm::ELF::NT_ARM_SVE;

    Flags opt_regsets;
    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, sizeof(sve_header))
            .Success())
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskSVE);

    // We may have the Scalable Matrix Extension (SME) which adds a
    // streaming SVE mode. Systems can have SVE and/or SME.
    ioVec.iov_len = sizeof(sve_header);
    regset = llvm::ELF::NT_ARM_SSVE;
    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, sizeof(sve_header))
            .Success())
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskSSVE);

    sve::user_za_header za_header;
    ioVec.iov_base = &za_header;
    ioVec.iov_len = sizeof(za_header);
    regset = llvm::ELF::NT_ARM_ZA;
    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, sizeof(za_header))
            .Success())
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskZA);

    // SME's ZT0 is a 512 bit register.
    std::array<uint8_t, 64> zt_reg;
    ioVec.iov_base = zt_reg.data();
    ioVec.iov_len = zt_reg.size();
    regset = llvm::ELF::NT_ARM_ZT;
    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, zt_reg.size())
            .Success())
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskZT);

    NativeProcessLinux &process = native_thread.GetProcess();

    std::optional<uint64_t> auxv_at_hwcap =
        process.GetAuxValue(AuxVector::AUXV_AT_HWCAP);
    if (auxv_at_hwcap && (*auxv_at_hwcap & HWCAP_PACA))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskPAuth);

    std::optional<uint64_t> auxv_at_hwcap2 =
        process.GetAuxValue(AuxVector::AUXV_AT_HWCAP2);
    if (auxv_at_hwcap2) {
      if (*auxv_at_hwcap2 & HWCAP2_MTE)
        opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskMTE);
      if (*auxv_at_hwcap2 & HWCAP2_FPMR)
        opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskFPMR);
      if (auxv_at_hwcap.value_or(0) & HWCAP_GCS)
        opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskGCS);
      if (*auxv_at_hwcap2 & HWCAP2_POE)
        opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskPOE);
    }

    opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskTLS);

    std::optional<uint64_t> auxv_at_hwcap3 =
        process.GetAuxValue(AuxVector::AUXV_AT_HWCAP3);
    linux_arm64::RegisterConfig config;
    config.features = opt_regsets;
    config.hwcap = auxv_at_hwcap.value_or(0);
    config.hwcap2 = auxv_at_hwcap2.value_or(0);
    config.hwcap3 = auxv_at_hwcap3.value_or(0);
    return std::make_unique<NativeRegisterContextLinux_arm64>(
        target_arch, native_thread, config);
  }
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return DetermineArchitectureViaGPR(tid, sizeof(linux_arm64::GPR));
}

NativeRegisterContextLinux_arm64::NativeRegisterContextLinux_arm64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    linux_arm64::RegisterConfig config)
    : NativeRegisterContext(
          native_thread, target_arch,
          CreateLinuxArm64RegisterLayout(target_arch, config)),
      NativeRegisterContextLinux(native_thread), m_register_config(config) {
  // 16 is just a maximum value, query hardware for actual watchpoint count
  m_max_hwp_supported = 16;
  m_max_hbp_supported = 16;

  // SME adds the tpidr2 register
  m_tls_size =
      m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSSVE)
          ? sizeof(m_tls_regs)
                                                 : sizeof(m_tls_regs.tpidr_reg);

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSVE) ||
      m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSSVE))
    m_sve_state = SVEState::Unknown;
  else
    m_sve_state = SVEState::Disabled;
}

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_arm64::ReadRegister(RegisterID id) {
  RegisterValue reg_value;
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (RegisterInfo_arm64::IsSVERegVG(reg)) {
    Status error = ReadSVEHeader();
    if (error.Fail())
      return error.ToError();
    reg_value.SetUInt64(GetSVERegVG());
    return reg_value;
  }
  auto required = GetLayout().GetRequiredBanks(descriptor->id, false);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case linux_arm64::GP:
      status = ReadGPR();
      break;
    case linux_arm64::FP:
      status = ReadFPR();
      break;
    case linux_arm64::SVE:
      status = ReadAllSVE();
      break;
    case linux_arm64::PAC:
      status = ReadPAuthMask();
      break;
    case linux_arm64::MTE:
      status = ReadMTEControl();
      break;
    case linux_arm64::TLS:
      status = ReadTLS();
      break;
    case linux_arm64::SMEControl:
      status = ReadSMESVG();
      ReadSMEControl();
      break;
    case linux_arm64::ZA:
      status = ReadZAHeader();
      if (status.Success()) {
        if (m_za_header.size == sizeof(m_za_header))
          m_za_ptrace_payload.assign(
              sizeof(m_za_header) + descriptor->byte_size, 0);
        else
          status = ReadZA();
      }
      break;
    case linux_arm64::ZT:
      status = ReadZT();
      break;
    case linux_arm64::FPMR:
      status = ReadFPMR();
      break;
    case linux_arm64::GCS:
      status = ReadGCS();
      break;
    case linux_arm64::POE:
      status = ReadPOE();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{linux_arm64::GP,
        {reinterpret_cast<const uint8_t *>(&m_gpr_arm64), sizeof(m_gpr_arm64)}},
       {linux_arm64::FP,
        {reinterpret_cast<const uint8_t *>(&m_fpr), sizeof(m_fpr)}},
       {linux_arm64::SVE,
        {reinterpret_cast<const uint8_t *>(m_sve_ptrace_payload.data()),
         m_sve_ptrace_payload.size()}},
       {linux_arm64::PAC,
        {reinterpret_cast<const uint8_t *>(&m_pac_mask), sizeof(m_pac_mask)}},
       {linux_arm64::MTE,
        {reinterpret_cast<const uint8_t *>(&m_mte_ctrl_reg),
         sizeof(m_mte_ctrl_reg)}},
       {linux_arm64::TLS,
        {reinterpret_cast<const uint8_t *>(&m_tls_regs), m_tls_size}},
       {linux_arm64::SMEControl,
        {reinterpret_cast<const uint8_t *>(&m_sme_pseudo_regs),
         sizeof(m_sme_pseudo_regs)}},
       {linux_arm64::ZA,
        {reinterpret_cast<const uint8_t *>(m_za_ptrace_payload.data()),
         m_za_ptrace_payload.size()}},
       {linux_arm64::ZT,
        {reinterpret_cast<const uint8_t *>(m_zt_reg.data()), m_zt_reg.size()}},
       {linux_arm64::FPMR,
        {reinterpret_cast<const uint8_t *>(&m_fpmr_reg), sizeof(m_fpmr_reg)}},
       {linux_arm64::GCS,
        {reinterpret_cast<const uint8_t *>(&m_gcs_regs), sizeof(m_gcs_regs)}},
       {linux_arm64::POE,
        {reinterpret_cast<const uint8_t *>(&m_poe_regs), sizeof(m_poe_regs)}}});
  return value;
}

llvm::Error NativeRegisterContextLinux_arm64::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  Status error;
  if (RegisterInfo_arm64::IsSVERegVG(reg)) {
    uint64_t vg_value = reg_value.GetAsUInt64();

    if (sve::vl_valid(vg_value * 8)) {
      if (IsValid(RegisterSetType::SVE_HEADER) && vg_value == GetSVERegVG())
        return error.ToError();

      SetSVERegVG(vg_value);

      error = WriteSVEHeader();
      if (error.Success()) {
        // Changing VG during streaming mode also changes the size of ZA.
        if (m_sve_state == SVEState::Streaming)
          Invalidate(RegisterSetType::ZA_HEADER);
        ConfigureRegisterContext();
      }

      if (IsValid(RegisterSetType::SVE_HEADER) && vg_value == GetSVERegVG())
        return error.ToError();
    }

    return Status::FromErrorString("SVE vector length update failed.")
        .ToError();
  }

  if (RegisterInfo_arm64::IsPAuthReg(*GetDescription(), reg))
    return Status::FromErrorString("pointer authentication masks are read-only")
        .ToError();
  if (RegisterInfo_arm64::IsSMEReg(*GetDescription(), reg) &&
      !RegisterInfo_arm64::IsSMERegZA(*GetDescription(), reg) &&
      !RegisterInfo_arm64::IsSMERegZT(*GetDescription(), reg))
    return Status::FromErrorString("Writing to SVG or SVCR is not supported.")
        .ToError();
  if (m_sve_state == SVEState::StreamingFPSIMD &&
      (RegisterInfo_arm64::IsSVEPReg(reg) ||
       RegisterInfo_arm64::IsSVERegFFR(reg)))
    return Status::FromErrorString(
               "cannot write predicates outside streaming mode")
        .ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, true);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case linux_arm64::GP:
      status = ReadGPR();
      break;
    case linux_arm64::FP:
      status = ReadFPR();
      break;
    case linux_arm64::SVE:
      status = ReadAllSVE();
      break;
    case linux_arm64::PAC:
      status = ReadPAuthMask();
      break;
    case linux_arm64::MTE:
      status = ReadMTEControl();
      break;
    case linux_arm64::TLS:
      status = ReadTLS();
      break;
    case linux_arm64::SMEControl:
      status = ReadSMESVG();
      ReadSMEControl();
      break;
    case linux_arm64::ZA:
      status = ReadZA();
      break;
    case linux_arm64::ZT:
      status = ReadZT();
      break;
    case linux_arm64::FPMR:
      status = ReadFPMR();
      break;
    case linux_arm64::GCS:
      status = ReadGCS();
      break;
    case linux_arm64::POE:
      status = ReadPOE();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  RegisterValue write_value = reg_value;
  // An SME-only target outside streaming mode accepts the SIMD portion of Z.
  if (m_sve_state == SVEState::StreamingFPSIMD &&
      RegisterInfo_arm64::IsSVEZReg(reg)) {
    std::vector<uint8_t> bytes(descriptor->byte_size, 0);
    if (reg_value.GetByteSize() < 16)
      return Status::FromErrorString("truncated SIMD value").ToError();
    std::memcpy(bytes.data(), reg_value.GetBytes(), 16);
    write_value.SetFromMemoryData(descriptor->encoding, descriptor->byte_size,
                                  bytes.data(), bytes.size(), GetByteOrder(),
                                  error, descriptor->name);
    if (error.Fail())
      return error.ToError();
  }
  if (auto codec_error = WriteBankRegister(
          descriptor->id, write_value,
          {{linux_arm64::GP,
            {reinterpret_cast<uint8_t *>(&m_gpr_arm64), sizeof(m_gpr_arm64)}},
           {linux_arm64::FP,
            {reinterpret_cast<uint8_t *>(&m_fpr), sizeof(m_fpr)}},
           {linux_arm64::SVE,
            {reinterpret_cast<uint8_t *>(m_sve_ptrace_payload.data()),
             m_sve_ptrace_payload.size()}},
           {linux_arm64::PAC,
            {reinterpret_cast<uint8_t *>(&m_pac_mask), sizeof(m_pac_mask)}},
           {linux_arm64::MTE,
            {reinterpret_cast<uint8_t *>(&m_mte_ctrl_reg),
             sizeof(m_mte_ctrl_reg)}},
           {linux_arm64::TLS,
            {reinterpret_cast<uint8_t *>(&m_tls_regs), m_tls_size}},
           {linux_arm64::SMEControl,
            {reinterpret_cast<uint8_t *>(&m_sme_pseudo_regs),
             sizeof(m_sme_pseudo_regs)}},
           {linux_arm64::ZA,
            {reinterpret_cast<uint8_t *>(m_za_ptrace_payload.data()),
             m_za_ptrace_payload.size()}},
           {linux_arm64::ZT,
            {reinterpret_cast<uint8_t *>(m_zt_reg.data()), m_zt_reg.size()}},
           {linux_arm64::FPMR,
            {reinterpret_cast<uint8_t *>(&m_fpmr_reg), sizeof(m_fpmr_reg)}},
           {linux_arm64::GCS,
            {reinterpret_cast<uint8_t *>(&m_gcs_regs), sizeof(m_gcs_regs)}},
           {linux_arm64::POE,
            {reinterpret_cast<uint8_t *>(&m_poe_regs), sizeof(m_poe_regs)}}}))
    return std::move(codec_error);
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case linux_arm64::GP:
      status = WriteGPR();
      break;
    case linux_arm64::FP:
      status = WriteFPR();
      break;
    case linux_arm64::SVE:
      status = WriteAllSVE();
      break;
    case linux_arm64::PAC:
      status =
          Status::FromErrorString("pointer authentication masks are read-only");
      break;
    case linux_arm64::MTE:
      status = WriteMTEControl();
      break;
    case linux_arm64::TLS:
      status = WriteTLS();
      break;
    case linux_arm64::SMEControl:
      status =
          Status::FromErrorString("Writing to SVG or SVCR is not supported.");
      break;
    case linux_arm64::ZA:
      status = WriteZA();
      break;
    case linux_arm64::ZT:
      status = WriteZT();
      break;
    case linux_arm64::FPMR:
      status = WriteFPMR();
      break;
    case linux_arm64::GCS:
      status = WriteGCS();
      break;
    case linux_arm64::POE:
      status = WritePOE();
      break;
    default:
      llvm_unreachable("validated writable bank");
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

uint8_t *NativeRegisterContextLinux_arm64::AddRegisterSetType(
    uint8_t *dst, RegisterSetType register_set_type) {
  std::memcpy(dst, &register_set_type, sizeof(register_set_type));
  return dst + sizeof(RegisterSetType);
}

static uint8_t *AddSavedRegistersData(uint8_t *dst, void *src, size_t size) {
  ::memcpy(dst, src, size);
  return dst + size;
}

uint8_t *NativeRegisterContextLinux_arm64::AddSavedRegisters(
    uint8_t *dst, RegisterSetType register_set_type, void *src, size_t size) {
  dst = AddRegisterSetType(dst, register_set_type);
  return AddSavedRegistersData(dst, src, size);
}

Status
NativeRegisterContextLinux_arm64::CacheAllRegisters(uint32_t &cached_size) {
  Status error;
  cached_size = sizeof(RegisterSetType) + GetGPRBufferSize();
  error = ReadGPR();
  if (error.Fail())
    return error;

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskZA)) {
    error = ReadZAHeader();
    if (error.Fail())
      return error;
    // Use header size here because the buffer may contain fake data when ZA is
    // disabled. We do not want to write this fake data (all 0s) because this
    // would tell the kernel that we want ZA to become active. Which is the
    // opposite of what we want in the case where it is currently inactive.
    cached_size += sizeof(RegisterSetType) + m_za_header.size;
    // For the same reason, we need to force it to be re-read so that it will
    // always contain the real header.
    Invalidate(RegisterSetType::ZA);
    error = ReadZA();
    if (error.Fail())
      return error;

    // We will only be restoring ZT data if ZA is active. As writing to an
    // inactive ZT enables ZA, which may not be desireable.
    if (
        // If we have ZT0, or in other words, if we have SME2.
        m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskZT) &&
        // And ZA is active, which means that ZT0 is also active.
        m_za_header.size > sizeof(m_za_header)) {
      cached_size += sizeof(RegisterSetType) + GetSetSize(RegisterSetType::ZT);
      // The kernel handles an inactive ZT0 for us, and it will read as 0s if
      // inactive (unlike ZA where we fake that behaviour).
      error = ReadZT();
      if (error.Fail())
        return error;
    }
  }

  // If SVE is enabled we need not copy FPR separately, unless we are in the
  // non-streaming mode of a streaming only process (as its non-streaming mode
  // is FPSIMD, rather than SVE).
  if ((m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSVE) ||
       m_register_config.features.AnySet(
           RegisterInfo_arm64::eRegsetMaskSSVE)) &&
      m_sve_state != SVEState::StreamingFPSIMD) {
    // Store mode and register data.
    cached_size += sizeof(RegisterSetType) + sizeof(m_sve_state) +
                   GetSetSize(RegisterSetType::SVE);
    error = ReadAllSVE();
  } else {
    cached_size += sizeof(RegisterSetType) + GetFPRSize();
    error = ReadFPR();
  }
  if (error.Fail())
    return error;

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskMTE)) {
    cached_size += sizeof(RegisterSetType) + GetSetSize(RegisterSetType::MTE);
    error = ReadMTEControl();
    if (error.Fail())
      return error;
  }

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskFPMR)) {
    cached_size += sizeof(RegisterSetType) + GetSetSize(RegisterSetType::FPMR);
    error = ReadFPMR();
    if (error.Fail())
      return error;
  }

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskGCS)) {
    cached_size += sizeof(RegisterSetType) + GetSetSize(RegisterSetType::GCS);
    error = ReadGCS();
    if (error.Fail())
      return error;
  }

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskPOE)) {
    cached_size += sizeof(RegisterSetType) + GetSetSize(RegisterSetType::POE);
    error = ReadPOE();
    if (error.Fail())
      return error;
  }

  // tpidr is always present but tpidr2 depends on SME.
  cached_size += sizeof(RegisterSetType) + GetSetSize(RegisterSetType::TLS);
  error = ReadTLS();

  return error;
}

Status NativeRegisterContextLinux_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  // AArch64 register data must contain GPRs and either FPR or SVE registers.
  // SVE registers can be non-streaming (aka SVE) or streaming (aka SSVE).
  // Finally an optional MTE register. Pointer Authentication (PAC) registers
  // are read-only and will be skipped.

  // In order to create register data checkpoint we first read all register
  // values if not done already and calculate total size of register set data.
  // We store all register values in data_sp by copying full PTrace data that
  // corresponds to register sets enabled by current register context.

  uint32_t reg_data_byte_size = 0;
  Status error = CacheAllRegisters(reg_data_byte_size);
  if (error.Fail())
    return error;

  data_sp.reset(new DataBufferHeap(reg_data_byte_size, 0));
  uint8_t *dst = data_sp->GetBytes();

  dst = AddSavedRegisters(dst, RegisterSetType::GPR, GetGPRBuffer(),
                          GetGPRBufferSize());

  // Streaming SVE and the ZA register both use the streaming vector length.
  // When you change this, the kernel will invalidate parts of the process
  // state. Therefore we need a specific order of restoration for each mode, if
  // we also have ZA to restore.
  //
  // Streaming mode enabled, ZA enabled:
  // * Write streaming registers. This sets SVCR.SM and clears SVCR.ZA.
  // * Write ZA, this set SVCR.ZA. The register data we provide is written to
  // ZA.
  // * Result is SVCR.SM and SVCR.ZA set, with the expected data in both
  //   register sets.
  //
  // Streaming mode disabled, ZA enabled:
  // * Write ZA. This sets SVCR.ZA, and the ZA content. In the majority of cases
  //   the streaming vector length is changing, so the thread is converted into
  //   an FPSIMD thread if it is not already one. This also clears SVCR.SM.
  // * Write SVE registers, which also clears SVCR.SM but most importantly, puts
  //   us into full SVE mode instead of FPSIMD mode (where the registers are
  //   actually the 128 bit Neon registers).
  // * Result is we have SVCR.SM = 0, SVCR.ZA = 1 and the expected register
  //   state.
  //
  // Restoring in different orders leads to things like the SVE registers being
  // truncated due to the FPSIMD mode and ZA being disabled or filled with 0s
  // (disabled and 0s looks the same from inside lldb since we fake the value
  // when it's disabled).
  //
  // For more information on this, look up the uses of the relevant NT_ARM_
  // constants and the functions vec_set_vector_length, sve_set_common and
  // za_set in the Linux Kernel.

  if ((m_sve_state != SVEState::Streaming) &&
      m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskZA)) {
    // Use the header size not the buffer size, as we may be using the buffer
    // for fake data, which we do not want to write out.
    assert(m_za_header.size <= GetSetSize(RegisterSetType::ZA));
    dst =
        AddSavedRegisters(dst, RegisterSetType::ZA,
                          GetSetBuffer(RegisterSetType::ZA), m_za_header.size);
  }

  if ((m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSVE) ||
       m_register_config.features.AnySet(
           RegisterInfo_arm64::eRegsetMaskSSVE)) &&
      m_sve_state != SVEState::StreamingFPSIMD) {
    dst = AddRegisterSetType(dst, RegisterSetType::SVE);
    std::memcpy(dst, &m_sve_state, sizeof(m_sve_state));
    dst += sizeof(m_sve_state);
    dst = AddSavedRegistersData(dst, GetSetBuffer(RegisterSetType::SVE),
                                GetSetSize(RegisterSetType::SVE));
  } else {
    dst = AddSavedRegisters(dst, RegisterSetType::FPR, GetFPRBuffer(),
                            GetFPRSize());
  }

  if ((m_sve_state == SVEState::Streaming) &&
      m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskZA)) {
    assert(m_za_header.size <= GetSetSize(RegisterSetType::ZA));
    dst =
        AddSavedRegisters(dst, RegisterSetType::ZA,
                          GetSetBuffer(RegisterSetType::ZA), m_za_header.size);
  }

  // If ZT0 is present and we are going to be restoring an active ZA (which
  // implies an active ZT0), then restore ZT0 after ZA has been set. This
  // prevents us enabling ZA accidentally after the restore of ZA disabled it.
  // If we leave ZA/ZT0 inactive and read ZT0, the kernel returns 0s. Therefore
  // there's nothing for us to restore if ZA was originally inactive.
  if (
      // If we have SME2 and therefore ZT0.
      m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskZT) &&
      // And ZA is enabled.
      m_za_header.size > sizeof(m_za_header))
    dst = AddSavedRegisters(dst, RegisterSetType::ZT,
                            GetSetBuffer(RegisterSetType::ZT),
                            GetSetSize(RegisterSetType::ZT));

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskMTE)) {
    dst = AddSavedRegisters(dst, RegisterSetType::MTE,
                            GetSetBuffer(RegisterSetType::MTE),
                            GetSetSize(RegisterSetType::MTE));
  }

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskFPMR)) {
    dst = AddSavedRegisters(dst, RegisterSetType::FPMR,
                            GetSetBuffer(RegisterSetType::FPMR),
                            GetSetSize(RegisterSetType::FPMR));
  }

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskGCS)) {
    dst = AddSavedRegisters(dst, RegisterSetType::GCS,
                            GetSetBuffer(RegisterSetType::GCS),
                            GetSetSize(RegisterSetType::GCS));
  }

  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskPOE)) {
    dst = AddSavedRegisters(dst, RegisterSetType::POE,
                            GetSetBuffer(RegisterSetType::POE),
                            GetSetSize(RegisterSetType::POE));
  }

  dst = AddSavedRegisters(dst, RegisterSetType::TLS,
                          GetSetBuffer(RegisterSetType::TLS),
                          GetSetSize(RegisterSetType::TLS));

  return error;
}

Status NativeRegisterContextLinux_arm64::RestoreRegisters(
    void *buffer, const uint8_t **src, size_t len,
    const NativeRegisterContextLinux_arm64::RegisterSetType set,
    std::function<Status()> writer) {
  ::memcpy(buffer, *src, len);
  MakeValid(set);
  *src += len;
  return writer();
}

Status NativeRegisterContextLinux_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  // AArch64 register data must contain GPRs, either FPR or SVE registers
  // (which can be streaming or non-streaming) and optional MTE register.
  // Pointer Authentication (PAC) registers are read-only and will be skipped.

  // We store all register values in data_sp by copying full PTrace data that
  // corresponds to register sets enabled by current register context. In order
  // to restore from register data checkpoint we will first restore GPRs, based
  // on size of remaining register data either SVE or FPRs should be restored
  // next. SVE is not enabled if we have register data size less than or equal
  // to size of GPR + FPR + MTE.

  Status error;
  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_arm64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_arm64::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }

  uint64_t reg_data_min_size =
      GetGPRBufferSize() + GetFPRSize() + 2 * (sizeof(RegisterSetType));
  if (data_sp->GetByteSize() < reg_data_min_size) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_arm64::%s data_sp contained insufficient "
        "register data bytes, expected at least %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, reg_data_min_size, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *end = src + data_sp->GetByteSize();
  while (src < end) {
    RegisterSetType kind;
    std::memcpy(&kind, src, sizeof(kind));
    src += sizeof(RegisterSetType);

    switch (kind) {
    case RegisterSetType::GPR:
      error = RestoreRegisters(
          GetGPRBuffer(), &src, GetGPRBufferSize(), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteGPR, this));
      break;
    case RegisterSetType::SVE:
      // Restore to the correct mode, streaming or not.
      std::memcpy(&m_sve_state, src, sizeof(m_sve_state));
      src += sizeof(m_sve_state);

      // First write SVE header. We do not use RestoreRegisters because we do
      // not want src to be modified yet.
      ::memcpy(GetSetBuffer(RegisterSetType::SVE_HEADER), src,
               GetSetSize(RegisterSetType::SVE_HEADER));
      if (!sve::vl_valid(m_sve_header.vl)) {
        Invalidate(RegisterSetType::SVE_HEADER);
        error = Status::FromErrorStringWithFormat(
            "NativeRegisterContextLinux_arm64::%s "
            "Invalid SVE header in data_sp",
            __FUNCTION__);
        return error;
      }
      MakeValid(RegisterSetType::SVE_HEADER);
      error = WriteSVEHeader();
      if (error.Fail())
        return error;

      // SVE header has been written configure SVE vector length if needed.
      // This could change ZA data too, but that will be restored again later
      // anyway.
      ConfigureRegisterContext();

      // Write header and register data, incrementing src this time.
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::SVE), &src,
          GetSetSize(RegisterSetType::SVE), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteAllSVE, this));
      break;
    case RegisterSetType::FPR: {
      Invalidate(RegisterSetType::SVE_HEADER);
      m_sve_state = SVEState::Unknown;
      ConfigureRegisterContext();

      // If we are on an SME only system and currently in streaming mode, about
      // to restore non-streaming FP data.
      if (!m_register_config.features.AnySet(
              RegisterInfo_arm64::eRegsetMaskSVE) &&
          m_register_config.features.AnySet(
              RegisterInfo_arm64::eRegsetMaskSSVE) &&
          m_sve_state == SVEState::Streaming) {
        // We can only restore this data on kernel versions >= 6.19, so
        // attempt it and if it fails, we will skip restoring the data.
        //
        // To attempt the restore we write FPSIMD format data to NT_ARM_SVE,
        // with the vector length set to 0. If supported, this will switch
        // modes from streaming to non-streaming and update the FP registers
        // with the values we provided.
        //
        // This interface is only used by LLDB in this one specific
        // circumstance.

        size_t data_size = sve::ptrace_fpsimd_offset + GetFPRSize();
        // NT_ARM_SVE data must be a multiple of 128 bits, and the FPU data size
        // is not, round up.
        data_size =
            (data_size + sve::vq_bytes - 1) / sve::vq_bytes * sve::vq_bytes;
        std::vector<uint8_t> sve_fpsimd_data(data_size);

        sve::user_sve_header *header =
            reinterpret_cast<sve::user_sve_header *>(sve_fpsimd_data.data());
        std::memset(header, 0, sizeof(sve::user_sve_header));
        header->size = sve_fpsimd_data.size();
        // VL = 0 tells the process to exit streaming mode.
        header->vl = 0;
        header->flags = sve::ptrace_regs_fpsimd;
        std::memcpy(&sve_fpsimd_data[sve::ptrace_fpsimd_offset], src,
                    GetFPRSize());

        struct iovec ioVec;
        ioVec.iov_base = sve_fpsimd_data.data();
        ioVec.iov_len = sve_fpsimd_data.size();

        // Even though the system does not have SVE, NT_ARM_SVE is used when
        // exiting streaming mode.
        error = WriteRegisterSet(&ioVec, sve_fpsimd_data.size(),
                                 llvm::ELF::NT_ARM_SVE);

        // Consume FP register set.
        src += GetFPRSize();

        if (error.Success()) {
          Invalidate(RegisterSetType::FPR);
          m_sve_state = SVEState::Unknown;
          ConfigureRegisterContext();
        }
        // Else we failed to restore these registers, but we will try to restore
        // the others.
      } else {
        error = RestoreRegisters(
            GetFPRBuffer(), &src, GetFPRSize(), kind,
            std::bind(&NativeRegisterContextLinux_arm64::WriteFPR, this));
      }
      break;
    }
    case RegisterSetType::MTE:
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::MTE), &src,
          GetSetSize(RegisterSetType::MTE), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteMTEControl, this));
      break;
    case RegisterSetType::TLS:
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::TLS), &src,
          GetSetSize(RegisterSetType::TLS), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteTLS, this));
      break;
    case RegisterSetType::ZA:
      // To enable or disable ZA you write the regset with or without register
      // data. The kernel detects this by looking at the ioVec's length, not the
      // ZA header size you pass in. Therefore we must write header and register
      // data (if present) in one go every time. Read the header only first just
      // to get the size.
      ::memcpy(GetSetBuffer(RegisterSetType::ZA_HEADER), src,
               GetSetSize(RegisterSetType::ZA_HEADER));
      // Read the header and register data. Can't use the buffer size here, it
      // may be incorrect due to being filled with dummy data previously. Resize
      // this so WriteZA uses the correct size.
      m_za_ptrace_payload.resize(m_za_header.size);
      ::memcpy(GetSetBuffer(RegisterSetType::ZA), src,
               GetSetSize(RegisterSetType::ZA));
      MakeValid(RegisterSetType::ZA);

      error = WriteZA();
      if (error.Fail())
        return error;

      // Update size of ZA, which resizes the ptrace payload potentially
      // trashing our copy of the data we just wrote.
      ConfigureRegisterContext();

      // ZA buffer now has proper size, read back the data we wrote above, from
      // ptrace.
      error = ReadZA();
      src += GetSetSize(RegisterSetType::ZA);
      break;
    case RegisterSetType::ZT:
      // Doing this would activate an inactive ZA, however we will only get here
      // if the state we are restoring had an active ZA. Restoring ZT0 will
      // always come after restoring ZA.
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::ZT), &src,
          GetSetSize(RegisterSetType::ZT), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteZT, this));
      break;
    case RegisterSetType::FPMR:
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::FPMR), &src,
          GetSetSize(RegisterSetType::FPMR), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteFPMR, this));
      break;
    case RegisterSetType::GCS: {
      // It is not permitted to enable GCS via ptrace. We can disable it, but
      // to keep things simple we will not revert any change to the
      // PR_SHADOW_STACK_ENABLE bit. Instead patch in the current enable bit
      // into the registers we are about to restore.
      Invalidate(RegisterSetType::GCS);
      error = ReadGCS();
      if (error.Fail())
        return error;

      uint64_t enable_bit = m_gcs_regs.features_enabled & 1UL;
      gcs_regs new_gcs_regs;
      std::memcpy(&new_gcs_regs, src, sizeof(new_gcs_regs));
      new_gcs_regs.features_enabled =
          (new_gcs_regs.features_enabled & ~1UL) | enable_bit;

      const uint8_t *new_gcs_src =
          reinterpret_cast<const uint8_t *>(&new_gcs_regs);
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::GCS), &new_gcs_src,
          GetSetSize(RegisterSetType::GCS), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WriteGCS, this));
      src += GetSetSize(RegisterSetType::GCS);

      break;
    }
    case RegisterSetType::POE:
      error = RestoreRegisters(
          GetSetBuffer(RegisterSetType::POE), &src,
          GetSetSize(RegisterSetType::POE), kind,
          std::bind(&NativeRegisterContextLinux_arm64::WritePOE, this));
      break;
    case RegisterSetType::PAC:
    case RegisterSetType::SVE_HEADER:
    case RegisterSetType::ZA_HEADER:
      // These are not saved or restored.
      break;
    }

    if (error.Fail())
      return error;
  }

  return error;
}

llvm::Error NativeRegisterContextLinux_arm64::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info) {
    return llvm::Error::success();
  }

  ::pid_t tid = m_thread.GetID();

  Status error = arm64::ReadHardwareDebugInfo(tid, m_max_hwp_supported,
                                              m_max_hbp_supported);
  if (error.Fail())
    return error.ToError();

  m_refresh_hwdebug_info = false;

  return llvm::Error::success();
}

llvm::Error
NativeRegisterContextLinux_arm64::WriteHardwareDebugRegs(DREGType hwbType) {
  uint32_t max_supported =
      (hwbType == eDREGTypeWATCH) ? m_max_hwp_supported : m_max_hbp_supported;
  auto &regs = (hwbType == eDREGTypeWATCH) ? m_hwp_regs : m_hbp_regs;
  return arm64::WriteHardwareDebugRegs(hwbType, m_thread.GetID(), max_supported,
                                       regs)
      .ToError();
}

Status NativeRegisterContextLinux_arm64::ReadGPR() {
  Status error;

  if (IsValid(RegisterSetType::GPR))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRBufferSize();

  error = ReadRegisterSet(&ioVec, GetGPRBufferSize(),
                          GetPtraceSet(RegisterSetType::GPR));

  if (error.Success())
    MakeValid(RegisterSetType::GPR);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteGPR() {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRBufferSize();

  Invalidate(RegisterSetType::GPR);

  return WriteRegisterSet(&ioVec, GetGPRBufferSize(),
                          GetPtraceSet(RegisterSetType::GPR));
}

Status NativeRegisterContextLinux_arm64::ReadFPR() {
  Status error;

  if (IsValid(RegisterSetType::FPR))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  error =
      ReadRegisterSet(&ioVec, GetFPRSize(), GetPtraceSet(RegisterSetType::FPR));
  if (error.Success())
    MakeValid(RegisterSetType::FPR);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteFPR() {
  Status error = ReadFPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  Invalidate(RegisterSetType::FPR);

  return WriteRegisterSet(&ioVec, GetFPRSize(),
                          GetPtraceSet(RegisterSetType::FPR));
}

void NativeRegisterContextLinux_arm64::InvalidateAllRegisters() {
  m_validity = static_cast<RegisterSetType>(0);

  // Update SVE and ZA registers in case there is change in configuration.
  ConfigureRegisterContext();
}

Status NativeRegisterContextLinux_arm64::ReadSVEHeader() {
  Status error;

  if (IsValid(RegisterSetType::SVE_HEADER))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::SVE_HEADER);
  ioVec.iov_len = GetSetSize(RegisterSetType::SVE_HEADER);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::SVE_HEADER),
                          GetPtraceSet(RegisterSetType::SVE_HEADER));

  if (error.Success())
    MakeValid(RegisterSetType::SVE_HEADER);

  return error;
}

Status NativeRegisterContextLinux_arm64::ReadPAuthMask() {
  Status error;

  if (IsValid(RegisterSetType::PAC))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::PAC);
  ioVec.iov_len = GetSetSize(RegisterSetType::PAC);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::PAC),
                          GetPtraceSet(RegisterSetType::PAC));

  if (error.Success())
    MakeValid(RegisterSetType::PAC);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteSVEHeader() {
  Status error;

  error = ReadSVEHeader();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::SVE_HEADER);
  ioVec.iov_len = GetSetSize(RegisterSetType::SVE_HEADER);

  Invalidate(RegisterSetType::SVE_HEADER);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::SVE_HEADER),
                          GetPtraceSet(RegisterSetType::SVE_HEADER));
}

Status NativeRegisterContextLinux_arm64::ReadAllSVE() {
  Status error;
  if (IsValid(RegisterSetType::SVE))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::SVE);
  ioVec.iov_len = GetSetSize(RegisterSetType::SVE);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::SVE),
                          GetPtraceSet(RegisterSetType::SVE));

  if (error.Success())
    MakeValid(RegisterSetType::SVE);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteAllSVE() {
  Status error;

  error = ReadAllSVE();
  if (error.Fail())
    return error;

  struct iovec ioVec;

  ioVec.iov_base = GetSetBuffer(RegisterSetType::SVE);
  ioVec.iov_len = GetSetSize(RegisterSetType::SVE);

  Invalidate(RegisterSetType::SVE);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::SVE),
                          GetPtraceSet(RegisterSetType::SVE));
}

Status NativeRegisterContextLinux_arm64::ReadSMEControl() {
  // The real register is SVCR and is accessible from EL0. However we don't want
  // to have to JIT code into the target process so we'll just recreate it using
  // what we know from ptrace.

  // Bit 0 indicates whether streaming mode is active.
  m_sme_pseudo_regs.ctrl_reg = m_sve_state == SVEState::Streaming;

  // Bit 1 indicates whether the array storage is active.
  // It is active if we can read the header and the size field tells us that
  // there is register data following it.
  Status error = ReadZAHeader();
  if (error.Success() && (m_za_header.size > sizeof(m_za_header)))
    m_sme_pseudo_regs.ctrl_reg |= 2;

  return error;
}

Status NativeRegisterContextLinux_arm64::ReadMTEControl() {
  Status error;

  if (IsValid(RegisterSetType::MTE))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::MTE);
  ioVec.iov_len = GetSetSize(RegisterSetType::MTE);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::MTE),
                          GetPtraceSet(RegisterSetType::MTE));

  if (error.Success())
    MakeValid(RegisterSetType::MTE);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteMTEControl() {
  Status error;

  error = ReadMTEControl();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::MTE);
  ioVec.iov_len = GetSetSize(RegisterSetType::MTE);

  Invalidate(RegisterSetType::MTE);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::MTE),
                          GetPtraceSet(RegisterSetType::MTE));
}

Status NativeRegisterContextLinux_arm64::ReadTLS() {
  Status error;

  if (IsValid(RegisterSetType::TLS))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::TLS);
  ioVec.iov_len = GetSetSize(RegisterSetType::TLS);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::TLS),
                          GetPtraceSet(RegisterSetType::TLS));

  if (error.Success())
    MakeValid(RegisterSetType::TLS);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteTLS() {
  Status error;

  error = ReadTLS();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::TLS);
  ioVec.iov_len = GetSetSize(RegisterSetType::TLS);

  Invalidate(RegisterSetType::TLS);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::TLS),
                          GetPtraceSet(RegisterSetType::TLS));
}

Status NativeRegisterContextLinux_arm64::ReadGCS() {
  Status error;

  if (IsValid(RegisterSetType::GCS))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::GCS);
  ioVec.iov_len = GetSetSize(RegisterSetType::GCS);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::GCS),
                          GetPtraceSet(RegisterSetType::GCS));

  if (error.Success())
    MakeValid(RegisterSetType::GCS);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteGCS() {
  Status error;

  error = ReadGCS();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::GCS);
  ioVec.iov_len = GetSetSize(RegisterSetType::GCS);

  Invalidate(RegisterSetType::GCS);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::GCS),
                          GetPtraceSet(RegisterSetType::GCS));
}

Status NativeRegisterContextLinux_arm64::ReadZAHeader() {
  Status error;

  if (IsValid(RegisterSetType::ZA_HEADER))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::ZA_HEADER);
  ioVec.iov_len = GetSetSize(RegisterSetType::ZA_HEADER);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::ZA_HEADER),
                          GetPtraceSet(RegisterSetType::ZA));

  if (error.Success())
    MakeValid(RegisterSetType::ZA_HEADER);

  return error;
}

Status NativeRegisterContextLinux_arm64::ReadZA() {
  Status error;

  if (IsValid(RegisterSetType::ZA))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::ZA);
  ioVec.iov_len = GetSetSize(RegisterSetType::ZA);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::ZA),
                          GetPtraceSet(RegisterSetType::ZA));

  if (error.Success())
    MakeValid(RegisterSetType::ZA);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteZA() {
  // Note that because the ZA ptrace payload contains the header also, this
  // method will write both. This is done because writing only the header
  // will disable ZA, even if .size in the header is correct for an enabled ZA.
  Status error;

  error = ReadZA();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::ZA);
  ioVec.iov_len = GetSetSize(RegisterSetType::ZA);

  Invalidate(RegisterSetType::ZA);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::ZA),
                          GetPtraceSet(RegisterSetType::ZA));
}

Status NativeRegisterContextLinux_arm64::ReadZT() {
  Status error;

  if (IsValid(RegisterSetType::ZT))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::ZT);
  ioVec.iov_len = GetSetSize(RegisterSetType::ZT);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::ZT),
                          GetPtraceSet(RegisterSetType::ZT));
  if (error.Success())
    MakeValid(RegisterSetType::ZT);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteZT() {
  Status error;

  error = ReadZT();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::ZT);
  ioVec.iov_len = GetSetSize(RegisterSetType::ZT);

  Invalidate(RegisterSetType::ZT);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::ZT),
                          GetPtraceSet(RegisterSetType::ZT));
}

Status NativeRegisterContextLinux_arm64::ReadFPMR() {
  Status error;

  if (IsValid(RegisterSetType::FPMR))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::FPMR);
  ioVec.iov_len = GetSetSize(RegisterSetType::FPMR);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::FPMR),
                          GetPtraceSet(RegisterSetType::FPMR));

  if (error.Success())
    MakeValid(RegisterSetType::FPMR);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteFPMR() {
  Status error;

  error = ReadFPMR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::FPMR);
  ioVec.iov_len = GetSetSize(RegisterSetType::FPMR);

  Invalidate(RegisterSetType::FPMR);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::FPMR),
                          GetPtraceSet(RegisterSetType::FPMR));
}

Status NativeRegisterContextLinux_arm64::ReadPOE() {
  Status error;

  if (IsValid(RegisterSetType::POE))
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::POE);
  ioVec.iov_len = GetSetSize(RegisterSetType::POE);

  error = ReadRegisterSet(&ioVec, GetSetSize(RegisterSetType::POE),
                          GetPtraceSet(RegisterSetType::POE));

  if (error.Success())
    MakeValid(RegisterSetType::POE);

  return error;
}

Status NativeRegisterContextLinux_arm64::WritePOE() {
  Status error;

  error = ReadPOE();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetSetBuffer(RegisterSetType::POE);
  ioVec.iov_len = GetSetSize(RegisterSetType::POE);

  Invalidate(RegisterSetType::POE);

  return WriteRegisterSet(&ioVec, GetSetSize(RegisterSetType::POE),
                          GetPtraceSet(RegisterSetType::POE));
}

void NativeRegisterContextLinux_arm64::ConfigureRegisterContext() {
  // ConfigureRegisterContext gets called from InvalidateAllRegisters
  // on every stop and configures SVE vector length and whether we are in
  // streaming SVE mode.
  // If m_sve_state is set to SVEState::Disabled on first stop, code below will
  // be deemed non operational for the lifetime of current process.
  if (!IsValid(RegisterSetType::SVE_HEADER) &&
      m_sve_state != SVEState::Disabled) {
    // Systems may have SVE and/or SME. If they are SME only, the SVE regset
    // cannot be read from but the SME one can. If they have both SVE and SME,
    // only the active mode will return valid register data.

    // Check for SME.
    Invalidate(RegisterSetType::SVE_HEADER);
    m_sve_state = SVEState::Streaming;
    Status error = ReadSVEHeader();

    bool has_sme = error.Success();
    bool sme_is_active =
        has_sme &&
        ((m_sve_header.flags & sve::ptrace_regs_mask) == sve::ptrace_regs_sve);

    // Check for SVE.
    Invalidate(RegisterSetType::SVE_HEADER);
    m_sve_state = SVEState::Full;
    error = ReadSVEHeader();

    bool has_sve = error.Success();
    bool sve_is_active =
        has_sve &&
        ((m_sve_header.flags & sve::ptrace_regs_mask) == sve::ptrace_regs_sve);
    // We do not check this for streaming mode because the streaming mode regset
    // will never be in FP format.
    bool fp_is_active =
        has_sve && ((m_sve_header.flags & sve::ptrace_regs_mask) ==
                    sve::ptrace_regs_fpsimd);

    if (sme_is_active)
      m_sve_state = SVEState::Streaming;
    else if (sve_is_active)
      m_sve_state = SVEState::Full;
    else if (fp_is_active)
      m_sve_state = SVEState::FPSIMD;
    else if (has_sme) {
      // We are in the non-streaming mode of an SME only system.
      m_sve_state = SVEState::StreamingFPSIMD;
    } else
      m_sve_state = SVEState::Disabled;

    if (m_sve_state == SVEState::Full || m_sve_state == SVEState::FPSIMD ||
        m_sve_state == SVEState::Streaming ||
        m_sve_state == SVEState::StreamingFPSIMD) {
      Invalidate(RegisterSetType::SVE_HEADER);
      error = ReadSVEHeader();

      // On every stop we configure SVE vector length by calling
      // ConfigureVectorLengthSVE regardless of current SVEState of this thread.
      uint32_t vq = RegisterInfo_arm64::eVectorQuadwordAArch64SVE;
      if (sve::vl_valid(m_sve_header.vl))
        vq = sve::vq_from_vl(m_sve_header.vl);

      ConfigureVectorLengthSVE(vq);
      m_sve_ptrace_payload.resize(sve::PTraceSize(vq, sve::ptrace_regs_sve));
    }
  }

  if (!IsValid(RegisterSetType::ZA_HEADER)) {
    Status error = ReadZAHeader();
    if (error.Success()) {
      uint32_t vq = RegisterInfo_arm64::eVectorQuadwordAArch64SVE;
      if (sve::vl_valid(m_za_header.vl))
        vq = sve::vq_from_vl(m_za_header.vl);

      ConfigureVectorLengthZA(vq);
      m_za_ptrace_payload.resize(m_za_header.size);
      Invalidate(RegisterSetType::ZA);
    }
  }
  SetRegisterLayoutState(m_sve_state);
}

Status NativeRegisterContextLinux_arm64::ReadSMESVG() {
  // This register is the streaming vector length, so we will get it from
  // NT_ARM_ZA regardless of the current streaming mode.
  Status error = ReadZAHeader();
  if (error.Success())
    m_sme_pseudo_regs.svg_reg = m_za_header.vl / 8;

  return error;
}

std::vector<RegisterID> NativeRegisterContextLinux_arm64::GetExpeditedRegisters(
    ExpeditedRegs expType) const {
  auto ids = NativeRegisterContext::GetExpeditedRegisters(expType);
  auto description = GetDescription();
  // Non-streaming SVE vector length.
  if (m_sve_state == SVEState::FPSIMD || m_sve_state == SVEState::Full)
    if (const auto *vg = description->Find("vg"))
      ids.push_back(vg->id);
  // SME's streaming vector length is available outside streaming mode too.
  if (m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSSVE))
    if (const auto *svg = description->Find("svg"))
      ids.push_back(svg->id);
  return ids;
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

void NativeRegisterContextLinux_arm64::RebuildRegisterLayout() {
  SetRegisterLayout(CreateLinuxArm64RegisterLayout(
      GetTargetArchitecture(), m_register_config, GetDescription().get()));
}
uint32_t
NativeRegisterContextLinux_arm64::ConfigureVectorLengthSVE(uint32_t vq) {
  if (!RegisterInfo_arm64::VectorSizeIsValid(vq) ||
      m_register_config.sve_vq == vq ||
      (!m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSVE) &&
       !m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSSVE)))
    return m_register_config.sve_vq;
  m_register_config.sve_vq = std::max(1u, vq);
  RebuildRegisterLayout();
  return m_register_config.sve_vq;
}
void NativeRegisterContextLinux_arm64::ConfigureVectorLengthZA(uint32_t vq) {
  if (!RegisterInfo_arm64::VectorSizeIsValid(vq) ||
      m_register_config.za_vq == vq ||
      !m_register_config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSSVE))
    return;
  m_register_config.za_vq = vq;
  RebuildRegisterLayout();
}
void NativeRegisterContextLinux_arm64::SetRegisterLayoutState(SVEState state) {
  if (m_register_config.sve_state == state)
    return;
  m_register_config.sve_state = state;
  RebuildRegisterLayout();
}

llvm::Expected<RegisterCheckpoint>
NativeRegisterContextLinux_arm64::SaveCheckpoint() {
  WritableDataBufferSP data;
  if (auto error = ReadAllRegisterValues(data); error.Fail())
    return error.ToError();
  return CreateCheckpoint(
      [this, data = std::move(data)]() -> llvm::Error {
        auto &context = *this;
        // The saved SVE/SME headers reestablish vector sizes before value
        // decoding.
        return context.WriteAllRegisterValues(data).ToError();
      });
}

#endif // defined (__arm64__) || defined (__aarch64__)
