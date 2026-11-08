//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_arm64.h"
#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"
#include "llvm/Support/Endian.h"

#include "Plugins/Process/Utility/AuxVector.h"
#include "Plugins/Process/elf-core/ProcessElfCore.h"
#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"

#include <algorithm>
#include <cstddef>
#include <memory>


namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;

namespace {
// NetBSD and OpenBSD append TPIDR to the general-register core note. Their
// floating-point notes differ in the order of FPCR and FPSR.
struct BSDArm64GPR {
  uint64_t x[31];
  uint64_t sp;
  uint64_t pc;
  uint64_t spsr;
  uint64_t tpidr;
};
struct alignas(16) NetBSDArm64FPR {
  uint8_t v[32][16];
  uint32_t fpcr;
  uint32_t fpsr;
};
struct alignas(16) OpenBSDArm64FPR {
  uint8_t v[32][16];
  uint32_t fpsr;
  uint32_t fpcr;
};
static_assert(sizeof(BSDArm64GPR) == 280);
static_assert(sizeof(NetBSDArm64FPR) == 528);
static_assert(sizeof(OpenBSDArm64FPR) == 528);

template <typename FPR>
RegisterLayout CreateBSDArm64RegisterLayout(const ArchSpec &arch) {
  auto description = RegisterInfo_arm64::CreateDescription(0);
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(BSDArm64GPR)));
  llvm::cantFail(layout.AddBank(1, sizeof(FPR)));
  const bool big_endian = arch.GetByteOrder() == lldb::eByteOrderBig;
  for (const auto &reg : description->GetRegisters()) {
    RegisterBankID bank = reg.id <= reg_ids::gpr_w28 ? 0 : 1;
    uint32_t offset;
    if (reg.id <= reg_ids::gpr_cpsr)
      offset = reg.id * 8 + (reg.id == reg_ids::gpr_cpsr && big_endian ? 4 : 0);
    else if (reg.id <= reg_ids::gpr_w28)
      offset = (reg.id - reg_ids::gpr_w0) * 8 + (big_endian ? 4 : 0);
    else if (reg.id == reg_ids::fpu_fpsr)
      offset = offsetof(FPR, fpsr);
    else if (reg.id == reg_ids::fpu_fpcr)
      offset = offsetof(FPR, fpcr);
    else {
      uint32_t i = reg.id >= reg_ids::fpu_d0   ? reg.id - reg_ids::fpu_d0
                   : reg.id >= reg_ids::fpu_s0 ? reg.id - reg_ids::fpu_s0
                                               : reg.id - reg_ids::fpu_v0;
      offset = i * 16 + (big_endian ? 16 - reg.byte_size : 0);
    }
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, reg.byte_size});
    llvm::cantFail(layout.Add(reg.id, std::move(location)));
  }
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
} // namespace

llvm::Expected<std::unique_ptr<RegisterContextELFCore_arm64>>
RegisterContextELFCore_arm64::Create(Thread &thread, const ArchSpec &arch,
                                     const DataExtractor &gpregset,
                                     llvm::ArrayRef<CoreNote> notes) {
  std::optional<RegisterLayout> layout;
  std::optional<linux_arm64::RegisterConfig> config;
  switch (arch.GetTriple().getOS()) {
  case llvm::Triple::FreeBSD: {
    auto &process = static_cast<ProcessElfCore &>(*thread.GetProcess());
    AuxVector auxv(process.GetAuxvData());
    layout = CreateFreeBSDArm64RegisterLayout(
        arch, auxv.GetAuxValue(AuxVector::AUXV_FREEBSD_AT_HWCAP).value_or(0),
        auxv.GetAuxValue(AuxVector::AUXV_AT_HWCAP2).value_or(0));
    break;
  }
  case llvm::Triple::NetBSD:
    layout = CreateBSDArm64RegisterLayout<NetBSDArm64FPR>(arch);
    break;
  case llvm::Triple::OpenBSD:
    layout = CreateBSDArm64RegisterLayout<OpenBSDArm64FPR>(arch);
    break;
  case llvm::Triple::Linux: {
    Flags opt_regsets = RegisterInfo_arm64::eRegsetMaskDefault;

    DataExtractor ssve_data =
        getRegset(notes, arch.GetTriple(), AARCH64_SSVE_Desc);
    if (ssve_data.GetByteSize() >= sizeof(sve::user_sve_header))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskSSVE);

    DataExtractor sve_data =
        getRegset(notes, arch.GetTriple(), AARCH64_SVE_Desc);
    if (sve_data.GetByteSize() >= sizeof(sve::user_sve_header))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskSVE);

    // Pointer Authentication register set data is based on struct
    // user_pac_mask declared in ptrace.h. See reference implementation
    // in Linux kernel source at arch/arm64/include/uapi/asm/ptrace.h.
    DataExtractor pac_data =
        getRegset(notes, arch.GetTriple(), AARCH64_PAC_Desc);
    if (pac_data.GetByteSize() >= sizeof(linux_arm64::PAuthMasks))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskPAuth);

    DataExtractor tls_data =
        getRegset(notes, arch.GetTriple(), AARCH64_TLS_Desc);
    // A valid note will always contain at least one register, "tpidr". It may
    // expand in future.
    if (tls_data.GetByteSize() >= sizeof(linux_arm64::TLSRegisters::tpidr))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskTLS);

    DataExtractor za_data = getRegset(notes, arch.GetTriple(), AARCH64_ZA_Desc);
    // Nothing if ZA is not present, just the header if it is disabled.
    if (za_data.GetByteSize() >= sizeof(sve::user_za_header))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskZA);

    DataExtractor mte_data =
        getRegset(notes, arch.GetTriple(), AARCH64_MTE_Desc);
    if (mte_data.GetByteSize() >= sizeof(linux_arm64::MTEControl))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskMTE);

    DataExtractor zt_data = getRegset(notes, arch.GetTriple(), AARCH64_ZT_Desc);
    // Although ZT0 can be in a disabled state like ZA can, the kernel reports
    // its content as 0s in that state. Therefore even a disabled ZT0 will have
    // a note containing those 0s. ZT0 is a 512 bit / 64 byte register.
    if (zt_data.GetByteSize() >= sizeof(linux_arm64::ZTRegister))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskZT);

    DataExtractor fpmr_data =
        getRegset(notes, arch.GetTriple(), AARCH64_FPMR_Desc);
    if (fpmr_data.GetByteSize() >= sizeof(linux_arm64::FPMRRegister))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskFPMR);

    DataExtractor gcs_data =
        getRegset(notes, arch.GetTriple(), AARCH64_GCS_Desc);
    if (gcs_data.GetByteSize() >= sizeof(linux_arm64::GCSRegisters))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskGCS);

    DataExtractor poe_data =
        getRegset(notes, arch.GetTriple(), AARCH64_POE_Desc);
    if (poe_data.GetByteSize() >= sizeof(linux_arm64::POERegister))
      opt_regsets.Set(RegisterInfo_arm64::eRegsetMaskPOE);

    auto &process = static_cast<ProcessElfCore &>(*thread.GetProcess());
    AuxVector auxv(process.GetAuxvData());
    config.emplace();
    config->features = opt_regsets;
    config->hwcap = auxv.GetAuxValue(AuxVector::AUXV_AT_HWCAP).value_or(0);
    config->hwcap2 = auxv.GetAuxValue(AuxVector::AUXV_AT_HWCAP2).value_or(0);
    config->hwcap3 = auxv.GetAuxValue(AuxVector::AUXV_AT_HWCAP3).value_or(0);
    layout = CreateLinuxArm64RegisterLayout(arch, *config);
    break;
  }
  default:
    return llvm::createStringError("unsupported OS for AArch64 ELF registers");
  }
  return std::unique_ptr<RegisterContextELFCore_arm64>(
      new RegisterContextELFCore_arm64(thread, std::move(*layout), gpregset,
                                       notes, std::move(config)));
}

RegisterContextELFCore_arm64::RegisterContextELFCore_arm64(
    Thread &thread, RegisterLayout layout, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes,
    std::optional<linux_arm64::RegisterConfig> config)
    : ThreadRegisterContext(thread, 0, std::move(layout)),
      m_register_config(std::move(config)) {
  ::memset(&m_sme_pseudo_regs, 0, sizeof(m_sme_pseudo_regs));

  m_gpr_data.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                      gpregset.GetByteSize()));
  m_gpr_data.SetByteOrder(gpregset.GetByteOrder());

  const llvm::Triple &target_triple = GetTargetArchitecture().GetTriple();
  m_fpr_data = getRegset(notes, target_triple, FPR_Desc);

  if ((m_register_config && m_register_config->features.AnySet(
                                RegisterInfo_arm64::eRegsetMaskSSVE))) {
    m_sve_data = getRegset(notes, target_triple, AARCH64_SSVE_Desc);
    lldb::offset_t flags_offset = 12;
    uint16_t flags = m_sve_data.GetU32(&flags_offset);
    if ((flags & sve::ptrace_regs_mask) == sve::ptrace_regs_sve)
      m_sve_state = SVEState::Streaming;
  }

  if (m_sve_state != SVEState::Streaming &&
      (m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskSVE)))
    m_sve_data = getRegset(notes, target_triple, AARCH64_SVE_Desc);

  if ((m_register_config && m_register_config->features.AnySet(
                                RegisterInfo_arm64::eRegsetMaskPAuth)))
    m_pac_data = getRegset(notes, target_triple, AARCH64_PAC_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskTLS)))
    m_tls_data = getRegset(notes, target_triple, AARCH64_TLS_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskZA)))
    m_za_data = getRegset(notes, target_triple, AARCH64_ZA_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskMTE)))
    m_mte_data = getRegset(notes, target_triple, AARCH64_MTE_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskZT)))
    m_zt_data = getRegset(notes, target_triple, AARCH64_ZT_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskFPMR)))
    m_fpmr_data = getRegset(notes, target_triple, AARCH64_FPMR_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskGCS)))
    m_gcs_data = getRegset(notes, target_triple, AARCH64_GCS_Desc);

  if ((m_register_config &&
       m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskPOE)))
    m_poe_data = getRegset(notes, target_triple, AARCH64_POE_Desc);

  ConfigureRegisterContext();
}

RegisterContextELFCore_arm64::~RegisterContextELFCore_arm64() = default;

void RegisterContextELFCore_arm64::ConfigureRegisterContext() {
  if (m_sve_data.GetByteSize() > sizeof(sve::user_sve_header)) {
    uint64_t sve_header_field_offset = 8;
    m_sve_vector_length = m_sve_data.GetU16(&sve_header_field_offset);

    if (m_sve_state != SVEState::Streaming) {
      sve_header_field_offset = 12;
      uint16_t sve_header_flags_field =
          m_sve_data.GetU16(&sve_header_field_offset);
      if ((sve_header_flags_field & sve::ptrace_regs_mask) ==
          sve::ptrace_regs_fpsimd)
        m_sve_state = SVEState::FPSIMD;
      else if ((sve_header_flags_field & sve::ptrace_regs_mask) ==
               sve::ptrace_regs_sve)
        m_sve_state = SVEState::Full;
    }

    if (!sve::vl_valid(m_sve_vector_length)) {
      m_sve_state = SVEState::Disabled;
      m_sve_vector_length = 0;
    }
  } else
    m_sve_state = SVEState::Disabled;

  if (m_register_config && m_sve_state != SVEState::Disabled) {
    const uint32_t vq = sve::vq_from_vl(m_sve_vector_length);
    if (RegisterInfo_arm64::VectorSizeIsValid(vq))
      m_register_config->sve_vq = std::max(1u, vq);
  }

  if (m_sve_state == SVEState::Streaming)
    m_sme_pseudo_regs.ctrl_reg |= 1;

  if (m_za_data.GetByteSize() >= sizeof(sve::user_za_header)) {
    lldb::offset_t vlen_offset = 8;
    uint16_t svl = m_za_data.GetU16(&vlen_offset);
    m_sme_pseudo_regs.svg_reg = svl / 8;
    if (m_register_config &&
        m_register_config->features.AnySet(RegisterInfo_arm64::eRegsetMaskSSVE) &&
        RegisterInfo_arm64::VectorSizeIsValid(svl / 16))
      m_register_config->za_vq = svl / 16;

    // If there is register data then ZA is active. The size of the note may be
    // misleading here so we use the size field of the embedded header.
    lldb::offset_t size_offset = 0;
    uint32_t size = m_za_data.GetU32(&size_offset);
    if (size > sizeof(sve::user_za_header))
      m_sme_pseudo_regs.ctrl_reg |= 1 << 1;
  }
  if (m_register_config) {
    m_register_config->sve_state = m_sve_state;
    SetRegisterLayout(CreateLinuxArm64RegisterLayout(
        GetTargetArchitecture(), *m_register_config, GetDescription().get()));
  }
}

llvm::Expected<RegisterValue>
RegisterContextELFCore_arm64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (!reg_info)
    return llvm::createStringError("register value is unavailable");
  const auto *reg = GetDescription()->Find(reg_info->id);
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  std::vector<uint8_t> inactive_za;
  llvm::ArrayRef<uint8_t> za(m_za_data.GetDataStart(), m_za_data.GetByteSize());
  if (reg->name == "za") {
    if (m_za_data.GetByteSize() < sizeof(sve::user_za_header))
      return llvm::createStringError("register value is unavailable");
    lldb::offset_t offset = 0;
    if (m_za_data.GetU32(&offset) == sizeof(sve::user_za_header)) {
      inactive_za.resize(sizeof(sve::user_za_header) + reg->byte_size, 0);
      za = inactive_za;
    }
  }
  uint8_t pseudo[sizeof(linux_arm64::SMERegisters)];
  auto order = GetTargetArchitecture().GetByteOrder() == lldb::eByteOrderBig
                   ? llvm::endianness::big
                   : llvm::endianness::little;
  llvm::support::endian::write64(pseudo, m_sme_pseudo_regs.ctrl_reg, order);
  llvm::support::endian::write64(
      pseudo + offsetof(linux_arm64::SMERegisters, svg_reg),
      m_sme_pseudo_regs.svg_reg, order);
  auto result = ReadBankRegister(
      reg->id,
      {{linux_arm64::GP, {m_gpr_data.GetDataStart(), m_gpr_data.GetByteSize()}},
       {linux_arm64::FP, {m_fpr_data.GetDataStart(), m_fpr_data.GetByteSize()}},
       {linux_arm64::SVE,
        {m_sve_data.GetDataStart(), m_sve_data.GetByteSize()}},
       {linux_arm64::PAC,
        {m_pac_data.GetDataStart(), m_pac_data.GetByteSize()}},
       {linux_arm64::MTE,
        {m_mte_data.GetDataStart(), m_mte_data.GetByteSize()}},
       {linux_arm64::TLS,
        {m_tls_data.GetDataStart(), m_tls_data.GetByteSize()}},
       {linux_arm64::ZT, {m_zt_data.GetDataStart(), m_zt_data.GetByteSize()}},
       {linux_arm64::FPMR,
        {m_fpmr_data.GetDataStart(), m_fpmr_data.GetByteSize()}},
       {linux_arm64::GCS,
        {m_gcs_data.GetDataStart(), m_gcs_data.GetByteSize()}},
       {linux_arm64::POE,
        {m_poe_data.GetDataStart(), m_poe_data.GetByteSize()}},
       {linux_arm64::SMEControl, pseudo},
       {linux_arm64::ZA, za}});
  return result;
}

bool RegisterContextELFCore_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

llvm::Error
RegisterContextELFCore_arm64::WriteRegister(RegisterID id,
                                                   const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

bool RegisterContextELFCore_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextELFCore_arm64::HardwareSingleStep(bool enable) {
  return false;
}
