//===------ NativeRegisterContextAIX_ppc64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc64__)

#include <set>
#include "lldb/Register/Arch/lldb-ppc64-register-enums.h"

#include "NativeRegisterContextAIX_ppc64.h"
#include "NativeThreadAIX.h"
#include "lldb/Register/Common/RegisterInfo_ppc64.h"
#include "lldb/Register/RegisterValue.h"

namespace reg_ids = lldb_private::registers::ppc64::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::aix_ppc64;
using namespace lldb_private::process_aix;

static const uint32_t g_gpr_regnums_ppc64[] = {
    reg_ids::gpr_r0,    reg_ids::gpr_r1,  reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,    reg_ids::gpr_r5,  reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,    reg_ids::gpr_r9,  reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12,   reg_ids::gpr_r13, reg_ids::gpr_r14, reg_ids::gpr_r15,
    reg_ids::gpr_r16,   reg_ids::gpr_r17, reg_ids::gpr_r18, reg_ids::gpr_r19,
    reg_ids::gpr_r20,   reg_ids::gpr_r21, reg_ids::gpr_r22, reg_ids::gpr_r23,
    reg_ids::gpr_r24,   reg_ids::gpr_r25, reg_ids::gpr_r26, reg_ids::gpr_r27,
    reg_ids::gpr_r28,   reg_ids::gpr_r29, reg_ids::gpr_r30, reg_ids::gpr_r31,
    reg_ids::gpr_cr,    reg_ids::gpr_msr, reg_ids::gpr_xer, reg_ids::gpr_lr,
    reg_ids::gpr_ctr,   reg_ids::gpr_pc,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_fpr_regnums_ppc64[] = {
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

static const uint32_t g_vmx_regnums_ppc64[] = {
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

static const uint32_t g_vsx_regnums_ppc64[] = {
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

static const RegisterGroupNumbers g_reg_sets_ppc64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_ppc64,
                             g_gpr_regnums_ppc64 +
                                 reg_ids::k_num_gpr_registers)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpr_regnums_ppc64,
                             g_fpr_regnums_ppc64 +
                                 reg_ids::k_num_fpr_registers)},
    {"AltiVec/VMX Registers", "vmx",
     std::vector<RegisterID>(g_vmx_regnums_ppc64,
                             g_vmx_regnums_ppc64 +
                                 reg_ids::k_num_vmx_registers)},
    {"VSX Registers", "vsx",
     std::vector<RegisterID>(g_vsx_regnums_ppc64,
                             g_vsx_regnums_ppc64 +
                                 reg_ids::k_num_vsx_registers)}};

std::unique_ptr<NativeRegisterContextAIX>
NativeRegisterContextAIX::CreateHostNativeRegisterContextAIX(
    const ArchSpec &target_arch, NativeThreadAIX &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    return std::make_unique<NativeRegisterContextAIX_ppc64>(target_arch,
                                                            native_thread);
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

NativeRegisterContextAIX_ppc64::NativeRegisterContextAIX_ppc64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateAIXPPC64RegisterLayout(target_arch)),
      NativeRegisterContextAIX(native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::ppc:
    new (&m_gpr_storage.gpr32) GPR_PPC{};
    m_gpr = &m_gpr_storage.gpr32;
    break;
  case llvm::Triple::ppc64:
    new (&m_gpr_storage.gpr64) GPR_PPC64{};
    m_gpr = &m_gpr_storage.gpr64;
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (set_index < k_num_register_sets)
      return &g_reg_sets_ppc64[set_index];

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
NativeRegisterContextAIX_ppc64::ReadRegister(RegisterID id) {
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
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{0, {reinterpret_cast<const uint8_t *>(GetGPRBuffer()), GetGPRSize()}}});
  return value;
}

llvm::Error
NativeRegisterContextAIX_ppc64::WriteRegister(RegisterID id,
                                              const RegisterValue &reg_value) {
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
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (auto error = WriteBankRegister(
          descriptor->id, reg_value,
          {{0, {reinterpret_cast<uint8_t *>(GetGPRBuffer()), GetGPRSize()}}}))
    return error;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = WriteGPR();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextAIX_ppc64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return Status("unimplemented");
}

Status NativeRegisterContextAIX_ppc64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return Status("unimplemented");
}

bool NativeRegisterContextAIX_ppc64::IsGPR(unsigned reg) const {
  return reg <= reg_ids::k_last_gpr;
}

bool NativeRegisterContextAIX_ppc64::IsFPR(unsigned reg) const {
  return (reg_ids::k_first_fpr <= reg && reg <= reg_ids::k_last_fpr);
}

bool NativeRegisterContextAIX_ppc64::IsVMX(unsigned reg) const {
  return (reg >= reg_ids::k_first_vmx) && (reg <= reg_ids::k_last_vmx);
}

bool NativeRegisterContextAIX_ppc64::IsVSX(unsigned reg) const {
  return (reg >= reg_ids::k_first_vsx) && (reg <= reg_ids::k_last_vsx);
}

uint32_t NativeRegisterContextAIX_ppc64::CalculateFprOffset(
    const RegisterDescriptor *reg_info) const {
  return 0;
}

uint32_t NativeRegisterContextAIX_ppc64::CalculateVmxOffset(
    const RegisterDescriptor *reg_info) const {
  return 0;
}

uint32_t NativeRegisterContextAIX_ppc64::CalculateVsxOffset(
    const RegisterDescriptor *reg_info) const {
  return 0;
}

#endif // defined(__powerpc64__)
