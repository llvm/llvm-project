//===-- RegisterContextPOSIXCore_riscv32.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_riscv32.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Utility/DataBufferHeap.h"

#define GPR_OFFSET(idx) ((idx) * sizeof(uint32_t))
#define FPR_OFFSET(idx) ((idx) * sizeof(uint32_t))

#define DECLARE_REGISTER_INFOS_RISCV32_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_riscv32.h"
#undef DECLARE_REGISTER_INFOS_RISCV32_STRUCT

using namespace lldb_private;

static std::vector<uint32_t> CopyRegisterListToVector(const uint32_t *regs) {
  if (!regs)
    return {};

  const uint32_t *end = regs;
  while (*end != LLDB_INVALID_REGNUM)
    ++end;

  return std::vector<uint32_t>(regs, end);
}

std::unique_ptr<RegisterContextCorePOSIX_riscv32>
RegisterContextCorePOSIX_riscv32::Create(Thread &thread, const ArchSpec &arch,
                                         const DataExtractor &gpregset,
                                         llvm::ArrayRef<CoreNote> notes) {
  return std::unique_ptr<RegisterContextCorePOSIX_riscv32>(
      new RegisterContextCorePOSIX_riscv32(
          thread, std::make_unique<RegisterInfoPOSIXDynamic_riscv32>(arch),
          gpregset, notes));
}

RegisterContextCorePOSIX_riscv32::RegisterContextCorePOSIX_riscv32(
    Thread &thread,
    std::unique_ptr<RegisterInfoPOSIXDynamic_riscv32> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContext(thread, 0), m_reg_infos_up(std::move(register_info)) {
  // Compute the maximum register counts for GPR, FPR, and CSR.
  constexpr uint32_t k_num_gpr_registers =
      std::size(g_register_infos_riscv32_gpr);
  constexpr uint32_t k_num_fpr_registers =
      std::size(g_register_infos_riscv32_fpr);
  constexpr uint32_t k_num_csr_registers =
      std::size(g_register_infos_riscv32_csr);
  const ArchSpec &target_arch = m_reg_infos_up->GetTargetArchitecture();
  const llvm::Triple triple = target_arch.GetTriple();
  const lldb::ByteOrder byte_order = target_arch.GetByteOrder();

  std::vector<DynamicRegisterInfo::Register> registers;
  uint32_t byte_offset = 0;

  // Build dynamic register information for GPR.
  const lldb_private::ConstString gpr_set("GPR");
  m_gpregset.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                      gpregset.GetByteSize()));
  if (m_gpregset.GetByteSize() >= g_register_infos_riscv32_gpr[0].byte_size) {
    // GPR is available.
    assert((m_gpregset.GetByteSize() /
            g_register_infos_riscv32_gpr[0].byte_size) == k_num_gpr_registers &&
           "GPR has the wrong number of registers!");
    m_gpregset.SetByteOrder(gpregset.GetByteOrder());
    for (const auto &gpr : g_register_infos_riscv32_gpr) {
      registers.push_back(BuildDynamicRegister(gpr, gpr_set, byte_offset));
      byte_offset += gpr.byte_size;
    }
  }

  // Build dynamic register information for FPR.
  const lldb_private::ConstString fpr_set("FPR");
  m_fpregset = getRegset(notes, triple, FPR_Desc);
  if (m_fpregset.GetByteSize() >= g_register_infos_riscv32_fpr[0].byte_size) {
    // FPR is available.
    assert((m_fpregset.GetByteSize() /
            g_register_infos_riscv32_fpr[0].byte_size) == k_num_fpr_registers &&
           "FPR has the wrong number of registers!");
    m_fpregset.SetByteOrder(byte_order);
    for (const auto &fpr : g_register_infos_riscv32_fpr) {
      registers.push_back(BuildDynamicRegister(fpr, fpr_set, byte_offset));
      byte_offset += fpr.byte_size;
    }
  }

  // Build dynamic register information for CSR.
  const lldb_private::ConstString csr_set("CSR");
  m_csregset = getRegset(notes, triple, RISCV32_CSREGMAP_Desc);
  if (m_csregset.GetByteSize() >=
      (sizeof(csr_kv_t::addr) + sizeof(csr_kv_t::val))) {
    // CSR is available.
    m_csregset.SetByteOrder(byte_order);
    lldb::offset_t offset = 0;
    std::vector<uint32_t> csregset_regnums = {};
    while (m_csregset.BytesLeft(offset)) {
      uint32_t csr_addr = m_csregset.GetU32(&offset);
      if (csregset_regnums.size() == k_num_csr_registers) {
        Debugger::ReportWarning(
            llvm::formatv("parsed the permissible number of CSRs {0:x} but "
                          "NT_CSREGMAP has more; skipping the remaining CSRs",
                          k_num_csr_registers));
        break;
      }
      if (llvm::is_contained(csregset_regnums, csr_addr)) {
        Debugger::ReportWarning(
            llvm::formatv("encountered a duplicate CSR while parsing "
                          "NT_CSREGMAP: {0}; skipping",
                          g_register_infos_riscv32_csr[csr_addr].name));
      } else {
        csregset_regnums.push_back(csr_addr);
        const RegisterInfo &csr = g_register_infos_riscv32_csr[csr_addr];
        registers.push_back(BuildDynamicRegister(csr, csr_set, byte_offset));
        byte_offset += csr.byte_size;
      }
      // Consume and skip the CSR value to advance to the next entry.
      (void)m_csregset.GetU32(&offset);
    }
  }

  m_reg_infos_up->SetRegisterInfo(std::move(registers));
}

RegisterContextCorePOSIX_riscv32::~RegisterContextCorePOSIX_riscv32() = default;

void RegisterContextCorePOSIX_riscv32::InvalidateAllRegisters() {}

size_t RegisterContextCorePOSIX_riscv32::GetRegisterCount() {
  return m_reg_infos_up->GetRegisterCount();
}

const lldb_private::RegisterInfo *
RegisterContextCorePOSIX_riscv32::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < GetRegisterCount())
    return &m_reg_infos_up->GetRegisterInfo()[static_cast<uint32_t>(reg)];
  return nullptr;
}

size_t RegisterContextCorePOSIX_riscv32::GetRegisterSetCount() {
  return m_reg_infos_up->GetRegisterSetCount();
}

const lldb_private::RegisterSet *
RegisterContextCorePOSIX_riscv32::GetRegisterSet(size_t set) {
  return m_reg_infos_up->GetRegisterSet(static_cast<uint32_t>(set));
}

bool RegisterContextCorePOSIX_riscv32::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  const lldb_private::RegisterInfo *dyn_reg_info;
  const uint8_t *src = nullptr;
  if ((dyn_reg_info = m_reg_infos_up->GetRegisterInfo(reg_info->name))) {
    lldb::offset_t offset = dyn_reg_info->byte_offset;
    if (IsGPR(dyn_reg_info->kinds[lldb::eRegisterKindLLDB])) {
      // clang-format off
      //
      // |____|                                                 |_______________|
      // |    |                                          src -> |               |
      // | pc | <- dyn_reg_info->byte_offset = 0                | m_gpregset[0] | uint32_t
      // |____|                                   offset = 0 -> |_______________|
      // |    |                                                 |               |
      // | x1 |                                                 | m_gpregset[4] | uint32_t
      // |____|                                                 |_______________|
      //   ..                                                           ..
      // |____|                                                 |_______________|
      // |    |                                                 |               |
      // | xn | <- dyn_reg_info->byte_offset          offset -> | m_gpregset[n] | uint32_t
      // |____|                                                 |_______________|
      // |    |                                                 |               |
      //   ..                                                           ..
      //
      // clang-format on
      src = m_gpregset.GetDataStart();
    } else if (IsFPR(dyn_reg_info->kinds[lldb::eRegisterKindLLDB])) {
      // clang-format off
      //
      // |____|
      // |    |
      // | pc | <- dyn_reg_info->byte_offset = 0
      // |____|
      // |    |
      // | x1 |
      // |____|
      //   ..
      // |____|
      // |    |
      // | x0 |
      // |____|                                                 |_______________|
      // |    |                                          src -> |               |
      // | f0 |                                                 | m_fpregset[0] | uint32_t
      // |____|                                   offset = 0 -> |_______________|
      // |    |                                                 |               |
      // | f1 |                                                 | m_fpregset[4] | uint32_t
      // |____|                                                 |_______________|
      //   ..                                                           ..
      // |____|                                                 |_______________|
      // |    |                                                 |               |
      // | fn | <- dyn_reg_info->byte_offset          offset -> | m_fpregset[n] | uint32_t
      // |____|                                                 |_______________|
      // |    |                                                 |               |
      //    ..                                                          ..
      //
      // clang-format on
      src = m_fpregset.GetDataStart();
      offset -= (g_register_infos_riscv32_gpr[0].byte_size *
                 m_reg_infos_up->GetGPRSize());
    } else if (IsCSR(dyn_reg_info->kinds[lldb::eRegisterKindLLDB])) {
      // clang-format off
      //
      // |______|
      // |      |
      // |  pc  | <- dyn_reg_info->byte_offset = 0
      // |______|
      // |      |
      // |  x1  |
      // |______|
      //    ..
      // |______|
      // |      |
      // |  x0  |
      // |______|
      // |      |
      // |  f0  |
      // |______|
      // |      |
      // |  f1  |
      // |______|
      //    ..
      // |______|
      // |      |
      // |  f31 |
      // |______|                                                 |________________|
      // |      |                                          src -> |                |
      // |      |                                   csr0::addr -> | m_csregset[00] | uint32_t
      // |      |                                   offset = 0 -> |                | 
      // | csr0 |                                                 |----------------|
      // |      |                                                 |                |
      // |      |                                    csr0::val -> | m_csregset[04] | uint32_t
      // |______|                                                 |________________|
      // |      |                                                 |                |
      // |      |                                   csr1::addr -> | m_csregset[08] | uint32_t
      // |      |                                                 |                | 
      // | csr1 |                                                 |----------------|
      // |      |                                                 |                |
      // |      |                                    csr1::val -> | m_csregset[12] | uint32_t
      // |______|                                                 |________________|
      //    ..                                                            ..
      // |______|                                                 |________________|
      // |      |                                                 |                |
      // |      |                                   csrn::addr -> | m_csregset[..] | uint32_t
      // |      |                                                 |                | 
      // | csrn |                                                 |----------------|
      // |      |                                    csrn::val -> |                |
      // |      |                                                 | m_csregset[..] | uint32_t
      // |______|                                       offset -> |________________|
      // |      |                                                 |                |
      //    ..                                                            ..
      //
      // clang-format on
      src = m_csregset.GetDataStart();
      offset -= ((g_register_infos_riscv32_gpr[0].byte_size *
                  m_reg_infos_up->GetGPRSize()) +
                 (g_register_infos_riscv32_fpr[0].byte_size *
                  m_reg_infos_up->GetFPRSize()));
      offset *= 2;
      offset += dyn_reg_info->byte_size;
    } else {
      return false;
    }

    Status error;
    value.SetFromMemoryData(*dyn_reg_info, src + offset,
                            dyn_reg_info->byte_size, lldb::eByteOrderLittle,
                            error);
    return error.Success();
  } else {
    return false;
  }
}

bool RegisterContextCorePOSIX_riscv32::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}

bool RegisterContextCorePOSIX_riscv32::IsGPR(unsigned reg) {
  if (llvm::StringRef(
          GetRegisterSet(m_reg_infos_up->GetRegisterSetFromRegisterIndex(reg))
              ->name)
          .equals_insensitive(llvm::StringRef("GPR")))
    return true;
  return false;
}

bool RegisterContextCorePOSIX_riscv32::IsFPR(unsigned reg) {
  if (llvm::StringRef(
          GetRegisterSet(m_reg_infos_up->GetRegisterSetFromRegisterIndex(reg))
              ->name)
          .equals_insensitive(llvm::StringRef("FPR")))
    return true;
  return false;
}

bool RegisterContextCorePOSIX_riscv32::IsCSR(unsigned reg) {
  if (llvm::StringRef(
          GetRegisterSet(m_reg_infos_up->GetRegisterSetFromRegisterIndex(reg))
              ->name)
          .equals_insensitive(llvm::StringRef("CSR")))
    return true;
  return false;
}

lldb_private::DynamicRegisterInfo::Register
RegisterContextCorePOSIX_riscv32::BuildDynamicRegister(
    const lldb_private::RegisterInfo &reg_info,
    const lldb_private::ConstString &set_name, uint32_t byte_offset) {
  return DynamicRegisterInfo::Register{
      lldb_private::ConstString(reg_info.name),
      lldb_private::ConstString(reg_info.alt_name),
      set_name,
      reg_info.byte_size,
      byte_offset,
      reg_info.encoding,
      reg_info.format,
      reg_info.kinds[lldb::eRegisterKindDWARF],
      reg_info.kinds[lldb::eRegisterKindEHFrame],
      reg_info.kinds[lldb::eRegisterKindGeneric],
      reg_info.kinds[lldb::eRegisterKindProcessPlugin],
      CopyRegisterListToVector(reg_info.value_regs),
      CopyRegisterListToVector(reg_info.invalidate_regs),
      /*value_reg_offset=*/0,
      reg_info.flags_type};
}
