//===-- RegisterContextPOSIXCore_riscv32.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_riscv32.h"

#include "lldb/Utility/DataBufferHeap.h"

#define GPR_OFFSET(idx) ((idx) * sizeof(uint32_t))
#define FPR_OFFSET(idx) ((idx) * sizeof(uint32_t))
#define CSR_OFFSET(idx) ((idx) * sizeof(uint32_t))

#define DECLARE_REGISTER_INFOS_RISCV32_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_riscv32.h"
#undef DECLARE_REGISTER_INFOS_RISCV32_STRUCT

using namespace lldb_private;

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
  uint32_t k_num_gpr_registers = (sizeof(g_register_infos_riscv32_gpr) /
                                  sizeof(g_register_infos_riscv32_gpr[0]));
  uint32_t k_num_fpr_registers = (sizeof(g_register_infos_riscv32_fpr) /
                                  sizeof(g_register_infos_riscv32_fpr[0]));
  uint32_t k_num_csr_registers = (sizeof(g_register_infos_riscv32_csr) /
                                  sizeof(g_register_infos_riscv32_csr[0]));

  std::vector<DynamicRegisterInfo::Register> registers;
  uint32_t byte_offset = 0;

  // Build dynamic register information for GPR.
  m_gpregset.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                      gpregset.GetByteSize()));
  if (m_gpregset.GetByteSize() >= g_register_infos_riscv32_gpr[0].byte_size) {
    // GPR is available.
    assert((m_gpregset.GetByteSize() /
            g_register_infos_riscv32_gpr[0].byte_size) == k_num_gpr_registers &&
           "GPR has the wrong number of registers!");
    m_gpregset.SetByteOrder(gpregset.GetByteOrder());
    for (auto gpr : g_register_infos_riscv32_gpr) {
      std::vector<uint32_t> value_regs;
      std::vector<uint32_t> invalidate_regs;
      if (gpr.value_regs)
        for (int idx = 0; gpr.value_regs[idx] != LLDB_INVALID_REGNUM; ++idx)
          value_regs.push_back(gpr.value_regs[idx]);
      if (gpr.invalidate_regs)
        for (int idx = 0; gpr.invalidate_regs[idx] != LLDB_INVALID_REGNUM;
             ++idx)
          invalidate_regs.push_back(gpr.invalidate_regs[idx]);
      DynamicRegisterInfo::Register reg{
          lldb_private::ConstString(gpr.name),
          lldb_private::ConstString(gpr.alt_name),
          lldb_private::ConstString("GPR"),
          gpr.byte_size,
          byte_offset,
          gpr.encoding,
          gpr.format,
          gpr.kinds[lldb::eRegisterKindDWARF],
          gpr.kinds[lldb::eRegisterKindEHFrame],
          gpr.kinds[lldb::eRegisterKindGeneric],
          gpr.kinds[lldb::eRegisterKindProcessPlugin],
          value_regs,
          invalidate_regs,
          /* value_reg_offset */ 0,
          gpr.flags_type};

      registers.push_back(reg);
      byte_offset += gpr.byte_size;
    }
  }

  // Build dynamic register information for FPR.
  m_fpregset = getRegset(
      notes, m_reg_infos_up->GetTargetArchitecture().GetTriple(), FPR_Desc);
  if (m_fpregset.GetByteSize() >= g_register_infos_riscv32_fpr[0].byte_size) {
    // GPR is available.
    assert((m_fpregset.GetByteSize() /
            g_register_infos_riscv32_fpr[0].byte_size) == k_num_fpr_registers &&
           "FPR has the wrong number of registers!");
    m_fpregset.SetByteOrder(lldb::eByteOrderLittle);
    for (auto fpr : g_register_infos_riscv32_fpr) {
      std::vector<uint32_t> value_regs;
      std::vector<uint32_t> invalidate_regs;
      if (fpr.value_regs)
        for (int idx = 0; fpr.value_regs[idx] != LLDB_INVALID_REGNUM; ++idx)
          value_regs.push_back(fpr.value_regs[idx]);
      if (fpr.invalidate_regs)
        for (int idx = 0; fpr.invalidate_regs[idx] != LLDB_INVALID_REGNUM;
             ++idx)
          invalidate_regs.push_back(fpr.invalidate_regs[idx]);
      DynamicRegisterInfo::Register reg{
          lldb_private::ConstString(fpr.name),
          lldb_private::ConstString(fpr.alt_name),
          lldb_private::ConstString("FPR"),
          fpr.byte_size,
          byte_offset,
          fpr.encoding,
          fpr.format,
          fpr.kinds[lldb::eRegisterKindDWARF],
          fpr.kinds[lldb::eRegisterKindEHFrame],
          fpr.kinds[lldb::eRegisterKindGeneric],
          fpr.kinds[lldb::eRegisterKindProcessPlugin],
          value_regs,
          invalidate_regs,
          /* value_reg_offset */ 0,
          fpr.flags_type};

      registers.push_back(reg);
      byte_offset += fpr.byte_size;
    }
  }

  // Build dynamic register information for CSR.
  m_csregset =
      getRegset(notes, m_reg_infos_up->GetTargetArchitecture().GetTriple(),
                RISCV32_CSREGMAP_Desc);
  if (m_csregset.GetByteSize() >=
      (sizeof(csr_kv_t::addr) + sizeof(csr_kv_t::val))) {
    // CSR is available.
    m_csregset.SetByteOrder(lldb::eByteOrderLittle);
    lldb::offset_t offset = 0;
    while (m_csregset.BytesLeft(offset)) {
      RegisterInfo csr;
      uint32_t csr_addr = m_csregset.GetU32(&offset);
      if (m_csregset_regnums.size() == k_num_csr_registers) {
        printf("Parsed the permissible number of CSRs (%d) but NT_CSREGMAP has "
               "more! Skipping the remaining CSRs!\n",
               k_num_csr_registers);
        break;
      }
      if (llvm::is_contained(m_csregset_regnums, csr_addr)) {
        printf("Encountered a duplicate CSR while parsing NT_CSREGMAP: %s! "
               "Skipping!\n",
               g_register_infos_riscv32_csr[csr_addr].name);
      } else {
        m_csregset_regnums.push_back(csr_addr);
        csr = g_register_infos_riscv32_csr[csr_addr];
        std::vector<uint32_t> value_regs;
        std::vector<uint32_t> invalidate_regs;
        if (csr.value_regs)
          for (int idx = 0; csr.value_regs[idx] != LLDB_INVALID_REGNUM; ++idx)
            value_regs.push_back(csr.value_regs[idx]);
        if (csr.invalidate_regs)
          for (int idx = 0; csr.invalidate_regs[idx] != LLDB_INVALID_REGNUM;
               ++idx)
            invalidate_regs.push_back(csr.invalidate_regs[idx]);
        DynamicRegisterInfo::Register reg{
            lldb_private::ConstString(csr.name),
            lldb_private::ConstString(csr.alt_name),
            lldb_private::ConstString("CSR"),
            csr.byte_size,
            byte_offset,
            csr.encoding,
            csr.format,
            csr.kinds[lldb::eRegisterKindDWARF],
            csr.kinds[lldb::eRegisterKindEHFrame],
            csr.kinds[lldb::eRegisterKindGeneric],
            csr.kinds[lldb::eRegisterKindProcessPlugin],
            value_regs,
            invalidate_regs,
            /* value_reg_offset */ 0,
            csr.flags_type};
        registers.push_back(reg);
      }
      byte_offset += csr.byte_size;
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
      // |  f0 |
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
