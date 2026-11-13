//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_riscv32.h"

#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <cstdint>
#include <stddef.h>

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/SubtargetFeature.h"

namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

llvm::Expected<std::unique_ptr<RegisterContextELFCore_riscv32>>
RegisterContextELFCore_riscv32::Create(Thread &thread, const ArchSpec &arch,
                                         const DataExtractor &gpregset,
                                         llvm::ArrayRef<CoreNote> notes) {
  switch (arch.GetTriple().getOS()) {
  case llvm::Triple::UnknownOS:
    // The packed GPR/FPR and sparse CSR notes describe a bare-metal target.
    return std::unique_ptr<RegisterContextELFCore_riscv32>(
        new RegisterContextELFCore_riscv32(thread, arch, gpregset, notes));
  default:
    return llvm::createStringError("unsupported OS for RISC-V32 ELF registers");
  }
}

RegisterContextELFCore_riscv32::RegisterContextELFCore_riscv32(
    Thread &thread, const ArchSpec &arch, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes)
    : ThreadRegisterContext(thread, 0) {
  auto cpu = RegisterInfo_riscv32::CreateDescription();
  std::vector<RegisterDescriptor> g_register_infos_riscv32_gpr,
      g_register_infos_riscv32_fpr;
  for (size_t set_index = 0; set_index < cpu->GetGroups().size(); ++set_index) {
    const auto &set = cpu->GetGroups()[set_index];
    auto &values = set_index == 0 ? g_register_infos_riscv32_gpr
                                  : g_register_infos_riscv32_fpr;
    for (RegisterID id : set.registers)
      values.push_back(*cpu->Find(id));
  }
  const auto k_num_gpr_registers = g_register_infos_riscv32_gpr.size();
  const auto k_num_fpr_registers = g_register_infos_riscv32_fpr.size();
  llvm::SmallVector<std::string> features;
  GetFeatures(features);
  llvm::SmallVector<lldb_private::RegisterDescriptor> reg_infos_riscv32_csr;
  RegisterInfo_riscv32::BuildCSRegInfos(features, reg_infos_riscv32_csr);
  uint32_t k_num_csr_registers = reg_infos_riscv32_csr.size();
  const ArchSpec &target_arch = arch;
  const llvm::Triple triple = target_arch.GetTriple();
  const lldb::ByteOrder byte_order = target_arch.GetByteOrder();

  std::vector<RegisterDescriptor> registers;
  std::vector<RegisterGroup> groups{
      {"GPR", {}, {}}, {"FPR", {}, {}}, {"CSR", {}, {}}};
  auto append = [&](RegisterDescriptor value, size_t group) {
    value.numbers[lldb::eRegisterKindLLDB] = registers.size();
    groups[group].registers.push_back(value.id);
    registers.push_back(std::move(value));
  };
  std::map<RegisterID, uint64_t> csr_offsets;

  // Build dynamic register information for GPR.
  m_gpregset.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                      gpregset.GetByteSize()));
  if (m_gpregset.GetByteSize() >= g_register_infos_riscv32_gpr[0].byte_size) {
    // A note may omit a synthetic zero slot or end before all registers.
    m_gpregset.SetByteOrder(gpregset.GetByteOrder());
    for (const auto &gpr : g_register_infos_riscv32_gpr)
      append(gpr, 0);
  }

  // Build dynamic register information for FPR.
  m_fpregset = getRegset(notes, triple, FPR_Desc);
  if (m_fpregset.GetByteSize() >= g_register_infos_riscv32_fpr[0].byte_size) {
    // A note may omit a synthetic zero slot or end before all registers.
    m_fpregset.SetByteOrder(byte_order);
    for (const auto &fpr : g_register_infos_riscv32_fpr)
      append(fpr, 1);
  }

  // Build dynamic register information for CSR.
  m_csregset = getRegset(notes, triple, RISCV32_CSREGMAP_Desc);
  if (m_csregset.GetByteSize() >=
      (sizeof(csr_kv_t::addr) + sizeof(csr_kv_t::val))) {
    // CSR is available.
    m_csregset.SetByteOrder(byte_order);
    lldb::offset_t offset = 0;
    std::vector<uint32_t> csregset_regnums = {};
    while (m_csregset.BytesLeft(offset) >= sizeof(csr_kv_t)) {
      uint32_t csr_addr = m_csregset.GetU32(&offset);
      if (csr_addr >= k_num_csr_registers) {
        (void)m_csregset.GetU32(&offset);
        continue;
      }
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
                          reg_infos_riscv32_csr[csr_addr].name));
      } else {
        csregset_regnums.push_back(csr_addr);
        const RegisterDescriptor &csr = reg_infos_riscv32_csr[csr_addr];
        csr_offsets[csr.id] = offset;
        auto dynamic = csr;
        // FCSR can be present both in the FP note and in the CSR map. Keep
        // both source locations addressable without duplicating display names.
        if (llvm::any_of(registers, [&](const auto &reg) {
              return reg.name == dynamic.name;
            })) {
          dynamic.name =
              !dynamic.alt_name.empty() && dynamic.alt_name != dynamic.name
                  ? dynamic.alt_name
                  : llvm::formatv("csr_{0:x}", csr_addr).str();
          dynamic.alt_name = csr.name;
        }
        append(std::move(dynamic), 2);
      }
      // Consume and skip the CSR value to advance to the next entry.
      (void)m_csregset.GetU32(&offset);
    }
  }

  auto description = std::make_shared<RegisterDescription>(llvm::cantFail(
      RegisterDescription::Create(std::move(registers), std::move(groups))));
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, k_num_gpr_registers * 4));
  llvm::cantFail(layout.AddBank(1, k_num_fpr_registers * 4));
  llvm::cantFail(layout.AddBank(2, m_csregset.GetByteSize()));
  for (const auto &reg : description->GetRegisters()) {
    RegisterLocation location;

    if (reg.id == reg_ids::gpr_x0)
      continue;
    {
      location.kind = RegisterLocation::Stored;
      if (reg.id < reg_ids::fpr_first)
        location.pieces.push_back(
            {0, (reg.id - reg_ids::gpr_first) * 4, reg.byte_size});
      else if (reg.id <= reg_ids::fpr_last)
        location.pieces.push_back(
            {1, (reg.id - reg_ids::fpr_first) * 4, reg.byte_size});
      else
        location.pieces.push_back({2, csr_offsets.at(reg.id), reg.byte_size});
    }
    llvm::cantFail(layout.Add(reg.id, std::move(location)));
  }
  SetRegisterLayout(llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout))));
}

RegisterContextELFCore_riscv32::~RegisterContextELFCore_riscv32() = default;

void RegisterContextELFCore_riscv32::InvalidateAllRegisters() {}

llvm::Expected<RegisterValue>
RegisterContextELFCore_riscv32::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (!reg_info)
    return llvm::createStringError("register value is unavailable");
  auto *reg = GetDescription()->Find(reg_info->id);
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  auto result = ReadBankRegister(
      reg->id, {{0, {m_gpregset.GetDataStart(), m_gpregset.GetByteSize()}},
                {1, {m_fpregset.GetDataStart(), m_fpregset.GetByteSize()}},
                {2, {m_csregset.GetDataStart(), m_csregset.GetByteSize()}}});
  return result;
}

llvm::Error
RegisterContextELFCore_riscv32::WriteRegister(RegisterID id,
                                                const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

void RegisterContextELFCore_riscv32::GetFeatures(
    llvm::SmallVectorImpl<std::string> &features) const {
  Log *log = GetLog(LLDBLog::Process);
  features.clear();

  if (!m_thread.IsValid())
    return;

  auto process_sp = m_thread.GetProcess();
  if (!process_sp)
    return;

  auto target = process_sp->CalculateTarget();
  if (!target)
    return;

  auto module_sp = target->GetExecutableModule();
  if (!module_sp)
    return;

  // ObjectFileELF::ParseRISCVAttributes should've already parsed features and
  // stored them in the module's ArchSpec via ArchSpec::SetSubtargetFeatures.
  std::vector<std::string> raw_features =
      module_sp->GetArchitecture().GetSubtargetFeatures().getFeatures();

  if (raw_features.empty()) {
    LLDB_LOG(log, "no RISC-V subtarget feature found in module ArchSpec");
    return;
  }

  features.reserve(raw_features.size());
  for (const auto &feature : raw_features) {
    if (feature.empty() || !llvm::SubtargetFeatures::isEnabled(feature))
      continue;
    auto stripped = llvm::SubtargetFeatures::StripFlag(feature);
    if (!stripped.empty())
      features.emplace_back(stripped);
  }
}
