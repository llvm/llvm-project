//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_riscv32.h"
#include "lldb/Register/Arch/lldb-riscv-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"
#include "lldb/lldb-private-types.h"
#include "llvm/ADT/StringMap.h"
#include <cstddef>
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <iterator>

namespace dwarf_regs = lldb_private::registers::riscv::dwarf;
namespace ehframe_regs = lldb_private::registers::riscv::ehframe;
namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

namespace {
const std::array<llvm::ArrayRef<RegisterDescriptor>, 3> &Definitions() {
#define DECLARE_REGISTER_INFOS_RISCV32_STRUCT
#include "Register/Arch/RegisterDescriptions_riscv32.inc"
#undef DECLARE_REGISTER_INFOS_RISCV32_STRUCT

  static const std::array<llvm::ArrayRef<RegisterDescriptor>, 3> definitions = {
      llvm::ArrayRef(g_register_infos_riscv32_le),
      llvm::ArrayRef(g_register_infos_riscv32_csr_patch),
      llvm::ArrayRef(g_register_infos_riscv32_csr_xqciint_patch)};
  return definitions;
}
} // namespace

// Register counts used to validate the architecture tables.
enum {
  k_num_gpr_registers = reg_ids::gpr_last - reg_ids::gpr_first + 1,
  k_num_fpr_registers = reg_ids::fpr_last - reg_ids::fpr_first + 1
};

// RISC-V32 general purpose registers.
static const uint32_t g_gpr_regnums_riscv32[] = {
    reg_ids::gpr_pc,  reg_ids::gpr_ra,  reg_ids::gpr_sp,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,  reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_fp,  reg_ids::gpr_x9,  reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13, reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17, reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21, reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25, reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_x29, reg_ids::gpr_x30, reg_ids::gpr_x31,
    reg_ids::gpr_x0};

static_assert((sizeof g_gpr_regnums_riscv32 /
               sizeof g_gpr_regnums_riscv32[0]) == k_num_gpr_registers,
              "g_gpr_regnums_riscv32 has wrong number of register infos");

// RISC-V32 floating point registers.
static const uint32_t g_fpr_regnums_riscv32[] = {
    reg_ids::fpr_f0,  reg_ids::fpr_f1,  reg_ids::fpr_f2,  reg_ids::fpr_f3,
    reg_ids::fpr_f4,  reg_ids::fpr_f5,  reg_ids::fpr_f6,  reg_ids::fpr_f7,
    reg_ids::fpr_f8,  reg_ids::fpr_f9,  reg_ids::fpr_f10, reg_ids::fpr_f11,
    reg_ids::fpr_f12, reg_ids::fpr_f13, reg_ids::fpr_f14, reg_ids::fpr_f15,
    reg_ids::fpr_f16, reg_ids::fpr_f17, reg_ids::fpr_f18, reg_ids::fpr_f19,
    reg_ids::fpr_f20, reg_ids::fpr_f21, reg_ids::fpr_f22, reg_ids::fpr_f23,
    reg_ids::fpr_f24, reg_ids::fpr_f25, reg_ids::fpr_f26, reg_ids::fpr_f27,
    reg_ids::fpr_f28, reg_ids::fpr_f29, reg_ids::fpr_f30, reg_ids::fpr_f31,
    reg_ids::fpr_fcsr};

static_assert((sizeof g_fpr_regnums_riscv32 /
               sizeof g_fpr_regnums_riscv32[0]) == k_num_fpr_registers,
              "g_fpr_regnums_riscv32 has wrong number of register infos");

// Register groups for RISC-V32.
std::shared_ptr<const RegisterDescription>
RegisterInfo_riscv32::CreateDescription() {
  static const auto description = [] {
    std::vector<RegisterDescriptor> values(Definitions()[0].begin(),
                                           Definitions()[0].end());
    std::vector<RegisterGroup> groups = {
        {"General Purpose Registers", "gpr", {}},
        {"Floating Point Registers", "fpr", {}}};
    for (uint32_t id : llvm::ArrayRef(g_gpr_regnums_riscv32))
      groups[0].registers.push_back(id);
    for (uint32_t id : llvm::ArrayRef(g_fpr_regnums_riscv32))
      groups[1].registers.push_back(id);
    for (auto &reg : values)
      if (reg.id == reg_ids::gpr_x0)
        reg.constant_value.resize(reg.byte_size, 0);
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(values), std::move(groups))));
  }();
  return description;
}

void RegisterInfo_riscv32::BuildCSRegInfos(
    llvm::ArrayRef<std::string> features,
    llvm::SmallVectorImpl<lldb_private::RegisterDescriptor> &cs_reg_infos) {
  cs_reg_infos.clear();

  // Sort and deduplicate to make the list independent of caller ordering.
  llvm::SmallVector<llvm::StringRef> normalized_features(features.begin(),
                                                         features.end());
  llvm::sort(normalized_features);
  normalized_features.erase(llvm::unique(normalized_features),
                            normalized_features.end());

  const uint32_t k_num_csr_registers =
      reg_ids::csr_last - reg_ids::csr_first + 1;
  cs_reg_infos.reserve(k_num_csr_registers);

  // Construct default CS register information. CS register information entries
  // are stored in CSR-address order.
  for (uint32_t reg = 0; reg < k_num_csr_registers; ++reg) {
    lldb_private::RegisterDescriptor csr{};
    csr.name = llvm::formatv("csr_{0:x}", reg).str();
    csr.alt_name = csr.name;
    csr.byte_size = 4;
    csr.encoding = lldb::eEncodingUint;
    csr.format = lldb::eFormatHex;
    csr.numbers[lldb::eRegisterKindEHFrame] =
        ehframe_regs::first_csr + reg;
    csr.numbers[lldb::eRegisterKindDWARF] = dwarf_regs::first_csr + reg;
    csr.numbers[lldb::eRegisterKindGeneric] = LLDB_INVALID_REGNUM;
    csr.numbers[lldb::eRegisterKindProcessPlugin] = LLDB_INVALID_REGNUM;
    csr.id = reg_ids::csr_first + reg;
    csr.numbers[lldb::eRegisterKindLLDB] = csr.id;

    cs_reg_infos.push_back(csr);
  }

  // Patch application is order-dependent; later patches override earlier ones
  // for the same CSR address.
  ConfigureCSRegInfos("default", cs_reg_infos);

  for (const auto &feature : normalized_features)
    ConfigureCSRegInfos(feature, cs_reg_infos);
}

void RegisterInfo_riscv32::ConfigureCSRegInfos(
    llvm::StringRef feature,
    llvm::SmallVectorImpl<lldb_private::RegisterDescriptor> &cs_reg_infos) {
  // Initialized on first use so that the patch set does not depend on the
  // initialization order of globals in other translation units.
  static const llvm::StringMap<llvm::ArrayRef<lldb_private::RegisterDescriptor>>
      g_register_infos_riscv32_csr_patches = {{"default", Definitions()[1]},
                                              {"xqciint", Definitions()[2]}};

  auto it = g_register_infos_riscv32_csr_patches.find(feature);
  if (it == g_register_infos_riscv32_csr_patches.end())
    return;

  for (const auto &csr : it->second) {
    const uint32_t lldb_reg = csr.id;
    if (lldb_reg < reg_ids::csr_first || lldb_reg > reg_ids::csr_last)
      continue;
    uint32_t idx = lldb_reg - reg_ids::csr_first;
    if (idx < cs_reg_infos.size())
      cs_reg_infos[idx] = csr;
  }
}
