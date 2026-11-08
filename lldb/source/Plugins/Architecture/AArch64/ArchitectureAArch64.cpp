//===-- ArchitectureAArch64.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/AArch64/ArchitectureAArch64.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"

#include "llvm/Support/Endian.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ArchitectureAArch64)

void ArchitectureAArch64::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "AArch64-specific algorithms",
                                &ArchitectureAArch64::Create);
}

void ArchitectureAArch64::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureAArch64::Create);
}

std::unique_ptr<Architecture>
ArchitectureAArch64::Create(const ArchSpec &arch) {
  auto machine = arch.GetMachine();
  if (machine != llvm::Triple::aarch64 && machine != llvm::Triple::aarch64_be &&
      machine != llvm::Triple::aarch64_32) {
    return nullptr;
  }
  return std::unique_ptr<Architecture>(new ArchitectureAArch64());
}

static void
UpdateARM64SVERegisterSizes(llvm::MutableArrayRef<RegisterDescriptor> regs,
                             uint64_t vg) {
  // SVE Z register size is vg x 8 bytes.
  uint32_t z_reg_byte_size = vg * 8;

  // The adapter publishes the resized logical values and recalculates the
  // packet layout together, after this callback finishes.
  for (auto &reg : regs) {
    if (reg.value_registers.empty()) {
      if (reg.name.size() > 1 && reg.name[0] == 'z' && isdigit(reg.name[1]))
        reg.byte_size = z_reg_byte_size;
      else if (reg.name.size() > 1 && reg.name[0] == 'p' &&
               isdigit(reg.name[1]))
        reg.byte_size = vg;
      else if (reg.name == "ffr")
        reg.byte_size = vg;
    }
  }
}

static void
UpdateARM64SMERegisterSizes(llvm::MutableArrayRef<RegisterDescriptor> regs,
                             uint64_t svg) {
  for (auto &reg : regs) {
    if (reg.name == "za") {
      // ZA is a register with size (svg*8) * (svg*8). A square essentially.
      reg.byte_size = (svg * 8) * (svg * 8);
    }
  }
}

bool ArchitectureAArch64::ReconfigureRegisterLayout(
    std::shared_ptr<const RegisterLayout> &layout, DataExtractor &reg_data,
    ThreadRegisterContext &reg_context

) const {
  // Once we start to reconfigure registers, we cannot read any of them.
  // So we must read VG and SVG up front.

  const uint64_t fail_value = LLDB_INVALID_ADDRESS;
  std::optional<uint64_t> vg_reg_value;
  const RegisterDescriptor *vg_reg_info = layout->GetDescription()->Find("vg");
  if (vg_reg_info) {
    uint32_t vg_reg_num = vg_reg_info->id;
    uint64_t reg_value =
        reg_context.ReadRegisterAsUnsigned(vg_reg_num, fail_value);
    if (reg_value != fail_value && reg_value <= 32)
      vg_reg_value = reg_value;
  }

  std::optional<uint64_t> svg_reg_value;
  const RegisterDescriptor *svg_reg_info =
      layout->GetDescription()->Find("svg");
  if (svg_reg_info) {
    uint32_t svg_reg_num = svg_reg_info->id;
    uint64_t reg_value =
        reg_context.ReadRegisterAsUnsigned(svg_reg_num, fail_value);
    if (reg_value != fail_value && reg_value <= 32)
      svg_reg_value = reg_value;
  }
  if (!svg_reg_value) {
    const RegisterDescriptor *darwin_svg_reg_info =
        layout->GetDescription()->Find("svl");
    if (darwin_svg_reg_info) {
      uint32_t svg_reg_num = darwin_svg_reg_info->id;
      uint64_t reg_value =
          reg_context.ReadRegisterAsUnsigned(svg_reg_num, fail_value);
      // UpdateARM64SVERegisterSizes and UpdateARM64SMERegisterSizes
      // expect the number of 8-byte granules; darwin provides number of
      // bytes.
      if (reg_value != fail_value && reg_value <= 256) {
        svg_reg_value = reg_value / 8;
        // Apple hardware only implements Streaming SVE mode, so
        // the non-streaming Vector Length is not reported by the
        // kernel. Set both svg and vg to this svl value.
        if (!vg_reg_value)
          vg_reg_value = reg_value / 8;
      }
    }
  }

  if (!vg_reg_value && !svg_reg_value)
    return false;

  if (!vg_reg_value) {
    // This must be an SME only system, so VG == SVG.
    vg_reg_value = svg_reg_value;
  }

  auto resized = ReconfigureRegisterImageLayout(*layout, [&](auto registers) {
    if (vg_reg_value)
      UpdateARM64SVERegisterSizes(registers, *vg_reg_value);
    if (svg_reg_value)
      UpdateARM64SMERegisterSizes(registers, *svg_reg_value);
  });
  if (!resized) {
    llvm::consumeError(resized.takeError());
    return false;
  }

  layout = std::make_shared<RegisterLayout>(std::move(*resized));
  reg_data.SetData(std::make_shared<DataBufferHeap>(layout->GetBankSize(0), 0));
  reg_data.SetByteOrder(reg_context.GetByteOrder());

  return true;
}

bool ArchitectureAArch64::IsValidTrapInstruction(
    llvm::ArrayRef<uint8_t> reference, llvm::ArrayRef<uint8_t> observed) const {
  if (reference.size() < 4 || observed.size() < 4)
    return false;
  auto ref_bytes = llvm::support::endian::read32le(reference.data());
  auto bytes = llvm::support::endian::read32le(observed.data());
  // Only the 11 highest bits define the breakpoint instruction, the others
  // include an immediate value that we will explicitly check against.
  uint32_t mask = 0xFFE00000;
  // Check that the masked bytes match the reference, but also check that the
  // immediate in the instruction is the default output by llvm.debugtrap.
  // The reference has the immediate set as all-zero, so mask and check here.
  return (ref_bytes == (bytes & mask)) && ((bytes & ~mask) >> 5 == 0xF000);
}
