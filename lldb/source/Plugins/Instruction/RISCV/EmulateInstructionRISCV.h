//===-- EmulateInstructionRISCV.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_INSTRUCTION_RISCV_EMULATEINSTRUCTIONRISCV_H
#define LLDB_SOURCE_PLUGINS_INSTRUCTION_RISCV_EMULATEINSTRUCTIONRISCV_H

#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {

constexpr uint32_t DecodeRD(uint32_t inst) { return (inst & 0xF80) >> 7; }
constexpr uint32_t DecodeRS1(uint32_t inst) { return (inst & 0xF8000) >> 15; }
constexpr uint32_t DecodeRS2(uint32_t inst) { return (inst & 0x1F00000) >> 20; }

class EmulateInstructionRISCV;

struct InstrPattern {
  const char *name;
  /// Bit mask to check the type of a instruction (B-Type, I-Type, J-Type, etc.)
  uint32_t type_mask;
  /// Characteristic value after bitwise-and with type_mask.
  uint32_t eigen;
  bool (*exec)(EmulateInstructionRISCV &emulator, uint32_t inst,
               bool ignore_cond);
};

class EmulateInstructionRISCV : public EmulateInstruction {
public:
  static llvm::StringRef GetPluginNameStatic() { return "riscv"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Emulate instructions for the RISC-V architecture.";
  }

  static bool SupportsThisInstructionType(InstructionType inst_type) {
    switch (inst_type) {
    case eInstructionTypeAny:
    case eInstructionTypePCModifying:
      return true;
    case eInstructionTypePrologueEpilogue:
    case eInstructionTypeAll:
      return false;
    }
    llvm_unreachable("Fully covered switch above!");
  }

  static bool SupportsThisArch(const ArchSpec &arch);

  static lldb_private::EmulateInstruction *
  CreateInstance(const lldb_private::ArchSpec &arch, InstructionType inst_type);

  static void Initialize();

  static void Terminate();

public:
  EmulateInstructionRISCV(const ArchSpec &arch) : EmulateInstruction(arch) {}

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  bool SupportsEmulatingInstructionsOfType(InstructionType inst_type) override {
    return SupportsThisInstructionType(inst_type);
  }

  bool SetTargetTriple(const ArchSpec &arch) override;
  bool ReadInstruction() override;
  bool EvaluateInstruction(uint32_t options) override;
  bool TestEmulation(Stream *out_stream, ArchSpec &arch,
                     OptionValueDictionary *test_data) override;
  bool GetRegisterInfo(lldb::RegisterKind reg_kind, uint32_t reg_num,
                       RegisterInfo &reg_info) override;

  lldb::addr_t ReadPC(bool *success);
  bool WritePC(lldb::addr_t pc);

  const InstrPattern *Decode(uint32_t inst);
  bool DecodeAndExecute(uint32_t inst, bool ignore_cond);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_INSTRUCTION_RISCV_EMULATEINSTRUCTIONRISCV_H
