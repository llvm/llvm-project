//===---EmulateInstructionLoongArch.h--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_INSTRUCTION_LOONGARCH_EMULATEINSTRUCTIONLOONGARCH_H
#define LLDB_SOURCE_PLUGINS_INSTRUCTION_LOONGARCH_EMULATEINSTRUCTIONLOONGARCH_H

#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {

class EmulateInstructionLoongArch : public EmulateInstruction {
public:
  static llvm::StringRef GetPluginNameStatic() { return "LoongArch"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Emulate instructions for the LoongArch architecture.";
  }

  static bool SupportsThisInstructionType(InstructionType inst_type) {
    return inst_type == eInstructionTypePCModifying;
  }

  static bool SupportsThisArch(const ArchSpec &arch);

  static lldb_private::EmulateInstruction *
  CreateInstance(const lldb_private::ArchSpec &arch, InstructionType inst_type);

  static void Initialize();

  static void Terminate();

public:
  EmulateInstructionLoongArch(const ArchSpec &arch)
      : EmulateInstruction(arch) {}

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  bool SupportsEmulatingInstructionsOfType(InstructionType inst_type) override {
    return SupportsThisInstructionType(inst_type);
  }

  bool SetTargetTriple(const ArchSpec &arch) override;
  bool ReadInstruction() override;
  bool EvaluateInstruction(uint32_t options) override;
  bool TestEmulation(Stream *out_stream, ArchSpec &arch,
                     OptionValueDictionary *test_data) override;

  llvm::Optional<RegisterInfo> GetRegisterInfo(lldb::RegisterKind reg_kind,
                                               uint32_t reg_num) override;
  lldb::addr_t ReadPC(bool *success);
  bool WritePC(lldb::addr_t pc);

private:
  struct Opcode {
    uint32_t mask;
    uint32_t value;
    bool (EmulateInstructionLoongArch::*callback)(uint32_t opcode);
    const char *name;
  };

  Opcode *GetOpcodeForInstruction(uint32_t inst);

  bool EmulateNonJMP(uint32_t inst);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_INSTRUCTION_LOONGARCH_EMULATEINSTRUCTIONLOONGARCH_H
