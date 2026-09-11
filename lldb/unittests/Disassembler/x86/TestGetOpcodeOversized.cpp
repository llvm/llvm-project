//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class TestGetOpcodeOversized : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

void TestGetOpcodeOversized::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestGetOpcodeOversized::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}
} // namespace

// A long enough run of redundant prefixes makes the x86 disassembler report an
// instruction whose length is greater than Opcode's fixed-size byte buffer.
TEST_F(TestGetOpcodeOversized, OversizedInstructionIsTruncated) {
  ArchSpec arch("x86_64-*-linux");

  // 20 operand-size (0x66) prefixes followed by a NOP: decodes successfully as
  // a 21-byte instruction, overlowing the 15-byte x86 limit and the 16-byte
  // opcode buffer.
  uint8_t data[21];
  memset(data, 0x66, sizeof(data));
  data[sizeof(data) - 1] = 0x90; // nop
  static_assert(sizeof(data) > Opcode::kMaxByteSize,
                "test input must exceed the opcode buffer");

  Address start_addr(0x100);
  DisassemblerSP disass_sp = Disassembler::DisassembleBytes(
      arch, nullptr, nullptr, nullptr, nullptr, start_addr, &data, sizeof(data),
      /*num_instructions=*/1, false);

  if (!disass_sp)
    return;

  const InstructionList &inst_list = disass_sp->GetInstructionList();
  ASSERT_EQ(1u, inst_list.GetSize());

  // The instruction is still produced, but its recorded opcode is clamped to
  // the buffer capacity rather than overflowing it.
  InstructionSP inst_sp = inst_list.GetInstructionAtIndex(0);
  EXPECT_LE(inst_sp->GetOpcode().GetByteSize(), Opcode::kMaxByteSize);
}

// A normal instruction is unaffected by the truncation logic.
TEST_F(TestGetOpcodeOversized, NormalInstructionIsUnchanged) {
  ArchSpec arch("x86_64-*-linux");

  uint8_t data[] = {0x48, 0x83, 0xec, 0x18}; // subq $0x18, %rsp

  Address start_addr(0x100);
  DisassemblerSP disass_sp = Disassembler::DisassembleBytes(
      arch, nullptr, nullptr, nullptr, nullptr, start_addr, &data, sizeof(data),
      /*num_instructions=*/1, false);

  if (!disass_sp)
    return;

  const InstructionList &inst_list = disass_sp->GetInstructionList();
  ASSERT_EQ(1u, inst_list.GetSize());
  EXPECT_EQ(4u, inst_list.GetInstructionAtIndex(0)->GetOpcode().GetByteSize());
}
