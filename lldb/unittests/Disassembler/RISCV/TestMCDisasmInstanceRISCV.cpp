//===-- TestMCDisasmInstanceRISCV.cpp -------------------------------------===//

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
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/StreamString.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class TestMCDisasmInstanceRISCV : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

protected:
};

void TestMCDisasmInstanceRISCV::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestMCDisasmInstanceRISCV::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}
} // namespace

TEST_F(TestMCDisasmInstanceRISCV, TestRISCV32Instruction) {
  ArchSpec arch("riscv32-*-linux");

  const unsigned num_of_instructions = 5;
  uint8_t data[] = {
      0xef, 0x00, 0x00, 0x00, // call -- jal x1, 0
      0xe7, 0x00, 0x00, 0x00, // call -- jalr x1, x0, 0
      0x6f, 0x00, 0x00, 0x00, // jump -- jal x0, 0
      0x67, 0x00, 0x00, 0x00, // jump -- jalr x0, x0, 0
      0x67, 0x80, 0x00, 0x00  // ret  -- jalr x0, x1, 0
  };

  DisassemblerSP disass_sp;
  Address start_addr(0x100);
  disass_sp = Disassembler::DisassembleBytes(
      arch, nullptr, nullptr, nullptr, nullptr, start_addr, &data, sizeof(data),
      num_of_instructions, false);

  const InstructionList inst_list(disass_sp->GetInstructionList());
  EXPECT_EQ(num_of_instructions, inst_list.GetSize());

  InstructionSP inst_sp;
  inst_sp = inst_list.GetInstructionAtIndex(0);
  EXPECT_TRUE(inst_sp->IsCall());
  EXPECT_TRUE(inst_sp->DoesBranch());

  inst_sp = inst_list.GetInstructionAtIndex(1);
  EXPECT_TRUE(inst_sp->IsCall());
  EXPECT_TRUE(inst_sp->DoesBranch());

  inst_sp = inst_list.GetInstructionAtIndex(2);
  EXPECT_FALSE(inst_sp->IsCall());
  EXPECT_TRUE(inst_sp->DoesBranch());

  inst_sp = inst_list.GetInstructionAtIndex(3);
  EXPECT_FALSE(inst_sp->IsCall());
  EXPECT_TRUE(inst_sp->DoesBranch());

  inst_sp = inst_list.GetInstructionAtIndex(4);
  EXPECT_FALSE(inst_sp->IsCall());
  EXPECT_TRUE(inst_sp->DoesBranch());
}

TEST_F(TestMCDisasmInstanceRISCV, TestOpcodeBytePrinter) {
  ArchSpec arch("riscv32-*-linux");

  const unsigned num_of_instructions = 7;
  // clang-format off
  uint8_t data[] = {
      0x41, 0x11,             // addi   sp, sp, -0x10
      0x06, 0xc6,             // sw     ra, 0xc(sp)
      0x23, 0x2a, 0xa4, 0xfe, // sw     a0, -0xc(s0)
      0x23, 0x28, 0xa4, 0xfe, // sw     a0, -0x10(s0)
      0x22, 0x44,             // lw     s0, 0x8(sp)

      0x3f, 0x00, 0x40, 0x09, // Fake 64-bit instruction
      0x20, 0x00, 0x20, 0x00,

      0x1f, 0x02,             // 48 bit xqci.e.li rd=8 imm=0x1000
      0x00, 0x00, 
      0x00, 0x10,
  };
  // clang-format on

  // clang-format off
  const char *expected_outputs[] = {
    "1141",
    "c606",
    "fea42a23",
    "fea42823",
    "4422",
    "0940003f 00200020",
    "021f 0000 1000"
  };
  // clang-format on
  const unsigned num_of_expected_outputs =
      sizeof(expected_outputs) / sizeof(char *);

  EXPECT_EQ(num_of_instructions, num_of_expected_outputs);

  DisassemblerSP disass_sp;
  Address start_addr(0x100);
  disass_sp = Disassembler::DisassembleBytes(
      arch, nullptr, nullptr, nullptr, nullptr, start_addr, &data, sizeof(data),
      num_of_instructions, false);

  const InstructionList inst_list(disass_sp->GetInstructionList());
  EXPECT_EQ(num_of_instructions, inst_list.GetSize());

  for (size_t i = 0; i < num_of_instructions; i++) {
    InstructionSP inst_sp;
    StreamString s;
    inst_sp = inst_list.GetInstructionAtIndex(i);
    inst_sp->GetOpcode().Dump(&s, 1);
    ASSERT_STREQ(s.GetString().str().c_str(), expected_outputs[i]);
  }
}
