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

  // If we failed to get a disassembler, we can assume it is because
  // the llvm we linked against was not built with the riscv target,
  // and we should skip these tests without marking anything as failing.
  if (!disass_sp)
    return;

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
