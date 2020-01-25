//===-- TestArm64Disassembly.cpp ------------------------------------------===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Target/ExecutionContext.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestArm64Disassembly : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  //  virtual void SetUp() override { }
  //  virtual void TearDown() override { }

protected:
};

void TestArm64Disassembly::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestArm64Disassembly::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}

TEST_F(TestArm64Disassembly, TestArmv81Instruction) {
  ArchSpec arch("arm64-apple-ios");

  const unsigned num_of_instructions = 2;
  uint8_t data[] = {
      0xff, 0x43, 0x00, 0xd1, // 0xd10043ff :  sub    sp, sp, #0x10
      0x62, 0x7c, 0xa1, 0xc8, // 0xc8a17c62 :  cas    x1, x2, [x3] (cas defined in ARM v8.1 & newer)
  };

  DisassemblerSP disass_sp;
  Address start_addr(0x100);
  disass_sp = Disassembler::DisassembleBytes(arch, nullptr, nullptr, start_addr,
                                 &data, sizeof (data), num_of_instructions, false);

  // If we failed to get a disassembler, we can assume it is because
  // the llvm we linked against was not built with the ARM target,
  // and we should skip these tests without marking anything as failing.

  if (disass_sp) {
    const InstructionList inst_list (disass_sp->GetInstructionList());
    EXPECT_EQ (num_of_instructions, inst_list.GetSize());

    InstructionSP inst_sp;
    const char *mnemonic;
    ExecutionContext exe_ctx (nullptr, nullptr, nullptr);
    inst_sp = inst_list.GetInstructionAtIndex (0);
    mnemonic = inst_sp->GetMnemonic(&exe_ctx);
    ASSERT_STREQ ("sub", mnemonic);

    inst_sp = inst_list.GetInstructionAtIndex (1);
    mnemonic = inst_sp->GetMnemonic(&exe_ctx);
    ASSERT_STREQ ("cas", mnemonic);
  }
}
