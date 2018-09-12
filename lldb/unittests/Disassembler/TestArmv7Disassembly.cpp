//===-- TestArmv7Disassembly.cpp ------------------------------------*- C++
//-*-===//

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Target/ExecutionContext.h"

#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestArmv7Disassembly : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  //  virtual void SetUp() override { }
  //  virtual void TearDown() override { }

protected:
};

void TestArmv7Disassembly::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestArmv7Disassembly::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}

TEST_F(TestArmv7Disassembly, TestCortexFPDisass) {
  ArchSpec arch("armv7em--");

  const unsigned num_of_instructions = 3;
  uint8_t data[] = {
      0x00, 0xee, 0x10, 0x2a, // 0xee002a10 :  vmov   s0, r2
      0xb8, 0xee, 0xc0, 0x0b, // 0xeeb80bc0 :  vcvt.f64.s32 d0, s0
      0xb6, 0xee, 0x00, 0x0a, // 0xeeb60a00 :  vmov.f32 s0, #5.000000e-01
  };

  // these can be disassembled by hand with llvm-mc, e.g.
  //
  // 0x00, 0xee, 0x10, 0x2a, // 0xee002a10 :  vmov   s0, r2
  //
  // echo 0x00 0xee 0x10  0x2a | llvm-mc -arch thumb -disassemble -mattr=+fp-armv8
	//       vmov s0, r2

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
    ASSERT_STREQ ("vmov", mnemonic);

    inst_sp = inst_list.GetInstructionAtIndex (1);
    mnemonic = inst_sp->GetMnemonic(&exe_ctx);
    ASSERT_STREQ ("vcvt.f64.s32", mnemonic);

    inst_sp = inst_list.GetInstructionAtIndex (2);
    mnemonic = inst_sp->GetMnemonic(&exe_ctx);
    ASSERT_STREQ ("vmov.f32", mnemonic);
  }
}
