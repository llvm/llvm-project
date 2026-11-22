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
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class TestWebAssemblyDisassembly : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

void TestWebAssemblyDisassembly::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestWebAssemblyDisassembly::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}
} // namespace

TEST_F(TestWebAssemblyDisassembly, TestNextBranchStopsAtTailCall) {
  ArchSpec arch("wasm32-unknown-unknown-wasm");

  const unsigned num_of_instructions = 3;
  uint8_t data[] = {
      0x10, 0x00,      // call 0
      0x12, 0x00,      // return_call 0
      0x13, 0x00, 0x00 // return_call_indirect 0
  };

  DisassemblerSP disass_sp = Disassembler::DisassembleBytes(
      arch, nullptr, nullptr, nullptr, nullptr, Address(0x100), &data,
      sizeof(data), num_of_instructions, false);
  ASSERT_TRUE(disass_sp);

  const InstructionList &inst_list = disass_sp->GetInstructionList();
  ASSERT_EQ(num_of_instructions, inst_list.GetSize());

  // Stepping over runs over the call, but not past a tail call, which does not
  // return to the next instruction.
  bool found_calls = false;
  EXPECT_EQ(1u, inst_list.GetIndexOfNextBranchInstruction(
                    0, /*ignore_calls=*/true, &found_calls));
  EXPECT_TRUE(found_calls);
  EXPECT_EQ(2u, inst_list.GetIndexOfNextBranchInstruction(
                    2, /*ignore_calls=*/true, &found_calls));
  EXPECT_FALSE(found_calls);
}
