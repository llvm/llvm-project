//===-- TextX86GetControlFlowKind.cpp ------------------------------------------===//

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

class TestGetControlFlowKindx86 : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

protected:
};

void TestGetControlFlowKindx86::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestGetControlFlowKindx86::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}

TEST_F(TestGetControlFlowKindx86, TestX86_64Instruction) {
  ArchSpec arch("x86_64-*-linux");

  const unsigned num_of_instructions = 29;
  uint8_t data[] = {
      0x55,                               // other -- pushq %rbp
      0x48, 0x89, 0xe5,                   // other -- movq %rsp, %rbp

      0xe8, 0xfc, 0xfe, 0xff, 0xff,       // call -- callq 0x4004c0
      0x41, 0xff, 0x14, 0xdc,             // call -- callq *(%r12,%rbx,8)
      0xff, 0x50, 0x18,                   // call -- callq *0x18(%rax)
      0xe8, 0x48, 0x0d, 0x00, 0x00,       // call -- callq 0x94fe0

      0xc3,                               // return -- retq

      0xeb, 0xd3,                         // jump -- jmp 0x92dab
      0xe9, 0x22, 0xff, 0xff, 0xff,       // jump -- jmp 0x933ae
      0xff, 0xe0,                         // jump -- jmpq *%rax
      0xf2, 0xff, 0x25, 0x75, 0xe7, 0x39, 0x00, // jump -- repne jmpq *0x39e775

      0x73, 0xc2,                         // cond jump -- jae 0x9515c
      0x74, 0x1f,                         // cond jump -- je 0x400626
      0x75, 0xea,                         // cond jump -- jne 0x400610
      0x76, 0x10,                         // cond jump -- jbe 0x94d10
      0x77, 0x58,                         // cond jump -- ja 0x1208c8
      0x7e, 0x67,                         // cond jump -- jle 0x92180
      0x78, 0x0b,                         // cond jump -- js 0x92dc3
      0x0f, 0x82, 0x17, 0x01, 0x00, 0x00, // cond jump -- jb 0x9c7b0
      0x0f, 0x83, 0xa7, 0x00, 0x00, 0x00, // cond jump -- jae 0x895c8
      0x0f, 0x84, 0x8c, 0x00, 0x00, 0x00, // cond jump -- je 0x941f0
      0x0f, 0x85, 0x51, 0xff, 0xff, 0xff, // cond jump -- jne 0x8952c
      0x0f, 0x86, 0xa3, 0x02, 0x00, 0x00, // cond jump -- jbe 0x9ae10
      0x0f, 0x87, 0xff, 0x00, 0x00, 0x00, // cond jump -- ja 0x9ab60
      0x0f, 0x8e, 0x7e, 0x00, 0x00, 0x00, // cond jump -- jle 0x92dd8
      0x0f, 0x86, 0xdf, 0x00, 0x00, 0x00, // cond jump -- jbe 0x921b0

      0x0f, 0x05,                         // far call -- syscall

      0x0f, 0x07,                         // far return -- sysret
      0xcf,                               // far return -- interrupt ret
  };

  InstructionControlFlowKind result[] = {
      eInstructionControlFlowKindOther,
      eInstructionControlFlowKindOther,

      eInstructionControlFlowKindCall,
      eInstructionControlFlowKindCall,
      eInstructionControlFlowKindCall,
      eInstructionControlFlowKindCall,

      eInstructionControlFlowKindReturn,

      eInstructionControlFlowKindJump,
      eInstructionControlFlowKindJump,
      eInstructionControlFlowKindJump,
      eInstructionControlFlowKindJump,

      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,
      eInstructionControlFlowKindCondJump,

      eInstructionControlFlowKindFarCall,

      eInstructionControlFlowKindFarReturn,
      eInstructionControlFlowKindFarReturn,
  };

  DisassemblerSP disass_sp;
  Address start_addr(0x100);
  disass_sp =
      Disassembler::DisassembleBytes(arch, nullptr, nullptr, start_addr, &data,
                                    sizeof (data), num_of_instructions, false);

  // If we failed to get a disassembler, we can assume it is because
  // the llvm we linked against was not built with the i386 target,
  // and we should skip these tests without marking anything as failing.

  if (disass_sp) {
    const InstructionList inst_list(disass_sp->GetInstructionList());
    EXPECT_EQ(num_of_instructions, inst_list.GetSize());

    for (size_t i = 0; i < num_of_instructions; ++i) {
      InstructionSP inst_sp;
      inst_sp = inst_list.GetInstructionAtIndex(i);
      InstructionControlFlowKind kind = inst_sp->GetControlFlowKind(arch);
      EXPECT_EQ(kind, result[i]);
    }
  }
}
