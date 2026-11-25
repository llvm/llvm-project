//===- InlineAsmOperandCommentTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests TargetInstrInfo::createMIROperandComment on inline asm operand flags.
//
// The MIR printer writes inline asm flags in their symbolic form, so it doesn't
// print the comment for them, and a lit test can't check it. Only X86 creates
// "func" flags (function operands of "p" and "*m"), so the test uses X86.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class InlineAsmOperandCommentTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MIRParser> Parser;
  std::unique_ptr<Module> M;

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
  }

  void SetUp() override {
    Triple TT("x86_64-unknown-linux-gnu");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      GTEST_SKIP() << Error;
    TM.reset(T->createTargetMachine(TT, "", "", TargetOptions(), std::nullopt,
                                    std::nullopt, CodeGenOptLevel::Default));

    // One inline asm call per function, each with a single operand: a "p"
    // function operand, a "*m" function operand, and a "*m" memory operand.
    static const char *MIRString = R"MIR(
--- |
  declare void @g()

  define void @test_func_p() {
    call void asm sideeffect "call ${0:P}", "p"(ptr @g)
    ret void
  }

  define void @test_func_m() {
    call void asm sideeffect "call ${0:P}", "*m"(ptr elementtype(void ()) @g)
    ret void
  }

  define void @test_mem_m(ptr %p) {
    call void asm sideeffect "incl $0", "*m"(ptr elementtype(i32) %p)
    ret void
  }
...
---
name: test_func_p
body: |
  bb.0:
    INLINEASM &"call ${0:P}", sideeffect attdialect, func:p ops:5, target-flags(x86-plt) @g, 1, $noreg, 0, $noreg
    RET 0
...
---
name: test_func_m
body: |
  bb.0:
    INLINEASM &"call ${0:P}", sideeffect mayload attdialect, func:m ops:5, target-flags(x86-plt) @g, 1, $noreg, 0, $noreg
    RET 0
...
---
name: test_mem_m
tracksRegLiveness: true
body: |
  bb.0:
    liveins: $rdi
    INLINEASM &"incl $0", sideeffect mayload maystore attdialect, mem:m ops:5, $rdi, 1, $noreg, 0, $noreg
    RET 0
...
)MIR";

    Parser = createMIRParser(MemoryBuffer::getMemBuffer(MIRString), Context);
    ASSERT_TRUE(Parser);
    M = Parser->parseIRModule();
    ASSERT_TRUE(M);
    M->setTargetTriple(TM->getTargetTriple());
    M->setDataLayout(TM->getTargetTriple().computeDataLayout());
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    ASSERT_FALSE(Parser->parseMachineFunctions(*M, *MMI));
  }

  /// Returns the comment for the flag of the only operand of the inline asm
  /// call at the start of function \p Name.
  std::string getFlagComment(StringRef Name) {
    MachineFunction *MF = MMI->getMachineFunction(*M->getFunction(Name));
    const MachineInstr &MI = MF->front().front();
    const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
    return TII->createMIROperandComment(
        MI, MI.getOperand(InlineAsm::MIOp_FirstOperand),
        InlineAsm::MIOp_FirstOperand);
  }
};

// A "func" flag holds a memory constraint code, as a "mem" flag does, and no
// register class. Its comment should give the constraint code.
TEST_F(InlineAsmOperandCommentTest, FuncFlag) {
  EXPECT_EQ(getFlagComment("test_func_p"), "func:p");
  EXPECT_EQ(getFlagComment("test_func_m"), "func:m");
}

TEST_F(InlineAsmOperandCommentTest, MemFlag) {
  EXPECT_EQ(getFlagComment("test_mem_m"), "mem:m");
}

} // end anonymous namespace
