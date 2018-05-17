//===-- AssemblerTest.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "X86InstrInfo.h"

namespace exegesis {
namespace {

using llvm::MCInstBuilder;
using llvm::X86::EAX;
using llvm::X86::MOV32ri;
using llvm::X86::MOV64ri32;
using llvm::X86::RAX;
using llvm::X86::XOR32rr;

class X86MachineFunctionGeneratorTest
    : public MachineFunctionGeneratorBaseTest {
protected:
  X86MachineFunctionGeneratorTest()
      : MachineFunctionGeneratorBaseTest("x86_64-unknown-linux", "haswell") {}

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
  }
};

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunction) {
  Check(llvm::MCInst(), 0xc3);
}

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunctionXOR32rr) {
  Check(MCInstBuilder(XOR32rr).addReg(EAX).addReg(EAX).addReg(EAX), 0x31, 0xc0,
        0xc3);
}

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunctionMOV64ri) {
  Check(MCInstBuilder(MOV64ri32).addReg(RAX).addImm(42), 0x48, 0xc7, 0xc0, 0x2a,
        0x00, 0x00, 0x00, 0xc3);
}

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunctionMOV32ri) {
  Check(MCInstBuilder(MOV32ri).addReg(EAX).addImm(42), 0xb8, 0x2a, 0x00, 0x00,
        0x00, 0xc3);
}

} // namespace
} // namespace exegesis
