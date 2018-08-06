//===-- AssemblerTest.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "ARMInstrInfo.h"

namespace exegesis {
namespace {

using llvm::MCInstBuilder;

class ARMMachineFunctionGeneratorTest
    : public MachineFunctionGeneratorBaseTest {
protected:
  ARMMachineFunctionGeneratorTest()
      : MachineFunctionGeneratorBaseTest("armv7-none-linux-gnueabi", "") {}

  static void SetUpTestCase() {
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTargetMC();
    LLVMInitializeARMTarget();
    LLVMInitializeARMAsmPrinter();
  }
};

TEST_F(ARMMachineFunctionGeneratorTest, DISABLED_JitFunction) {
  Check(llvm::MCInst(), 0x1e, 0xff, 0x2f, 0xe1);
}

TEST_F(ARMMachineFunctionGeneratorTest, DISABLED_JitFunctionADDrr) {
  Check(MCInstBuilder(llvm::ARM::ADDrr)
            .addReg(llvm::ARM::R0)
            .addReg(llvm::ARM::R0)
            .addReg(llvm::ARM::R0)
            .addImm(llvm::ARMCC::AL)
            .addReg(0)
            .addReg(0),
        0x00, 0x00, 0x80, 0xe0, 0x1e, 0xff, 0x2f, 0xe1);
}

} // namespace
} // namespace exegesis
