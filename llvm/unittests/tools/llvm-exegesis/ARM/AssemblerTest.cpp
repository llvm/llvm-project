//===-- AssemblerTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "ARMInstrInfo.h"

namespace llvm {
namespace exegesis {
namespace {

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
  Check({}, MCInst(), 0x1e, 0xff, 0x2f, 0xe1);
}

TEST_F(ARMMachineFunctionGeneratorTest, DISABLED_JitFunctionADDrr) {
  Check({{ARM::R0, APInt()}},
        MCInstBuilder(ARM::ADDrr)
            .addReg(ARM::R0)
            .addReg(ARM::R0)
            .addReg(ARM::R0)
            .addImm(ARMCC::AL)
            .addReg(0)
            .addReg(0),
        0x00, 0x00, 0x80, 0xe0, 0x1e, 0xff, 0x2f, 0xe1);
}

} // namespace
} // namespace exegesis
} // namespace llvm
