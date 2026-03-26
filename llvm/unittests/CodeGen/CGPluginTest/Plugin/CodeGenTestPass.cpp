//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenTestPass.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "codegen-test"
#define CODEGEN_TEST_NAME "CodeGen Test Pass"

CodeGenTest::CodeGenTest() : MachineFunctionPass(ID) {}

bool CodeGenTest::runOnMachineFunction(MachineFunction &MF) {
  outs() << CODEGEN_TEST_NAME << " running on " << MF.getName()
         << "\n"; // used for the lit test
  if (RunCallback)
    RunCallback();
  return true;
}

StringRef CodeGenTest::getPassName() const { return CODEGEN_TEST_NAME; }

char CodeGenTest::ID = 0;
std::function<void()> CodeGenTest::RunCallback;

INITIALIZE_PASS(CodeGenTest, DEBUG_TYPE, CODEGEN_TEST_NAME, false, false)
