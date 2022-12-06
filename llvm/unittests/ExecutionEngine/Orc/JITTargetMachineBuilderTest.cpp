//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "OrcTestCommon.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(ExecutionUtilsTest, JITTargetMachineBuilder) {
  // Tests basic API usage.
  // Bails out on error, as it is valid to run this test without any targets
  // built.

  // Make sure LLVM has been initialized.
  OrcNativeTarget::initialize();

  auto JTMB = cantFail(JITTargetMachineBuilder::detectHost());

  // Test API by performing a bunch of no-ops.
  JTMB.setCPU("");
  JTMB.setRelocationModel(std::nullopt);
  JTMB.setCodeModel(std::nullopt);
  JTMB.setCodeGenOptLevel(CodeGenOpt::None);
  JTMB.addFeatures(std::vector<std::string>());
  SubtargetFeatures &STF = JTMB.getFeatures();
  (void)STF;
  TargetOptions &TO = JTMB.getOptions();
  (void)TO;
  Triple &TT = JTMB.getTargetTriple();
  (void)TT;

  auto TM = JTMB.createTargetMachine();

  if (!TM)
    consumeError(TM.takeError());
  else {
    EXPECT_NE(TM.get(), nullptr)
        << "JITTargetMachineBuilder should return a non-null TargetMachine "
           "on success";
  }
}

} // namespace
