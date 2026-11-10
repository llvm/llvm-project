//===- SPIRVProcessorTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/IntelGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

class SPIRVProcessorTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
  }

  void SetUp() override {
    Triple TT("spirv64-unknown-unknown");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      GTEST_SKIP();
    STI.reset(T->createMCSubtargetInfo(TT, "", ""));
    ASSERT_TRUE(STI);
  }

  std::unique_ptr<MCSubtargetInfo> STI;
};

// Intel IGCA target tests:
//
// SPIRV.td lists the Intel IGCA targets by hand, so check that it agrees with
// IntelGPUTargetParser.def.

// Every IGCA target that a device in IntelGPUTargetParser.def can run is a
// SPIR-V processor.
TEST_F(SPIRVProcessorTest, EveryIGCATargetIsAProcessor) {
  SmallVector<StringRef> IGCATargets;
  IntelGPU::fillValidIGCATargetList(IGCATargets);
  ASSERT_FALSE(IGCATargets.empty());
  for (StringRef Name : IGCATargets)
    EXPECT_TRUE(STI->isCPUStringValid(Name))
        << Name << " is missing from SPIRV.td";
}

// Every IGCA processor in SPIRV.td is still a valid IGCA target.
TEST_F(SPIRVProcessorTest, EveryIGCAProcessorIsAnIGCATarget) {
  for (const SubtargetSubTypeKV &Proc : STI->getAllProcessorDescriptions()) {
    StringRef Name = Proc.key();
    if (Name.starts_with("igca_"))
      EXPECT_TRUE(IntelGPU::parseIGCATarget(Name))
          << Name << " in SPIRV.td is not a valid IGCA target";
  }
}

} // namespace
