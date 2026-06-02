//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64Subtarget.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class AArch64CacheLineSizeTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
  }

  unsigned getCacheLineSizeForCPU(StringRef CPU) {
    std::string Error;
    Triple TT("aarch64--");
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    std::unique_ptr<TargetMachine> TM(
        T->createTargetMachine(TT, CPU, "", TargetOptions(), std::nullopt));
    AArch64Subtarget ST(TT, CPU, CPU, TM->getTargetFeatureString(), *TM, true);
    return ST.getCacheLineSize();
  }
};

TEST_F(AArch64CacheLineSizeTest, IsCorrect) {
  EXPECT_EQ(getCacheLineSizeForCPU("generic"), 0u);
  EXPECT_EQ(getCacheLineSizeForCPU("a64fx"), 256u);
  EXPECT_EQ(getCacheLineSizeForCPU("ampere1"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("apple-m5"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("carmel"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("cortex-a57"), 0u);
  EXPECT_EQ(getCacheLineSizeForCPU("cortex-a725"), 0u);
  EXPECT_EQ(getCacheLineSizeForCPU("cortex-x4"), 0u);
  EXPECT_EQ(getCacheLineSizeForCPU("cortex-x925"), 0u);
  EXPECT_EQ(getCacheLineSizeForCPU("falkor"), 128u);
  EXPECT_EQ(getCacheLineSizeForCPU("grace"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("kryo"), 128u);
  EXPECT_EQ(getCacheLineSizeForCPU("neoverse-v1"), 0u);
  EXPECT_EQ(getCacheLineSizeForCPU("neoverse-v3"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("olympus"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("oryon-1"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("thunderx"), 128u);
  EXPECT_EQ(getCacheLineSizeForCPU("thunderx2t99"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("thunderx3t110"), 64u);
  EXPECT_EQ(getCacheLineSizeForCPU("tsv110"), 64u);
}

} // namespace
