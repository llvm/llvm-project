//===- TargetTransformInfoTest.cpp - TargetTransformInfo unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/DataLayout.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// A target that describes some of its address spaces, but not all of them.
class FakeTTIImpl : public TargetTransformInfoImplBase {
public:
  explicit FakeTTIImpl(const DataLayout &DL)
      : TargetTransformInfoImplBase(DL) {}

  SmallVector<TTI::PointerInfo, 8> getPointerInfos() const override {
    return {TTI::PointerInfo{1, "global"}, TTI::PointerInfo{3, "local"},
            TTI::PointerInfo{7, "private"}};
  }
};

TargetTransformInfo makeFakeTTI(const DataLayout &DL) {
  return TargetTransformInfo(std::make_unique<const FakeTTIImpl>(DL));
}

TEST(TargetTransformInfoTest, PointerInfosDefaultsToEmpty) {
  DataLayout DL("p1:32:32-p2:64:64");
  TargetTransformInfo TTI(DL);

  EXPECT_THAT(TTI.getPointerInfos(), ::testing::IsEmpty());
}

TEST(TargetTransformInfoTest, PointerInfoDefaultsToNullopt) {
  DataLayout DL("p1:32:32");
  TargetTransformInfo TTI(DL);

  EXPECT_FALSE(TTI.getPointerInfo(0).has_value());
}

TEST(TargetTransformInfoTest, PointerInfoFindsDescribedAddrSpace) {
  DataLayout DL("");
  TargetTransformInfo TTI = makeFakeTTI(DL);

  std::optional<TTI::PointerInfo> PI = TTI.getPointerInfo(3);
  ASSERT_TRUE(PI.has_value());
  EXPECT_EQ(PI->AddrSpace, 3u);
  EXPECT_EQ(PI->Name, "local");
}

TEST(TargetTransformInfoTest, PointerInfoFindsEveryReportedAddrSpace) {
  DataLayout DL("p1:32:32");
  TargetTransformInfo TTI = makeFakeTTI(DL);

  for (const TTI::PointerInfo &Reported : TTI.getPointerInfos()) {
    std::optional<TTI::PointerInfo> PI = TTI.getPointerInfo(Reported.AddrSpace);
    ASSERT_TRUE(PI.has_value()) << "address space " << Reported.AddrSpace;
    EXPECT_EQ(PI->AddrSpace, Reported.AddrSpace);
    EXPECT_EQ(PI->Name, Reported.Name);
  }

  EXPECT_FALSE(TTI.getPointerInfo(999).has_value());
}

} // end anonymous namespace
