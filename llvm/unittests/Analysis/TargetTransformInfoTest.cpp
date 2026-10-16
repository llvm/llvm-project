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
class FakeTTIImpl : public TargetTransformInfoImplBase {
public:
  FakeTTIImpl(const DataLayout &DL, SmallVector<unsigned, 8> AddrSpaces)
      : TargetTransformInfoImplBase(DL), AddrSpaces(std::move(AddrSpaces)) {}

  SmallVector<unsigned, 8> getAddressSpaces() const override {
    return AddrSpaces;
  }

private:
  SmallVector<unsigned, 8> AddrSpaces;
};

TargetTransformInfo makeFakeTTI(const DataLayout &DL,
                                SmallVector<unsigned, 8> AddrSpaces) {
  return TargetTransformInfo(
      std::make_unique<const FakeTTIImpl>(DL, std::move(AddrSpaces)));
}

TEST(TargetTransformInfoTest, AddressSpacesDefaultsToEmpty) {
  DataLayout DL("p1:32:32-p2:64:64");
  TargetTransformInfo TTI(DL);

  EXPECT_THAT(TTI.getAddressSpaces(), ::testing::IsEmpty());
}

TEST(TargetTransformInfoTest, AddressSpacesReportsDescribedAddrSpaces) {
  DataLayout DL("p1:32:32");
  TargetTransformInfo TTI = makeFakeTTI(DL, {1, 3, 7});

  EXPECT_THAT(TTI.getAddressSpaces(), ::testing::ElementsAre(1u, 3u, 7u));
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(TargetTransformInfoTest, AddressSpacesOutOfOrderOrDuplicatedIsRejected) {
  DataLayout DL("");
  TargetTransformInfo TTI = makeFakeTTI(DL, {3, 1, 7});

  EXPECT_DEATH(TTI.getAddressSpaces(), "out of order or duplicated");

  TTI = makeFakeTTI(DL, {1, 3, 3});

  EXPECT_DEATH(TTI.getAddressSpaces(), "out of order or duplicated");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

} // end anonymous namespace
