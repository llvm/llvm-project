//===------------------ DLangDemangleTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <utility>

struct DLangDemangleTestFixture
    : public testing::TestWithParam<std::pair<const char *, const char *>> {
  char *Demangled;

  void SetUp() override { Demangled = llvm::dlangDemangle(GetParam().first); }

  void TearDown() override { std::free(Demangled); }
};

TEST_P(DLangDemangleTestFixture, DLangDemangleTest) {
  EXPECT_STREQ(Demangled, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(DLangDemangleTest, DLangDemangleTestFixture,
                         testing::Values(std::make_pair("_Dmain", "D main"),
                                         std::make_pair(nullptr, nullptr),
                                         std::make_pair("_Z", nullptr),
                                         std::make_pair("_DDD", nullptr)));
