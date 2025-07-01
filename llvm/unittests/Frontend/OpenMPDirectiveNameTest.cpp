//===- llvm/unittests/Frontend/OpenMPDirectiveNameTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::omp;

const DenseMap<Directive, StringRef> &Expected52() {
  static const DenseMap<Directive, StringRef> Names{
      {OMPD_begin_declare_target, "begin declare target"},
      {OMPD_begin_declare_variant, "begin declare variant"},
      {OMPD_cancellation_point, "cancellation point"},
      {OMPD_declare_mapper, "declare mapper"},
      {OMPD_declare_reduction, "declare reduction"},
      {OMPD_declare_simd, "declare simd"},
      {OMPD_declare_target, "declare target"},
      {OMPD_declare_variant, "declare variant"},
      {OMPD_end_declare_target, "end declare target"},
      {OMPD_end_declare_variant, "end declare variant"},
      {OMPD_target_data, "target data"},
      {OMPD_target_enter_data, "target enter data"},
      {OMPD_target_exit_data, "target exit data"},
      {OMPD_target_update, "target update"},
  };
  return Names;
}

const DenseMap<Directive, StringRef> &Expected60() {
  static const DenseMap<Directive, StringRef> Names{
      {OMPD_begin_declare_target, "begin declare_target"},
      {OMPD_begin_declare_variant, "begin declare_variant"},
      {OMPD_cancellation_point, "cancellation_point"},
      {OMPD_declare_mapper, "declare_mapper"},
      {OMPD_declare_reduction, "declare_reduction"},
      {OMPD_declare_simd, "declare_simd"},
      {OMPD_declare_target, "declare_target"},
      {OMPD_declare_variant, "declare_variant"},
      {OMPD_end_declare_target, "end declare_target"},
      {OMPD_end_declare_variant, "end declare_variant"},
      {OMPD_target_data, "target_data"},
      {OMPD_target_enter_data, "target_enter_data"},
      {OMPD_target_exit_data, "target_exit_data"},
      {OMPD_target_update, "target_update"},
  };
  return Names;
}

class VersionTest : public testing::TestWithParam<unsigned> {
public:
  void SetUp() override {
    Version = GetParam();

    if (Version < 60)
      KindToName = &Expected52();
    else
      KindToName = &Expected60();
  }

  const DenseMap<Directive, StringRef> *KindToName;
  unsigned Version;
};

INSTANTIATE_TEST_SUITE_P(OpenMPDirectiveNames, VersionTest,
                         testing::ValuesIn(getOpenMPVersions()));

TEST_P(VersionTest, DirectiveName) {
  for (auto [Kind, Name] : *KindToName)
    ASSERT_EQ(Name, getOpenMPDirectiveName(Kind, Version));
}

TEST(OpenMPDirectiveNames, DirectiveKind52) {
  for (auto [Kind, Name] : Expected52()) {
    auto [K, R] = getOpenMPDirectiveKindAndVersions(Name);
    ASSERT_EQ(K, Kind);
    // Expect the name to be valid in 5.2, but not in 6.0.
    EXPECT_TRUE(52 <= R.Max && R.Max < 60);
  }
}

TEST(OpenMPDirectiveNames, DirectiveKind60) {
  for (auto [Kind, Name] : Expected60()) {
    auto [K, R] = getOpenMPDirectiveKindAndVersions(Name);
    ASSERT_EQ(K, Kind);
    // Expect the name to be valid in 6.0 and later.
    EXPECT_TRUE(60 <= R.Min);
  }
}
