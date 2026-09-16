//===- RTTICrossDylibTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Confirms that RTTI identity (isA<>, and casting on the strength of it)
// survives a real cross-library boundary: RTTICrossDylibTestLib links its
// own copy of orc-rt-bedrock, so the CrossDylibTestError it constructs has a
// LibraryID distinct from this binary's. That forces isA<>() through the
// strcmp fallback (see RTTIRoot::isA in orc-rt/support/RTTI.h) rather than
// the same-library pointer-equality fast path.
//
// CoreTests links RTTICrossDylibTestLib directly (see CMakeLists.txt), so
// the two get their own LibraryIDs the same way any two independently-linked
// consumers of orc-rt would; nothing here needs to touch dlopen/dlsym.
//
//===----------------------------------------------------------------------===//

#include "../Inputs/RTTICrossDylibTestError.h"
#include "../Inputs/RTTICrossDylibTestLib.h"
#include "gtest/gtest.h"

using namespace orc_rt;
using orc_rt_test::CrossDylibTestError;

namespace {

class UnrelatedError : public ErrorExtends<UnrelatedError, ErrorInfoBase> {
public:
  static constexpr const char *RTTIName =
      "orc_rt_test::RTTICrossDylibTest_UnrelatedError";
  std::string toString() const noexcept override { return {}; }
};

} // namespace

TEST(RTTICrossDylibTest, LibraryIDsAreDistinct) {
  // Sanity-check the premise of every other test in this file: the two
  // libraries must genuinely have different LibraryIDs, or the tests below
  // would only ever exercise the same-library fast path.
  CrossDylibTestError Local(0);
  EXPECT_NE(Local.libraryID(), rttiCrossDylibTest_libraryID());
}

TEST(RTTICrossDylibTest, IsACrossesLibraryBoundary) {
  ErrorInfoBase *E = rttiCrossDylibTest_makeError(42);
  ASSERT_NE(E, nullptr);

  EXPECT_TRUE(E->isA<CrossDylibTestError>());

  rttiCrossDylibTest_destroyError(E);
}

TEST(RTTICrossDylibTest, IsABaseTypeCrossesLibraryBoundary) {
  ErrorInfoBase *E = rttiCrossDylibTest_makeError(42);
  ASSERT_NE(E, nullptr);

  // Unlike the leaf-type query above, these succeed only by walking up the
  // hierarchy: differentDylibIsA has to recurse into ParentT after its own
  // strcmp fails. Every other cross-library check here either matches on the
  // first comparison or fails at all of them, so this is the only one that
  // sees the recursion return true across an image boundary.
  EXPECT_TRUE(E->isA<ErrorInfoBase>());
  EXPECT_TRUE(E->isA<RTTIRoot>());

  rttiCrossDylibTest_destroyError(E);
}

TEST(RTTICrossDylibTest, CastAfterIsACrossesLibraryBoundary) {
  ErrorInfoBase *E = rttiCrossDylibTest_makeError(42);
  ASSERT_NE(E, nullptr);
  ASSERT_TRUE(E->isA<CrossDylibTestError>());

  // isA<>() returning true is a contract that E is genuinely a
  // CrossDylibTestError -- the same C++ class, defined once in the shared
  // header and compiled into both libraries -- so casting back and calling a
  // real method on it must work, not just the boolean check.
  auto *Cast = static_cast<CrossDylibTestError *>(E);
  EXPECT_EQ(Cast->getCode(), 42);
  EXPECT_EQ(Cast->toString(), "CrossDylibTestError(42)");

  rttiCrossDylibTest_destroyError(E);
}

TEST(RTTICrossDylibTest, IsANotFooledByWrongType) {
  ErrorInfoBase *E = rttiCrossDylibTest_makeError(7);
  ASSERT_NE(E, nullptr);

  EXPECT_FALSE(E->isA<UnrelatedError>());
  EXPECT_TRUE(E->isA<CrossDylibTestError>());

  rttiCrossDylibTest_destroyError(E);
}
