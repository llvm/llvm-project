//===- ErrnoTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's sys/Errno.h APIs.
//
// The exact wording of a description is the system's business, so these only
// check the properties strError promises.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/support/sys/Errno.h"
#include "gtest/gtest.h"

#include <errno.h>

using namespace orc_rt;

TEST(ErrnoTest, KnownValuesDescribed) {
  EXPECT_FALSE(sys::strError(EINVAL).empty());
  EXPECT_FALSE(sys::strError(ENOENT).empty());

  // Distinct values should not share a description, or callers can't tell what
  // went wrong.
  EXPECT_NE(sys::strError(EINVAL), sys::strError(ENOENT));
}

TEST(ErrnoTest, UnknownValueStillDescribed) {
  // strError promises a non-empty result even where the system has no
  // description, so that an error message never comes out blank.
  EXPECT_FALSE(sys::strError(999999).empty());
}

TEST(ErrnoTest, ResultIsNulFree) {
  // The result is built from a fixed-size buffer; it must be trimmed to the
  // description rather than padded out to the buffer length.
  auto S = sys::strError(EINVAL);
  EXPECT_EQ(S.find('\0'), std::string::npos);
  EXPECT_EQ(S.size(), strlen(S.c_str()));
}
