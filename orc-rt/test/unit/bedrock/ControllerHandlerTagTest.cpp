//===- ControllerHandlerTagTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG.
//
// These tests check that the macro produces a definition (rather than only a
// declaration) in both C++ and C. Each references a tag defined with the
// macro, so a failure shows up as a link error rather than a test failure.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/bedrock/ControllerHandlerTag.h"

#include "gtest/gtest.h"

ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG(orc_rt_tag_test_controllerHandlerTagCxx);

// Defined in ControllerHandlerTagCTest.c, which is compiled as C.
extern "C" orc_rt_ControllerHandlerTag *orc_rt_test_controllerHandlerTagC(void);

// Addresses are stored through volatile pointers so that the compiler can't
// fold the null checks below, and with them the references to the tags.

TEST(ControllerHandlerTagTest, DefinedInCxx) {
  orc_rt_ControllerHandlerTag *volatile Tag =
      &orc_rt_tag_test_controllerHandlerTagCxx;
  EXPECT_NE(Tag, nullptr);
}

TEST(ControllerHandlerTagTest, DefinedInC) {
  orc_rt_ControllerHandlerTag *volatile Tag =
      orc_rt_test_controllerHandlerTagC();
  EXPECT_NE(Tag, nullptr);
}
