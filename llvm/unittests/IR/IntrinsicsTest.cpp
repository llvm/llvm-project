//===- llvm/unittest/IR/IntrinsicsTest.cpp - ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicInst.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static const char *const NameTable1[] = {
  "llvm.foo",
  "llvm.foo.a",
  "llvm.foo.b",
  "llvm.foo.b.a",
  "llvm.foo.c",
};

TEST(IntrinNameLookup, Basic) {
  int I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo");
  EXPECT_EQ(0, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.f64");
  EXPECT_EQ(0, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.b");
  EXPECT_EQ(2, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.b.a");
  EXPECT_EQ(3, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.c");
  EXPECT_EQ(4, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.c.f64");
  EXPECT_EQ(4, I);
}

} // end namespace
