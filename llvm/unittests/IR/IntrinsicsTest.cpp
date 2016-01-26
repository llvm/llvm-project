//===- llvm/unittest/IR/IntrinsicsTest.cpp - ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
