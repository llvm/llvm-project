//===- unittests/Threading.cpp - Thread tests -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(Threading, PhysicalConcurrency) {
  auto Num = heavyweight_hardware_concurrency();
  // Since Num is unsigned this will also catch us trying to
  // return -1.
  ASSERT_LE(Num, thread::hardware_concurrency());
}

} // end anon namespace
