//===- llvm/unittest/LeakDetector/LeakDetector.cpp - LeakDetector tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Support/LeakDetector.h"

using namespace llvm;

namespace {

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(LeakDetector, Death1) {
  LeakDetector::addGarbageObject((void*) 1);
  LeakDetector::addGarbageObject((void*) 2);

  EXPECT_DEATH(LeakDetector::addGarbageObject((void*) 1),
               ".*Ts.count\\(o\\) == 0 && \"Object already in set!\"");
  EXPECT_DEATH(LeakDetector::addGarbageObject((void*) 2),
               "Cache != o && \"Object already in set!\"");
}
#endif
#endif

}
