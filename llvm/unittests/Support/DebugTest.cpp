//===- llvm/unittest/Support/DebugTest.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>
using namespace llvm;

#ifndef NDEBUG
TEST(DebugTest, Basic) {
  std::string s1, s2;
  raw_string_ostream os1(s1), os2(s2);
  static const char *DT[] = {"A", "B"};  
  
  llvm::DebugFlag = true;
  setCurrentDebugTypes(DT, 2);
  DEBUG_WITH_TYPE("A", os1 << "A");
  DEBUG_WITH_TYPE("B", os1 << "B");
  EXPECT_EQ("AB", os1.str());

  setCurrentDebugType("A");
  DEBUG_WITH_TYPE("A", os2 << "A");
  DEBUG_WITH_TYPE("B", os2 << "B");
  EXPECT_EQ("A", os2.str());
}
#endif
