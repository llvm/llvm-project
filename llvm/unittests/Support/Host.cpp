//========- unittests/Support/Host.cpp - Host.cpp tests --------------========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Host.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"

#include "gtest/gtest.h"

using namespace llvm;

class HostTest : public testing::Test {
  Triple Host;

protected:
  bool isSupportedArchAndOS() {
    // Initially this is only testing detection of the number of
    // physical cores, which is currently only supported/tested for
    // x86_64 Linux and Darwin.
    return (Host.getArch() == Triple::x86_64 &&
            (Host.isOSDarwin() || Host.getOS() == Triple::Linux));
  }

  HostTest() : Host(Triple::normalize(sys::getProcessTriple())) {}
};

TEST_F(HostTest, NumPhysicalCores) {
  int Num = sys::getHostNumPhysicalCores();

  if (isSupportedArchAndOS())
    ASSERT_GT(Num, 0);
  else
    ASSERT_EQ(Num, -1);
}
