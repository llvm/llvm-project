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
  SmallVector<std::pair<Triple::ArchType, Triple::OSType>, 4> SupportedArchAndOSs;

protected:
  bool isSupportedArchAndOS() {
    if (is_contained(SupportedArchAndOSs, std::make_pair(Host.getArch(), Host.getOS())))
      return true;

    return false;
  }

  HostTest() {
    Host.setTriple(Triple::normalize(sys::getProcessTriple()));

    // Initially this is only testing detection of the number of
    // physical cores, which is currently only supported/tested for
    // x86_64 Linux and Darwin.
    SupportedArchAndOSs.push_back(std::make_pair(Triple::x86_64, Triple::Linux));
    SupportedArchAndOSs.push_back(std::make_pair(Triple::x86_64, Triple::Darwin));
    SupportedArchAndOSs.push_back(std::make_pair(Triple::x86_64, Triple::MacOSX));
  }
};

TEST_F(HostTest, NumPhysicalCores) {
  int Num = sys::getHostNumPhysicalCores();

  if (isSupportedArchAndOS())
    ASSERT_GT(Num, 0);
  else
    ASSERT_EQ(Num, -1);
}
