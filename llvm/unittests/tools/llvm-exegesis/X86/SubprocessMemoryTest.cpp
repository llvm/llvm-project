//===-- SubprocessMemoryTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubprocessMemory.h"

#include "X86/TestBase.h"
#include "gtest/gtest.h"
#include <unordered_map>

#ifdef __linux__
#include <endian.h>
#include <fcntl.h>
#include <sys/mman.h>
#endif // __linux__

namespace llvm {
namespace exegesis {

#if defined(__linux__) && !defined(__ANDROID__)

class SubprocessMemoryTest : public X86TestBase {
protected:
  void
  testCommon(std::unordered_map<std::string, MemoryValue> MemoryDefinitions,
             const int MainProcessPID) {
    EXPECT_FALSE(SM.initializeSubprocessMemory(MainProcessPID));
    EXPECT_FALSE(SM.addMemoryDefinition(MemoryDefinitions, MainProcessPID));
  }

  void checkSharedMemoryDefinition(const std::string &DefinitionName,
                                   size_t DefinitionSize,
                                   std::vector<uint8_t> ExpectedValue) {
    int SharedMemoryFD =
        shm_open(DefinitionName.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    uint8_t *SharedMemoryMapping = (uint8_t *)mmap(
        NULL, DefinitionSize, PROT_READ, MAP_SHARED, SharedMemoryFD, 0);
    EXPECT_NE((intptr_t)SharedMemoryMapping, -1);
    for (size_t I = 0; I < ExpectedValue.size(); ++I) {
      EXPECT_EQ(SharedMemoryMapping[I], ExpectedValue[I]);
    }
    munmap(SharedMemoryMapping, DefinitionSize);
  }

  SubprocessMemory SM;
};

// Some of the tests below are failing on s390x and PPC due to the shared
// memory calls not working in some cases, so they have been disabled.
// TODO(boomanaiden154): Investigate and fix this issue on PPC.

#if defined(__powerpc__) || defined(__s390x__)
TEST_F(SubprocessMemoryTest, DISABLED_OneDefinition) {
#else
TEST_F(SubprocessMemoryTest, OneDefinition) {
#endif
  testCommon({{"test1", {APInt(8, 0xff), 4096, 0}}}, 0);
  checkSharedMemoryDefinition("/0memdef0", 4096, {0xff});
}

#if defined(__powerpc__) || defined(__s390x__)
TEST_F(SubprocessMemoryTest, DISABLED_MultipleDefinitions) {
#else
TEST_F(SubprocessMemoryTest, MultipleDefinitions) {
#endif
  testCommon({{"test1", {APInt(8, 0xaa), 4096, 0}},
              {"test2", {APInt(8, 0xbb), 4096, 1}},
              {"test3", {APInt(8, 0xcc), 4096, 2}}},
             1);
  checkSharedMemoryDefinition("/1memdef0", 4096, {0xaa});
  checkSharedMemoryDefinition("/1memdef1", 4096, {0xbb});
  checkSharedMemoryDefinition("/1memdef2", 4096, {0xcc});
}

#if defined(__powerpc__) || defined(__s390x__)
TEST_F(SubprocessMemoryTest, DISABLED_DefinitionFillsCompletely) {
#else
TEST_F(SubprocessMemoryTest, DefinitionFillsCompletely) {
#endif
  testCommon({{"test1", {APInt(8, 0xaa), 4096, 0}},
              {"test2", {APInt(16, 0xbbbb), 4096, 1}},
              {"test3", {APInt(24, 0xcccccc), 4096, 2}}},
             2);
  std::vector<uint8_t> Test1Expected(512, 0xaa);
  std::vector<uint8_t> Test2Expected(512, 0xbb);
  std::vector<uint8_t> Test3Expected(512, 0xcc);
  checkSharedMemoryDefinition("/2memdef0", 4096, Test1Expected);
  checkSharedMemoryDefinition("/2memdef1", 4096, Test2Expected);
  checkSharedMemoryDefinition("/2memdef2", 4096, Test3Expected);
}

// The following test is only supported on little endian systems.
#if defined(__powerpc__) || defined(__s390x__) || __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
TEST_F(SubprocessMemoryTest, DISABLED_DefinitionEndTruncation) {
#else
TEST_F(SubprocessMemoryTest, DefinitionEndTruncation) {
#endif
  testCommon({{"test1", {APInt(48, 0xaabbccddeeff), 4096, 0}}}, 3);
  std::vector<uint8_t> Test1Expected(512, 0);
  // order is reversed since we're assuming a little endian system.
  for (size_t I = 0; I < Test1Expected.size(); ++I) {
    switch (I % 6) {
    case 0:
      Test1Expected[I] = 0xff;
      break;
    case 1:
      Test1Expected[I] = 0xee;
      break;
    case 2:
      Test1Expected[I] = 0xdd;
      break;
    case 3:
      Test1Expected[I] = 0xcc;
      break;
    case 4:
      Test1Expected[I] = 0xbb;
      break;
    case 5:
      Test1Expected[I] = 0xaa;
    }
  }
  checkSharedMemoryDefinition("/3memdef0", 4096, Test1Expected);
}

#endif // defined(__linux__) && !defined(__ANDROID__)

} // namespace exegesis
} // namespace llvm
