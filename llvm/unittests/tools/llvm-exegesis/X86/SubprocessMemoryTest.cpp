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
#include <string>
#include <unordered_map>

#ifdef __linux__
#include <endian.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif // __linux__

// This needs to be updated anytime a test is added or removed from the test
// suite.
static constexpr const size_t TestCount = 4;

namespace llvm {
namespace exegesis {

#if defined(__linux__) && !defined(__ANDROID__)

class SubprocessMemoryTest : public X86TestBase {
protected:
  int getSharedMemoryNumber(const unsigned TestNumber) {
    // Do a process similar to 2D array indexing so that each process gets it's
    // own shared memory space to avoid collisions. This will not overflow as
    // the maximum value a PID can take on is 10^22.
    return getpid() * TestCount + TestNumber;
  }

  void
  testCommon(std::unordered_map<std::string, MemoryValue> MemoryDefinitions,
             const unsigned TestNumber) {
    EXPECT_FALSE(
        SM.initializeSubprocessMemory(getSharedMemoryNumber(TestNumber)));
    EXPECT_FALSE(SM.addMemoryDefinition(MemoryDefinitions,
                                        getSharedMemoryNumber(TestNumber)));
  }

  std::string getSharedMemoryName(const unsigned TestNumber,
                                  const unsigned DefinitionNumber) {
    return "/" + std::to_string(getSharedMemoryNumber(TestNumber)) + "memdef" +
           std::to_string(DefinitionNumber);
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
  checkSharedMemoryDefinition(getSharedMemoryName(0, 0), 4096, {0xff});
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
  checkSharedMemoryDefinition(getSharedMemoryName(1, 0), 4096, {0xaa});
  checkSharedMemoryDefinition(getSharedMemoryName(1, 1), 4096, {0xbb});
  checkSharedMemoryDefinition(getSharedMemoryName(1, 2), 4096, {0xcc});
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
  checkSharedMemoryDefinition(getSharedMemoryName(2, 0), 4096, Test1Expected);
  checkSharedMemoryDefinition(getSharedMemoryName(2, 1), 4096, Test2Expected);
  checkSharedMemoryDefinition(getSharedMemoryName(2, 2), 4096, Test3Expected);
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
  checkSharedMemoryDefinition(getSharedMemoryName(3, 0), 4096, Test1Expected);
}

#endif // defined(__linux__) && !defined(__ANDROID__)

} // namespace exegesis
} // namespace llvm
