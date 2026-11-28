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

#ifdef __linux__
#include <endian.h>
#include <sys/mman.h>
#endif // __linux__

namespace llvm {
namespace exegesis {

#if defined(__linux__) && !defined(__ANDROID__) &&                             \
    !(defined(__powerpc__) || defined(__s390x__) || defined(__sparc__))

class SubprocessMemoryTest : public X86TestBase {
protected:
  void testCommon(StringMap<MemoryValue> MemoryDefinitions) {
    EXPECT_FALSE(SM.initializeSubprocessMemory());
    EXPECT_FALSE(SM.addMemoryDefinition(std::move(MemoryDefinitions)));
  }

  void checkSharedMemoryDefinition(const unsigned DefinitionNumber,
                                   size_t DefinitionSize,
                                   std::vector<uint8_t> ExpectedValue) {
    Expected<int> SharedMemoryFDOrErr =
        SM.getMemoryDefinitionFD(DefinitionNumber);
    ASSERT_TRUE(static_cast<bool>(SharedMemoryFDOrErr));
    uint8_t *SharedMemoryMapping =
        static_cast<uint8_t *>(mmap(nullptr, DefinitionSize, PROT_READ,
                                    MAP_SHARED, *SharedMemoryFDOrErr, 0));
    ASSERT_NE(SharedMemoryMapping, MAP_FAILED);
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

TEST_F(SubprocessMemoryTest, OneDefinition) {
  testCommon({{"test1", {APInt(8, 0xff), 4096, 0}}});
  checkSharedMemoryDefinition(0, 4096, {0xff});
}

TEST_F(SubprocessMemoryTest, MultipleDefinitions) {
  testCommon({{"test1", {APInt(8, 0xaa), 4096, 0}},
              {"test2", {APInt(8, 0xbb), 4096, 1}},
              {"test3", {APInt(8, 0xcc), 4096, 2}}});
  checkSharedMemoryDefinition(0, 4096, {0xaa});
  checkSharedMemoryDefinition(1, 4096, {0xbb});
  checkSharedMemoryDefinition(2, 4096, {0xcc});
}

TEST_F(SubprocessMemoryTest, DefinitionFillsCompletely) {
  testCommon({{"test1", {APInt(8, 0xaa), 4096, 0}},
              {"test2", {APInt(16, 0xbbbb), 4096, 1}},
              {"test3", {APInt(24, 0xcccccc), 4096, 2}}});
  std::vector<uint8_t> Test1Expected(512, 0xaa);
  std::vector<uint8_t> Test2Expected(512, 0xbb);
  std::vector<uint8_t> Test3Expected(512, 0xcc);
  checkSharedMemoryDefinition(0, 4096, Test1Expected);
  checkSharedMemoryDefinition(1, 4096, Test2Expected);
  checkSharedMemoryDefinition(2, 4096, Test3Expected);
}

// The following test is only supported on little endian systems.
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
TEST_F(SubprocessMemoryTest, DISABLED_DefinitionEndTruncation) {
#else
TEST_F(SubprocessMemoryTest, DefinitionEndTruncation) {
#endif
  testCommon({{"test1", {APInt(48, 0xaabbccddeeff), 4096, 0}}});
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
  checkSharedMemoryDefinition(0, 4096, Test1Expected);
}

#endif // __linux__ && !__ANDROID__ && !(__powerpc__ || __s390x__ || __sparc__)

} // namespace exegesis
} // namespace llvm
