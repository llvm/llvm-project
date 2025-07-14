//===------- Offload API tests - olCreateProgram --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olLinkProgramTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLinkProgramTest);

TEST_P(olLinkProgramTest, SuccessSingle) {

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  ol_program_link_buffer_t Buffers[1] = {
      {const_cast<char *>(DeviceBin->getBufferStart()),
       DeviceBin->getBufferSize()},
  };

  ol_program_handle_t Program;
  ASSERT_SUCCESS(olLinkProgram(Device, Buffers, 1, &Program));
  ASSERT_NE(Program, nullptr);

  ASSERT_SUCCESS(olDestroyProgram(Program));
}

TEST_P(olLinkProgramTest, SuccessBuild) {
  std::unique_ptr<llvm::MemoryBuffer> ABin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("link_a", Device, ABin));
  std::unique_ptr<llvm::MemoryBuffer> BBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("link_b", Device, BBin));

  ol_program_link_buffer_t Buffers[2] = {
      {const_cast<char *>(ABin->getBufferStart()), ABin->getBufferSize()},
      {const_cast<char *>(BBin->getBufferStart()), BBin->getBufferSize()},
  };

  ol_program_handle_t Program;
  auto LinkResult = olLinkProgram(Device, Buffers, 2, &Program);
  if (LinkResult && LinkResult->Code == OL_ERRC_UNSUPPORTED)
    GTEST_SKIP() << "Linking unsupported: " << LinkResult->Details;
  ASSERT_SUCCESS(LinkResult);
  ASSERT_NE(Program, nullptr);

  ol_symbol_handle_t Kernel;
  ASSERT_SUCCESS(
      olGetSymbol(Program, "link_a", OL_SYMBOL_KIND_KERNEL, &Kernel));

  void *Mem;
  ASSERT_SUCCESS(
      olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 2 * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};
  ol_kernel_launch_size_args_t LaunchArgs{};
  LaunchArgs.GroupSize = {1, 1, 1};
  LaunchArgs.NumGroups = {1, 1, 1};
  LaunchArgs.Dimensions = 1;

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  ASSERT_EQ(Data[0], 200);
  ASSERT_EQ(Data[1], 100);

  ASSERT_SUCCESS(olMemFree(Mem));
  ASSERT_SUCCESS(olDestroyProgram(Program));
}

TEST_P(olLinkProgramTest, InvalidNotBitcode) {
  char FakeElf[] =
      "\177ELF0000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000";

  ol_program_link_buffer_t Buffers[1] = {
      {FakeElf, sizeof(FakeElf)},
  };

  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_BINARY,
               olLinkProgram(Device, Buffers, 1, &Program));
}

TEST_P(olLinkProgramTest, InvalidSize) {
  ol_program_link_buffer_t Buffers[0] = {};

  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olLinkProgram(Device, Buffers, 0, &Program));
}
