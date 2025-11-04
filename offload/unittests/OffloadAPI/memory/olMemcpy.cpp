//===------- Offload API tests - olMemcpy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemcpyTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemcpyTest);

struct olMemcpyGlobalTest : OffloadGlobalTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadGlobalTest::SetUp());
    ASSERT_SUCCESS(
        olGetSymbol(Program, "read", OL_SYMBOL_KIND_KERNEL, &ReadKernel));
    ASSERT_SUCCESS(
        olGetSymbol(Program, "write", OL_SYMBOL_KIND_KERNEL, &WriteKernel));
    ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
    ASSERT_SUCCESS(olGetSymbolInfo(
        Global, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS, sizeof(Addr), &Addr));

    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSize = {64, 1, 1};
    LaunchArgs.NumGroups = {1, 1, 1};

    LaunchArgs.DynSharedMemory = 0;
  }

  ol_kernel_launch_size_args_t LaunchArgs{};
  void *Addr;
  ol_symbol_handle_t ReadKernel;
  ol_symbol_handle_t WriteKernel;
  ol_queue_handle_t Queue;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemcpyGlobalTest);

TEST_P(olMemcpyTest, SuccessHtoD) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size, 42);
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size));
  olSyncQueue(Queue);
  olMemFree(Alloc);
}

TEST_P(olMemcpyTest, SuccessDtoH) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, Alloc, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemcpyTest, SuccessDtoD) {
  constexpr size_t Size = 1024;
  void *AllocA;
  void *AllocB;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocA));
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocB));
  ASSERT_SUCCESS(olMemcpy(Queue, AllocA, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, AllocB, Device, AllocA, Device, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, AllocB, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(AllocA));
  ASSERT_SUCCESS(olMemFree(AllocB));
}

TEST_P(olMemcpyTest, SuccessHtoHSync) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemcpy(nullptr, Output.data(), Host, Input.data(), Host, Size));

  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
}

TEST_P(olMemcpyTest, SuccessDtoHSync) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemcpy(nullptr, Alloc, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Alloc, Device, Size));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemcpyTest, SuccessSizeZero) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  // As with std::memcpy, size 0 is allowed. Keep all other arguments valid even
  // if they aren't used.
  ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Input.data(), Host, 0));
}

TEST_P(olMemcpyGlobalTest, SuccessRoundTrip) {
  void *SourceMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            64 * sizeof(uint32_t), &SourceMem));
  uint32_t *SourceData = (uint32_t *)SourceMem;
  for (auto I = 0; I < 64; I++)
    SourceData[I] = I;

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            64 * sizeof(uint32_t), &DestMem));

  ASSERT_SUCCESS(
      olMemcpy(Queue, Addr, Device, SourceMem, Host, 64 * sizeof(uint32_t)));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DestMem, Host, Addr, Device, 64 * sizeof(uint32_t)));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I);

  ASSERT_SUCCESS(olMemFree(DestMem));
  ASSERT_SUCCESS(olMemFree(SourceMem));
}

TEST_P(olMemcpyGlobalTest, SuccessWrite) {
  void *SourceMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &SourceMem));
  uint32_t *SourceData = (uint32_t *)SourceMem;
  for (auto I = 0; I < 64; I++)
    SourceData[I] = I;

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &DestMem));
  struct {
    void *Mem;
  } Args{DestMem};

  ASSERT_SUCCESS(
      olMemcpy(Queue, Addr, Device, SourceMem, Host, 64 * sizeof(uint32_t)));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, ReadKernel, &Args, sizeof(Args),
                                &LaunchArgs));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I);

  ASSERT_SUCCESS(olMemFree(DestMem));
  ASSERT_SUCCESS(olMemFree(SourceMem));
}

TEST_P(olMemcpyGlobalTest, SuccessRead) {
  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &DestMem));

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, WriteKernel, nullptr, 0, &LaunchArgs));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DestMem, Host, Addr, Device, 64 * sizeof(uint32_t)));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I * 2);

  ASSERT_SUCCESS(olMemFree(DestMem));
}
