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

#include <cstring>

using olMemcpyTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemcpyTest);

struct olMemcpyGlobalTest : OffloadGlobalTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadGlobalTest::SetUp());
    ASSERT_SUCCESS(
        olGetSymbol(Program, "read", OL_SYMBOL_KIND_KERNEL, &ReadKernel));
    ASSERT_SUCCESS(
        olGetSymbol(Program, "write", OL_SYMBOL_KIND_KERNEL, &WriteKernel));
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
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

struct olMemcpyMultiDeviceTest : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    for (auto D : Devices) {
      ol_queue_handle_t Queue = nullptr;
      ASSERT_SUCCESS(olCreateQueue(Context, D, &Queue));
      Queues.push_back(Queue);
    }
  }

  void TearDown() override {
    for (auto Queue : Queues)
      olDestroyQueue(Queue);
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::TearDown());
  }

  std::vector<ol_queue_handle_t> Queues;
};
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(olMemcpyMultiDeviceTest);

TEST_P(olMemcpyTest, SuccessHtoD) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size, 42);
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size));
  olSyncQueue(Queue);
  olMemFree(Context, Alloc);
}

TEST_P(olMemcpyTest, SuccessDtoH) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, Alloc, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpyTest, SuccessDtoD) {
  constexpr size_t Size = 1024;
  void *AllocA;
  void *AllocB;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocA));
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocB));
  ASSERT_SUCCESS(olMemcpy(Queue, AllocA, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, AllocB, Device, AllocA, Device, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, AllocB, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Context, AllocA));
  ASSERT_SUCCESS(olMemFree(Context, AllocB));
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

TEST_P(olMemcpyTest, SuccessHtoHQueuedOrdering) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Intermediate(Size, 0);
  std::vector<uint8_t> Copied(Size, 0);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(
      olMemcpy(Queue, Intermediate.data(), Host, Alloc, Device, Size));
  ASSERT_SUCCESS(
      olMemcpy(Queue, Copied.data(), Host, Intermediate.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Copied.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, Alloc, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (uint8_t Val : Output)
    ASSERT_EQ(Val, 42);

  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpyTest, SuccessHtoHQueuedOrderingHostAlloc) {
  constexpr size_t Size = 1024;
  void *Alloc;
  void *Input;
  void *Intermediate;
  void *Copied;
  void *Output;

  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemAllocHost(Context, Device, Size, &Input));
  ASSERT_SUCCESS(olMemAllocHost(Context, Device, Size, &Intermediate));
  ASSERT_SUCCESS(olMemAllocHost(Context, Device, Size, &Copied));
  ASSERT_SUCCESS(olMemAllocHost(Context, Device, Size, &Output));

  std::memset(Input, 42, Size);
  std::memset(Intermediate, 0, Size);
  std::memset(Copied, 0, Size);
  std::memset(Output, 0, Size);

  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input, Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Intermediate, Host, Alloc, Device, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Copied, Host, Intermediate, Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Copied, Host, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output, Host, Alloc, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (size_t I = 0; I < Size; ++I)
    ASSERT_EQ(static_cast<uint8_t *>(Output)[I], 42);

  ASSERT_SUCCESS(olMemFree(Context, Output));
  ASSERT_SUCCESS(olMemFree(Context, Copied));
  ASSERT_SUCCESS(olMemFree(Context, Intermediate));
  ASSERT_SUCCESS(olMemFree(Context, Input));
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpyTest, SuccessDtoHSync) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemcpy(nullptr, Alloc, Device, Input.data(), Host, Size));
  ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Alloc, Device, Size));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpyTest, SuccessNullHostDevice) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), nullptr, Size));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), nullptr, Alloc, Device, Size));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpyTest, SuccessNullHostDeviceHtoH) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemcpy(nullptr, Output.data(), nullptr, Input.data(), nullptr, Size));

  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
}

TEST_P(olMemcpyTest, SuccessSizeZero) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  // As with std::memcpy, size 0 is allowed. Keep all other arguments valid even
  // if they aren't used.
  ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Input.data(), Host, 0));
}

TEST_P(olMemcpyTest, SuccessHtoHQueuedSizeZero) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, Input.data(), Host, 0));
  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olMemcpyGlobalTest, SuccessRoundTrip) {
  void *SourceMem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
                            64 * sizeof(uint32_t), &SourceMem));
  uint32_t *SourceData = (uint32_t *)SourceMem;
  for (auto I = 0; I < 64; I++)
    SourceData[I] = I;

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
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

  ASSERT_SUCCESS(olMemFree(Context, DestMem));
  ASSERT_SUCCESS(olMemFree(Context, SourceMem));
}

TEST_P(olMemcpyGlobalTest, SuccessWrite) {
  void *SourceMem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &SourceMem));
  uint32_t *SourceData = (uint32_t *)SourceMem;
  for (auto I = 0; I < 64; I++)
    SourceData[I] = I;

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &DestMem));
  void *ArgPtrs[] = {&DestMem};
  size_t ArgSizes[] = {sizeof(DestMem)};

  ASSERT_SUCCESS(
      olMemcpy(Queue, Addr, Device, SourceMem, Host, 64 * sizeof(uint32_t)));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, ReadKernel, &LaunchArgs, nullptr,
                                1, ArgPtrs, ArgSizes));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I);

  ASSERT_SUCCESS(olMemFree(Context, DestMem));
  ASSERT_SUCCESS(olMemFree(Context, SourceMem));
}

TEST_P(olMemcpyGlobalTest, SuccessRead) {
  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &DestMem));

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, WriteKernel, &LaunchArgs,
                                nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DestMem, Host, Addr, Device, 64 * sizeof(uint32_t)));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I * 2);

  ASSERT_SUCCESS(olMemFree(Context, DestMem));
}

TEST_P(olMemcpyMultiDeviceTest, SuccessDtoD) {
  constexpr size_t Size = 1024;

  for (size_t Src = 0; Src < Devices.size(); ++Src) {
    for (size_t Dst = 0; Dst < Devices.size(); ++Dst) {
      if (Src == Dst)
        continue;
      SCOPED_TRACE("Device #" + std::to_string(Src) + " -> Device #" +
                   std::to_string(Dst));
      ol_device_handle_t SrcDevice = Devices[Src];
      ol_device_handle_t DstDevice = Devices[Dst];
      ol_queue_handle_t Queue = Queues[Src];

      void *SrcAlloc;
      void *DstAlloc;
      std::vector<uint8_t> Input(Size, static_cast<uint8_t>(Src + 1));
      std::vector<uint8_t> Output(Size, 0);

      ASSERT_SUCCESS(olMemAlloc(Context, SrcDevice, OL_ALLOC_TYPE_DEVICE, Size,
                                &SrcAlloc));
      ASSERT_SUCCESS(olMemAlloc(Context, DstDevice, OL_ALLOC_TYPE_DEVICE, Size,
                                &DstAlloc));
      ASSERT_SUCCESS(
          olMemcpy(Queue, SrcAlloc, SrcDevice, Input.data(), Host, Size));
      ASSERT_SUCCESS(
          olMemcpy(Queue, DstAlloc, DstDevice, SrcAlloc, SrcDevice, Size));
      ASSERT_SUCCESS(
          olMemcpy(Queue, Output.data(), Host, DstAlloc, DstDevice, Size));
      ASSERT_SUCCESS(olSyncQueue(Queue));
      for (uint8_t Val : Output)
        ASSERT_EQ(Val, Src + 1);

      ASSERT_SUCCESS(olMemFree(Context, DstAlloc));
      ASSERT_SUCCESS(olMemFree(Context, SrcAlloc));
    }
  }
}

TEST_P(olMemcpyMultiDeviceTest, SuccessDtoDSync) {
  constexpr size_t Size = 1024;

  for (size_t Src = 0; Src < Devices.size(); ++Src) {
    for (size_t Dst = 0; Dst < Devices.size(); ++Dst) {
      if (Src == Dst)
        continue;
      SCOPED_TRACE("Device #" + std::to_string(Src) + " -> Device #" +
                   std::to_string(Dst));
      ol_device_handle_t SrcDevice = Devices[Src];
      ol_device_handle_t DstDevice = Devices[Dst];

      void *SrcAlloc;
      void *DstAlloc;
      std::vector<uint8_t> Input(Size, static_cast<uint8_t>(Src + 1));
      std::vector<uint8_t> Output(Size, 0);

      ASSERT_SUCCESS(olMemAlloc(Context, SrcDevice, OL_ALLOC_TYPE_DEVICE, Size,
                                &SrcAlloc));
      ASSERT_SUCCESS(olMemAlloc(Context, DstDevice, OL_ALLOC_TYPE_DEVICE, Size,
                                &DstAlloc));
      ASSERT_SUCCESS(
          olMemcpy(nullptr, SrcAlloc, SrcDevice, Input.data(), Host, Size));
      ASSERT_SUCCESS(
          olMemcpy(nullptr, DstAlloc, DstDevice, SrcAlloc, SrcDevice, Size));
      ASSERT_SUCCESS(
          olMemcpy(nullptr, Output.data(), Host, DstAlloc, DstDevice, Size));
      for (uint8_t Val : Output)
        ASSERT_EQ(Val, Src + 1);

      ASSERT_SUCCESS(olMemFree(Context, DstAlloc));
      ASSERT_SUCCESS(olMemFree(Context, SrcAlloc));
    }
  }
}

TEST_P(olMemcpyMultiDeviceTest, SuccessDtoDChain) {
  constexpr size_t Size = 1024;
  std::vector<void *> Allocs(Devices.size(), nullptr);
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  for (size_t I = 0; I < Devices.size(); ++I)
    ASSERT_SUCCESS(olMemAlloc(Context, Devices[I], OL_ALLOC_TYPE_DEVICE, Size,
                              &Allocs[I]));

  // Pass the data through every device in turn, each copy enqueued on the queue
  // of the source device and synchronized before the next one starts.
  ASSERT_SUCCESS(
      olMemcpy(Queues[0], Allocs[0], Devices[0], Input.data(), Host, Size));
  ASSERT_SUCCESS(olSyncQueue(Queues[0]));
  for (size_t I = 1; I < Devices.size(); ++I) {
    ASSERT_SUCCESS(olMemcpy(Queues[I - 1], Allocs[I], Devices[I], Allocs[I - 1],
                            Devices[I - 1], Size));
    ASSERT_SUCCESS(olSyncQueue(Queues[I - 1]));
  }
  ASSERT_SUCCESS(olMemcpy(Queues.back(), Output.data(), Host, Allocs.back(),
                          Devices.back(), Size));
  ASSERT_SUCCESS(olSyncQueue(Queues.back()));
  for (uint8_t Val : Output)
    ASSERT_EQ(Val, 42);

  for (void *Alloc : Allocs)
    ASSERT_SUCCESS(olMemFree(Context, Alloc));
}
