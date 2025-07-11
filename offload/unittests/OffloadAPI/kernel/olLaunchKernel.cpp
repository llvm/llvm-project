//===------- Offload API tests - olLaunchKernel --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct LaunchKernelTestBase : OffloadQueueTest {
  void SetUpKernel(const char *kernel) {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary(kernel, Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));
    ASSERT_SUCCESS(olGetKernel(Program, kernel, &Kernel));
    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSize = {64, 1, 1};
    LaunchArgs.NumGroups = {1, 1, 1};

    LaunchArgs.DynSharedMemory = 0;
  }

  void TearDown() override {
    if (Program) {
      olDestroyProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::TearDown());
  }

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ol_program_handle_t Program = nullptr;
  ol_kernel_handle_t Kernel = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
};

#define KERNEL_TEST(NAME, KERNEL)                                              \
  struct olLaunchKernel##NAME##Test : LaunchKernelTestBase {                   \
    void SetUp() override { LaunchKernelTestBase::SetUpKernel(#KERNEL); }      \
  };                                                                           \
  OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchKernel##NAME##Test);

KERNEL_TEST(Foo, foo)
KERNEL_TEST(NoArgs, noargs)
KERNEL_TEST(LocalMem, localmem)
KERNEL_TEST(LocalMemReduction, localmem_reduction)
KERNEL_TEST(LocalMemStatic, localmem_static)

TEST_P(olLaunchKernelFooTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelNoArgsTest, Success) {
  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, nullptr, 0, &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));
}

TEST_P(olLaunchKernelFooTest, SuccessSynchronous) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(nullptr, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelLocalMemTest, Success) {
  LaunchArgs.NumGroups.x = 4;
  LaunchArgs.DynSharedMemory = 64 * sizeof(uint32_t);

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * LaunchArgs.NumGroups.x *
                                sizeof(uint32_t),
                            &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.GroupSize.x * LaunchArgs.NumGroups.x; i++)
    ASSERT_EQ(Data[i], (i % 64) * 2);

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelLocalMemReductionTest, Success) {
  LaunchArgs.NumGroups.x = 4;
  LaunchArgs.DynSharedMemory = 64 * sizeof(uint32_t);

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.NumGroups.x * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.NumGroups.x; i++)
    ASSERT_EQ(Data[i], 2 * LaunchArgs.GroupSize.x);

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelLocalMemStaticTest, Success) {
  LaunchArgs.NumGroups.x = 4;
  LaunchArgs.DynSharedMemory = 0;

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.NumGroups.x * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.NumGroups.x; i++)
    ASSERT_EQ(Data[i], 2 * LaunchArgs.GroupSize.x);

  ASSERT_SUCCESS(olMemFree(Mem));
}
