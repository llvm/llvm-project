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
  void SetUpProgram(const char *program) {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary(program, Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));

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
  ol_kernel_launch_size_args_t LaunchArgs{};
};

struct LaunchSingleKernelTestBase : LaunchKernelTestBase {
  void SetUpKernel(const char *kernel) {
    RETURN_ON_FATAL_FAILURE(SetUpProgram(kernel));
    ASSERT_SUCCESS(
        olGetSymbol(Program, kernel, OL_SYMBOL_KIND_KERNEL, &Kernel));
  }

  ol_symbol_handle_t Kernel = nullptr;
};

#define KERNEL_TEST(NAME, KERNEL)                                              \
  struct olLaunchKernel##NAME##Test : LaunchSingleKernelTestBase {             \
    void SetUp() override { SetUpKernel(#KERNEL); }                            \
  };                                                                           \
  OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchKernel##NAME##Test);

KERNEL_TEST(Foo, foo)
KERNEL_TEST(NoArgs, noargs)
KERNEL_TEST(Byte, byte)
KERNEL_TEST(LocalMem, localmem)
KERNEL_TEST(LocalMemReduction, localmem_reduction)
KERNEL_TEST(LocalMemStatic, localmem_static)
KERNEL_TEST(GlobalCtor, global_ctor)
KERNEL_TEST(GlobalDtor, global_dtor)

struct LaunchMultipleKernelTestBase : LaunchKernelTestBase {
  void SetUpKernels(const char *program, std::vector<const char *> kernels) {
    RETURN_ON_FATAL_FAILURE(SetUpProgram(program));

    Kernels.resize(kernels.size());
    size_t I = 0;
    for (auto K : kernels)
      ASSERT_SUCCESS(
          olGetSymbol(Program, K, OL_SYMBOL_KIND_KERNEL, &Kernels[I++]));
  }

  std::vector<ol_symbol_handle_t> Kernels;
};

#define KERNEL_MULTI_TEST(NAME, PROGRAM, ...)                                  \
  struct olLaunchKernel##NAME##Test : LaunchMultipleKernelTestBase {           \
    void SetUp() override { SetUpKernels(#PROGRAM, {__VA_ARGS__}); }           \
  };                                                                           \
  OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchKernel##NAME##Test);

KERNEL_MULTI_TEST(Global, global, "write", "read")

TEST_P(olLaunchKernelFooTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelFooTest, SuccessThreaded) {
  threadify([&](size_t) {
    void *Mem;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                              LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));
    struct {
      void *Mem;
    } Args{Mem};

    ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                  &LaunchArgs));

    ASSERT_SUCCESS(olSyncQueue(Queue));

    uint32_t *Data = (uint32_t *)Mem;
    for (uint32_t i = 0; i < 64; i++) {
      ASSERT_EQ(Data[i], i);
    }

    ASSERT_SUCCESS(olMemFree(Mem));
  });
}

TEST_P(olLaunchKernelNoArgsTest, Success) {
  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, nullptr, 0, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olLaunchKernelFooTest, SuccessSynchronous) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(nullptr, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs));

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

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

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

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

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

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.NumGroups.x; i++)
    ASSERT_EQ(Data[i], 2 * LaunchArgs.GroupSize.x);

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelGlobalTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernels[0], nullptr, 0, &LaunchArgs));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernels[1], &Args, sizeof(Args),
                                &LaunchArgs));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i * 2);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelGlobalTest, InvalidNotAKernel) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_SUCCESS(
      olGetSymbol(Program, "global", OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Global));
  ASSERT_ERROR(OL_ERRC_SYMBOL_KIND,
               olLaunchKernel(Queue, Device, Global, nullptr, 0, &LaunchArgs));
}

TEST_P(olLaunchKernelGlobalCtorTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i + 100);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelGlobalDtorTest, Success) {
  // TODO: We can't inspect the result of a destructor yet, once we
  // find/implement a way, update this test. For now we just check that nothing
  // crashes
  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, nullptr, 0, &LaunchArgs));
  ASSERT_SUCCESS(olSyncQueue(Queue));
}
