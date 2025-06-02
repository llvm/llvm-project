//===------- Offload API tests - olLaunchKernelSuggestedGroupSize ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

static constexpr uint32_t COMBOS[6][4] = {
    {1, 64, 1, 1},  {1, 63, 1, 1},   {2, 64, 64, 1},
    {2, 40, 40, 1}, {3, 64, 64, 64}, {3, 128, 20, 12},
};

struct olLaunchKernelSuggestedGroupSizeTest : OffloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));
    ASSERT_SUCCESS(olGetKernel(Program, "foo", &Kernel));
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
};

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchKernelSuggestedGroupSizeTest);

TEST_P(olLaunchKernelSuggestedGroupSizeTest, Success) {
  for (auto C : COMBOS) {
    std::string scope{};
    llvm::raw_string_ostream os{scope};
    os << "{ " << C[0] << ", " << C[1] << ", " << C[2] << ", " << C[3] << "}";
    os.flush();
    SCOPED_TRACE(scope);

    auto NumItems = C[1] * C[2] * C[3];

    ol_kernel_launch_size_suggested_args_t LaunchArgs{};
    LaunchArgs.Dimensions = C[0];
    LaunchArgs.NumItemsX = C[1];
    LaunchArgs.NumItemsY = C[2];
    LaunchArgs.NumItemsZ = C[3];

    void *Mem;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                              NumItems * sizeof(int), &Mem));
    struct {
      void *Mem;
    } Args{Mem};

    ASSERT_SUCCESS(olLaunchKernelSuggestedGroupSize(
        Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs, nullptr));

    ASSERT_SUCCESS(olWaitQueue(Queue));

    int *Data = (int *)Mem;
    for (int i = 0; i < static_cast<int>(NumItems); i++) {
      ASSERT_EQ(Data[i], i);
    }

    ASSERT_SUCCESS(olMemFree(Mem));
  }
}

TEST_P(olLaunchKernelSuggestedGroupSizeTest, SuccessSynchronous) {
  for (auto C : COMBOS) {
    std::string scope{};
    llvm::raw_string_ostream os{scope};
    os << "{ " << C[0] << ", " << C[1] << ", " << C[2] << ", " << C[3] << "}";
    os.flush();
    SCOPED_TRACE(scope);

    auto NumItems = C[1] * C[2] * C[3];

    ol_kernel_launch_size_suggested_args_t LaunchArgs{};
    LaunchArgs.Dimensions = C[0];
    LaunchArgs.NumItemsX = C[1];
    LaunchArgs.NumItemsY = C[2];
    LaunchArgs.NumItemsZ = C[3];

    void *Mem;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                              NumItems * sizeof(int), &Mem));
    struct {
      void *Mem;
    } Args{Mem};

    ASSERT_SUCCESS(olLaunchKernelSuggestedGroupSize(
        nullptr, Device, Kernel, &Args, sizeof(Args), &LaunchArgs, nullptr));

    ASSERT_SUCCESS(olWaitQueue(Queue));

    int *Data = (int *)Mem;
    for (int i = 0; i < static_cast<int>(NumItems); i++) {
      ASSERT_EQ(Data[i], i);
    }

    ASSERT_SUCCESS(olMemFree(Mem));
  }
}
