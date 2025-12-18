//===------- Offload API tests - olLaunchKernel Cooperative ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olLaunchKernelCooperativeTest : LaunchSingleKernelTestBase {
  void SetUp() override {
    SetUpKernel("noargs");

    bool SupportsCooperative = false;
    auto Result =
        olGetDeviceInfo(Device, OL_DEVICE_INFO_COOPERATIVE_LAUNCH_SUPPORT,
                        sizeof(bool), &SupportsCooperative);

    if (Result) {
      if (Result->Code == OL_ERRC_UNIMPLEMENTED) {
        GTEST_SKIP()
            << "Device does not provide cooperative launch support information";
      }
      if (Result->Code != OL_ERRC_SUCCESS) {
        GTEST_FAIL() << "olGetDeviceInfo failed with unexpected error: "
                     << Result->Code << ": " << Result->Details;
      }
    }

    if (!SupportsCooperative) {
      GTEST_SKIP() << "Device does not support cooperative kernel launch";
    }
  }
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchKernelCooperativeTest);

TEST_P(olLaunchKernelCooperativeTest, GetMaxCooperativeGroupCount) {
  uint32_t MaxGroupCount = 0;
  ASSERT_SUCCESS(olGetKernelMaxCooperativeGroupCount(
      Device, Kernel, &LaunchArgs, &MaxGroupCount));
  ASSERT_GT(MaxGroupCount, 0u);
}

TEST_P(olLaunchKernelCooperativeTest, SuccessCooperative) {
  uint32_t MaxGroupCount = 0;
  ASSERT_SUCCESS(olGetKernelMaxCooperativeGroupCount(
      Device, Kernel, &LaunchArgs, &MaxGroupCount));

  LaunchArgs.NumGroups.x = MaxGroupCount;

  bool IsCooperative = true;
  ol_kernel_launch_prop_t Props[] = {
      {OL_KERNEL_LAUNCH_PROP_TYPE_IS_COOPERATIVE, &IsCooperative},
      OL_KERNEL_LAUNCH_PROP_END};

  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, nullptr, 0, &LaunchArgs, Props));
  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olLaunchKernelCooperativeTest, SuccessNonCooperative) {
  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, nullptr, 0, &LaunchArgs, nullptr));
  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olLaunchKernelCooperativeTest, TooManyGroups) {
  uint32_t MaxGroupCount = 0;
  ASSERT_SUCCESS(olGetKernelMaxCooperativeGroupCount(
      Device, Kernel, &LaunchArgs, &MaxGroupCount));

  LaunchArgs.NumGroups.x = MaxGroupCount * 2;

  bool IsCooperative = true;
  ol_kernel_launch_prop_t Props[] = {
      {OL_KERNEL_LAUNCH_PROP_TYPE_IS_COOPERATIVE, &IsCooperative},
      OL_KERNEL_LAUNCH_PROP_END};

  ASSERT_ANY_ERROR(
      olLaunchKernel(Queue, Device, Kernel, nullptr, 0, &LaunchArgs, Props));
}

TEST_P(olLaunchKernelCooperativeTest, SynchronousLaunch) {
  uint32_t MaxGroupCount = 0;
  ASSERT_SUCCESS(olGetKernelMaxCooperativeGroupCount(
      Device, Kernel, &LaunchArgs, &MaxGroupCount));

  LaunchArgs.NumGroups.x = std::min(MaxGroupCount, 2u);

  bool IsCooperative = true;
  ol_kernel_launch_prop_t Props[] = {
      {OL_KERNEL_LAUNCH_PROP_TYPE_IS_COOPERATIVE, &IsCooperative},
      OL_KERNEL_LAUNCH_PROP_END};

  ASSERT_SUCCESS(
      olLaunchKernel(nullptr, Device, Kernel, nullptr, 0, &LaunchArgs, Props));
}

TEST_P(olLaunchKernelCooperativeTest, InvalidNullHandleKernel) {
  uint32_t MaxGroupCount = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernelMaxCooperativeGroupCount(Device, nullptr, &LaunchArgs,
                                                   &MaxGroupCount));
}

TEST_P(olLaunchKernelCooperativeTest, InvalidNullHandleDevice) {
  uint32_t MaxGroupCount = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernelMaxCooperativeGroupCount(nullptr, Kernel, &LaunchArgs,
                                                   &MaxGroupCount));
}

TEST_P(olLaunchKernelCooperativeTest, InvalidNullPointerGroupCountRet) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernelMaxCooperativeGroupCount(Device, Kernel, &LaunchArgs,
                                                   nullptr));
}
