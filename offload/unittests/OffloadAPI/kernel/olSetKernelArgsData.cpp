//===------- Offload API tests - olSetKernelArgsData ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olSetKernelArgsDataTest = offloadKernelTest;

// Don't actually test execution of the kernel in these tests, that is covered
// by the olEnqueueKernelLaunch tests

TEST_F(olSetKernelArgsDataTest, Success) {
  // Kernel takes a single int* argument
  void *Ptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 64, &Ptr));
  std::vector<uintptr_t> ArgsData;
  ArgsData.push_back(reinterpret_cast<uintptr_t>(Ptr));

  ASSERT_SUCCESS(olSetKernelArgsData(Kernel, ArgsData.data(),
                                     ArgsData.size() * sizeof(uintptr_t)));
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Ptr));
}

TEST_F(olSetKernelArgsDataTest, InvalidNullHandle) {
  void *Ptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 64, &Ptr));
  std::vector<uintptr_t> ArgsData;
  ArgsData.push_back(reinterpret_cast<uintptr_t>(Ptr));

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olSetKernelArgsData(nullptr, ArgsData.data(),
                                   ArgsData.size() * sizeof(uintptr_t)));
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Ptr));
}
