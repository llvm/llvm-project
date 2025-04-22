//===------- Offload API tests - olMemFree --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemFreeTest = OffloadDeviceTest;

TEST_F(olMemFreeTest, SuccessFreeManaged) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_F(olMemFreeTest, SuccessFreeHost) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_F(olMemFreeTest, SuccessFreeDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_F(olMemFreeTest, InvalidNullPtr) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olMemFree(nullptr));
  ASSERT_SUCCESS(olMemFree(Alloc));
}
