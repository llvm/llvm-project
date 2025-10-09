//===------- Offload API tests - olGetMemInfoSize -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

struct olGetMemInfoSizeTest : OffloadDeviceTest {
  void *OffsetPtr() { return &reinterpret_cast<char *>(Ptr)[123]; }

  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 0x1024, &Ptr));
  }

  void TearDown() override {
    ASSERT_SUCCESS(olMemFree(Ptr));
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::TearDown());
  }

  void *Ptr;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetMemInfoSizeTest);

TEST_P(olGetMemInfoSizeTest, SuccessDevice) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetMemInfoSize(Ptr, OL_MEM_INFO_DEVICE, &Size));
  ASSERT_EQ(Size, sizeof(ol_device_handle_t));
}

TEST_P(olGetMemInfoSizeTest, SuccessBase) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetMemInfoSize(Ptr, OL_MEM_INFO_BASE, &Size));
  ASSERT_EQ(Size, sizeof(void *));
}

TEST_P(olGetMemInfoSizeTest, SuccessSize) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetMemInfoSize(Ptr, OL_MEM_INFO_SIZE, &Size));
  ASSERT_EQ(Size, sizeof(size_t));
}

TEST_P(olGetMemInfoSizeTest, SuccessType) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetMemInfoSize(Ptr, OL_MEM_INFO_TYPE, &Size));
  ASSERT_EQ(Size, sizeof(ol_alloc_type_t));
}

TEST_P(olGetMemInfoSizeTest, InvalidSymbolInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetMemInfoSize(Ptr, OL_MEM_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetMemInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetMemInfoSize(Ptr, OL_MEM_INFO_DEVICE, nullptr));
}
