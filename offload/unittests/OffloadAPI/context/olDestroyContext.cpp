//===------- Offload API tests - olDestroyContext -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyContextTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyContextTest);

TEST_P(olDestroyContextTest, Success) {
  ol_context_handle_t Ctx = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Device, &Ctx));
  ASSERT_SUCCESS(olDestroyContext(Ctx));
}

TEST_P(olDestroyContextTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyContext(nullptr));
}
