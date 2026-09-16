//===------- Offload API tests - olRegisterRPCCallback --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olRegisterRPCCallbackTest = OffloadPlatformTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olRegisterRPCCallbackTest);

static unsigned callback(void *Raw, unsigned NumLanes) {}

TEST_P(olRegisterRPCCallbackTest, SuccessBackend) {
  ASSERT_SUCCESS(olPlatformRegisterRPCCallback(Platform, &callback));
}

TEST_P(olRegisterRPCCallbackTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olPlatformRegisterRPCCallback(nullptr, &callback));
}

TEST_P(olRegisterRPCCallbackTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olPlatformRegisterRPCCallback(Platform, nullptr));
}
