//===------- Offload API tests - olGetSymbolInfoSize ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Properties.hpp"

using olGetSymbolInfoSizeKernelTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetSymbolInfoSizeKernelTest);

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olGetSymbolInfoSizeGlobalTest, SymbolGlobalProperties,
    defaultPropertyTestPrinter<ol_symbol_info_t>);

TEST_P(olGetSymbolInfoSizeKernelTest, SuccessPropertySize) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetSymbolInfoSize(Kernel, OL_SYMBOL_INFO_KIND, &Size));
  ASSERT_EQ(Size, sizeof(ol_symbol_kind_t));
}

TEST_P(olGetSymbolInfoSizeGlobalTest, SuccessPropertySize) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetSymbolInfoSize(Global, Property, &Size));
  ASSERT_EQ(Size, PropertySize);
}

TEST_P(olGetSymbolInfoSizeKernelTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetSymbolInfoSize(nullptr, OL_SYMBOL_INFO_KIND, &Size));
}

TEST_P(olGetSymbolInfoSizeKernelTest, InvalidSymbolInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetSymbolInfoSize(Kernel, OL_SYMBOL_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetSymbolInfoSizeKernelTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetSymbolInfoSize(Kernel, OL_SYMBOL_INFO_KIND, nullptr));
}
