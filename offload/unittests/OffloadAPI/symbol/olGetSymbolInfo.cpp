//===------- Offload API tests - olGetSymbolInfo --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetSymbolInfoKernelTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetSymbolInfoKernelTest);

using olGetSymbolInfoGlobalTest = OffloadGlobalTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetSymbolInfoGlobalTest);

TEST_P(olGetSymbolInfoKernelTest, SuccessKind) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_SUCCESS(olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_KIND,
                                 sizeof(RetrievedKind), &RetrievedKind));
  ASSERT_EQ(RetrievedKind, OL_SYMBOL_KIND_KERNEL);
}

TEST_P(olGetSymbolInfoGlobalTest, SuccessKind) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_SUCCESS(olGetSymbolInfo(Global, OL_SYMBOL_INFO_KIND,
                                 sizeof(RetrievedKind), &RetrievedKind));
  ASSERT_EQ(RetrievedKind, OL_SYMBOL_KIND_GLOBAL_VARIABLE);
}

TEST_P(olGetSymbolInfoKernelTest, InvalidAddress) {
  void *RetrievedAddr;
  ASSERT_ERROR(OL_ERRC_SYMBOL_KIND,
               olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
                               sizeof(RetrievedAddr), &RetrievedAddr));
}

TEST_P(olGetSymbolInfoGlobalTest, SuccessAddress) {
  void *RetrievedAddr = nullptr;
  ASSERT_SUCCESS(olGetSymbolInfo(Global, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
                                 sizeof(RetrievedAddr), &RetrievedAddr));
  ASSERT_NE(RetrievedAddr, nullptr);
}

TEST_P(olGetSymbolInfoKernelTest, InvalidSize) {
  size_t RetrievedSize;
  ASSERT_ERROR(OL_ERRC_SYMBOL_KIND,
               olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                               sizeof(RetrievedSize), &RetrievedSize));
}

TEST_P(olGetSymbolInfoGlobalTest, SuccessSize) {
  size_t RetrievedSize = 0;
  ASSERT_SUCCESS(olGetSymbolInfo(Global, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                                 sizeof(RetrievedSize), &RetrievedSize));
  ASSERT_EQ(RetrievedSize, 64 * sizeof(uint32_t));
}

TEST_P(olGetSymbolInfoKernelTest, InvalidNullHandle) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetSymbolInfo(nullptr, OL_SYMBOL_INFO_KIND,
                               sizeof(RetrievedKind), &RetrievedKind));
}

TEST_P(olGetSymbolInfoKernelTest, InvalidSymbolInfoEnumeration) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_FORCE_UINT32,
                               sizeof(RetrievedKind), &RetrievedKind));
}

TEST_P(olGetSymbolInfoKernelTest, InvalidSizeZero) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_KIND, 0, &RetrievedKind));
}

TEST_P(olGetSymbolInfoKernelTest, InvalidSizeSmall) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_KIND,
                               sizeof(RetrievedKind) - 1, &RetrievedKind));
}

TEST_P(olGetSymbolInfoKernelTest, InvalidNullPointerPropValue) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_KIND,
                               sizeof(RetrievedKind), nullptr));
}
