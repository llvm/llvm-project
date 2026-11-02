//===------- Offload API tests - olGetSymbolInfo --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

#include <atomic>
#include <thread>

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

TEST_P(olGetSymbolInfoKernelTest, SuccessName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetSymbolInfoSize(Kernel, OL_SYMBOL_INFO_NAME, &Size));
  std::vector<char> Name(Size);
  ASSERT_SUCCESS(
      olGetSymbolInfo(Kernel, OL_SYMBOL_INFO_NAME, Size, Name.data()));
  ASSERT_STREQ(Name.data(), "foo");
}

TEST_P(olGetSymbolInfoGlobalTest, SuccessName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetSymbolInfoSize(Global, OL_SYMBOL_INFO_NAME, &Size));
  std::vector<char> Name(Size);
  ASSERT_SUCCESS(
      olGetSymbolInfo(Global, OL_SYMBOL_INFO_NAME, Size, Name.data()));
  ASSERT_STREQ(Name.data(), "global");
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

using olGetSymbolInfoImageVariableTest = OffloadImageVariableTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetSymbolInfoImageVariableTest);

TEST_P(olGetSymbolInfoImageVariableTest, SuccessKind) {
  ol_symbol_kind_t RetrievedKind;
  ASSERT_SUCCESS(olGetSymbolInfo(Image, OL_SYMBOL_INFO_KIND,
                                 sizeof(RetrievedKind), &RetrievedKind));
  ASSERT_EQ(RetrievedKind, OL_SYMBOL_KIND_GLOBAL_VARIABLE);
}

TEST_P(olGetSymbolInfoImageVariableTest, SuccessSize) {
  size_t RetrievedSize = 0;
  ASSERT_SUCCESS(olGetSymbolInfo(Image, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                                 sizeof(RetrievedSize), &RetrievedSize));
  ASSERT_EQ(RetrievedSize, 4 * sizeof(uint32_t));
}

TEST_P(olGetSymbolInfoImageVariableTest, SuccessAddressContents) {
  void *RetrievedAddr = nullptr;
  ASSERT_SUCCESS(olGetSymbolInfo(Image,
                                 OL_SYMBOL_INFO_GLOBAL_VARIABLE_HOST_ADDRESS,
                                 sizeof(RetrievedAddr), &RetrievedAddr));
  ASSERT_NE(RetrievedAddr, nullptr);
  const uint32_t *Data = static_cast<const uint32_t *>(RetrievedAddr);
  for (uint32_t I = 0; I < 4; I++)
    ASSERT_EQ(Data[I], I + 1);
}

// The symbol's addresses and size are resolved lazily; hammer the first query
// from many threads at once and check they all observe the same values.
TEST_P(olGetSymbolInfoImageVariableTest, ConcurrentLazyResolution) {
  constexpr size_t NumThreads = 16;
  struct Result {
    bool Ok = false;
    void *DeviceAddr = nullptr;
    void *HostAddr = nullptr;
    size_t Size = 0;
  };
  std::vector<Result> Results(NumThreads);
  std::atomic<bool> Go{false};

  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; I++)
    Threads.emplace_back([&, I] {
      while (!Go.load())
        std::this_thread::yield();
      Result &R = Results[I];
      // Vary the query order so every first-query path is exercised.
      for (size_t Step = 0; Step < 3; Step++) {
        ol_result_t Err = nullptr;
        switch ((I + Step) % 3) {
        case 0:
          Err = olGetSymbolInfo(Image, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                                sizeof(R.Size), &R.Size);
          break;
        case 1:
          Err = olGetSymbolInfo(Image, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
                                sizeof(R.DeviceAddr), &R.DeviceAddr);
          break;
        case 2:
          Err = olGetSymbolInfo(Image,
                                OL_SYMBOL_INFO_GLOBAL_VARIABLE_HOST_ADDRESS,
                                sizeof(R.HostAddr), &R.HostAddr);
          break;
        }
        if (Err)
          return;
      }
      R.Ok = true;
    });
  Go = true;
  for (auto &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_TRUE(R.Ok);
    ASSERT_EQ(R.Size, Results[0].Size);
    ASSERT_EQ(R.DeviceAddr, Results[0].DeviceAddr);
    ASSERT_EQ(R.HostAddr, Results[0].HostAddr);
  }
  ASSERT_EQ(Results[0].Size, 4 * sizeof(uint32_t));
  ASSERT_NE(Results[0].HostAddr, nullptr);
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
