//===--------------- Offload API tests - olMemAllocAligned ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Properties.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemAllocAlignedTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemAllocAlignedTest);

struct olMemAllocAlignedTypesTest
    : OffloadDeviceTestWithParam<ol_alloc_type_t> {
  ol_result_t allocateDeviceOrHost(size_t Size, size_t Alignment,
                                   void **Alloc) {
    ol_alloc_type_t AllocType = getTestParam();
    if (AllocType == OL_ALLOC_TYPE_HOST) {
      return olMemAllocAlignedHost(this->Device, Size, Alignment, Alloc);
    }

    return olMemAllocAligned(this->Device, AllocType, Size, Alignment, Alloc);
  }
};

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olMemAllocAlignedTypesTest, AllocTypes,
    defaultPrinterWithParam<ol_alloc_type_t>);

constexpr size_t DefaultAlignment = 16;

TEST_P(olMemAllocAlignedTest, SuccessAllocMany) {
  std::vector<void *> Allocs;
  Allocs.reserve(TestAllocsNum);

  for (size_t I = 1; I < TestAllocsNum; I++) {
    void *Alloc = nullptr;
    ol_alloc_type_t AllocType = AllocTypes[I % 3];
    if (AllocType == OL_ALLOC_TYPE_HOST) {
      ASSERT_SUCCESS(olMemAllocAlignedHost(Device, DefaultAllocSize * I,
                                           DefaultAlignment, &Alloc));
    } else {
      ASSERT_SUCCESS(olMemAllocAligned(Device, AllocType, DefaultAllocSize * I,
                                       DefaultAlignment, &Alloc));
    }
    ASSERT_NE(Alloc, nullptr);

    Allocs.push_back(Alloc);
  }

  for (auto *A : Allocs) {
    olMemFree(A);
  }
}

TEST_P(olMemAllocAlignedTest, InvalidNullDevice) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAllocAligned(nullptr, OL_ALLOC_TYPE_DEVICE, 1024,
                                 DefaultAlignment, &Alloc));
}

TEST_P(olMemAllocAlignedTest, InvalidNullOutPtr) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAllocAligned(Device, OL_ALLOC_TYPE_DEVICE, 1024,
                                 DefaultAlignment, nullptr));
}

TEST_P(olMemAllocAlignedTest, InvalidAlignmentZero) {
  void *Alloc = nullptr;

  ASSERT_ERROR(
      OL_ERRC_INVALID_ARGUMENT,
      olMemAllocAligned(Device, OL_ALLOC_TYPE_DEVICE, 1024, 0, &Alloc));
}

TEST_P(olMemAllocAlignedTest, InvalidAlignmentNotAPowerOfTwo) {
  void *Alloc = nullptr;

  ASSERT_ERROR(
      OL_ERRC_INVALID_ARGUMENT,
      olMemAllocAligned(Device, OL_ALLOC_TYPE_DEVICE, 1024, 3, &Alloc));
}

TEST_P(olMemAllocAlignedTest, InvalidHostType) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olMemAllocAligned(Device, OL_ALLOC_TYPE_HOST, 1024,
                                 DefaultAlignment, &Alloc));
}

TEST_P(olMemAllocAlignedTest, CudaExceedDefaultAlignment) {
  if (getPlatformBackend() != OL_PLATFORM_BACKEND_CUDA) {
    GTEST_SKIP() << "Test inteded for CUDA backend";
  }

  void *Alloc = nullptr;
  // The default page size for cuda is 64 KB.
  ASSERT_ERROR(OL_ERRC_UNSUPPORTED,
               olMemAllocAligned(Device, OL_ALLOC_TYPE_DEVICE, 1024,
                                 1024 * 64 * 64 * 64, &Alloc));
  ASSERT_EQ(Alloc, nullptr);
}

TEST_P(olMemAllocAlignedTypesTest, SuccessAllocDifferentAlignments) {
  void *Alloc = nullptr;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t NumAlignments = sizeof(Alignments) / sizeof(Alignments[0]);
  size_t Alignment;

  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));
    ASSERT_SUCCESS(allocateDeviceOrHost(DefaultAllocSize, Alignment, &Alloc));
    ASSERT_NE(Alloc, nullptr);
    olMemFree(Alloc);
  }
}

TEST_P(olMemAllocAlignedTypesTest, SuccessMemcpyDiferentAlignments) {
  void *Alloc;
  std::vector<uint8_t> Input(DefaultAllocSize, 42);
  std::vector<uint8_t> Output(DefaultAllocSize, 0);

  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;
  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));
    ASSERT_SUCCESS(allocateDeviceOrHost(DefaultAllocSize, Alignment, &Alloc));
    // memcpy is synchronous when queue is unspecified.
    ASSERT_SUCCESS(
        olMemcpy(nullptr, Alloc, Device, Input.data(), Host, DefaultAllocSize));
    ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Alloc, Device,
                            DefaultAllocSize));

    for (uint8_t Val : Output) {
      ASSERT_EQ(Val, 42);
    }

    ASSERT_SUCCESS(olMemFree(Alloc));
  }
}
