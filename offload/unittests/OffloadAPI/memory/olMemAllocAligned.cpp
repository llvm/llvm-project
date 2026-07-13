//===--------------- Offload API tests - olMemAllocAligned ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemAllocAlignedTest = OffloadDeviceTest;

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemAllocAlignedTest);

constexpr size_t DefaultAlignment = 16;
constexpr size_t TestAllocsNum = 1000;

TEST_P(olMemAllocAlignedTest, SuccessAllocMany) {
  std::vector<void *> Allocs;
  Allocs.reserve(1000);

  for (size_t I = 1; I < TestAllocsNum; I++) {
    void *Alloc = nullptr;
    switch (I % 3) {
    case 0:
      ASSERT_SUCCESS(
          olMemAllocAlignedDevice(Device, 1024 * I, DefaultAlignment, &Alloc));
      break;
    case 1:
      ASSERT_SUCCESS(
          olMemAllocAlignedManaged(Device, 1024 * I, DefaultAlignment, &Alloc));
      break;
    case 2:
      ASSERT_SUCCESS(
          olMemAllocAlignedHost(Device, 1024 * I, DefaultAlignment, &Alloc));
      break;
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
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_HANDLE,
      olMemAllocAlignedDevice(nullptr, 1024, DefaultAlignment, &Alloc));
}

TEST_P(olMemAllocAlignedTest, InvalidNullOutPtr) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olMemAllocAlignedDevice(Device, 1024, DefaultAlignment, nullptr));
}

TEST_P(olMemAllocAlignedTest, InvalidAlignmentZero) {
  void *Alloc = nullptr;

  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olMemAllocAlignedDevice(Device, 1024, 0, &Alloc));
}

TEST_P(olMemAllocAlignedTest, InvalidAlignmentNotAPowerOfTwo) {
  void *Alloc = nullptr;

  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olMemAllocAlignedDevice(Device, 1024, 3, &Alloc));
}

TEST_P(olMemAllocAlignedTest, CudaExceedDefaultAlignment) {
  if (getPlatformBackend() != OL_PLATFORM_BACKEND_CUDA) {
    GTEST_SKIP() << "Test inteded for CUDA backend";
  }

  void *Alloc = nullptr;
  // The default page size for cuda is 64 KB.
  ASSERT_ERROR(OL_ERRC_UNSUPPORTED,
               olMemAllocAlignedDevice(Device, 1024, 1024 * 64 * 64, &Alloc));
  ASSERT_EQ(Alloc, nullptr);
}

TEST_P(olMemAllocAlignedTest, SuccessAllocManagedDifferentAlignments) {
  void *Alloc = nullptr;
  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;

  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));
    ASSERT_SUCCESS(olMemAllocAlignedManaged(Device, 1024, Alignment, &Alloc));
    ASSERT_NE(Alloc, nullptr);
    olMemFree(Alloc);
  }
}

TEST_P(olMemAllocAlignedTest, SuccessAllocHostDifferentAlignments) {
  void *Alloc = nullptr;
  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;

  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));
    ASSERT_SUCCESS(olMemAllocAlignedHost(Device, 1024, Alignment, &Alloc));
    ASSERT_NE(Alloc, nullptr);
    olMemFree(Alloc);
  }
}

TEST_P(olMemAllocAlignedTest, SuccessAllocDeviceDifferentAlignments) {
  void *Alloc = nullptr;
  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;

  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));
    ASSERT_SUCCESS(olMemAllocAlignedDevice(Device, 1024, Alignment, &Alloc));
    ASSERT_NE(Alloc, nullptr);

    olMemFree(Alloc);
  }
}

TEST_P(olMemAllocAlignedTest, SuccessMemcpyManagedDiferentAlignments) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;
  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));

    ASSERT_SUCCESS(olMemAllocAlignedManaged(Device, Size, Alignment, &Alloc));
    // memcpy is synchronous when queue is unspecified.
    ASSERT_SUCCESS(olMemcpy(nullptr, Alloc, Device, Input.data(), Host, Size));
    ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Alloc, Device, Size));

    for (uint8_t Val : Output) {
      ASSERT_EQ(Val, 42);
    }

    ASSERT_SUCCESS(olMemFree(Alloc));
  }
}

TEST_P(olMemAllocAlignedTest, SuccessMemcpyDeviceDiferentAlignments) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;
  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));

    ASSERT_SUCCESS(olMemAllocAlignedDevice(Device, Size, Alignment, &Alloc));
    // memcpy is synchronous when queue is unspecified.
    ASSERT_SUCCESS(olMemcpy(nullptr, Alloc, Device, Input.data(), Host, Size));
    ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Alloc, Device, Size));

    for (uint8_t Val : Output) {
      ASSERT_EQ(Val, 42);
    }

    ASSERT_SUCCESS(olMemFree(Alloc));
  }
}

TEST_P(olMemAllocAlignedTest, SuccessMemcpyHostDiferentAlignments) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  size_t NumAlignments = 6;
  size_t Alignments[] = {8, 16, 32, 64, 128, 256};
  size_t Alignment;
  for (size_t i = 0; i < NumAlignments; i++) {
    Alignment = Alignments[i];
    SCOPED_TRACE("alignment: " + std::to_string(Alignment));

    ASSERT_SUCCESS(olMemAllocAlignedHost(Device, Size, Alignment, &Alloc));
    // memcpy is synchronous when queue is unspecified.
    ASSERT_SUCCESS(olMemcpy(nullptr, Alloc, Device, Input.data(), Host, Size));
    ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Alloc, Device, Size));

    for (uint8_t Val : Output) {
      ASSERT_EQ(Val, 42);
    }

    ASSERT_SUCCESS(olMemFree(Alloc));
  }
}
