//===------- Offload API tests - olMemcpyRect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

constexpr ol_dimensions_t FULL_SIZE = {16, 8, 4};
constexpr size_t BYTES = FULL_SIZE.x * FULL_SIZE.y * FULL_SIZE.z;

constexpr ol_dimensions_t COPY_SIZE = {4, 3, 2};
constexpr ol_dimensions_t COPY_OFFSET = {8, 2, 1};

struct olMemcpyRectTest : OffloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());

    ol_platform_handle_t Platform;
    ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                   sizeof(Platform), &Platform));
    ol_platform_backend_t Backend;
    ASSERT_SUCCESS(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                     sizeof(Backend), &Backend));
    if (Backend == OL_PLATFORM_BACKEND_CUDA)
      GTEST_SKIP() << "CUDA does not yet support this entry point\n";

    Buff.fill('h');
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, BYTES, &HostPtr));
    ASSERT_SUCCESS(
        olMemcpy(nullptr, HostPtr, Device, Buff.data(), Host, BYTES));

    Buff.fill('d');
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, BYTES, &DevicePtr));
    ASSERT_SUCCESS(
        olMemcpy(nullptr, DevicePtr, Device, Buff.data(), Host, BYTES));

    Buff.fill('D');
    ASSERT_SUCCESS(
        olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, BYTES, &DevicePtr2));
    ASSERT_SUCCESS(
        olMemcpy(nullptr, DevicePtr2, Device, Buff.data(), Host, BYTES));

    SrcRect.offset = DstRect.offset = COPY_OFFSET;
    SrcRect.pitch = DstRect.pitch = FULL_SIZE.x;
    SrcRect.slice = DstRect.slice = FULL_SIZE.y * FULL_SIZE.x;
  }

  void TearDown() override {
    ASSERT_SUCCESS(olMemFree(HostPtr));
    ASSERT_SUCCESS(olMemFree(DevicePtr));
  }

  void checkPattern(void *CheckBuffer, const char *Template) {
    ASSERT_SUCCESS(
        olMemcpy(nullptr, Buff.data(), Host, CheckBuffer, Device, BYTES));
    bool Failed = false;

    for (size_t I = 0; I < BYTES; I++) {
      if (Buff[I] != Template[I]) {
        ADD_FAILURE() << "Failure at location " << I << "\n";
        Failed = true;
        break;
      }
    }

    if (Failed) {
      std::cerr << "Expected:\n";
      printSlices([&](size_t I) -> char { return Template[I]; });
      std::cerr << "Got:\n";
      printSlices([&](size_t I) -> char { return Buff[I]; });
      std::cerr << "Delta:\n";
      printSlices(
          [&](size_t I) -> char { return Buff[I] == Template[I] ? '.' : 'X'; });
    }
  }

  template <typename F> void printSlices(F Getter) {
    for (size_t Y = 0; Y < FULL_SIZE.y; Y++) {
      for (size_t Z = 0; Z < FULL_SIZE.z; Z++) {
        for (size_t X = 0; X < FULL_SIZE.x; X++) {
          std::cerr << Getter(X + (Y * FULL_SIZE.x) +
                              (Z * FULL_SIZE.y * FULL_SIZE.x));
        }
        std::cerr << "    ";
      }

      std::cerr << "\n";
    }
  }

  std::array<uint8_t, BYTES> Buff;
  void *HostPtr;
  void *DevicePtr;
  void *DevicePtr2;
  ol_memcpy_rect_t SrcRect;
  ol_memcpy_rect_t DstRect;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemcpyRectTest);

TEST_P(olMemcpyRectTest, SuccessHtoD) {
  DstRect.buffer = DevicePtr;
  SrcRect.buffer = HostPtr;

  ASSERT_SUCCESS(
      olMemcpyRect(Queue, DstRect, Device, SrcRect, Host, COPY_SIZE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  // clang-format off
  checkPattern(DevicePtr,
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"

    "dddddddddddddddd"
    "dddddddddddddddd"
    "ddddddddhhhhdddd"
    "ddddddddhhhhdddd"
    "ddddddddhhhhdddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"

    "dddddddddddddddd"
    "dddddddddddddddd"
    "ddddddddhhhhdddd"
    "ddddddddhhhhdddd"
    "ddddddddhhhhdddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"

    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
  );
  // clang-format on
}

TEST_P(olMemcpyRectTest, SuccessDtoH) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = DevicePtr;

  ASSERT_SUCCESS(
      olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  // clang-format off
  checkPattern(HostPtr,
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"

    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhddddhhhh"
    "hhhhhhhhddddhhhh"
    "hhhhhhhhddddhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"

    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhddddhhhh"
    "hhhhhhhhddddhhhh"
    "hhhhhhhhddddhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"

    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
    "hhhhhhhhhhhhhhhh"
  );
  // clang-format on
}

TEST_P(olMemcpyRectTest, SuccessDtoD) {
  DstRect.buffer = DevicePtr;
  SrcRect.buffer = DevicePtr2;

  ASSERT_SUCCESS(
      olMemcpyRect(Queue, DstRect, Device, SrcRect, Device, COPY_SIZE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  // clang-format off
  checkPattern(DevicePtr,
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"

    "dddddddddddddddd"
    "dddddddddddddddd"
    "ddddddddDDDDdddd"
    "ddddddddDDDDdddd"
    "ddddddddDDDDdddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"

    "dddddddddddddddd"
    "dddddddddddddddd"
    "ddddddddDDDDdddd"
    "ddddddddDDDDdddd"
    "ddddddddDDDDdddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"

    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
    "dddddddddddddddd"
  );
  // clang-format on
}

TEST_P(olMemcpyRectTest, InvalidDstPtr) {
  DstRect.buffer = nullptr;
  SrcRect.buffer = HostPtr;

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemcpyRect(Queue, DstRect, Device, SrcRect, Host, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidSrcPtr) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = nullptr;

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemcpyRect(Queue, DstRect, Device, SrcRect, Host, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidDstDevice) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemcpyRect(Queue, DstRect, nullptr, SrcRect, Host, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidSrcDevice) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, nullptr, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidSize) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, {0, 0, 0}));
}

TEST_P(olMemcpyRectTest, InvalidSrcPtrAlign) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = &static_cast<char *>(DevicePtr)[2];

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidDstPtrAlign) {
  DstRect.buffer = &static_cast<char *>(HostPtr)[2];
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidDstPitchAlign) {
  DstRect.buffer = HostPtr;
  DstRect.pitch = 2;
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidSrcPitchAlign) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = DevicePtr;
  SrcRect.pitch = 2;

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidDstSliceAlign) {
  DstRect.buffer = HostPtr;
  DstRect.slice = 2;
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidSrcSliceAlign) {
  DstRect.buffer = HostPtr;
  SrcRect.buffer = DevicePtr;
  SrcRect.slice = 2;

  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidDstUnalloc) {
  DstRect.buffer = Buff.data();
  SrcRect.buffer = DevicePtr;

  ASSERT_ERROR(OL_ERRC_INVALID_VALUE,
               olMemcpyRect(Queue, DstRect, Host, SrcRect, Device, COPY_SIZE));
}

TEST_P(olMemcpyRectTest, InvalidSrcUnalloc) {
  DstRect.buffer = DevicePtr;
  SrcRect.buffer = Buff.data();

  ASSERT_ERROR(OL_ERRC_INVALID_VALUE,
               olMemcpyRect(Queue, DstRect, Device, SrcRect, Host, COPY_SIZE));
}
