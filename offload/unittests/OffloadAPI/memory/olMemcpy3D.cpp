//===------- Offload API tests - olMemcpy3D -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using olMemcpy3DTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemcpy3DTest);

namespace {
ol_result_t copy3D(ol_queue_handle_t Queue, void *DstPtr,
                   ol_device_handle_t DstDevice, ol_dimensions_t DstOffset,
                   size_t DstPitch, size_t DstSlicePitch, const void *SrcPtr,
                   ol_device_handle_t SrcDevice, ol_dimensions_t SrcOffset,
                   size_t SrcPitch, size_t SrcSlicePitch,
                   ol_dimensions_t Size) {
  ol_memcpy3d_layout_t Dst = {DstOffset, DstPitch, DstSlicePitch};
  ol_memcpy3d_layout_t Src = {SrcOffset, SrcPitch, SrcSlicePitch};
  return olMemcpy3D(Queue, DstPtr, DstDevice, &Dst, SrcPtr, SrcDevice, &Src,
                    Size);
}
constexpr size_t Pitch = 32;
constexpr size_t Rows = 8;
constexpr size_t Slices = 4;
constexpr size_t SlicePitch = Pitch * Rows;
constexpr size_t Total = SlicePitch * Slices;
} // namespace

TEST_P(olMemcpy3DTest, Success3DRoundTrip) {
  std::vector<uint8_t> Input(Total);
  std::iota(Input.begin(), Input.end(), 0);
  std::vector<uint8_t> Output(Total, 0);

  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Total, &Alloc));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Total));

  // Copy a 10x3x2 region at (4,2,1) in the device buffer to (1,1,0) in the
  // host buffer.
  ol_dimensions_t SrcOff = {4, 2, 1};
  ol_dimensions_t DstOff = {1, 1, 0};
  ol_dimensions_t Extent = {10, 3, 2};
  ASSERT_SUCCESS(copy3D(Queue, Output.data(), Host, DstOff, Pitch, SlicePitch,
                        Alloc, Device, SrcOff, Pitch, SlicePitch, Extent));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (size_t Z = 0; Z < Slices; ++Z)
    for (size_t Y = 0; Y < Rows; ++Y)
      for (size_t X = 0; X < Pitch; ++X) {
        bool In = X >= DstOff.x && X < DstOff.x + Extent.x && Y >= DstOff.y &&
                  Y < DstOff.y + Extent.y && Z >= DstOff.z &&
                  Z < DstOff.z + Extent.z;
        uint8_t Expected = In ? Input[(Z - DstOff.z + SrcOff.z) * SlicePitch +
                                      (Y - DstOff.y + SrcOff.y) * Pitch +
                                      (X - DstOff.x) + SrcOff.x]
                              : 0;
        ASSERT_EQ(Output[Z * SlicePitch + Y * Pitch + X], Expected);
      }
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpy3DTest, Success2DHostToDevice) {
  std::vector<uint8_t> Input(SlicePitch);
  std::iota(Input.begin(), Input.end(), 1);
  std::vector<uint8_t> Output(SlicePitch, 0);

  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, SlicePitch, &Alloc));
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, 1, Output.data(), SlicePitch));

  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, 1};
  ASSERT_SUCCESS(copy3D(Queue, Alloc, Device, Zero, Pitch, SlicePitch,
                        Input.data(), Host, Zero, Pitch, SlicePitch, Extent));
  ASSERT_SUCCESS(
      olMemcpy(Queue, Output.data(), Host, Alloc, Device, SlicePitch));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_EQ(Input, Output);
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpy3DTest, InvalidPitch) {
  std::vector<uint8_t> Buf(Total);
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, 1, 1};
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               copy3D(nullptr, Buf.data(), Host, Zero, Pitch - 1, SlicePitch,
                      Buf.data(), Host, Zero, Pitch, SlicePitch, Extent));
}

TEST_P(olMemcpy3DTest, SuccessSyncNoQueue) {
  std::vector<uint8_t> Input(SlicePitch);
  std::iota(Input.begin(), Input.end(), 1);
  std::vector<uint8_t> Output(SlicePitch, 0);

  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, SlicePitch, &Alloc));
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, 1};
  ASSERT_SUCCESS(copy3D(nullptr, Alloc, Device, Zero, Pitch, SlicePitch,
                        Input.data(), Host, Zero, Pitch, SlicePitch, Extent));
  ASSERT_SUCCESS(copy3D(nullptr, Output.data(), Host, Zero, Pitch, SlicePitch,
                        Alloc, Device, Zero, Pitch, SlicePitch, Extent));
  ASSERT_EQ(Input, Output);
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpy3DTest, SuccessDtoD) {
  std::vector<uint8_t> Input(Total);
  std::iota(Input.begin(), Input.end(), 0);
  std::vector<uint8_t> Output(Total, 0);

  void *A, *B;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Total, &A));
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Total, &B));
  ASSERT_SUCCESS(olMemcpy(Queue, A, Device, Input.data(), Host, Total));
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, Slices};
  ASSERT_SUCCESS(copy3D(Queue, B, Device, Zero, Pitch, SlicePitch, A, Device,
                        Zero, Pitch, SlicePitch, Extent));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, B, Device, Total));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_EQ(Input, Output);
  ASSERT_SUCCESS(olMemFree(Context, A));
  ASSERT_SUCCESS(olMemFree(Context, B));
}

TEST_P(olMemcpy3DTest, SuccessHtoHNullDevices) {
  std::vector<uint8_t> Input(Total);
  std::iota(Input.begin(), Input.end(), 0);
  std::vector<uint8_t> Output(Total, 0);
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, Slices};
  ASSERT_SUCCESS(copy3D(nullptr, Output.data(), nullptr, Zero, Pitch,
                        SlicePitch, Input.data(), nullptr, Zero, Pitch,
                        SlicePitch, Extent));
  ASSERT_EQ(Input, Output);
}

TEST_P(olMemcpy3DTest, SuccessZeroExtent) {
  std::vector<uint8_t> Buf(Total, 7);
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {0, Rows, 1};
  ASSERT_SUCCESS(copy3D(nullptr, Buf.data(), Host, Zero, Pitch, SlicePitch,
                        Buf.data(), Host, Zero, Pitch, SlicePitch, Extent));
}

TEST_P(olMemcpy3DTest, InvalidSlicePitch) {
  std::vector<uint8_t> Buf(Total);
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, 2};
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               copy3D(nullptr, Buf.data(), Host, Zero, Pitch, SlicePitch - 1,
                      Buf.data(), Host, Zero, Pitch, SlicePitch, Extent));
  ol_dimensions_t Huge = {Pitch, 0xFFFFFFFFu, 2};
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               copy3D(nullptr, Buf.data(), Host, Zero, SIZE_MAX / 2, SlicePitch,
                      Buf.data(), Host, Zero, Pitch, SlicePitch, Huge));
}

TEST_P(olMemcpy3DTest, SuccessSingleSliceZOffset) {
  std::vector<uint8_t> Input(Total);
  std::iota(Input.begin(), Input.end(), 0);
  std::vector<uint8_t> Output(Total, 0);

  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Total, &Alloc));
  ASSERT_SUCCESS(olMemcpy(Queue, Alloc, Device, Input.data(), Host, Total));
  ol_dimensions_t SrcOff = {4, 2, 1};
  ol_dimensions_t DstOff = {0, 0, 2};
  ol_dimensions_t Extent = {10, 3, 1};
  ASSERT_SUCCESS(copy3D(Queue, Output.data(), Host, DstOff, Pitch, SlicePitch,
                        Alloc, Device, SrcOff, Pitch, SlicePitch, Extent));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  for (size_t Y = 0; Y < 3; ++Y)
    for (size_t X = 0; X < 10; ++X)
      ASSERT_EQ(Output[2 * SlicePitch + Y * Pitch + X],
                Input[SlicePitch + (Y + 2) * Pitch + X + 4]);
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpy3DTest, SuccessHtoHQueue) {
  std::vector<uint8_t> Input(Total);
  std::iota(Input.begin(), Input.end(), 0);
  std::vector<uint8_t> Output(Total, 0);
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, Slices};
  ASSERT_SUCCESS(copy3D(Queue, Output.data(), Host, Zero, Pitch, SlicePitch,
                        Input.data(), Host, Zero, Pitch, SlicePitch, Extent));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_EQ(Input, Output);
}

TEST_P(olMemcpy3DTest, InvalidSlicePitchNotMultiple) {
  std::vector<uint8_t> Buf(Total);
  ol_dimensions_t Zero = {0, 0, 0};
  ol_dimensions_t Extent = {Pitch, Rows, 2};
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               copy3D(nullptr, Buf.data(), Host, Zero, Pitch, SlicePitch + 1,
                      Buf.data(), Host, Zero, Pitch, SlicePitch, Extent));
}
