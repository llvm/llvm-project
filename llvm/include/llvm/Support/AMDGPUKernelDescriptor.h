//===--- AMDGPUKernelDescriptor.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU kernel descriptor definitions. For more information, visit
/// https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor-for-gfx6-gfx9
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUKERNELDESCRIPTOR_H
#define LLVM_SUPPORT_AMDGPUKERNELDESCRIPTOR_H

#include <cstdint>

// Creates enumeration entries used for packing bits into integers. Enumeration
// entries include bit shift amount, bit width, and bit mask.
#define AMDGPU_BITS_ENUM_ENTRY(name, shift, width) \
  name ## _SHIFT = (shift),                        \
  name ## _WIDTH = (width),                        \
  name = (((1 << (width)) - 1) << (shift))         \

// Gets bits for specified bit mask from specified source.
#define AMDGPU_BITS_GET(src, mask) \
  ((src & mask) >> mask ## _SHIFT) \

// Sets bits for specified bit mask in specified destination.
#define AMDGPU_BITS_SET(dst, mask, val)     \
  dst &= (~(1 << mask ## _SHIFT) & ~mask);  \
  dst |= (((val) << mask ## _SHIFT) & mask) \

namespace llvm {
namespace AMDGPU {
namespace HSAKD {

/// Floating point rounding modes.
enum : uint8_t {
  AMDGPU_FLOAT_ROUND_MODE_NEAR_EVEN      = 0,
  AMDGPU_FLOAT_ROUND_MODE_PLUS_INFINITY  = 1,
  AMDGPU_FLOAT_ROUND_MODE_MINUS_INFINITY = 2,
  AMDGPU_FLOAT_ROUND_MODE_ZERO           = 3,
};

/// Floating point denorm modes.
enum : uint8_t {
  AMDGPU_FLOAT_DENORM_MODE_FLUSH_SRC_DST = 0,
  AMDGPU_FLOAT_DENORM_MODE_FLUSH_DST     = 1,
  AMDGPU_FLOAT_DENORM_MODE_FLUSH_SRC     = 2,
  AMDGPU_FLOAT_DENORM_MODE_FLUSH_NONE    = 3,
};

/// System VGPR workitem IDs.
enum : uint8_t {
  AMDGPU_SYSTEM_VGPR_WORKITEM_ID_X         = 0,
  AMDGPU_SYSTEM_VGPR_WORKITEM_ID_X_Y       = 1,
  AMDGPU_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z     = 2,
  AMDGPU_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED = 3,
};

/// Compute program resource register one layout.
enum ComputePgmRsrc1 {
  AMDGPU_BITS_ENUM_ENTRY(GRANULATED_WORKITEM_VGPR_COUNT, 0, 6),
  AMDGPU_BITS_ENUM_ENTRY(GRANULATED_WAVEFRONT_SGPR_COUNT, 6, 4),
  AMDGPU_BITS_ENUM_ENTRY(PRIORITY, 10, 2),
  AMDGPU_BITS_ENUM_ENTRY(FLOAT_ROUND_MODE_32, 12, 2),
  AMDGPU_BITS_ENUM_ENTRY(FLOAT_ROUND_MODE_16_64, 14, 2),
  AMDGPU_BITS_ENUM_ENTRY(FLOAT_DENORM_MODE_32, 16, 2),
  AMDGPU_BITS_ENUM_ENTRY(FLOAT_DENORM_MODE_16_64, 18, 2),
  AMDGPU_BITS_ENUM_ENTRY(PRIV, 20, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_DX10_CLAMP, 21, 1),
  AMDGPU_BITS_ENUM_ENTRY(DEBUG_MODE, 22, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_IEEE_MODE, 23, 1),
  AMDGPU_BITS_ENUM_ENTRY(BULKY, 24, 1),
  AMDGPU_BITS_ENUM_ENTRY(CDBG_USER, 25, 1),
  AMDGPU_BITS_ENUM_ENTRY(FP16_OVFL, 26, 1),
  AMDGPU_BITS_ENUM_ENTRY(RESERVED0, 27, 5),
};

/// Compute program resource register two layout.
enum ComputePgmRsrc2 {
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_OFFSET, 0, 1),
  AMDGPU_BITS_ENUM_ENTRY(USER_SGPR_COUNT, 1, 5),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_TRAP_HANDLER, 6, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_SGPR_WORKGROUP_ID_X, 7, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_SGPR_WORKGROUP_ID_Y, 8, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_SGPR_WORKGROUP_ID_Z, 9, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_SGPR_WORKGROUP_INFO, 10, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_VGPR_WORKITEM_ID, 11, 2),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_ADDRESS_WATCH, 13, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_MEMORY, 14, 1),
  AMDGPU_BITS_ENUM_ENTRY(GRANULATED_LDS_SIZE, 15, 9),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION, 24, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_FP_DENORMAL_SOURCE, 25, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO, 26, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW, 27, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW, 28, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_IEEE_754_FP_INEXACT, 29, 1),
  AMDGPU_BITS_ENUM_ENTRY(ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO, 30, 1),
  AMDGPU_BITS_ENUM_ENTRY(RESERVED1, 31, 1),
};

/// Kernel descriptor layout. This layout should be kept backwards
/// compatible as it is consumed by the command processor.
struct KernelDescriptor final {
  uint32_t GroupSegmentFixedSize;
  uint32_t PrivateSegmentFixedSize;
  uint32_t MaxFlatWorkGroupSize;
  uint64_t IsDynamicCallStack : 1;
  uint64_t IsXNACKEnabled : 1;
  uint64_t Reserved0 : 30;
  int64_t KernelCodeEntryByteOffset;
  uint64_t Reserved1[3];
  uint32_t ComputePgmRsrc1;
  uint32_t ComputePgmRsrc2;
  uint64_t EnableSGPRPrivateSegmentBuffer : 1;
  uint64_t EnableSGPRDispatchPtr : 1;
  uint64_t EnableSGPRQueuePtr : 1;
  uint64_t EnableSGPRKernargSegmentPtr : 1;
  uint64_t EnableSGPRDispatchID : 1;
  uint64_t EnableSGPRFlatScratchInit : 1;
  uint64_t EnableSGPRPrivateSegmentSize : 1;
  uint64_t EnableSGPRGridWorkgroupCountX : 1;
  uint64_t EnableSGPRGridWorkgroupCountY : 1;
  uint64_t EnableSGPRGridWorkgroupCountZ : 1;
  uint64_t Reserved2 : 54;

  KernelDescriptor() = default;
};

} // end namespace HSAKD
} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_SUPPORT_AMDGPUKERNELDESCRIPTOR_H
