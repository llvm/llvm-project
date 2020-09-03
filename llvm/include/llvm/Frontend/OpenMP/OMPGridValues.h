//====--- OMPGridValues.h - Language-specific address spaces --*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides definitions for Target specific Grid Values
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_GRIDVALUES_H
#define LLVM_OPENMP_GRIDVALUES_H

namespace llvm {

namespace omp {

/// \brief Defines various target-specific GPU grid values that must be
///        consistent between host RTL (plugin), device RTL, and clang.
///        We can change grid values for a "fat" binary so that different
///        passes get the correct values when generating code for a
///        multi-target binary. Both amdgcn and nvptx values are stored in
///        this file. In the future, should there be differences between GPUs
///        of the same architecture, then simply make a different array and
///        use the new array name.
///
/// Example usage in clang:
///   const unsigned slot_size = ctx.GetTargetInfo().getGridValue(GV_Warp_Size);
///
/// Example usage in libomptarget/deviceRTLs:
///   #include "OMPGridValues.h"
///   #ifdef __AMDGPU__
///     #define GRIDVAL AMDGPUGpuGridValues
///   #else
///     #define GRIDVAL NVPTXGpuGridValues
///   #endif
///   ... Then use this reference for GV_Warp_Size in the deviceRTL source.
///   GRIDVAL[GV_Warp_Size]
///
/// Example usage in libomptarget hsa plugin:
///   #include "OMPGridValues.h"
///   #define GRIDVAL AMDGPUGpuGridValues
///   ... Then use this reference to access GV_Warp_Size in the hsa plugin.
///   GRIDVAL[GV_Warp_Size]
///
/// Example usage in libomptarget cuda plugin:
///    #include "OMPGridValues.h"
///    #define GRIDVAL NVPTXGpuGridValues
///   ... Then use this reference to access GV_Warp_Size in the cuda plugin.
///    GRIDVAL[GV_Warp_Size]
///
enum GVIDX {
  /// The maximum number of workers in a kernel.
  /// (THREAD_ABSOLUTE_LIMIT) - (GV_Warp_Size), might be issue for blockDim.z
  GV_Threads,
  /// The size reserved for data in a shared memory slot.
  GV_Slot_Size,
  /// The default value of maximum number of threads in a worker warp.
  GV_Warp_Size,
  /// Alternate warp size for some AMDGCN architectures. Same as GV_Warp_Size
  /// for NVPTX.
  GV_Warp_Size_32,
  /// The number of bits required to represent the max number of threads in warp
  GV_Warp_Size_Log2,
  /// GV_Warp_Size * GV_Slot_Size,
  GV_Warp_Slot_Size,
  /// the maximum number of teams.
  GV_Max_Teams,
  /// Global Memory Alignment
  GV_Mem_Align,
  /// (~0u >> (GV_Warp_Size - GV_Warp_Size_Log2))
  GV_Warp_Size_Log2_Mask,
  // An alternative to the heavy data sharing infrastructure that uses global
  // memory is one that uses device __shared__ memory.  The amount of such space
  // (in bytes) reserved by the OpenMP runtime is noted here.
  GV_SimpleBufferSize,
  // The absolute maximum team size for a working group
  GV_Max_WG_Size,
  // The default maximum team size for a working group
  GV_Default_WG_Size,
  // This is GV_Max_WG_Size / GV_WarpSize. 32 for NVPTX and 16 for AMDGCN.
  GV_Max_Warp_Number,
  /// The slot size that should be reserved for a working warp.
  /// (~0u >> (GV_Warp_Size - GV_Warp_Size_Log2))
  GV_Warp_Size_Log2_MaskL
};

/// For AMDGPU GPUs
static constexpr unsigned AMDGPUGpuGridValues[] = {
    448,       // GV_Threads
    256,       // GV_Slot_Size
    64,        // GV_Warp_Size
    32,        // GV_Warp_Size_32
    6,         // GV_Warp_Size_Log2
    64 * 256,  // GV_Warp_Slot_Size
    128,       // GV_Max_Teams
    256,       // GV_Mem_Align
    63,        // GV_Warp_Size_Log2_Mask
    896,       // GV_SimpleBufferSize
    1024,      // GV_Max_WG_Size,
    256,       // GV_Defaut_WG_Size
    1024 / 64, // GV_Max_WG_Size / GV_WarpSize
    63         // GV_Warp_Size_Log2_MaskL
};

/// For Nvidia GPUs
static constexpr unsigned NVPTXGpuGridValues[] = {
    992,               // GV_Threads
    256,               // GV_Slot_Size
    32,                // GV_Warp_Size
    32,                // GV_Warp_Size_32
    5,                 // GV_Warp_Size_Log2
    32 * 256,          // GV_Warp_Slot_Size
    1024,              // GV_Max_Teams
    256,               // GV_Mem_Align
    (~0u >> (32 - 5)), // GV_Warp_Size_Log2_Mask
    896,               // GV_SimpleBufferSize
    1024,              // GV_Max_WG_Size
    128,               // GV_Defaut_WG_Size
    1024 / 32,         // GV_Max_WG_Size / GV_WarpSize
    31                 // GV_Warp_Size_Log2_MaskL
};

} // namespace omp
} // namespace llvm

#endif // LLVM_OPENMP_GRIDVALUES_H
