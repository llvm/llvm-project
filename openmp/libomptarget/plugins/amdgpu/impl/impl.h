//===--- amdgpu/impl/impl.h --------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_IMPL_H_
#define INCLUDE_IMPL_H_

#define ROCM_VERSION_MAJOR 3
#define ROCM_VERSION_MINOR 2

/** \defgroup enumerations Enumerated Types
 * @{
 */

/**
 * @brief Device Types.
 */
typedef enum impl_devtype_s {
  IMPL_DEVTYPE_CPU = 0x0001,
  IMPL_DEVTYPE_iGPU = 0x0010,                               // Integrated GPU
  IMPL_DEVTYPE_dGPU = 0x0100,                               // Discrete GPU
  IMPL_DEVTYPE_GPU = IMPL_DEVTYPE_iGPU | IMPL_DEVTYPE_dGPU, // Any GPU
  IMPL_DEVTYPE_ALL = 0x111 // Union of all device types
} impl_devtype_t;

/**
 * @brief Memory Access Type.
 */
typedef enum impl_memtype_s {
  IMPL_MEMTYPE_FINE_GRAINED = 0,
  IMPL_MEMTYPE_COARSE_GRAINED = 1,
  IMPL_MEMTYPE_ANY
} impl_memtype_t;

/** @} */
#endif // INCLUDE_IMPL_H_
