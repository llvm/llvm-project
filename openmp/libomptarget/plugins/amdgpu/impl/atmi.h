//===--- amdgpu/impl/atmi.h --------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_ATMI_H_
#define INCLUDE_ATMI_H_

#define ROCM_VERSION_MAJOR 3
#define ROCM_VERSION_MINOR 2

/** \defgroup enumerations Enumerated Types
 * @{
 */

/**
 * @brief Device Types.
 */
typedef enum atmi_devtype_s {
  ATMI_DEVTYPE_CPU = 0x0001,
  ATMI_DEVTYPE_iGPU = 0x0010,                               // Integrated GPU
  ATMI_DEVTYPE_dGPU = 0x0100,                               // Discrete GPU
  ATMI_DEVTYPE_GPU = ATMI_DEVTYPE_iGPU | ATMI_DEVTYPE_dGPU, // Any GPU
  ATMI_DEVTYPE_ALL = 0x111 // Union of all device types
} atmi_devtype_t;

/**
 * @brief Memory Access Type.
 */
typedef enum atmi_memtype_s {
  ATMI_MEMTYPE_FINE_GRAINED = 0,
  ATMI_MEMTYPE_COARSE_GRAINED = 1,
  ATMI_MEMTYPE_ANY
} atmi_memtype_t;

/** @} */
#endif // INCLUDE_ATMI_H_
