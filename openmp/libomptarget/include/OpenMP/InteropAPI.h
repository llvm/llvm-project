//===-- OpenMP/InteropAPI.h - OpenMP interoperability types and API - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_INTEROP_API_H
#define OMPTARGET_OPENMP_INTEROP_API_H

#include "omp.h"

#include "omptarget.h"

extern "C" {

typedef enum kmp_interop_type_t {
  kmp_interop_type_unknown = -1,
  kmp_interop_type_platform,
  kmp_interop_type_device,
  kmp_interop_type_tasksync,
} kmp_interop_type_t;

/// The interop value type, aka. the interop object.
typedef struct omp_interop_val_t {
  /// Device and interop-type are determined at construction time and fix.
  omp_interop_val_t(intptr_t device_id, kmp_interop_type_t interop_type)
      : interop_type(interop_type), device_id(device_id) {}
  const char *err_str = nullptr;
  __tgt_async_info *async_info = nullptr;
  __tgt_device_info device_info;
  const kmp_interop_type_t interop_type;
  const intptr_t device_id;
  const omp_foreign_runtime_ids_t vendor_id = cuda;
  const intptr_t backend_type_id = omp_interop_backend_type_cuda_1;
} omp_interop_val_t;

} // extern "C"

#endif // OMPTARGET_OPENMP_INTEROP_API_H
