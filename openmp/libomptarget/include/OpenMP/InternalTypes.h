//===-- OpenMP/InternalTypes.h -- Internal OpenMP Types ------------- C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private type declarations and helper macros for OpenMP.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_INTERNAL_TYPES_H
#define OMPTARGET_OPENMP_INTERNAL_TYPES_H

#include <cstddef>
#include <cstdint>

typedef intptr_t kmp_intptr_t;

extern "C" {

// Compiler sends us this info:
typedef struct kmp_depend_info {
  kmp_intptr_t base_addr;
  size_t len;
  struct {
    bool in : 1;
    bool out : 1;
    bool mtx : 1;
  } flags;
} kmp_depend_info_t;

} // extern "C"

#endif // OMPTARGET_OPENMP_INTERNAL_TYPES_H
