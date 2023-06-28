//===-- OpenMP/omp.h - Copies of OpenMP user facing types and APIs - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This copies some OpenMP user facing types and APIs for easy reach within the
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_OMP_H
#define OMPTARGET_OPENMP_OMP_H

#include <cstdint>

#if defined(_WIN32)
#define __KAI_KMPC_CONVENTION __cdecl
#ifndef __KMP_IMP
#define __KMP_IMP __declspec(dllimport)
#endif
#else
#define __KAI_KMPC_CONVENTION
#ifndef __KMP_IMP
#define __KMP_IMP
#endif
#endif

extern "C" {

/// Type declarations
///{

typedef void *omp_depend_t;

///}

/// API declarations
///{

int omp_get_default_device(void) __attribute__((weak));

///}

} // extern "C"

#endif // OMPTARGET_OPENMP_OMP_H
