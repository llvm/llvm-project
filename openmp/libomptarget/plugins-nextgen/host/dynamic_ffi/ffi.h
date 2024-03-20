//===--- generic-elf-64bit/dynamic_ffi/ffi.cpp -------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a mirror to the parts of the FFI interface that the plugins require.
//
// libffi
//   - Copyright (c) 2011, 2014, 2019, 2021, 2022 Anthony Green
//   - Copyright (c) 1996-2003, 2007, 2008 Red Hat, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMIC_FFI_FFI_H
#define DYNAMIC_FFI_FFI_H

#include <stddef.h>
#include <stdint.h>

#define USES_DYNAMIC_FFI

uint32_t ffi_init();

typedef struct _ffi_type {
  size_t size;
  unsigned short alignment;
  unsigned short type;
  struct _ffi_type **elements;
} ffi_type;

typedef enum {
  FFI_OK = 0,
  FFI_BAD_TYPEDEF,
  FFI_BAD_ABI,
  FFI_BAD_ARGTYPE
} ffi_status;

// These are target depenent so we set them manually for each ABI by referencing
// the FFI source.
typedef enum ffi_abi {
#if (defined(_M_X64) || defined(__x86_64__))
  FFI_DEFAULT_ABI = 2, // FFI_UNIX64.
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
  FFI_DEFAULT_ABI = 1, // FFI_SYSV.
#elif defined(__powerpc64__)
  FFI_DEFAULT_ABI = 8, // FFI_LINUX.
#elif defined(__s390x__)
  FFI_DEFAULT_ABI = 1, // FFI_SYSV.
#else
#error "Unknown ABI"
#endif
} ffi_cif;

#ifdef __cplusplus
extern "C" {
#endif

#define FFI_EXTERN extern
#define FFI_API

FFI_EXTERN ffi_type ffi_type_void;
FFI_EXTERN ffi_type ffi_type_pointer;

FFI_API
void ffi_call(ffi_cif *cif, void (*fn)(void), void *rvalue, void **avalue);

FFI_API
ffi_status ffi_prep_cif(ffi_cif *cif, ffi_abi abi, unsigned int nargs,
                        ffi_type *rtype, ffi_type **atypes);

#ifdef __cplusplus
}
#endif

#endif // DYNAMIC_FFI_FFI_H
