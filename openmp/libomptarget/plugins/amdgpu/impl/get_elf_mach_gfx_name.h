//===--- amdgpu/impl/get_elf_mach_gfx_name.h ---------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GET_ELF_MACH_GFX_NAME_H_INCLUDED
#define GET_ELF_MACH_GFX_NAME_H_INCLUDED

#include <stdint.h>

const char *get_elf_mach_gfx_name(uint32_t EFlags);

enum IMPLICITARGS : uint16_t {
  COV4_SIZE = 56,
  COV4_HOSTCALL_PTR_OFFSET = 24,
  HOSTCALL_PTR_SIZE = 8,

  COV5_SIZE = 256,

  COV5_BLOCK_COUNT_X_OFFSET = 0,
  COV5_BLOCK_COUNT_X_SIZE = 4,

  COV5_BLOCK_COUNT_Y_OFFSET = 4,
  COV5_BLOCK_COUNT_Y_SIZE = 4,

  COV5_BLOCK_COUNT_Z_OFFSET = 8,
  COV5_BLOCK_COUNT_Z_SIZE = 4,

  COV5_GROUP_SIZE_X_OFFSET = 12,
  COV5_GROUP_SIZE_X_SIZE = 2,

  COV5_GROUP_SIZE_Y_OFFSET = 14,
  COV5_GROUP_SIZE_Y_SIZE = 2,

  COV5_GROUP_SIZE_Z_OFFSET = 16,
  COV5_GROUP_SIZE_Z_SIZE = 2,

  COV5_REMAINDER_X_OFFSET = 18,
  COV5_REMAINDER_X_SIZE = 2,

  COV5_REMAINDER_Y_OFFSET = 20,
  COV5_REMAINDER_Y_SIZE = 2,

  COV5_REMAINDER_Z_OFFSET = 22,
  COV5_REMAINDER_Z_SIZE = 2,

  COV5_GRID_DIMS_OFFSET = 64,
  COV5_GRID_DIMS_SIZE = 2,

  COV5_HOSTCALL_PTR_OFFSET = 80,

  COV5_HEAPV1_PTR_OFFSET = 96,
  COV5_HEAPV1_PTR_SIZE = 8
};

const uint16_t implicitArgsSize(uint16_t Version);

#endif
