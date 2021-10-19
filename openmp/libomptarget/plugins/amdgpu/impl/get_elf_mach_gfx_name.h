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

#endif
