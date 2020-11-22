//===-- elf_amd.h - Elf functions needed by amd plugin rtl.cpp ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Convenience functions for rtl.cpp to query various AMD Elf attributes.

extern int get_elf_mach_gfx(__tgt_device_image *image);

extern const char* get_elf_mach_gfx_name(__tgt_device_image *image);

extern bool elf_machine_id_is_amdgcn(__tgt_device_image *image);
