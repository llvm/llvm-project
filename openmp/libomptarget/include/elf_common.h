//===-- elf_common.h - Elf functions needed from elf_common.cpp ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Get the e_rflags from ethe elf imange
uint32_t elf_e_flags(__tgt_device_image *image);

// Verifies that the target_id for the system matches the image
int32_t elf_check_machine(__tgt_device_image *image,
                            uint16_t target_id);


