//===-- elf_common.h - Common ELF functionality -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common ELF functionality for target plugins.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_ELF_COMMON_ELF_COMMON_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_ELF_COMMON_ELF_COMMON_H

#include "omptargetplugin.h"
#include <cstdint>

/// Return non-zero, if the given \p image is an ELF object, which
/// e_machine matches \p target_id; return zero otherwise.
EXTERN int32_t elf_check_machine(__tgt_device_image *Image, uint16_t TargetId);

/// Return non-zero, if the given \p image is an ET_DYN ELF object;
/// return zero otherwise.
EXTERN int32_t elf_is_dynamic(__tgt_device_image *Image);

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_ELF_COMMON_ELF_COMMON_H
