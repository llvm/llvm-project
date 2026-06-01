//===-- EJitRegistryTable.c - Default Registry Tables ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Default (empty) weak definitions for __ejit_registry_bitcode[] and
// __ejit_registry_period[].  Overridden by AOT pass output (PASS1/PASS2)
// which defines strong versions when --embed-bitcode is active.
//
// Must be a .c file — C++ weak declarations on global arrays require
// explicit visibility annotations.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRegistryEntry.h"

const ejit_reg_entry_t __ejit_registry_bitcode[]
    __attribute__((weak)) = {{EJIT_REG_NONE, 0, 0, 0, 0}};

const ejit_reg_entry_t __ejit_registry_period[]
    __attribute__((weak)) = {{EJIT_REG_NONE, 0, 0, 0, 0}};
