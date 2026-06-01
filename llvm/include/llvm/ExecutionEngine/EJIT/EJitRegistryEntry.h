//===-- EJitRegistryEntry.h - Static Registry Table Entry -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Global constant array entry type for bare-metal auto-registration.
// PASS1/PASS2 generate an array of these entries (__ejit_registry[]) that
// ejit_init() walks when constructor-based registration is unavailable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITREGISTRYENTRY_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITREGISTRYENTRY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  EJIT_REG_BITCODE      = 0,
  EJIT_REG_PERIOD_ARRAY = 1,
  EJIT_REG_STATIC_VAR   = 2,
  EJIT_REG_SYMBOL       = 3,
  EJIT_REG_NONE         = 4  // sentinel
} ejit_reg_type_t;

typedef struct {
  ejit_reg_type_t type;
  const char *name1;       // funcName / periodName / varName / symbolName
  const char *name2;       // varName (period/static), NULL otherwise
  const void *ptr;         // bitcode data / baseAddr / symbol addr
  uint64_t     size;       // bitcode size / array size / 0
} ejit_reg_entry_t;

#ifdef __cplusplus
}
#endif

#endif
