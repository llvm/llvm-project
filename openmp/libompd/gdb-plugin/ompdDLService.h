/*
 * ompdDLService.h -- Load libompd and look up OMPD API symbols.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPD_GDB_PLUGIN_OMPD_DL_SERVICE_H
#define OPENMP_LIBOMPD_GDB_PLUGIN_OMPD_DL_SERVICE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Handle of the loaded OMPD library, or NULL if none is loaded. */
extern void *ompd_library;

/* Load the OMPD library at name.
 * Returns 0 on success. On failure, returns -1 and ompd_get_dl_error()
 * describes the problem. */
int ompd_load_library(const char *name);

/* Look up name in the loaded OMPD library.
 * Returns the symbol address, or NULL on failure. */
void *ompd_get_symbol(const char *name);

/* Last load/lookup error, or NULL if the last helper call succeeded.
 * The string stays valid until the next helper call. */
const char *ompd_get_dl_error(void);

#ifdef __cplusplus
}
#endif

#endif
