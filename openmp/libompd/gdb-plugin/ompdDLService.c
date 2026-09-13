/*
 * ompdDLService.c -- Load libompd and look up OMPD API symbols.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ompdDLService.h"

#include <dlfcn.h>
#include <string.h>

void *ompd_library = NULL;

static char last_error[256];

static void set_error(const char *msg) {
  if (!msg || !msg[0]) {
    last_error[0] = '\0';
    return;
  }
  strncpy(last_error, msg, sizeof(last_error) - 1);
  last_error[sizeof(last_error) - 1] = '\0';
}

static void clear_error(void) {
  last_error[0] = '\0';
  (void)dlerror();
}

int ompd_load_library(const char *name) {
  const char *dlerr;

  clear_error();
  if (!name || !name[0]) {
    set_error("OMPD library path is empty");
    ompd_library = NULL;
    return -1;
  }

  ompd_library = dlopen(name, RTLD_LAZY);
  dlerr = dlerror();
  if (dlerr) {
    set_error(dlerr);
    ompd_library = NULL;
    return -1;
  }
  if (!ompd_library) {
    set_error("dlopen returned NULL");
    return -1;
  }
  return 0;
}

void *ompd_get_symbol(const char *name) {
  const char *dlerr;
  void *sym;

  clear_error();
  if (!ompd_library) {
    set_error("OMPD library is not loaded");
    return NULL;
  }
  if (!name || !name[0]) {
    set_error("OMPD symbol name is empty");
    return NULL;
  }

  sym = dlsym(ompd_library, name);
  dlerr = dlerror();
  if (dlerr) {
    set_error(dlerr);
    return NULL;
  }
  return sym;
}

const char *ompd_get_dl_error(void) {
  if (!last_error[0])
    return NULL;
  return last_error;
}
