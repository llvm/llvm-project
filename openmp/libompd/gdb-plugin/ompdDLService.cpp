/*
 * ompdDLService.cpp -- Load libompd and look up OMPD API symbols.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ompdDLService.h"

#include "llvm/Support/DynamicLibrary.h"

#include <cstdio>
#include <cstring>
#include <string>

void *ompd_library = NULL;

static llvm::sys::DynamicLibrary LoadedLib;
static char last_error[256];

static void set_error(const char *msg) {
  if (!msg || !msg[0]) {
    last_error[0] = '\0';
    return;
  }
  strncpy(last_error, msg, sizeof(last_error) - 1);
  last_error[sizeof(last_error) - 1] = '\0';
}

static void clear_error(void) { last_error[0] = '\0'; }

int ompd_load_library(const char *name) {
  clear_error();
  if (!name || !name[0]) {
    set_error("OMPD library path is empty");
    return -1;
  }

  std::string ErrMsg;
  llvm::sys::DynamicLibrary NewLib =
      llvm::sys::DynamicLibrary::getLibrary(name, &ErrMsg);
  if (!NewLib.isValid()) {
    set_error(ErrMsg.empty() ? "failed to load OMPD library" : ErrMsg.c_str());
    return -1;
  }

  if (LoadedLib.isValid())
    llvm::sys::DynamicLibrary::closeLibrary(LoadedLib);
  LoadedLib = NewLib;
  ompd_library = LoadedLib.getOSSpecificHandle();
  return 0;
}

void *ompd_get_symbol(const char *name) {
  void *sym;

  clear_error();
  if (!LoadedLib.isValid()) {
    set_error("OMPD library is not loaded");
    return NULL;
  }
  if (!name || !name[0]) {
    set_error("OMPD symbol name is empty");
    return NULL;
  }

  sym = LoadedLib.getAddressOfSymbol(name);
  if (!sym) {
    snprintf(last_error, sizeof(last_error), "could not find symbol '%s'",
             name);
    return NULL;
  }
  return sym;
}

const char *ompd_get_dl_error(void) {
  if (!last_error[0])
    return NULL;
  return last_error;
}
