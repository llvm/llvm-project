/*
 * DLSymService.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include <memory>

bool ErrorOccured = false;
std::shared_ptr<llvm::sys::DynamicLibrary> OMPDLibrary = nullptr;

void *getSymbolForFunction(const char *name) {
  if (!OMPDLibrary || !OMPDLibrary->isValid()) {
    ErrorOccured = true;
    return nullptr;
  }

  auto SymAddr = OMPDLibrary->getAddressOfSymbol(name);
  if (!SymAddr) {
    ErrorOccured = true;
  }
  // Leave cast to user
  return SymAddr;
}

void loadLibraryWithName(const char *name) {
  if (OMPDLibrary && OMPDLibrary->isValid()) {
    return;
  }

  std::string errMsg;
  OMPDLibrary = std::make_shared<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(name, &errMsg));
  if (!OMPDLibrary->isValid()) {
    ErrorOccured = true;
  }
  ErrorOccured = false;
}

bool errorOccured() {
  bool oldVal = ErrorOccured;
  ErrorOccured = false;
  return oldVal;
}

const char *getErrorStr() {
  return "An error occured";
}

extern "C" {
void *get_dlsym_for_name(const char *name) {
  return getSymbolForFunction(name);
}

void get_library_with_name(const char *name) {
  return loadLibraryWithName(name);
}

const char *get_error() {
  if (!errorOccured()) {
    return nullptr;
  }
  return getErrorStr();
}
}