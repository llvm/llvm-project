//===- offload_lib.cpp - Entry points for the new LLVM/Offload API --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file pulls in the tablegen'd API entry point functions.
//
//===----------------------------------------------------------------------===//

#include "OffloadImpl.hpp"
#include <OffloadAPI.h>
#include <OffloadPrint.hpp>

#include <iostream>

llvm::StringSet<> &errorStrs() {
  static llvm::StringSet<> ErrorStrs;
  return ErrorStrs;
}

ErrSetT &errors() {
  static ErrSetT Errors{};
  return Errors;
}

ol_code_location_t *&currentCodeLocation() {
  thread_local ol_code_location_t *CodeLoc = nullptr;
  return CodeLoc;
}

OffloadConfig &offloadConfig() {
  static OffloadConfig Config{};
  return Config;
}

// Pull in the declarations for the implementation funtions. The actual entry
// points in this file wrap these.
#include "OffloadImplFuncDecls.inc"

// Pull in the tablegen'd entry point definitions.
#include "OffloadEntryPoints.inc"
