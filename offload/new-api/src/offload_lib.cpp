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

#include "offload_impl.hpp"
#include <offload_api.h>
#include <offload_print.hpp>

#include <iostream>

llvm::StringSet<> &ErrorStrs() {
  static llvm::StringSet<> ErrorStrs;
  return ErrorStrs;
}

ErrSetT &Errors() {
  static ErrSetT Errors{};
  return Errors;
}

// Pull in the declarations for the implementation funtions. The actual entry
// points in this file wrap these.
#include "offload_impl_func_decls.inc"

// Pull in the tablegen'd entry point definitions.
#include "offload_entry_points.inc"
