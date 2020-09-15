//===- Support.cpp - Helpers for C interface to MLIR API ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Support.h"

#include <cstring>

MlirStringRef mlirStringRefCreateFromCString(const char *str) {
  return mlirStringRefCreate(str, strlen(str));
}
