//===- TypeSize.cpp - Wrapper around type sizes------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TypeSize.h"
#include "llvm/Support/Error.h"

using namespace llvm;

TypeSize::operator TypeSize::ScalarTy() const {
  if (isScalable()) {
    reportFatalInternalError(
        "Cannot implicitly convert a scalable size to a fixed-width size in "
        "`TypeSize::operator ScalarTy()`");
    return getKnownMinValue();
  }
  return getFixedValue();
}
