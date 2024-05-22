//===- RawOstreamExtras.cpp - Extensions to LLVM's raw_ostream ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/RawOstreamExtras.h"
#include "llvm/Support/raw_ostream.h"

llvm::raw_ostream &mlir::thread_safe_nulls() {
  static thread_local llvm::raw_null_ostream stream;
  return stream;
}
