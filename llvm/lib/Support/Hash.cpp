//===- Hash.cpp - Hash functions ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hash functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Hash.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

uint32_t llvm::getKCFITypeID(StringRef MangledTypeName) {
  return static_cast<uint32_t>(xxHash64(MangledTypeName));
}
