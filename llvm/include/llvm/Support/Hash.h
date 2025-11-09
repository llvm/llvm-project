//===- llvm/Support/Hash.h - Hash functions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides hash functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HASH_H
#define LLVM_SUPPORT_HASH_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {

/// Compute KCFI type ID from mangled type name using FNV-1a hash.
uint32_t getKCFITypeID(StringRef MangledTypeName);

} // end namespace llvm

#endif // LLVM_SUPPORT_HASH_H
