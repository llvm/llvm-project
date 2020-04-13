//===-- StringPool.cpp - Intern'd string pool -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringPool class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StringPool.h"
using namespace llvm;

StringPool::StringPool() {}

StringPool::~StringPool() {
  assert(internTable.empty() && "PooledStringPtr leaked!");
}

PooledStringPtr StringPool::intern(StringRef key) {
  auto it = internTable.find(key);
  if (it != internTable.end())
    return PooledStringPtr(&*it);

  MallocAllocator allocator;
  auto *entry = Entry::Create(key, allocator);
  entry->getValue().pool = this;
  internTable.insert(entry);

  return PooledStringPtr(entry);
}
