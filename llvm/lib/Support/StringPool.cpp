//===-- StringPool.cpp - Interned string pool -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringPool class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StringPool.h"
#include "llvm/Support/Streams.h"

using namespace llvm;

StringPool::StringPool() {}

StringPool::~StringPool() {
  assert(InternTable.empty() && "PooledStringPtr leaked!");
}

PooledStringPtr StringPool::intern(const char *Begin, const char *End) {
  table_t::iterator I = InternTable.find(Begin, End);
  if (I != InternTable.end())
    return PooledStringPtr(&*I);
  
  entry_t *S = entry_t::Create(Begin, End);
  S->getValue().Pool = this;
  InternTable.insert(S);
  
  return PooledStringPtr(S);
}
