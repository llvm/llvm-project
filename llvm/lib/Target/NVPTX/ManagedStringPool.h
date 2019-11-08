//===-- ManagedStringPool.h - Managed String Pool ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The strings allocated from a managed string pool are owned by the string
// pool and will be deleted together with the managed string pool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MANAGEDSTRINGPOOL_H
#define LLVM_LIB_TARGET_NVPTX_MANAGEDSTRINGPOOL_H

#include "llvm/ADT/SmallVector.h"
#include <string>

namespace llvm {

/// ManagedStringPool - The strings allocated from a managed string pool are
/// owned by the string pool and will be deleted together with the managed
/// string pool.
class ManagedStringPool {
  SmallVector<std::string *, 8> Pool;

public:
  ManagedStringPool() = default;

  ~ManagedStringPool() {
    SmallVectorImpl<std::string *>::iterator Current = Pool.begin();
    while (Current != Pool.end()) {
      delete *Current;
      ++Current;
    }
  }

  std::string *getManagedString(const char *S) {
    std::string *Str = new std::string(S);
    Pool.push_back(Str);
    return Str;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_MANAGEDSTRINGPOOL_H
