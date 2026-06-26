//===-- EJitRwLock.h - Lightweight rwlock for EJIT taskpool --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITRWLOCK_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITRWLOCK_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include <cstdint>

namespace llvm {
namespace ejit {

class EJitRwLock {
public:
  bool tryRead() {
    if (writeFlag_.loadAcquire() != 0)
      return false;
    readers_.fetchAdd(1);
    if (writeFlag_.loadAcquire() != 0) {
      readers_.fetchSub(1);
      return false;
    }
    return true;
  }

  void readRelease() {
    uint32_t cur = readers_.loadAcquire();
    while (cur != 0) {
      uint32_t next = cur - 1;
      if (readers_.compareExchange(cur, next))
        return;
    }
  }

  bool tryWrite() {
    uint32_t expected = 0;
    if (!writeFlag_.compareExchange(expected, 1))
      return false;
    if (readers_.loadAcquire() != 0) {
      writeFlag_.storeRelease(0);
      return false;
    }
    return true;
  }

  void write() {
    uint32_t expected = 0;
    while (!writeFlag_.compareExchange(expected, 1))
      expected = 0;
    while (readers_.loadAcquire() != 0) {
    }
  }

  void writeRelease() { writeFlag_.storeRelease(0); }

private:
  EJitAtomicU32 writeFlag_{0};
  EJitAtomicU32 readers_{0};
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITRWLOCK_H
