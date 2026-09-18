//===- Memory.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/Memory.h"

namespace orc_rt::sys {

Expected<void *> reserveMemory(uint64_t) {
  return make_error<StringError>("Windows reserveMemory not implemented");
}

Error releaseMemory(void *, uint64_t) {
  return make_error<StringError>("Windows releaseMemory not implemented");
}

Error protectMemory(void *, uint64_t, MemProt) {
  return make_error<StringError>("Windows protectMemory not implemented");
}

} // namespace orc_rt::sys
