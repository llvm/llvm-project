//===--- special-case-list-fuzzer.cpp - Fuzzer for special case lists -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SpecialCaseList.h"

#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::unique_ptr<llvm::MemoryBuffer> Buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(Data), Size), "", false);

  if (!Buf)
    return 0;

  std::string Error;
  llvm::SpecialCaseList::create(Buf.get(), Error);

  return 0;
}
