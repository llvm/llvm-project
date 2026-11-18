//===-- llvm-smybol-reader-fuzzer.cpp - Fuzzer for symbol prof parsing ---====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SymbolRemappingReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);

  std::string Payload(reinterpret_cast<const char *>(Data), Size);
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Payload);

  SymbolRemappingReader Reader;
  Error E = Reader.read(*Buf);
  if ((bool)E) {
    logAllUnhandledErrors(std::move(E), OS);
  }

  return 0;
}
