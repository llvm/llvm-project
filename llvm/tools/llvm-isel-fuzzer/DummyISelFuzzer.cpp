//===--- DummyFuzzerMain.cpp - Entry point to sanity check the fuzzer -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of main so we can build and test without linking libFuzzer.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv);

int main(int argc, char *argv[]) {
  errs() << "*** This tool was not linked to libFuzzer.\n"
         << "*** No fuzzing will be performed.\n";
  if (int RC = LLVMFuzzerInitialize(&argc, &argv)) {
    errs() << "Initialization failed\n";
    return RC;
  }

  for (int I = 1; I < argc; ++I) {
    StringRef Arg(argv[I]);
    if (Arg.startswith("-")) {
      if (Arg.equals("-ignore_remaining_args=1"))
        break;
      continue;
    }

    auto BufOrErr = MemoryBuffer::getFile(Arg, /*FileSize-*/ -1,
                                          /*RequiresNullTerminator=*/false);
    if (std::error_code EC = BufOrErr.getError()) {
      errs() << "Error reading file: " << Arg << ": " << EC.message() << "\n";
      return 1;
    }
    std::unique_ptr<MemoryBuffer> Buf = std::move(BufOrErr.get());
    errs() << "Running: " << Arg << " (" << Buf->getBufferSize() << " bytes)\n";
    LLVMFuzzerTestOneInput(
        reinterpret_cast<const uint8_t *>(Buf->getBufferStart()),
        Buf->getBufferSize());
  }
}
