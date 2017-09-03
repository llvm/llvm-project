//===-- FuzzerCLI.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/FuzzerCLI.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void llvm::parseFuzzerCLOpts(int ArgC, char *ArgV[]) {
  std::vector<const char *> CLArgs;
  CLArgs.push_back(ArgV[0]);

  int I = 1;
  while (I < ArgC)
    if (StringRef(ArgV[I++]).equals("-ignore_remaining_args=1"))
      break;
  while (I < ArgC)
    CLArgs.push_back(ArgV[I++]);

  cl::ParseCommandLineOptions(CLArgs.size(), CLArgs.data());
}

int llvm::runFuzzerOnInputs(int ArgC, char *ArgV[], FuzzerTestFun TestOne,
                            FuzzerInitFun Init) {
  errs() << "*** This tool was not linked to libFuzzer.\n"
         << "*** No fuzzing will be performed.\n";
  if (int RC = Init(&ArgC, &ArgV)) {
    errs() << "Initialization failed\n";
    return RC;
  }

  for (int I = 1; I < ArgC; ++I) {
    StringRef Arg(ArgV[I]);
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
    TestOne(reinterpret_cast<const uint8_t *>(Buf->getBufferStart()),
            Buf->getBufferSize());
  }
  return 0;
}
