//===-- FuzzerCLI.h - Common logic for CLIs of fuzzers ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common logic needed to implement LLVM's fuzz targets' CLIs - including LLVM
// concepts like cl::opt and libFuzzer concepts like -ignore_remaining_args=1.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZMUTATE_FUZZER_CLI_H
#define LLVM_FUZZMUTATE_FUZZER_CLI_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

/// Parse cl::opts from a fuzz target commandline.
///
/// This handles all arguments after -ignore_remaining_args=1 as cl::opts.
void parseFuzzerCLOpts(int ArgC, char *ArgV[]);

using FuzzerTestFun = int (*)(const uint8_t *Data, size_t Size);
using FuzzerInitFun = int (*)(int *argc, char ***argv);

/// Runs a fuzz target on the inputs specified on the command line.
///
/// Useful for testing fuzz targets without linking to libFuzzer. Finds inputs
/// in the argument list in a libFuzzer compatible way.
int runFuzzerOnInputs(int ArgC, char *ArgV[], FuzzerTestFun TestOne,
                      FuzzerInitFun Init = [](int *, char ***) { return 0; });

} // end llvm namespace

#endif // LLVM_FUZZMUTATE_FUZZER_CLI_H
