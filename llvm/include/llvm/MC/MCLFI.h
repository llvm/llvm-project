//===- MCLFI.h - LFI-specific code for MC -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file was written by the LFI and Native Client authors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCContext;
class MCStreamer;
class Triple;

LLVM_ABI void initializeLFIMCStreamer(MCStreamer &Streamer, MCContext &Ctx,
                                      const Triple &TheTriple);

} // namespace llvm
