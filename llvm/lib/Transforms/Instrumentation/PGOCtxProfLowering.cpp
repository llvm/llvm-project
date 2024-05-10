//===- PGOCtxProfLowering.cpp - Contextual PGO Instr. Lowering ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "llvm/Transforms/Instrumentation/PGOCtxProfLowering.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::list<std::string> ContextRoots(
    "profile-context-root", cl::Hidden,
    cl::desc(
        "A function name, assumed to be global, which will be treated as the "
        "root of an interesting graph, which will be profiled independently "
        "from other similar graphs."));

bool PGOCtxProfLoweringPass::isContextualIRPGOEnabled() {
  return !ContextRoots.empty();
}
