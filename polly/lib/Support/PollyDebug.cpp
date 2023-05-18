//===-PollyDebug.cpp -Provide support for debugging Polly passes-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to aid printing Debug Info of all polly passes.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/PollyDebug.h"
#include "llvm/Support/CommandLine.h"

using namespace polly;
using namespace llvm;

bool PollyDebugFlag;
bool polly::getPollyDebugFlag() { return PollyDebugFlag; }

// -debug - Command line option to enable the DEBUG statements in the passes.
// This flag may only be enabled in debug builds.
static cl::opt<bool, true>
    PollyDebug("polly-debug",
               cl::desc("Enable debug output for only polly passes."),
               cl::Hidden, cl::location(PollyDebugFlag), cl::ZeroOrMore);
