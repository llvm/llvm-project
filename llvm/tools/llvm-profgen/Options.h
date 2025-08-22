//===-- Options.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_PROFGEN_OPTIONS_H
#define LLVM_TOOLS_LLVM_PROFGEN_OPTIONS_H

#include "llvm/Support/CommandLine.h"

namespace llvm {

extern cl::OptionCategory ProfGenCategory;

extern cl::opt<std::string> OutputFilename;
extern cl::opt<bool> ShowDisassemblyOnly;
extern cl::opt<bool> ShowSourceLocations;
extern cl::opt<bool> SkipSymbolization;
extern cl::opt<bool> ShowDetailedWarning;
extern cl::opt<bool> InferMissingFrames;
extern cl::opt<bool> EnableCSPreInliner;
extern cl::opt<bool> UseContextCostForPreInliner;

} // end namespace llvm

#endif
