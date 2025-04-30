//===- SymbolTracker.h - Symbol Tracking Pass -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the prototypes and definitions related to the Symbol
// Tracker pass.
//
// The Symbol Tracker pass embeds global variables in the IR module to allow
// a runtime program analysis tool that has access to the IR that was used
// to generate the native code, to match IR symbols to their physical
// native location at runtime within the process address space, and do so
// in a target-independent way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SYMBOLTRACKER_H
#define LLVM_TRANSFORMS_UTILS_SYMBOLTRACKER_H

#include "llvm/IR/PassManager.h"
#include <list>
#include <memory>
#include <string>

namespace llvm {

class MemoryBuffer;
class Module;
class ModulePass;

class EmbedSymbolTrackersPass : public PassInfoMixin<EmbedSymbolTrackersPass> {
public:
  EmbedSymbolTrackersPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SYMBOLTRACKER_H
