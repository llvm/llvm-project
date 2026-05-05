//===- Transforms/IPO/PGOVerify.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the pass-instrumentation registration hook for
/// `-verify-ipgo` diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PGOVERIFY_H
#define LLVM_TRANSFORMS_IPO_PGOVERIFY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class Function;
class Loop;
class Module;
class PassInstrumentationCallbacks;

/// Registers `-verify-ipgo` diagnostics with pass instrumentation.
class IPGOVerifier {
public:
  /// Register post-pass callback hooks used by `-verify-ipgo` diagnostics.
  ///
  /// \param PIC Pass instrumentation callback registry.
  LLVM_ABI void registerCallbacks(PassInstrumentationCallbacks &PIC);

  /// Dispatch post-pass handling by IR unit type.
  ///
  /// \param PassID Name of the pass that completed.
  /// \param IR IR unit received from pass instrumentation callbacks.
  LLVM_ABI void runAfterPass(StringRef PassID, Any IR);

private:
  /// Handle module callbacks by delegating each function to function handler.
  void runAfterPass(const Module *M);

  /// Per-function callback handler.
  void runAfterPass(const Function *F);

  /// Handle SCC callbacks by delegating each function to function handler.
  void runAfterPass(const LazyCallGraph::SCC *C);

  /// Handle loop callbacks by delegating to containing function handler.
  void runAfterPass(const Loop *L);
};

} // end namespace llvm
#endif // LLVM_TRANSFORMS_IPO_PGOVERIFY_H
