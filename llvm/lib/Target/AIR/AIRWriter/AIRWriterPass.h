//===- AIRWriterPass.h - Emit a .metallib container -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Module pass that serializes the module to a AIR `.metallib` container.
/// This is the final step of the AIR backend's object-file emission path; it
/// reconstructs typed-pointer info (AIR v1 bitcode requires typed pointers) and
/// hands it to the forked metallib writer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_AIRWRITERPASS_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_AIRWRITERPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
class raw_pwrite_stream;

ModulePass *createAIRWriterPass(raw_pwrite_stream &Out);

/// New pass-manager wrapper that emits a `.metallib` container to \p Out.
/// The output stream must outlive the pass.
class AIRWriterPass : public PassInfoMixin<AIRWriterPass> {
  raw_pwrite_stream &OS;

public:
  explicit AIRWriterPass(raw_pwrite_stream &Out) : OS(Out) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRWRITER_AIRWRITERPASS_H
