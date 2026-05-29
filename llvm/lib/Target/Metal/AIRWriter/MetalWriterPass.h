//===- MetalWriterPass.h - Emit a .metallib container -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Module pass that serializes the module to a Metal `.metallib` container.
/// This is the final step of the Metal backend's object-file emission path; it
/// reconstructs typed-pointer info (AIR v1 bitcode requires typed pointers) and
/// hands it to the forked metallib writer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_METALWRITERPASS_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_METALWRITERPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
class raw_pwrite_stream;

ModulePass *createMetalWriterPass(raw_pwrite_stream &Out);

/// New pass-manager wrapper that emits a `.metallib` container to \p Out.
/// The output stream must outlive the pass.
class MetalWriterPass : public PassInfoMixin<MetalWriterPass> {
  raw_pwrite_stream &OS;

public:
  explicit MetalWriterPass(raw_pwrite_stream &Out) : OS(Out) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_METALWRITERPASS_H
