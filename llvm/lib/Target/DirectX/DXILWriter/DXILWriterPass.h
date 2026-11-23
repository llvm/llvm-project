//===-- DXILWriterPass.h - Bitcode writing pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides a bitcode writing pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_DXILWriterPass_H
#define LLVM_BITCODE_DXILWriterPass_H

#include "DirectX.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class raw_ostream;

/// Create and return a pass that writes the module to the specified
/// ostream.
class DXILWriterPass : public RequiredPassInfoMixin<DXILWriterPass> {
  raw_ostream &OS;

public:
  explicit DXILWriterPass(raw_ostream &OS = dbgs()) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createDXILWriterLegacyPass(raw_ostream &Str);

/// Create and return a pass that writes the module to a global variable in the
/// module for later emission in the MCStreamer.
class DXILEmbedderPass : public RequiredPassInfoMixin<DXILEmbedderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createDXILEmbedderLegacyPass();

} // namespace llvm

#endif
