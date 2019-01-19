//===- Transforms/Instrumentation/PGOInstrumentation.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the interface for IR based instrumentation passes (
/// (profile-gen, and profile-use).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PGOINSTRUMENTATION_H
#define LLVM_TRANSFORMS_PGOINSTRUMENTATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/PassManager.h"
#include <cstdint>
#include <string>

namespace llvm {

class Function;
class Instruction;
class Module;

/// The instrumentation (profile-instr-gen) pass for IR based PGO.
class PGOInstrumentationGen : public PassInfoMixin<PGOInstrumentationGen> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// The profile annotation (profile-instr-use) pass for IR based PGO.
class PGOInstrumentationUse : public PassInfoMixin<PGOInstrumentationUse> {
public:
  PGOInstrumentationUse(std::string Filename = "",
                        std::string RemappingFilename = "");

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::string ProfileFileName;
  std::string ProfileRemappingFileName;
};

/// The indirect function call promotion pass.
class PGOIndirectCallPromotion : public PassInfoMixin<PGOIndirectCallPromotion> {
public:
  PGOIndirectCallPromotion(bool IsInLTO = false, bool SamplePGO = false)
      : InLTO(IsInLTO), SamplePGO(SamplePGO) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  bool InLTO;
  bool SamplePGO;
};

/// The profile size based optimization pass for memory intrinsics.
class PGOMemOPSizeOpt : public PassInfoMixin<PGOMemOPSizeOpt> {
public:
  PGOMemOPSizeOpt() = default;

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void setProfMetadata(Module *M, Instruction *TI, ArrayRef<uint64_t> EdgeCounts,
                     uint64_t MaxCount);

void setIrrLoopHeaderMetadata(Module *M, Instruction *TI, uint64_t Count);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_PGOINSTRUMENTATION_H
