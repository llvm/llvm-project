//===- Transforms/PGOInstrumentation.h - PGO gen/use passes  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for IR based instrumentation passes (
/// (profile-gen, and profile-use).
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PGOINSTRUMENTATION_H
#define LLVM_TRANSFORMS_PGOINSTRUMENTATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation.h"

namespace llvm {

/// The instrumentation (profile-instr-gen) pass for IR based PGO.
class PGOInstrumentationGen : public PassInfoMixin<PGOInstrumentationGen> {
public:
  PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);
};

/// The profile annotation (profile-instr-use) pass for IR based PGO.
class PGOInstrumentationUse : public PassInfoMixin<PGOInstrumentationUse> {
public:
  PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);
  PGOInstrumentationUse(std::string Filename = "");

private:
  std::string ProfileFileName;
};

/// The indirect function call promotion pass.
class PGOIndirectCallPromotion : public PassInfoMixin<PGOIndirectCallPromotion> {
public:
  PGOIndirectCallPromotion(bool IsInLTO = false) : InLTO(IsInLTO) {}
  PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);
private:
  bool InLTO;
};

} // End llvm namespace
#endif
