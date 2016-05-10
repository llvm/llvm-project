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

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Instrumentation.h"

namespace llvm {

/// The instrumentation (profile-instr-gen) pass for IR based PGO.
class PGOInstrumentationGen : public PassInfoMixin<PGOInstrumentationGen> {
public:
  PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);
};

} // End llvm namespace
#endif
