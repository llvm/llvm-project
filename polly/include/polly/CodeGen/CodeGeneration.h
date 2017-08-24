//===------ polly/CodeGeneration.h - The Polly code generator *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGENERATION_H
#define POLLY_CODEGENERATION_H

#include "IRBuilder.h"
#include "polly/Config/config.h"
#include "polly/ScopPass.h"
#include "isl/map.h"
#include "isl/set.h"

namespace polly {
enum VectorizerChoice {
  VECTORIZER_NONE,
  VECTORIZER_STRIPMINE,
  VECTORIZER_POLLY,
};
extern VectorizerChoice PollyVectorizerChoice;

/// Mark a basic block unreachable.
///
/// Marks the basic block @p Block unreachable by equipping it with an
/// UnreachableInst.
void markBlockUnreachable(BasicBlock &Block, PollyIRBuilder &Builder);

struct CodeGenerationPass : public PassInfoMixin<CodeGenerationPass> {
  PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                        ScopStandardAnalysisResults &AR, SPMUpdater &U);
};

extern bool PerfMonitoring;
} // namespace polly

#endif // POLLY_CODEGENERATION_H
