//===- polly/CodeGeneration.h - The Polly code generator --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGENERATION_H
#define POLLY_CODEGENERATION_H

#include "polly/CodeGen/IRBuilder.h"

namespace llvm {
class RegionInfo;
}

namespace polly {
class IslAstInfo;

using llvm::BasicBlock;

enum VectorizerChoice {
  VECTORIZER_NONE,
  VECTORIZER_STRIPMINE,
};
extern VectorizerChoice PollyVectorizerChoice;

/// Mark a basic block unreachable.
///
/// Marks the basic block @p Block unreachable by equipping it with an
/// UnreachableInst.
void markBlockUnreachable(BasicBlock &Block, PollyIRBuilder &Builder);

extern bool PerfMonitoring;

bool runCodeGeneration(Scop &S, llvm::RegionInfo &RI, IslAstInfo &AI);
} // namespace polly

#endif // POLLY_CODEGENERATION_H
