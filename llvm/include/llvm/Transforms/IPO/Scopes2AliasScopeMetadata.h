//=== Scopes2AliasScopeMetadata.h - Transform !alias.scope metadata. C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Currently this pass is C language specific
//
// Propagates !alias.scope metadata based on scope metadata emitted by the
// frontend. The !alias.scope metadata nodes contain information about pointer
// alias annotations (such as the restrict qualifier in C) and the scopes in
// which these pointers were declared. According to certain rules described at
// https://discourse.llvm.org/t/rfc-yet-another-llvm-restrict-support/87612,
// we can annotate certain loads/stores with !noalias metadata to enable further
// optimizations.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SCOPES2ALIASSCOPEMETADATA_H
#define LLVM_TRANSFORMS_IPO_SCOPES2ALIASSCOPEMETADATA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class MDNode;
class Instruction;
class LoadInst;

struct Scopes2AliasScopeMetadataPass
    : public PassInfoMixin<Scopes2AliasScopeMetadataPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  bool convertScopeInfo2AliasScopeMetadata(Module &M);

private:
  std::unordered_map<std::string, MDNode *> ScopeMap;
  SmallVector<std::pair<Instruction *, std::string>> NoAliasInstrs;

  bool runOnFunction(Function &F);

  bool propagateAliasScopes(Function &F);

  void setNoAliasMetadata() const;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_SCOPES2ALIASSCOPEMETADATA_H
