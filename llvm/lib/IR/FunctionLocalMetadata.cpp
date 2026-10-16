//===- FunctionLocalMetadata.cpp - Function-Local Metadata Info -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the function-local metadata types and the
// MDFunctionLocalContext class which stores them.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/FunctionLocalMetadata.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

MDFunctionLocalContext *
MDFunctionLocalContext::getDistinct(LLVMContext &Context, bool NonDebug) {
  MDFunctionLocalContext *NewMD =
      new (0u, Distinct) MDFunctionLocalContext(Context, Distinct, NonDebug);
  NewMD->storeDistinctInContext();
  return NewMD;
}

void MDFunctionLocalContext::getUsedMDNodes(SmallVectorImpl<MDNode*> &Nodes) const {
  // Scopes should have no duplicate entries, just append.
  for (auto Scope : getScopes())
    Nodes.push_back(Scope.get());
  // InlineeContexts may be repeated, so use a set to avoid duplicate entries.
  SmallPtrSet<MDFunctionLocalContext *, 4> SeenInlineeContexts;
  for (auto InlinedCall : getInlinedCalls())
    if (SeenInlineeContexts.insert(InlinedCall.InlineeContext).second)
      Nodes.push_back(InlinedCall.InlineeContext);
}

FLMDBuilder::FLMDBuilder(MDFunctionLocalContext *FLContext,
                         const DISubprogram *SP)
    : FLContext(FLContext) {
  assert(!FLContext->isNonDebug() &&
         "Trying to build debug info in a NonDebug FLContext?");
  assert(SP && "FLMDBuilder requires a valid DISubprogram");
  assert(
      (FLContext->getScopes().empty() || FLContext->getScope(0).Scope == SP) &&
      "FLContext already contains non-matching subprogram?");
  /// If there are no existing FLMD entries, create the required DISubprogram
  /// scope and then insert some commonly-required srclocs.
  if (FLContext->getScopes().empty()) {
    assert(FLContext->getSrcLocs().empty() &&
           FLContext->getInlinedCalls().empty() &&
           "FLContext contains SrcLocs or InlinedCalls without Scopes?");
    getScope(SP);
    getSrcLoc(0, 0, 0);
    getSrcLoc(SP->getLine(), 0, 0);
    getSrcLoc(SP->getScopeLine(), 0, 0);
    return;
  }
  /// Otherwise, load all existing FLMD entries into the builder map.
  for (auto [Idx, SrcLoc] : enumerate(FLContext->getSrcLocs()))
    SrcLocMap.insert({SrcLoc, Idx});
  for (auto [Idx, Scope] : enumerate(FLContext->getScopes()))
    ScopeMap.insert({Scope, Idx});
}
