//===- llvm/IR/DebugInfoODRUniquer.cpp - Debug Information Builder --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a class used to merge debug info for ODR types.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoODRUniquer.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

DISubprogram *DebugInfoODRUniquer::getODRSubprogramDecl(DIScope *Scope,
                                                        StringRef LinkageName) {
  // Only methods, which have a type scope, are eligable for ODR uniquing.
  auto *CT = dyn_cast_or_null<DICompositeType>(Scope);
  if (!CT || !CT->getRawIdentifier())
    return nullptr;

  auto R =
      FnDecls.find_as(DISubprogramODRKey(CT->getIdentifier(), LinkageName));
  if (R == FnDecls.end())
    return nullptr;

  assert(!(*R)->isDefinition() && "definition unexpectedly ODR-uniqued");
  return *R;
}

void DebugInfoODRUniquer::addSubprogramDecl(DISubprogram *SP) {
  assert(!SP->isDefinition() &&
         "only expect declarations DISubprogram ODR uniquing");
  assert(!SP->isDistinct() && "expect declarations to be uniqued");

  if (SP->getLinkageName().empty())
    return;
  // Only methods, which have a type scope, are eligable for ODR uniquing.
  auto *CT = dyn_cast_or_null<DICompositeType>(SP->getScope());
  if (!CT || !CT->getRawIdentifier())
    return;

  FnDecls.insert(SP);
}

void DebugInfoODRUniquer::addUnresolvedODRSubprogramDecl(TempDISubprogram SP) {
  assert(!SP->isDefinition() && "expected declarations only");
  if (SP->getLinkageName().empty())
    return;
  PendingFnDecls.push_back(std::move(SP));
}

void DebugInfoODRUniquer::finalizeUnresolvedSubprogramDecls() {
  while (!PendingFnDecls.empty()) {
    TempDISubprogram SP = PendingFnDecls.pop_back_val();
    if (MDNode *Scope = dyn_cast_or_null<MDNode>(SP->getRawScope())) {
      assert(!SP->getLinkageName().empty() && "expcted linkage name");
      assert(!Scope->isTemporary() &&
             "expected temporary scope to be replaced");
      if (DISubprogram *Existing = getODRSubprogramDecl(cast<DIScope>(Scope),
                                                        SP->getLinkageName())) {
        SP->replaceAllUsesWith(Existing);
        continue;
      }
    }

    // Else, keep this one for good. All of this machinery is operating on
    // declarations, which shouldn't be distinct. Save it in our map in case
    // other instances need to be ODR-uniqued to it.
    auto *NewSP = DISubprogram::replaceWithPermanent(std::move(SP));
    assert(NewSP->isResolved() && "expected SP to be resolved");
    addSubprogramDecl(NewSP);
  }
}