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

DISubprogram *
DebugInfoODRUniquer::getODRSubprogramDecl(Metadata *Scope,
                                          StringRef LinkageName, Metadata *Type,
                                          Metadata *TemplateParams) {
  auto R = FnDecls.find_as(
      DISubprogramODRKey(Scope, LinkageName, Type, TemplateParams));
  if (R == FnDecls.end())
    return nullptr;
  assert(!(*R)->isDefinition() && "definition unexpectedly ODR-uniqued");
  return *R;
}

void DebugInfoODRUniquer::addSubprogramDecl(DISubprogram *SP) {
  assert(!SP->isDefinition() &&
         "only expect declarations DISubprogram ODR uniquing");
  FnDecls.insert(SP);
}
