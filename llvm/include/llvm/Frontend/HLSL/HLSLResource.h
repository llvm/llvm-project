//===- HLSLResource.h - HLSL Resource helper objects ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Resources.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLRESOURCE_H
#define LLVM_FRONTEND_HLSL_HLSLRESOURCE_H

#include "llvm/IR/Metadata.h"

namespace llvm {
class Module;
class GlobalVariable;

namespace hlsl {

class FrontendResource {
  MDNode *Entry;

public:
  FrontendResource(MDNode *E) : Entry(E) {
    assert(Entry->getNumOperands() == 3 && "Unexpected metadata shape");
  }

  FrontendResource(GlobalVariable *GV, StringRef TypeStr, uint32_t Counter);

  GlobalVariable *getGlobalVariable();
  StringRef getSourceType();
  Constant *getID();

  MDNode *getMetadata() { return Entry; }
};
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLRESOURCE_H
