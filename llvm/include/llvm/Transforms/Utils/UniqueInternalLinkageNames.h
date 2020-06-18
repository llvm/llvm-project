//===-- UniqueInternalLinkageNames.h - Uniq. Int. Linkage Names -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unique naming of internal linkage symbols with option
// -funique-internal-linkage-symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNIQUEINTERNALLINKAGENAMES_H
#define LLVM_TRANSFORMS_UTILS_UNIQUEINTERNALLINKAGENAMES_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Simple pass that provides a name to every anonymous globals.
class UniqueInternalLinkageNamesPass
    : public PassInfoMixin<UniqueInternalLinkageNamesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_UNIQUEINTERNALLINKAGENAMES_H
