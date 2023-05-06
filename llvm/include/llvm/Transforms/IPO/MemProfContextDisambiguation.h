//==- MemProfContextDisambiguation.h - Context Disambiguation ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements support for context disambiguation of allocation calls for profile
// guided heap optimization using memprof metadata. See implementation file for
// details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H
#define LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/PassManager.h"
#include <functional>

namespace llvm {
class GlobalValueSummary;
class Module;
class ModuleSummaryIndex;

class MemProfContextDisambiguation
    : public PassInfoMixin<MemProfContextDisambiguation> {
  /// Run the context disambiguator on \p M, returns true if any changes made.
  bool processModule(Module &M);

public:
  MemProfContextDisambiguation() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  void run(ModuleSummaryIndex &Index,
           function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
               isPrevailing);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H
