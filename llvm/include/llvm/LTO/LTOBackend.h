//===-LTOBackend.h - LLVM Link Time Optimizer Backend ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the "backend" phase of LTO, i.e. it performs
// optimization and code generation on a loaded module. It is generally used
// internally by the LTO class but can also be used independently, for example
// to implement a standalone ThinLTO backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_LTOBACKEND_H
#define LLVM_LTO_LTOBACKEND_H

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/FunctionImport.h"

namespace llvm {

class BitcodeModule;
class Error;
class Module;
class Target;

namespace lto {

/// Runs a regular LTO backend. The regular LTO backend can also act as the
/// regular LTO phase of ThinLTO, which may need to access the combined index.
Error backend(const Config &C, AddStreamFn AddStream,
              unsigned ParallelCodeGenParallelismLevel,
              std::unique_ptr<Module> M, ModuleSummaryIndex &CombinedIndex);

/// Runs a ThinLTO backend.
Error thinBackend(const Config &C, unsigned Task, AddStreamFn AddStream,
                  Module &M, const ModuleSummaryIndex &CombinedIndex,
                  const FunctionImporter::ImportMapTy &ImportList,
                  const GVSummaryMapTy &DefinedGlobals,
                  MapVector<StringRef, BitcodeModule> &ModuleMap);

Error finalizeOptimizationRemarks(
    std::unique_ptr<ToolOutputFile> DiagOutputFile);
}
}

#endif
