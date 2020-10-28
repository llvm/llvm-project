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
                  MapVector<StringRef, BitcodeModule> &ModuleMap,
                  const std::vector<uint8_t> *CmdArgs = nullptr);

Error finalizeOptimizationRemarks(
    std::unique_ptr<ToolOutputFile> DiagOutputFile);

/// Returns the BitcodeModule that is ThinLTO.
BitcodeModule *findThinLTOModule(MutableArrayRef<BitcodeModule> BMs);

/// Variant of the above.
Expected<BitcodeModule> findThinLTOModule(MemoryBufferRef MBRef);

/// Distributed ThinLTO: load the referenced modules, keeping their buffers
/// alive in the provided OwnedImportLifetimeManager. Returns false if the
/// operation failed.
bool loadReferencedModules(
    const Module &M, const ModuleSummaryIndex &CombinedIndex,
    FunctionImporter::ImportMapTy &ImportList,
    MapVector<llvm::StringRef, llvm::BitcodeModule> &ModuleMap,
    std::vector<std::unique_ptr<llvm::MemoryBuffer>>
        &OwnedImportsLifetimeManager);
}
}

#endif
