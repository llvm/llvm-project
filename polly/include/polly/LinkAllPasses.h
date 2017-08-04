//===- polly/LinkAllPasses.h ----------- Reference All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_LINKALLPASSES_H
#define POLLY_LINKALLPASSES_H

#include "polly/CodeGen/PPCGCodeGeneration.h"
#include "polly/Config/config.h"
#include "polly/PruneUnprofitable.h"
#include "polly/Simplify.h"
#include "polly/Support/DumpModulePass.h"
#include "llvm/ADT/StringRef.h"
#include <cstdlib>

namespace llvm {
class Pass;
class PassInfo;
class PassRegistry;
class RegionPass;
} // namespace llvm

namespace polly {
llvm::Pass *createCodePreparationPass();
llvm::Pass *createDeadCodeElimPass();
llvm::Pass *createDependenceInfoPass();
llvm::Pass *createDependenceInfoWrapperPassPass();
llvm::Pass *createDOTOnlyPrinterPass();
llvm::Pass *createDOTOnlyViewerPass();
llvm::Pass *createDOTPrinterPass();
llvm::Pass *createDOTViewerPass();
llvm::Pass *createJSONExporterPass();
llvm::Pass *createJSONImporterPass();
llvm::Pass *createPollyCanonicalizePass();
llvm::Pass *createPolyhedralInfoPass();
llvm::Pass *createScopDetectionWrapperPassPass();
llvm::Pass *createScopInfoRegionPassPass();
llvm::Pass *createScopInfoWrapperPassPass();
llvm::Pass *createIslAstInfoWrapperPassPass();
llvm::Pass *createCodeGenerationPass();
#ifdef GPU_CODEGEN
llvm::Pass *createPPCGCodeGenerationPass(GPUArch Arch = GPUArch::NVPTX64,
                                         GPURuntime Runtime = GPURuntime::CUDA);
#endif
llvm::Pass *createIslScheduleOptimizerPass();
llvm::Pass *createFlattenSchedulePass();
llvm::Pass *createDeLICMPass();

extern char &CodePreparationID;
} // namespace polly

namespace {
struct PollyForcePassLinking {
  PollyForcePassLinking() {
    // We must reference the passes in such a way that compilers will not
    // delete it all as dead code, even with whole program optimization,
    // yet is effectively a NO-OP. As the compiler isn't smart enough
    // to know that getenv() never returns -1, this will do the job.
    if (std::getenv("bar") != (char *)-1)
      return;

    polly::createCodePreparationPass();
    polly::createDeadCodeElimPass();
    polly::createDependenceInfoPass();
    polly::createDOTOnlyPrinterPass();
    polly::createDOTOnlyViewerPass();
    polly::createDOTPrinterPass();
    polly::createDOTViewerPass();
    polly::createJSONExporterPass();
    polly::createJSONImporterPass();
    polly::createScopDetectionWrapperPassPass();
    polly::createScopInfoRegionPassPass();
    polly::createPollyCanonicalizePass();
    polly::createPolyhedralInfoPass();
    polly::createIslAstInfoWrapperPassPass();
    polly::createCodeGenerationPass();
#ifdef GPU_CODEGEN
    polly::createPPCGCodeGenerationPass();
#endif
    polly::createIslScheduleOptimizerPass();
    polly::createFlattenSchedulePass();
    polly::createDeLICMPass();
    polly::createDumpModulePass("", true);
    polly::createSimplifyPass();
    polly::createPruneUnprofitablePass();
  }
} PollyForcePassLinking; // Force link by creating a global definition.
} // namespace

namespace llvm {
class PassRegistry;
void initializeCodePreparationPass(llvm::PassRegistry &);
void initializeDeadCodeElimPass(llvm::PassRegistry &);
void initializeJSONExporterPass(llvm::PassRegistry &);
void initializeJSONImporterPass(llvm::PassRegistry &);
void initializeIslAstInfoWrapperPassPass(llvm::PassRegistry &);
void initializeCodeGenerationPass(llvm::PassRegistry &);
#ifdef GPU_CODEGEN
void initializePPCGCodeGenerationPass(llvm::PassRegistry &);
#endif
void initializeIslScheduleOptimizerPass(llvm::PassRegistry &);
void initializePollyCanonicalizePass(llvm::PassRegistry &);
void initializeFlattenSchedulePass(llvm::PassRegistry &);
void initializeDeLICMPass(llvm::PassRegistry &);
} // namespace llvm

#endif
