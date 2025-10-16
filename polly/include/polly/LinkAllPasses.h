//===- polly/LinkAllPasses.h ----------- Reference All Passes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_LINKALLPASSES_H
#define POLLY_LINKALLPASSES_H

#include "polly/Config/config.h"
#include "polly/Support/DumpFunctionPass.h"
#include "polly/Support/DumpModulePass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AlwaysTrue.h"

namespace llvm {
class Pass;
class PassRegistry;
} // namespace llvm

namespace polly {
llvm::Pass *createCodePreparationPass();
llvm::Pass *createScopInlinerPass();
llvm::Pass *createDeadCodeElimWrapperPass();
llvm::Pass *createDependenceInfoPass();
llvm::Pass *createDependenceInfoPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createDependenceInfoWrapperPassPass();
llvm::Pass *
createDependenceInfoPrinterLegacyFunctionPass(llvm::raw_ostream &OS);
llvm::Pass *createDOTOnlyPrinterWrapperPass();
llvm::Pass *createDOTOnlyViewerWrapperPass();
llvm::Pass *createDOTPrinterWrapperPass();
llvm::Pass *createDOTViewerWrapperPass();
llvm::Pass *createJSONExporterPass();
llvm::Pass *createJSONImporterPass();
llvm::Pass *createJSONImporterPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createPollyCanonicalizePass();
llvm::Pass *createScopDetectionWrapperPassPass();
llvm::Pass *createScopDetectionPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createScopInfoRegionPassPass();
llvm::Pass *createScopInfoPrinterLegacyRegionPass(llvm::raw_ostream &OS);
llvm::Pass *createScopInfoWrapperPassPass();
llvm::Pass *createScopInfoPrinterLegacyFunctionPass(llvm::raw_ostream &OS);
llvm::Pass *createIslAstInfoWrapperPassPass();
llvm::Pass *createIslAstInfoPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createCodeGenerationPass();
llvm::Pass *createIslScheduleOptimizerWrapperPass();
llvm::Pass *createIslScheduleOptimizerPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createFlattenSchedulePass();
llvm::Pass *createFlattenSchedulePrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createForwardOpTreeWrapperPass();
llvm::Pass *createForwardOpTreePrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createDeLICMWrapperPass();
llvm::Pass *createDeLICMPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createMaximalStaticExpansionPass();
llvm::Pass *createSimplifyWrapperPass(int);
llvm::Pass *createSimplifyPrinterLegacyPass(llvm::raw_ostream &OS);
llvm::Pass *createPruneUnprofitableWrapperPass();

extern char &CodePreparationID;
} // namespace polly

namespace {
struct PollyForcePassLinking {
  PollyForcePassLinking() {
    // We must reference the passes in such a way that compilers will not delete
    // it all as dead code, even with whole program optimization, yet is
    // effectively a NO-OP.
    if (llvm::getNonFoldableAlwaysTrue())
      return;

    polly::createCodePreparationPass();
    polly::createDeadCodeElimWrapperPass();
    polly::createDependenceInfoPass();
    polly::createDependenceInfoPrinterLegacyPass(llvm::outs());
    polly::createDependenceInfoWrapperPassPass();
    polly::createDependenceInfoPrinterLegacyFunctionPass(llvm::outs());
    polly::createDOTOnlyPrinterWrapperPass();
    polly::createDOTOnlyViewerWrapperPass();
    polly::createDOTPrinterWrapperPass();
    polly::createDOTViewerWrapperPass();
    polly::createJSONExporterPass();
    polly::createJSONImporterPass();
    polly::createJSONImporterPrinterLegacyPass(llvm::outs());
    polly::createScopDetectionWrapperPassPass();
    polly::createScopDetectionPrinterLegacyPass(llvm::outs());
    polly::createScopInfoRegionPassPass();
    polly::createScopInfoPrinterLegacyRegionPass(llvm::outs());
    polly::createScopInfoWrapperPassPass();
    polly::createScopInfoPrinterLegacyFunctionPass(llvm::outs());
    polly::createPollyCanonicalizePass();
    polly::createIslAstInfoWrapperPassPass();
    polly::createIslAstInfoPrinterLegacyPass(llvm::outs());
    polly::createCodeGenerationPass();
    polly::createIslScheduleOptimizerWrapperPass();
    polly::createIslScheduleOptimizerPrinterLegacyPass(llvm::outs());
    polly::createMaximalStaticExpansionPass();
    polly::createFlattenSchedulePass();
    polly::createFlattenSchedulePrinterLegacyPass(llvm::errs());
    polly::createForwardOpTreeWrapperPass();
    polly::createForwardOpTreePrinterLegacyPass(llvm::errs());
    polly::createDeLICMWrapperPass();
    polly::createDeLICMPrinterLegacyPass(llvm::outs());
    polly::createDumpModuleWrapperPass("", true);
    polly::createDumpFunctionWrapperPass("");
    polly::createSimplifyWrapperPass(0);
    polly::createSimplifyPrinterLegacyPass(llvm::outs());
    polly::createPruneUnprofitableWrapperPass();
  }
} PollyForcePassLinking; // Force link by creating a global definition.
} // namespace

namespace llvm {
void initializeCodePreparationPass(llvm::PassRegistry &);
void initializeScopInlinerPass(llvm::PassRegistry &);
void initializeScopDetectionWrapperPassPass(llvm::PassRegistry &);
void initializeScopDetectionPrinterLegacyPassPass(llvm::PassRegistry &);
void initializeScopInfoRegionPassPass(PassRegistry &);
void initializeScopInfoPrinterLegacyRegionPassPass(llvm::PassRegistry &);
void initializeScopInfoWrapperPassPass(PassRegistry &);
void initializeScopInfoPrinterLegacyFunctionPassPass(PassRegistry &);
void initializeDeadCodeElimWrapperPassPass(llvm::PassRegistry &);
void initializeJSONExporterPass(llvm::PassRegistry &);
void initializeJSONImporterPass(llvm::PassRegistry &);
void initializeJSONImporterPrinterLegacyPassPass(llvm::PassRegistry &);
void initializeDependenceInfoPass(llvm::PassRegistry &);
void initializeDependenceInfoPrinterLegacyPassPass(llvm::PassRegistry &);
void initializeDependenceInfoWrapperPassPass(llvm::PassRegistry &);
void initializeDependenceInfoPrinterLegacyFunctionPassPass(
    llvm::PassRegistry &);
void initializeIslAstInfoWrapperPassPass(llvm::PassRegistry &);
void initializeIslAstInfoPrinterLegacyPassPass(llvm::PassRegistry &);
void initializeCodeGenerationPass(llvm::PassRegistry &);
void initializeIslScheduleOptimizerWrapperPassPass(llvm::PassRegistry &);
void initializeIslScheduleOptimizerPrinterLegacyPassPass(llvm::PassRegistry &);
void initializeMaximalStaticExpanderWrapperPassPass(llvm::PassRegistry &);
void initializePollyCanonicalizePass(llvm::PassRegistry &);
void initializeFlattenSchedulePass(llvm::PassRegistry &);
void initializeFlattenSchedulePrinterLegacyPassPass(llvm::PassRegistry &);
void initializeForwardOpTreeWrapperPassPass(llvm::PassRegistry &);
void initializeForwardOpTreePrinterLegacyPassPass(PassRegistry &);
void initializeDeLICMWrapperPassPass(llvm::PassRegistry &);
void initializeDeLICMPrinterLegacyPassPass(llvm::PassRegistry &);
void initializeSimplifyWrapperPassPass(llvm::PassRegistry &);
void initializeSimplifyPrinterLegacyPassPass(llvm::PassRegistry &);
void initializePruneUnprofitableWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif
