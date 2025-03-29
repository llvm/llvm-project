//===- PassManagerOptions.cpp - PassManager Command Line Options ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

namespace {
struct PassManagerOptions {
  //===--------------------------------------------------------------------===//
  // Crash Reproducer Generator
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<std::string> reproducerFile{
      "mlir-pass-pipeline-crash-reproducer",
      llvm::cl::desc("Generate a .mlir reproducer file at the given output path"
                     " if the pass manager crashes or fails")};
  llvm::cl::opt<bool> localReproducer{
      "mlir-pass-pipeline-local-reproducer",
      llvm::cl::desc("When generating a crash reproducer, attempt to generated "
                     "a reproducer with the smallest pipeline."),
      llvm::cl::init(false)};

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//
  PassNameCLParser printBefore{"mlir-print-ir-before",
                               "Print IR before specified passes"};
  PassNameCLParser printAfter{"mlir-print-ir-after",
                              "Print IR after specified passes"};
  llvm::cl::opt<bool> printBeforeAll{
      "mlir-print-ir-before-all", llvm::cl::desc("Print IR before each pass"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterAll{"mlir-print-ir-after-all",
                                    llvm::cl::desc("Print IR after each pass"),
                                    llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterChange{
      "mlir-print-ir-after-change",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the IR changed"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterFailure{
      "mlir-print-ir-after-failure",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the pass failed"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printModuleScope{
      "mlir-print-ir-module-scope",
      llvm::cl::desc("When printing IR for print-ir-[before|after]{-all} "
                     "always print the top-level operation"),
      llvm::cl::init(false)};
  llvm::cl::opt<std::string> printTreeDir{
      "mlir-print-ir-tree-dir",
      llvm::cl::desc("When printing the IR before/after a pass, print file "
                     "tree rooted at this directory. Use in conjunction with "
                     "mlir-print-ir-* flags")};

  /// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
  void addPrinterInstrumentation(PassManager &pm);

  //===--------------------------------------------------------------------===//
  // Pass Statistics
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> passStatistics{
      "mlir-pass-statistics",
      llvm::cl::desc("Display the statistics of each pass")};
  llvm::cl::opt<PassDisplayMode> passStatisticsDisplayMode{
      "mlir-pass-statistics-display",
      llvm::cl::desc("Display method for pass statistics"),
      llvm::cl::init(PassDisplayMode::Pipeline),
      llvm::cl::values(
          clEnumValN(
              PassDisplayMode::List, "list",
              "display the results in a merged list sorted by pass name"),
          clEnumValN(PassDisplayMode::Pipeline, "pipeline",
                     "display the results with a nested pipeline view"))};
};
} // namespace

static llvm::ManagedStatic<PassManagerOptions> options;

/// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
void PassManagerOptions::addPrinterInstrumentation(PassManager &pm) {
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  // Handle print-before.
  if (printBeforeAll) {
    // If we are printing before all, then just return true for the filter.
    shouldPrintBeforePass = [](Pass *, Operation *) { return true; };
  } else if (printBefore.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print before, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintBeforePass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && printBefore.contains(passInfo);
    };
  }

  // Handle print-after.
  if (printAfterAll || printAfterFailure) {
    // If we are printing after all or failure, then just return true for the
    // filter.
    shouldPrintAfterPass = [](Pass *, Operation *) { return true; };
  } else if (printAfter.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print after, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintAfterPass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && printAfter.contains(passInfo);
    };
  }

  // If there are no valid printing filters, then just return.
  if (!shouldPrintBeforePass && !shouldPrintAfterPass)
    return;

  // Otherwise, add the IR printing instrumentation.
  if (!printTreeDir.empty()) {
    pm.enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass,
                                  printModuleScope, printAfterChange,
                                  printAfterFailure, printTreeDir);
    return;
  }

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printModuleScope, printAfterChange, printAfterFailure,
                      llvm::errs());
}

void mlir::registerPassManagerCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;
}

LogicalResult mlir::applyPassManagerCLOptions(PassManager &pm) {
  if (!options.isConstructed())
    return failure();

  // Generate a reproducer on crash/failure.
  if (options->reproducerFile.getNumOccurrences())
    pm.enableCrashReproducerGeneration(options->reproducerFile,
                                       options->localReproducer);

  // Enable statistics dumping.
  if (options->passStatistics)
    pm.enableStatistics(options->passStatisticsDisplayMode);

  if (options->printModuleScope && pm.getContext()->isMultithreadingEnabled()) {
    emitError(UnknownLoc::get(pm.getContext()))
        << "IR print for module scope can't be setup on a pass-manager "
           "without disabling multi-threading first.\n";
    return failure();
  }

  // Add the IR printing instrumentation.
  options->addPrinterInstrumentation(pm);
  return success();
}

void mlir::applyDefaultTimingPassManagerCLOptions(PassManager &pm) {
  // Create a temporary timing manager for the PM to own, apply its CL options,
  // and pass it to the PM.
  auto tm = std::make_unique<DefaultTimingManager>();
  applyDefaultTimingManagerCLOptions(*tm);
  pm.enableTiming(std::move(tm));
}
