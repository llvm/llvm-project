//===- Standard pass instrumentations handling ----------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines IR-printing pass instrumentation callbacks as well as
/// StandardInstrumentations class that manages standard pass instrumentations.
///
//===----------------------------------------------------------------------===//

#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
namespace PrintIR {

//===----------------------------------------------------------------------===//
// IR-printing instrumentation
//===----------------------------------------------------------------------===//

/// Generic IR-printing helper that unpacks a pointer to IRUnit wrapped into
/// llvm::Any and does actual print job.
void unwrapAndPrint(StringRef Banner, Any IR) {
  if (any_isa<const CallGraphSCC *>(IR) ||
      any_isa<const LazyCallGraph::SCC *>(IR))
    return;

  SmallString<40> Extra{"\n"};
  const Module *M = nullptr;
  if (any_isa<const Module *>(IR)) {
    M = any_cast<const Module *>(IR);
  } else if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    if (!llvm::isFunctionInPrintList(F->getName()))
      return;
    if (!llvm::forcePrintModuleIR()) {
      dbgs() << Banner << Extra << static_cast<const Value &>(*F);
      return;
    }
    M = F->getParent();
    Extra = formatv(" (function: {0})\n", F->getName());
  } else if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    const Function *F = L->getHeader()->getParent();
    if (!isFunctionInPrintList(F->getName()))
      return;
    if (!llvm::forcePrintModuleIR()) {
      llvm::printLoop(const_cast<Loop &>(*L), dbgs(), Banner);
      return;
    }
    M = F->getParent();
    {
      std::string LoopName;
      raw_string_ostream ss(LoopName);
      L->getHeader()->printAsOperand(ss, false);
      Extra = formatv(" (loop: {0})\n", ss.str());
    }
  }
  if (M) {
    dbgs() << Banner << Extra;
    M->print(dbgs(), nullptr, false);
  } else {
    llvm_unreachable("Unknown wrapped IR type");
  }
}

bool printBeforePass(StringRef PassID, Any IR) {
  if (!llvm::shouldPrintBeforePass(PassID))
    return true;

  if (PassID.startswith("PassManager<") || PassID.contains("PassAdaptor<"))
    return true;

  SmallString<20> Banner = formatv("*** IR Dump Before {0} ***", PassID);
  unwrapAndPrint(Banner, IR);
  return true;
}

void printAfterPass(StringRef PassID, Any IR) {
  if (!llvm::shouldPrintAfterPass(PassID))
    return;

  if (PassID.startswith("PassManager<") || PassID.contains("PassAdaptor<"))
    return;

  SmallString<20> Banner = formatv("*** IR Dump After {0} ***", PassID);
  unwrapAndPrint(Banner, IR);
  return;
}
} // namespace PrintIR
} // namespace

void StandardInstrumentations::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (llvm::shouldPrintBeforePass())
    PIC.registerBeforePassCallback(PrintIR::printBeforePass);
  if (llvm::shouldPrintAfterPass())
    PIC.registerAfterPassCallback(PrintIR::printAfterPass);
}
