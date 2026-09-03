//===--- IRPrintingPasses.cpp - Module and Function printing passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PrintModulePass and PrintFunctionPass implementations for the legacy pass
// manager.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

static void printModule(raw_ostream &OS, StringRef Banner,
                        bool ShouldPreserveUseListOrder, Module &M) {
  if (llvm::isFunctionInPrintList("*")) {
    if (!Banner.empty())
      OS << Banner << "\n";
    M.print(OS, nullptr, ShouldPreserveUseListOrder);
    return;
  }

  bool BannerPrinted = false;
  for (const auto &F : M.functions()) {
    if (!llvm::isFunctionInPrintList(F.getName()))
      continue;
    if (!BannerPrinted && !Banner.empty()) {
      OS << Banner << "\n";
      BannerPrinted = true;
    }
    F.print(OS);
  }
}

class PrintModulePassWrapper : public ModulePass {
  raw_ostream &OS;
  std::string Banner;
  bool ShouldPreserveUseListOrder;
  bool ShouldRenumberMetadata;

public:
  static char ID;
  PrintModulePassWrapper()
      : ModulePass(ID), OS(dbgs()), ShouldPreserveUseListOrder(false),
        ShouldRenumberMetadata(false) {}
  PrintModulePassWrapper(raw_ostream &OS, const std::string &Banner,
                         bool ShouldPreserveUseListOrder,
                         bool ShouldRenumberMetadata)
      : ModulePass(ID), OS(OS), Banner(Banner),
        ShouldPreserveUseListOrder(ShouldPreserveUseListOrder),
        ShouldRenumberMetadata(ShouldRenumberMetadata) {}

  bool runOnModule(Module &M) override {
    if (ShouldRenumberMetadata)
      M.renumberMetadataForAssembly();
    printModule(OS, Banner, ShouldPreserveUseListOrder, M);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  StringRef getPassName() const override { return "Print Module IR"; }
};

class PrintFunctionPassWrapper : public FunctionPass {
  raw_ostream &OS;
  std::string Banner;

public:
  static char ID;
  PrintFunctionPassWrapper() : FunctionPass(ID), OS(dbgs()) {}
  PrintFunctionPassWrapper(raw_ostream &OS, const std::string &Banner)
      : FunctionPass(ID), OS(OS), Banner(Banner) {}

  // This pass just prints a banner followed by the function as it's processed.
  bool runOnFunction(Function &F) override {
    if (isFunctionInPrintList(F.getName())) {
      if (forcePrintModuleIR()) {
        OS << Banner << " (function: " << F.getName() << ")\n";
        F.getParent()->print(OS, nullptr);
      } else
        OS << Banner << '\n' << static_cast<Value &>(F);
    }

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  StringRef getPassName() const override { return "Print Function IR"; }
};

} // namespace

char PrintModulePassWrapper::ID = 0;
INITIALIZE_PASS(PrintModulePassWrapper, "print-module",
                "Print module to stderr", false, true)
char PrintFunctionPassWrapper::ID = 0;
INITIALIZE_PASS(PrintFunctionPassWrapper, "print-function",
                "Print function to stderr", false, true)

ModulePass *llvm::createPrintModulePass(llvm::raw_ostream &OS,
                                        const std::string &Banner,
                                        bool ShouldPreserveUseListOrder) {
  return createPrintModulePass(OS, Banner, ShouldPreserveUseListOrder,
                               /*ShouldRenumberMetadata=*/false);
}

ModulePass *llvm::createPrintModulePass(llvm::raw_ostream &OS,
                                        const std::string &Banner,
                                        bool ShouldPreserveUseListOrder,
                                        bool ShouldRenumberMetadata) {
  return new PrintModulePassWrapper(OS, Banner, ShouldPreserveUseListOrder,
                                    ShouldRenumberMetadata);
}

FunctionPass *llvm::createPrintFunctionPass(llvm::raw_ostream &OS,
                                            const std::string &Banner) {
  return new PrintFunctionPassWrapper(OS, Banner);
}

bool llvm::isIRPrintingPass(Pass *P) {
  const char *PID = (const char *)P->getPassID();

  return (PID == &PrintModulePassWrapper::ID) ||
         (PID == &PrintFunctionPassWrapper::ID);
}
