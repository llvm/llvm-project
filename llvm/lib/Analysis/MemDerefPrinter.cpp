//===- MemDerefPrinter.cpp - Printer for isDereferenceablePointer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct MemDerefPrinter : public FunctionPass {
    SmallVector<Value *, 4> Deref;
    SmallPtrSet<Value *, 4> DerefAndAligned;

    static char ID; // Pass identification, replacement for typeid
    MemDerefPrinter() : FunctionPass(ID) {
      initializeMemDerefPrinterPass(*PassRegistry::getPassRegistry());
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
    bool runOnFunction(Function &F) override;
    void print(raw_ostream &OS, const Module * = nullptr) const override;
    void releaseMemory() override {
      Deref.clear();
      DerefAndAligned.clear();
    }
  };
}

char MemDerefPrinter::ID = 0;
INITIALIZE_PASS_BEGIN(MemDerefPrinter, "print-memderefs",
                      "Memory Dereferenciblity of pointers in function", false, true)
INITIALIZE_PASS_END(MemDerefPrinter, "print-memderefs",
                    "Memory Dereferenciblity of pointers in function", false, true)

FunctionPass *llvm::createMemDerefPrinter() {
  return new MemDerefPrinter();
}

bool MemDerefPrinter::runOnFunction(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  for (auto &I: instructions(F)) {
    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      Value *PO = LI->getPointerOperand();
      if (isDereferenceablePointer(PO, LI->getType(), DL))
        Deref.push_back(PO);
      if (isDereferenceableAndAlignedPointer(
              PO, LI->getType(), MaybeAlign(LI->getAlignment()), DL))
        DerefAndAligned.insert(PO);
    }
  }
  return false;
}

void MemDerefPrinter::print(raw_ostream &OS, const Module *M) const {
  OS << "The following are dereferenceable:\n";
  for (Value *V: Deref) {
    V->print(OS);
    if (DerefAndAligned.count(V))
      OS << "\t(aligned)";
    else
      OS << "\t(unaligned)";
    OS << "\n\n";
  }
}
