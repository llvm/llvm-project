//===- PassPrinters.cpp - Utilities to print analysis info for passes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities to print analysis info for various kinds of passes.
///
//===----------------------------------------------------------------------===//

#include "PassPrinters.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

namespace {

struct FunctionPassPrinter : public FunctionPass {
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  static char ID;
  std::string PassName;

  FunctionPassPrinter(const PassInfo *PI, raw_ostream &out)
      : FunctionPass(ID), PassToPrint(PI), Out(out) {
    std::string PassToPrintName = std::string(PassToPrint->getPassName());
    PassName = "FunctionPass Printer: " + PassToPrintName;
  }

  bool runOnFunction(Function &F) override {
    Out << "Printing analysis '" << PassToPrint->getPassName()
        << "' for function '" << F.getName() << "':\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out, F.getParent());
    return false;
  }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char FunctionPassPrinter::ID = 0;

struct CallGraphSCCPassPrinter : public CallGraphSCCPass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  CallGraphSCCPassPrinter(const PassInfo *PI, raw_ostream &out)
      : CallGraphSCCPass(ID), PassToPrint(PI), Out(out) {
    std::string PassToPrintName = std::string(PassToPrint->getPassName());
    PassName = "CallGraphSCCPass Printer: " + PassToPrintName;
  }

  bool runOnSCC(CallGraphSCC &SCC) override {
    Out << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

    // Get and print pass...
    for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
      Function *F = (*I)->getFunction();
      if (F)
        getAnalysisID<Pass>(PassToPrint->getTypeInfo())
            .print(Out, F->getParent());
    }
    return false;
  }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char CallGraphSCCPassPrinter::ID = 0;

struct ModulePassPrinter : public ModulePass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  ModulePassPrinter(const PassInfo *PI, raw_ostream &out)
      : ModulePass(ID), PassToPrint(PI), Out(out) {
    std::string PassToPrintName = std::string(PassToPrint->getPassName());
    PassName = "ModulePass Printer: " + PassToPrintName;
  }

  bool runOnModule(Module &M) override {
    Out << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out, &M);
    return false;
  }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char ModulePassPrinter::ID = 0;

struct LoopPassPrinter : public LoopPass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  LoopPassPrinter(const PassInfo *PI, raw_ostream &out)
      : LoopPass(ID), PassToPrint(PI), Out(out) {
    std::string PassToPrintName = std::string(PassToPrint->getPassName());
    PassName = "LoopPass Printer: " + PassToPrintName;
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    Out << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo())
        .print(Out, L->getHeader()->getParent()->getParent());
    return false;
  }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char LoopPassPrinter::ID = 0;

struct RegionPassPrinter : public RegionPass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  RegionPassPrinter(const PassInfo *PI, raw_ostream &out)
      : RegionPass(ID), PassToPrint(PI), Out(out) {
    std::string PassToPrintName = std::string(PassToPrint->getPassName());
    PassName = "RegionPass Printer: " + PassToPrintName;
  }

  bool runOnRegion(Region *R, RGPassManager &RGM) override {
    Out << "Printing analysis '" << PassToPrint->getPassName() << "' for "
        << "region: '" << R->getNameStr() << "' in function '"
        << R->getEntry()->getParent()->getName() << "':\n";
    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo())
        .print(Out, R->getEntry()->getParent()->getParent());
    return false;
  }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char RegionPassPrinter::ID = 0;

} // end anonymous namespace

FunctionPass *llvm::createFunctionPassPrinter(const PassInfo *PI,
                                              raw_ostream &OS) {
  return new FunctionPassPrinter(PI, OS);
}

CallGraphSCCPass *llvm::createCallGraphPassPrinter(const PassInfo *PI,
                                                   raw_ostream &OS) {
  return new CallGraphSCCPassPrinter(PI, OS);
}

ModulePass *llvm::createModulePassPrinter(const PassInfo *PI, raw_ostream &OS) {
  return new ModulePassPrinter(PI, OS);
}

LoopPass *llvm::createLoopPassPrinter(const PassInfo *PI, raw_ostream &OS) {
  return new LoopPassPrinter(PI, OS);
}

RegionPass *llvm::createRegionPassPrinter(const PassInfo *PI, raw_ostream &OS) {
  return new RegionPassPrinter(PI, OS);
}
