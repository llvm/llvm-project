//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "hello"

STATISTIC(HelloCounter, "Counts number of functions greeted");

namespace {
// Hello - The first implementation, without getAnalysisUsage.
struct Hello : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  Hello() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    ++HelloCounter;
    errs() << "Hello: ";
    errs().write_escaped(F.getName()) << '\n';
    return false;
  }
};
} // namespace

char Hello::ID = 0;
static RegisterPass<Hello> X("hello", "Hello World Pass");

namespace {
// Hello2 - The second implementation with getAnalysisUsage implemented.
struct Hello2 : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  Hello2() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    ++HelloCounter;
    errs() << "Hello: ";
    errs().write_escaped(F.getName()) << '\n';
    return false;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char Hello2::ID = 0;
static RegisterPass<Hello2>
    Y("hello2", "Hello World Pass (with getAnalysisUsage implemented)");

namespace {
// Hello - The first implementation, without getAnalysisUsage.
struct MyHello : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  MyHello() : FunctionPass(ID) {}
  int Count = 0;
  int CountBb = 0;
  llvm::DenseMap<llvm::StringRef, int> CountI;
  llvm::DenseMap<llvm::StringRef, int> CountSuc;
  llvm::DenseMap<llvm::StringRef, int> CountPred;

  bool runOnFunction(Function &F) override {

    // Count the number of instruction in a function
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      ++Count;
    }

    // count the no of BB in a Function
    for (BasicBlock &BB : F) {
      // Print out the name of the basic block if it has one, and then the
      // number of instructions that it contains
      // errs() << "Basic block (name=" << BB.getName() << ") has "
      //       << BB.size() << " instructions.\n";
      ++CountBb;
    }

    // Find the basic block with maximum instructions.
    for (BasicBlock &BB : F) {
      CountI[BB.getName()] = BB.size();
    }

    // for (llvm::DenseMap<llvm::StringRef, int>::iterator V = CountI.begin(),
    //                                               E = CountI.end();
    //      V != E; ++V) {
    //   errs() << V->first << " :" << V->second << "\n";
    // }

    int CurrentMax = 0;
    llvm::StringRef Maax;
    for (llvm::DenseMap<llvm::StringRef, int>::iterator V = CountI.begin(),
                                                        E = CountI.end();
         V != E; ++V) {
      if (V->second > CurrentMax) {
        Maax = V->first;
        CurrentMax = V->second;
      }
    }

    // Find the basic block with maximum successors.
    BasicBlock *Target = nullptr;
    int PredCount = 0;

    for (BasicBlock &BB : F) {
      Target = &BB;
      PredCount = 0;
      CountPred[Target->getName()] = PredCount;
      for (BasicBlock *Pred : predecessors(Target)) {
        PredCount++;
        CountPred[Target->getName()] = PredCount;
        // errs() << "Basic block name=" << Target->getName() << "\t"
        //        << Pred->getName() << "\n";
      }
    }

    // for (llvm::DenseMap<llvm::StringRef, int>::iterator V =
    // CountPred.begin(),
    //                                               E = CountPred.end();
    //      V != E; ++V) {
    //   errs() << V->first << " :" << V->second << "\n";
    // }

    int CurrentPredMax = 0;
    llvm::StringRef MaaxPred;
    for (llvm::DenseMap<llvm::StringRef, int>::iterator V = CountPred.begin(),
                                                        E = CountPred.end();
         V != E; ++V) {
      if (V->second > CurrentPredMax) {
        MaaxPred = V->first;
        CurrentPredMax = V->second;
      }
    }

    // Find the basic block with maximum successors.
    BasicBlock *TargetS = nullptr;
    int SuccCount = 0;

    for (BasicBlock &BB : F) {
      TargetS = &BB;
      SuccCount = 0;
      // CountSuc[TargetS->getName()] = SuccCount;
      for (BasicBlock *Succ : successors(TargetS)) {
        SuccCount++;
        CountSuc[TargetS->getName()] = SuccCount;
        // errs() << "Basic block name=" << TargetS->getName() << "\t"
        //        << Succ->getName() << "\n";
      }
    }

    // for (llvm::DenseMap<llvm::StringRef, int>::iterator V = CountSuc.begin(),
    //                                               E = CountSuc.end();
    //      V != E; ++V) {
    //   errs() << V->first << " :" << V->second << "\n";
    // }

    int CurrentSuccMax = 0;
    llvm::StringRef MaaxSucc;
    for (llvm::DenseMap<llvm::StringRef, int>::iterator V = CountSuc.begin(),
                                                        E = CountSuc.end();
         V != E; ++V) {
      if (V->second > CurrentSuccMax) {
        MaaxSucc = V->first;
        CurrentSuccMax = V->second;
      }
    }

    // Adding global variable using IRBuilder class and store zero to the new
    // global variable in entry block
    IRBuilder<> Builder((F.begin())->getFirstNonPHI());
    GlobalVariable *GV = new llvm::GlobalVariable(
        *F.getParent(), IntegerType::getInt32Ty((F.getContext())), false,
        llvm::GlobalValue::InternalLinkage, Builder.getInt32(0), "G");
    //Builder.CreateStore(Builder.getInt32(0), GV);

    // Store to different sequential numbers from 1 to all other blocks
    int Counter = 0;
    for (BasicBlock &BB : F) {
      Builder.SetInsertPoint(&*BB.begin());
      auto CountVal = APInt(32, Counter);
      auto *Var = Builder.getInt(CountVal);
      Builder.CreateStore(Var, GV);
      Counter++;
    }

    errs() << "Total no of instruction in a Function: " << Count << "\n";
    errs() << "Total no of BBs in a Function: " << CountBb << "\n";
    errs() << "BasicBlock with max instructions: " << Maax << "->" << CurrentMax
           << " instructions"
           << "\n";
    errs() << "BasicBlock with max predecessors: " << MaaxPred << "->"
           << CurrentPredMax << " predecessors "
           << "\n";
    errs() << "BasicBlock with max successors: " << MaaxSucc << "->"
           << CurrentSuccMax << " successors "
           << "\n";
    return false;
  }
};
} // namespace

char MyHello::ID = 0;
static RegisterPass<MyHello> Z("myhello", "Hello World Pass");