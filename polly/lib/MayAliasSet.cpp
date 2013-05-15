//===---------- MayAliasSet.cpp  - May-Alais Set for base pointers --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MayAliasSet class
//
//===----------------------------------------------------------------------===//

#include "polly/TempScopInfo.h"
#include "polly/MayAliasSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace polly;

void MayAliasSet::print(raw_ostream &OS) const {
  OS << "Must alias {";

  for (const_iterator I = mustalias_begin(), E = mustalias_end(); I != E; ++I) {
    WriteAsOperand(OS, *I, false);
    OS << ", ";
  }

  OS << "} May alias {";
  OS << '}';
}

void MayAliasSet::dump() const { print(dbgs()); }

void MayAliasSetInfo::buildMayAliasSets(TempScop &Scop, AliasAnalysis &AA) {}
