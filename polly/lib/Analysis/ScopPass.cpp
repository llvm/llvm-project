//===- ScopPass.cpp - The base class of Passes that operate on Polly IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the ScopPass members.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopPass.h"
#include "polly/ScopInfo.h"

using namespace llvm;
using namespace polly;

bool ScopPass::runOnRegion(Region *R, RGPassManager &RGM) {
  S = 0;

  if ((S = getAnalysis<ScopInfo>().getScop()))
    return runOnScop(*S);

  return false;
}

void ScopPass::print(raw_ostream &OS, const Module *M) const {
  if (S)
    printScop(OS);
}

void ScopPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ScopInfo>();
  AU.setPreservesAll();
}
