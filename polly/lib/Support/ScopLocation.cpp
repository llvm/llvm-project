//=== ScopLocation.cpp - Debug location for ScopDetection ----- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Helper function for extracting region debug information.
//
//===----------------------------------------------------------------------===//
//
#include "polly/Support/ScopLocation.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"

using namespace llvm;

namespace polly {

void getDebugLocation(const Region *R, unsigned &LineBegin, unsigned &LineEnd,
                      std::string &FileName) {
  LineBegin = -1;
  LineEnd = 0;

  for (const BasicBlock *BB : R->blocks())
    for (const Instruction &Inst : *BB) {
      DebugLoc DL = Inst.getDebugLoc();
      if (!DL)
        continue;

      auto *Scope = cast<DIScope>(DL.getScope());

      if (FileName.empty())
        FileName = Scope->getFilename();

      unsigned NewLine = DL.getLine();

      LineBegin = std::min(LineBegin, NewLine);
      LineEnd = std::max(LineEnd, NewLine);
    }
}
} // namespace polly
