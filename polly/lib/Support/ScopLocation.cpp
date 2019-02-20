//=== ScopLocation.cpp - Debug location for ScopDetection ----- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
