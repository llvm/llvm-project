//===- StableBasicBlockNumbering.h - Provide BB identifiers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class provides a *stable* numbering of basic blocks that does not depend
// on their address in memory (which is nondeterministic).  When requested, this
// class simply provides a unique ID for each basic block in the function
// specified and the inverse mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STABLEBASICBLOCKNUMBERING_H
#define LLVM_SUPPORT_STABLEBASICBLOCKNUMBERING_H

#include "llvm/Function.h"
#include "llvm/ADT/UniqueVector.h"

namespace llvm {
  class StableBasicBlockNumbering {
    // BBNumbering - Holds the numbering.
    UniqueVector<BasicBlock*> BBNumbering;
  public:
    StableBasicBlockNumbering(Function *F = 0) {
      if (F) compute(*F);
    }

    /// compute - If we have not computed a numbering for the function yet, do
    /// so.
    void compute(Function &F) {
      if (BBNumbering.empty()) {
        for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
          BBNumbering.insert(I);
      }
    }

    /// getNumber - Return the ID number for the specified BasicBlock.
    ///
    unsigned getNumber(BasicBlock *BB) const {
      unsigned Idx = BBNumbering.idFor(BB);
      assert(Idx && "Invalid basic block or numbering not computed!");
      return Idx-1;
    }

    /// getBlock - Return the BasicBlock corresponding to a particular ID.
    ///
    BasicBlock *getBlock(unsigned N) const {
      assert(N < BBNumbering.size() &&
             "Block ID out of range or numbering not computed!");
      return BBNumbering[N+1];
    }
  };
}

#endif
