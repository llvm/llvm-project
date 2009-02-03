//===- RSProfiling.h - Various profiling using random sampling ----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// See notes in RSProfiling.cpp
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/RSProfiling.h"
#include <set>

namespace llvm {
  /// RSProfilers_std - a simple support class for profilers that handles most
  /// of the work of chaining and tracking inserted code.
  struct RSProfilers_std : public RSProfilers {
    static char ID;
    std::set<Value*> profcode;
    // Lookup up values in profcode
    virtual bool isProfiling(Value* v);
    // handles required chaining
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    // places counter updates in basic blocks and recordes added instructions in
    // profcode
    void IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                                 GlobalValue *CounterArray);
  };
}
