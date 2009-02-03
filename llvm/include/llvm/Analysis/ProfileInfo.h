//===- llvm/Analysis/ProfileInfo.h - Profile Info Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the generic ProfileInfo interface, which is used as the
// common interface used by all clients of profiling information, and
// implemented either by making static guestimations, or by actually reading in
// profiling information gathered by running the program.
//
// Note that to be useful, all profile-based optimizations should preserve
// ProfileInfo, which requires that they notify it when changes to the CFG are
// made.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILEINFO_H
#define LLVM_ANALYSIS_PROFILEINFO_H

#include <string>
#include <map>

namespace llvm {
  class BasicBlock;
  class Pass;

  /// ProfileInfo Class - This class holds and maintains edge profiling
  /// information for some unit of code.
  class ProfileInfo {
  protected:
    // EdgeCounts - Count the number of times a transition between two blocks is
    // executed.  As a special case, we also hold an edge from the null
    // BasicBlock to the entry block to indicate how many times the function was
    // entered.
    std::map<std::pair<BasicBlock*, BasicBlock*>, unsigned> EdgeCounts;
  public:
    static char ID; // Class identification, replacement for typeinfo
    virtual ~ProfileInfo();  // We want to be subclassed

    //===------------------------------------------------------------------===//
    /// Profile Information Queries
    ///
    unsigned getExecutionCount(BasicBlock *BB) const;

    unsigned getEdgeWeight(BasicBlock *Src, BasicBlock *Dest) const {
      std::map<std::pair<BasicBlock*, BasicBlock*>, unsigned>::const_iterator I=
        EdgeCounts.find(std::make_pair(Src, Dest));
      return I != EdgeCounts.end() ? I->second : 0;
    }

    //===------------------------------------------------------------------===//
    /// Analysis Update Methods
    ///

  };

  /// createProfileLoaderPass - This function returns a Pass that loads the
  /// profiling information for the module from the specified filename, making
  /// it available to the optimizers.
  Pass *createProfileLoaderPass(const std::string &Filename);
} // End llvm namespace

#endif
