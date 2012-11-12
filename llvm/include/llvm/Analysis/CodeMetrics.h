//===- CodeMetrics.h - Code cost measurements -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements various weight measurements for code, helping
// the Inliner and other passes decide whether to duplicate its contents.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CODEMETRICS_H
#define LLVM_ANALYSIS_CODEMETRICS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CallSite.h"

namespace llvm {
  class BasicBlock;
  class Function;
  class Instruction;
  class DataLayout;
  class Value;

  /// \brief Check whether an instruction is likely to be "free" when lowered.
  bool isInstructionFree(const Instruction *I, const DataLayout *TD = 0);

  /// \brief Check whether a call will lower to something small.
  ///
  /// This tests checks whether this callsite will lower to something
  /// significantly cheaper than a traditional call, often a single
  /// instruction. Note that if isInstructionFree(CS.getInstruction()) would
  /// return true, so will this function.
  bool callIsSmall(ImmutableCallSite CS);

  /// \brief Utility to calculate the size and a few similar metrics for a set
  /// of basic blocks.
  struct CodeMetrics {
    /// \brief True if this function contains a call to setjmp or other functions
    /// with attribute "returns twice" without having the attribute itself.
    bool exposesReturnsTwice;

    /// \brief True if this function calls itself.
    bool isRecursive;

    /// \brief True if this function contains one or more indirect branches.
    bool containsIndirectBr;

    /// \brief True if this function calls alloca (in the C sense).
    bool usesDynamicAlloca;

    /// \brief Number of instructions in the analyzed blocks.
    unsigned NumInsts;

    /// \brief Number of analyzed blocks.
    unsigned NumBlocks;

    /// \brief Keeps track of basic block code size estimates.
    DenseMap<const BasicBlock *, unsigned> NumBBInsts;

    /// \brief Keep track of the number of calls to 'big' functions.
    unsigned NumCalls;

    /// \brief The number of calls to internal functions with a single caller.
    ///
    /// These are likely targets for future inlining, likely exposed by
    /// interleaved devirtualization.
    unsigned NumInlineCandidates;

    /// \brief How many instructions produce vector values.
    ///
    /// The inliner is more aggressive with inlining vector kernels.
    unsigned NumVectorInsts;

    /// \brief How many 'ret' instructions the blocks contain.
    unsigned NumRets;

    CodeMetrics() : exposesReturnsTwice(false), isRecursive(false),
                    containsIndirectBr(false), usesDynamicAlloca(false),
                    NumInsts(0), NumBlocks(0), NumCalls(0),
                    NumInlineCandidates(0), NumVectorInsts(0),
                    NumRets(0) {}

    /// \brief Add information about a block to the current state.
    void analyzeBasicBlock(const BasicBlock *BB, const DataLayout *TD = 0);

    /// \brief Add information about a function to the current state.
    void analyzeFunction(Function *F, const DataLayout *TD = 0);
  };
}

#endif
