//===------ Support/ScopHelper.h -- Some Helper Functions for Scop. --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with LLVM-IR.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_IRHELPER_H
#define POLLY_SUPPORT_IRHELPER_H

namespace llvm {
  class Instruction;
  class LoopInfo;
  class Loop;
  class ScalarEvolution;
  class SCEV;
  class Value;
  class PHINode;
  class Region;
  class Pass;
  class BasicBlock;
}

namespace polly {
  // Helper function for Scop.
  //===----------------------------------------------------------------------===//
  /// Temporary Hack for extended regiontree.
  ///
  /// @brief Cast the region to loop.
  ///
  /// @param R  The Region to be casted.
  /// @param LI The LoopInfo to help the casting.
  ///
  /// @return If there is a a loop that has the same entry and exit as the region,
  ///         return the loop, otherwise, return null.
  llvm::Loop *castToLoop(const llvm::Region &R, llvm::LoopInfo &LI);

  //===----------------------------------------------------------------------===//
  // Functions for checking affine functions.
  bool isIndVar(const llvm::SCEV *Var, llvm::Region &RefRegion,
                llvm::LoopInfo &LI, llvm::ScalarEvolution &SE);

  /// @brief Check if the instruction I is the induction variable of a loop.
  ///
  /// @param I The instruction to check.
  /// @param LI The LoopInfo analysis.
  ///
  /// @return Return true if I is the induction variable of a loop, false
  ///         otherwise.
  bool isIndVar(const llvm::Instruction *I, const llvm::LoopInfo *LI);

  /// @brief Check if the PHINode has any incoming Invoke edge.
  ///
  /// @param PN The PHINode to check.
  ///
  /// @return If the PHINode has an incoming BB that jumps to the parent BB
  ///         of the PHINode with an invoke instruction, return true,
  ///         otherwise, return false.
  bool hasInvokeEdge(const llvm::PHINode *PN);

  llvm::Value *getPointerOperand(llvm::Instruction &Inst);

  // Helper function for LLVM-IR about Scop.
  llvm::BasicBlock *createSingleExitEdge(llvm::Region *R, llvm::Pass *P);

  /// @brief Split the entry block of a function to store the newly inserted
  ///        allocations outside of all Scops.
  ///
  /// @param EntryBlock The entry block of the current function.
  /// @param P          The pass that currently running.
  ///
  void splitEntryBlockForAlloca(llvm::BasicBlock *EntryBlock, llvm::Pass *P);
}
#endif
