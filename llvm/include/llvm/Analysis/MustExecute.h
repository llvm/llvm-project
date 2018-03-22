//===- MustExecute.h - Is an instruction known to execute--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// Contains a collection of routines for determining if a given instruction is
/// guaranteed to execute if a given point in control flow is reached.  The most
/// common example is an instruction within a loop being provably executed if we
/// branch to the header of it's containing loop.  
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MUSTEXECUTE_H
#define LLVM_ANALYSIS_MUSTEXECUTE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

class Instruction;
class DominatorTree;
class Loop;

/// \brief Captures loop safety information.
/// It keep information for loop & its header may throw exception or otherwise
/// exit abnormaly on any iteration of the loop which might actually execute
/// at runtime.  The primary way to consume this infromation is via
/// isGuaranteedToExecute below, but some callers bailout or fallback to
/// alternate reasoning if a loop contains any implicit control flow.
struct LoopSafetyInfo {
  bool MayThrow = false;       // The current loop contains an instruction which
                               // may throw.
  bool HeaderMayThrow = false; // Same as previous, but specific to loop header
  // Used to update funclet bundle operands.
  DenseMap<BasicBlock *, ColorVector> BlockColors;

  LoopSafetyInfo() = default;
};

/// \brief Computes safety information for a loop checks loop body & header for
/// the possibility of may throw exception, it takes LoopSafetyInfo and loop as
/// argument. Updates safety information in LoopSafetyInfo argument.
/// Note: This is defined to clear and reinitialize an already initialized
/// LoopSafetyInfo.  Some callers rely on this fact.
void computeLoopSafetyInfo(LoopSafetyInfo *, Loop *);

/// Returns true if the instruction in a loop is guaranteed to execute at least
/// once (under the assumption that the loop is entered).
bool isGuaranteedToExecute(const Instruction &Inst, const DominatorTree *DT,
                           const Loop *CurLoop,
                           const LoopSafetyInfo *SafetyInfo);
  
}

#endif
