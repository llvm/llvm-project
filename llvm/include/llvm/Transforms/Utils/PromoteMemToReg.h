//===- PromoteMemToReg.h - Promote Allocas to Scalars -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to promote alloca instructions to SSA
// registers, by using the SSA construction algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_UTILS_PROMOTEMEMTOREG_H
#define TRANSFORMS_UTILS_PROMOTEMEMTOREG_H

#include <vector>

namespace llvm {

class AllocaInst;
class DominatorTree;
class AliasSetTracker;

/// isAllocaPromotable - Return true if this alloca is legal for promotion.
/// This is true if there are only loads and stores to the alloca...
///
bool isAllocaPromotable(const AllocaInst *AI);

/// PromoteMemToReg - Promote the specified list of alloca instructions into
/// scalar registers, inserting PHI nodes as appropriate.  This function makes
/// use of DominanceFrontier information.  This function does not modify the CFG
/// of the function at all.  All allocas must be from the same function.
///
/// If AST is specified, the specified tracker is updated to reflect changes
/// made to the IR.
///
void PromoteMemToReg(const std::vector<AllocaInst*> &Allocas,
                     DominatorTree &DT, AliasSetTracker *AST = 0);

} // End llvm namespace

#endif
