//===- Transforms/Utils/ControlFlowUtils.h --------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to manipulate the CFG and restore SSA for the new control flow.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CONTROLFLOWUTILS_H
#define LLVM_TRANSFORMS_UTILS_CONTROLFLOWUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class BasicBlock;
class DomTreeUpdater;

/// Given a set of branch descriptors [BB, Succ0, Succ1], create a "hub" such
/// that the control flow from each BB to a successor is now split into two
/// edges, one from BB to the hub and another from the hub to the successor. The
/// hub consists of a series of guard blocks, one for each outgoing block. Each
/// guard block conditionally branches to the corresponding outgoing block, or
/// the next guard block in the chain. These guard blocks are returned in the
/// argument vector.
///
/// This also updates any PHINodes in the successor. For each such PHINode, the
/// operands corresponding to incoming blocks are moved to a new PHINode in the
/// hub, and the hub is made an operand of the original PHINode.
///
/// Note that for some block BB with a conditional branch, it is not necessary
/// that both successors are rerouted. The client specifies this by setting
/// either Succ0 or Succ1 to nullptr, in which case, the corresponding successor
/// is not rerouted.
///
/// Input CFG:
/// ----------
///
///                    Def
///                     |
///                     v
///           In1      In2
///            |        |
///            |        |
///            v        v
///  Foo ---> Out1     Out2
///                     |
///                     v
///                    Use
///
///
/// Create hub: Incoming = {In1, In2}, Outgoing = {Out1, Out2}
/// ----------------------------------------------------------
///
///             Def
///              |
///              v
///  In1        In2          Foo
///   |    Hub   |            |
///   |    + - - | - - +      |
///   |    '     v     '      V
///   +------> Guard1 -----> Out1
///        '     |     '
///        '     v     '
///        '   Guard2 -----> Out2
///        '           '      |
///        + - - - - - +      |
///                           v
///                          Use
///
/// Limitations:
/// -----------
/// 1. This assumes that all terminators in the CFG are direct branches (the
///    "br" instruction). The presence of any other control flow such as
///    indirectbr, switch or callbr will cause an assert.
///
/// 2. The updates to the PHINodes are not sufficient to restore SSA
///    form. Consider a definition Def, its use Use, incoming block In2 and
///    outgoing block Out2, such that:
///    a. In2 is reachable from D or contains D.
///    b. U is reachable from Out2 or is contained in Out2.
///    c. U is not a PHINode if U is contained in Out2.
///
///    Clearly, Def dominates Out2 since the program is valid SSA. But when the
///    hub is introduced, there is a new path through the hub along which Use is
///    reachable from entry without passing through Def, and SSA is no longer
///    valid. To fix this, we need to look at all the blocks post-dominated by
///    the hub on the one hand, and dominated by Out2 on the other. This is left
///    for the caller to accomplish, since each specific use of this function
///    may have additional information which simplifies this fixup. For example,
///    see restoreSSA() in the UnifyLoopExits pass.
struct ControlFlowHub {
  struct BranchDescriptor {
    BasicBlock *BB;
    BasicBlock *Succ0;
    BasicBlock *Succ1;

    BranchDescriptor(BasicBlock *BB, BasicBlock *Succ0, BasicBlock *Succ1)
        : BB(BB), Succ0(Succ0), Succ1(Succ1) {}
  };

  void addBranch(BasicBlock *BB, BasicBlock *Succ0, BasicBlock *Succ1) {
    assert(BB);
    assert(Succ0 || Succ1);
    Branches.emplace_back(BB, Succ0, Succ1);
  }

  BasicBlock *
  finalize(DomTreeUpdater *DTU, SmallVectorImpl<BasicBlock *> &GuardBlocks,
           const StringRef Prefix,
           std::optional<unsigned> MaxControlFlowBooleans = std::nullopt);

  SmallVector<BranchDescriptor> Branches;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CONTROLFLOWUTILS_H
