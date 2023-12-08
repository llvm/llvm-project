//===- Utils.h - Utility functions for code generation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the code generation.
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGEN_UTILS_H
#define POLLY_CODEGEN_UTILS_H

#include <utility>

namespace llvm {
class Pass;
class Value;
class BasicBlock;
class DominatorTree;
class RegionInfo;
class LoopInfo;
class BranchInst;
} // namespace llvm

namespace polly {

class Scop;

using BBPair = std::pair<llvm::BasicBlock *, llvm::BasicBlock *>;
/// Execute a Scop conditionally wrt @p RTC.
///
/// In the CFG the optimized code of the Scop is generated next to the
/// original code. Both the new and the original version of the code remain
/// in the CFG. A branch statement decides which version is executed based on
/// the runtime value of @p RTC.
///
/// Before transformation:
///
///                        bb0
///                         |
///                     orig_scop
///                         |
///                        bb1
///
/// After transformation:
///                        bb0
///                         |
///                  polly.splitBlock
///                     /       \.
///                     |     startBlock
///                     |        |
///               orig_scop   new_scop
///                     \      /
///                      \    /
///                        bb1 (joinBlock)
///
/// @param S   The Scop to execute conditionally.
/// @param P   A reference to the pass calling this function.
/// @param RTC The runtime condition checked before executing the new SCoP.
///
/// @return  An std::pair:
///              - The first element is a BBPair of (StartBlock, EndBlock).
///              - The second element is the BranchInst which conditionally
///                branches to the SCoP based on the RTC.
///
std::pair<BBPair, llvm::BranchInst *>
executeScopConditionally(Scop &S, llvm::Value *RTC, llvm::DominatorTree &DT,
                         llvm::RegionInfo &RI, llvm::LoopInfo &LI);
} // namespace polly
#endif
