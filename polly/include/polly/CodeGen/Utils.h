//===- Utils.h - Utility functions for code generation ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the code generation.
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGEN_UTILS_H
#define POLLY_CODEGEN_UTILS_H

namespace llvm {
class Pass;
class BasicBlock;
}

namespace polly {

class Scop;

/// @brief Execute a Scop conditionally.
///
/// In the CFG the optimized code of the Scop is generated next to the
/// original code. Both the new and the original version of the code remain
/// in the CFG. A branch statement decides which version is executed.
/// For now, we always execute the new version (the old one is dead code
/// eliminated by the cleanup passes). In the future we may decide to execute
/// the new version only if certain run time checks succeed. This will be
/// useful to support constructs for which we cannot prove all assumptions at
/// compile time.
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
/// @param S The Scop to execute conditionally.
/// @param PassInfo A reference to the pass calling this function.
/// @return BasicBlock The 'StartBlock' to which new code can be added.
llvm::BasicBlock *executeScopConditionally(Scop &S, llvm::Pass *PassInfo);

}
#endif
