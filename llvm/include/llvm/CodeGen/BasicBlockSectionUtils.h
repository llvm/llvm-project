//===- BasicBlockSectionUtils.h - Utilities for basic block sections     --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BASICBLOCKSECTIONUTILS_H
#define LLVM_CODEGEN_BASICBLOCKSECTIONUTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

extern cl::opt<std::string> BBSectionsColdTextPrefix;

class MachineFunction;
class MachineBasicBlock;

using MachineBasicBlockComparator =
    function_ref<bool(const MachineBasicBlock &, const MachineBasicBlock &)>;

void sortBasicBlocksAndUpdateBranches(MachineFunction &MF,
                                      MachineBasicBlockComparator MBBCmp);

void avoidZeroOffsetLandingPad(MachineFunction &MF);

} // end namespace llvm

#endif // LLVM_CODEGEN_BASICBLOCKSECTIONUTILS_H
