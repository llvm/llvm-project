//===---- PPCCCState.h - CCState with PowerPC specific extensions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PPCCCSTATE_H
#define PPCCCSTATE_H

#include "PPCISelLowering.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {

class PPCCCState : public CCState {
public:

  void
  PreAnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs);
  void
  PreAnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins);

private:

  // Records whether the value has been lowered from an ppcf128.
  SmallVector<bool, 4> OriginalArgWasPPCF128;

public:
  PPCCCState(CallingConv::ID CC, bool isVarArg, MachineFunction &MF,
             SmallVectorImpl<CCValAssign> &locs, LLVMContext &C)
        : CCState(CC, isVarArg, MF, locs, C) {}

  bool WasOriginalArgPPCF128(unsigned ValNo) { return OriginalArgWasPPCF128[ValNo]; }
  void clearWasPPCF128() { OriginalArgWasPPCF128.clear(); }
};

} // end namespace llvm

#endif
