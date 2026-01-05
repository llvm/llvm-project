//===---- MipsCCState.h - CCState with Mips specific extensions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIPSCCSTATE_H
#define MIPSCCSTATE_H

#include "MipsISelLowering.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {
class SDNode;
class MipsSubtarget;

class MipsCCState : public CCState {
public:
  enum SpecialCallingConvType { Mips16RetHelperConv, NoSpecialCallingConv };

  /// Determine the SpecialCallingConvType for the given callee
  static SpecialCallingConvType
  getSpecialCallingConvForCallee(const SDNode *Callee,
                                 const MipsSubtarget &Subtarget);

private:
  // Used to handle MIPS16-specific calling convention tweaks.
  // FIXME: This should probably be a fully fledged calling convention.
  SpecialCallingConvType SpecialCallingConv;

public:
  MipsCCState(CallingConv::ID CC, bool isVarArg, MachineFunction &MF,
              SmallVectorImpl<CCValAssign> &locs, LLVMContext &C,
              SpecialCallingConvType SpecialCC = NoSpecialCallingConv)
      : CCState(CC, isVarArg, MF, locs, C), SpecialCallingConv(SpecialCC) {}

  SpecialCallingConvType getSpecialCallingConv() { return SpecialCallingConv; }
};
}

#endif
