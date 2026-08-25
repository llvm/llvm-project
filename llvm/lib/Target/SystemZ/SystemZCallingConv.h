//===-- SystemZCallingConv.h - Calling conventions for SystemZ --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZCALLINGCONV_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZCALLINGCONV_H

#include "SystemZSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
namespace SystemZ {
  const unsigned ELFNumArgGPRs = 5;
  extern const MCPhysReg ELFArgGPRs[ELFNumArgGPRs];

  const unsigned ELFNumArgFPRs = 4;
  extern const MCPhysReg ELFArgFPRs[ELFNumArgFPRs];

  const unsigned XPLINK64NumArgGPRs = 3;
  extern const MCPhysReg XPLINK64ArgGPRs[XPLINK64NumArgGPRs];

  const unsigned XPLINK64NumArgFPRs = 4;
  extern const MCPhysReg XPLINK64ArgFPRs[XPLINK64NumArgFPRs];
} // end namespace SystemZ

// Handle i128 argument types.  These need to be passed by implicit
// reference.  This could be as simple as the following .td line:
//    CCIfType<[i128], CCPassIndirect<i64>>,
// except that i128 is not a legal type, and therefore gets split by
// common code into a pair of i64 arguments.
inline bool CC_SystemZ_I128Indirect(unsigned &ValNo, MVT &ValVT,
                                    MVT &LocVT,
                                    CCValAssign::LocInfo &LocInfo,
                                    ISD::ArgFlagsTy &ArgFlags,
                                    CCState &State) {
  SmallVectorImpl<CCValAssign> &PendingMembers = State.getPendingLocs();

  // ArgFlags.isSplit() is true on the first part of a i128 argument;
  // PendingMembers.empty() is false on all subsequent parts.
  if (!ArgFlags.isSplit() && PendingMembers.empty())
    return false;

  // Push a pending Indirect value location for each part.
  LocVT = MVT::i64;
  LocInfo = CCValAssign::Indirect;
  PendingMembers.push_back(CCValAssign::getPending(ValNo, ValVT,
                                                   LocVT, LocInfo));
  if (!ArgFlags.isSplitEnd())
    return true;

  // OK, we've collected all parts in the pending list.  Allocate
  // the location (register or stack slot) for the indirect pointer.
  // (This duplicates the usual i64 calling convention rules.)
  unsigned Reg;
  const SystemZSubtarget &Subtarget =
      State.getMachineFunction().getSubtarget<SystemZSubtarget>();
  if (Subtarget.isTargetELF())
    Reg = State.AllocateReg(SystemZ::ELFArgGPRs);
  else if (Subtarget.isTargetXPLINK64())
    Reg = State.AllocateReg(SystemZ::XPLINK64ArgGPRs);
  else
    llvm_unreachable("Unknown Calling Convention!");

  unsigned Offset = Reg && !Subtarget.isTargetXPLINK64()
                        ? 0
                        : State.AllocateStack(8, Align(8));

  // Use that same location for all the pending parts.
  for (auto &It : PendingMembers) {
    if (Reg)
      It.convertToReg(Reg);
    else
      It.convertToMem(Offset);
    State.addLoc(It);
  }

  PendingMembers.clear();

  return true;
}

// Extends CCState with:
//   - a flag to distinguish formal-argument lowering from outgoing-call
//     lowering (used by CC_XPLINK_Promote_i32)
//   - the pre-legalization original MVT for each argument (needed because
//     i8/i16 are both legalized to i32 before the CC table runs, so ValVT
//     alone cannot distinguish them)
class SystemZCCState : public CCState {
  bool IsFormalArgLowering = false;
  SmallVector<MVT, 8> ArgOrigVTs;

public:
  using CCState::CCState;

  void AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                              CCAssignFn Fn) {
    // Record the pre-legalization original type for each argument.
    // Use MVT::Other as a sentinel when ArgVT is not a simple type (e.g.
    // split or extended EVTs that occur in 32-bit mode).
    ArgOrigVTs.clear();
    for (const auto &In : Ins)
      ArgOrigVTs.push_back(In.ArgVT.isSimple() ? In.ArgVT.getSimpleVT()
                                               : MVT::Other);
    CCState::AnalyzeFormalArguments(Ins, Fn);
  }

  // Returns the pre-legalization MVT for argument ValNo.
  // Only valid after AnalyzeFormalArguments; returns MVT::Other if out of
  // range (e.g. called from AnalyzeCallOperands context where ArgOrigVTs
  // was never populated).
  MVT getArgOrigVT(unsigned ValNo) const {
    if (ValNo >= ArgOrigVTs.size())
      return MVT::Other;
    return ArgOrigVTs[ValNo];
  }

  bool isFormalArgLowering() const { return IsFormalArgLowering; }
  void setIsFormalArgLowering() { IsFormalArgLowering = true; }
};

// A pointer in 64bit mode is always passed as 64bit.
inline bool CC_XPLINK64_Pointer(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                CCValAssign::LocInfo &LocInfo,
                                ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  if (LocVT != MVT::i64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::ZExt;
  }
  return false;
}

inline bool CC_XPLINK_Promote_i32(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                  CCValAssign::LocInfo &LocInfo,
                                  ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  SystemZCCState *SZState = static_cast<SystemZCCState *>(&State);
  if (SZState->isFormalArgLowering()) {
    // For formal arguments, use the pre-legalization original MVT to
    // distinguish i8/i16 from i32 — all three are legalized to i32 (ValVT)
    // before the CC table runs, so ValVT alone cannot tell them apart.
    MVT OrigVT = SZState->getArgOrigVT(ValNo);
    if (OrigVT == MVT::i8 || OrigVT == MVT::i16) {
      // Keep LocVT=i32 (GR32 live-in) only if a GR32 register is still free.
      // The CC table assigns i32 args to R1L/R2L/R3L in XPLINK64; if all
      // three are taken, this arg goes to an 8-byte stack slot.  The caller
      // stores a sign-extended i64 in that slot (value in bytes 6-7), so we
      // must use LocVT=i64 to load all 8 bytes correctly.
      // Keep LocVT=i32 (GR32 live-in) only if a GR32 register is still free.
      // XPLINK64 assigns i32 to R1L/R2L/R3L; if all three are taken this arg
      // goes to an 8-byte stack slot where the value lives in bytes 6-7 (the
      // caller stores a sign-extended i64).  In that case we must use
      // LocVT=i64 so the 8-byte slot is loaded correctly.
      static const MCPhysReg GPR32s[] = {SystemZ::R1L, SystemZ::R2L,
                                         SystemZ::R3L};
      if (State.getFirstUnallocated(GPR32s) < 3) {
        LocInfo = CCValAssign::AExt;
        return false;
      }
      // All GR32s taken — stack path: promote to i64.
    }
    // i32 formal (or stack-bound i8/i16): promote to GR64 via AExt.
    LocVT = MVT::i64;
    LocInfo = CCValAssign::AExt;
  } else {
    LocVT = MVT::i64;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }
  return false;
}

inline bool CC_XPLINK64_Shadow_Reg(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  if (LocVT == MVT::f32 || LocVT == MVT::f64) {
    State.AllocateReg(SystemZ::XPLINK64ArgGPRs);
  }
  if (LocVT == MVT::f128 || LocVT.is128BitVector()) {
    // Shadow next two GPRs, if available.
    State.AllocateReg(SystemZ::XPLINK64ArgGPRs);
    State.AllocateReg(SystemZ::XPLINK64ArgGPRs);

    // Quad precision floating point needs to
    // go inside pre-defined FPR pair.
    if (LocVT == MVT::f128) {
      for (unsigned I = 0; I < SystemZ::XPLINK64NumArgFPRs; I += 2)
        if (State.isAllocated(SystemZ::XPLINK64ArgFPRs[I]))
          State.AllocateReg(SystemZ::XPLINK64ArgFPRs[I + 1]);
    }
  }
  return false;
}

inline bool CC_XPLINK64_Allocate128BitVararg(unsigned &ValNo, MVT &ValVT,
                                             MVT &LocVT,
                                             CCValAssign::LocInfo &LocInfo,
                                             ISD::ArgFlagsTy &ArgFlags,
                                             CCState &State) {
  // For any C or C++ program, this should always be
  // false, since it is illegal to have a function
  // where the first argument is variadic. Therefore
  // the first fixed argument should already have
  // allocated GPR1 either through shadowing it or
  // using it for parameter passing.
  State.AllocateReg(SystemZ::R1D);

  bool AllocGPR2 = State.AllocateReg(SystemZ::R2D);
  bool AllocGPR3 = State.AllocateReg(SystemZ::R3D);

  // If GPR2 and GPR3 are available, then we may pass vararg in R2Q.
  // If only GPR3 is available, we need to set custom handling to copy
  // hi bits into GPR3.
  // Either way, we allocate on the stack.
  if (AllocGPR3) {
    // For f128 and vector var arg case, set the bitcast flag to bitcast to
    // i128.
    LocVT = MVT::i128;
    LocInfo = CCValAssign::BCvt;
    auto Offset = State.AllocateStack(16, Align(8));
    if (AllocGPR2)
      State.addLoc(
          CCValAssign::getReg(ValNo, ValVT, SystemZ::R2Q, LocVT, LocInfo));
    else
      State.addLoc(
          CCValAssign::getCustomMem(ValNo, ValVT, Offset, LocVT, LocInfo));
    return true;
  }

  return false;
}

inline bool RetCC_SystemZ_Error(unsigned &, MVT &, MVT &,
                                CCValAssign::LocInfo &, ISD::ArgFlagsTy &,
                                CCState &) {
  llvm_unreachable("Return value calling convention currently unsupported.");
}

inline bool CC_SystemZ_Error(unsigned &, MVT &, MVT &, CCValAssign::LocInfo &,
                             ISD::ArgFlagsTy &, CCState &) {
  llvm_unreachable("Argument calling convention currently unsupported.");
}

inline bool CC_SystemZ_GHC_Error(unsigned &, MVT &, MVT &,
                                 CCValAssign::LocInfo &, ISD::ArgFlagsTy &,
                                 CCState &) {
  report_fatal_error("No registers left in GHC calling convention");
  return false;
}

} // end namespace llvm

#endif
