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

  static bool originalTypeIsF128(const Type *Ty);
  static bool originalEVTTypeIsVectorFloat(EVT Ty);
  static bool originalTypeIsVectorFloat(const Type *Ty);

  void PreAnalyzeCallOperand(const Type *ArgTy);

  void PreAnalyzeFormalArgument(const Type *ArgTy, ISD::ArgFlagsTy Flags);
  void PreAnalyzeReturnValue(EVT ArgVT);

private:
  /// Identify lowered values that originated from f128 arguments and record
  /// this for use by RetCC_MipsN.
  void PreAnalyzeCallResultForF128(const SmallVectorImpl<ISD::InputArg> &Ins,
                                   const Type *RetTy);

  /// Identify lowered values that originated from f128 arguments and record
  /// this for use by RetCC_MipsN.
  void PreAnalyzeCallReturnForF128(const SmallVectorImpl<ISD::OutputArg> &Outs, const Type *RetTy);

  /// Identify lowered values that originated from f128 arguments and record
  /// this.
  void
  PreAnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                         std::vector<TargetLowering::ArgListEntry> &FuncArgs);

  /// Identify lowered values that originated from f128 arguments and record
  /// this for use by RetCC_MipsN.
  void
  PreAnalyzeFormalArgumentsForF128(const SmallVectorImpl<ISD::InputArg> &Ins);

  void
  PreAnalyzeCallResultForVectorFloat(const SmallVectorImpl<ISD::InputArg> &Ins,
                                     const Type *RetTy);

  void PreAnalyzeFormalArgumentsForVectorFloat(
      const SmallVectorImpl<ISD::InputArg> &Ins);

  void
  PreAnalyzeReturnForVectorFloat(const SmallVectorImpl<ISD::OutputArg> &Outs);

  /// Records whether the value has been lowered from an f128.
  SmallVector<bool, 4> OriginalArgWasF128;

  /// Records whether the value has been lowered from float.
  SmallVector<bool, 4> OriginalArgWasFloat;

  /// Records whether the value has been lowered from a floating point vector.
  SmallVector<bool, 4> OriginalArgWasFloatVector;

  /// Records whether the return value has been lowered from a floating point
  /// vector.
  SmallVector<bool, 4> OriginalRetWasFloatVector;

  // Used to handle MIPS16-specific calling convention tweaks.
  // FIXME: This should probably be a fully fledged calling convention.
  SpecialCallingConvType SpecialCallingConv;

public:
  MipsCCState(CallingConv::ID CC, bool isVarArg, MachineFunction &MF,
              SmallVectorImpl<CCValAssign> &locs, LLVMContext &C,
              SpecialCallingConvType SpecialCC = NoSpecialCallingConv)
      : CCState(CC, isVarArg, MF, locs, C), SpecialCallingConv(SpecialCC) {}

  void
  PreAnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                         CCAssignFn Fn,
                         std::vector<TargetLowering::ArgListEntry> &FuncArgs) {
    OriginalArgWasF128.clear();
    OriginalArgWasFloat.clear();
    OriginalArgWasFloatVector.clear();
    PreAnalyzeCallOperands(Outs, FuncArgs);
  }

  void
  AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                      CCAssignFn Fn,
                      std::vector<TargetLowering::ArgListEntry> &FuncArgs) {
    PreAnalyzeCallOperands(Outs, Fn, FuncArgs);
    CCState::AnalyzeCallOperands(Outs, Fn);
  }

  // The AnalyzeCallOperands in the base class is not usable since we must
  // provide a means of accessing ArgListEntry::IsFixed. Delete them from this
  // class. This doesn't stop them being used via the base class though.
  void AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                           CCAssignFn Fn) = delete;
  void AnalyzeCallOperands(const SmallVectorImpl<MVT> &Outs,
                           SmallVectorImpl<ISD::ArgFlagsTy> &Flags,
                           CCAssignFn Fn) = delete;

  void PreAnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                                 CCAssignFn Fn) {
    OriginalArgWasFloat.clear();
    OriginalArgWasF128.clear();
    OriginalArgWasFloatVector.clear();
    PreAnalyzeFormalArgumentsForF128(Ins);
  }

  void AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                              CCAssignFn Fn) {
    PreAnalyzeFormalArguments(Ins, Fn);
    CCState::AnalyzeFormalArguments(Ins, Fn);
  }

  void PreAnalyzeCallResult(const Type *RetTy) {
    OriginalArgWasF128.push_back(originalTypeIsF128(RetTy));
    OriginalArgWasFloat.push_back(RetTy->isFloatingPointTy());
    OriginalRetWasFloatVector.push_back(originalTypeIsVectorFloat(RetTy));
  }

  void PreAnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                            CCAssignFn Fn, const Type *RetTy) {
    OriginalArgWasFloat.clear();
    OriginalArgWasF128.clear();
    OriginalArgWasFloatVector.clear();
    PreAnalyzeCallResultForF128(Ins, RetTy);
    PreAnalyzeCallResultForVectorFloat(Ins, RetTy);
  }

  void AnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                         CCAssignFn Fn, const Type *RetTy) {
    PreAnalyzeCallResult(Ins, Fn, RetTy);
    CCState::AnalyzeCallResult(Ins, Fn);
  }

  void PreAnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                        CCAssignFn Fn) {
    const MachineFunction &MF = getMachineFunction();
    OriginalArgWasFloat.clear();
    OriginalArgWasF128.clear();
    OriginalArgWasFloatVector.clear();
    PreAnalyzeCallReturnForF128(Outs, MF.getFunction().getReturnType());
    PreAnalyzeReturnForVectorFloat(Outs);
  }

  void AnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                     CCAssignFn Fn) {
    PreAnalyzeReturn(Outs, Fn);
    CCState::AnalyzeReturn(Outs, Fn);
  }

  bool CheckReturn(const SmallVectorImpl<ISD::OutputArg> &ArgsFlags,
                   CCAssignFn Fn) {
    const MachineFunction &MF = getMachineFunction();
    PreAnalyzeCallReturnForF128(ArgsFlags, MF.getFunction().getReturnType());
    PreAnalyzeReturnForVectorFloat(ArgsFlags);
    bool Return = CCState::CheckReturn(ArgsFlags, Fn);
    OriginalArgWasFloat.clear();
    OriginalArgWasF128.clear();
    OriginalArgWasFloatVector.clear();
    return Return;
  }

  bool CheckCallReturn(const SmallVectorImpl<ISD::OutputArg> &ArgsFlags,
                   CCAssignFn Fn, const Type *RetTy) {
    PreAnalyzeCallReturnForF128(ArgsFlags, RetTy);
    PreAnalyzeReturnForVectorFloat(ArgsFlags);
    bool Return = CCState::CheckReturn(ArgsFlags, Fn);
    OriginalArgWasFloat.clear();
    OriginalArgWasF128.clear();
    OriginalArgWasFloatVector.clear();
    return Return;
  }
  bool WasOriginalArgF128(unsigned ValNo) { return OriginalArgWasF128[ValNo]; }
  bool WasOriginalArgFloat(unsigned ValNo) {
      return OriginalArgWasFloat[ValNo];
  }
  bool WasOriginalArgVectorFloat(unsigned ValNo) const {
    return OriginalArgWasFloatVector[ValNo];
  }
  bool WasOriginalRetVectorFloat(unsigned ValNo) const {
    return OriginalRetWasFloatVector[ValNo];
  }
  SpecialCallingConvType getSpecialCallingConv() { return SpecialCallingConv; }
};
}

#endif
