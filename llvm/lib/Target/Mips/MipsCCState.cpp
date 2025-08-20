//===---- MipsCCState.cpp - CCState with Mips specific extensions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsCCState.h"
#include "MipsSubtarget.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// This function returns true if Ty is fp128, {f128} or i128 which was
/// originally a fp128.
bool MipsCCState::originalTypeIsF128(const Type *Ty) {
  if (Ty->isFP128Ty())
    return true;

  if (Ty->isStructTy() && Ty->getStructNumElements() == 1 &&
      Ty->getStructElementType(0)->isFP128Ty())
    return true;

  return false;
}

/// Return true if the original type was vXfXX.
bool MipsCCState::originalEVTTypeIsVectorFloat(EVT Ty) {
  if (Ty.isVector() && Ty.getVectorElementType().isFloatingPoint())
    return true;

  return false;
}

/// Return true if the original type was vXfXX / vXfXX.
bool MipsCCState::originalTypeIsVectorFloat(const Type *Ty) {
  if (Ty->isVectorTy() && Ty->isFPOrFPVectorTy())
    return true;

  return false;
}

MipsCCState::SpecialCallingConvType
MipsCCState::getSpecialCallingConvForCallee(const SDNode *Callee,
                                            const MipsSubtarget &Subtarget) {
  MipsCCState::SpecialCallingConvType SpecialCallingConv = NoSpecialCallingConv;
  if (Subtarget.inMips16HardFloat()) {
    if (const GlobalAddressSDNode *G =
            dyn_cast<const GlobalAddressSDNode>(Callee)) {
      llvm::StringRef Sym = G->getGlobal()->getName();
      Function *F = G->getGlobal()->getParent()->getFunction(Sym);
      if (F && F->hasFnAttribute("__Mips16RetHelper")) {
        SpecialCallingConv = Mips16RetHelperConv;
      }
    }
  }
  return SpecialCallingConv;
}

void MipsCCState::PreAnalyzeCallResultForF128(
    const SmallVectorImpl<ISD::InputArg> &Ins, const Type *RetTy) {
  for (unsigned i = 0; i < Ins.size(); ++i) {
    OriginalArgWasF128.push_back(originalTypeIsF128(RetTy));
    OriginalArgWasFloat.push_back(RetTy->isFloatingPointTy());
  }
}

/// Identify lowered values that originated from f128 or float arguments and
/// record this for use by RetCC_MipsN.
void MipsCCState::PreAnalyzeCallReturnForF128(
    const SmallVectorImpl<ISD::OutputArg> &Outs, const Type *RetTy) {
  for (unsigned i = 0; i < Outs.size(); ++i) {
    OriginalArgWasF128.push_back(originalTypeIsF128(RetTy));
    OriginalArgWasFloat.push_back(
        RetTy->isFloatingPointTy());
  }
}

/// Identify lower values that originated from vXfXX and record
/// this.
void MipsCCState::PreAnalyzeCallResultForVectorFloat(
    const SmallVectorImpl<ISD::InputArg> &Ins, const Type *RetTy) {
  for (unsigned i = 0; i < Ins.size(); ++i) {
    OriginalRetWasFloatVector.push_back(originalTypeIsVectorFloat(RetTy));
  }
}

/// Identify lowered values that originated from vXfXX arguments and record
/// this.
void MipsCCState::PreAnalyzeReturnForVectorFloat(
    const SmallVectorImpl<ISD::OutputArg> &Outs) {
  for (unsigned i = 0; i < Outs.size(); ++i) {
    ISD::OutputArg Out = Outs[i];
    OriginalRetWasFloatVector.push_back(
        originalEVTTypeIsVectorFloat(Out.ArgVT));
  }
}

void MipsCCState::PreAnalyzeReturnValue(EVT ArgVT) {
  OriginalRetWasFloatVector.push_back(originalEVTTypeIsVectorFloat(ArgVT));
}

void MipsCCState::PreAnalyzeCallOperand(const Type *ArgTy) {
  OriginalArgWasF128.push_back(originalTypeIsF128(ArgTy));
  OriginalArgWasFloat.push_back(ArgTy->isFloatingPointTy());
  OriginalArgWasFloatVector.push_back(ArgTy->isVectorTy());
}

/// Identify lowered values that originated from f128, float and sret to vXfXX
/// arguments and record this.
void MipsCCState::PreAnalyzeCallOperands(
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    std::vector<TargetLowering::ArgListEntry> &FuncArgs) {
  for (unsigned i = 0; i < Outs.size(); ++i) {
    TargetLowering::ArgListEntry FuncArg = FuncArgs[Outs[i].OrigArgIndex];

    OriginalArgWasF128.push_back(originalTypeIsF128(FuncArg.OrigTy));
    OriginalArgWasFloat.push_back(FuncArg.OrigTy->isFloatingPointTy());
    OriginalArgWasFloatVector.push_back(FuncArg.OrigTy->isVectorTy());
  }
}

void MipsCCState::PreAnalyzeFormalArgument(const Type *ArgTy,
                                           ISD::ArgFlagsTy Flags) {
  // SRet arguments cannot originate from f128 or {f128} returns so we just
  // push false. We have to handle this specially since SRet arguments
  // aren't mapped to an original argument.
  if (Flags.isSRet()) {
    OriginalArgWasF128.push_back(false);
    OriginalArgWasFloat.push_back(false);
    OriginalArgWasFloatVector.push_back(false);
    return;
  }

  OriginalArgWasF128.push_back(originalTypeIsF128(ArgTy));
  OriginalArgWasFloat.push_back(ArgTy->isFloatingPointTy());

  // The MIPS vector ABI exhibits a corner case of sorts or quirk; if the
  // first argument is actually an SRet pointer to a vector, then the next
  // argument slot is $a2.
  OriginalArgWasFloatVector.push_back(ArgTy->isVectorTy());
}

/// Identify lowered values that originated from f128, float and vXfXX arguments
/// and record this.
void MipsCCState::PreAnalyzeFormalArgumentsForF128(
    const SmallVectorImpl<ISD::InputArg> &Ins) {
  const MachineFunction &MF = getMachineFunction();
  for (unsigned i = 0; i < Ins.size(); ++i) {
    Function::const_arg_iterator FuncArg = MF.getFunction().arg_begin();

    // SRet arguments cannot originate from f128 or {f128} returns so we just
    // push false. We have to handle this specially since SRet arguments
    // aren't mapped to an original argument.
    if (Ins[i].Flags.isSRet()) {
      OriginalArgWasF128.push_back(false);
      OriginalArgWasFloat.push_back(false);
      OriginalArgWasFloatVector.push_back(false);
      continue;
    }

    assert(Ins[i].getOrigArgIndex() < MF.getFunction().arg_size());
    std::advance(FuncArg, Ins[i].getOrigArgIndex());

    OriginalArgWasF128.push_back(originalTypeIsF128(FuncArg->getType()));
    OriginalArgWasFloat.push_back(FuncArg->getType()->isFloatingPointTy());

    // The MIPS vector ABI exhibits a corner case of sorts or quirk; if the
    // first argument is actually an SRet pointer to a vector, then the next
    // argument slot is $a2.
    OriginalArgWasFloatVector.push_back(FuncArg->getType()->isVectorTy());
  }
}
