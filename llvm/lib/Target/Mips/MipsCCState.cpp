//===---- MipsCCState.cpp - CCState with Mips specific extensions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsCCState.h"
#include "MipsSubtarget.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// This function returns true if CallSym is a long double emulation routine.
static bool isF128SoftLibCall(const char *CallSym) {
  const char *const LibCalls[] = {
      "__addtf3",      "__divtf3",     "__eqtf2",       "__extenddftf2",
      "__extendsftf2", "__fixtfdi",    "__fixtfsi",     "__fixtfti",
      "__fixunstfdi",  "__fixunstfsi", "__fixunstfti",  "__floatditf",
      "__floatsitf",   "__floattitf",  "__floatunditf", "__floatunsitf",
      "__floatuntitf", "__getf2",      "__gttf2",       "__letf2",
      "__lttf2",       "__multf3",     "__netf2",       "__powitf2",
      "__subtf3",      "__trunctfdf2", "__trunctfsf2",  "__unordtf2",
      "ceill",         "copysignl",    "cosl",          "exp2l",
      "expl",          "floorl",       "fmal",          "fmodl",
      "log10l",        "log2l",        "logl",          "nearbyintl",
      "powl",          "rintl",        "sinl",          "sqrtl",
      "truncl"};

  const char *const *End = LibCalls + array_lengthof(LibCalls);

  // Check that LibCalls is sorted alphabetically.
  MipsTargetLowering::LTStr Comp;

#ifndef NDEBUG
  for (const char *const *I = LibCalls; I < End - 1; ++I)
    assert(Comp(*I, *(I + 1)));
#endif

  return std::binary_search(LibCalls, End, CallSym, Comp);
}

/// This function returns true if Ty is fp128, {f128} or i128 which was
/// originally a fp128.
static bool originalTypeIsF128(const Type *Ty, const SDNode *CallNode) {
  if (Ty->isFP128Ty())
    return true;

  if (Ty->isStructTy() && Ty->getStructNumElements() == 1 &&
      Ty->getStructElementType(0)->isFP128Ty())
    return true;

  const ExternalSymbolSDNode *ES =
      dyn_cast_or_null<const ExternalSymbolSDNode>(CallNode);

  // If the Ty is i128 and the function being called is a long double emulation
  // routine, then the original type is f128.
  return (ES && Ty->isIntegerTy(128) && isF128SoftLibCall(ES->getSymbol()));
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
    const SmallVectorImpl<ISD::InputArg> &Ins,
    const TargetLowering::CallLoweringInfo &CLI) {
  for (unsigned i = 0; i < Ins.size(); ++i) {
    OriginalArgWasF128.push_back(
        originalTypeIsF128(CLI.RetTy, CLI.Callee.getNode()));
    OriginalArgWasFloat.push_back(CLI.RetTy->isFloatingPointTy());
  }
}

/// Identify lowered values that originated from f128 arguments and record
/// this for use by RetCC_MipsN.
void MipsCCState::PreAnalyzeReturnForF128(
    const SmallVectorImpl<ISD::OutputArg> &Outs) {
  const MachineFunction &MF = getMachineFunction();
  for (unsigned i = 0; i < Outs.size(); ++i) {
    OriginalArgWasF128.push_back(
        originalTypeIsF128(MF.getFunction()->getReturnType(), nullptr));
    OriginalArgWasFloat.push_back(
        MF.getFunction()->getReturnType()->isFloatingPointTy());
  }
}

/// Identify lowered values that originated from f128 arguments and record
/// this.
void MipsCCState::PreAnalyzeCallOperands(
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    std::vector<TargetLowering::ArgListEntry> &FuncArgs,
    const SDNode *CallNode) {
  for (unsigned i = 0; i < Outs.size(); ++i) {
    OriginalArgWasF128.push_back(
        originalTypeIsF128(FuncArgs[Outs[i].OrigArgIndex].Ty, CallNode));
    OriginalArgWasFloat.push_back(
        FuncArgs[Outs[i].OrigArgIndex].Ty->isFloatingPointTy());
    CallOperandIsFixed.push_back(Outs[i].IsFixed);
  }
}

/// Identify lowered values that originated from f128 arguments and record
/// this.
void MipsCCState::PreAnalyzeFormalArgumentsForF128(
    const SmallVectorImpl<ISD::InputArg> &Ins) {
  const MachineFunction &MF = getMachineFunction();
  for (unsigned i = 0; i < Ins.size(); ++i) {
    Function::const_arg_iterator FuncArg = MF.getFunction()->arg_begin();

    // SRet arguments cannot originate from f128 or {f128} returns so we just
    // push false. We have to handle this specially since SRet arguments
    // aren't mapped to an original argument.
    if (Ins[i].Flags.isSRet()) {
      OriginalArgWasF128.push_back(false);
      OriginalArgWasFloat.push_back(false);
      continue;
    }

    assert(Ins[i].getOrigArgIndex() < MF.getFunction()->arg_size());
    std::advance(FuncArg, Ins[i].getOrigArgIndex());

    OriginalArgWasF128.push_back(
        originalTypeIsF128(FuncArg->getType(), nullptr));
    OriginalArgWasFloat.push_back(FuncArg->getType()->isFloatingPointTy());
  }
}
