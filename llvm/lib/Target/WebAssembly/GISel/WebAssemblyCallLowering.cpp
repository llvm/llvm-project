//===-- WebAssemblyCallLowering.cpp - Call lowering for GlobalISel -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyCallLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyISelLowering.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Value.h"

#define DEBUG_TYPE "wasm-call-lowering"

using namespace llvm;

// Test whether the given calling convention is supported.
static bool callingConvSupported(CallingConv::ID CallConv) {
  // We currently support the language-independent target-independent
  // conventions. We don't yet have a way to annotate calls with properties like
  // "cold", and we don't have any call-clobbered registers, so these are mostly
  // all handled the same.
  return CallConv == CallingConv::C || CallConv == CallingConv::Fast ||
         CallConv == CallingConv::Cold ||
         CallConv == CallingConv::PreserveMost ||
         CallConv == CallingConv::PreserveAll ||
         CallConv == CallingConv::CXX_FAST_TLS ||
         CallConv == CallingConv::WASM_EmscriptenInvoke ||
         CallConv == CallingConv::Swift;
}

WebAssemblyCallLowering::WebAssemblyCallLowering(
    const WebAssemblyTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool WebAssemblyCallLowering::canLowerReturn(MachineFunction &MF,
                                             CallingConv::ID CallConv,
                                             SmallVectorImpl<BaseArgInfo> &Outs,
                                             bool IsVarArg) const {
  return WebAssembly::canLowerReturn(Outs.size(),
                                     &MF.getSubtarget<WebAssemblySubtarget>());
}

bool WebAssemblyCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                          const Value *Val,
                                          ArrayRef<Register> VRegs,
                                          FunctionLoweringInfo &FLI,
                                          Register SwiftErrorVReg) const {
  if (!Val)
    return true; // allow only void returns for now

  return false;
}

static unsigned getWASMArgumentOpcode(MVT ArgType) {
  switch (ArgType.SimpleTy) {
  case MVT::i32:
    return WebAssembly::ARGUMENT_i32;
  case MVT::i64:
    return WebAssembly::ARGUMENT_i64;
  case MVT::f32:
    return WebAssembly::ARGUMENT_f32;
  case MVT::f64:
    return WebAssembly::ARGUMENT_f64;

  case MVT::funcref:
    return WebAssembly::ARGUMENT_funcref;
  case MVT::externref:
    return WebAssembly::ARGUMENT_externref;
  case MVT::exnref:
    return WebAssembly::ARGUMENT_exnref;

  case MVT::v16i8:
    return WebAssembly::ARGUMENT_v16i8;
  case MVT::v8i16:
    return WebAssembly::ARGUMENT_v8i16;
  case MVT::v4i32:
    return WebAssembly::ARGUMENT_v4i32;
  case MVT::v2i64:
    return WebAssembly::ARGUMENT_v2i64;
  case MVT::v8f16:
    return WebAssembly::ARGUMENT_v8f16;
  case MVT::v4f32:
    return WebAssembly::ARGUMENT_v4f32;
  case MVT::v2f64:
    return WebAssembly::ARGUMENT_v2f64;
  default:
    break;
  }
  llvm_unreachable("Found unexpected type for WASM argument");
}

bool WebAssemblyCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo *MFI = MF.getInfo<WebAssemblyFunctionInfo>();
  const DataLayout &DL = F.getDataLayout();
  const WebAssemblyTargetLowering &TLI = *getTLI<WebAssemblyTargetLowering>();
  const WebAssemblySubtarget &Subtarget =
      MF.getSubtarget<WebAssemblySubtarget>();
  const WebAssemblyRegisterInfo &TRI = *Subtarget.getRegisterInfo();
  const WebAssemblyInstrInfo &TII = *Subtarget.getInstrInfo();
  const RegisterBankInfo &RBI = *Subtarget.getRegBankInfo();

  LLVMContext &Ctx = MIRBuilder.getContext();
  const CallingConv::ID CallConv = F.getCallingConv();

  if (!callingConvSupported(CallConv)) {
    return false;
  }

  MF.getRegInfo().addLiveIn(WebAssembly::ARGUMENTS);
  MF.front().addLiveIn(WebAssembly::ARGUMENTS);

  SmallVector<ArgInfo, 8> SplitArgs;

  if (!FLI.CanLowerReturn) {
    insertSRetIncomingArgument(F, SplitArgs, FLI.DemoteRegister, MRI, DL);
  }

  unsigned ArgIdx = 0;
  bool HasSwiftErrorArg = false;
  bool HasSwiftSelfArg = false;
  for (const Argument &Arg : F.args()) {
    ArgInfo OrigArg{VRegs[ArgIdx], Arg.getType(), ArgIdx};
    setArgFlags(OrigArg, ArgIdx + AttributeList::FirstArgIndex, DL, F);

    HasSwiftSelfArg |= Arg.hasSwiftSelfAttr();
    HasSwiftErrorArg |= Arg.hasSwiftErrorAttr();
    if (Arg.hasInAllocaAttr()) {
      return false;
    }
    if (Arg.hasNestAttr()) {
      return false;
    }
    splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    ++ArgIdx;
  }

  unsigned FinalArgIdx = 0;
  for (ArgInfo &Arg : SplitArgs) {
    EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
    MVT NewVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
    LLT OrigLLT = getLLTForType(*Arg.Ty, DL);
    LLT NewLLT = getLLTForMVT(NewVT);

    // If we need to split the type over multiple regs, check it's a scenario
    // we currently support.
    unsigned NumParts =
        TLI.getNumRegistersForCallingConv(Ctx, CallConv, OrigVT);

    ISD::ArgFlagsTy OrigFlags = Arg.Flags[0];
    Arg.Flags.clear();

    for (unsigned Part = 0; Part < NumParts; ++Part) {
      ISD::ArgFlagsTy Flags = OrigFlags;
      if (Part == 0) {
        Flags.setSplit();
      } else {
        Flags.setOrigAlign(Align(1));
        if (Part == NumParts - 1)
          Flags.setSplitEnd();
      }

      Arg.Flags.push_back(Flags);
    }

    Arg.OrigRegs.assign(Arg.Regs.begin(), Arg.Regs.end());
    if (NumParts != 1 || OrigVT != NewVT) {
      // If we can't directly assign the register, we need one or more
      // intermediate values.
      Arg.Regs.resize(NumParts);

      // For each split register, create and assign a vreg that will store
      // the incoming component of the larger value. These will later be
      // merged to form the final vreg.
      for (unsigned Part = 0; Part < NumParts; ++Part) {
        Arg.Regs[Part] = MRI.createGenericVirtualRegister(NewLLT);
      }
    }

    for (unsigned Part = 0; Part < NumParts; ++Part) {
      MachineInstrBuilder ArgInst =
          MIRBuilder.buildInstr(getWASMArgumentOpcode(NewVT))
              .addDef(Arg.Regs[Part])
              .addImm(FinalArgIdx);

      constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *ArgInst,
                               ArgInst->getDesc(), ArgInst->getOperand(0), 0);
      MFI->addParam(NewVT);
      ++FinalArgIdx;
    }

    if (NumParts != 1 || OrigVT != NewVT) {
      buildCopyFromRegs(MIRBuilder, Arg.OrigRegs, Arg.Regs, OrigLLT, NewLLT,
                        Arg.Flags[0]);
    }
  }

  // For swiftcc, emit additional swiftself and swifterror arguments
  // if there aren't. These additional arguments are also added for callee
  // signature They are necessary to match callee and caller signature for
  // indirect call.
  MVT PtrVT = TLI.getPointerTy(DL);
  if (CallConv == CallingConv::Swift) {
    if (!HasSwiftSelfArg) {
      MFI->addParam(PtrVT);
    }
    if (!HasSwiftErrorArg) {
      MFI->addParam(PtrVT);
    }
  }

  // Varargs are copied into a buffer allocated by the caller, and a pointer to
  // the buffer is passed as an argument.
  if (F.isVarArg()) {
    MVT PtrVT = TLI.getPointerTy(DL, 0);
    LLT PtrLLT = LLT::pointer(0, DL.getPointerSizeInBits(0));
    Register VarargVreg = MF.getRegInfo().createGenericVirtualRegister(PtrLLT);

    MFI->setVarargBufferVreg(VarargVreg);

    MachineInstrBuilder ArgInst =
        MIRBuilder.buildInstr(getWASMArgumentOpcode(PtrVT))
            .addDef(VarargVreg)
            .addImm(FinalArgIdx);

    constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *ArgInst,
                             ArgInst->getDesc(), ArgInst->getOperand(0), 0);

    MFI->addParam(PtrVT);
    ++FinalArgIdx;
  }

  // Record the number and types of arguments and results.
  SmallVector<MVT, 4> Params;
  SmallVector<MVT, 4> Results;
  computeSignatureVTs(MF.getFunction().getFunctionType(), &MF.getFunction(),
                      MF.getFunction(), MF.getTarget(), Params, Results);
  for (MVT VT : Results)
    MFI->addResult(VT);

  // TODO: Use signatures in WebAssemblyMachineFunctionInfo too and unify
  // the param logic here with ComputeSignatureVTs
  assert(MFI->getParams().size() == Params.size() &&
         std::equal(MFI->getParams().begin(), MFI->getParams().end(),
                    Params.begin()));
  return true;
}

bool WebAssemblyCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                        CallLoweringInfo &Info) const {
  return false;
}
