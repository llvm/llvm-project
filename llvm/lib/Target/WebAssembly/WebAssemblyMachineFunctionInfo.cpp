//=- WebAssemblyMachineFunctionInfo.cpp - WebAssembly Machine Function Info -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements WebAssembly-specific per-machine-function
/// information.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMachineFunctionInfo.h"
#include "Utils/WebAssemblyTypeUtilities.h"
#include "WebAssemblyISelLowering.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

WebAssemblyFunctionInfo::~WebAssemblyFunctionInfo() = default; // anchor.

MachineFunctionInfo *WebAssemblyFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<WebAssemblyFunctionInfo>(*this);
}

void WebAssemblyFunctionInfo::initWARegs(MachineRegisterInfo &MRI) {
  assert(WARegs.empty());
  unsigned Reg = WebAssembly::UnusedReg;
  WARegs.resize(MRI.getNumVirtRegs(), Reg);
}

void llvm::computeLegalValueVTs(const WebAssemblyTargetLowering &TLI,
                                LLVMContext &Ctx, const DataLayout &DL,
                                Type *Ty, SmallVectorImpl<MVT> &ValueVTs) {
  SmallVector<EVT, 4> VTs;
  ComputeValueVTs(TLI, DL, Ty, VTs);

  for (EVT VT : VTs) {
    unsigned NumRegs = TLI.getNumRegisters(Ctx, VT);
    MVT RegisterVT = TLI.getRegisterType(Ctx, VT);
    for (unsigned I = 0; I != NumRegs; ++I)
      ValueVTs.push_back(RegisterVT);
  }
}

void llvm::computeLegalValueVTs(const Function &F, const TargetMachine &TM,
                                Type *Ty, SmallVectorImpl<MVT> &ValueVTs) {
  const DataLayout &DL(F.getDataLayout());
  const WebAssemblyTargetLowering &TLI =
      *TM.getSubtarget<WebAssemblySubtarget>(F).getTargetLowering();
  computeLegalValueVTs(TLI, F.getContext(), DL, Ty, ValueVTs);
}

void llvm::computeSignatureVTs(const FunctionType *Ty,
                               const Function *TargetFunc,
                               const Function &ContextFunc,
                               const TargetMachine &TM,
                               SmallVectorImpl<MVT> &Params,
                               SmallVectorImpl<MVT> &Results) {
  computeLegalValueVTs(ContextFunc, TM, Ty->getReturnType(), Results);

  MVT PtrVT = MVT::getIntegerVT(TM.createDataLayout().getPointerSizeInBits());
  if (!WebAssembly::canLowerReturn(
          Results.size(),
          &TM.getSubtarget<WebAssemblySubtarget>(ContextFunc))) {
    // WebAssembly can't lower returns of multiple values without demoting to
    // sret unless multivalue is enabled (see
    // WebAssemblyTargetLowering::CanLowerReturn). So replace multiple return
    // values with a poitner parameter.
    Results.clear();
    Params.push_back(PtrVT);
  }

  for (auto *Param : Ty->params())
    computeLegalValueVTs(ContextFunc, TM, Param, Params);
  if (Ty->isVarArg())
    Params.push_back(PtrVT);

  // For swiftcc and swifttailcc, emit additional swiftself, swifterror, and
  // (for swifttailcc) swiftasync parameters if there aren't. These additional
  // parameters are also passed for caller. They are necessary to match callee
  // and caller signature for indirect call.

  if (TargetFunc && (TargetFunc->getCallingConv() == CallingConv::Swift ||
                     TargetFunc->getCallingConv() == CallingConv::SwiftTail)) {
    MVT PtrVT = MVT::getIntegerVT(TM.createDataLayout().getPointerSizeInBits());
    bool HasSwiftErrorArg = false;
    bool HasSwiftSelfArg = false;
    bool HasSwiftAsyncArg = false;
    for (const auto &Arg : TargetFunc->args()) {
      HasSwiftErrorArg |= Arg.hasAttribute(Attribute::SwiftError);
      HasSwiftSelfArg |= Arg.hasAttribute(Attribute::SwiftSelf);
      HasSwiftAsyncArg |= Arg.hasAttribute(Attribute::SwiftAsync);
    }
    if (!HasSwiftSelfArg)
      Params.push_back(PtrVT);
    if (!HasSwiftErrorArg)
      Params.push_back(PtrVT);
    if (TargetFunc->getCallingConv() == CallingConv::SwiftTail &&
        !HasSwiftAsyncArg)
      Params.push_back(PtrVT);
  }
}

void llvm::valTypesFromMVTs(ArrayRef<MVT> In,
                            SmallVectorImpl<wasm::ValType> &Out) {
  for (MVT Ty : In)
    Out.push_back(WebAssembly::toValType(Ty));
}

wasm::WasmSignature *
llvm::signatureFromMVTs(MCContext &Ctx, const SmallVectorImpl<MVT> &Results,
                        const SmallVectorImpl<MVT> &Params) {
  auto Sig = Ctx.createWasmSignature();
  valTypesFromMVTs(Results, Sig->Returns);
  valTypesFromMVTs(Params, Sig->Params);
  return Sig;
}

yaml::WebAssemblyFunctionInfo::WebAssemblyFunctionInfo(
    const llvm::MachineFunction &MF, const llvm::WebAssemblyFunctionInfo &MFI)
    : CFGStackified(MFI.isCFGStackified()) {
  for (auto VT : MFI.getParams())
    Params.push_back(EVT(VT).getEVTString());
  for (auto VT : MFI.getResults())
    Results.push_back(EVT(VT).getEVTString());
}

void yaml::WebAssemblyFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<WebAssemblyFunctionInfo>::mapping(YamlIO, *this);
}

void WebAssemblyFunctionInfo::initializeBaseYamlFields(
    MachineFunction &MF, const yaml::WebAssemblyFunctionInfo &YamlMFI) {
  CFGStackified = YamlMFI.CFGStackified;
  for (auto VT : YamlMFI.Params)
    addParam(WebAssembly::parseMVT(VT.Value));
  for (auto VT : YamlMFI.Results)
    addResult(WebAssembly::parseMVT(VT.Value));
}
