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
#include "WebAssemblyISelLowering.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

WebAssemblyFunctionInfo::~WebAssemblyFunctionInfo() = default; // anchor.

void WebAssemblyFunctionInfo::initWARegs() {
  assert(WARegs.empty());
  unsigned Reg = UnusedReg;
  WARegs.resize(MF.getRegInfo().getNumVirtRegs(), Reg);
}

void llvm::computeLegalValueVTs(const Function &F, const TargetMachine &TM,
                                Type *Ty, SmallVectorImpl<MVT> &ValueVTs) {
  const DataLayout &DL(F.getParent()->getDataLayout());
  const WebAssemblyTargetLowering &TLI =
      *TM.getSubtarget<WebAssemblySubtarget>(F).getTargetLowering();
  SmallVector<EVT, 4> VTs;
  ComputeValueVTs(TLI, DL, Ty, VTs);

  for (EVT VT : VTs) {
    unsigned NumRegs = TLI.getNumRegisters(F.getContext(), VT);
    MVT RegisterVT = TLI.getRegisterType(F.getContext(), VT);
    for (unsigned I = 0; I != NumRegs; ++I)
      ValueVTs.push_back(RegisterVT);
  }
}

void llvm::computeSignatureVTs(const FunctionType *Ty,
                               const Function *TargetFunc,
                               const Function &ContextFunc,
                               const TargetMachine &TM,
                               SmallVectorImpl<MVT> &Params,
                               SmallVectorImpl<MVT> &Results) {
  computeLegalValueVTs(ContextFunc, TM, Ty->getReturnType(), Results);

  MVT PtrVT = MVT::getIntegerVT(TM.createDataLayout().getPointerSizeInBits());
  if (Results.size() > 1 &&
      !TM.getSubtarget<WebAssemblySubtarget>(ContextFunc).hasMultivalue()) {
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

  // For swiftcc, emit additional swiftself and swifterror parameters
  // if there aren't. These additional parameters are also passed for caller.
  // They are necessary to match callee and caller signature for indirect
  // call.

  if (TargetFunc && TargetFunc->getCallingConv() == CallingConv::Swift) {
    MVT PtrVT = MVT::getIntegerVT(TM.createDataLayout().getPointerSizeInBits());
    bool HasSwiftErrorArg = false;
    bool HasSwiftSelfArg = false;
    for (const auto &Arg : TargetFunc->args()) {
      HasSwiftErrorArg |= Arg.hasAttribute(Attribute::SwiftError);
      HasSwiftSelfArg |= Arg.hasAttribute(Attribute::SwiftSelf);
    }
    if (!HasSwiftErrorArg)
      Params.push_back(PtrVT);
    if (!HasSwiftSelfArg)
      Params.push_back(PtrVT);
  }
}

void llvm::valTypesFromMVTs(const ArrayRef<MVT> &In,
                            SmallVectorImpl<wasm::ValType> &Out) {
  for (MVT Ty : In)
    Out.push_back(WebAssembly::toValType(Ty));
}

std::unique_ptr<wasm::WasmSignature>
llvm::signatureFromMVTs(const SmallVectorImpl<MVT> &Results,
                        const SmallVectorImpl<MVT> &Params) {
  auto Sig = std::make_unique<wasm::WasmSignature>();
  valTypesFromMVTs(Results, Sig->Returns);
  valTypesFromMVTs(Params, Sig->Params);
  return Sig;
}

yaml::WebAssemblyFunctionInfo::WebAssemblyFunctionInfo(
    const llvm::WebAssemblyFunctionInfo &MFI)
    : CFGStackified(MFI.isCFGStackified()) {}

void yaml::WebAssemblyFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<WebAssemblyFunctionInfo>::mapping(YamlIO, *this);
}

void WebAssemblyFunctionInfo::initializeBaseYamlFields(
    const yaml::WebAssemblyFunctionInfo &YamlMFI) {
  CFGStackified = YamlMFI.CFGStackified;
}
