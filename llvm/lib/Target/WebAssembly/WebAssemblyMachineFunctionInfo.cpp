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
#include "MCTargetDesc/WebAssemblyInstPrinter.h"
#include "Utils/WebAssemblyTypeUtilities.h"
#include "WebAssemblyISelLowering.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

WebAssemblyFunctionInfo::~WebAssemblyFunctionInfo() = default; // anchor.

void WebAssemblyFunctionInfo::initWARegs(MachineRegisterInfo &MRI) {
  assert(WARegs.empty());
  unsigned Reg = UnusedReg;
  WARegs.resize(MRI.getNumVirtRegs(), Reg);
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
    : CFGStackified(MFI.isCFGStackified()) {
  auto *EHInfo = MFI.getWasmEHFuncInfo();
  const llvm::MachineFunction &MF = MFI.getMachineFunction();

  for (auto VT : MFI.getParams())
    Params.push_back(EVT(VT).getEVTString());
  for (auto VT : MFI.getResults())
    Results.push_back(EVT(VT).getEVTString());

  //  MFI.getWasmEHFuncInfo() is non-null only for functions with the
  //  personality function.
  if (EHInfo) {
    // SrcToUnwindDest can contain stale mappings in case BBs are removed in
    // optimizations, in case, for example, they are unreachable. We should not
    // include their info.
    SmallPtrSet<const MachineBasicBlock *, 16> MBBs;
    for (const auto &MBB : MF)
      MBBs.insert(&MBB);
    for (auto KV : EHInfo->SrcToUnwindDest) {
      auto *SrcBB = KV.first.get<MachineBasicBlock *>();
      auto *DestBB = KV.second.get<MachineBasicBlock *>();
      if (MBBs.count(SrcBB) && MBBs.count(DestBB))
        SrcToUnwindDest[SrcBB->getNumber()] = DestBB->getNumber();
    }
  }
}

void yaml::WebAssemblyFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<WebAssemblyFunctionInfo>::mapping(YamlIO, *this);
}

void WebAssemblyFunctionInfo::initializeBaseYamlFields(
    const yaml::WebAssemblyFunctionInfo &YamlMFI) {
  CFGStackified = YamlMFI.CFGStackified;
  for (auto VT : YamlMFI.Params)
    addParam(WebAssembly::parseMVT(VT.Value));
  for (auto VT : YamlMFI.Results)
    addResult(WebAssembly::parseMVT(VT.Value));
  if (WasmEHInfo) {
    for (auto KV : YamlMFI.SrcToUnwindDest)
      WasmEHInfo->setUnwindDest(MF.getBlockNumbered(KV.first),
                                MF.getBlockNumbered(KV.second));
  }
}
