//===-- RISCVSelectionDAGTargetInfo.cpp - RISCV SelectionDAG Info
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCVSelectionDAGTargetInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVSelectionDAGTargetInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-selectiondag-target-info"

static cl::opt<unsigned> MaxStrcmpSpecializeLength(
    "riscv-max-strcmp-specialize-length", cl::Hidden,
    cl::desc("Do not specialize strcmp if the length of constant string is "
             "greater or equal to this parameter"),
    cl::init(0));

static bool canSpecializeStrcmp(const GlobalAddressSDNode *GA) {
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(GA->getGlobal());
  if (!GV || !GV->isConstant() || !GV->hasInitializer())
    return false;
  // NOTE: this doesn't work for empty strings
  const ConstantDataArray *CDA =
      dyn_cast<ConstantDataArray>(GV->getInitializer());
  if (!CDA || !CDA->isCString())
    return false;

  StringRef CString = CDA->getAsCString();
  if (CString.str().length() >= MaxStrcmpSpecializeLength)
    return false;

  return true;
}

std::pair<SDValue, SDValue>
RISCVSelectionDAGTargetInfo::EmitTargetCodeForStrcmp(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Chain, SDValue Src1,
    SDValue Src2, MachinePointerInfo Op1PtrInfo,
    MachinePointerInfo Op2PtrInfo) const {
  // This is the default setting, so exit early if the optimization is turned
  // off.
  if (MaxStrcmpSpecializeLength == 0)
    return std::make_pair(SDValue(), Chain);

  const RISCVSubtarget &Subtarget =
      DAG.getMachineFunction().getSubtarget<RISCVSubtarget>();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  MVT XLenVT = Subtarget.getXLenVT();
  const DataLayout &DLayout = DAG.getDataLayout();

  Align NeededAlignment = Align(XLenVT.getSizeInBits() / 8);
  Align Src1Align;
  Align Src2Align;
  if (const Value *Src1V = dyn_cast_if_present<const Value *>(Op1PtrInfo.V)) {
    Src1Align = Src1V->getPointerAlignment(DLayout);
  }
  if (const Value *Src2V = dyn_cast_if_present<const Value *>(Op2PtrInfo.V)) {
    Src2Align = Src2V->getPointerAlignment(DLayout);
  }
  if (!(Src1Align < NeededAlignment || Src2Align < NeededAlignment))
    return std::make_pair(SDValue(), Chain);

  const GlobalAddressSDNode *CStringGA = nullptr;
  SDValue Other;
  MachinePointerInfo MPI;
  bool ConstantStringIsSecond = false;

  const GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Src1);
  if (GA && canSpecializeStrcmp(GA)) {
    CStringGA = GA;
    Other = Src2;
    MPI = Op2PtrInfo;
  }
  if (!CStringGA) {
    GA = dyn_cast<GlobalAddressSDNode>(Src2);
    if (GA && canSpecializeStrcmp(GA)) {
      ConstantStringIsSecond = true;
      CStringGA = GA;
      Other = Src1;
      MPI = Op1PtrInfo;
    }
  }

  if (!CStringGA)
    return std::make_pair(SDValue(), Chain);

  // It could be that the non-constant string is actually aligned, but
  // we can't prove it, so getPointerAlignment will return Align(1).
  // In this case, if the constant string is sufficiently aligned, it's better
  // to call to libc's strcmp.
  // Align ConstantStrAlignment = ConstantStringIsSecond ? Src2Align :
  // Src1Align; if (ConstantStrAlignment >= NeededAlignment)
  //  return std::make_pair(SDValue(), Chain);

  SDValue TGA = DAG.getTargetGlobalAddress(CStringGA->getGlobal(), DL,
                                           TLI.getPointerTy(DLayout), 0,
                                           CStringGA->getTargetFlags());

  SDValue Str1 = TGA;
  SDValue Str2 = Other;
  if (ConstantStringIsSecond)
    std::swap(Str1, Str2);

  MachineFunction &MF = DAG.getMachineFunction();
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MPI, MachineMemOperand::MOLoad, LLT(MVT::i8), Align(1));
  // TODO: what should be the MemVT?
  SDValue STRCMPNode = DAG.getMemIntrinsicNode(
      RISCVISD::STRCMP, DL, DAG.getVTList(XLenVT, MVT::Other),
      {Chain, Str1, Str2}, MVT::i8, MMO);

  SDValue ChainOut = STRCMPNode.getValue(1);
  return std::make_pair(STRCMPNode, ChainOut);
}
