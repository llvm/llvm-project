//===-- lib/CodeGen/GlobalISel/CallLowering.cpp - Call lowering -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements some simple delegations needed for call lowering.
///
//===----------------------------------------------------------------------===//


#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetLowering.h"

using namespace llvm;

bool CallLowering::lowerCall(
    MachineIRBuilder &MIRBuilder, const CallInst &CI, unsigned ResReg,
    ArrayRef<unsigned> ArgRegs, std::function<unsigned()> GetCalleeReg) const {
  auto &DL = CI.getParent()->getParent()->getParent()->getDataLayout();

  // First step is to marshall all the function's parameters into the correct
  // physregs and memory locations. Gather the sequence of argument types that
  // we'll pass to the assigner function.
  SmallVector<ArgInfo, 8> OrigArgs;
  unsigned i = 0;
  for (auto &Arg : CI.arg_operands()) {
    ArgInfo OrigArg{ArgRegs[i], Arg->getType(), ISD::ArgFlagsTy{}};
    setArgFlags(OrigArg, i + 1, DL, CI);
    OrigArgs.push_back(OrigArg);
    ++i;
  }

  MachineOperand Callee = MachineOperand::CreateImm(0);
  if (Function *F = CI.getCalledFunction())
    Callee = MachineOperand::CreateGA(F, 0);
  else
    Callee = MachineOperand::CreateReg(GetCalleeReg(), false);

  ArgInfo OrigRet{ResReg, CI.getType(), ISD::ArgFlagsTy{}};
  if (!OrigRet.Ty->isVoidTy())
    setArgFlags(OrigRet, AttributeSet::ReturnIndex, DL, CI);

  return lowerCall(MIRBuilder, Callee, OrigRet, OrigArgs);
}

template <typename FuncInfoTy>
void CallLowering::setArgFlags(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                               const DataLayout &DL,
                               const FuncInfoTy &FuncInfo) const {
  const AttributeSet &Attrs = FuncInfo.getAttributes();
  if (Attrs.hasAttribute(OpIdx, Attribute::ZExt))
    Arg.Flags.setZExt();
  if (Attrs.hasAttribute(OpIdx, Attribute::SExt))
    Arg.Flags.setSExt();
  if (Attrs.hasAttribute(OpIdx, Attribute::InReg))
    Arg.Flags.setInReg();
  if (Attrs.hasAttribute(OpIdx, Attribute::StructRet))
    Arg.Flags.setSRet();
  if (Attrs.hasAttribute(OpIdx, Attribute::SwiftSelf))
    Arg.Flags.setSwiftSelf();
  if (Attrs.hasAttribute(OpIdx, Attribute::SwiftError))
    Arg.Flags.setSwiftError();
  if (Attrs.hasAttribute(OpIdx, Attribute::ByVal))
    Arg.Flags.setByVal();
  if (Attrs.hasAttribute(OpIdx, Attribute::InAlloca))
    Arg.Flags.setInAlloca();

  if (Arg.Flags.isByVal() || Arg.Flags.isInAlloca()) {
    Type *ElementTy = cast<PointerType>(Arg.Ty)->getElementType();
    Arg.Flags.setByValSize(DL.getTypeAllocSize(ElementTy));
    // For ByVal, alignment should be passed from FE.  BE will guess if
    // this info is not there but there are cases it cannot get right.
    unsigned FrameAlign;
    if (FuncInfo.getParamAlignment(OpIdx))
      FrameAlign = FuncInfo.getParamAlignment(OpIdx);
    else
      FrameAlign = getTLI()->getByValTypeAlignment(ElementTy, DL);
    Arg.Flags.setByValAlign(FrameAlign);
  }
  if (Attrs.hasAttribute(OpIdx, Attribute::Nest))
    Arg.Flags.setNest();
  Arg.Flags.setOrigAlign(DL.getABITypeAlignment(Arg.Ty));
}

template void
CallLowering::setArgFlags<Function>(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                                    const DataLayout &DL,
                                    const Function &FuncInfo) const;

template void
CallLowering::setArgFlags<CallInst>(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                                    const DataLayout &DL,
                                    const CallInst &FuncInfo) const;
