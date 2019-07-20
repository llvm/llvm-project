//===-- lib/CodeGen/GlobalISel/CallLowering.cpp - Call lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements some simple delegations needed for call lowering.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "call-lowering"

using namespace llvm;

void CallLowering::anchor() {}

bool CallLowering::lowerCall(MachineIRBuilder &MIRBuilder, ImmutableCallSite CS,
                             ArrayRef<Register> ResRegs,
                             ArrayRef<ArrayRef<Register>> ArgRegs,
                             Register SwiftErrorVReg,
                             std::function<unsigned()> GetCalleeReg) const {
  auto &DL = CS.getParent()->getParent()->getParent()->getDataLayout();

  // First step is to marshall all the function's parameters into the correct
  // physregs and memory locations. Gather the sequence of argument types that
  // we'll pass to the assigner function.
  SmallVector<ArgInfo, 8> OrigArgs;
  unsigned i = 0;
  unsigned NumFixedArgs = CS.getFunctionType()->getNumParams();
  for (auto &Arg : CS.args()) {
    ArgInfo OrigArg{ArgRegs[i], Arg->getType(), ISD::ArgFlagsTy{},
                    i < NumFixedArgs};
    setArgFlags(OrigArg, i + AttributeList::FirstArgIndex, DL, CS);
    // We don't currently support swiftself args.
    if (OrigArg.Flags.isSwiftSelf())
      return false;
    OrigArgs.push_back(OrigArg);
    ++i;
  }

  MachineOperand Callee = MachineOperand::CreateImm(0);
  if (const Function *F = CS.getCalledFunction())
    Callee = MachineOperand::CreateGA(F, 0);
  else
    Callee = MachineOperand::CreateReg(GetCalleeReg(), false);

  ArgInfo OrigRet{ResRegs, CS.getType(), ISD::ArgFlagsTy{}};
  if (!OrigRet.Ty->isVoidTy())
    setArgFlags(OrigRet, AttributeList::ReturnIndex, DL, CS);

  return lowerCall(MIRBuilder, CS.getCallingConv(), Callee, OrigRet, OrigArgs,
                   SwiftErrorVReg);
}

template <typename FuncInfoTy>
void CallLowering::setArgFlags(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                               const DataLayout &DL,
                               const FuncInfoTy &FuncInfo) const {
  const AttributeList &Attrs = FuncInfo.getAttributes();
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

    auto Ty = Attrs.getAttribute(OpIdx, Attribute::ByVal).getValueAsType();
    Arg.Flags.setByValSize(DL.getTypeAllocSize(Ty ? Ty : ElementTy));

    // For ByVal, alignment should be passed from FE.  BE will guess if
    // this info is not there but there are cases it cannot get right.
    unsigned FrameAlign;
    if (FuncInfo.getParamAlignment(OpIdx - 2))
      FrameAlign = FuncInfo.getParamAlignment(OpIdx - 2);
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

Register CallLowering::packRegs(ArrayRef<Register> SrcRegs, Type *PackedTy,
                                MachineIRBuilder &MIRBuilder) const {
  assert(SrcRegs.size() > 1 && "Nothing to pack");

  const DataLayout &DL = MIRBuilder.getMF().getDataLayout();
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();

  LLT PackedLLT = getLLTForType(*PackedTy, DL);

  SmallVector<LLT, 8> LLTs;
  SmallVector<uint64_t, 8> Offsets;
  computeValueLLTs(DL, *PackedTy, LLTs, &Offsets);
  assert(LLTs.size() == SrcRegs.size() && "Regs / types mismatch");

  Register Dst = MRI->createGenericVirtualRegister(PackedLLT);
  MIRBuilder.buildUndef(Dst);
  for (unsigned i = 0; i < SrcRegs.size(); ++i) {
    Register NewDst = MRI->createGenericVirtualRegister(PackedLLT);
    MIRBuilder.buildInsert(NewDst, Dst, SrcRegs[i], Offsets[i]);
    Dst = NewDst;
  }

  return Dst;
}

void CallLowering::unpackRegs(ArrayRef<Register> DstRegs, Register SrcReg,
                              Type *PackedTy,
                              MachineIRBuilder &MIRBuilder) const {
  assert(DstRegs.size() > 1 && "Nothing to unpack");

  const DataLayout &DL = MIRBuilder.getMF().getDataLayout();

  SmallVector<LLT, 8> LLTs;
  SmallVector<uint64_t, 8> Offsets;
  computeValueLLTs(DL, *PackedTy, LLTs, &Offsets);
  assert(LLTs.size() == DstRegs.size() && "Regs / types mismatch");

  for (unsigned i = 0; i < DstRegs.size(); ++i)
    MIRBuilder.buildExtract(DstRegs[i], SrcReg, Offsets[i]);
}

bool CallLowering::handleAssignments(MachineIRBuilder &MIRBuilder,
                                     ArrayRef<ArgInfo> Args,
                                     ValueHandler &Handler) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());
  return handleAssignments(CCInfo, ArgLocs, MIRBuilder, Args, Handler);
}

bool CallLowering::handleAssignments(CCState &CCInfo,
                                     SmallVectorImpl<CCValAssign> &ArgLocs,
                                     MachineIRBuilder &MIRBuilder,
                                     ArrayRef<ArgInfo> Args,
                                     ValueHandler &Handler) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();

  unsigned NumArgs = Args.size();
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT CurVT = MVT::getVT(Args[i].Ty);
    if (Handler.assignArg(i, CurVT, CurVT, CCValAssign::Full, Args[i], CCInfo)) {
      // Try to use the register type if we couldn't assign the VT.
      if (!Handler.isArgumentHandler() || !CurVT.isValid())
        return false;
      CurVT = TLI->getRegisterTypeForCallingConv(
          F.getContext(), F.getCallingConv(), EVT(CurVT));
      if (Handler.assignArg(i, CurVT, CurVT, CCValAssign::Full, Args[i], CCInfo))
        return false;
    }
  }

  for (unsigned i = 0, e = Args.size(), j = 0; i != e; ++i, ++j) {
    assert(j < ArgLocs.size() && "Skipped too many arg locs");

    CCValAssign &VA = ArgLocs[j];
    assert(VA.getValNo() == i && "Location doesn't correspond to current arg");

    if (VA.needsCustom()) {
      j += Handler.assignCustomValue(Args[i], makeArrayRef(ArgLocs).slice(j));
      continue;
    }

    assert(Args[i].Regs.size() == 1 &&
           "Can't handle multiple virtual regs yet");

    // FIXME: Pack registers if we have more than one.
    Register ArgReg = Args[i].Regs[0];

    if (VA.isRegLoc()) {
      MVT OrigVT = MVT::getVT(Args[i].Ty);
      MVT VAVT = VA.getValVT();
      if (Handler.isArgumentHandler() && VAVT != OrigVT) {
        if (VAVT.getSizeInBits() < OrigVT.getSizeInBits())
          return false; // Can't handle this type of arg yet.
        const LLT VATy(VAVT);
        Register NewReg =
            MIRBuilder.getMRI()->createGenericVirtualRegister(VATy);
        Handler.assignValueToReg(NewReg, VA.getLocReg(), VA);
        // If it's a vector type, we either need to truncate the elements
        // or do an unmerge to get the lower block of elements.
        if (VATy.isVector() &&
            VATy.getNumElements() > OrigVT.getVectorNumElements()) {
          const LLT OrigTy(OrigVT);
          // Just handle the case where the VA type is 2 * original type.
          if (VATy.getNumElements() != OrigVT.getVectorNumElements() * 2) {
            LLVM_DEBUG(dbgs()
                       << "Incoming promoted vector arg has too many elts");
            return false;
          }
          auto Unmerge = MIRBuilder.buildUnmerge({OrigTy, OrigTy}, {NewReg});
          MIRBuilder.buildCopy(ArgReg, Unmerge.getReg(0));
        } else {
          MIRBuilder.buildTrunc(ArgReg, {NewReg}).getReg(0);
        }
      } else {
        Handler.assignValueToReg(ArgReg, VA.getLocReg(), VA);
      }
    } else if (VA.isMemLoc()) {
      MVT VT = MVT::getVT(Args[i].Ty);
      unsigned Size = VT == MVT::iPTR ? DL.getPointerSize()
                                      : alignTo(VT.getSizeInBits(), 8) / 8;
      unsigned Offset = VA.getLocMemOffset();
      MachinePointerInfo MPO;
      Register StackAddr = Handler.getStackAddress(Size, Offset, MPO);
      Handler.assignValueToAddress(ArgReg, StackAddr, Size, MPO, VA);
    } else {
      // FIXME: Support byvals and other weirdness
      return false;
    }
  }
  return true;
}

Register CallLowering::ValueHandler::extendRegister(Register ValReg,
                                                    CCValAssign &VA) {
  LLT LocTy{VA.getLocVT()};
  if (LocTy.getSizeInBits() == MRI.getType(ValReg).getSizeInBits())
    return ValReg;
  switch (VA.getLocInfo()) {
  default: break;
  case CCValAssign::Full:
  case CCValAssign::BCvt:
    // FIXME: bitconverting between vector types may or may not be a
    // nop in big-endian situations.
    return ValReg;
  case CCValAssign::AExt: {
    auto MIB = MIRBuilder.buildAnyExt(LocTy, ValReg);
    return MIB->getOperand(0).getReg();
  }
  case CCValAssign::SExt: {
    Register NewReg = MRI.createGenericVirtualRegister(LocTy);
    MIRBuilder.buildSExt(NewReg, ValReg);
    return NewReg;
  }
  case CCValAssign::ZExt: {
    Register NewReg = MRI.createGenericVirtualRegister(LocTy);
    MIRBuilder.buildZExt(NewReg, ValReg);
    return NewReg;
  }
  }
  llvm_unreachable("unable to extend register");
}

void CallLowering::ValueHandler::anchor() {}
