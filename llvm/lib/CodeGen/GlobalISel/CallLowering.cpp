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

#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "call-lowering"

using namespace llvm;

void CallLowering::anchor() {}

/// Helper function which updates \p Flags when \p AttrFn returns true.
static void
addFlagsUsingAttrFn(ISD::ArgFlagsTy &Flags,
                    const std::function<bool(Attribute::AttrKind)> &AttrFn) {
  if (AttrFn(Attribute::SExt))
    Flags.setSExt();
  if (AttrFn(Attribute::ZExt))
    Flags.setZExt();
  if (AttrFn(Attribute::InReg))
    Flags.setInReg();
  if (AttrFn(Attribute::StructRet))
    Flags.setSRet();
  if (AttrFn(Attribute::Nest))
    Flags.setNest();
  if (AttrFn(Attribute::ByVal))
    Flags.setByVal();
  if (AttrFn(Attribute::Preallocated))
    Flags.setPreallocated();
  if (AttrFn(Attribute::InAlloca))
    Flags.setInAlloca();
  if (AttrFn(Attribute::Returned))
    Flags.setReturned();
  if (AttrFn(Attribute::SwiftSelf))
    Flags.setSwiftSelf();
  if (AttrFn(Attribute::SwiftError))
    Flags.setSwiftError();
}

ISD::ArgFlagsTy CallLowering::getAttributesForArgIdx(const CallBase &Call,
                                                     unsigned ArgIdx) const {
  ISD::ArgFlagsTy Flags;
  addFlagsUsingAttrFn(Flags, [&Call, &ArgIdx](Attribute::AttrKind Attr) {
    return Call.paramHasAttr(ArgIdx, Attr);
  });
  return Flags;
}

void CallLowering::addArgFlagsFromAttributes(ISD::ArgFlagsTy &Flags,
                                             const AttributeList &Attrs,
                                             unsigned OpIdx) const {
  addFlagsUsingAttrFn(Flags, [&Attrs, &OpIdx](Attribute::AttrKind Attr) {
    return Attrs.hasAttribute(OpIdx, Attr);
  });
}

bool CallLowering::lowerCall(MachineIRBuilder &MIRBuilder, const CallBase &CB,
                             ArrayRef<Register> ResRegs,
                             ArrayRef<ArrayRef<Register>> ArgRegs,
                             Register SwiftErrorVReg,
                             std::function<unsigned()> GetCalleeReg) const {
  CallLoweringInfo Info;
  const DataLayout &DL = MIRBuilder.getDataLayout();
  MachineFunction &MF = MIRBuilder.getMF();
  bool CanBeTailCalled = CB.isTailCall() &&
                         isInTailCallPosition(CB, MF.getTarget()) &&
                         (MF.getFunction()
                              .getFnAttribute("disable-tail-calls")
                              .getValueAsString() != "true");

  CallingConv::ID CallConv = CB.getCallingConv();
  Type *RetTy = CB.getType();
  bool IsVarArg = CB.getFunctionType()->isVarArg();

  SmallVector<BaseArgInfo, 4> SplitArgs;
  getReturnInfo(CallConv, RetTy, CB.getAttributes(), SplitArgs, DL);
  Info.CanLowerReturn = canLowerReturn(MF, CallConv, SplitArgs, IsVarArg);

  if (!Info.CanLowerReturn) {
    // Callee requires sret demotion.
    insertSRetOutgoingArgument(MIRBuilder, CB, Info);

    // The sret demotion isn't compatible with tail-calls, since the sret
    // argument points into the caller's stack frame.
    CanBeTailCalled = false;
  }

  // First step is to marshall all the function's parameters into the correct
  // physregs and memory locations. Gather the sequence of argument types that
  // we'll pass to the assigner function.
  unsigned i = 0;
  unsigned NumFixedArgs = CB.getFunctionType()->getNumParams();
  for (auto &Arg : CB.args()) {
    ArgInfo OrigArg{ArgRegs[i], Arg->getType(), getAttributesForArgIdx(CB, i),
                    i < NumFixedArgs};
    setArgFlags(OrigArg, i + AttributeList::FirstArgIndex, DL, CB);

    // If we have an explicit sret argument that is an Instruction, (i.e., it
    // might point to function-local memory), we can't meaningfully tail-call.
    if (OrigArg.Flags[0].isSRet() && isa<Instruction>(&Arg))
      CanBeTailCalled = false;

    Info.OrigArgs.push_back(OrigArg);
    ++i;
  }

  // Try looking through a bitcast from one function type to another.
  // Commonly happens with calls to objc_msgSend().
  const Value *CalleeV = CB.getCalledOperand()->stripPointerCasts();
  if (const Function *F = dyn_cast<Function>(CalleeV))
    Info.Callee = MachineOperand::CreateGA(F, 0);
  else
    Info.Callee = MachineOperand::CreateReg(GetCalleeReg(), false);

  Info.OrigRet = ArgInfo{ResRegs, RetTy, ISD::ArgFlagsTy{}};
  if (!Info.OrigRet.Ty->isVoidTy())
    setArgFlags(Info.OrigRet, AttributeList::ReturnIndex, DL, CB);

  Info.KnownCallees = CB.getMetadata(LLVMContext::MD_callees);
  Info.CallConv = CallConv;
  Info.SwiftErrorVReg = SwiftErrorVReg;
  Info.IsMustTailCall = CB.isMustTailCall();
  Info.IsTailCall = CanBeTailCalled;
  Info.IsVarArg = IsVarArg;
  return lowerCall(MIRBuilder, Info);
}

template <typename FuncInfoTy>
void CallLowering::setArgFlags(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                               const DataLayout &DL,
                               const FuncInfoTy &FuncInfo) const {
  auto &Flags = Arg.Flags[0];
  const AttributeList &Attrs = FuncInfo.getAttributes();
  addArgFlagsFromAttributes(Flags, Attrs, OpIdx);

  if (Flags.isByVal() || Flags.isInAlloca() || Flags.isPreallocated()) {
    Type *ElementTy = cast<PointerType>(Arg.Ty)->getElementType();

    auto Ty = Attrs.getAttribute(OpIdx, Attribute::ByVal).getValueAsType();
    Flags.setByValSize(DL.getTypeAllocSize(Ty ? Ty : ElementTy));

    // For ByVal, alignment should be passed from FE.  BE will guess if
    // this info is not there but there are cases it cannot get right.
    Align FrameAlign;
    if (auto ParamAlign = FuncInfo.getParamAlign(OpIdx - 2))
      FrameAlign = *ParamAlign;
    else
      FrameAlign = Align(getTLI()->getByValTypeAlignment(ElementTy, DL));
    Flags.setByValAlign(FrameAlign);
  }
  Flags.setOrigAlign(DL.getABITypeAlign(Arg.Ty));

  // Don't try to use the returned attribute if the argument is marked as
  // swiftself, since it won't be passed in x0.
  if (Flags.isSwiftSelf())
    Flags.setReturned(false);
}

template void
CallLowering::setArgFlags<Function>(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                                    const DataLayout &DL,
                                    const Function &FuncInfo) const;

template void
CallLowering::setArgFlags<CallBase>(CallLowering::ArgInfo &Arg, unsigned OpIdx,
                                    const DataLayout &DL,
                                    const CallBase &FuncInfo) const;

void CallLowering::splitToValueTypes(const ArgInfo &OrigArg,
                                     SmallVectorImpl<ArgInfo> &SplitArgs,
                                     const DataLayout &DL,
                                     CallingConv::ID CallConv) const {
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(*TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  if (SplitVTs.size() == 0)
    return;

  if (SplitVTs.size() == 1) {
    // No splitting to do, but we want to replace the original type (e.g. [1 x
    // double] -> double).
    SplitArgs.emplace_back(OrigArg.Regs[0], SplitVTs[0].getTypeForEVT(Ctx),
                           OrigArg.Flags[0], OrigArg.IsFixed);
    return;
  }

  // Create one ArgInfo for each virtual register in the original ArgInfo.
  assert(OrigArg.Regs.size() == SplitVTs.size() && "Regs / types mismatch");

  bool NeedsRegBlock = TLI->functionArgumentNeedsConsecutiveRegisters(
      OrigArg.Ty, CallConv, false);
  for (unsigned i = 0, e = SplitVTs.size(); i < e; ++i) {
    Type *SplitTy = SplitVTs[i].getTypeForEVT(Ctx);
    SplitArgs.emplace_back(OrigArg.Regs[i], SplitTy, OrigArg.Flags[0],
                           OrigArg.IsFixed);
    if (NeedsRegBlock)
      SplitArgs.back().Flags[0].setInConsecutiveRegs();
  }

  SplitArgs.back().Flags[0].setInConsecutiveRegsLast();
}

void CallLowering::unpackRegs(ArrayRef<Register> DstRegs, Register SrcReg,
                              Type *PackedTy,
                              MachineIRBuilder &MIRBuilder) const {
  assert(DstRegs.size() > 1 && "Nothing to unpack");

  const DataLayout &DL = MIRBuilder.getDataLayout();

  SmallVector<LLT, 8> LLTs;
  SmallVector<uint64_t, 8> Offsets;
  computeValueLLTs(DL, *PackedTy, LLTs, &Offsets);
  assert(LLTs.size() == DstRegs.size() && "Regs / types mismatch");

  for (unsigned i = 0; i < DstRegs.size(); ++i)
    MIRBuilder.buildExtract(DstRegs[i], SrcReg, Offsets[i]);
}

/// Pack values \p SrcRegs to cover the vector type result \p DstRegs.
static MachineInstrBuilder
mergeVectorRegsToResultRegs(MachineIRBuilder &B, ArrayRef<Register> DstRegs,
                            ArrayRef<Register> SrcRegs) {
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT LLTy = MRI.getType(DstRegs[0]);
  LLT PartLLT = MRI.getType(SrcRegs[0]);

  // Deal with v3s16 split into v2s16
  LLT LCMTy = getLCMType(LLTy, PartLLT);
  if (LCMTy == LLTy) {
    // Common case where no padding is needed.
    assert(DstRegs.size() == 1);
    return B.buildConcatVectors(DstRegs[0], SrcRegs);
  }

  // We need to create an unmerge to the result registers, which may require
  // widening the original value.
  Register UnmergeSrcReg;
  if (LCMTy != PartLLT) {
    // e.g. A <3 x s16> value was split to <2 x s16>
    // %register_value0:_(<2 x s16>)
    // %register_value1:_(<2 x s16>)
    // %undef:_(<2 x s16>) = G_IMPLICIT_DEF
    // %concat:_<6 x s16>) = G_CONCAT_VECTORS %reg_value0, %reg_value1, %undef
    // %dst_reg:_(<3 x s16>), %dead:_(<3 x s16>) = G_UNMERGE_VALUES %concat
    const int NumWide = LCMTy.getSizeInBits() / PartLLT.getSizeInBits();
    Register Undef = B.buildUndef(PartLLT).getReg(0);

    // Build vector of undefs.
    SmallVector<Register, 8> WidenedSrcs(NumWide, Undef);

    // Replace the first sources with the real registers.
    std::copy(SrcRegs.begin(), SrcRegs.end(), WidenedSrcs.begin());
    UnmergeSrcReg = B.buildConcatVectors(LCMTy, WidenedSrcs).getReg(0);
  } else {
    // We don't need to widen anything if we're extracting a scalar which was
    // promoted to a vector e.g. s8 -> v4s8 -> s8
    assert(SrcRegs.size() == 1);
    UnmergeSrcReg = SrcRegs[0];
  }

  int NumDst = LCMTy.getSizeInBits() / LLTy.getSizeInBits();

  SmallVector<Register, 8> PadDstRegs(NumDst);
  std::copy(DstRegs.begin(), DstRegs.end(), PadDstRegs.begin());

  // Create the excess dead defs for the unmerge.
  for (int I = DstRegs.size(); I != NumDst; ++I)
    PadDstRegs[I] = MRI.createGenericVirtualRegister(LLTy);

  return B.buildUnmerge(PadDstRegs, UnmergeSrcReg);
}

/// Create a sequence of instructions to combine pieces split into register
/// typed values to the original IR value. \p OrigRegs contains the destination
/// value registers of type \p LLTy, and \p Regs contains the legalized pieces
/// with type \p PartLLT. This is used for incoming values (physregs to vregs).
static void buildCopyFromRegs(MachineIRBuilder &B, ArrayRef<Register> OrigRegs,
                              ArrayRef<Register> Regs, LLT LLTy, LLT PartLLT) {
  MachineRegisterInfo &MRI = *B.getMRI();

  // We could just insert a regular copy, but this is unreachable at the moment.
  assert(LLTy != PartLLT && "identical part types shouldn't reach here");

  if (PartLLT.isVector() == LLTy.isVector() &&
      PartLLT.getScalarSizeInBits() > LLTy.getScalarSizeInBits()) {
    assert(OrigRegs.size() == 1 && Regs.size() == 1);
    B.buildTrunc(OrigRegs[0], Regs[0]);
    return;
  }

  if (!LLTy.isVector() && !PartLLT.isVector()) {
    assert(OrigRegs.size() == 1);
    LLT OrigTy = MRI.getType(OrigRegs[0]);

    unsigned SrcSize = PartLLT.getSizeInBits() * Regs.size();
    if (SrcSize == OrigTy.getSizeInBits())
      B.buildMerge(OrigRegs[0], Regs);
    else {
      auto Widened = B.buildMerge(LLT::scalar(SrcSize), Regs);
      B.buildTrunc(OrigRegs[0], Widened);
    }

    return;
  }

  if (PartLLT.isVector()) {
    assert(OrigRegs.size() == 1 &&
           LLTy.getScalarType() == PartLLT.getElementType());
    mergeVectorRegsToResultRegs(B, OrigRegs, Regs);
    return;
  }

  assert(LLTy.isVector() && !PartLLT.isVector());

  LLT DstEltTy = LLTy.getElementType();

  // Pointer information was discarded. We'll need to coerce some register types
  // to avoid violating type constraints.
  LLT RealDstEltTy = MRI.getType(OrigRegs[0]).getElementType();

  assert(DstEltTy.getSizeInBits() == RealDstEltTy.getSizeInBits());

  if (DstEltTy == PartLLT) {
    // Vector was trivially scalarized.

    if (RealDstEltTy.isPointer()) {
      for (Register Reg : Regs)
        MRI.setType(Reg, RealDstEltTy);
    }

    B.buildBuildVector(OrigRegs[0], Regs);
  } else if (DstEltTy.getSizeInBits() > PartLLT.getSizeInBits()) {
    // Deal with vector with 64-bit elements decomposed to 32-bit
    // registers. Need to create intermediate 64-bit elements.
    SmallVector<Register, 8> EltMerges;
    int PartsPerElt = DstEltTy.getSizeInBits() / PartLLT.getSizeInBits();

    assert(DstEltTy.getSizeInBits() % PartLLT.getSizeInBits() == 0);

    for (int I = 0, NumElts = LLTy.getNumElements(); I != NumElts; ++I) {
      auto Merge = B.buildMerge(RealDstEltTy, Regs.take_front(PartsPerElt));
      // Fix the type in case this is really a vector of pointers.
      MRI.setType(Merge.getReg(0), RealDstEltTy);
      EltMerges.push_back(Merge.getReg(0));
      Regs = Regs.drop_front(PartsPerElt);
    }

    B.buildBuildVector(OrigRegs[0], EltMerges);
  } else {
    // Vector was split, and elements promoted to a wider type.
    // FIXME: Should handle floating point promotions.
    LLT BVType = LLT::vector(LLTy.getNumElements(), PartLLT);
    auto BV = B.buildBuildVector(BVType, Regs);
    B.buildTrunc(OrigRegs[0], BV);
  }
}

/// Create a sequence of instructions to expand the value in \p SrcReg (of type
/// \p SrcTy) to the types in \p DstRegs (of type \p PartTy). \p ExtendOp should
/// contain the type of scalar value extension if necessary.
///
/// This is used for outgoing values (vregs to physregs)
static void buildCopyToRegs(MachineIRBuilder &B, ArrayRef<Register> DstRegs,
                            Register SrcReg, LLT SrcTy, LLT PartTy,
                            unsigned ExtendOp = TargetOpcode::G_ANYEXT) {
  // We could just insert a regular copy, but this is unreachable at the moment.
  assert(SrcTy != PartTy && "identical part types shouldn't reach here");

  const unsigned PartSize = PartTy.getSizeInBits();

  if (PartTy.isVector() == SrcTy.isVector() &&
      PartTy.getScalarSizeInBits() > SrcTy.getScalarSizeInBits()) {
    assert(DstRegs.size() == 1);
    B.buildInstr(ExtendOp, {DstRegs[0]}, {SrcReg});
    return;
  }

  if (SrcTy.isVector() && !PartTy.isVector() &&
      PartSize > SrcTy.getElementType().getSizeInBits()) {
    // Vector was scalarized, and the elements extended.
    auto UnmergeToEltTy = B.buildUnmerge(SrcTy.getElementType(), SrcReg);
    for (int i = 0, e = DstRegs.size(); i != e; ++i)
      B.buildAnyExt(DstRegs[i], UnmergeToEltTy.getReg(i));
    return;
  }

  LLT GCDTy = getGCDType(SrcTy, PartTy);
  if (GCDTy == PartTy) {
    // If this already evenly divisible, we can create a simple unmerge.
    B.buildUnmerge(DstRegs, SrcReg);
    return;
  }

  MachineRegisterInfo &MRI = *B.getMRI();
  LLT DstTy = MRI.getType(DstRegs[0]);
  LLT LCMTy = getLCMType(SrcTy, PartTy);

  const unsigned LCMSize = LCMTy.getSizeInBits();
  const unsigned DstSize = DstTy.getSizeInBits();
  const unsigned SrcSize = SrcTy.getSizeInBits();

  Register UnmergeSrc = SrcReg;
  if (LCMSize != SrcSize) {
    // Widen to the common type.
    Register Undef = B.buildUndef(SrcTy).getReg(0);
    SmallVector<Register, 8> MergeParts(1, SrcReg);
    for (unsigned Size = SrcSize; Size != LCMSize; Size += SrcSize)
      MergeParts.push_back(Undef);

    UnmergeSrc = B.buildMerge(LCMTy, MergeParts).getReg(0);
  }

  // Unmerge to the original registers and pad with dead defs.
  SmallVector<Register, 8> UnmergeResults(DstRegs.begin(), DstRegs.end());
  for (unsigned Size = DstSize * DstRegs.size(); Size != LCMSize;
       Size += DstSize) {
    UnmergeResults.push_back(MRI.createGenericVirtualRegister(DstTy));
  }

  B.buildUnmerge(UnmergeResults, UnmergeSrc);
}

bool CallLowering::handleAssignments(MachineIRBuilder &MIRBuilder,
                                     SmallVectorImpl<ArgInfo> &Args,
                                     ValueHandler &Handler,
                                     CallingConv::ID CallConv, bool IsVarArg,
                                     Register ThisReturnReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  SmallVector<CCValAssign, 16> ArgLocs;

  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, F.getContext());
  return handleAssignments(CCInfo, ArgLocs, MIRBuilder, Args, Handler,
                           ThisReturnReg);
}

static unsigned extendOpFromFlags(llvm::ISD::ArgFlagsTy Flags) {
  if (Flags.isSExt())
    return TargetOpcode::G_SEXT;
  if (Flags.isZExt())
    return TargetOpcode::G_ZEXT;
  return TargetOpcode::G_ANYEXT;
}

bool CallLowering::handleAssignments(CCState &CCInfo,
                                     SmallVectorImpl<CCValAssign> &ArgLocs,
                                     MachineIRBuilder &MIRBuilder,
                                     SmallVectorImpl<ArgInfo> &Args,
                                     ValueHandler &Handler,
                                     Register ThisReturnReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();

  unsigned NumArgs = Args.size();
  for (unsigned i = 0; i != NumArgs; ++i) {
    EVT CurVT = EVT::getEVT(Args[i].Ty);
    if (CurVT.isSimple() &&
        !Handler.assignArg(i, CurVT.getSimpleVT(), CurVT.getSimpleVT(),
                           CCValAssign::Full, Args[i], Args[i].Flags[0],
                           CCInfo))
      continue;

    MVT NewVT = TLI->getRegisterTypeForCallingConv(
        F.getContext(), CCInfo.getCallingConv(), EVT(CurVT));

    // If we need to split the type over multiple regs, check it's a scenario
    // we currently support.
    unsigned NumParts = TLI->getNumRegistersForCallingConv(
        F.getContext(), CCInfo.getCallingConv(), CurVT);

    if (NumParts == 1) {
      // Try to use the register type if we couldn't assign the VT.
      if (Handler.assignArg(i, NewVT, NewVT, CCValAssign::Full, Args[i],
                            Args[i].Flags[0], CCInfo))
        return false;

      // If we couldn't directly assign this part, some casting may be
      // necessary. Create the new register, but defer inserting the conversion
      // instructions.
      assert(Args[i].OrigRegs.empty());
      Args[i].OrigRegs.push_back(Args[i].Regs[0]);
      assert(Args[i].Regs.size() == 1);

      const LLT VATy(NewVT);
      Args[i].Regs[0] = MRI.createGenericVirtualRegister(VATy);
      continue;
    }

    const LLT NewLLT(NewVT);

    // For incoming arguments (physregs to vregs), we could have values in
    // physregs (or memlocs) which we want to extract and copy to vregs.
    // During this, we might have to deal with the LLT being split across
    // multiple regs, so we have to record this information for later.
    //
    // If we have outgoing args, then we have the opposite case. We have a
    // vreg with an LLT which we want to assign to a physical location, and
    // we might have to record that the value has to be split later.
    if (Handler.isIncomingArgumentHandler()) {
      // We're handling an incoming arg which is split over multiple regs.
      // E.g. passing an s128 on AArch64.
      ISD::ArgFlagsTy OrigFlags = Args[i].Flags[0];
      Args[i].OrigRegs.push_back(Args[i].Regs[0]);
      Args[i].Regs.clear();
      Args[i].Flags.clear();
      // For each split register, create and assign a vreg that will store
      // the incoming component of the larger value. These will later be
      // merged to form the final vreg.
      for (unsigned Part = 0; Part < NumParts; ++Part) {
        Register Reg = MRI.createGenericVirtualRegister(NewLLT);
        ISD::ArgFlagsTy Flags = OrigFlags;
        if (Part == 0) {
          Flags.setSplit();
        } else {
          Flags.setOrigAlign(Align(1));
          if (Part == NumParts - 1)
            Flags.setSplitEnd();
        }
        Args[i].Regs.push_back(Reg);
        Args[i].Flags.push_back(Flags);
        if (Handler.assignArg(i, NewVT, NewVT, CCValAssign::Full, Args[i],
                              Args[i].Flags[Part], CCInfo)) {
          // Still couldn't assign this smaller part type for some reason.
          return false;
        }
      }
    } else {
      assert(Args[i].Regs.size() == 1);

      // This type is passed via multiple registers in the calling convention.
      // We need to extract the individual parts.
      assert(Args[i].OrigRegs.empty());
      Args[i].OrigRegs.push_back(Args[i].Regs[0]);

      ISD::ArgFlagsTy OrigFlags = Args[i].Flags[0];
      // We're going to replace the regs and flags with the split ones.
      Args[i].Regs.clear();
      Args[i].Flags.clear();
      for (unsigned PartIdx = 0; PartIdx < NumParts; ++PartIdx) {
        ISD::ArgFlagsTy Flags = OrigFlags;
        if (PartIdx == 0) {
          Flags.setSplit();
        } else {
          Flags.setOrigAlign(Align(1));
          if (PartIdx == NumParts - 1)
            Flags.setSplitEnd();
        }

        // TODO: Also check if there is a valid extension that preserves the
        // bits. However currently this call lowering doesn't support non-exact
        // split parts, so that can't be tested.
        if (OrigFlags.isReturned() &&
            (NumParts * NewVT.getSizeInBits() != CurVT.getSizeInBits())) {
          Flags.setReturned(false);
        }

        Register NewReg = MRI.createGenericVirtualRegister(NewLLT);

        Args[i].Regs.push_back(NewReg);
        Args[i].Flags.push_back(Flags);
        if (Handler.assignArg(i, NewVT, NewVT, CCValAssign::Full,
                              Args[i], Args[i].Flags[PartIdx], CCInfo))
          return false;
      }
    }
  }

  for (unsigned i = 0, e = Args.size(), j = 0; i != e; ++i, ++j) {
    assert(j < ArgLocs.size() && "Skipped too many arg locs");

    CCValAssign &VA = ArgLocs[j];
    assert(VA.getValNo() == i && "Location doesn't correspond to current arg");

    if (VA.needsCustom()) {
      unsigned NumArgRegs =
          Handler.assignCustomValue(Args[i], makeArrayRef(ArgLocs).slice(j));
      if (!NumArgRegs)
        return false;
      j += NumArgRegs;
      continue;
    }

    EVT VAVT = VA.getValVT();
    const LLT OrigTy = getLLTForType(*Args[i].Ty, DL);
    const LLT VATy(VAVT.getSimpleVT());

    // Expected to be multiple regs for a single incoming arg.
    // There should be Regs.size() ArgLocs per argument.
    unsigned NumArgRegs = Args[i].Regs.size();
    assert((j + (NumArgRegs - 1)) < ArgLocs.size() &&
           "Too many regs for number of args");

    // Coerce into outgoing value types before register assignment.
    if (!Handler.isIncomingArgumentHandler() && OrigTy != VATy) {
      assert(Args[i].OrigRegs.size() == 1);
      buildCopyToRegs(MIRBuilder, Args[i].Regs, Args[i].OrigRegs[0], OrigTy,
                      VATy, extendOpFromFlags(Args[i].Flags[0]));
    }

    for (unsigned Part = 0; Part < NumArgRegs; ++Part) {
      Register ArgReg = Args[i].Regs[Part];
      // There should be Regs.size() ArgLocs per argument.
      VA = ArgLocs[j + Part];
      if (VA.isMemLoc()) {
        // Individual pieces may have been spilled to the stack and others
        // passed in registers.

        // FIXME: Use correct address space for pointer size
        EVT LocVT = VA.getValVT();
        unsigned MemSize = LocVT == MVT::iPTR ? DL.getPointerSize()
                                              : LocVT.getStoreSize();
        unsigned Offset = VA.getLocMemOffset();
        MachinePointerInfo MPO;
        Register StackAddr = Handler.getStackAddress(MemSize, Offset, MPO);
        Handler.assignValueToAddress(Args[i], Part, StackAddr, MemSize, MPO,
                                     VA);
        continue;
      }

      assert(VA.isRegLoc() && "custom loc should have been handled already");

      if (i == 0 && ThisReturnReg.isValid() &&
          Handler.isIncomingArgumentHandler() &&
          isTypeIsValidForThisReturn(VAVT)) {
        Handler.assignValueToReg(Args[i].Regs[i], ThisReturnReg, VA);
        continue;
      }

      Handler.assignValueToReg(ArgReg, VA.getLocReg(), VA);
    }

    // Now that all pieces have been assigned, re-pack the register typed values
    // into the original value typed registers.
    if (Handler.isIncomingArgumentHandler() && OrigTy != VATy) {
      // Merge the split registers into the expected larger result vregs of
      // the original call.
      buildCopyFromRegs(MIRBuilder, Args[i].OrigRegs, Args[i].Regs, OrigTy,
                        VATy);
    }

    j += NumArgRegs - 1;
  }

  return true;
}

void CallLowering::insertSRetLoads(MachineIRBuilder &MIRBuilder, Type *RetTy,
                                   ArrayRef<Register> VRegs, Register DemoteReg,
                                   int FI) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const DataLayout &DL = MF.getDataLayout();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(*TLI, DL, RetTy, SplitVTs, &Offsets, 0);

  assert(VRegs.size() == SplitVTs.size());

  unsigned NumValues = SplitVTs.size();
  Align BaseAlign = DL.getPrefTypeAlign(RetTy);
  Type *RetPtrTy = RetTy->getPointerTo(DL.getAllocaAddrSpace());
  LLT OffsetLLTy = getLLTForType(*DL.getIntPtrType(RetPtrTy), DL);

  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);

  for (unsigned I = 0; I < NumValues; ++I) {
    Register Addr;
    MIRBuilder.materializePtrAdd(Addr, DemoteReg, OffsetLLTy, Offsets[I]);
    auto *MMO = MF.getMachineMemOperand(PtrInfo, MachineMemOperand::MOLoad,
                                        MRI.getType(VRegs[I]).getSizeInBytes(),
                                        commonAlignment(BaseAlign, Offsets[I]));
    MIRBuilder.buildLoad(VRegs[I], Addr, *MMO);
  }
}

void CallLowering::insertSRetStores(MachineIRBuilder &MIRBuilder, Type *RetTy,
                                    ArrayRef<Register> VRegs,
                                    Register DemoteReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const DataLayout &DL = MF.getDataLayout();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(*TLI, DL, RetTy, SplitVTs, &Offsets, 0);

  assert(VRegs.size() == SplitVTs.size());

  unsigned NumValues = SplitVTs.size();
  Align BaseAlign = DL.getPrefTypeAlign(RetTy);
  unsigned AS = DL.getAllocaAddrSpace();
  LLT OffsetLLTy =
      getLLTForType(*DL.getIntPtrType(RetTy->getPointerTo(AS)), DL);

  MachinePointerInfo PtrInfo(AS);

  for (unsigned I = 0; I < NumValues; ++I) {
    Register Addr;
    MIRBuilder.materializePtrAdd(Addr, DemoteReg, OffsetLLTy, Offsets[I]);
    auto *MMO = MF.getMachineMemOperand(PtrInfo, MachineMemOperand::MOStore,
                                        MRI.getType(VRegs[I]).getSizeInBytes(),
                                        commonAlignment(BaseAlign, Offsets[I]));
    MIRBuilder.buildStore(VRegs[I], Addr, *MMO);
  }
}

void CallLowering::insertSRetIncomingArgument(
    const Function &F, SmallVectorImpl<ArgInfo> &SplitArgs, Register &DemoteReg,
    MachineRegisterInfo &MRI, const DataLayout &DL) const {
  unsigned AS = DL.getAllocaAddrSpace();
  DemoteReg = MRI.createGenericVirtualRegister(
      LLT::pointer(AS, DL.getPointerSizeInBits(AS)));

  Type *PtrTy = PointerType::get(F.getReturnType(), AS);

  SmallVector<EVT, 1> ValueVTs;
  ComputeValueVTs(*TLI, DL, PtrTy, ValueVTs);

  // NOTE: Assume that a pointer won't get split into more than one VT.
  assert(ValueVTs.size() == 1);

  ArgInfo DemoteArg(DemoteReg, ValueVTs[0].getTypeForEVT(PtrTy->getContext()));
  setArgFlags(DemoteArg, AttributeList::ReturnIndex, DL, F);
  DemoteArg.Flags[0].setSRet();
  SplitArgs.insert(SplitArgs.begin(), DemoteArg);
}

void CallLowering::insertSRetOutgoingArgument(MachineIRBuilder &MIRBuilder,
                                              const CallBase &CB,
                                              CallLoweringInfo &Info) const {
  const DataLayout &DL = MIRBuilder.getDataLayout();
  Type *RetTy = CB.getType();
  unsigned AS = DL.getAllocaAddrSpace();
  LLT FramePtrTy = LLT::pointer(AS, DL.getPointerSizeInBits(AS));

  int FI = MIRBuilder.getMF().getFrameInfo().CreateStackObject(
      DL.getTypeAllocSize(RetTy), DL.getPrefTypeAlign(RetTy), false);

  Register DemoteReg = MIRBuilder.buildFrameIndex(FramePtrTy, FI).getReg(0);
  ArgInfo DemoteArg(DemoteReg, PointerType::get(RetTy, AS));
  setArgFlags(DemoteArg, AttributeList::ReturnIndex, DL, CB);
  DemoteArg.Flags[0].setSRet();

  Info.OrigArgs.insert(Info.OrigArgs.begin(), DemoteArg);
  Info.DemoteStackIndex = FI;
  Info.DemoteRegister = DemoteReg;
}

bool CallLowering::checkReturn(CCState &CCInfo,
                               SmallVectorImpl<BaseArgInfo> &Outs,
                               CCAssignFn *Fn) const {
  for (unsigned I = 0, E = Outs.size(); I < E; ++I) {
    MVT VT = MVT::getVT(Outs[I].Ty);
    if (Fn(I, VT, VT, CCValAssign::Full, Outs[I].Flags[0], CCInfo))
      return false;
  }
  return true;
}

void CallLowering::getReturnInfo(CallingConv::ID CallConv, Type *RetTy,
                                 AttributeList Attrs,
                                 SmallVectorImpl<BaseArgInfo> &Outs,
                                 const DataLayout &DL) const {
  LLVMContext &Context = RetTy->getContext();
  ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();

  SmallVector<EVT, 4> SplitVTs;
  ComputeValueVTs(*TLI, DL, RetTy, SplitVTs);
  addArgFlagsFromAttributes(Flags, Attrs, AttributeList::ReturnIndex);

  for (EVT VT : SplitVTs) {
    unsigned NumParts =
        TLI->getNumRegistersForCallingConv(Context, CallConv, VT);
    MVT RegVT = TLI->getRegisterTypeForCallingConv(Context, CallConv, VT);
    Type *PartTy = EVT(RegVT).getTypeForEVT(Context);

    for (unsigned I = 0; I < NumParts; ++I) {
      Outs.emplace_back(PartTy, Flags);
    }
  }
}

bool CallLowering::checkReturnTypeForCallConv(MachineFunction &MF) const {
  const auto &F = MF.getFunction();
  Type *ReturnType = F.getReturnType();
  CallingConv::ID CallConv = F.getCallingConv();

  SmallVector<BaseArgInfo, 4> SplitArgs;
  getReturnInfo(CallConv, ReturnType, F.getAttributes(), SplitArgs,
                MF.getDataLayout());
  return canLowerReturn(MF, CallConv, SplitArgs, F.isVarArg());
}

bool CallLowering::analyzeArgInfo(CCState &CCState,
                                  SmallVectorImpl<ArgInfo> &Args,
                                  CCAssignFn &AssignFnFixed,
                                  CCAssignFn &AssignFnVarArg) const {
  for (unsigned i = 0, e = Args.size(); i < e; ++i) {
    MVT VT = MVT::getVT(Args[i].Ty);
    CCAssignFn &Fn = Args[i].IsFixed ? AssignFnFixed : AssignFnVarArg;
    if (Fn(i, VT, VT, CCValAssign::Full, Args[i].Flags[0], CCState)) {
      // Bail out on anything we can't handle.
      LLVM_DEBUG(dbgs() << "Cannot analyze " << EVT(VT).getEVTString()
                        << " (arg number = " << i << "\n");
      return false;
    }
  }
  return true;
}

bool CallLowering::parametersInCSRMatch(
    const MachineRegisterInfo &MRI, const uint32_t *CallerPreservedMask,
    const SmallVectorImpl<CCValAssign> &OutLocs,
    const SmallVectorImpl<ArgInfo> &OutArgs) const {
  for (unsigned i = 0; i < OutLocs.size(); ++i) {
    auto &ArgLoc = OutLocs[i];
    // If it's not a register, it's fine.
    if (!ArgLoc.isRegLoc())
      continue;

    MCRegister PhysReg = ArgLoc.getLocReg();

    // Only look at callee-saved registers.
    if (MachineOperand::clobbersPhysReg(CallerPreservedMask, PhysReg))
      continue;

    LLVM_DEBUG(
        dbgs()
        << "... Call has an argument passed in a callee-saved register.\n");

    // Check if it was copied from.
    const ArgInfo &OutInfo = OutArgs[i];

    if (OutInfo.Regs.size() > 1) {
      LLVM_DEBUG(
          dbgs() << "... Cannot handle arguments in multiple registers.\n");
      return false;
    }

    // Check if we copy the register, walking through copies from virtual
    // registers. Note that getDefIgnoringCopies does not ignore copies from
    // physical registers.
    MachineInstr *RegDef = getDefIgnoringCopies(OutInfo.Regs[0], MRI);
    if (!RegDef || RegDef->getOpcode() != TargetOpcode::COPY) {
      LLVM_DEBUG(
          dbgs()
          << "... Parameter was not copied into a VReg, cannot tail call.\n");
      return false;
    }

    // Got a copy. Verify that it's the same as the register we want.
    Register CopyRHS = RegDef->getOperand(1).getReg();
    if (CopyRHS != PhysReg) {
      LLVM_DEBUG(dbgs() << "... Callee-saved register was not copied into "
                           "VReg, cannot tail call.\n");
      return false;
    }
  }

  return true;
}

bool CallLowering::resultsCompatible(CallLoweringInfo &Info,
                                     MachineFunction &MF,
                                     SmallVectorImpl<ArgInfo> &InArgs,
                                     CCAssignFn &CalleeAssignFnFixed,
                                     CCAssignFn &CalleeAssignFnVarArg,
                                     CCAssignFn &CallerAssignFnFixed,
                                     CCAssignFn &CallerAssignFnVarArg) const {
  const Function &F = MF.getFunction();
  CallingConv::ID CalleeCC = Info.CallConv;
  CallingConv::ID CallerCC = F.getCallingConv();

  if (CallerCC == CalleeCC)
    return true;

  SmallVector<CCValAssign, 16> ArgLocs1;
  CCState CCInfo1(CalleeCC, false, MF, ArgLocs1, F.getContext());
  if (!analyzeArgInfo(CCInfo1, InArgs, CalleeAssignFnFixed,
                      CalleeAssignFnVarArg))
    return false;

  SmallVector<CCValAssign, 16> ArgLocs2;
  CCState CCInfo2(CallerCC, false, MF, ArgLocs2, F.getContext());
  if (!analyzeArgInfo(CCInfo2, InArgs, CallerAssignFnFixed,
                      CalleeAssignFnVarArg))
    return false;

  // We need the argument locations to match up exactly. If there's more in
  // one than the other, then we are done.
  if (ArgLocs1.size() != ArgLocs2.size())
    return false;

  // Make sure that each location is passed in exactly the same way.
  for (unsigned i = 0, e = ArgLocs1.size(); i < e; ++i) {
    const CCValAssign &Loc1 = ArgLocs1[i];
    const CCValAssign &Loc2 = ArgLocs2[i];

    // We need both of them to be the same. So if one is a register and one
    // isn't, we're done.
    if (Loc1.isRegLoc() != Loc2.isRegLoc())
      return false;

    if (Loc1.isRegLoc()) {
      // If they don't have the same register location, we're done.
      if (Loc1.getLocReg() != Loc2.getLocReg())
        return false;

      // They matched, so we can move to the next ArgLoc.
      continue;
    }

    // Loc1 wasn't a RegLoc, so they both must be MemLocs. Check if they match.
    if (Loc1.getLocMemOffset() != Loc2.getLocMemOffset())
      return false;
  }

  return true;
}

Register CallLowering::ValueHandler::extendRegister(Register ValReg,
                                                    CCValAssign &VA,
                                                    unsigned MaxSizeBits) {
  LLT LocTy{VA.getLocVT()};
  LLT ValTy = MRI.getType(ValReg);
  if (LocTy.getSizeInBits() == ValTy.getSizeInBits())
    return ValReg;

  if (LocTy.isScalar() && MaxSizeBits && MaxSizeBits < LocTy.getSizeInBits()) {
    if (MaxSizeBits <= ValTy.getSizeInBits())
      return ValReg;
    LocTy = LLT::scalar(MaxSizeBits);
  }

  switch (VA.getLocInfo()) {
  default: break;
  case CCValAssign::Full:
  case CCValAssign::BCvt:
    // FIXME: bitconverting between vector types may or may not be a
    // nop in big-endian situations.
    return ValReg;
  case CCValAssign::AExt: {
    auto MIB = MIRBuilder.buildAnyExt(LocTy, ValReg);
    return MIB.getReg(0);
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
