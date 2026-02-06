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
#include "Utils/WasmAddressSpaces.h"
#include "WebAssemblyISelLowering.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "wasm-call-lowering"

using namespace llvm;

// Several of the following methods are internal utilities defined in
// CodeGen/GlobalIsel/CallLowering.cpp
// TODO: Find a better solution?

// Internal utility from CallLowering.cpp
static unsigned extendOpFromFlags(ISD::ArgFlagsTy Flags) {
  if (Flags.isSExt())
    return TargetOpcode::G_SEXT;
  if (Flags.isZExt())
    return TargetOpcode::G_ZEXT;
  return TargetOpcode::G_ANYEXT;
}

// Internal utility from CallLowering.cpp
/// Pack values \p SrcRegs to cover the vector type result \p DstRegs.
static MachineInstrBuilder
mergeVectorRegsToResultRegs(MachineIRBuilder &B, ArrayRef<Register> DstRegs,
                            ArrayRef<Register> SrcRegs) {
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT LLTy = MRI.getType(DstRegs[0]);
  LLT PartLLT = MRI.getType(SrcRegs[0]);

  // Deal with v3s16 split into v2s16
  LLT LCMTy = getCoverTy(LLTy, PartLLT);
  if (LCMTy == LLTy) {
    // Common case where no padding is needed.
    assert(DstRegs.size() == 1);
    return B.buildConcatVectors(DstRegs[0], SrcRegs);
  }

  // We need to create an unmerge to the result registers, which may require
  // widening the original value.
  Register UnmergeSrcReg;
  if (LCMTy != PartLLT) {
    assert(DstRegs.size() == 1);
    return B.buildDeleteTrailingVectorElements(
        DstRegs[0], B.buildMergeLikeInstr(LCMTy, SrcRegs));
  } else {
    // We don't need to widen anything if we're extracting a scalar which was
    // promoted to a vector e.g. s8 -> v4s8 -> s8
    assert(SrcRegs.size() == 1);
    UnmergeSrcReg = SrcRegs[0];
  }

  int NumDst = LCMTy.getSizeInBits() / LLTy.getSizeInBits();

  SmallVector<Register, 8> PadDstRegs(NumDst);
  llvm::copy(DstRegs, PadDstRegs.begin());

  // Create the excess dead defs for the unmerge.
  for (int I = DstRegs.size(); I != NumDst; ++I)
    PadDstRegs[I] = MRI.createGenericVirtualRegister(LLTy);

  if (PadDstRegs.size() == 1)
    return B.buildDeleteTrailingVectorElements(DstRegs[0], UnmergeSrcReg);
  return B.buildUnmerge(PadDstRegs, UnmergeSrcReg);
}

// Internal utility from CallLowering.cpp
/// Create a sequence of instructions to combine pieces split into register
/// typed values to the original IR value. \p OrigRegs contains the destination
/// value registers of type \p LLTy, and \p Regs contains the legalized pieces
/// with type \p PartLLT. This is used for incoming values (physregs to vregs).

// Modified to account for floating-point extends/truncations
static void buildCopyFromRegs(MachineIRBuilder &B, ArrayRef<Register> OrigRegs,
                              ArrayRef<Register> Regs, LLT LLTy, LLT PartLLT,
                              const ISD::ArgFlagsTy Flags,
                              bool IsFloatingPoint) {
  MachineRegisterInfo &MRI = *B.getMRI();

  if (PartLLT == LLTy) {
    // We should have avoided introducing a new virtual register, and just
    // directly assigned here.
    assert(OrigRegs[0] == Regs[0]);
    return;
  }

  if (PartLLT.getSizeInBits() == LLTy.getSizeInBits() && OrigRegs.size() == 1 &&
      Regs.size() == 1) {
    B.buildBitcast(OrigRegs[0], Regs[0]);
    return;
  }

  // A vector PartLLT needs extending to LLTy's element size.
  // E.g. <2 x s64> = G_SEXT <2 x s32>.
  if (PartLLT.isVector() == LLTy.isVector() &&
      PartLLT.getScalarSizeInBits() > LLTy.getScalarSizeInBits() &&
      (!PartLLT.isVector() ||
       PartLLT.getElementCount() == LLTy.getElementCount()) &&
      OrigRegs.size() == 1 && Regs.size() == 1) {
    Register SrcReg = Regs[0];

    LLT LocTy = MRI.getType(SrcReg);

    if (Flags.isSExt()) {
      SrcReg = B.buildAssertSExt(LocTy, SrcReg, LLTy.getScalarSizeInBits())
                   .getReg(0);
    } else if (Flags.isZExt()) {
      SrcReg = B.buildAssertZExt(LocTy, SrcReg, LLTy.getScalarSizeInBits())
                   .getReg(0);
    }

    // Sometimes pointers are passed zero extended.
    LLT OrigTy = MRI.getType(OrigRegs[0]);
    if (OrigTy.isPointer()) {
      LLT IntPtrTy = LLT::scalar(OrigTy.getSizeInBits());
      B.buildIntToPtr(OrigRegs[0], B.buildTrunc(IntPtrTy, SrcReg));
      return;
    }

    if (IsFloatingPoint)
      B.buildFPTrunc(OrigRegs[0], SrcReg);
    else
      B.buildTrunc(OrigRegs[0], SrcReg);
    return;
  }

  if (!LLTy.isVector() && !PartLLT.isVector()) {
    assert(OrigRegs.size() == 1);
    LLT OrigTy = MRI.getType(OrigRegs[0]);

    unsigned SrcSize = PartLLT.getSizeInBits().getFixedValue() * Regs.size();
    if (SrcSize == OrigTy.getSizeInBits())
      B.buildMergeValues(OrigRegs[0], Regs);
    else {
      auto Widened = B.buildMergeLikeInstr(LLT::scalar(SrcSize), Regs);

      if (IsFloatingPoint)
        B.buildFPTrunc(OrigRegs[0], Widened);
      else
        B.buildTrunc(OrigRegs[0], Widened);
    }

    return;
  }

  if (PartLLT.isVector()) {
    assert(OrigRegs.size() == 1);
    SmallVector<Register> CastRegs(Regs);

    // If PartLLT is a mismatched vector in both number of elements and element
    // size, e.g. PartLLT == v2s64 and LLTy is v3s32, then first coerce it to
    // have the same elt type, i.e. v4s32.
    // TODO: Extend this coersion to element multiples other than just 2.
    if (TypeSize::isKnownGT(PartLLT.getSizeInBits(), LLTy.getSizeInBits()) &&
        PartLLT.getScalarSizeInBits() == LLTy.getScalarSizeInBits() * 2 &&
        Regs.size() == 1) {
      LLT NewTy = PartLLT.changeElementType(LLTy.getElementType())
                      .changeElementCount(PartLLT.getElementCount() * 2);
      CastRegs[0] = B.buildBitcast(NewTy, Regs[0]).getReg(0);
      PartLLT = NewTy;
    }

    if (LLTy.getScalarType() == PartLLT.getElementType()) {
      mergeVectorRegsToResultRegs(B, OrigRegs, CastRegs);
    } else {
      unsigned I = 0;
      LLT GCDTy = getGCDType(LLTy, PartLLT);

      // We are both splitting a vector, and bitcasting its element types. Cast
      // the source pieces into the appropriate number of pieces with the result
      // element type.
      for (Register SrcReg : CastRegs)
        CastRegs[I++] = B.buildBitcast(GCDTy, SrcReg).getReg(0);
      mergeVectorRegsToResultRegs(B, OrigRegs, CastRegs);
    }

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
    int PartsPerElt =
        divideCeil(DstEltTy.getSizeInBits(), PartLLT.getSizeInBits());
    LLT ExtendedPartTy = LLT::scalar(PartLLT.getSizeInBits() * PartsPerElt);

    for (int I = 0, NumElts = LLTy.getNumElements(); I != NumElts; ++I) {
      auto Merge =
          B.buildMergeLikeInstr(ExtendedPartTy, Regs.take_front(PartsPerElt));
      if (ExtendedPartTy.getSizeInBits() > RealDstEltTy.getSizeInBits())
        Merge = B.buildTrunc(RealDstEltTy, Merge);
      // Fix the type in case this is really a vector of pointers.
      MRI.setType(Merge.getReg(0), RealDstEltTy);
      EltMerges.push_back(Merge.getReg(0));
      Regs = Regs.drop_front(PartsPerElt);
    }

    B.buildBuildVector(OrigRegs[0], EltMerges);
  } else {
    // Vector was split, and elements promoted to a wider type.
    // FIXME: Should handle floating point promotions.
    unsigned NumElts = LLTy.getNumElements();
    LLT BVType = LLT::fixed_vector(NumElts, PartLLT);

    Register BuildVec;
    if (NumElts == Regs.size())
      BuildVec = B.buildBuildVector(BVType, Regs).getReg(0);
    else {
      // Vector elements are packed in the inputs.
      // e.g. we have a <4 x s16> but 2 x s32 in regs.
      assert(NumElts > Regs.size());
      LLT SrcEltTy = MRI.getType(Regs[0]);

      LLT OriginalEltTy = MRI.getType(OrigRegs[0]).getElementType();

      // Input registers contain packed elements.
      // Determine how many elements per reg.
      assert((SrcEltTy.getSizeInBits() % OriginalEltTy.getSizeInBits()) == 0);
      unsigned EltPerReg =
          (SrcEltTy.getSizeInBits() / OriginalEltTy.getSizeInBits());

      SmallVector<Register, 0> BVRegs;
      BVRegs.reserve(Regs.size() * EltPerReg);
      for (Register R : Regs) {
        auto Unmerge = B.buildUnmerge(OriginalEltTy, R);
        for (unsigned K = 0; K < EltPerReg; ++K)
          BVRegs.push_back(B.buildAnyExt(PartLLT, Unmerge.getReg(K)).getReg(0));
      }

      // We may have some more elements in BVRegs, e.g. if we have 2 s32 pieces
      // for a <3 x s16> vector. We should have less than EltPerReg extra items.
      if (BVRegs.size() > NumElts) {
        assert((BVRegs.size() - NumElts) < EltPerReg);
        BVRegs.truncate(NumElts);
      }
      BuildVec = B.buildBuildVector(BVType, BVRegs).getReg(0);
    }
    B.buildTrunc(OrigRegs[0], BuildVec);
  }
}

// Internal utility from CallLowering.cpp
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

  const TypeSize PartSize = PartTy.getSizeInBits();

  if (PartTy.isVector() == SrcTy.isVector() &&
      PartTy.getScalarSizeInBits() > SrcTy.getScalarSizeInBits()) {
    assert(DstRegs.size() == 1);
    B.buildInstr(ExtendOp, {DstRegs[0]}, {SrcReg});
    return;
  }

  if (SrcTy.isVector() && !PartTy.isVector() &&
      TypeSize::isKnownGT(PartSize, SrcTy.getElementType().getSizeInBits())) {
    // Vector was scalarized, and the elements extended.
    auto UnmergeToEltTy = B.buildUnmerge(SrcTy.getElementType(), SrcReg);
    for (int i = 0, e = DstRegs.size(); i != e; ++i)
      B.buildAnyExt(DstRegs[i], UnmergeToEltTy.getReg(i));
    return;
  }

  if (SrcTy.isVector() && PartTy.isVector() &&
      PartTy.getSizeInBits() == SrcTy.getSizeInBits() &&
      ElementCount::isKnownLT(SrcTy.getElementCount(),
                              PartTy.getElementCount())) {
    // A coercion like: v2f32 -> v4f32 or nxv2f32 -> nxv4f32
    Register DstReg = DstRegs.front();
    B.buildPadVectorWithUndefElements(DstReg, SrcReg);
    return;
  }

  LLT GCDTy = getGCDType(SrcTy, PartTy);
  if (GCDTy == PartTy) {
    // If this already evenly divisible, we can create a simple unmerge.
    B.buildUnmerge(DstRegs, SrcReg);
    return;
  }

  if (SrcTy.isVector() && !PartTy.isVector() &&
      SrcTy.getScalarSizeInBits() > PartTy.getSizeInBits()) {
    LLT ExtTy =
        LLT::vector(SrcTy.getElementCount(),
                    LLT::scalar(PartTy.getScalarSizeInBits() * DstRegs.size() /
                                SrcTy.getNumElements()));
    auto Ext = B.buildAnyExt(ExtTy, SrcReg);
    B.buildUnmerge(DstRegs, Ext);
    return;
  }

  MachineRegisterInfo &MRI = *B.getMRI();
  LLT DstTy = MRI.getType(DstRegs[0]);
  LLT LCMTy = getCoverTy(SrcTy, PartTy);

  if (PartTy.isVector() && LCMTy == PartTy) {
    assert(DstRegs.size() == 1);
    B.buildPadVectorWithUndefElements(DstRegs[0], SrcReg);
    return;
  }

  const unsigned DstSize = DstTy.getSizeInBits();
  const unsigned SrcSize = SrcTy.getSizeInBits();
  unsigned CoveringSize = LCMTy.getSizeInBits();

  Register UnmergeSrc = SrcReg;

  if (!LCMTy.isVector() && CoveringSize != SrcSize) {
    // For scalars, it's common to be able to use a simple extension.
    if (SrcTy.isScalar() && DstTy.isScalar()) {
      CoveringSize = alignTo(SrcSize, DstSize);
      LLT CoverTy = LLT::scalar(CoveringSize);
      UnmergeSrc = B.buildInstr(ExtendOp, {CoverTy}, {SrcReg}).getReg(0);
    } else {
      // Widen to the common type.
      // FIXME: This should respect the extend type
      Register Undef = B.buildUndef(SrcTy).getReg(0);
      SmallVector<Register, 8> MergeParts(1, SrcReg);
      for (unsigned Size = SrcSize; Size != CoveringSize; Size += SrcSize)
        MergeParts.push_back(Undef);
      UnmergeSrc = B.buildMergeLikeInstr(LCMTy, MergeParts).getReg(0);
    }
  }

  if (LCMTy.isVector() && CoveringSize != SrcSize)
    UnmergeSrc = B.buildPadVectorWithUndefElements(LCMTy, SrcReg).getReg(0);

  B.buildUnmerge(DstRegs, UnmergeSrc);
}

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

static void fail(MachineIRBuilder &MIRBuilder, const char *Msg) {
  MachineFunction &MF = MIRBuilder.getMF();
  MIRBuilder.getContext().diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), Msg, MIRBuilder.getDL()));
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
  auto MIB = MIRBuilder.buildInstrNoInsert(WebAssembly::RETURN);
  MachineFunction &MF = MIRBuilder.getMF();
  auto &Subtarget = MF.getSubtarget<WebAssemblySubtarget>();
  auto &RBI = *Subtarget.getRegBankInfo();

  assert(((Val && !VRegs.empty()) || (!Val && VRegs.empty())) &&
         "Return value without a vreg");

  if (Val && !FLI.CanLowerReturn) {
    insertSRetStores(MIRBuilder, Val->getType(), VRegs, FLI.DemoteRegister);
  } else if (!VRegs.empty()) {
    MachineFunction &MF = MIRBuilder.getMF();
    const Function &F = MF.getFunction();
    MachineRegisterInfo &MRI = MF.getRegInfo();
    const WebAssemblyTargetLowering &TLI = *getTLI<WebAssemblyTargetLowering>();
    auto &DL = F.getDataLayout();
    LLVMContext &Ctx = Val->getType()->getContext();

    SmallVector<EVT, 4> SplitEVTs;
    ComputeValueVTs(TLI, DL, Val->getType(), SplitEVTs);
    assert(VRegs.size() == SplitEVTs.size() &&
           "For each split Type there should be exactly one VReg.");

    SmallVector<ArgInfo, 8> SplitArgs;
    CallingConv::ID CallConv = F.getCallingConv();

    unsigned i = 0;
    for (auto SplitEVT : SplitEVTs) {
      Register CurVReg = VRegs[i];
      ArgInfo CurArgInfo = ArgInfo{CurVReg, SplitEVT.getTypeForEVT(Ctx), 0};
      setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, F);

      splitToValueTypes(CurArgInfo, SplitArgs, DL, CallConv);
      ++i;
    }

    for (auto &Arg : SplitArgs) {
      EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
      MVT NewVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
      LLT OrigLLT = getLLTForType(*Arg.Ty, DL);
      LLT NewLLT = getLLTForMVT(NewVT);
      const TargetRegisterClass &NewRegClass = *TLI.getRegClassFor(NewVT);

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
        buildCopyToRegs(MIRBuilder, Arg.Regs, Arg.OrigRegs[0], OrigLLT, NewLLT,
                        Arg.Ty->isFloatingPointTy()
                            ? TargetOpcode::G_FPEXT
                            : extendOpFromFlags(Arg.Flags[0]));
      }

      for (unsigned Part = 0; Part < NumParts; ++Part) {
        auto NewOutReg = MRI.createGenericVirtualRegister(NewLLT);
        assert(RBI.constrainGenericRegister(NewOutReg, NewRegClass, MRI) &&
               "Couldn't constrain brand-new register?");
        MIRBuilder.buildCopy(NewOutReg, Arg.Regs[Part]);
        MIB.addUse(NewOutReg);
      }
    }
  }

  if (SwiftErrorVReg) {
    llvm_unreachable("WASM does not `supportSwiftError`, yet SwiftErrorVReg is "
                     "improperly valid.");
  }

  MIRBuilder.insertInstr(MIB);
  return true;
}

static unsigned getWASMArgOpcode(MVT ArgType) {
#define MVT_CASE(type)                                                         \
  case MVT::type:                                                              \
    return WebAssembly::ARGUMENT_##type;

  switch (ArgType.SimpleTy) {
    MVT_CASE(i32)
    MVT_CASE(i64)
    MVT_CASE(f32)
    MVT_CASE(f64)
    MVT_CASE(funcref)
    MVT_CASE(externref)
    MVT_CASE(exnref)
    MVT_CASE(v16i8)
    MVT_CASE(v8i16)
    MVT_CASE(v4i32)
    MVT_CASE(v2i64)
    MVT_CASE(v4f32)
    MVT_CASE(v2f64)
    MVT_CASE(v8f16)
  default:
    break;
  }
  llvm_unreachable("Found unexpected type for WASM argument");

#undef MVT_CASE
}

bool WebAssemblyCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {

  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo *MFI = MF.getInfo<WebAssemblyFunctionInfo>();
  const DataLayout &DL = F.getDataLayout();
  auto &TLI = *getTLI<WebAssemblyTargetLowering>();
  auto &Subtarget = MF.getSubtarget<WebAssemblySubtarget>();
  auto &TRI = *Subtarget.getRegisterInfo();
  auto &TII = *Subtarget.getInstrInfo();
  auto &RBI = *Subtarget.getRegBankInfo();

  LLVMContext &Ctx = MIRBuilder.getContext();
  const CallingConv::ID CallConv = F.getCallingConv();

  if (!callingConvSupported(F.getCallingConv())) {
    fail(MIRBuilder, "WebAssembly doesn't support non-C calling conventions");
    return false;
  }

  // Set up the live-in for the incoming ARGUMENTS.
  MF.getRegInfo().addLiveIn(WebAssembly::ARGUMENTS);

  SmallVector<ArgInfo, 8> SplitArgs;

  if (!FLI.CanLowerReturn) {
    insertSRetIncomingArgument(F, SplitArgs, FLI.DemoteRegister, MRI, DL);
  }
  unsigned i = 0;

  bool HasSwiftErrorArg = false;
  bool HasSwiftSelfArg = false;
  for (const auto &Arg : F.args()) {
    ArgInfo OrigArg{VRegs[i], Arg.getType(), i};
    setArgFlags(OrigArg, i + AttributeList::FirstArgIndex, DL, F);

    HasSwiftSelfArg |= Arg.hasSwiftSelfAttr();
    HasSwiftErrorArg |= Arg.hasSwiftErrorAttr();
    if (Arg.hasInAllocaAttr()) {
      fail(MIRBuilder, "WebAssembly hasn't implemented inalloca arguments");
      return false;
    }
    if (Arg.hasNestAttr()) {
      fail(MIRBuilder, "WebAssembly hasn't implemented nest arguments");
      return false;
    }
    splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    ++i;
  }

  unsigned FinalArgIdx = 0;
  for (auto &Arg : SplitArgs) {
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
      auto ArgInst = MIRBuilder.buildInstr(getWASMArgOpcode(NewVT))
                         .addDef(Arg.Regs[Part])
                         .addImm(FinalArgIdx);

      constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *ArgInst,
                               ArgInst->getDesc(), ArgInst->getOperand(0), 0);
      MFI->addParam(NewVT);
      ++FinalArgIdx;
    }

    if (NumParts != 1 || OrigVT != NewVT) {
      buildCopyFromRegs(MIRBuilder, Arg.OrigRegs, Arg.Regs, OrigLLT, NewLLT,
                        Arg.Flags[0], Arg.Ty->isFloatingPointTy());
    }
  }

  /**/

  // For swiftcc, emit additional swiftself and swifterror arguments
  // if there aren't. These additional arguments are also added for callee
  // signature They are necessary to match callee and caller signature for
  // indirect call.
  auto PtrVT = TLI.getPointerTy(DL);
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
    auto PtrVT = TLI.getPointerTy(DL);
    Register VarargVreg = MF.getRegInfo().createGenericVirtualRegister(
        getLLTForType(*PointerType::get(Ctx, 0), DL));
    MFI->setVarargBufferVreg(VarargVreg);

    auto ArgInst = MIRBuilder.buildInstr(getWASMArgOpcode(PtrVT))
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
  MachineFunction &MF = MIRBuilder.getMF();
  auto DL = MIRBuilder.getDataLayout();
  LLVMContext &Ctx = MIRBuilder.getContext();
  const WebAssemblyTargetLowering &TLI = *getTLI<WebAssemblyTargetLowering>();
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  const WebAssemblySubtarget &Subtarget =
      MF.getSubtarget<WebAssemblySubtarget>();
  auto &TRI = *Subtarget.getRegisterInfo();
  auto &TII = *Subtarget.getInstrInfo();
  auto &RBI = *Subtarget.getRegBankInfo();

  CallingConv::ID CallConv = Info.CallConv;
  if (!callingConvSupported(CallConv)) {
    fail(MIRBuilder,
         "WebAssembly doesn't support language-specific or target-specific "
         "calling conventions yet");
    return false;
  }

  // TODO: investigate "PatchPoint"
  /*
  if (Info.IsPatchPoint) {
    fail(MIRBuilder, "WebAssembly doesn't support patch point yet");
    return false;
    }
    */

  if (Info.IsTailCall) {
    Info.LoweredTailCall = true;
    auto NoTail = [&](const char *Msg) {
      if (Info.CB && Info.CB->isMustTailCall())
        fail(MIRBuilder, Msg);
      Info.LoweredTailCall = false;
    };

    if (!Subtarget.hasTailCall())
      NoTail("WebAssembly 'tail-call' feature not enabled");

    // Varargs calls cannot be tail calls because the buffer is on the stack
    if (Info.IsVarArg)
      NoTail("WebAssembly does not support varargs tail calls");

    // Do not tail call unless caller and callee return types match
    const Function &F = MF.getFunction();
    const TargetMachine &TM = TLI.getTargetMachine();
    Type *RetTy = F.getReturnType();
    SmallVector<MVT, 4> CallerRetTys;
    SmallVector<MVT, 4> CalleeRetTys;
    computeLegalValueVTs(F, TM, RetTy, CallerRetTys);
    computeLegalValueVTs(F, TM, Info.OrigRet.Ty, CalleeRetTys);
    bool TypesMatch = CallerRetTys.size() == CalleeRetTys.size() &&
                      std::equal(CallerRetTys.begin(), CallerRetTys.end(),
                                 CalleeRetTys.begin());
    if (!TypesMatch)
      NoTail("WebAssembly tail call requires caller and callee return types to "
             "match");

    // If pointers to local stack values are passed, we cannot tail call
    if (Info.CB) {
      for (auto &Arg : Info.CB->args()) {
        Value *Val = Arg.get();
        // Trace the value back through pointer operations
        while (true) {
          Value *Src = Val->stripPointerCastsAndAliases();
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Src))
            Src = GEP->getPointerOperand();
          if (Val == Src)
            break;
          Val = Src;
        }
        if (isa<AllocaInst>(Val)) {
          NoTail(
              "WebAssembly does not support tail calling with stack arguments");
          break;
        }
      }
    }
  }

  if (Info.LoweredTailCall) {
    MF.getFrameInfo().setHasTailCall();
  }

  MachineInstrBuilder CallInst;

  bool IsIndirect = false;
  Register IndirectIdx;

  // Use indirect calls when callee is pointer in a register
  // or it's a weak (interposable) or non-function global.
  // Makes sure aliases behave.
  if (Info.Callee.isReg() ||
      (Info.Callee.isGlobal() &&
       (Info.Callee.getGlobal()->isInterposable() ||
        !Info.Callee.getGlobal()->getValueType()->isFunctionTy()))) {
    IsIndirect = true;
    CallInst = MIRBuilder.buildInstr(Info.LoweredTailCall
                                         ? WebAssembly::RET_CALL_INDIRECT
                                         : WebAssembly::CALL_INDIRECT);
  } else {
    CallInst = MIRBuilder.buildInstr(
        Info.LoweredTailCall ? WebAssembly::RET_CALL : WebAssembly::CALL);
  }

  if (!Info.LoweredTailCall) {
    if (Info.CanLowerReturn && !Info.OrigRet.Ty->isVoidTy()) {
      SmallVector<EVT, 4> SplitEVTs;
      ComputeValueVTs(TLI, DL, Info.OrigRet.Ty, SplitEVTs);
      assert(Info.OrigRet.Regs.size() == SplitEVTs.size() &&
             "For each split Type there should be exactly one VReg.");

      SmallVector<ArgInfo, 8> SplitReturns;

      unsigned i = 0;
      for (auto SplitEVT : SplitEVTs) {
        Register CurVReg = Info.OrigRet.Regs[i];
        ArgInfo CurArgInfo = ArgInfo{CurVReg, SplitEVT.getTypeForEVT(Ctx), 0};
        if (Info.CB) {
          setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, *Info.CB);
        } else {
          // we don't have a call base, so chances are we're looking at a
          // libcall (external symbol).

          // TODO: figure out how to get ALL the correct attributes
          auto &Flags = CurArgInfo.Flags[0];
          PointerType *PtrTy =
              dyn_cast<PointerType>(CurArgInfo.Ty->getScalarType());
          if (PtrTy) {
            Flags.setPointer();
            Flags.setPointerAddrSpace(PtrTy->getPointerAddressSpace());
          }
          Align MemAlign = DL.getABITypeAlign(CurArgInfo.Ty);
          Flags.setMemAlign(MemAlign);
          Flags.setOrigAlign(MemAlign);
        }
        splitToValueTypes(CurArgInfo, SplitReturns, DL, CallConv);
        ++i;
      }

      for (auto &Ret : SplitReturns) {
        EVT OrigVT = TLI.getValueType(DL, Ret.Ty);
        MVT NewVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
        LLT OrigLLT = getLLTForType(*Ret.Ty, DL);
        LLT NewLLT = getLLTForMVT(NewVT);
        const TargetRegisterClass &NewRegClass = *TLI.getRegClassFor(NewVT);

        // If we need to split the type over multiple regs, check it's a
        // scenario we currently support.
        unsigned NumParts =
            TLI.getNumRegistersForCallingConv(Ctx, CallConv, OrigVT);

        ISD::ArgFlagsTy OrigFlags = Ret.Flags[0];
        Ret.Flags.clear();

        for (unsigned Part = 0; Part < NumParts; ++Part) {
          ISD::ArgFlagsTy Flags = OrigFlags;
          if (Part == 0) {
            Flags.setSplit();
          } else {
            Flags.setOrigAlign(Align(1));
            if (Part == NumParts - 1)
              Flags.setSplitEnd();
          }

          Ret.Flags.push_back(Flags);
        }

        Ret.OrigRegs.assign(Ret.Regs.begin(), Ret.Regs.end());
        if (NumParts != 1 || OrigVT != NewVT) {
          // If we can't directly assign the register, we need one or more
          // intermediate values.
          Ret.Regs.resize(NumParts);

          // For each split register, create and assign a vreg that will store
          // the incoming component of the larger value. These will later be
          // merged to form the final vreg.
          for (unsigned Part = 0; Part < NumParts; ++Part) {
            Ret.Regs[Part] = MRI.createGenericVirtualRegister(NewLLT);
          }
          buildCopyFromRegs(MIRBuilder, Ret.OrigRegs, Ret.Regs, OrigLLT, NewLLT,
                            Ret.Flags[0], Ret.Ty->isFloatingPointTy());
        }

        for (unsigned Part = 0; Part < NumParts; ++Part) {
          auto NewRetReg = Ret.Regs[Part];
          if (!RBI.constrainGenericRegister(NewRetReg, NewRegClass, MRI)) {
            NewRetReg = MRI.createGenericVirtualRegister(NewLLT);
            assert(RBI.constrainGenericRegister(NewRetReg, NewRegClass, MRI) &&
                   "Couldn't constrain brand-new register?");
            MIRBuilder.buildCopy(NewRetReg, Ret.Regs[Part]);
          }
          CallInst.addDef(Ret.Regs[Part]);
        }
      }
    }

    if (!Info.CanLowerReturn) {
      insertSRetLoads(MIRBuilder, Info.OrigRet.Ty, Info.OrigRet.Regs,
                      Info.DemoteRegister, Info.DemoteStackIndex);
    }
  }
  auto SavedInsertPt = MIRBuilder.getInsertPt();
  MIRBuilder.setInstr(*CallInst);

  if (Info.Callee.isReg()) {
    LLT CalleeType = MRI.getType(Info.Callee.getReg());
    assert(CalleeType.isPointer() &&
           "Trying to lower a call with a Callee other than a pointer???");

    // Placeholder for the type index.
    // This gets replaced with the correct value in WebAssemblyMCInstLower.cpp
    CallInst.addImm(0);

    MCSymbolWasm *Table;
    if (CalleeType.getAddressSpace() ==
        WebAssembly::WASM_ADDRESS_SPACE_DEFAULT) {
      Table = WebAssembly::getOrCreateFunctionTableSymbol(MF.getContext(),
                                                          &Subtarget);
      IndirectIdx = Info.Callee.getReg();

      auto PtrSize = CalleeType.getSizeInBits();
      auto PtrIntLLT = LLT::scalar(PtrSize);

      IndirectIdx = MIRBuilder.buildPtrToInt(PtrIntLLT, IndirectIdx).getReg(0);
    } else if (CalleeType.getAddressSpace() ==
               WebAssembly::WASM_ADDRESS_SPACE_FUNCREF) {
      Table = WebAssembly::getOrCreateFuncrefCallTableSymbol(MF.getContext(),
                                                             &Subtarget);

      Type *PtrTy = PointerType::getUnqual(Ctx);
      LLT PtrLLT = getLLTForType(*PtrTy, DL);
      auto PtrIntLLT = LLT::scalar(PtrLLT.getSizeInBits());

      IndirectIdx = MIRBuilder.buildConstant(PtrIntLLT, 0).getReg(0);

      auto TableSetInstr =
          MIRBuilder.buildInstr(WebAssembly::TABLE_SET_FUNCREF);
      TableSetInstr.addSym(Table);
      TableSetInstr.addUse(IndirectIdx);
      TableSetInstr.addUse(Info.Callee.getReg());

      constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *TableSetInstr,
                               TableSetInstr->getDesc(),
                               TableSetInstr->getOperand(1), 1);
      constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *TableSetInstr,
                               TableSetInstr->getDesc(),
                               TableSetInstr->getOperand(2), 2);

    } else {
      fail(MIRBuilder, "Invalid address space for indirect call");
      return false;
    }

    if (Subtarget.hasCallIndirectOverlong()) {
      CallInst.addSym(Table);
    } else {
      // For the MVP there is at most one table whose number is 0, but we can't
      // write a table symbol or issue relocations.  Instead we just ensure the
      // table is live and write a zero.
      Table->setNoStrip();
      CallInst.addImm(0);
    }
  } else {
    if (Info.Callee.isGlobal()) {
      if (IsIndirect) {
        // Placeholder for the type index.
        // This gets replaced with the correct value in
        // WebAssemblyMCInstLower.cpp
        CallInst.addImm(0);

        Type *PtrTy = PointerType::getUnqual(Ctx);
        LLT PtrLLT = getLLTForType(*PtrTy, DL);
        auto PtrSize = PtrLLT.getSizeInBits();
        auto PtrIntLLT = LLT::scalar(PtrSize);

        IndirectIdx =
            MIRBuilder.buildGlobalValue(PtrLLT, Info.Callee.getGlobal())
                .getReg(0);
        IndirectIdx =
            MIRBuilder.buildPtrToInt(PtrIntLLT, IndirectIdx).getReg(0);

        auto *Table = WebAssembly::getOrCreateFunctionTableSymbol(
            MF.getContext(), &Subtarget);
        if (Subtarget.hasCallIndirectOverlong()) {
          CallInst.addSym(Table);
        } else {
          // For the MVP there is at most one table whose number is 0, but we
          // can't write a table symbol or issue relocations.  Instead we just
          // ensure the table is live and write a zero.
          Table->setNoStrip();
          CallInst.addImm(0);
        }
      } else {
        CallInst.addGlobalAddress(Info.Callee.getGlobal());
      }
    } else if (Info.Callee.isSymbol()) {
      CallInst.addExternalSymbol(Info.Callee.getSymbolName());
    } else {
      llvm_unreachable("Trying to lower call with a callee other than reg, "
                       "global, or a symbol.");
    }
  }

  SmallVector<ArgInfo, 8> SplitArgs;

  bool HasSwiftErrorArg = false;
  bool HasSwiftSelfArg = false;

  for (const auto &Arg : Info.OrigArgs) {
    HasSwiftSelfArg |= Arg.Flags[0].isSwiftSelf();
    HasSwiftErrorArg |= Arg.Flags[0].isSwiftError();
    if (Arg.Flags[0].isNest()) {
      fail(MIRBuilder, "WebAssembly hasn't implemented nest arguments");
      return false;
    }
    if (Arg.Flags[0].isInAlloca()) {
      fail(MIRBuilder, "WebAssembly hasn't implemented inalloca arguments");
      return false;
    }
    if (Arg.Flags[0].isInConsecutiveRegs()) {
      fail(MIRBuilder, "WebAssembly hasn't implemented cons regs arguments");
      return false;
    }
    if (Arg.Flags[0].isInConsecutiveRegsLast()) {
      fail(MIRBuilder,
           "WebAssembly hasn't implemented cons regs last arguments");
      return false;
    }

    if (Arg.Flags[0].isByVal() && Arg.Flags[0].getByValSize() != 0) {
      MachineFrameInfo &MFI = MF.getFrameInfo();

      unsigned MemSize = Arg.Flags[0].getByValSize();
      Align MemAlign = Arg.Flags[0].getNonZeroByValAlign();
      int FI = MFI.CreateStackObject(Arg.Flags[0].getByValSize(), MemAlign,
                                     /*isSS=*/false);

      auto StackAddrSpace = DL.getAllocaAddrSpace();
      auto PtrLLT =
          LLT::pointer(StackAddrSpace, DL.getPointerSizeInBits(StackAddrSpace));

      Register StackObjPtrVreg =
          MF.getRegInfo().createGenericVirtualRegister(PtrLLT);
      MRI.setRegClass(StackObjPtrVreg, TLI.getRepRegClassFor(TLI.getPointerTy(
                                           DL, StackAddrSpace)));

      MIRBuilder.buildFrameIndex(StackObjPtrVreg, FI);

      MachinePointerInfo DstPtrInfo = MachinePointerInfo::getFixedStack(MF, FI);

      MachinePointerInfo SrcPtrInfo(Arg.OrigValue);
      if (!Arg.OrigValue) {
        // We still need to accurately track the stack address space if we
        // don't know the underlying value.
        SrcPtrInfo = MachinePointerInfo::getUnknownStack(MF);
      }

      Align DstAlign =
          std::max(MemAlign, inferAlignFromPtrInfo(MF, DstPtrInfo));

      Align SrcAlign =
          std::max(MemAlign, inferAlignFromPtrInfo(MF, SrcPtrInfo));

      MachineMemOperand *SrcMMO = MF.getMachineMemOperand(
          SrcPtrInfo,
          MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable,
          MemSize, SrcAlign);

      MachineMemOperand *DstMMO = MF.getMachineMemOperand(
          DstPtrInfo,
          MachineMemOperand::MOStore | MachineMemOperand::MODereferenceable,
          MemSize, DstAlign);

      const LLT SizeTy = LLT::scalar(PtrLLT.getSizeInBits());

      auto SizeConst = MIRBuilder.buildConstant(SizeTy, MemSize);
      MIRBuilder.buildMemCpy(StackObjPtrVreg, Arg.Regs[0], SizeConst, *DstMMO,
                             *SrcMMO);
    }

    splitToValueTypes(Arg, SplitArgs, DL, CallConv);
  }

  unsigned NumFixedArgs = 0;

  for (auto &Arg : SplitArgs) {
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

      buildCopyToRegs(MIRBuilder, Arg.Regs, Arg.OrigRegs[0], OrigLLT, NewLLT,
                      Arg.Ty->isFloatingPointTy()
                          ? TargetOpcode::G_FPEXT
                          : extendOpFromFlags(Arg.Flags[0]));
    }

    if (!Arg.Flags[0].isVarArg()) {
      for (unsigned Part = 0; Part < NumParts; ++Part) {
        auto NewArgReg = MRI.createGenericVirtualRegister(LLT(NewVT));
        MRI.setRegClass(NewArgReg, TLI.getRegClassFor(NewVT));
        MIRBuilder.buildCopy(NewArgReg, Arg.Regs[Part]);
        CallInst.addUse(Arg.Regs[Part]);
      }
      ++NumFixedArgs;
    }
  }

  if (CallConv == CallingConv::Swift) {
    Type *PtrTy = PointerType::getUnqual(Ctx);
    LLT PtrLLT = getLLTForType(*PtrTy, DL);
    auto &PtrRegClass = *TLI.getRegClassFor(TLI.getSimpleValueType(DL, PtrTy));

    if (!HasSwiftSelfArg) {
      auto NewUndefReg = MIRBuilder.buildUndef(PtrLLT).getReg(0);
      MRI.setRegClass(NewUndefReg, &PtrRegClass);
      CallInst.addUse(NewUndefReg);
    }
    if (!HasSwiftErrorArg) {
      auto NewUndefReg = MIRBuilder.buildUndef(PtrLLT).getReg(0);
      MRI.setRegClass(NewUndefReg, &PtrRegClass);
      CallInst.addUse(NewUndefReg);
    }
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, Info.IsVarArg, MF, ArgLocs, Ctx);

  if (Info.IsVarArg) {
    // Outgoing non-fixed arguments are placed in a buffer. First
    // compute their offsets and the total amount of buffer space needed.
    for (ArgInfo &Arg : drop_begin(SplitArgs, NumFixedArgs)) {
      EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
      MVT PartVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
      Type *Ty = EVT(PartVT).getTypeForEVT(Ctx);

      for (unsigned Part = 0; Part < Arg.Regs.size(); ++Part) {
        Align Alignment = std::max(Arg.Flags[Part].getNonZeroOrigAlign(),
                                   DL.getABITypeAlign(Ty));
        unsigned Offset =
            CCInfo.AllocateStack(DL.getTypeAllocSize(Ty), Alignment);
        CCInfo.addLoc(CCValAssign::getMem(ArgLocs.size(), PartVT, Offset,
                                          PartVT, CCValAssign::Full));
      }
    }
  }

  unsigned NumBytes = CCInfo.getAlignedCallFrameSize();

  auto StackAddrSpace = DL.getAllocaAddrSpace();
  auto PtrLLT = LLT::pointer(StackAddrSpace, DL.getPointerSizeInBits(0));
  auto SizeLLT = LLT::scalar(DL.getPointerSizeInBits(StackAddrSpace));
  auto *PtrRegClass = TLI.getRegClassFor(TLI.getPointerTy(DL, StackAddrSpace));

  if (Info.IsVarArg && NumBytes) {
    Register VarArgStackPtr =
        MF.getRegInfo().createGenericVirtualRegister(PtrLLT);
    MRI.setRegClass(VarArgStackPtr, PtrRegClass);

    MaybeAlign StackAlign = DL.getStackAlignment();
    assert(StackAlign && "data layout string is missing stack alignment");
    int FI = MF.getFrameInfo().CreateStackObject(NumBytes, *StackAlign,
                                                 /*isSS=*/false);

    MIRBuilder.buildFrameIndex(VarArgStackPtr, FI);

    unsigned ValNo = 0;
    for (ArgInfo &Arg : drop_begin(SplitArgs, NumFixedArgs)) {
      EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
      MVT PartVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
      Type *Ty = EVT(PartVT).getTypeForEVT(Ctx);

      for (unsigned Part = 0; Part < Arg.Regs.size(); ++Part) {
        Align Alignment = std::max(Arg.Flags[Part].getNonZeroOrigAlign(),
                                   DL.getABITypeAlign(Ty));

        unsigned Offset = ArgLocs[ValNo++].getLocMemOffset();

        Register DstPtr =
            MIRBuilder
                .buildPtrAdd(
                    PtrLLT, VarArgStackPtr,
                    MIRBuilder.buildConstant(SizeLLT, Offset).getReg(0),
                    MachineInstr::MIFlag::NoUWrap)
                .getReg(0);

        MachineMemOperand *DstMMO = MF.getMachineMemOperand(
            MachinePointerInfo::getFixedStack(MF, FI, Offset),
            MachineMemOperand::MOStore | MachineMemOperand::MODereferenceable,
            PartVT.getStoreSize(), Alignment);

        MIRBuilder.buildStore(Arg.Regs[Part], DstPtr, *DstMMO);
      }
    }

    CallInst.addUse(VarArgStackPtr);
  } else if (Info.IsVarArg) {
    auto NewArgReg = MIRBuilder.buildConstant(PtrLLT, 0).getReg(0);
    MRI.setRegClass(NewArgReg, PtrRegClass);
    CallInst.addUse(NewArgReg);
  }

  if (IsIndirect) {
    auto NewArgReg =
        constrainRegToClass(MRI, TII, RBI, IndirectIdx, *PtrRegClass);
    if (IndirectIdx != NewArgReg)
      MIRBuilder.buildCopy(NewArgReg, IndirectIdx);
    CallInst.addUse(IndirectIdx);
  }

  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), SavedInsertPt);

  return true;
}
