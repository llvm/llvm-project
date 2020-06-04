//===-- llvm/CodeGen/GlobalISel/LegalizerHelper.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the LegalizerHelper class to legalize
/// individual instructions and the LegalizeMachineIR wrapper pass for the
/// primary legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "legalizer"

using namespace llvm;
using namespace LegalizeActions;

/// Try to break down \p OrigTy into \p NarrowTy sized pieces.
///
/// Returns the number of \p NarrowTy elements needed to reconstruct \p OrigTy,
/// with any leftover piece as type \p LeftoverTy
///
/// Returns -1 in the first element of the pair if the breakdown is not
/// satisfiable.
static std::pair<int, int>
getNarrowTypeBreakDown(LLT OrigTy, LLT NarrowTy, LLT &LeftoverTy) {
  assert(!LeftoverTy.isValid() && "this is an out argument");

  unsigned Size = OrigTy.getSizeInBits();
  unsigned NarrowSize = NarrowTy.getSizeInBits();
  unsigned NumParts = Size / NarrowSize;
  unsigned LeftoverSize = Size - NumParts * NarrowSize;
  assert(Size > NarrowSize);

  if (LeftoverSize == 0)
    return {NumParts, 0};

  if (NarrowTy.isVector()) {
    unsigned EltSize = OrigTy.getScalarSizeInBits();
    if (LeftoverSize % EltSize != 0)
      return {-1, -1};
    LeftoverTy = LLT::scalarOrVector(LeftoverSize / EltSize, EltSize);
  } else {
    LeftoverTy = LLT::scalar(LeftoverSize);
  }

  int NumLeftover = LeftoverSize / LeftoverTy.getSizeInBits();
  return std::make_pair(NumParts, NumLeftover);
}

static Type *getFloatTypeForLLT(LLVMContext &Ctx, LLT Ty) {

  if (!Ty.isScalar())
    return nullptr;

  switch (Ty.getSizeInBits()) {
  case 16:
    return Type::getHalfTy(Ctx);
  case 32:
    return Type::getFloatTy(Ctx);
  case 64:
    return Type::getDoubleTy(Ctx);
  case 128:
    return Type::getFP128Ty(Ctx);
  default:
    return nullptr;
  }
}

LegalizerHelper::LegalizerHelper(MachineFunction &MF,
                                 GISelChangeObserver &Observer,
                                 MachineIRBuilder &Builder)
    : MIRBuilder(Builder), MRI(MF.getRegInfo()),
      LI(*MF.getSubtarget().getLegalizerInfo()), Observer(Observer) {
  MIRBuilder.setMF(MF);
  MIRBuilder.setChangeObserver(Observer);
}

LegalizerHelper::LegalizerHelper(MachineFunction &MF, const LegalizerInfo &LI,
                                 GISelChangeObserver &Observer,
                                 MachineIRBuilder &B)
    : MIRBuilder(B), MRI(MF.getRegInfo()), LI(LI), Observer(Observer) {
  MIRBuilder.setMF(MF);
  MIRBuilder.setChangeObserver(Observer);
}
LegalizerHelper::LegalizeResult
LegalizerHelper::legalizeInstrStep(MachineInstr &MI) {
  LLVM_DEBUG(dbgs() << "Legalizing: "; MI.print(dbgs()));

  if (MI.getOpcode() == TargetOpcode::G_INTRINSIC ||
      MI.getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS)
    return LI.legalizeIntrinsic(MI, MIRBuilder, Observer) ? Legalized
                                                          : UnableToLegalize;
  auto Step = LI.getAction(MI, MRI);
  switch (Step.Action) {
  case Legal:
    LLVM_DEBUG(dbgs() << ".. Already legal\n");
    return AlreadyLegal;
  case Libcall:
    LLVM_DEBUG(dbgs() << ".. Convert to libcall\n");
    return libcall(MI);
  case NarrowScalar:
    LLVM_DEBUG(dbgs() << ".. Narrow scalar\n");
    return narrowScalar(MI, Step.TypeIdx, Step.NewType);
  case WidenScalar:
    LLVM_DEBUG(dbgs() << ".. Widen scalar\n");
    return widenScalar(MI, Step.TypeIdx, Step.NewType);
  case Bitcast:
    LLVM_DEBUG(dbgs() << ".. Bitcast type\n");
    return bitcast(MI, Step.TypeIdx, Step.NewType);
  case Lower:
    LLVM_DEBUG(dbgs() << ".. Lower\n");
    return lower(MI, Step.TypeIdx, Step.NewType);
  case FewerElements:
    LLVM_DEBUG(dbgs() << ".. Reduce number of elements\n");
    return fewerElementsVector(MI, Step.TypeIdx, Step.NewType);
  case MoreElements:
    LLVM_DEBUG(dbgs() << ".. Increase number of elements\n");
    return moreElementsVector(MI, Step.TypeIdx, Step.NewType);
  case Custom:
    LLVM_DEBUG(dbgs() << ".. Custom legalization\n");
    return LI.legalizeCustom(MI, MRI, MIRBuilder, Observer) ? Legalized
                                                            : UnableToLegalize;
  default:
    LLVM_DEBUG(dbgs() << ".. Unable to legalize\n");
    return UnableToLegalize;
  }
}

void LegalizerHelper::extractParts(Register Reg, LLT Ty, int NumParts,
                                   SmallVectorImpl<Register> &VRegs) {
  for (int i = 0; i < NumParts; ++i)
    VRegs.push_back(MRI.createGenericVirtualRegister(Ty));
  MIRBuilder.buildUnmerge(VRegs, Reg);
}

bool LegalizerHelper::extractParts(Register Reg, LLT RegTy,
                                   LLT MainTy, LLT &LeftoverTy,
                                   SmallVectorImpl<Register> &VRegs,
                                   SmallVectorImpl<Register> &LeftoverRegs) {
  assert(!LeftoverTy.isValid() && "this is an out argument");

  unsigned RegSize = RegTy.getSizeInBits();
  unsigned MainSize = MainTy.getSizeInBits();
  unsigned NumParts = RegSize / MainSize;
  unsigned LeftoverSize = RegSize - NumParts * MainSize;

  // Use an unmerge when possible.
  if (LeftoverSize == 0) {
    for (unsigned I = 0; I < NumParts; ++I)
      VRegs.push_back(MRI.createGenericVirtualRegister(MainTy));
    MIRBuilder.buildUnmerge(VRegs, Reg);
    return true;
  }

  if (MainTy.isVector()) {
    unsigned EltSize = MainTy.getScalarSizeInBits();
    if (LeftoverSize % EltSize != 0)
      return false;
    LeftoverTy = LLT::scalarOrVector(LeftoverSize / EltSize, EltSize);
  } else {
    LeftoverTy = LLT::scalar(LeftoverSize);
  }

  // For irregular sizes, extract the individual parts.
  for (unsigned I = 0; I != NumParts; ++I) {
    Register NewReg = MRI.createGenericVirtualRegister(MainTy);
    VRegs.push_back(NewReg);
    MIRBuilder.buildExtract(NewReg, Reg, MainSize * I);
  }

  for (unsigned Offset = MainSize * NumParts; Offset < RegSize;
       Offset += LeftoverSize) {
    Register NewReg = MRI.createGenericVirtualRegister(LeftoverTy);
    LeftoverRegs.push_back(NewReg);
    MIRBuilder.buildExtract(NewReg, Reg, Offset);
  }

  return true;
}

void LegalizerHelper::insertParts(Register DstReg,
                                  LLT ResultTy, LLT PartTy,
                                  ArrayRef<Register> PartRegs,
                                  LLT LeftoverTy,
                                  ArrayRef<Register> LeftoverRegs) {
  if (!LeftoverTy.isValid()) {
    assert(LeftoverRegs.empty());

    if (!ResultTy.isVector()) {
      MIRBuilder.buildMerge(DstReg, PartRegs);
      return;
    }

    if (PartTy.isVector())
      MIRBuilder.buildConcatVectors(DstReg, PartRegs);
    else
      MIRBuilder.buildBuildVector(DstReg, PartRegs);
    return;
  }

  unsigned PartSize = PartTy.getSizeInBits();
  unsigned LeftoverPartSize = LeftoverTy.getSizeInBits();

  Register CurResultReg = MRI.createGenericVirtualRegister(ResultTy);
  MIRBuilder.buildUndef(CurResultReg);

  unsigned Offset = 0;
  for (Register PartReg : PartRegs) {
    Register NewResultReg = MRI.createGenericVirtualRegister(ResultTy);
    MIRBuilder.buildInsert(NewResultReg, CurResultReg, PartReg, Offset);
    CurResultReg = NewResultReg;
    Offset += PartSize;
  }

  for (unsigned I = 0, E = LeftoverRegs.size(); I != E; ++I) {
    // Use the original output register for the final insert to avoid a copy.
    Register NewResultReg = (I + 1 == E) ?
      DstReg : MRI.createGenericVirtualRegister(ResultTy);

    MIRBuilder.buildInsert(NewResultReg, CurResultReg, LeftoverRegs[I], Offset);
    CurResultReg = NewResultReg;
    Offset += LeftoverPartSize;
  }
}

/// Return the result registers of G_UNMERGE_VALUES \p MI in \p Regs
static void getUnmergeResults(SmallVectorImpl<Register> &Regs,
                              const MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES);

  const int NumResults = MI.getNumOperands() - 1;
  Regs.resize(NumResults);
  for (int I = 0; I != NumResults; ++I)
    Regs[I] = MI.getOperand(I).getReg();
}

LLT LegalizerHelper::extractGCDType(SmallVectorImpl<Register> &Parts, LLT DstTy,
                                    LLT NarrowTy, Register SrcReg) {
  LLT SrcTy = MRI.getType(SrcReg);

  LLT GCDTy = getGCDType(DstTy, getGCDType(SrcTy, NarrowTy));
  if (SrcTy == GCDTy) {
    // If the source already evenly divides the result type, we don't need to do
    // anything.
    Parts.push_back(SrcReg);
  } else {
    // Need to split into common type sized pieces.
    auto Unmerge = MIRBuilder.buildUnmerge(GCDTy, SrcReg);
    getUnmergeResults(Parts, *Unmerge);
  }

  return GCDTy;
}

LLT LegalizerHelper::buildLCMMergePieces(LLT DstTy, LLT NarrowTy, LLT GCDTy,
                                         SmallVectorImpl<Register> &VRegs,
                                         unsigned PadStrategy) {
  LLT LCMTy = getLCMType(DstTy, NarrowTy);

  int NumParts = LCMTy.getSizeInBits() / NarrowTy.getSizeInBits();
  int NumSubParts = NarrowTy.getSizeInBits() / GCDTy.getSizeInBits();
  int NumOrigSrc = VRegs.size();

  Register PadReg;

  // Get a value we can use to pad the source value if the sources won't evenly
  // cover the result type.
  if (NumOrigSrc < NumParts * NumSubParts) {
    if (PadStrategy == TargetOpcode::G_ZEXT)
      PadReg = MIRBuilder.buildConstant(GCDTy, 0).getReg(0);
    else if (PadStrategy == TargetOpcode::G_ANYEXT)
      PadReg = MIRBuilder.buildUndef(GCDTy).getReg(0);
    else {
      assert(PadStrategy == TargetOpcode::G_SEXT);

      // Shift the sign bit of the low register through the high register.
      auto ShiftAmt =
        MIRBuilder.buildConstant(LLT::scalar(64), GCDTy.getSizeInBits() - 1);
      PadReg = MIRBuilder.buildAShr(GCDTy, VRegs.back(), ShiftAmt).getReg(0);
    }
  }

  // Registers for the final merge to be produced.
  SmallVector<Register, 4> Remerge(NumParts);

  // Registers needed for intermediate merges, which will be merged into a
  // source for Remerge.
  SmallVector<Register, 4> SubMerge(NumSubParts);

  // Once we've fully read off the end of the original source bits, we can reuse
  // the same high bits for remaining padding elements.
  Register AllPadReg;

  // Build merges to the LCM type to cover the original result type.
  for (int I = 0; I != NumParts; ++I) {
    bool AllMergePartsArePadding = true;

    // Build the requested merges to the requested type.
    for (int J = 0; J != NumSubParts; ++J) {
      int Idx = I * NumSubParts + J;
      if (Idx >= NumOrigSrc) {
        SubMerge[J] = PadReg;
        continue;
      }

      SubMerge[J] = VRegs[Idx];

      // There are meaningful bits here we can't reuse later.
      AllMergePartsArePadding = false;
    }

    // If we've filled up a complete piece with padding bits, we can directly
    // emit the natural sized constant if applicable, rather than a merge of
    // smaller constants.
    if (AllMergePartsArePadding && !AllPadReg) {
      if (PadStrategy == TargetOpcode::G_ANYEXT)
        AllPadReg = MIRBuilder.buildUndef(NarrowTy).getReg(0);
      else if (PadStrategy == TargetOpcode::G_ZEXT)
        AllPadReg = MIRBuilder.buildConstant(NarrowTy, 0).getReg(0);

      // If this is a sign extension, we can't materialize a trivial constant
      // with the right type and have to produce a merge.
    }

    if (AllPadReg) {
      // Avoid creating additional instructions if we're just adding additional
      // copies of padding bits.
      Remerge[I] = AllPadReg;
      continue;
    }

    if (NumSubParts == 1)
      Remerge[I] = SubMerge[0];
    else
      Remerge[I] = MIRBuilder.buildMerge(NarrowTy, SubMerge).getReg(0);

    // In the sign extend padding case, re-use the first all-signbit merge.
    if (AllMergePartsArePadding && !AllPadReg)
      AllPadReg = Remerge[I];
  }

  VRegs = std::move(Remerge);
  return LCMTy;
}

void LegalizerHelper::buildWidenedRemergeToDst(Register DstReg, LLT LCMTy,
                                               ArrayRef<Register> RemergeRegs) {
  LLT DstTy = MRI.getType(DstReg);

  // Create the merge to the widened source, and extract the relevant bits into
  // the result.

  if (DstTy == LCMTy) {
    MIRBuilder.buildMerge(DstReg, RemergeRegs);
    return;
  }

  auto Remerge = MIRBuilder.buildMerge(LCMTy, RemergeRegs);
  if (DstTy.isScalar() && LCMTy.isScalar()) {
    MIRBuilder.buildTrunc(DstReg, Remerge);
    return;
  }

  if (LCMTy.isVector()) {
    MIRBuilder.buildExtract(DstReg, Remerge, 0);
    return;
  }

  llvm_unreachable("unhandled case");
}

static RTLIB::Libcall getRTLibDesc(unsigned Opcode, unsigned Size) {
#define RTLIBCASE(LibcallPrefix)                                               \
  do {                                                                         \
    switch (Size) {                                                            \
    case 32:                                                                   \
      return RTLIB::LibcallPrefix##32;                                         \
    case 64:                                                                   \
      return RTLIB::LibcallPrefix##64;                                         \
    case 128:                                                                  \
      return RTLIB::LibcallPrefix##128;                                        \
    default:                                                                   \
      llvm_unreachable("unexpected size");                                     \
    }                                                                          \
  } while (0)

  assert((Size == 32 || Size == 64 || Size == 128) && "Unsupported size");

  switch (Opcode) {
  case TargetOpcode::G_SDIV:
    RTLIBCASE(SDIV_I);
  case TargetOpcode::G_UDIV:
    RTLIBCASE(UDIV_I);
  case TargetOpcode::G_SREM:
    RTLIBCASE(SREM_I);
  case TargetOpcode::G_UREM:
    RTLIBCASE(UREM_I);
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
    RTLIBCASE(CTLZ_I);
  case TargetOpcode::G_FADD:
    RTLIBCASE(ADD_F);
  case TargetOpcode::G_FSUB:
    RTLIBCASE(SUB_F);
  case TargetOpcode::G_FMUL:
    RTLIBCASE(MUL_F);
  case TargetOpcode::G_FDIV:
    RTLIBCASE(DIV_F);
  case TargetOpcode::G_FEXP:
    RTLIBCASE(EXP_F);
  case TargetOpcode::G_FEXP2:
    RTLIBCASE(EXP2_F);
  case TargetOpcode::G_FREM:
    RTLIBCASE(REM_F);
  case TargetOpcode::G_FPOW:
    RTLIBCASE(POW_F);
  case TargetOpcode::G_FMA:
    RTLIBCASE(FMA_F);
  case TargetOpcode::G_FSIN:
    RTLIBCASE(SIN_F);
  case TargetOpcode::G_FCOS:
    RTLIBCASE(COS_F);
  case TargetOpcode::G_FLOG10:
    RTLIBCASE(LOG10_F);
  case TargetOpcode::G_FLOG:
    RTLIBCASE(LOG_F);
  case TargetOpcode::G_FLOG2:
    RTLIBCASE(LOG2_F);
  case TargetOpcode::G_FCEIL:
    RTLIBCASE(CEIL_F);
  case TargetOpcode::G_FFLOOR:
    RTLIBCASE(FLOOR_F);
  case TargetOpcode::G_FMINNUM:
    RTLIBCASE(FMIN_F);
  case TargetOpcode::G_FMAXNUM:
    RTLIBCASE(FMAX_F);
  case TargetOpcode::G_FSQRT:
    RTLIBCASE(SQRT_F);
  case TargetOpcode::G_FRINT:
    RTLIBCASE(RINT_F);
  case TargetOpcode::G_FNEARBYINT:
    RTLIBCASE(NEARBYINT_F);
  }
  llvm_unreachable("Unknown libcall function");
}

/// True if an instruction is in tail position in its caller. Intended for
/// legalizing libcalls as tail calls when possible.
static bool isLibCallInTailPosition(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const Function &F = MBB.getParent()->getFunction();

  // Conservatively require the attributes of the call to match those of
  // the return. Ignore NoAlias and NonNull because they don't affect the
  // call sequence.
  AttributeList CallerAttrs = F.getAttributes();
  if (AttrBuilder(CallerAttrs, AttributeList::ReturnIndex)
          .removeAttribute(Attribute::NoAlias)
          .removeAttribute(Attribute::NonNull)
          .hasAttributes())
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  if (CallerAttrs.hasAttribute(AttributeList::ReturnIndex, Attribute::ZExt) ||
      CallerAttrs.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt))
    return false;

  // Only tail call if the following instruction is a standard return.
  auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  auto Next = next_nodbg(MI.getIterator(), MBB.instr_end());
  if (Next == MBB.instr_end() || TII.isTailCall(*Next) || !Next->isReturn())
    return false;

  return true;
}

LegalizerHelper::LegalizeResult
llvm::createLibcall(MachineIRBuilder &MIRBuilder, const char *Name,
                    const CallLowering::ArgInfo &Result,
                    ArrayRef<CallLowering::ArgInfo> Args,
                    const CallingConv::ID CC) {
  auto &CLI = *MIRBuilder.getMF().getSubtarget().getCallLowering();

  CallLowering::CallLoweringInfo Info;
  Info.CallConv = CC;
  Info.Callee = MachineOperand::CreateES(Name);
  Info.OrigRet = Result;
  std::copy(Args.begin(), Args.end(), std::back_inserter(Info.OrigArgs));
  if (!CLI.lowerCall(MIRBuilder, Info))
    return LegalizerHelper::UnableToLegalize;

  return LegalizerHelper::Legalized;
}

LegalizerHelper::LegalizeResult
llvm::createLibcall(MachineIRBuilder &MIRBuilder, RTLIB::Libcall Libcall,
                    const CallLowering::ArgInfo &Result,
                    ArrayRef<CallLowering::ArgInfo> Args) {
  auto &TLI = *MIRBuilder.getMF().getSubtarget().getTargetLowering();
  const char *Name = TLI.getLibcallName(Libcall);
  const CallingConv::ID CC = TLI.getLibcallCallingConv(Libcall);
  return createLibcall(MIRBuilder, Name, Result, Args, CC);
}

// Useful for libcalls where all operands have the same type.
static LegalizerHelper::LegalizeResult
simpleLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder, unsigned Size,
              Type *OpType) {
  auto Libcall = getRTLibDesc(MI.getOpcode(), Size);

  SmallVector<CallLowering::ArgInfo, 3> Args;
  for (unsigned i = 1; i < MI.getNumOperands(); i++)
    Args.push_back({MI.getOperand(i).getReg(), OpType});
  return createLibcall(MIRBuilder, Libcall, {MI.getOperand(0).getReg(), OpType},
                       Args);
}

LegalizerHelper::LegalizeResult
llvm::createMemLibcall(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS);
  auto &Ctx = MIRBuilder.getMF().getFunction().getContext();

  SmallVector<CallLowering::ArgInfo, 3> Args;
  // Add all the args, except for the last which is an imm denoting 'tail'.
  for (unsigned i = 1; i < MI.getNumOperands() - 1; i++) {
    Register Reg = MI.getOperand(i).getReg();

    // Need derive an IR type for call lowering.
    LLT OpLLT = MRI.getType(Reg);
    Type *OpTy = nullptr;
    if (OpLLT.isPointer())
      OpTy = Type::getInt8PtrTy(Ctx, OpLLT.getAddressSpace());
    else
      OpTy = IntegerType::get(Ctx, OpLLT.getSizeInBits());
    Args.push_back({Reg, OpTy});
  }

  auto &CLI = *MIRBuilder.getMF().getSubtarget().getCallLowering();
  auto &TLI = *MIRBuilder.getMF().getSubtarget().getTargetLowering();
  Intrinsic::ID ID = MI.getOperand(0).getIntrinsicID();
  RTLIB::Libcall RTLibcall;
  switch (ID) {
  case Intrinsic::memcpy:
    RTLibcall = RTLIB::MEMCPY;
    break;
  case Intrinsic::memset:
    RTLibcall = RTLIB::MEMSET;
    break;
  case Intrinsic::memmove:
    RTLibcall = RTLIB::MEMMOVE;
    break;
  default:
    return LegalizerHelper::UnableToLegalize;
  }
  const char *Name = TLI.getLibcallName(RTLibcall);

  MIRBuilder.setInstrAndDebugLoc(MI);

  CallLowering::CallLoweringInfo Info;
  Info.CallConv = TLI.getLibcallCallingConv(RTLibcall);
  Info.Callee = MachineOperand::CreateES(Name);
  Info.OrigRet = CallLowering::ArgInfo({0}, Type::getVoidTy(Ctx));
  Info.IsTailCall = MI.getOperand(MI.getNumOperands() - 1).getImm() == 1 &&
                    isLibCallInTailPosition(MI);

  std::copy(Args.begin(), Args.end(), std::back_inserter(Info.OrigArgs));
  if (!CLI.lowerCall(MIRBuilder, Info))
    return LegalizerHelper::UnableToLegalize;

  if (Info.LoweredTailCall) {
    assert(Info.IsTailCall && "Lowered tail call when it wasn't a tail call?");
    // We must have a return following the call (or debug insts) to get past
    // isLibCallInTailPosition.
    do {
      MachineInstr *Next = MI.getNextNode();
      assert(Next && (Next->isReturn() || Next->isDebugInstr()) &&
             "Expected instr following MI to be return or debug inst?");
      // We lowered a tail call, so the call is now the return from the block.
      // Delete the old return.
      Next->eraseFromParent();
    } while (MI.getNextNode());
  }

  return LegalizerHelper::Legalized;
}

static RTLIB::Libcall getConvRTLibDesc(unsigned Opcode, Type *ToType,
                                       Type *FromType) {
  auto ToMVT = MVT::getVT(ToType);
  auto FromMVT = MVT::getVT(FromType);

  switch (Opcode) {
  case TargetOpcode::G_FPEXT:
    return RTLIB::getFPEXT(FromMVT, ToMVT);
  case TargetOpcode::G_FPTRUNC:
    return RTLIB::getFPROUND(FromMVT, ToMVT);
  case TargetOpcode::G_FPTOSI:
    return RTLIB::getFPTOSINT(FromMVT, ToMVT);
  case TargetOpcode::G_FPTOUI:
    return RTLIB::getFPTOUINT(FromMVT, ToMVT);
  case TargetOpcode::G_SITOFP:
    return RTLIB::getSINTTOFP(FromMVT, ToMVT);
  case TargetOpcode::G_UITOFP:
    return RTLIB::getUINTTOFP(FromMVT, ToMVT);
  }
  llvm_unreachable("Unsupported libcall function");
}

static LegalizerHelper::LegalizeResult
conversionLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder, Type *ToType,
                  Type *FromType) {
  RTLIB::Libcall Libcall = getConvRTLibDesc(MI.getOpcode(), ToType, FromType);
  return createLibcall(MIRBuilder, Libcall, {MI.getOperand(0).getReg(), ToType},
                       {{MI.getOperand(1).getReg(), FromType}});
}

LegalizerHelper::LegalizeResult
LegalizerHelper::libcall(MachineInstr &MI) {
  LLT LLTy = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = LLTy.getSizeInBits();
  auto &Ctx = MIRBuilder.getMF().getFunction().getContext();

  MIRBuilder.setInstrAndDebugLoc(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF: {
    Type *HLTy = IntegerType::get(Ctx, Size);
    auto Status = simpleLibcall(MI, MIRBuilder, Size, HLTy);
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FNEARBYINT: {
    Type *HLTy = getFloatTypeForLLT(Ctx, LLTy);
    if (!HLTy || (Size != 32 && Size != 64 && Size != 128)) {
      LLVM_DEBUG(dbgs() << "No libcall available for size " << Size << ".\n");
      return UnableToLegalize;
    }
    auto Status = simpleLibcall(MI, MIRBuilder, Size, HLTy);
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FPEXT:
  case TargetOpcode::G_FPTRUNC: {
    Type *FromTy = getFloatTypeForLLT(Ctx,  MRI.getType(MI.getOperand(1).getReg()));
    Type *ToTy = getFloatTypeForLLT(Ctx, MRI.getType(MI.getOperand(0).getReg()));
    if (!FromTy || !ToTy)
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(MI, MIRBuilder, ToTy, FromTy );
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI: {
    // FIXME: Support other types
    unsigned FromSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned ToSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if ((ToSize != 32 && ToSize != 64) || (FromSize != 32 && FromSize != 64))
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(
        MI, MIRBuilder,
        ToSize == 32 ? Type::getInt32Ty(Ctx) : Type::getInt64Ty(Ctx),
        FromSize == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx));
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP: {
    // FIXME: Support other types
    unsigned FromSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned ToSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if ((FromSize != 32 && FromSize != 64) || (ToSize != 32 && ToSize != 64))
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(
        MI, MIRBuilder,
        ToSize == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx),
        FromSize == 32 ? Type::getInt32Ty(Ctx) : Type::getInt64Ty(Ctx));
    if (Status != Legalized)
      return Status;
    break;
  }
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult LegalizerHelper::narrowScalar(MachineInstr &MI,
                                                              unsigned TypeIdx,
                                                              LLT NarrowTy) {
  MIRBuilder.setInstrAndDebugLoc(MI);

  uint64_t SizeOp0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  uint64_t NarrowSize = NarrowTy.getSizeInBits();

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_IMPLICIT_DEF: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);

    // If SizeOp0 is not an exact multiple of NarrowSize, emit
    // G_ANYEXT(G_IMPLICIT_DEF). Cast result to vector if needed.
    // FIXME: Although this would also be legal for the general case, it causes
    //  a lot of regressions in the emitted code (superfluous COPYs, artifact
    //  combines not being hit). This seems to be a problem related to the
    //  artifact combiner.
    if (SizeOp0 % NarrowSize != 0) {
      LLT ImplicitTy = NarrowTy;
      if (DstTy.isVector())
        ImplicitTy = LLT::vector(DstTy.getNumElements(), ImplicitTy);

      Register ImplicitReg = MIRBuilder.buildUndef(ImplicitTy).getReg(0);
      MIRBuilder.buildAnyExt(DstReg, ImplicitReg);

      MI.eraseFromParent();
      return Legalized;
    }

    int NumParts = SizeOp0 / NarrowSize;

    SmallVector<Register, 2> DstRegs;
    for (int i = 0; i < NumParts; ++i)
      DstRegs.push_back(MIRBuilder.buildUndef(NarrowTy).getReg(0));

    if (DstTy.isVector())
      MIRBuilder.buildBuildVector(DstReg, DstRegs);
    else
      MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    const APInt &Val = MI.getOperand(1).getCImm()->getValue();
    unsigned TotalSize = Ty.getSizeInBits();
    unsigned NarrowSize = NarrowTy.getSizeInBits();
    int NumParts = TotalSize / NarrowSize;

    SmallVector<Register, 4> PartRegs;
    for (int I = 0; I != NumParts; ++I) {
      unsigned Offset = I * NarrowSize;
      auto K = MIRBuilder.buildConstant(NarrowTy,
                                        Val.lshr(Offset).trunc(NarrowSize));
      PartRegs.push_back(K.getReg(0));
    }

    LLT LeftoverTy;
    unsigned LeftoverBits = TotalSize - NumParts * NarrowSize;
    SmallVector<Register, 1> LeftoverRegs;
    if (LeftoverBits != 0) {
      LeftoverTy = LLT::scalar(LeftoverBits);
      auto K = MIRBuilder.buildConstant(
        LeftoverTy,
        Val.lshr(NumParts * NarrowSize).trunc(LeftoverBits));
      LeftoverRegs.push_back(K.getReg(0));
    }

    insertParts(MI.getOperand(0).getReg(),
                Ty, NarrowTy, PartRegs, LeftoverTy, LeftoverRegs);

    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_ANYEXT:
    return narrowScalarExt(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_TRUNC: {
    if (TypeIdx != 1)
      return UnableToLegalize;

    uint64_t SizeOp1 = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    if (NarrowTy.getSizeInBits() * 2 != SizeOp1) {
      LLVM_DEBUG(dbgs() << "Can't narrow trunc to type " << NarrowTy << "\n");
      return UnableToLegalize;
    }

    auto Unmerge = MIRBuilder.buildUnmerge(NarrowTy, MI.getOperand(1));
    MIRBuilder.buildCopy(MI.getOperand(0), Unmerge.getReg(0));
    MI.eraseFromParent();
    return Legalized;
  }

  case TargetOpcode::G_FREEZE:
    return reduceOperationWidth(MI, TypeIdx, NarrowTy);

  case TargetOpcode::G_ADD: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    // Expand in terms of carry-setting/consuming G_ADDE instructions.
    int NumParts = SizeOp0 / NarrowTy.getSizeInBits();

    SmallVector<Register, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    Register CarryIn;
    for (int i = 0; i < NumParts; ++i) {
      Register DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      Register CarryOut = MRI.createGenericVirtualRegister(LLT::scalar(1));

      if (i == 0)
        MIRBuilder.buildUAddo(DstReg, CarryOut, Src1Regs[i], Src2Regs[i]);
      else {
        MIRBuilder.buildUAdde(DstReg, CarryOut, Src1Regs[i],
                              Src2Regs[i], CarryIn);
      }

      DstRegs.push_back(DstReg);
      CarryIn = CarryOut;
    }
    Register DstReg = MI.getOperand(0).getReg();
    if(MRI.getType(DstReg).isVector())
      MIRBuilder.buildBuildVector(DstReg, DstRegs);
    else
      MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SUB: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;

    int NumParts = SizeOp0 / NarrowTy.getSizeInBits();

    SmallVector<Register, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    Register DstReg = MRI.createGenericVirtualRegister(NarrowTy);
    Register BorrowOut = MRI.createGenericVirtualRegister(LLT::scalar(1));
    MIRBuilder.buildInstr(TargetOpcode::G_USUBO, {DstReg, BorrowOut},
                          {Src1Regs[0], Src2Regs[0]});
    DstRegs.push_back(DstReg);
    Register BorrowIn = BorrowOut;
    for (int i = 1; i < NumParts; ++i) {
      DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      BorrowOut = MRI.createGenericVirtualRegister(LLT::scalar(1));

      MIRBuilder.buildInstr(TargetOpcode::G_USUBE, {DstReg, BorrowOut},
                            {Src1Regs[i], Src2Regs[i], BorrowIn});

      DstRegs.push_back(DstReg);
      BorrowIn = BorrowOut;
    }
    MIRBuilder.buildMerge(MI.getOperand(0), DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_UMULH:
    return narrowScalarMul(MI, NarrowTy);
  case TargetOpcode::G_EXTRACT:
    return narrowScalarExtract(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_INSERT:
    return narrowScalarInsert(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_LOAD: {
    const auto &MMO = **MI.memoperands_begin();
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy.isVector())
      return UnableToLegalize;

    if (8 * MMO.getSize() != DstTy.getSizeInBits()) {
      Register TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
      auto &MMO = **MI.memoperands_begin();
      MIRBuilder.buildLoad(TmpReg, MI.getOperand(1), MMO);
      MIRBuilder.buildAnyExt(DstReg, TmpReg);
      MI.eraseFromParent();
      return Legalized;
    }

    return reduceLoadStoreWidth(MI, TypeIdx, NarrowTy);
  }
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_SEXTLOAD: {
    bool ZExt = MI.getOpcode() == TargetOpcode::G_ZEXTLOAD;
    Register DstReg = MI.getOperand(0).getReg();
    Register PtrReg = MI.getOperand(1).getReg();

    Register TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
    auto &MMO = **MI.memoperands_begin();
    if (MMO.getSizeInBits() == NarrowSize) {
      MIRBuilder.buildLoad(TmpReg, PtrReg, MMO);
    } else {
      MIRBuilder.buildLoadInstr(MI.getOpcode(), TmpReg, PtrReg, MMO);
    }

    if (ZExt)
      MIRBuilder.buildZExt(DstReg, TmpReg);
    else
      MIRBuilder.buildSExt(DstReg, TmpReg);

    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_STORE: {
    const auto &MMO = **MI.memoperands_begin();

    Register SrcReg = MI.getOperand(0).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    if (SrcTy.isVector())
      return UnableToLegalize;

    int NumParts = SizeOp0 / NarrowSize;
    unsigned HandledSize = NumParts * NarrowTy.getSizeInBits();
    unsigned LeftoverBits = SrcTy.getSizeInBits() - HandledSize;
    if (SrcTy.isVector() && LeftoverBits != 0)
      return UnableToLegalize;

    if (8 * MMO.getSize() != SrcTy.getSizeInBits()) {
      Register TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
      auto &MMO = **MI.memoperands_begin();
      MIRBuilder.buildTrunc(TmpReg, SrcReg);
      MIRBuilder.buildStore(TmpReg, MI.getOperand(1), MMO);
      MI.eraseFromParent();
      return Legalized;
    }

    return reduceLoadStoreWidth(MI, 0, NarrowTy);
  }
  case TargetOpcode::G_SELECT:
    return narrowScalarSelect(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    // Legalize bitwise operation:
    // A = BinOp<Ty> B, C
    // into:
    // B1, ..., BN = G_UNMERGE_VALUES B
    // C1, ..., CN = G_UNMERGE_VALUES C
    // A1 = BinOp<Ty/N> B1, C2
    // ...
    // AN = BinOp<Ty/N> BN, CN
    // A = G_MERGE_VALUES A1, ..., AN
    return narrowScalarBasic(MI, TypeIdx, NarrowTy);
  }
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR:
    return narrowScalarShift(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTPOP:
    if (TypeIdx == 1)
      switch (MI.getOpcode()) {
      case TargetOpcode::G_CTLZ:
      case TargetOpcode::G_CTLZ_ZERO_UNDEF:
        return narrowScalarCTLZ(MI, TypeIdx, NarrowTy);
      case TargetOpcode::G_CTTZ:
      case TargetOpcode::G_CTTZ_ZERO_UNDEF:
        return narrowScalarCTTZ(MI, TypeIdx, NarrowTy);
      case TargetOpcode::G_CTPOP:
        return narrowScalarCTPOP(MI, TypeIdx, NarrowTy);
      default:
        return UnableToLegalize;
      }

    Observer.changingInstr(MI);
    narrowScalarDst(MI, NarrowTy, 0, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_INTTOPTR:
    if (TypeIdx != 1)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    narrowScalarSrc(MI, NarrowTy, 1);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PTRTOINT:
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    narrowScalarDst(MI, NarrowTy, 0, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PHI: {
    unsigned NumParts = SizeOp0 / NarrowSize;
    SmallVector<Register, 2> DstRegs(NumParts);
    SmallVector<SmallVector<Register, 2>, 2> SrcRegs(MI.getNumOperands() / 2);
    Observer.changingInstr(MI);
    for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
      MachineBasicBlock &OpMBB = *MI.getOperand(i + 1).getMBB();
      MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());
      extractParts(MI.getOperand(i).getReg(), NarrowTy, NumParts,
                   SrcRegs[i / 2]);
    }
    MachineBasicBlock &MBB = *MI.getParent();
    MIRBuilder.setInsertPt(MBB, MI);
    for (unsigned i = 0; i < NumParts; ++i) {
      DstRegs[i] = MRI.createGenericVirtualRegister(NarrowTy);
      MachineInstrBuilder MIB =
          MIRBuilder.buildInstr(TargetOpcode::G_PHI).addDef(DstRegs[i]);
      for (unsigned j = 1; j < MI.getNumOperands(); j += 2)
        MIB.addUse(SrcRegs[j / 2][i]).add(MI.getOperand(j + 1));
    }
    MIRBuilder.setInsertPt(MBB, MBB.getFirstNonPHI());
    MIRBuilder.buildMerge(MI.getOperand(0), DstRegs);
    Observer.changedInstr(MI);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
  case TargetOpcode::G_INSERT_VECTOR_ELT: {
    if (TypeIdx != 2)
      return UnableToLegalize;

    int OpIdx = MI.getOpcode() == TargetOpcode::G_EXTRACT_VECTOR_ELT ? 2 : 3;
    Observer.changingInstr(MI);
    narrowScalarSrc(MI, NarrowTy, OpIdx);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_ICMP: {
    uint64_t SrcSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
    if (NarrowSize * 2 != SrcSize)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    Register LHSL = MRI.createGenericVirtualRegister(NarrowTy);
    Register LHSH = MRI.createGenericVirtualRegister(NarrowTy);
    MIRBuilder.buildUnmerge({LHSL, LHSH}, MI.getOperand(2));

    Register RHSL = MRI.createGenericVirtualRegister(NarrowTy);
    Register RHSH = MRI.createGenericVirtualRegister(NarrowTy);
    MIRBuilder.buildUnmerge({RHSL, RHSH}, MI.getOperand(3));

    CmpInst::Predicate Pred =
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
    LLT ResTy = MRI.getType(MI.getOperand(0).getReg());

    if (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) {
      MachineInstrBuilder XorL = MIRBuilder.buildXor(NarrowTy, LHSL, RHSL);
      MachineInstrBuilder XorH = MIRBuilder.buildXor(NarrowTy, LHSH, RHSH);
      MachineInstrBuilder Or = MIRBuilder.buildOr(NarrowTy, XorL, XorH);
      MachineInstrBuilder Zero = MIRBuilder.buildConstant(NarrowTy, 0);
      MIRBuilder.buildICmp(Pred, MI.getOperand(0), Or, Zero);
    } else {
      MachineInstrBuilder CmpH = MIRBuilder.buildICmp(Pred, ResTy, LHSH, RHSH);
      MachineInstrBuilder CmpHEQ =
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, ResTy, LHSH, RHSH);
      MachineInstrBuilder CmpLU = MIRBuilder.buildICmp(
          ICmpInst::getUnsignedPredicate(Pred), ResTy, LHSL, RHSL);
      MIRBuilder.buildSelect(MI.getOperand(0), CmpHEQ, CmpLU, CmpH);
    }
    Observer.changedInstr(MI);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SEXT_INREG: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    int64_t SizeInBits = MI.getOperand(2).getImm();

    // So long as the new type has more bits than the bits we're extending we
    // don't need to break it apart.
    if (NarrowTy.getScalarSizeInBits() >= SizeInBits) {
      Observer.changingInstr(MI);
      // We don't lose any non-extension bits by truncating the src and
      // sign-extending the dst.
      MachineOperand &MO1 = MI.getOperand(1);
      auto TruncMIB = MIRBuilder.buildTrunc(NarrowTy, MO1);
      MO1.setReg(TruncMIB.getReg(0));

      MachineOperand &MO2 = MI.getOperand(0);
      Register DstExt = MRI.createGenericVirtualRegister(NarrowTy);
      MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
      MIRBuilder.buildSExt(MO2, DstExt);
      MO2.setReg(DstExt);
      Observer.changedInstr(MI);
      return Legalized;
    }

    // Break it apart. Components below the extension point are unmodified. The
    // component containing the extension point becomes a narrower SEXT_INREG.
    // Components above it are ashr'd from the component containing the
    // extension point.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    int NumParts = SizeOp0 / NarrowSize;

    // List the registers where the destination will be scattered.
    SmallVector<Register, 2> DstRegs;
    // List the registers where the source will be split.
    SmallVector<Register, 2> SrcRegs;

    // Create all the temporary registers.
    for (int i = 0; i < NumParts; ++i) {
      Register SrcReg = MRI.createGenericVirtualRegister(NarrowTy);

      SrcRegs.push_back(SrcReg);
    }

    // Explode the big arguments into smaller chunks.
    MIRBuilder.buildUnmerge(SrcRegs, MI.getOperand(1));

    Register AshrCstReg =
        MIRBuilder.buildConstant(NarrowTy, NarrowTy.getScalarSizeInBits() - 1)
            .getReg(0);
    Register FullExtensionReg = 0;
    Register PartialExtensionReg = 0;

    // Do the operation on each small part.
    for (int i = 0; i < NumParts; ++i) {
      if ((i + 1) * NarrowTy.getScalarSizeInBits() < SizeInBits)
        DstRegs.push_back(SrcRegs[i]);
      else if (i * NarrowTy.getScalarSizeInBits() > SizeInBits) {
        assert(PartialExtensionReg &&
               "Expected to visit partial extension before full");
        if (FullExtensionReg) {
          DstRegs.push_back(FullExtensionReg);
          continue;
        }
        DstRegs.push_back(
            MIRBuilder.buildAShr(NarrowTy, PartialExtensionReg, AshrCstReg)
                .getReg(0));
        FullExtensionReg = DstRegs.back();
      } else {
        DstRegs.push_back(
            MIRBuilder
                .buildInstr(
                    TargetOpcode::G_SEXT_INREG, {NarrowTy},
                    {SrcRegs[i], SizeInBits % NarrowTy.getScalarSizeInBits()})
                .getReg(0));
        PartialExtensionReg = DstRegs.back();
      }
    }

    // Gather the destination registers into the final destination.
    Register DstReg = MI.getOperand(0).getReg();
    MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_BSWAP:
  case TargetOpcode::G_BITREVERSE: {
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    SmallVector<Register, 2> SrcRegs, DstRegs;
    unsigned NumParts = SizeOp0 / NarrowSize;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, SrcRegs);

    for (unsigned i = 0; i < NumParts; ++i) {
      auto DstPart = MIRBuilder.buildInstr(MI.getOpcode(), {NarrowTy},
                                           {SrcRegs[NumParts - 1 - i]});
      DstRegs.push_back(DstPart.getReg(0));
    }

    MIRBuilder.buildMerge(MI.getOperand(0), DstRegs);

    Observer.changedInstr(MI);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_PTRMASK: {
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    narrowScalarSrc(MI, NarrowTy, 2);
    Observer.changedInstr(MI);
    return Legalized;
  }
  }
}

Register LegalizerHelper::coerceToScalar(Register Val) {
  LLT Ty = MRI.getType(Val);
  if (Ty.isScalar())
    return Val;

  const DataLayout &DL = MIRBuilder.getDataLayout();
  LLT NewTy = LLT::scalar(Ty.getSizeInBits());
  if (Ty.isPointer()) {
    if (DL.isNonIntegralAddressSpace(Ty.getAddressSpace()))
      return Register();
    return MIRBuilder.buildPtrToInt(NewTy, Val).getReg(0);
  }

  Register NewVal = Val;

  assert(Ty.isVector());
  LLT EltTy = Ty.getElementType();
  if (EltTy.isPointer())
    NewVal = MIRBuilder.buildPtrToInt(NewTy, NewVal).getReg(0);
  return MIRBuilder.buildBitcast(NewTy, NewVal).getReg(0);
}

void LegalizerHelper::widenScalarSrc(MachineInstr &MI, LLT WideTy,
                                     unsigned OpIdx, unsigned ExtOpcode) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  auto ExtB = MIRBuilder.buildInstr(ExtOpcode, {WideTy}, {MO});
  MO.setReg(ExtB.getReg(0));
}

void LegalizerHelper::narrowScalarSrc(MachineInstr &MI, LLT NarrowTy,
                                      unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  auto ExtB = MIRBuilder.buildTrunc(NarrowTy, MO);
  MO.setReg(ExtB.getReg(0));
}

void LegalizerHelper::widenScalarDst(MachineInstr &MI, LLT WideTy,
                                     unsigned OpIdx, unsigned TruncOpcode) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  Register DstExt = MRI.createGenericVirtualRegister(WideTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildInstr(TruncOpcode, {MO}, {DstExt});
  MO.setReg(DstExt);
}

void LegalizerHelper::narrowScalarDst(MachineInstr &MI, LLT NarrowTy,
                                      unsigned OpIdx, unsigned ExtOpcode) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  Register DstTrunc = MRI.createGenericVirtualRegister(NarrowTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildInstr(ExtOpcode, {MO}, {DstTrunc});
  MO.setReg(DstTrunc);
}

void LegalizerHelper::moreElementsVectorDst(MachineInstr &MI, LLT WideTy,
                                            unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  Register DstExt = MRI.createGenericVirtualRegister(WideTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildExtract(MO, DstExt, 0);
  MO.setReg(DstExt);
}

void LegalizerHelper::moreElementsVectorSrc(MachineInstr &MI, LLT MoreTy,
                                            unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);

  LLT OldTy = MRI.getType(MO.getReg());
  unsigned OldElts = OldTy.getNumElements();
  unsigned NewElts = MoreTy.getNumElements();

  unsigned NumParts = NewElts / OldElts;

  // Use concat_vectors if the result is a multiple of the number of elements.
  if (NumParts * OldElts == NewElts) {
    SmallVector<Register, 8> Parts;
    Parts.push_back(MO.getReg());

    Register ImpDef = MIRBuilder.buildUndef(OldTy).getReg(0);
    for (unsigned I = 1; I != NumParts; ++I)
      Parts.push_back(ImpDef);

    auto Concat = MIRBuilder.buildConcatVectors(MoreTy, Parts);
    MO.setReg(Concat.getReg(0));
    return;
  }

  Register MoreReg = MRI.createGenericVirtualRegister(MoreTy);
  Register ImpDef = MIRBuilder.buildUndef(MoreTy).getReg(0);
  MIRBuilder.buildInsert(MoreReg, ImpDef, MO.getReg(), 0);
  MO.setReg(MoreReg);
}

void LegalizerHelper::bitcastSrc(MachineInstr &MI, LLT CastTy, unsigned OpIdx) {
  MachineOperand &Op = MI.getOperand(OpIdx);
  Op.setReg(MIRBuilder.buildBitcast(CastTy, Op).getReg(0));
}

void LegalizerHelper::bitcastDst(MachineInstr &MI, LLT CastTy, unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  Register CastDst = MRI.createGenericVirtualRegister(CastTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildBitcast(MO, CastDst);
  MO.setReg(CastDst);
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarMergeValues(MachineInstr &MI, unsigned TypeIdx,
                                        LLT WideTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  if (DstTy.isVector())
    return UnableToLegalize;

  Register Src1 = MI.getOperand(1).getReg();
  LLT SrcTy = MRI.getType(Src1);
  const int DstSize = DstTy.getSizeInBits();
  const int SrcSize = SrcTy.getSizeInBits();
  const int WideSize = WideTy.getSizeInBits();
  const int NumMerge = (DstSize + WideSize - 1) / WideSize;

  unsigned NumOps = MI.getNumOperands();
  unsigned NumSrc = MI.getNumOperands() - 1;
  unsigned PartSize = DstTy.getSizeInBits() / NumSrc;

  if (WideSize >= DstSize) {
    // Directly pack the bits in the target type.
    Register ResultReg = MIRBuilder.buildZExt(WideTy, Src1).getReg(0);

    for (unsigned I = 2; I != NumOps; ++I) {
      const unsigned Offset = (I - 1) * PartSize;

      Register SrcReg = MI.getOperand(I).getReg();
      assert(MRI.getType(SrcReg) == LLT::scalar(PartSize));

      auto ZextInput = MIRBuilder.buildZExt(WideTy, SrcReg);

      Register NextResult = I + 1 == NumOps && WideTy == DstTy ? DstReg :
        MRI.createGenericVirtualRegister(WideTy);

      auto ShiftAmt = MIRBuilder.buildConstant(WideTy, Offset);
      auto Shl = MIRBuilder.buildShl(WideTy, ZextInput, ShiftAmt);
      MIRBuilder.buildOr(NextResult, ResultReg, Shl);
      ResultReg = NextResult;
    }

    if (WideSize > DstSize)
      MIRBuilder.buildTrunc(DstReg, ResultReg);
    else if (DstTy.isPointer())
      MIRBuilder.buildIntToPtr(DstReg, ResultReg);

    MI.eraseFromParent();
    return Legalized;
  }

  // Unmerge the original values to the GCD type, and recombine to the next
  // multiple greater than the original type.
  //
  // %3:_(s12) = G_MERGE_VALUES %0:_(s4), %1:_(s4), %2:_(s4) -> s6
  // %4:_(s2), %5:_(s2) = G_UNMERGE_VALUES %0
  // %6:_(s2), %7:_(s2) = G_UNMERGE_VALUES %1
  // %8:_(s2), %9:_(s2) = G_UNMERGE_VALUES %2
  // %10:_(s6) = G_MERGE_VALUES %4, %5, %6
  // %11:_(s6) = G_MERGE_VALUES %7, %8, %9
  // %12:_(s12) = G_MERGE_VALUES %10, %11
  //
  // Padding with undef if necessary:
  //
  // %2:_(s8) = G_MERGE_VALUES %0:_(s4), %1:_(s4) -> s6
  // %3:_(s2), %4:_(s2) = G_UNMERGE_VALUES %0
  // %5:_(s2), %6:_(s2) = G_UNMERGE_VALUES %1
  // %7:_(s2) = G_IMPLICIT_DEF
  // %8:_(s6) = G_MERGE_VALUES %3, %4, %5
  // %9:_(s6) = G_MERGE_VALUES %6, %7, %7
  // %10:_(s12) = G_MERGE_VALUES %8, %9

  const int GCD = greatestCommonDivisor(SrcSize, WideSize);
  LLT GCDTy = LLT::scalar(GCD);

  SmallVector<Register, 8> Parts;
  SmallVector<Register, 8> NewMergeRegs;
  SmallVector<Register, 8> Unmerges;
  LLT WideDstTy = LLT::scalar(NumMerge * WideSize);

  // Decompose the original operands if they don't evenly divide.
  for (int I = 1, E = MI.getNumOperands(); I != E; ++I) {
    Register SrcReg = MI.getOperand(I).getReg();
    if (GCD == SrcSize) {
      Unmerges.push_back(SrcReg);
    } else {
      auto Unmerge = MIRBuilder.buildUnmerge(GCDTy, SrcReg);
      for (int J = 0, JE = Unmerge->getNumOperands() - 1; J != JE; ++J)
        Unmerges.push_back(Unmerge.getReg(J));
    }
  }

  // Pad with undef to the next size that is a multiple of the requested size.
  if (static_cast<int>(Unmerges.size()) != NumMerge * WideSize) {
    Register UndefReg = MIRBuilder.buildUndef(GCDTy).getReg(0);
    for (int I = Unmerges.size(); I != NumMerge * WideSize; ++I)
      Unmerges.push_back(UndefReg);
  }

  const int PartsPerGCD = WideSize / GCD;

  // Build merges of each piece.
  ArrayRef<Register> Slicer(Unmerges);
  for (int I = 0; I != NumMerge; ++I, Slicer = Slicer.drop_front(PartsPerGCD)) {
    auto Merge = MIRBuilder.buildMerge(WideTy, Slicer.take_front(PartsPerGCD));
    NewMergeRegs.push_back(Merge.getReg(0));
  }

  // A truncate may be necessary if the requested type doesn't evenly divide the
  // original result type.
  if (DstTy.getSizeInBits() == WideDstTy.getSizeInBits()) {
    MIRBuilder.buildMerge(DstReg, NewMergeRegs);
  } else {
    auto FinalMerge = MIRBuilder.buildMerge(WideDstTy, NewMergeRegs);
    MIRBuilder.buildTrunc(DstReg, FinalMerge.getReg(0));
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarUnmergeValues(MachineInstr &MI, unsigned TypeIdx,
                                          LLT WideTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  int NumDst = MI.getNumOperands() - 1;
  Register SrcReg = MI.getOperand(NumDst).getReg();
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy.isVector())
    return UnableToLegalize;

  Register Dst0Reg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst0Reg);
  if (!DstTy.isScalar())
    return UnableToLegalize;

  if (WideTy.getSizeInBits() >= SrcTy.getSizeInBits()) {
    if (SrcTy.isPointer()) {
      const DataLayout &DL = MIRBuilder.getDataLayout();
      if (DL.isNonIntegralAddressSpace(SrcTy.getAddressSpace())) {
        LLVM_DEBUG(
            dbgs() << "Not casting non-integral address space integer\n");
        return UnableToLegalize;
      }

      SrcTy = LLT::scalar(SrcTy.getSizeInBits());
      SrcReg = MIRBuilder.buildPtrToInt(SrcTy, SrcReg).getReg(0);
    }

    // Widen SrcTy to WideTy. This does not affect the result, but since the
    // user requested this size, it is probably better handled than SrcTy and
    // should reduce the total number of legalization artifacts
    if (WideTy.getSizeInBits() > SrcTy.getSizeInBits()) {
      SrcTy = WideTy;
      SrcReg = MIRBuilder.buildAnyExt(WideTy, SrcReg).getReg(0);
    }

    // Theres no unmerge type to target. Directly extract the bits from the
    // source type
    unsigned DstSize = DstTy.getSizeInBits();

    MIRBuilder.buildTrunc(Dst0Reg, SrcReg);
    for (int I = 1; I != NumDst; ++I) {
      auto ShiftAmt = MIRBuilder.buildConstant(SrcTy, DstSize * I);
      auto Shr = MIRBuilder.buildLShr(SrcTy, SrcReg, ShiftAmt);
      MIRBuilder.buildTrunc(MI.getOperand(I), Shr);
    }

    MI.eraseFromParent();
    return Legalized;
  }

  // Extend the source to a wider type.
  LLT LCMTy = getLCMType(SrcTy, WideTy);

  Register WideSrc = SrcReg;
  if (LCMTy.getSizeInBits() != SrcTy.getSizeInBits()) {
    // TODO: If this is an integral address space, cast to integer and anyext.
    if (SrcTy.isPointer()) {
      LLVM_DEBUG(dbgs() << "Widening pointer source types not implemented\n");
      return UnableToLegalize;
    }

    WideSrc = MIRBuilder.buildAnyExt(LCMTy, WideSrc).getReg(0);
  }

  auto Unmerge = MIRBuilder.buildUnmerge(WideTy, WideSrc);

  // Create a sequence of unmerges to the original results. since we may have
  // widened the source, we will need to pad the results with dead defs to cover
  // the source register.
  // e.g. widen s16 to s32:
  // %1:_(s16), %2:_(s16), %3:_(s16) = G_UNMERGE_VALUES %0:_(s48)
  //
  // =>
  //  %4:_(s64) = G_ANYEXT %0:_(s48)
  //  %5:_(s32), %6:_(s32) = G_UNMERGE_VALUES %4 ; Requested unmerge
  //  %1:_(s16), %2:_(s16) = G_UNMERGE_VALUES %5 ; unpack to original regs
  //  %3:_(s16), dead %7 = G_UNMERGE_VALUES %6 ; original reg + extra dead def

  const int NumUnmerge = Unmerge->getNumOperands() - 1;
  const int PartsPerUnmerge = WideTy.getSizeInBits() / DstTy.getSizeInBits();

  for (int I = 0; I != NumUnmerge; ++I) {
    auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_UNMERGE_VALUES);

    for (int J = 0; J != PartsPerUnmerge; ++J) {
      int Idx = I * PartsPerUnmerge + J;
      if (Idx < NumDst)
        MIB.addDef(MI.getOperand(Idx).getReg());
      else {
        // Create dead def for excess components.
        MIB.addDef(MRI.createGenericVirtualRegister(DstTy));
      }
    }

    MIB.addUse(Unmerge.getReg(I));
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarExtract(MachineInstr &MI, unsigned TypeIdx,
                                    LLT WideTy) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT SrcTy = MRI.getType(SrcReg);

  LLT DstTy = MRI.getType(DstReg);
  unsigned Offset = MI.getOperand(2).getImm();

  if (TypeIdx == 0) {
    if (SrcTy.isVector() || DstTy.isVector())
      return UnableToLegalize;

    SrcOp Src(SrcReg);
    if (SrcTy.isPointer()) {
      // Extracts from pointers can be handled only if they are really just
      // simple integers.
      const DataLayout &DL = MIRBuilder.getDataLayout();
      if (DL.isNonIntegralAddressSpace(SrcTy.getAddressSpace()))
        return UnableToLegalize;

      LLT SrcAsIntTy = LLT::scalar(SrcTy.getSizeInBits());
      Src = MIRBuilder.buildPtrToInt(SrcAsIntTy, Src);
      SrcTy = SrcAsIntTy;
    }

    if (DstTy.isPointer())
      return UnableToLegalize;

    if (Offset == 0) {
      // Avoid a shift in the degenerate case.
      MIRBuilder.buildTrunc(DstReg,
                            MIRBuilder.buildAnyExtOrTrunc(WideTy, Src));
      MI.eraseFromParent();
      return Legalized;
    }

    // Do a shift in the source type.
    LLT ShiftTy = SrcTy;
    if (WideTy.getSizeInBits() > SrcTy.getSizeInBits()) {
      Src = MIRBuilder.buildAnyExt(WideTy, Src);
      ShiftTy = WideTy;
    } else if (WideTy.getSizeInBits() > SrcTy.getSizeInBits())
      return UnableToLegalize;

    auto LShr = MIRBuilder.buildLShr(
      ShiftTy, Src, MIRBuilder.buildConstant(ShiftTy, Offset));
    MIRBuilder.buildTrunc(DstReg, LShr);
    MI.eraseFromParent();
    return Legalized;
  }

  if (SrcTy.isScalar()) {
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    Observer.changedInstr(MI);
    return Legalized;
  }

  if (!SrcTy.isVector())
    return UnableToLegalize;

  if (DstTy != SrcTy.getElementType())
    return UnableToLegalize;

  if (Offset % SrcTy.getScalarSizeInBits() != 0)
    return UnableToLegalize;

  Observer.changingInstr(MI);
  widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);

  MI.getOperand(2).setImm((WideTy.getSizeInBits() / SrcTy.getSizeInBits()) *
                          Offset);
  widenScalarDst(MI, WideTy.getScalarType(), 0);
  Observer.changedInstr(MI);
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarInsert(MachineInstr &MI, unsigned TypeIdx,
                                   LLT WideTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;
  Observer.changingInstr(MI);
  widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
  widenScalarDst(MI, WideTy);
  Observer.changedInstr(MI);
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalar(MachineInstr &MI, unsigned TypeIdx, LLT WideTy) {
  MIRBuilder.setInstrAndDebugLoc(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_EXTRACT:
    return widenScalarExtract(MI, TypeIdx, WideTy);
  case TargetOpcode::G_INSERT:
    return widenScalarInsert(MI, TypeIdx, WideTy);
  case TargetOpcode::G_MERGE_VALUES:
    return widenScalarMergeValues(MI, TypeIdx, WideTy);
  case TargetOpcode::G_UNMERGE_VALUES:
    return widenScalarUnmergeValues(MI, TypeIdx, WideTy);
  case TargetOpcode::G_UADDO:
  case TargetOpcode::G_USUBO: {
    if (TypeIdx == 1)
      return UnableToLegalize; // TODO
    auto LHSZext = MIRBuilder.buildZExt(WideTy, MI.getOperand(2));
    auto RHSZext = MIRBuilder.buildZExt(WideTy, MI.getOperand(3));
    unsigned Opcode = MI.getOpcode() == TargetOpcode::G_UADDO
                          ? TargetOpcode::G_ADD
                          : TargetOpcode::G_SUB;
    // Do the arithmetic in the larger type.
    auto NewOp = MIRBuilder.buildInstr(Opcode, {WideTy}, {LHSZext, RHSZext});
    LLT OrigTy = MRI.getType(MI.getOperand(0).getReg());
    APInt Mask =
        APInt::getLowBitsSet(WideTy.getSizeInBits(), OrigTy.getSizeInBits());
    auto AndOp = MIRBuilder.buildAnd(
        WideTy, NewOp, MIRBuilder.buildConstant(WideTy, Mask));
    // There is no overflow if the AndOp is the same as NewOp.
    MIRBuilder.buildICmp(CmpInst::ICMP_NE, MI.getOperand(1), NewOp, AndOp);
    // Now trunc the NewOp to the original result.
    MIRBuilder.buildTrunc(MI.getOperand(0), NewOp);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTPOP: {
    if (TypeIdx == 0) {
      Observer.changingInstr(MI);
      widenScalarDst(MI, WideTy, 0);
      Observer.changedInstr(MI);
      return Legalized;
    }

    Register SrcReg = MI.getOperand(1).getReg();

    // First ZEXT the input.
    auto MIBSrc = MIRBuilder.buildZExt(WideTy, SrcReg);
    LLT CurTy = MRI.getType(SrcReg);
    if (MI.getOpcode() == TargetOpcode::G_CTTZ) {
      // The count is the same in the larger type except if the original
      // value was zero.  This can be handled by setting the bit just off
      // the top of the original type.
      auto TopBit =
          APInt::getOneBitSet(WideTy.getSizeInBits(), CurTy.getSizeInBits());
      MIBSrc = MIRBuilder.buildOr(
        WideTy, MIBSrc, MIRBuilder.buildConstant(WideTy, TopBit));
    }

    // Perform the operation at the larger size.
    auto MIBNewOp = MIRBuilder.buildInstr(MI.getOpcode(), {WideTy}, {MIBSrc});
    // This is already the correct result for CTPOP and CTTZs
    if (MI.getOpcode() == TargetOpcode::G_CTLZ ||
        MI.getOpcode() == TargetOpcode::G_CTLZ_ZERO_UNDEF) {
      // The correct result is NewOp - (Difference in widety and current ty).
      unsigned SizeDiff = WideTy.getSizeInBits() - CurTy.getSizeInBits();
      MIBNewOp = MIRBuilder.buildSub(
          WideTy, MIBNewOp, MIRBuilder.buildConstant(WideTy, SizeDiff));
    }

    MIRBuilder.buildZExtOrTrunc(MI.getOperand(0), MIBNewOp);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_BSWAP: {
    Observer.changingInstr(MI);
    Register DstReg = MI.getOperand(0).getReg();

    Register ShrReg = MRI.createGenericVirtualRegister(WideTy);
    Register DstExt = MRI.createGenericVirtualRegister(WideTy);
    Register ShiftAmtReg = MRI.createGenericVirtualRegister(WideTy);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);

    MI.getOperand(0).setReg(DstExt);

    MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());

    LLT Ty = MRI.getType(DstReg);
    unsigned DiffBits = WideTy.getScalarSizeInBits() - Ty.getScalarSizeInBits();
    MIRBuilder.buildConstant(ShiftAmtReg, DiffBits);
    MIRBuilder.buildLShr(ShrReg, DstExt, ShiftAmtReg);

    MIRBuilder.buildTrunc(DstReg, ShrReg);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_BITREVERSE: {
    Observer.changingInstr(MI);

    Register DstReg = MI.getOperand(0).getReg();
    LLT Ty = MRI.getType(DstReg);
    unsigned DiffBits = WideTy.getScalarSizeInBits() - Ty.getScalarSizeInBits();

    Register DstExt = MRI.createGenericVirtualRegister(WideTy);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    MI.getOperand(0).setReg(DstExt);
    MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());

    auto ShiftAmt = MIRBuilder.buildConstant(WideTy, DiffBits);
    auto Shift = MIRBuilder.buildLShr(WideTy, DstExt, ShiftAmt);
    MIRBuilder.buildTrunc(DstReg, Shift);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_FREEZE:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_ADD:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SUB:
    // Perform operation at larger width (any extension is fines here, high bits
    // don't affect the result) and then truncate the result back to the
    // original type.
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ANYEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SHL:
    Observer.changingInstr(MI);

    if (TypeIdx == 0) {
      widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
      widenScalarDst(MI, WideTy);
    } else {
      assert(TypeIdx == 1);
      // The "number of bits to shift" operand must preserve its value as an
      // unsigned integer:
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    }

    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_SEXT);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_SEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_LSHR:
    Observer.changingInstr(MI);

    if (TypeIdx == 0) {
      unsigned CvtOp = MI.getOpcode() == TargetOpcode::G_ASHR ?
        TargetOpcode::G_SEXT : TargetOpcode::G_ZEXT;

      widenScalarSrc(MI, WideTy, 1, CvtOp);
      widenScalarDst(MI, WideTy);
    } else {
      assert(TypeIdx == 1);
      // The "number of bits to shift" operand must preserve its value as an
      // unsigned integer:
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    }

    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ZEXT);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SELECT:
    Observer.changingInstr(MI);
    if (TypeIdx == 0) {
      // Perform operation at larger width (any extension is fine here, high
      // bits don't affect the result) and then truncate the result back to the
      // original type.
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ANYEXT);
      widenScalarSrc(MI, WideTy, 3, TargetOpcode::G_ANYEXT);
      widenScalarDst(MI, WideTy);
    } else {
      bool IsVec = MRI.getType(MI.getOperand(1).getReg()).isVector();
      // Explicit extension is required here since high bits affect the result.
      widenScalarSrc(MI, WideTy, 1, MIRBuilder.getBoolExtOp(IsVec, false));
    }
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
    Observer.changingInstr(MI);

    if (TypeIdx == 0)
      widenScalarDst(MI, WideTy);
    else
      widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_FPEXT);

    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_SITOFP:
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_SEXT);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_UITOFP:
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD:
    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_STORE: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    if (!isPowerOf2_32(Ty.getSizeInBits()))
      return UnableToLegalize;

    Observer.changingInstr(MI);

    unsigned ExtType = Ty.getScalarSizeInBits() == 1 ?
      TargetOpcode::G_ZEXT : TargetOpcode::G_ANYEXT;
    widenScalarSrc(MI, WideTy, 0, ExtType);

    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    MachineOperand &SrcMO = MI.getOperand(1);
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    unsigned ExtOpc = LI.getExtOpcodeForWideningConstant(
        MRI.getType(MI.getOperand(0).getReg()));
    assert((ExtOpc == TargetOpcode::G_ZEXT || ExtOpc == TargetOpcode::G_SEXT ||
            ExtOpc == TargetOpcode::G_ANYEXT) &&
           "Illegal Extend");
    const APInt &SrcVal = SrcMO.getCImm()->getValue();
    const APInt &Val = (ExtOpc == TargetOpcode::G_SEXT)
                           ? SrcVal.sext(WideTy.getSizeInBits())
                           : SrcVal.zext(WideTy.getSizeInBits());
    Observer.changingInstr(MI);
    SrcMO.setCImm(ConstantInt::get(Ctx, Val));

    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_FCONSTANT: {
    MachineOperand &SrcMO = MI.getOperand(1);
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    APFloat Val = SrcMO.getFPImm()->getValueAPF();
    bool LosesInfo;
    switch (WideTy.getSizeInBits()) {
    case 32:
      Val.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                  &LosesInfo);
      break;
    case 64:
      Val.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &LosesInfo);
      break;
    default:
      return UnableToLegalize;
    }

    assert(!LosesInfo && "extend should always be lossless");

    Observer.changingInstr(MI);
    SrcMO.setFPImm(ConstantFP::get(Ctx, Val));

    widenScalarDst(MI, WideTy, 0, TargetOpcode::G_FPTRUNC);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_IMPLICIT_DEF: {
    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_BRCOND:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 0, MIRBuilder.getBoolExtOp(false, false));
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_FCMP:
    Observer.changingInstr(MI);
    if (TypeIdx == 0)
      widenScalarDst(MI, WideTy);
    else {
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_FPEXT);
      widenScalarSrc(MI, WideTy, 3, TargetOpcode::G_FPEXT);
    }
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_ICMP:
    Observer.changingInstr(MI);
    if (TypeIdx == 0)
      widenScalarDst(MI, WideTy);
    else {
      unsigned ExtOpcode = CmpInst::isSigned(static_cast<CmpInst::Predicate>(
                               MI.getOperand(1).getPredicate()))
                               ? TargetOpcode::G_SEXT
                               : TargetOpcode::G_ZEXT;
      widenScalarSrc(MI, WideTy, 2, ExtOpcode);
      widenScalarSrc(MI, WideTy, 3, ExtOpcode);
    }
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_PTR_ADD:
    assert(TypeIdx == 1 && "unable to legalize pointer of G_PTR_ADD");
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_SEXT);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_PHI: {
    assert(TypeIdx == 0 && "Expecting only Idx 0");

    Observer.changingInstr(MI);
    for (unsigned I = 1; I < MI.getNumOperands(); I += 2) {
      MachineBasicBlock &OpMBB = *MI.getOperand(I + 1).getMBB();
      MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());
      widenScalarSrc(MI, WideTy, I, TargetOpcode::G_ANYEXT);
    }

    MachineBasicBlock &MBB = *MI.getParent();
    MIRBuilder.setInsertPt(MBB, --MBB.getFirstNonPHI());
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    if (TypeIdx == 0) {
      Register VecReg = MI.getOperand(1).getReg();
      LLT VecTy = MRI.getType(VecReg);
      Observer.changingInstr(MI);

      widenScalarSrc(MI, LLT::vector(VecTy.getNumElements(),
                                     WideTy.getSizeInBits()),
                     1, TargetOpcode::G_SEXT);

      widenScalarDst(MI, WideTy, 0);
      Observer.changedInstr(MI);
      return Legalized;
    }

    if (TypeIdx != 2)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    // TODO: Probably should be zext
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_SEXT);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_INSERT_VECTOR_ELT: {
    if (TypeIdx == 1) {
      Observer.changingInstr(MI);

      Register VecReg = MI.getOperand(1).getReg();
      LLT VecTy = MRI.getType(VecReg);
      LLT WideVecTy = LLT::vector(VecTy.getNumElements(), WideTy);

      widenScalarSrc(MI, WideVecTy, 1, TargetOpcode::G_ANYEXT);
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ANYEXT);
      widenScalarDst(MI, WideVecTy, 0);
      Observer.changedInstr(MI);
      return Legalized;
    }

    if (TypeIdx == 2) {
      Observer.changingInstr(MI);
      // TODO: Probably should be zext
      widenScalarSrc(MI, WideTy, 3, TargetOpcode::G_SEXT);
      Observer.changedInstr(MI);
    }

    return Legalized;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FMAD:
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_FABS:
  case TargetOpcode::G_FCANONICALIZE:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMAXNUM_IEEE:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_INTRINSIC_ROUND:
    assert(TypeIdx == 0);
    Observer.changingInstr(MI);

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I)
      widenScalarSrc(MI, WideTy, I, TargetOpcode::G_FPEXT);

    widenScalarDst(MI, WideTy, 0, TargetOpcode::G_FPTRUNC);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_INTTOPTR:
    if (TypeIdx != 1)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PTRTOINT:
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_BUILD_VECTOR: {
    Observer.changingInstr(MI);

    const LLT WideEltTy = TypeIdx == 1 ? WideTy : WideTy.getElementType();
    for (int I = 1, E = MI.getNumOperands(); I != E; ++I)
      widenScalarSrc(MI, WideEltTy, I, TargetOpcode::G_ANYEXT);

    // Avoid changing the result vector type if the source element type was
    // requested.
    if (TypeIdx == 1) {
      auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
      MI.setDesc(TII.get(TargetOpcode::G_BUILD_VECTOR_TRUNC));
    } else {
      widenScalarDst(MI, WideTy, 0);
    }

    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_SEXT_INREG:
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    widenScalarDst(MI, WideTy, 0, TargetOpcode::G_TRUNC);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PTRMASK: {
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  }
  }
}

static void getUnmergePieces(SmallVectorImpl<Register> &Pieces,
                             MachineIRBuilder &B, Register Src, LLT Ty) {
  auto Unmerge = B.buildUnmerge(Ty, Src);
  for (int I = 0, E = Unmerge->getNumOperands() - 1; I != E; ++I)
    Pieces.push_back(Unmerge.getReg(I));
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerBitcast(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (SrcTy.isVector() && !DstTy.isVector()) {
    SmallVector<Register, 8> SrcRegs;
    getUnmergePieces(SrcRegs, MIRBuilder, Src, SrcTy.getElementType());
    MIRBuilder.buildMerge(Dst, SrcRegs);
    MI.eraseFromParent();
    return Legalized;
  }

  if (DstTy.isVector() && !SrcTy.isVector()) {
    SmallVector<Register, 8> SrcRegs;
    getUnmergePieces(SrcRegs, MIRBuilder, Src, DstTy.getElementType());
    MIRBuilder.buildMerge(Dst, SrcRegs);
    MI.eraseFromParent();
    return Legalized;
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::bitcast(MachineInstr &MI, unsigned TypeIdx, LLT CastTy) {
  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  case TargetOpcode::G_LOAD: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    bitcastDst(MI, CastTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_STORE: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    bitcastSrc(MI, CastTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_SELECT: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    if (MRI.getType(MI.getOperand(1).getReg()).isVector()) {
      LLVM_DEBUG(
          dbgs() << "bitcast action not implemented for vector select\n");
      return UnableToLegalize;
    }

    Observer.changingInstr(MI);
    bitcastSrc(MI, CastTy, 2);
    bitcastSrc(MI, CastTy, 3);
    bitcastDst(MI, CastTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    Observer.changingInstr(MI);
    bitcastSrc(MI, CastTy, 1);
    bitcastSrc(MI, CastTy, 2);
    bitcastDst(MI, CastTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  default:
    return UnableToLegalize;
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lower(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  using namespace TargetOpcode;
  MIRBuilder.setInstrAndDebugLoc(MI);

  switch(MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_BITCAST:
    return lowerBitcast(MI);
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM: {
    auto Quot =
        MIRBuilder.buildInstr(MI.getOpcode() == G_SREM ? G_SDIV : G_UDIV, {Ty},
                              {MI.getOperand(1), MI.getOperand(2)});

    auto Prod = MIRBuilder.buildMul(Ty, Quot, MI.getOperand(2));
    MIRBuilder.buildSub(MI.getOperand(0), MI.getOperand(1), Prod);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SADDO:
  case TargetOpcode::G_SSUBO:
    return lowerSADDO_SSUBO(MI);
  case TargetOpcode::G_SMULO:
  case TargetOpcode::G_UMULO: {
    // Generate G_UMULH/G_SMULH to check for overflow and a normal G_MUL for the
    // result.
    Register Res = MI.getOperand(0).getReg();
    Register Overflow = MI.getOperand(1).getReg();
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();

    unsigned Opcode = MI.getOpcode() == TargetOpcode::G_SMULO
                          ? TargetOpcode::G_SMULH
                          : TargetOpcode::G_UMULH;

    Observer.changingInstr(MI);
    const auto &TII = MIRBuilder.getTII();
    MI.setDesc(TII.get(TargetOpcode::G_MUL));
    MI.RemoveOperand(1);
    Observer.changedInstr(MI);

    MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());

    auto HiPart = MIRBuilder.buildInstr(Opcode, {Ty}, {LHS, RHS});
    auto Zero = MIRBuilder.buildConstant(Ty, 0);

    // For *signed* multiply, overflow is detected by checking:
    // (hi != (lo >> bitwidth-1))
    if (Opcode == TargetOpcode::G_SMULH) {
      auto ShiftAmt = MIRBuilder.buildConstant(Ty, Ty.getSizeInBits() - 1);
      auto Shifted = MIRBuilder.buildAShr(Ty, Res, ShiftAmt);
      MIRBuilder.buildICmp(CmpInst::ICMP_NE, Overflow, HiPart, Shifted);
    } else {
      MIRBuilder.buildICmp(CmpInst::ICMP_NE, Overflow, HiPart, Zero);
    }
    return Legalized;
  }
  case TargetOpcode::G_FNEG: {
    // TODO: Handle vector types once we are able to
    // represent them.
    if (Ty.isVector())
      return UnableToLegalize;
    Register Res = MI.getOperand(0).getReg();
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    Type *ZeroTy = getFloatTypeForLLT(Ctx, Ty);
    if (!ZeroTy)
      return UnableToLegalize;
    ConstantFP &ZeroForNegation =
        *cast<ConstantFP>(ConstantFP::getZeroValueForNegation(ZeroTy));
    auto Zero = MIRBuilder.buildFConstant(Ty, ZeroForNegation);
    Register SubByReg = MI.getOperand(1).getReg();
    Register ZeroReg = Zero.getReg(0);
    MIRBuilder.buildFSub(Res, ZeroReg, SubByReg, MI.getFlags());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FSUB: {
    // Lower (G_FSUB LHS, RHS) to (G_FADD LHS, (G_FNEG RHS)).
    // First, check if G_FNEG is marked as Lower. If so, we may
    // end up with an infinite loop as G_FSUB is used to legalize G_FNEG.
    if (LI.getAction({G_FNEG, {Ty}}).Action == Lower)
      return UnableToLegalize;
    Register Res = MI.getOperand(0).getReg();
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();
    Register Neg = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildFNeg(Neg, RHS);
    MIRBuilder.buildFAdd(Res, LHS, Neg, MI.getFlags());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FMAD:
    return lowerFMad(MI);
  case TargetOpcode::G_FFLOOR:
    return lowerFFloor(MI);
  case TargetOpcode::G_INTRINSIC_ROUND:
    return lowerIntrinsicRound(MI);
  case TargetOpcode::G_ATOMIC_CMPXCHG_WITH_SUCCESS: {
    Register OldValRes = MI.getOperand(0).getReg();
    Register SuccessRes = MI.getOperand(1).getReg();
    Register Addr = MI.getOperand(2).getReg();
    Register CmpVal = MI.getOperand(3).getReg();
    Register NewVal = MI.getOperand(4).getReg();
    MIRBuilder.buildAtomicCmpXchg(OldValRes, Addr, CmpVal, NewVal,
                                  **MI.memoperands_begin());
    MIRBuilder.buildICmp(CmpInst::ICMP_EQ, SuccessRes, OldValRes, CmpVal);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD: {
    // Lower to a memory-width G_LOAD and a G_SEXT/G_ZEXT/G_ANYEXT
    Register DstReg = MI.getOperand(0).getReg();
    Register PtrReg = MI.getOperand(1).getReg();
    LLT DstTy = MRI.getType(DstReg);
    auto &MMO = **MI.memoperands_begin();

    if (DstTy.getSizeInBits() == MMO.getSizeInBits()) {
      if (MI.getOpcode() == TargetOpcode::G_LOAD) {
        // This load needs splitting into power of 2 sized loads.
        if (DstTy.isVector())
          return UnableToLegalize;
        if (isPowerOf2_32(DstTy.getSizeInBits()))
          return UnableToLegalize; // Don't know what we're being asked to do.

        // Our strategy here is to generate anyextending loads for the smaller
        // types up to next power-2 result type, and then combine the two larger
        // result values together, before truncating back down to the non-pow-2
        // type.
        // E.g. v1 = i24 load =>
        // v2 = i32 zextload (2 byte)
        // v3 = i32 load (1 byte)
        // v4 = i32 shl v3, 16
        // v5 = i32 or v4, v2
        // v1 = i24 trunc v5
        // By doing this we generate the correct truncate which should get
        // combined away as an artifact with a matching extend.
        uint64_t LargeSplitSize = PowerOf2Floor(DstTy.getSizeInBits());
        uint64_t SmallSplitSize = DstTy.getSizeInBits() - LargeSplitSize;

        MachineFunction &MF = MIRBuilder.getMF();
        MachineMemOperand *LargeMMO =
            MF.getMachineMemOperand(&MMO, 0, LargeSplitSize / 8);
        MachineMemOperand *SmallMMO = MF.getMachineMemOperand(
            &MMO, LargeSplitSize / 8, SmallSplitSize / 8);

        LLT PtrTy = MRI.getType(PtrReg);
        unsigned AnyExtSize = NextPowerOf2(DstTy.getSizeInBits());
        LLT AnyExtTy = LLT::scalar(AnyExtSize);
        Register LargeLdReg = MRI.createGenericVirtualRegister(AnyExtTy);
        Register SmallLdReg = MRI.createGenericVirtualRegister(AnyExtTy);
        auto LargeLoad = MIRBuilder.buildLoadInstr(
            TargetOpcode::G_ZEXTLOAD, LargeLdReg, PtrReg, *LargeMMO);

        auto OffsetCst = MIRBuilder.buildConstant(
            LLT::scalar(PtrTy.getSizeInBits()), LargeSplitSize / 8);
        Register PtrAddReg = MRI.createGenericVirtualRegister(PtrTy);
        auto SmallPtr =
            MIRBuilder.buildPtrAdd(PtrAddReg, PtrReg, OffsetCst.getReg(0));
        auto SmallLoad = MIRBuilder.buildLoad(SmallLdReg, SmallPtr.getReg(0),
                                              *SmallMMO);

        auto ShiftAmt = MIRBuilder.buildConstant(AnyExtTy, LargeSplitSize);
        auto Shift = MIRBuilder.buildShl(AnyExtTy, SmallLoad, ShiftAmt);
        auto Or = MIRBuilder.buildOr(AnyExtTy, Shift, LargeLoad);
        MIRBuilder.buildTrunc(DstReg, {Or.getReg(0)});
        MI.eraseFromParent();
        return Legalized;
      }
      MIRBuilder.buildLoad(DstReg, PtrReg, MMO);
      MI.eraseFromParent();
      return Legalized;
    }

    if (DstTy.isScalar()) {
      Register TmpReg =
          MRI.createGenericVirtualRegister(LLT::scalar(MMO.getSizeInBits()));
      MIRBuilder.buildLoad(TmpReg, PtrReg, MMO);
      switch (MI.getOpcode()) {
      default:
        llvm_unreachable("Unexpected opcode");
      case TargetOpcode::G_LOAD:
        MIRBuilder.buildExtOrTrunc(TargetOpcode::G_ANYEXT, DstReg, TmpReg);
        break;
      case TargetOpcode::G_SEXTLOAD:
        MIRBuilder.buildSExt(DstReg, TmpReg);
        break;
      case TargetOpcode::G_ZEXTLOAD:
        MIRBuilder.buildZExt(DstReg, TmpReg);
        break;
      }
      MI.eraseFromParent();
      return Legalized;
    }

    return UnableToLegalize;
  }
  case TargetOpcode::G_STORE: {
    // Lower a non-power of 2 store into multiple pow-2 stores.
    // E.g. split an i24 store into an i16 store + i8 store.
    // We do this by first extending the stored value to the next largest power
    // of 2 type, and then using truncating stores to store the components.
    // By doing this, likewise with G_LOAD, generate an extend that can be
    // artifact-combined away instead of leaving behind extracts.
    Register SrcReg = MI.getOperand(0).getReg();
    Register PtrReg = MI.getOperand(1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    MachineMemOperand &MMO = **MI.memoperands_begin();
    if (SrcTy.getSizeInBits() != MMO.getSizeInBits())
      return UnableToLegalize;
    if (SrcTy.isVector())
      return UnableToLegalize;
    if (isPowerOf2_32(SrcTy.getSizeInBits()))
      return UnableToLegalize; // Don't know what we're being asked to do.

    // Extend to the next pow-2.
    const LLT ExtendTy = LLT::scalar(NextPowerOf2(SrcTy.getSizeInBits()));
    auto ExtVal = MIRBuilder.buildAnyExt(ExtendTy, SrcReg);

    // Obtain the smaller value by shifting away the larger value.
    uint64_t LargeSplitSize = PowerOf2Floor(SrcTy.getSizeInBits());
    uint64_t SmallSplitSize = SrcTy.getSizeInBits() - LargeSplitSize;
    auto ShiftAmt = MIRBuilder.buildConstant(ExtendTy, LargeSplitSize);
    auto SmallVal = MIRBuilder.buildLShr(ExtendTy, ExtVal, ShiftAmt);

    // Generate the PtrAdd and truncating stores.
    LLT PtrTy = MRI.getType(PtrReg);
    auto OffsetCst = MIRBuilder.buildConstant(
            LLT::scalar(PtrTy.getSizeInBits()), LargeSplitSize / 8);
    Register PtrAddReg = MRI.createGenericVirtualRegister(PtrTy);
    auto SmallPtr =
        MIRBuilder.buildPtrAdd(PtrAddReg, PtrReg, OffsetCst.getReg(0));

    MachineFunction &MF = MIRBuilder.getMF();
    MachineMemOperand *LargeMMO =
        MF.getMachineMemOperand(&MMO, 0, LargeSplitSize / 8);
    MachineMemOperand *SmallMMO =
        MF.getMachineMemOperand(&MMO, LargeSplitSize / 8, SmallSplitSize / 8);
    MIRBuilder.buildStore(ExtVal.getReg(0), PtrReg, *LargeMMO);
    MIRBuilder.buildStore(SmallVal.getReg(0), SmallPtr.getReg(0), *SmallMMO);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTPOP:
    return lowerBitCount(MI, TypeIdx, Ty);
  case G_UADDO: {
    Register Res = MI.getOperand(0).getReg();
    Register CarryOut = MI.getOperand(1).getReg();
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();

    MIRBuilder.buildAdd(Res, LHS, RHS);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CarryOut, Res, RHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_UADDE: {
    Register Res = MI.getOperand(0).getReg();
    Register CarryOut = MI.getOperand(1).getReg();
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();
    Register CarryIn = MI.getOperand(4).getReg();
    LLT Ty = MRI.getType(Res);

    auto TmpRes = MIRBuilder.buildAdd(Ty, LHS, RHS);
    auto ZExtCarryIn = MIRBuilder.buildZExt(Ty, CarryIn);
    MIRBuilder.buildAdd(Res, TmpRes, ZExtCarryIn);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CarryOut, Res, LHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_USUBO: {
    Register Res = MI.getOperand(0).getReg();
    Register BorrowOut = MI.getOperand(1).getReg();
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();

    MIRBuilder.buildSub(Res, LHS, RHS);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, BorrowOut, LHS, RHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_USUBE: {
    Register Res = MI.getOperand(0).getReg();
    Register BorrowOut = MI.getOperand(1).getReg();
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();
    Register BorrowIn = MI.getOperand(4).getReg();
    const LLT CondTy = MRI.getType(BorrowOut);
    const LLT Ty = MRI.getType(Res);

    auto TmpRes = MIRBuilder.buildSub(Ty, LHS, RHS);
    auto ZExtBorrowIn = MIRBuilder.buildZExt(Ty, BorrowIn);
    MIRBuilder.buildSub(Res, TmpRes, ZExtBorrowIn);

    auto LHS_EQ_RHS = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, CondTy, LHS, RHS);
    auto LHS_ULT_RHS = MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CondTy, LHS, RHS);
    MIRBuilder.buildSelect(BorrowOut, LHS_EQ_RHS, BorrowIn, LHS_ULT_RHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_UITOFP:
    return lowerUITOFP(MI, TypeIdx, Ty);
  case G_SITOFP:
    return lowerSITOFP(MI, TypeIdx, Ty);
  case G_FPTOUI:
    return lowerFPTOUI(MI, TypeIdx, Ty);
  case G_FPTOSI:
    return lowerFPTOSI(MI);
  case G_FPTRUNC:
    return lowerFPTRUNC(MI, TypeIdx, Ty);
  case G_SMIN:
  case G_SMAX:
  case G_UMIN:
  case G_UMAX:
    return lowerMinMax(MI, TypeIdx, Ty);
  case G_FCOPYSIGN:
    return lowerFCopySign(MI, TypeIdx, Ty);
  case G_FMINNUM:
  case G_FMAXNUM:
    return lowerFMinNumMaxNum(MI);
  case G_MERGE_VALUES:
    return lowerMergeValues(MI);
  case G_UNMERGE_VALUES:
    return lowerUnmergeValues(MI);
  case TargetOpcode::G_SEXT_INREG: {
    assert(MI.getOperand(2).isImm() && "Expected immediate");
    int64_t SizeInBits = MI.getOperand(2).getImm();

    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    LLT DstTy = MRI.getType(DstReg);
    Register TmpRes = MRI.createGenericVirtualRegister(DstTy);

    auto MIBSz = MIRBuilder.buildConstant(DstTy, DstTy.getScalarSizeInBits() - SizeInBits);
    MIRBuilder.buildShl(TmpRes, SrcReg, MIBSz->getOperand(0));
    MIRBuilder.buildAShr(DstReg, TmpRes, MIBSz->getOperand(0));
    MI.eraseFromParent();
    return Legalized;
  }
  case G_SHUFFLE_VECTOR:
    return lowerShuffleVector(MI);
  case G_DYN_STACKALLOC:
    return lowerDynStackAlloc(MI);
  case G_EXTRACT:
    return lowerExtract(MI);
  case G_INSERT:
    return lowerInsert(MI);
  case G_BSWAP:
    return lowerBswap(MI);
  case G_BITREVERSE:
    return lowerBitreverse(MI);
  case G_READ_REGISTER:
  case G_WRITE_REGISTER:
    return lowerReadWriteRegister(MI);
  }
}

LegalizerHelper::LegalizeResult LegalizerHelper::fewerElementsVectorImplicitDef(
    MachineInstr &MI, unsigned TypeIdx, LLT NarrowTy) {
  SmallVector<Register, 2> DstRegs;

  unsigned NarrowSize = NarrowTy.getSizeInBits();
  Register DstReg = MI.getOperand(0).getReg();
  unsigned Size = MRI.getType(DstReg).getSizeInBits();
  int NumParts = Size / NarrowSize;
  // FIXME: Don't know how to handle the situation where the small vectors
  // aren't all the same size yet.
  if (Size % NarrowSize != 0)
    return UnableToLegalize;

  for (int i = 0; i < NumParts; ++i) {
    Register TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
    MIRBuilder.buildUndef(TmpReg);
    DstRegs.push_back(TmpReg);
  }

  if (NarrowTy.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

// Handle splitting vector operations which need to have the same number of
// elements in each type index, but each type index may have a different element
// type.
//
// e.g.  <4 x s64> = G_SHL <4 x s64>, <4 x s32> ->
//       <2 x s64> = G_SHL <2 x s64>, <2 x s32>
//       <2 x s64> = G_SHL <2 x s64>, <2 x s32>
//
// Also handles some irregular breakdown cases, e.g.
// e.g.  <3 x s64> = G_SHL <3 x s64>, <3 x s32> ->
//       <2 x s64> = G_SHL <2 x s64>, <2 x s32>
//             s64 = G_SHL s64, s32
LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorMultiEltType(
  MachineInstr &MI, unsigned TypeIdx, LLT NarrowTyArg) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  const LLT NarrowTy0 = NarrowTyArg;
  const unsigned NewNumElts =
      NarrowTy0.isVector() ? NarrowTy0.getNumElements() : 1;

  const Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT LeftoverTy0;

  // All of the operands need to have the same number of elements, so if we can
  // determine a type breakdown for the result type, we can for all of the
  // source types.
  int NumParts = getNarrowTypeBreakDown(DstTy, NarrowTy0, LeftoverTy0).first;
  if (NumParts < 0)
    return UnableToLegalize;

  SmallVector<MachineInstrBuilder, 4> NewInsts;

  SmallVector<Register, 4> DstRegs, LeftoverDstRegs;
  SmallVector<Register, 4> PartRegs, LeftoverRegs;

  for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I) {
    LLT LeftoverTy;
    Register SrcReg = MI.getOperand(I).getReg();
    LLT SrcTyI = MRI.getType(SrcReg);
    LLT NarrowTyI = LLT::scalarOrVector(NewNumElts, SrcTyI.getScalarType());
    LLT LeftoverTyI;

    // Split this operand into the requested typed registers, and any leftover
    // required to reproduce the original type.
    if (!extractParts(SrcReg, SrcTyI, NarrowTyI, LeftoverTyI, PartRegs,
                      LeftoverRegs))
      return UnableToLegalize;

    if (I == 1) {
      // For the first operand, create an instruction for each part and setup
      // the result.
      for (Register PartReg : PartRegs) {
        Register PartDstReg = MRI.createGenericVirtualRegister(NarrowTy0);
        NewInsts.push_back(MIRBuilder.buildInstrNoInsert(MI.getOpcode())
                               .addDef(PartDstReg)
                               .addUse(PartReg));
        DstRegs.push_back(PartDstReg);
      }

      for (Register LeftoverReg : LeftoverRegs) {
        Register PartDstReg = MRI.createGenericVirtualRegister(LeftoverTy0);
        NewInsts.push_back(MIRBuilder.buildInstrNoInsert(MI.getOpcode())
                               .addDef(PartDstReg)
                               .addUse(LeftoverReg));
        LeftoverDstRegs.push_back(PartDstReg);
      }
    } else {
      assert(NewInsts.size() == PartRegs.size() + LeftoverRegs.size());

      // Add the newly created operand splits to the existing instructions. The
      // odd-sized pieces are ordered after the requested NarrowTyArg sized
      // pieces.
      unsigned InstCount = 0;
      for (unsigned J = 0, JE = PartRegs.size(); J != JE; ++J)
        NewInsts[InstCount++].addUse(PartRegs[J]);
      for (unsigned J = 0, JE = LeftoverRegs.size(); J != JE; ++J)
        NewInsts[InstCount++].addUse(LeftoverRegs[J]);
    }

    PartRegs.clear();
    LeftoverRegs.clear();
  }

  // Insert the newly built operations and rebuild the result register.
  for (auto &MIB : NewInsts)
    MIRBuilder.insertInstr(MIB);

  insertParts(DstReg, DstTy, NarrowTy0, DstRegs, LeftoverTy0, LeftoverDstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorCasts(MachineInstr &MI, unsigned TypeIdx,
                                          LLT NarrowTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);

  LLT NarrowTy0 = NarrowTy;
  LLT NarrowTy1;
  unsigned NumParts;

  if (NarrowTy.isVector()) {
    // Uneven breakdown not handled.
    NumParts = DstTy.getNumElements() / NarrowTy.getNumElements();
    if (NumParts * NarrowTy.getNumElements() != DstTy.getNumElements())
      return UnableToLegalize;

    NarrowTy1 = LLT::vector(NumParts, SrcTy.getElementType().getSizeInBits());
  } else {
    NumParts = DstTy.getNumElements();
    NarrowTy1 = SrcTy.getElementType();
  }

  SmallVector<Register, 4> SrcRegs, DstRegs;
  extractParts(SrcReg, NarrowTy1, NumParts, SrcRegs);

  for (unsigned I = 0; I < NumParts; ++I) {
    Register DstReg = MRI.createGenericVirtualRegister(NarrowTy0);
    MachineInstr *NewInst =
        MIRBuilder.buildInstr(MI.getOpcode(), {DstReg}, {SrcRegs[I]});

    NewInst->setFlags(MI.getFlags());
    DstRegs.push_back(DstReg);
  }

  if (NarrowTy.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorCmp(MachineInstr &MI, unsigned TypeIdx,
                                        LLT NarrowTy) {
  Register DstReg = MI.getOperand(0).getReg();
  Register Src0Reg = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(Src0Reg);

  unsigned NumParts;
  LLT NarrowTy0, NarrowTy1;

  if (TypeIdx == 0) {
    unsigned NewElts = NarrowTy.isVector() ? NarrowTy.getNumElements() : 1;
    unsigned OldElts = DstTy.getNumElements();

    NarrowTy0 = NarrowTy;
    NumParts = NarrowTy.isVector() ? (OldElts / NewElts) : DstTy.getNumElements();
    NarrowTy1 = NarrowTy.isVector() ?
      LLT::vector(NarrowTy.getNumElements(), SrcTy.getScalarSizeInBits()) :
      SrcTy.getElementType();

  } else {
    unsigned NewElts = NarrowTy.isVector() ? NarrowTy.getNumElements() : 1;
    unsigned OldElts = SrcTy.getNumElements();

    NumParts = NarrowTy.isVector() ? (OldElts / NewElts) :
      NarrowTy.getNumElements();
    NarrowTy0 = LLT::vector(NarrowTy.getNumElements(),
                            DstTy.getScalarSizeInBits());
    NarrowTy1 = NarrowTy;
  }

  // FIXME: Don't know how to handle the situation where the small vectors
  // aren't all the same size yet.
  if (NarrowTy1.isVector() &&
      NarrowTy1.getNumElements() * NumParts != DstTy.getNumElements())
    return UnableToLegalize;

  CmpInst::Predicate Pred
    = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());

  SmallVector<Register, 2> Src1Regs, Src2Regs, DstRegs;
  extractParts(MI.getOperand(2).getReg(), NarrowTy1, NumParts, Src1Regs);
  extractParts(MI.getOperand(3).getReg(), NarrowTy1, NumParts, Src2Regs);

  for (unsigned I = 0; I < NumParts; ++I) {
    Register DstReg = MRI.createGenericVirtualRegister(NarrowTy0);
    DstRegs.push_back(DstReg);

    if (MI.getOpcode() == TargetOpcode::G_ICMP)
      MIRBuilder.buildICmp(Pred, DstReg, Src1Regs[I], Src2Regs[I]);
    else {
      MachineInstr *NewCmp
        = MIRBuilder.buildFCmp(Pred, DstReg, Src1Regs[I], Src2Regs[I]);
      NewCmp->setFlags(MI.getFlags());
    }
  }

  if (NarrowTy1.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorSelect(MachineInstr &MI, unsigned TypeIdx,
                                           LLT NarrowTy) {
  Register DstReg = MI.getOperand(0).getReg();
  Register CondReg = MI.getOperand(1).getReg();

  unsigned NumParts = 0;
  LLT NarrowTy0, NarrowTy1;

  LLT DstTy = MRI.getType(DstReg);
  LLT CondTy = MRI.getType(CondReg);
  unsigned Size = DstTy.getSizeInBits();

  assert(TypeIdx == 0 || CondTy.isVector());

  if (TypeIdx == 0) {
    NarrowTy0 = NarrowTy;
    NarrowTy1 = CondTy;

    unsigned NarrowSize = NarrowTy0.getSizeInBits();
    // FIXME: Don't know how to handle the situation where the small vectors
    // aren't all the same size yet.
    if (Size % NarrowSize != 0)
      return UnableToLegalize;

    NumParts = Size / NarrowSize;

    // Need to break down the condition type
    if (CondTy.isVector()) {
      if (CondTy.getNumElements() == NumParts)
        NarrowTy1 = CondTy.getElementType();
      else
        NarrowTy1 = LLT::vector(CondTy.getNumElements() / NumParts,
                                CondTy.getScalarSizeInBits());
    }
  } else {
    NumParts = CondTy.getNumElements();
    if (NarrowTy.isVector()) {
      // TODO: Handle uneven breakdown.
      if (NumParts * NarrowTy.getNumElements() != CondTy.getNumElements())
        return UnableToLegalize;

      return UnableToLegalize;
    } else {
      NarrowTy0 = DstTy.getElementType();
      NarrowTy1 = NarrowTy;
    }
  }

  SmallVector<Register, 2> DstRegs, Src0Regs, Src1Regs, Src2Regs;
  if (CondTy.isVector())
    extractParts(MI.getOperand(1).getReg(), NarrowTy1, NumParts, Src0Regs);

  extractParts(MI.getOperand(2).getReg(), NarrowTy0, NumParts, Src1Regs);
  extractParts(MI.getOperand(3).getReg(), NarrowTy0, NumParts, Src2Regs);

  for (unsigned i = 0; i < NumParts; ++i) {
    Register DstReg = MRI.createGenericVirtualRegister(NarrowTy0);
    MIRBuilder.buildSelect(DstReg, CondTy.isVector() ? Src0Regs[i] : CondReg,
                           Src1Regs[i], Src2Regs[i]);
    DstRegs.push_back(DstReg);
  }

  if (NarrowTy0.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorPhi(MachineInstr &MI, unsigned TypeIdx,
                                        LLT NarrowTy) {
  const Register DstReg = MI.getOperand(0).getReg();
  LLT PhiTy = MRI.getType(DstReg);
  LLT LeftoverTy;

  // All of the operands need to have the same number of elements, so if we can
  // determine a type breakdown for the result type, we can for all of the
  // source types.
  int NumParts, NumLeftover;
  std::tie(NumParts, NumLeftover)
    = getNarrowTypeBreakDown(PhiTy, NarrowTy, LeftoverTy);
  if (NumParts < 0)
    return UnableToLegalize;

  SmallVector<Register, 4> DstRegs, LeftoverDstRegs;
  SmallVector<MachineInstrBuilder, 4> NewInsts;

  const int TotalNumParts = NumParts + NumLeftover;

  // Insert the new phis in the result block first.
  for (int I = 0; I != TotalNumParts; ++I) {
    LLT Ty = I < NumParts ? NarrowTy : LeftoverTy;
    Register PartDstReg = MRI.createGenericVirtualRegister(Ty);
    NewInsts.push_back(MIRBuilder.buildInstr(TargetOpcode::G_PHI)
                       .addDef(PartDstReg));
    if (I < NumParts)
      DstRegs.push_back(PartDstReg);
    else
      LeftoverDstRegs.push_back(PartDstReg);
  }

  MachineBasicBlock *MBB = MI.getParent();
  MIRBuilder.setInsertPt(*MBB, MBB->getFirstNonPHI());
  insertParts(DstReg, PhiTy, NarrowTy, DstRegs, LeftoverTy, LeftoverDstRegs);

  SmallVector<Register, 4> PartRegs, LeftoverRegs;

  // Insert code to extract the incoming values in each predecessor block.
  for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
    PartRegs.clear();
    LeftoverRegs.clear();

    Register SrcReg = MI.getOperand(I).getReg();
    MachineBasicBlock &OpMBB = *MI.getOperand(I + 1).getMBB();
    MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());

    LLT Unused;
    if (!extractParts(SrcReg, PhiTy, NarrowTy, Unused, PartRegs,
                      LeftoverRegs))
      return UnableToLegalize;

    // Add the newly created operand splits to the existing instructions. The
    // odd-sized pieces are ordered after the requested NarrowTyArg sized
    // pieces.
    for (int J = 0; J != TotalNumParts; ++J) {
      MachineInstrBuilder MIB = NewInsts[J];
      MIB.addUse(J < NumParts ? PartRegs[J] : LeftoverRegs[J - NumParts]);
      MIB.addMBB(&OpMBB);
    }
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorUnmergeValues(MachineInstr &MI,
                                                  unsigned TypeIdx,
                                                  LLT NarrowTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  const int NumDst = MI.getNumOperands() - 1;
  const Register SrcReg = MI.getOperand(NumDst).getReg();
  LLT SrcTy = MRI.getType(SrcReg);

  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  // TODO: Create sequence of extracts.
  if (DstTy == NarrowTy)
    return UnableToLegalize;

  LLT GCDTy = getGCDType(SrcTy, NarrowTy);
  if (DstTy == GCDTy) {
    // This would just be a copy of the same unmerge.
    // TODO: Create extracts, pad with undef and create intermediate merges.
    return UnableToLegalize;
  }

  auto Unmerge = MIRBuilder.buildUnmerge(GCDTy, SrcReg);
  const int NumUnmerge = Unmerge->getNumOperands() - 1;
  const int PartsPerUnmerge = NumDst / NumUnmerge;

  for (int I = 0; I != NumUnmerge; ++I) {
    auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_UNMERGE_VALUES);

    for (int J = 0; J != PartsPerUnmerge; ++J)
      MIB.addDef(MI.getOperand(I * PartsPerUnmerge + J).getReg());
    MIB.addUse(Unmerge.getReg(I));
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorBuildVector(MachineInstr &MI,
                                                unsigned TypeIdx,
                                                LLT NarrowTy) {
  assert(TypeIdx == 0 && "not a vector type index");
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = DstTy.getElementType();

  int DstNumElts = DstTy.getNumElements();
  int NarrowNumElts = NarrowTy.getNumElements();
  int NumConcat = (DstNumElts + NarrowNumElts - 1) / NarrowNumElts;
  LLT WidenedDstTy = LLT::vector(NarrowNumElts * NumConcat, SrcTy);

  SmallVector<Register, 8> ConcatOps;
  SmallVector<Register, 8> SubBuildVector;

  Register UndefReg;
  if (WidenedDstTy != DstTy)
    UndefReg = MIRBuilder.buildUndef(SrcTy).getReg(0);

  // Create a G_CONCAT_VECTORS of NarrowTy pieces, padding with undef as
  // necessary.
  //
  // %3:_(<3 x s16>) = G_BUILD_VECTOR %0, %1, %2
  //   -> <2 x s16>
  //
  // %4:_(s16) = G_IMPLICIT_DEF
  // %5:_(<2 x s16>) = G_BUILD_VECTOR %0, %1
  // %6:_(<2 x s16>) = G_BUILD_VECTOR %2, %4
  // %7:_(<4 x s16>) = G_CONCAT_VECTORS %5, %6
  // %3:_(<3 x s16>) = G_EXTRACT %7, 0
  for (int I = 0; I != NumConcat; ++I) {
    for (int J = 0; J != NarrowNumElts; ++J) {
      int SrcIdx = NarrowNumElts * I + J;

      if (SrcIdx < DstNumElts) {
        Register SrcReg = MI.getOperand(SrcIdx + 1).getReg();
        SubBuildVector.push_back(SrcReg);
      } else
        SubBuildVector.push_back(UndefReg);
    }

    auto BuildVec = MIRBuilder.buildBuildVector(NarrowTy, SubBuildVector);
    ConcatOps.push_back(BuildVec.getReg(0));
    SubBuildVector.clear();
  }

  if (DstTy == WidenedDstTy)
    MIRBuilder.buildConcatVectors(DstReg, ConcatOps);
  else {
    auto Concat = MIRBuilder.buildConcatVectors(WidenedDstTy, ConcatOps);
    MIRBuilder.buildExtract(DstReg, Concat, 0);
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::reduceLoadStoreWidth(MachineInstr &MI, unsigned TypeIdx,
                                      LLT NarrowTy) {
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0)
    return UnableToLegalize;

  MachineMemOperand *MMO = *MI.memoperands_begin();

  // This implementation doesn't work for atomics. Give up instead of doing
  // something invalid.
  if (MMO->getOrdering() != AtomicOrdering::NotAtomic ||
      MMO->getFailureOrdering() != AtomicOrdering::NotAtomic)
    return UnableToLegalize;

  bool IsLoad = MI.getOpcode() == TargetOpcode::G_LOAD;
  Register ValReg = MI.getOperand(0).getReg();
  Register AddrReg = MI.getOperand(1).getReg();
  LLT ValTy = MRI.getType(ValReg);

  // FIXME: Do we need a distinct NarrowMemory legalize action?
  if (ValTy.getSizeInBits() != 8 * MMO->getSize()) {
    LLVM_DEBUG(dbgs() << "Can't narrow extload/truncstore\n");
    return UnableToLegalize;
  }

  int NumParts = -1;
  int NumLeftover = -1;
  LLT LeftoverTy;
  SmallVector<Register, 8> NarrowRegs, NarrowLeftoverRegs;
  if (IsLoad) {
    std::tie(NumParts, NumLeftover) = getNarrowTypeBreakDown(ValTy, NarrowTy, LeftoverTy);
  } else {
    if (extractParts(ValReg, ValTy, NarrowTy, LeftoverTy, NarrowRegs,
                     NarrowLeftoverRegs)) {
      NumParts = NarrowRegs.size();
      NumLeftover = NarrowLeftoverRegs.size();
    }
  }

  if (NumParts == -1)
    return UnableToLegalize;

  const LLT OffsetTy = LLT::scalar(MRI.getType(AddrReg).getScalarSizeInBits());

  unsigned TotalSize = ValTy.getSizeInBits();

  // Split the load/store into PartTy sized pieces starting at Offset. If this
  // is a load, return the new registers in ValRegs. For a store, each elements
  // of ValRegs should be PartTy. Returns the next offset that needs to be
  // handled.
  auto splitTypePieces = [=](LLT PartTy, SmallVectorImpl<Register> &ValRegs,
                             unsigned Offset) -> unsigned {
    MachineFunction &MF = MIRBuilder.getMF();
    unsigned PartSize = PartTy.getSizeInBits();
    for (unsigned Idx = 0, E = NumParts; Idx != E && Offset < TotalSize;
         Offset += PartSize, ++Idx) {
      unsigned ByteSize = PartSize / 8;
      unsigned ByteOffset = Offset / 8;
      Register NewAddrReg;

      MIRBuilder.materializePtrAdd(NewAddrReg, AddrReg, OffsetTy, ByteOffset);

      MachineMemOperand *NewMMO =
        MF.getMachineMemOperand(MMO, ByteOffset, ByteSize);

      if (IsLoad) {
        Register Dst = MRI.createGenericVirtualRegister(PartTy);
        ValRegs.push_back(Dst);
        MIRBuilder.buildLoad(Dst, NewAddrReg, *NewMMO);
      } else {
        MIRBuilder.buildStore(ValRegs[Idx], NewAddrReg, *NewMMO);
      }
    }

    return Offset;
  };

  unsigned HandledOffset = splitTypePieces(NarrowTy, NarrowRegs, 0);

  // Handle the rest of the register if this isn't an even type breakdown.
  if (LeftoverTy.isValid())
    splitTypePieces(LeftoverTy, NarrowLeftoverRegs, HandledOffset);

  if (IsLoad) {
    insertParts(ValReg, ValTy, NarrowTy, NarrowRegs,
                LeftoverTy, NarrowLeftoverRegs);
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::reduceOperationWidth(MachineInstr &MI, unsigned int TypeIdx,
                                      LLT NarrowTy) {
  assert(TypeIdx == 0 && "only one type index expected");

  const unsigned Opc = MI.getOpcode();
  const int NumOps = MI.getNumOperands() - 1;
  const Register DstReg = MI.getOperand(0).getReg();
  const unsigned Flags = MI.getFlags();
  const unsigned NarrowSize = NarrowTy.getSizeInBits();
  const LLT NarrowScalarTy = LLT::scalar(NarrowSize);

  assert(NumOps <= 3 && "expected instruction with 1 result and 1-3 sources");

  // First of all check whether we are narrowing (changing the element type)
  // or reducing the vector elements
  const LLT DstTy = MRI.getType(DstReg);
  const bool IsNarrow = NarrowTy.getScalarType() != DstTy.getScalarType();

  SmallVector<Register, 8> ExtractedRegs[3];
  SmallVector<Register, 8> Parts;

  unsigned NarrowElts = NarrowTy.isVector() ? NarrowTy.getNumElements() : 1;

  // Break down all the sources into NarrowTy pieces we can operate on. This may
  // involve creating merges to a wider type, padded with undef.
  for (int I = 0; I != NumOps; ++I) {
    Register SrcReg = MI.getOperand(I + 1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);

    // The type to narrow SrcReg to. For narrowing, this is a smaller scalar.
    // For fewerElements, this is a smaller vector with the same element type.
    LLT OpNarrowTy;
    if (IsNarrow) {
      OpNarrowTy = NarrowScalarTy;

      // In case of narrowing, we need to cast vectors to scalars for this to
      // work properly
      // FIXME: Can we do without the bitcast here if we're narrowing?
      if (SrcTy.isVector()) {
        SrcTy = LLT::scalar(SrcTy.getSizeInBits());
        SrcReg = MIRBuilder.buildBitcast(SrcTy, SrcReg).getReg(0);
      }
    } else {
      OpNarrowTy = LLT::scalarOrVector(NarrowElts, SrcTy.getScalarType());
    }

    LLT GCDTy = extractGCDType(ExtractedRegs[I], SrcTy, OpNarrowTy, SrcReg);

    // Build a sequence of NarrowTy pieces in ExtractedRegs for this operand.
    buildLCMMergePieces(SrcTy, OpNarrowTy, GCDTy, ExtractedRegs[I],
                        TargetOpcode::G_ANYEXT);
  }

  SmallVector<Register, 8> ResultRegs;

  // Input operands for each sub-instruction.
  SmallVector<SrcOp, 4> InputRegs(NumOps, Register());

  int NumParts = ExtractedRegs[0].size();
  const unsigned DstSize = DstTy.getSizeInBits();
  const LLT DstScalarTy = LLT::scalar(DstSize);

  // Narrowing needs to use scalar types
  LLT DstLCMTy, NarrowDstTy;
  if (IsNarrow) {
    DstLCMTy = getLCMType(DstScalarTy, NarrowScalarTy);
    NarrowDstTy = NarrowScalarTy;
  } else {
    DstLCMTy = getLCMType(DstTy, NarrowTy);
    NarrowDstTy = NarrowTy;
  }

  // We widened the source registers to satisfy merge/unmerge size
  // constraints. We'll have some extra fully undef parts.
  const int NumRealParts = (DstSize + NarrowSize - 1) / NarrowSize;

  for (int I = 0; I != NumRealParts; ++I) {
    // Emit this instruction on each of the split pieces.
    for (int J = 0; J != NumOps; ++J)
      InputRegs[J] = ExtractedRegs[J][I];

    auto Inst = MIRBuilder.buildInstr(Opc, {NarrowDstTy}, InputRegs, Flags);
    ResultRegs.push_back(Inst.getReg(0));
  }

  // Fill out the widened result with undef instead of creating instructions
  // with undef inputs.
  int NumUndefParts = NumParts - NumRealParts;
  if (NumUndefParts != 0)
    ResultRegs.append(NumUndefParts,
                      MIRBuilder.buildUndef(NarrowDstTy).getReg(0));

  // Extract the possibly padded result. Use a scratch register if we need to do
  // a final bitcast, otherwise use the original result register.
  Register MergeDstReg;
  if (IsNarrow && DstTy.isVector())
    MergeDstReg = MRI.createGenericVirtualRegister(DstScalarTy);
  else
    MergeDstReg = DstReg;

  buildWidenedRemergeToDst(MergeDstReg, DstLCMTy, ResultRegs);

  // Recast to vector if we narrowed a vector
  if (IsNarrow && DstTy.isVector())
    MIRBuilder.buildBitcast(DstReg, MergeDstReg);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorSextInReg(MachineInstr &MI, unsigned TypeIdx,
                                              LLT NarrowTy) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  int64_t Imm = MI.getOperand(2).getImm();

  LLT DstTy = MRI.getType(DstReg);

  SmallVector<Register, 8> Parts;
  LLT GCDTy = extractGCDType(Parts, DstTy, NarrowTy, SrcReg);
  LLT LCMTy = buildLCMMergePieces(DstTy, NarrowTy, GCDTy, Parts);

  for (Register &R : Parts)
    R = MIRBuilder.buildSExtInReg(NarrowTy, R, Imm).getReg(0);

  buildWidenedRemergeToDst(DstReg, LCMTy, Parts);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                     LLT NarrowTy) {
  using namespace TargetOpcode;

  MIRBuilder.setInstrAndDebugLoc(MI);
  switch (MI.getOpcode()) {
  case G_IMPLICIT_DEF:
    return fewerElementsVectorImplicitDef(MI, TypeIdx, NarrowTy);
  case G_TRUNC:
  case G_AND:
  case G_OR:
  case G_XOR:
  case G_ADD:
  case G_SUB:
  case G_MUL:
  case G_SMULH:
  case G_UMULH:
  case G_FADD:
  case G_FMUL:
  case G_FSUB:
  case G_FNEG:
  case G_FABS:
  case G_FCANONICALIZE:
  case G_FDIV:
  case G_FREM:
  case G_FMA:
  case G_FMAD:
  case G_FPOW:
  case G_FEXP:
  case G_FEXP2:
  case G_FLOG:
  case G_FLOG2:
  case G_FLOG10:
  case G_FNEARBYINT:
  case G_FCEIL:
  case G_FFLOOR:
  case G_FRINT:
  case G_INTRINSIC_ROUND:
  case G_INTRINSIC_TRUNC:
  case G_FCOS:
  case G_FSIN:
  case G_FSQRT:
  case G_BSWAP:
  case G_BITREVERSE:
  case G_SDIV:
  case G_UDIV:
  case G_SREM:
  case G_UREM:
  case G_SMIN:
  case G_SMAX:
  case G_UMIN:
  case G_UMAX:
  case G_FMINNUM:
  case G_FMAXNUM:
  case G_FMINNUM_IEEE:
  case G_FMAXNUM_IEEE:
  case G_FMINIMUM:
  case G_FMAXIMUM:
  case G_FSHL:
  case G_FSHR:
  case G_FREEZE:
    return reduceOperationWidth(MI, TypeIdx, NarrowTy);
  case G_SHL:
  case G_LSHR:
  case G_ASHR:
  case G_CTLZ:
  case G_CTLZ_ZERO_UNDEF:
  case G_CTTZ:
  case G_CTTZ_ZERO_UNDEF:
  case G_CTPOP:
  case G_FCOPYSIGN:
    return fewerElementsVectorMultiEltType(MI, TypeIdx, NarrowTy);
  case G_ZEXT:
  case G_SEXT:
  case G_ANYEXT:
  case G_FPEXT:
  case G_FPTRUNC:
  case G_SITOFP:
  case G_UITOFP:
  case G_FPTOSI:
  case G_FPTOUI:
  case G_INTTOPTR:
  case G_PTRTOINT:
  case G_ADDRSPACE_CAST:
    return fewerElementsVectorCasts(MI, TypeIdx, NarrowTy);
  case G_ICMP:
  case G_FCMP:
    return fewerElementsVectorCmp(MI, TypeIdx, NarrowTy);
  case G_SELECT:
    return fewerElementsVectorSelect(MI, TypeIdx, NarrowTy);
  case G_PHI:
    return fewerElementsVectorPhi(MI, TypeIdx, NarrowTy);
  case G_UNMERGE_VALUES:
    return fewerElementsVectorUnmergeValues(MI, TypeIdx, NarrowTy);
  case G_BUILD_VECTOR:
    return fewerElementsVectorBuildVector(MI, TypeIdx, NarrowTy);
  case G_LOAD:
  case G_STORE:
    return reduceLoadStoreWidth(MI, TypeIdx, NarrowTy);
  case G_SEXT_INREG:
    return fewerElementsVectorSextInReg(MI, TypeIdx, NarrowTy);
  default:
    return UnableToLegalize;
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarShiftByConstant(MachineInstr &MI, const APInt &Amt,
                                             const LLT HalfTy, const LLT AmtTy) {

  Register InL = MRI.createGenericVirtualRegister(HalfTy);
  Register InH = MRI.createGenericVirtualRegister(HalfTy);
  MIRBuilder.buildUnmerge({InL, InH}, MI.getOperand(1));

  if (Amt.isNullValue()) {
    MIRBuilder.buildMerge(MI.getOperand(0), {InL, InH});
    MI.eraseFromParent();
    return Legalized;
  }

  LLT NVT = HalfTy;
  unsigned NVTBits = HalfTy.getSizeInBits();
  unsigned VTBits = 2 * NVTBits;

  SrcOp Lo(Register(0)), Hi(Register(0));
  if (MI.getOpcode() == TargetOpcode::G_SHL) {
    if (Amt.ugt(VTBits)) {
      Lo = Hi = MIRBuilder.buildConstant(NVT, 0);
    } else if (Amt.ugt(NVTBits)) {
      Lo = MIRBuilder.buildConstant(NVT, 0);
      Hi = MIRBuilder.buildShl(NVT, InL,
                               MIRBuilder.buildConstant(AmtTy, Amt - NVTBits));
    } else if (Amt == NVTBits) {
      Lo = MIRBuilder.buildConstant(NVT, 0);
      Hi = InL;
    } else {
      Lo = MIRBuilder.buildShl(NVT, InL, MIRBuilder.buildConstant(AmtTy, Amt));
      auto OrLHS =
          MIRBuilder.buildShl(NVT, InH, MIRBuilder.buildConstant(AmtTy, Amt));
      auto OrRHS = MIRBuilder.buildLShr(
          NVT, InL, MIRBuilder.buildConstant(AmtTy, -Amt + NVTBits));
      Hi = MIRBuilder.buildOr(NVT, OrLHS, OrRHS);
    }
  } else if (MI.getOpcode() == TargetOpcode::G_LSHR) {
    if (Amt.ugt(VTBits)) {
      Lo = Hi = MIRBuilder.buildConstant(NVT, 0);
    } else if (Amt.ugt(NVTBits)) {
      Lo = MIRBuilder.buildLShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, Amt - NVTBits));
      Hi = MIRBuilder.buildConstant(NVT, 0);
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = MIRBuilder.buildConstant(NVT, 0);
    } else {
      auto ShiftAmtConst = MIRBuilder.buildConstant(AmtTy, Amt);

      auto OrLHS = MIRBuilder.buildLShr(NVT, InL, ShiftAmtConst);
      auto OrRHS = MIRBuilder.buildShl(
          NVT, InH, MIRBuilder.buildConstant(AmtTy, -Amt + NVTBits));

      Lo = MIRBuilder.buildOr(NVT, OrLHS, OrRHS);
      Hi = MIRBuilder.buildLShr(NVT, InH, ShiftAmtConst);
    }
  } else {
    if (Amt.ugt(VTBits)) {
      Hi = Lo = MIRBuilder.buildAShr(
          NVT, InH, MIRBuilder.buildConstant(AmtTy, NVTBits - 1));
    } else if (Amt.ugt(NVTBits)) {
      Lo = MIRBuilder.buildAShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, Amt - NVTBits));
      Hi = MIRBuilder.buildAShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, NVTBits - 1));
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = MIRBuilder.buildAShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, NVTBits - 1));
    } else {
      auto ShiftAmtConst = MIRBuilder.buildConstant(AmtTy, Amt);

      auto OrLHS = MIRBuilder.buildLShr(NVT, InL, ShiftAmtConst);
      auto OrRHS = MIRBuilder.buildShl(
          NVT, InH, MIRBuilder.buildConstant(AmtTy, -Amt + NVTBits));

      Lo = MIRBuilder.buildOr(NVT, OrLHS, OrRHS);
      Hi = MIRBuilder.buildAShr(NVT, InH, ShiftAmtConst);
    }
  }

  MIRBuilder.buildMerge(MI.getOperand(0), {Lo, Hi});
  MI.eraseFromParent();

  return Legalized;
}

// TODO: Optimize if constant shift amount.
LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarShift(MachineInstr &MI, unsigned TypeIdx,
                                   LLT RequestedTy) {
  if (TypeIdx == 1) {
    Observer.changingInstr(MI);
    narrowScalarSrc(MI, RequestedTy, 2);
    Observer.changedInstr(MI);
    return Legalized;
  }

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  if (DstTy.isVector())
    return UnableToLegalize;

  Register Amt = MI.getOperand(2).getReg();
  LLT ShiftAmtTy = MRI.getType(Amt);
  const unsigned DstEltSize = DstTy.getScalarSizeInBits();
  if (DstEltSize % 2 != 0)
    return UnableToLegalize;

  // Ignore the input type. We can only go to exactly half the size of the
  // input. If that isn't small enough, the resulting pieces will be further
  // legalized.
  const unsigned NewBitSize = DstEltSize / 2;
  const LLT HalfTy = LLT::scalar(NewBitSize);
  const LLT CondTy = LLT::scalar(1);

  if (const MachineInstr *KShiftAmt =
          getOpcodeDef(TargetOpcode::G_CONSTANT, Amt, MRI)) {
    return narrowScalarShiftByConstant(
        MI, KShiftAmt->getOperand(1).getCImm()->getValue(), HalfTy, ShiftAmtTy);
  }

  // TODO: Expand with known bits.

  // Handle the fully general expansion by an unknown amount.
  auto NewBits = MIRBuilder.buildConstant(ShiftAmtTy, NewBitSize);

  Register InL = MRI.createGenericVirtualRegister(HalfTy);
  Register InH = MRI.createGenericVirtualRegister(HalfTy);
  MIRBuilder.buildUnmerge({InL, InH}, MI.getOperand(1));

  auto AmtExcess = MIRBuilder.buildSub(ShiftAmtTy, Amt, NewBits);
  auto AmtLack = MIRBuilder.buildSub(ShiftAmtTy, NewBits, Amt);

  auto Zero = MIRBuilder.buildConstant(ShiftAmtTy, 0);
  auto IsShort = MIRBuilder.buildICmp(ICmpInst::ICMP_ULT, CondTy, Amt, NewBits);
  auto IsZero = MIRBuilder.buildICmp(ICmpInst::ICMP_EQ, CondTy, Amt, Zero);

  Register ResultRegs[2];
  switch (MI.getOpcode()) {
  case TargetOpcode::G_SHL: {
    // Short: ShAmt < NewBitSize
    auto LoS = MIRBuilder.buildShl(HalfTy, InL, Amt);

    auto LoOr = MIRBuilder.buildLShr(HalfTy, InL, AmtLack);
    auto HiOr = MIRBuilder.buildShl(HalfTy, InH, Amt);
    auto HiS = MIRBuilder.buildOr(HalfTy, LoOr, HiOr);

    // Long: ShAmt >= NewBitSize
    auto LoL = MIRBuilder.buildConstant(HalfTy, 0);         // Lo part is zero.
    auto HiL = MIRBuilder.buildShl(HalfTy, InL, AmtExcess); // Hi from Lo part.

    auto Lo = MIRBuilder.buildSelect(HalfTy, IsShort, LoS, LoL);
    auto Hi = MIRBuilder.buildSelect(
        HalfTy, IsZero, InH, MIRBuilder.buildSelect(HalfTy, IsShort, HiS, HiL));

    ResultRegs[0] = Lo.getReg(0);
    ResultRegs[1] = Hi.getReg(0);
    break;
  }
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR: {
    // Short: ShAmt < NewBitSize
    auto HiS = MIRBuilder.buildInstr(MI.getOpcode(), {HalfTy}, {InH, Amt});

    auto LoOr = MIRBuilder.buildLShr(HalfTy, InL, Amt);
    auto HiOr = MIRBuilder.buildShl(HalfTy, InH, AmtLack);
    auto LoS = MIRBuilder.buildOr(HalfTy, LoOr, HiOr);

    // Long: ShAmt >= NewBitSize
    MachineInstrBuilder HiL;
    if (MI.getOpcode() == TargetOpcode::G_LSHR) {
      HiL = MIRBuilder.buildConstant(HalfTy, 0);            // Hi part is zero.
    } else {
      auto ShiftAmt = MIRBuilder.buildConstant(ShiftAmtTy, NewBitSize - 1);
      HiL = MIRBuilder.buildAShr(HalfTy, InH, ShiftAmt);    // Sign of Hi part.
    }
    auto LoL = MIRBuilder.buildInstr(MI.getOpcode(), {HalfTy},
                                     {InH, AmtExcess});     // Lo from Hi part.

    auto Lo = MIRBuilder.buildSelect(
        HalfTy, IsZero, InL, MIRBuilder.buildSelect(HalfTy, IsShort, LoS, LoL));

    auto Hi = MIRBuilder.buildSelect(HalfTy, IsShort, HiS, HiL);

    ResultRegs[0] = Lo.getReg(0);
    ResultRegs[1] = Hi.getReg(0);
    break;
  }
  default:
    llvm_unreachable("not a shift");
  }

  MIRBuilder.buildMerge(DstReg, ResultRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::moreElementsVectorPhi(MachineInstr &MI, unsigned TypeIdx,
                                       LLT MoreTy) {
  assert(TypeIdx == 0 && "Expecting only Idx 0");

  Observer.changingInstr(MI);
  for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
    MachineBasicBlock &OpMBB = *MI.getOperand(I + 1).getMBB();
    MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());
    moreElementsVectorSrc(MI, MoreTy, I);
  }

  MachineBasicBlock &MBB = *MI.getParent();
  MIRBuilder.setInsertPt(MBB, --MBB.getFirstNonPHI());
  moreElementsVectorDst(MI, MoreTy, 0);
  Observer.changedInstr(MI);
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::moreElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                    LLT MoreTy) {
  MIRBuilder.setInstrAndDebugLoc(MI);
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case TargetOpcode::G_IMPLICIT_DEF:
  case TargetOpcode::G_LOAD: {
    if (TypeIdx != 0)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_STORE:
    if (TypeIdx != 0)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMAXNUM_IEEE:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXIMUM: {
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 1);
    moreElementsVectorSrc(MI, MoreTy, 2);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_EXTRACT:
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 1);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_INSERT:
  case TargetOpcode::G_FREEZE:
    if (TypeIdx != 0)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 1);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_SELECT:
    if (TypeIdx != 0)
      return UnableToLegalize;
    if (MRI.getType(MI.getOperand(1).getReg()).isVector())
      return UnableToLegalize;

    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 2);
    moreElementsVectorSrc(MI, MoreTy, 3);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_UNMERGE_VALUES: {
    if (TypeIdx != 1)
      return UnableToLegalize;

    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    int NumDst = MI.getNumOperands() - 1;
    moreElementsVectorSrc(MI, MoreTy, NumDst);

    auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_UNMERGE_VALUES);
    for (int I = 0; I != NumDst; ++I)
      MIB.addDef(MI.getOperand(I).getReg());

    int NewNumDst = MoreTy.getSizeInBits() / DstTy.getSizeInBits();
    for (int I = NumDst; I != NewNumDst; ++I)
      MIB.addDef(MRI.createGenericVirtualRegister(DstTy));

    MIB.addUse(MI.getOperand(NumDst).getReg());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_PHI:
    return moreElementsVectorPhi(MI, TypeIdx, MoreTy);
  default:
    return UnableToLegalize;
  }
}

void LegalizerHelper::multiplyRegisters(SmallVectorImpl<Register> &DstRegs,
                                        ArrayRef<Register> Src1Regs,
                                        ArrayRef<Register> Src2Regs,
                                        LLT NarrowTy) {
  MachineIRBuilder &B = MIRBuilder;
  unsigned SrcParts = Src1Regs.size();
  unsigned DstParts = DstRegs.size();

  unsigned DstIdx = 0; // Low bits of the result.
  Register FactorSum =
      B.buildMul(NarrowTy, Src1Regs[DstIdx], Src2Regs[DstIdx]).getReg(0);
  DstRegs[DstIdx] = FactorSum;

  unsigned CarrySumPrevDstIdx;
  SmallVector<Register, 4> Factors;

  for (DstIdx = 1; DstIdx < DstParts; DstIdx++) {
    // Collect low parts of muls for DstIdx.
    for (unsigned i = DstIdx + 1 < SrcParts ? 0 : DstIdx - SrcParts + 1;
         i <= std::min(DstIdx, SrcParts - 1); ++i) {
      MachineInstrBuilder Mul =
          B.buildMul(NarrowTy, Src1Regs[DstIdx - i], Src2Regs[i]);
      Factors.push_back(Mul.getReg(0));
    }
    // Collect high parts of muls from previous DstIdx.
    for (unsigned i = DstIdx < SrcParts ? 0 : DstIdx - SrcParts;
         i <= std::min(DstIdx - 1, SrcParts - 1); ++i) {
      MachineInstrBuilder Umulh =
          B.buildUMulH(NarrowTy, Src1Regs[DstIdx - 1 - i], Src2Regs[i]);
      Factors.push_back(Umulh.getReg(0));
    }
    // Add CarrySum from additions calculated for previous DstIdx.
    if (DstIdx != 1) {
      Factors.push_back(CarrySumPrevDstIdx);
    }

    Register CarrySum;
    // Add all factors and accumulate all carries into CarrySum.
    if (DstIdx != DstParts - 1) {
      MachineInstrBuilder Uaddo =
          B.buildUAddo(NarrowTy, LLT::scalar(1), Factors[0], Factors[1]);
      FactorSum = Uaddo.getReg(0);
      CarrySum = B.buildZExt(NarrowTy, Uaddo.getReg(1)).getReg(0);
      for (unsigned i = 2; i < Factors.size(); ++i) {
        MachineInstrBuilder Uaddo =
            B.buildUAddo(NarrowTy, LLT::scalar(1), FactorSum, Factors[i]);
        FactorSum = Uaddo.getReg(0);
        MachineInstrBuilder Carry = B.buildZExt(NarrowTy, Uaddo.getReg(1));
        CarrySum = B.buildAdd(NarrowTy, CarrySum, Carry).getReg(0);
      }
    } else {
      // Since value for the next index is not calculated, neither is CarrySum.
      FactorSum = B.buildAdd(NarrowTy, Factors[0], Factors[1]).getReg(0);
      for (unsigned i = 2; i < Factors.size(); ++i)
        FactorSum = B.buildAdd(NarrowTy, FactorSum, Factors[i]).getReg(0);
    }

    CarrySumPrevDstIdx = CarrySum;
    DstRegs[DstIdx] = FactorSum;
    Factors.clear();
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarMul(MachineInstr &MI, LLT NarrowTy) {
  Register DstReg = MI.getOperand(0).getReg();
  Register Src1 = MI.getOperand(1).getReg();
  Register Src2 = MI.getOperand(2).getReg();

  LLT Ty = MRI.getType(DstReg);
  if (Ty.isVector())
    return UnableToLegalize;

  unsigned SrcSize = MRI.getType(Src1).getSizeInBits();
  unsigned DstSize = Ty.getSizeInBits();
  unsigned NarrowSize = NarrowTy.getSizeInBits();
  if (DstSize % NarrowSize != 0 || SrcSize % NarrowSize != 0)
    return UnableToLegalize;

  unsigned NumDstParts = DstSize / NarrowSize;
  unsigned NumSrcParts = SrcSize / NarrowSize;
  bool IsMulHigh = MI.getOpcode() == TargetOpcode::G_UMULH;
  unsigned DstTmpParts = NumDstParts * (IsMulHigh ? 2 : 1);

  SmallVector<Register, 2> Src1Parts, Src2Parts;
  SmallVector<Register, 2> DstTmpRegs(DstTmpParts);
  extractParts(Src1, NarrowTy, NumSrcParts, Src1Parts);
  extractParts(Src2, NarrowTy, NumSrcParts, Src2Parts);
  multiplyRegisters(DstTmpRegs, Src1Parts, Src2Parts, NarrowTy);

  // Take only high half of registers if this is high mul.
  ArrayRef<Register> DstRegs(
      IsMulHigh ? &DstTmpRegs[DstTmpParts / 2] : &DstTmpRegs[0], NumDstParts);
  MIRBuilder.buildMerge(DstReg, DstRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarExtract(MachineInstr &MI, unsigned TypeIdx,
                                     LLT NarrowTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  uint64_t NarrowSize = NarrowTy.getSizeInBits();

  int64_t SizeOp1 = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
  // FIXME: add support for when SizeOp1 isn't an exact multiple of
  // NarrowSize.
  if (SizeOp1 % NarrowSize != 0)
    return UnableToLegalize;
  int NumParts = SizeOp1 / NarrowSize;

  SmallVector<Register, 2> SrcRegs, DstRegs;
  SmallVector<uint64_t, 2> Indexes;
  extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, SrcRegs);

  Register OpReg = MI.getOperand(0).getReg();
  uint64_t OpStart = MI.getOperand(2).getImm();
  uint64_t OpSize = MRI.getType(OpReg).getSizeInBits();
  for (int i = 0; i < NumParts; ++i) {
    unsigned SrcStart = i * NarrowSize;

    if (SrcStart + NarrowSize <= OpStart || SrcStart >= OpStart + OpSize) {
      // No part of the extract uses this subregister, ignore it.
      continue;
    } else if (SrcStart == OpStart && NarrowTy == MRI.getType(OpReg)) {
      // The entire subregister is extracted, forward the value.
      DstRegs.push_back(SrcRegs[i]);
      continue;
    }

    // OpSegStart is where this destination segment would start in OpReg if it
    // extended infinitely in both directions.
    int64_t ExtractOffset;
    uint64_t SegSize;
    if (OpStart < SrcStart) {
      ExtractOffset = 0;
      SegSize = std::min(NarrowSize, OpStart + OpSize - SrcStart);
    } else {
      ExtractOffset = OpStart - SrcStart;
      SegSize = std::min(SrcStart + NarrowSize - OpStart, OpSize);
    }

    Register SegReg = SrcRegs[i];
    if (ExtractOffset != 0 || SegSize != NarrowSize) {
      // A genuine extract is needed.
      SegReg = MRI.createGenericVirtualRegister(LLT::scalar(SegSize));
      MIRBuilder.buildExtract(SegReg, SrcRegs[i], ExtractOffset);
    }

    DstRegs.push_back(SegReg);
  }

  Register DstReg = MI.getOperand(0).getReg();
  if (MRI.getType(DstReg).isVector())
    MIRBuilder.buildBuildVector(DstReg, DstRegs);
  else if (DstRegs.size() > 1)
    MIRBuilder.buildMerge(DstReg, DstRegs);
  else
    MIRBuilder.buildCopy(DstReg, DstRegs[0]);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarInsert(MachineInstr &MI, unsigned TypeIdx,
                                    LLT NarrowTy) {
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0)
    return UnableToLegalize;

  uint64_t SizeOp0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  uint64_t NarrowSize = NarrowTy.getSizeInBits();

  // FIXME: add support for when SizeOp0 isn't an exact multiple of
  // NarrowSize.
  if (SizeOp0 % NarrowSize != 0)
    return UnableToLegalize;

  int NumParts = SizeOp0 / NarrowSize;

  SmallVector<Register, 2> SrcRegs, DstRegs;
  SmallVector<uint64_t, 2> Indexes;
  extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, SrcRegs);

  Register OpReg = MI.getOperand(2).getReg();
  uint64_t OpStart = MI.getOperand(3).getImm();
  uint64_t OpSize = MRI.getType(OpReg).getSizeInBits();
  for (int i = 0; i < NumParts; ++i) {
    unsigned DstStart = i * NarrowSize;

    if (DstStart + NarrowSize <= OpStart || DstStart >= OpStart + OpSize) {
      // No part of the insert affects this subregister, forward the original.
      DstRegs.push_back(SrcRegs[i]);
      continue;
    } else if (DstStart == OpStart && NarrowTy == MRI.getType(OpReg)) {
      // The entire subregister is defined by this insert, forward the new
      // value.
      DstRegs.push_back(OpReg);
      continue;
    }

    // OpSegStart is where this destination segment would start in OpReg if it
    // extended infinitely in both directions.
    int64_t ExtractOffset, InsertOffset;
    uint64_t SegSize;
    if (OpStart < DstStart) {
      InsertOffset = 0;
      ExtractOffset = DstStart - OpStart;
      SegSize = std::min(NarrowSize, OpStart + OpSize - DstStart);
    } else {
      InsertOffset = OpStart - DstStart;
      ExtractOffset = 0;
      SegSize =
        std::min(NarrowSize - InsertOffset, OpStart + OpSize - DstStart);
    }

    Register SegReg = OpReg;
    if (ExtractOffset != 0 || SegSize != OpSize) {
      // A genuine extract is needed.
      SegReg = MRI.createGenericVirtualRegister(LLT::scalar(SegSize));
      MIRBuilder.buildExtract(SegReg, OpReg, ExtractOffset);
    }

    Register DstReg = MRI.createGenericVirtualRegister(NarrowTy);
    MIRBuilder.buildInsert(DstReg, SrcRegs[i], SegReg, InsertOffset);
    DstRegs.push_back(DstReg);
  }

  assert(DstRegs.size() == (unsigned)NumParts && "not all parts covered");
  Register DstReg = MI.getOperand(0).getReg();
  if(MRI.getType(DstReg).isVector())
    MIRBuilder.buildBuildVector(DstReg, DstRegs);
  else
    MIRBuilder.buildMerge(DstReg, DstRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarBasic(MachineInstr &MI, unsigned TypeIdx,
                                   LLT NarrowTy) {
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  assert(MI.getNumOperands() == 3 && TypeIdx == 0);

  SmallVector<Register, 4> DstRegs, DstLeftoverRegs;
  SmallVector<Register, 4> Src0Regs, Src0LeftoverRegs;
  SmallVector<Register, 4> Src1Regs, Src1LeftoverRegs;
  LLT LeftoverTy;
  if (!extractParts(MI.getOperand(1).getReg(), DstTy, NarrowTy, LeftoverTy,
                    Src0Regs, Src0LeftoverRegs))
    return UnableToLegalize;

  LLT Unused;
  if (!extractParts(MI.getOperand(2).getReg(), DstTy, NarrowTy, Unused,
                    Src1Regs, Src1LeftoverRegs))
    llvm_unreachable("inconsistent extractParts result");

  for (unsigned I = 0, E = Src1Regs.size(); I != E; ++I) {
    auto Inst = MIRBuilder.buildInstr(MI.getOpcode(), {NarrowTy},
                                        {Src0Regs[I], Src1Regs[I]});
    DstRegs.push_back(Inst.getReg(0));
  }

  for (unsigned I = 0, E = Src1LeftoverRegs.size(); I != E; ++I) {
    auto Inst = MIRBuilder.buildInstr(
      MI.getOpcode(),
      {LeftoverTy}, {Src0LeftoverRegs[I], Src1LeftoverRegs[I]});
    DstLeftoverRegs.push_back(Inst.getReg(0));
  }

  insertParts(DstReg, DstTy, NarrowTy, DstRegs,
              LeftoverTy, DstLeftoverRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarExt(MachineInstr &MI, unsigned TypeIdx,
                                 LLT NarrowTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();

  LLT DstTy = MRI.getType(DstReg);
  if (DstTy.isVector())
    return UnableToLegalize;

  SmallVector<Register, 8> Parts;
  LLT GCDTy = extractGCDType(Parts, DstTy, NarrowTy, SrcReg);
  LLT LCMTy = buildLCMMergePieces(DstTy, NarrowTy, GCDTy, Parts, MI.getOpcode());
  buildWidenedRemergeToDst(DstReg, LCMTy, Parts);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarSelect(MachineInstr &MI, unsigned TypeIdx,
                                    LLT NarrowTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  Register CondReg = MI.getOperand(1).getReg();
  LLT CondTy = MRI.getType(CondReg);
  if (CondTy.isVector()) // TODO: Handle vselect
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  SmallVector<Register, 4> DstRegs, DstLeftoverRegs;
  SmallVector<Register, 4> Src1Regs, Src1LeftoverRegs;
  SmallVector<Register, 4> Src2Regs, Src2LeftoverRegs;
  LLT LeftoverTy;
  if (!extractParts(MI.getOperand(2).getReg(), DstTy, NarrowTy, LeftoverTy,
                    Src1Regs, Src1LeftoverRegs))
    return UnableToLegalize;

  LLT Unused;
  if (!extractParts(MI.getOperand(3).getReg(), DstTy, NarrowTy, Unused,
                    Src2Regs, Src2LeftoverRegs))
    llvm_unreachable("inconsistent extractParts result");

  for (unsigned I = 0, E = Src1Regs.size(); I != E; ++I) {
    auto Select = MIRBuilder.buildSelect(NarrowTy,
                                         CondReg, Src1Regs[I], Src2Regs[I]);
    DstRegs.push_back(Select.getReg(0));
  }

  for (unsigned I = 0, E = Src1LeftoverRegs.size(); I != E; ++I) {
    auto Select = MIRBuilder.buildSelect(
      LeftoverTy, CondReg, Src1LeftoverRegs[I], Src2LeftoverRegs[I]);
    DstLeftoverRegs.push_back(Select.getReg(0));
  }

  insertParts(DstReg, DstTy, NarrowTy, DstRegs,
              LeftoverTy, DstLeftoverRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarCTLZ(MachineInstr &MI, unsigned TypeIdx,
                                  LLT NarrowTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  unsigned NarrowSize = NarrowTy.getSizeInBits();

  if (SrcTy.isScalar() && SrcTy.getSizeInBits() == 2 * NarrowSize) {
    const bool IsUndef = MI.getOpcode() == TargetOpcode::G_CTLZ_ZERO_UNDEF;

    MachineIRBuilder &B = MIRBuilder;
    auto UnmergeSrc = B.buildUnmerge(NarrowTy, SrcReg);
    // ctlz(Hi:Lo) -> Hi == 0 ? (NarrowSize + ctlz(Lo)) : ctlz(Hi)
    auto C_0 = B.buildConstant(NarrowTy, 0);
    auto HiIsZero = B.buildICmp(CmpInst::ICMP_EQ, LLT::scalar(1),
                                UnmergeSrc.getReg(1), C_0);
    auto LoCTLZ = IsUndef ?
      B.buildCTLZ_ZERO_UNDEF(DstTy, UnmergeSrc.getReg(0)) :
      B.buildCTLZ(DstTy, UnmergeSrc.getReg(0));
    auto C_NarrowSize = B.buildConstant(DstTy, NarrowSize);
    auto HiIsZeroCTLZ = B.buildAdd(DstTy, LoCTLZ, C_NarrowSize);
    auto HiCTLZ = B.buildCTLZ_ZERO_UNDEF(DstTy, UnmergeSrc.getReg(1));
    B.buildSelect(DstReg, HiIsZero, HiIsZeroCTLZ, HiCTLZ);

    MI.eraseFromParent();
    return Legalized;
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarCTTZ(MachineInstr &MI, unsigned TypeIdx,
                                  LLT NarrowTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  unsigned NarrowSize = NarrowTy.getSizeInBits();

  if (SrcTy.isScalar() && SrcTy.getSizeInBits() == 2 * NarrowSize) {
    const bool IsUndef = MI.getOpcode() == TargetOpcode::G_CTTZ_ZERO_UNDEF;

    MachineIRBuilder &B = MIRBuilder;
    auto UnmergeSrc = B.buildUnmerge(NarrowTy, SrcReg);
    // cttz(Hi:Lo) -> Lo == 0 ? (cttz(Hi) + NarrowSize) : cttz(Lo)
    auto C_0 = B.buildConstant(NarrowTy, 0);
    auto LoIsZero = B.buildICmp(CmpInst::ICMP_EQ, LLT::scalar(1),
                                UnmergeSrc.getReg(0), C_0);
    auto HiCTTZ = IsUndef ?
      B.buildCTTZ_ZERO_UNDEF(DstTy, UnmergeSrc.getReg(1)) :
      B.buildCTTZ(DstTy, UnmergeSrc.getReg(1));
    auto C_NarrowSize = B.buildConstant(DstTy, NarrowSize);
    auto LoIsZeroCTTZ = B.buildAdd(DstTy, HiCTTZ, C_NarrowSize);
    auto LoCTTZ = B.buildCTTZ_ZERO_UNDEF(DstTy, UnmergeSrc.getReg(0));
    B.buildSelect(DstReg, LoIsZero, LoIsZeroCTTZ, LoCTTZ);

    MI.eraseFromParent();
    return Legalized;
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarCTPOP(MachineInstr &MI, unsigned TypeIdx,
                                   LLT NarrowTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
  unsigned NarrowSize = NarrowTy.getSizeInBits();

  if (SrcTy.isScalar() && SrcTy.getSizeInBits() == 2 * NarrowSize) {
    auto UnmergeSrc = MIRBuilder.buildUnmerge(NarrowTy, MI.getOperand(1));

    auto LoCTPOP = MIRBuilder.buildCTPOP(DstTy, UnmergeSrc.getReg(0));
    auto HiCTPOP = MIRBuilder.buildCTPOP(DstTy, UnmergeSrc.getReg(1));
    MIRBuilder.buildAdd(DstReg, HiCTPOP, LoCTPOP);

    MI.eraseFromParent();
    return Legalized;
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerBitCount(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  unsigned Opc = MI.getOpcode();
  auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  auto isSupported = [this](const LegalityQuery &Q) {
    auto QAction = LI.getAction(Q).Action;
    return QAction == Legal || QAction == Libcall || QAction == Custom;
  };
  switch (Opc) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_CTLZ_ZERO_UNDEF: {
    // This trivially expands to CTLZ.
    Observer.changingInstr(MI);
    MI.setDesc(TII.get(TargetOpcode::G_CTLZ));
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_CTLZ: {
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    LLT DstTy = MRI.getType(DstReg);
    LLT SrcTy = MRI.getType(SrcReg);
    unsigned Len = SrcTy.getSizeInBits();

    if (isSupported({TargetOpcode::G_CTLZ_ZERO_UNDEF, {DstTy, SrcTy}})) {
      // If CTLZ_ZERO_UNDEF is supported, emit that and a select for zero.
      auto CtlzZU = MIRBuilder.buildCTLZ_ZERO_UNDEF(DstTy, SrcReg);
      auto ZeroSrc = MIRBuilder.buildConstant(SrcTy, 0);
      auto ICmp = MIRBuilder.buildICmp(
          CmpInst::ICMP_EQ, SrcTy.changeElementSize(1), SrcReg, ZeroSrc);
      auto LenConst = MIRBuilder.buildConstant(DstTy, Len);
      MIRBuilder.buildSelect(DstReg, ICmp, LenConst, CtlzZU);
      MI.eraseFromParent();
      return Legalized;
    }
    // for now, we do this:
    // NewLen = NextPowerOf2(Len);
    // x = x | (x >> 1);
    // x = x | (x >> 2);
    // ...
    // x = x | (x >>16);
    // x = x | (x >>32); // for 64-bit input
    // Upto NewLen/2
    // return Len - popcount(x);
    //
    // Ref: "Hacker's Delight" by Henry Warren
    Register Op = SrcReg;
    unsigned NewLen = PowerOf2Ceil(Len);
    for (unsigned i = 0; (1U << i) <= (NewLen / 2); ++i) {
      auto MIBShiftAmt = MIRBuilder.buildConstant(SrcTy, 1ULL << i);
      auto MIBOp = MIRBuilder.buildOr(
          SrcTy, Op, MIRBuilder.buildLShr(SrcTy, Op, MIBShiftAmt));
      Op = MIBOp.getReg(0);
    }
    auto MIBPop = MIRBuilder.buildCTPOP(DstTy, Op);
    MIRBuilder.buildSub(MI.getOperand(0), MIRBuilder.buildConstant(DstTy, Len),
                        MIBPop);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CTTZ_ZERO_UNDEF: {
    // This trivially expands to CTTZ.
    Observer.changingInstr(MI);
    MI.setDesc(TII.get(TargetOpcode::G_CTTZ));
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_CTTZ: {
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    LLT DstTy = MRI.getType(DstReg);
    LLT SrcTy = MRI.getType(SrcReg);

    unsigned Len = SrcTy.getSizeInBits();
    if (isSupported({TargetOpcode::G_CTTZ_ZERO_UNDEF, {DstTy, SrcTy}})) {
      // If CTTZ_ZERO_UNDEF is legal or custom, emit that and a select with
      // zero.
      auto CttzZU = MIRBuilder.buildCTTZ_ZERO_UNDEF(DstTy, SrcReg);
      auto Zero = MIRBuilder.buildConstant(SrcTy, 0);
      auto ICmp = MIRBuilder.buildICmp(
          CmpInst::ICMP_EQ, DstTy.changeElementSize(1), SrcReg, Zero);
      auto LenConst = MIRBuilder.buildConstant(DstTy, Len);
      MIRBuilder.buildSelect(DstReg, ICmp, LenConst, CttzZU);
      MI.eraseFromParent();
      return Legalized;
    }
    // for now, we use: { return popcount(~x & (x - 1)); }
    // unless the target has ctlz but not ctpop, in which case we use:
    // { return 32 - nlz(~x & (x-1)); }
    // Ref: "Hacker's Delight" by Henry Warren
    auto MIBCstNeg1 = MIRBuilder.buildConstant(Ty, -1);
    auto MIBNot = MIRBuilder.buildXor(Ty, SrcReg, MIBCstNeg1);
    auto MIBTmp = MIRBuilder.buildAnd(
        Ty, MIBNot, MIRBuilder.buildAdd(Ty, SrcReg, MIBCstNeg1));
    if (!isSupported({TargetOpcode::G_CTPOP, {Ty, Ty}}) &&
        isSupported({TargetOpcode::G_CTLZ, {Ty, Ty}})) {
      auto MIBCstLen = MIRBuilder.buildConstant(Ty, Len);
      MIRBuilder.buildSub(MI.getOperand(0), MIBCstLen,
                          MIRBuilder.buildCTLZ(Ty, MIBTmp));
      MI.eraseFromParent();
      return Legalized;
    }
    MI.setDesc(TII.get(TargetOpcode::G_CTPOP));
    MI.getOperand(1).setReg(MIBTmp.getReg(0));
    return Legalized;
  }
  case TargetOpcode::G_CTPOP: {
    unsigned Size = Ty.getSizeInBits();
    MachineIRBuilder &B = MIRBuilder;

    // Count set bits in blocks of 2 bits. Default approach would be
    // B2Count = { val & 0x55555555 } + { (val >> 1) & 0x55555555 }
    // We use following formula instead:
    // B2Count = val - { (val >> 1) & 0x55555555 }
    // since it gives same result in blocks of 2 with one instruction less.
    auto C_1 = B.buildConstant(Ty, 1);
    auto B2Set1LoTo1Hi = B.buildLShr(Ty, MI.getOperand(1).getReg(), C_1);
    APInt B2Mask1HiTo0 = APInt::getSplat(Size, APInt(8, 0x55));
    auto C_B2Mask1HiTo0 = B.buildConstant(Ty, B2Mask1HiTo0);
    auto B2Count1Hi = B.buildAnd(Ty, B2Set1LoTo1Hi, C_B2Mask1HiTo0);
    auto B2Count = B.buildSub(Ty, MI.getOperand(1).getReg(), B2Count1Hi);

    // In order to get count in blocks of 4 add values from adjacent block of 2.
    // B4Count = { B2Count & 0x33333333 } + { (B2Count >> 2) & 0x33333333 }
    auto C_2 = B.buildConstant(Ty, 2);
    auto B4Set2LoTo2Hi = B.buildLShr(Ty, B2Count, C_2);
    APInt B4Mask2HiTo0 = APInt::getSplat(Size, APInt(8, 0x33));
    auto C_B4Mask2HiTo0 = B.buildConstant(Ty, B4Mask2HiTo0);
    auto B4HiB2Count = B.buildAnd(Ty, B4Set2LoTo2Hi, C_B4Mask2HiTo0);
    auto B4LoB2Count = B.buildAnd(Ty, B2Count, C_B4Mask2HiTo0);
    auto B4Count = B.buildAdd(Ty, B4HiB2Count, B4LoB2Count);

    // For count in blocks of 8 bits we don't have to mask high 4 bits before
    // addition since count value sits in range {0,...,8} and 4 bits are enough
    // to hold such binary values. After addition high 4 bits still hold count
    // of set bits in high 4 bit block, set them to zero and get 8 bit result.
    // B8Count = { B4Count + (B4Count >> 4) } & 0x0F0F0F0F
    auto C_4 = B.buildConstant(Ty, 4);
    auto B8HiB4Count = B.buildLShr(Ty, B4Count, C_4);
    auto B8CountDirty4Hi = B.buildAdd(Ty, B8HiB4Count, B4Count);
    APInt B8Mask4HiTo0 = APInt::getSplat(Size, APInt(8, 0x0F));
    auto C_B8Mask4HiTo0 = B.buildConstant(Ty, B8Mask4HiTo0);
    auto B8Count = B.buildAnd(Ty, B8CountDirty4Hi, C_B8Mask4HiTo0);

    assert(Size<=128 && "Scalar size is too large for CTPOP lower algorithm");
    // 8 bits can hold CTPOP result of 128 bit int or smaller. Mul with this
    // bitmask will set 8 msb in ResTmp to sum of all B8Counts in 8 bit blocks.
    auto MulMask = B.buildConstant(Ty, APInt::getSplat(Size, APInt(8, 0x01)));
    auto ResTmp = B.buildMul(Ty, B8Count, MulMask);

    // Shift count result from 8 high bits to low bits.
    auto C_SizeM8 = B.buildConstant(Ty, Size - 8);
    B.buildLShr(MI.getOperand(0).getReg(), ResTmp, C_SizeM8);

    MI.eraseFromParent();
    return Legalized;
  }
  }
}

// Expand s32 = G_UITOFP s64 using bit operations to an IEEE float
// representation.
LegalizerHelper::LegalizeResult
LegalizerHelper::lowerU64ToF32BitOps(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const LLT S64 = LLT::scalar(64);
  const LLT S32 = LLT::scalar(32);
  const LLT S1 = LLT::scalar(1);

  assert(MRI.getType(Src) == S64 && MRI.getType(Dst) == S32);

  // unsigned cul2f(ulong u) {
  //   uint lz = clz(u);
  //   uint e = (u != 0) ? 127U + 63U - lz : 0;
  //   u = (u << lz) & 0x7fffffffffffffffUL;
  //   ulong t = u & 0xffffffffffUL;
  //   uint v = (e << 23) | (uint)(u >> 40);
  //   uint r = t > 0x8000000000UL ? 1U : (t == 0x8000000000UL ? v & 1U : 0U);
  //   return as_float(v + r);
  // }

  auto Zero32 = MIRBuilder.buildConstant(S32, 0);
  auto Zero64 = MIRBuilder.buildConstant(S64, 0);

  auto LZ = MIRBuilder.buildCTLZ_ZERO_UNDEF(S32, Src);

  auto K = MIRBuilder.buildConstant(S32, 127U + 63U);
  auto Sub = MIRBuilder.buildSub(S32, K, LZ);

  auto NotZero = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1, Src, Zero64);
  auto E = MIRBuilder.buildSelect(S32, NotZero, Sub, Zero32);

  auto Mask0 = MIRBuilder.buildConstant(S64, (-1ULL) >> 1);
  auto ShlLZ = MIRBuilder.buildShl(S64, Src, LZ);

  auto U = MIRBuilder.buildAnd(S64, ShlLZ, Mask0);

  auto Mask1 = MIRBuilder.buildConstant(S64, 0xffffffffffULL);
  auto T = MIRBuilder.buildAnd(S64, U, Mask1);

  auto UShl = MIRBuilder.buildLShr(S64, U, MIRBuilder.buildConstant(S64, 40));
  auto ShlE = MIRBuilder.buildShl(S32, E, MIRBuilder.buildConstant(S32, 23));
  auto V = MIRBuilder.buildOr(S32, ShlE, MIRBuilder.buildTrunc(S32, UShl));

  auto C = MIRBuilder.buildConstant(S64, 0x8000000000ULL);
  auto RCmp = MIRBuilder.buildICmp(CmpInst::ICMP_UGT, S1, T, C);
  auto TCmp = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, S1, T, C);
  auto One = MIRBuilder.buildConstant(S32, 1);

  auto VTrunc1 = MIRBuilder.buildAnd(S32, V, One);
  auto Select0 = MIRBuilder.buildSelect(S32, TCmp, VTrunc1, Zero32);
  auto R = MIRBuilder.buildSelect(S32, RCmp, One, Select0);
  MIRBuilder.buildAdd(Dst, V, R);

  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerUITOFP(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (SrcTy == LLT::scalar(1)) {
    auto True = MIRBuilder.buildFConstant(DstTy, 1.0);
    auto False = MIRBuilder.buildFConstant(DstTy, 0.0);
    MIRBuilder.buildSelect(Dst, Src, True, False);
    MI.eraseFromParent();
    return Legalized;
  }

  if (SrcTy != LLT::scalar(64))
    return UnableToLegalize;

  if (DstTy == LLT::scalar(32)) {
    // TODO: SelectionDAG has several alternative expansions to port which may
    // be more reasonble depending on the available instructions. If a target
    // has sitofp, does not have CTLZ, or can efficiently use f64 as an
    // intermediate type, this is probably worse.
    return lowerU64ToF32BitOps(MI);
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerSITOFP(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  const LLT S64 = LLT::scalar(64);
  const LLT S32 = LLT::scalar(32);
  const LLT S1 = LLT::scalar(1);

  if (SrcTy == S1) {
    auto True = MIRBuilder.buildFConstant(DstTy, -1.0);
    auto False = MIRBuilder.buildFConstant(DstTy, 0.0);
    MIRBuilder.buildSelect(Dst, Src, True, False);
    MI.eraseFromParent();
    return Legalized;
  }

  if (SrcTy != S64)
    return UnableToLegalize;

  if (DstTy == S32) {
    // signed cl2f(long l) {
    //   long s = l >> 63;
    //   float r = cul2f((l + s) ^ s);
    //   return s ? -r : r;
    // }
    Register L = Src;
    auto SignBit = MIRBuilder.buildConstant(S64, 63);
    auto S = MIRBuilder.buildAShr(S64, L, SignBit);

    auto LPlusS = MIRBuilder.buildAdd(S64, L, S);
    auto Xor = MIRBuilder.buildXor(S64, LPlusS, S);
    auto R = MIRBuilder.buildUITOFP(S32, Xor);

    auto RNeg = MIRBuilder.buildFNeg(S32, R);
    auto SignNotZero = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1, S,
                                            MIRBuilder.buildConstant(S64, 0));
    MIRBuilder.buildSelect(Dst, SignNotZero, RNeg, R);
    return Legalized;
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerFPTOUI(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);
  const LLT S64 = LLT::scalar(64);
  const LLT S32 = LLT::scalar(32);

  if (SrcTy != S64 && SrcTy != S32)
    return UnableToLegalize;
  if (DstTy != S32 && DstTy != S64)
    return UnableToLegalize;

  // FPTOSI gives same result as FPTOUI for positive signed integers.
  // FPTOUI needs to deal with fp values that convert to unsigned integers
  // greater or equal to 2^31 for float or 2^63 for double. For brevity 2^Exp.

  APInt TwoPExpInt = APInt::getSignMask(DstTy.getSizeInBits());
  APFloat TwoPExpFP(SrcTy.getSizeInBits() == 32 ? APFloat::IEEEsingle()
                                                : APFloat::IEEEdouble(),
                    APInt::getNullValue(SrcTy.getSizeInBits()));
  TwoPExpFP.convertFromAPInt(TwoPExpInt, false, APFloat::rmNearestTiesToEven);

  MachineInstrBuilder FPTOSI = MIRBuilder.buildFPTOSI(DstTy, Src);

  MachineInstrBuilder Threshold = MIRBuilder.buildFConstant(SrcTy, TwoPExpFP);
  // For fp Value greater or equal to Threshold(2^Exp), we use FPTOSI on
  // (Value - 2^Exp) and add 2^Exp by setting highest bit in result to 1.
  MachineInstrBuilder FSub = MIRBuilder.buildFSub(SrcTy, Src, Threshold);
  MachineInstrBuilder ResLowBits = MIRBuilder.buildFPTOSI(DstTy, FSub);
  MachineInstrBuilder ResHighBit = MIRBuilder.buildConstant(DstTy, TwoPExpInt);
  MachineInstrBuilder Res = MIRBuilder.buildXor(DstTy, ResLowBits, ResHighBit);

  const LLT S1 = LLT::scalar(1);

  MachineInstrBuilder FCMP =
      MIRBuilder.buildFCmp(CmpInst::FCMP_ULT, S1, Src, Threshold);
  MIRBuilder.buildSelect(Dst, FCMP, FPTOSI, Res);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult LegalizerHelper::lowerFPTOSI(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);
  const LLT S64 = LLT::scalar(64);
  const LLT S32 = LLT::scalar(32);

  // FIXME: Only f32 to i64 conversions are supported.
  if (SrcTy.getScalarType() != S32 || DstTy.getScalarType() != S64)
    return UnableToLegalize;

  // Expand f32 -> i64 conversion
  // This algorithm comes from compiler-rt's implementation of fixsfdi:
  // https://github.com/llvm/llvm-project/blob/master/compiler-rt/lib/builtins/fixsfdi.c

  unsigned SrcEltBits = SrcTy.getScalarSizeInBits();

  auto ExponentMask = MIRBuilder.buildConstant(SrcTy, 0x7F800000);
  auto ExponentLoBit = MIRBuilder.buildConstant(SrcTy, 23);

  auto AndExpMask = MIRBuilder.buildAnd(SrcTy, Src, ExponentMask);
  auto ExponentBits = MIRBuilder.buildLShr(SrcTy, AndExpMask, ExponentLoBit);

  auto SignMask = MIRBuilder.buildConstant(SrcTy,
                                           APInt::getSignMask(SrcEltBits));
  auto AndSignMask = MIRBuilder.buildAnd(SrcTy, Src, SignMask);
  auto SignLowBit = MIRBuilder.buildConstant(SrcTy, SrcEltBits - 1);
  auto Sign = MIRBuilder.buildAShr(SrcTy, AndSignMask, SignLowBit);
  Sign = MIRBuilder.buildSExt(DstTy, Sign);

  auto MantissaMask = MIRBuilder.buildConstant(SrcTy, 0x007FFFFF);
  auto AndMantissaMask = MIRBuilder.buildAnd(SrcTy, Src, MantissaMask);
  auto K = MIRBuilder.buildConstant(SrcTy, 0x00800000);

  auto R = MIRBuilder.buildOr(SrcTy, AndMantissaMask, K);
  R = MIRBuilder.buildZExt(DstTy, R);

  auto Bias = MIRBuilder.buildConstant(SrcTy, 127);
  auto Exponent = MIRBuilder.buildSub(SrcTy, ExponentBits, Bias);
  auto SubExponent = MIRBuilder.buildSub(SrcTy, Exponent, ExponentLoBit);
  auto ExponentSub = MIRBuilder.buildSub(SrcTy, ExponentLoBit, Exponent);

  auto Shl = MIRBuilder.buildShl(DstTy, R, SubExponent);
  auto Srl = MIRBuilder.buildLShr(DstTy, R, ExponentSub);

  const LLT S1 = LLT::scalar(1);
  auto CmpGt = MIRBuilder.buildICmp(CmpInst::ICMP_SGT,
                                    S1, Exponent, ExponentLoBit);

  R = MIRBuilder.buildSelect(DstTy, CmpGt, Shl, Srl);

  auto XorSign = MIRBuilder.buildXor(DstTy, R, Sign);
  auto Ret = MIRBuilder.buildSub(DstTy, XorSign, Sign);

  auto ZeroSrcTy = MIRBuilder.buildConstant(SrcTy, 0);

  auto ExponentLt0 = MIRBuilder.buildICmp(CmpInst::ICMP_SLT,
                                          S1, Exponent, ZeroSrcTy);

  auto ZeroDstTy = MIRBuilder.buildConstant(DstTy, 0);
  MIRBuilder.buildSelect(Dst, ExponentLt0, ZeroDstTy, Ret);

  MI.eraseFromParent();
  return Legalized;
}

// f64 -> f16 conversion using round-to-nearest-even rounding mode.
LegalizerHelper::LegalizeResult
LegalizerHelper::lowerFPTRUNC_F64_TO_F16(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();

  if (MRI.getType(Src).isVector()) // TODO: Handle vectors directly.
    return UnableToLegalize;

  const unsigned ExpMask = 0x7ff;
  const unsigned ExpBiasf64 = 1023;
  const unsigned ExpBiasf16 = 15;
  const LLT S32 = LLT::scalar(32);
  const LLT S1 = LLT::scalar(1);

  auto Unmerge = MIRBuilder.buildUnmerge(S32, Src);
  Register U = Unmerge.getReg(0);
  Register UH = Unmerge.getReg(1);

  auto E = MIRBuilder.buildLShr(S32, UH, MIRBuilder.buildConstant(S32, 20));

  // Subtract the fp64 exponent bias (1023) to get the real exponent and
  // add the f16 bias (15) to get the biased exponent for the f16 format.
  E = MIRBuilder.buildAdd(
    S32, E, MIRBuilder.buildConstant(S32, -ExpBiasf64 + ExpBiasf16));
  E = MIRBuilder.buildAnd(S32, E, MIRBuilder.buildConstant(S32, ExpMask));

  auto M = MIRBuilder.buildLShr(S32, UH, MIRBuilder.buildConstant(S32, 8));
  M = MIRBuilder.buildAnd(S32, M, MIRBuilder.buildConstant(S32, 0xffe));

  auto MaskedSig = MIRBuilder.buildAnd(S32, UH,
                                       MIRBuilder.buildConstant(S32, 0x1ff));
  MaskedSig = MIRBuilder.buildOr(S32, MaskedSig, U);

  auto Zero = MIRBuilder.buildConstant(S32, 0);
  auto SigCmpNE0 = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1, MaskedSig, Zero);
  auto Lo40Set = MIRBuilder.buildZExt(S32, SigCmpNE0);
  M = MIRBuilder.buildOr(S32, M, Lo40Set);

  // (M != 0 ? 0x0200 : 0) | 0x7c00;
  auto Bits0x200 = MIRBuilder.buildConstant(S32, 0x0200);
  auto CmpM_NE0 = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1, M, Zero);
  auto SelectCC = MIRBuilder.buildSelect(S32, CmpM_NE0, Bits0x200, Zero);

  auto Bits0x7c00 = MIRBuilder.buildConstant(S32, 0x7c00);
  auto I = MIRBuilder.buildOr(S32, SelectCC, Bits0x7c00);

  // N = M | (E << 12);
  auto EShl12 = MIRBuilder.buildShl(S32, E, MIRBuilder.buildConstant(S32, 12));
  auto N = MIRBuilder.buildOr(S32, M, EShl12);

  // B = clamp(1-E, 0, 13);
  auto One = MIRBuilder.buildConstant(S32, 1);
  auto OneSubExp = MIRBuilder.buildSub(S32, One, E);
  auto B = MIRBuilder.buildSMax(S32, OneSubExp, Zero);
  B = MIRBuilder.buildSMin(S32, B, MIRBuilder.buildConstant(S32, 13));

  auto SigSetHigh = MIRBuilder.buildOr(S32, M,
                                       MIRBuilder.buildConstant(S32, 0x1000));

  auto D = MIRBuilder.buildLShr(S32, SigSetHigh, B);
  auto D0 = MIRBuilder.buildShl(S32, D, B);

  auto D0_NE_SigSetHigh = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1,
                                             D0, SigSetHigh);
  auto D1 = MIRBuilder.buildZExt(S32, D0_NE_SigSetHigh);
  D = MIRBuilder.buildOr(S32, D, D1);

  auto CmpELtOne = MIRBuilder.buildICmp(CmpInst::ICMP_SLT, S1, E, One);
  auto V = MIRBuilder.buildSelect(S32, CmpELtOne, D, N);

  auto VLow3 = MIRBuilder.buildAnd(S32, V, MIRBuilder.buildConstant(S32, 7));
  V = MIRBuilder.buildLShr(S32, V, MIRBuilder.buildConstant(S32, 2));

  auto VLow3Eq3 = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, S1, VLow3,
                                       MIRBuilder.buildConstant(S32, 3));
  auto V0 = MIRBuilder.buildZExt(S32, VLow3Eq3);

  auto VLow3Gt5 = MIRBuilder.buildICmp(CmpInst::ICMP_SGT, S1, VLow3,
                                       MIRBuilder.buildConstant(S32, 5));
  auto V1 = MIRBuilder.buildZExt(S32, VLow3Gt5);

  V1 = MIRBuilder.buildOr(S32, V0, V1);
  V = MIRBuilder.buildAdd(S32, V, V1);

  auto CmpEGt30 = MIRBuilder.buildICmp(CmpInst::ICMP_SGT,  S1,
                                       E, MIRBuilder.buildConstant(S32, 30));
  V = MIRBuilder.buildSelect(S32, CmpEGt30,
                             MIRBuilder.buildConstant(S32, 0x7c00), V);

  auto CmpEGt1039 = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, S1,
                                         E, MIRBuilder.buildConstant(S32, 1039));
  V = MIRBuilder.buildSelect(S32, CmpEGt1039, I, V);

  // Extract the sign bit.
  auto Sign = MIRBuilder.buildLShr(S32, UH, MIRBuilder.buildConstant(S32, 16));
  Sign = MIRBuilder.buildAnd(S32, Sign, MIRBuilder.buildConstant(S32, 0x8000));

  // Insert the sign bit
  V = MIRBuilder.buildOr(S32, Sign, V);

  MIRBuilder.buildTrunc(Dst, V);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerFPTRUNC(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();

  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);
  const LLT S64 = LLT::scalar(64);
  const LLT S16 = LLT::scalar(16);

  if (DstTy.getScalarType() == S16 && SrcTy.getScalarType() == S64)
    return lowerFPTRUNC_F64_TO_F16(MI);

  return UnableToLegalize;
}

static CmpInst::Predicate minMaxToCompare(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_SMIN:
    return CmpInst::ICMP_SLT;
  case TargetOpcode::G_SMAX:
    return CmpInst::ICMP_SGT;
  case TargetOpcode::G_UMIN:
    return CmpInst::ICMP_ULT;
  case TargetOpcode::G_UMAX:
    return CmpInst::ICMP_UGT;
  default:
    llvm_unreachable("not in integer min/max");
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerMinMax(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src0 = MI.getOperand(1).getReg();
  Register Src1 = MI.getOperand(2).getReg();

  const CmpInst::Predicate Pred = minMaxToCompare(MI.getOpcode());
  LLT CmpType = MRI.getType(Dst).changeElementSize(1);

  auto Cmp = MIRBuilder.buildICmp(Pred, CmpType, Src0, Src1);
  MIRBuilder.buildSelect(Dst, Cmp, Src0, Src1);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerFCopySign(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src0 = MI.getOperand(1).getReg();
  Register Src1 = MI.getOperand(2).getReg();

  const LLT Src0Ty = MRI.getType(Src0);
  const LLT Src1Ty = MRI.getType(Src1);

  const int Src0Size = Src0Ty.getScalarSizeInBits();
  const int Src1Size = Src1Ty.getScalarSizeInBits();

  auto SignBitMask = MIRBuilder.buildConstant(
    Src0Ty, APInt::getSignMask(Src0Size));

  auto NotSignBitMask = MIRBuilder.buildConstant(
    Src0Ty, APInt::getLowBitsSet(Src0Size, Src0Size - 1));

  auto And0 = MIRBuilder.buildAnd(Src0Ty, Src0, NotSignBitMask);
  MachineInstr *Or;

  if (Src0Ty == Src1Ty) {
    auto And1 = MIRBuilder.buildAnd(Src1Ty, Src1, SignBitMask);
    Or = MIRBuilder.buildOr(Dst, And0, And1);
  } else if (Src0Size > Src1Size) {
    auto ShiftAmt = MIRBuilder.buildConstant(Src0Ty, Src0Size - Src1Size);
    auto Zext = MIRBuilder.buildZExt(Src0Ty, Src1);
    auto Shift = MIRBuilder.buildShl(Src0Ty, Zext, ShiftAmt);
    auto And1 = MIRBuilder.buildAnd(Src0Ty, Shift, SignBitMask);
    Or = MIRBuilder.buildOr(Dst, And0, And1);
  } else {
    auto ShiftAmt = MIRBuilder.buildConstant(Src1Ty, Src1Size - Src0Size);
    auto Shift = MIRBuilder.buildLShr(Src1Ty, Src1, ShiftAmt);
    auto Trunc = MIRBuilder.buildTrunc(Src0Ty, Shift);
    auto And1 = MIRBuilder.buildAnd(Src0Ty, Trunc, SignBitMask);
    Or = MIRBuilder.buildOr(Dst, And0, And1);
  }

  // Be careful about setting nsz/nnan/ninf on every instruction, since the
  // constants are a nan and -0.0, but the final result should preserve
  // everything.
  if (unsigned Flags = MI.getFlags())
    Or->setFlags(Flags);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerFMinNumMaxNum(MachineInstr &MI) {
  unsigned NewOp = MI.getOpcode() == TargetOpcode::G_FMINNUM ?
    TargetOpcode::G_FMINNUM_IEEE : TargetOpcode::G_FMAXNUM_IEEE;

  Register Dst = MI.getOperand(0).getReg();
  Register Src0 = MI.getOperand(1).getReg();
  Register Src1 = MI.getOperand(2).getReg();
  LLT Ty = MRI.getType(Dst);

  if (!MI.getFlag(MachineInstr::FmNoNans)) {
    // Insert canonicalizes if it's possible we need to quiet to get correct
    // sNaN behavior.

    // Note this must be done here, and not as an optimization combine in the
    // absence of a dedicate quiet-snan instruction as we're using an
    // omni-purpose G_FCANONICALIZE.
    if (!isKnownNeverSNaN(Src0, MRI))
      Src0 = MIRBuilder.buildFCanonicalize(Ty, Src0, MI.getFlags()).getReg(0);

    if (!isKnownNeverSNaN(Src1, MRI))
      Src1 = MIRBuilder.buildFCanonicalize(Ty, Src1, MI.getFlags()).getReg(0);
  }

  // If there are no nans, it's safe to simply replace this with the non-IEEE
  // version.
  MIRBuilder.buildInstr(NewOp, {Dst}, {Src0, Src1}, MI.getFlags());
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult LegalizerHelper::lowerFMad(MachineInstr &MI) {
  // Expand G_FMAD a, b, c -> G_FADD (G_FMUL a, b), c
  Register DstReg = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(DstReg);
  unsigned Flags = MI.getFlags();

  auto Mul = MIRBuilder.buildFMul(Ty, MI.getOperand(1), MI.getOperand(2),
                                  Flags);
  MIRBuilder.buildFAdd(DstReg, Mul, MI.getOperand(3), Flags);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerIntrinsicRound(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  Register X = MI.getOperand(1).getReg();
  const unsigned Flags = MI.getFlags();
  const LLT Ty = MRI.getType(DstReg);
  const LLT CondTy = Ty.changeElementSize(1);

  // round(x) =>
  //  t = trunc(x);
  //  d = fabs(x - t);
  //  o = copysign(1.0f, x);
  //  return t + (d >= 0.5 ? o : 0.0);

  auto T = MIRBuilder.buildIntrinsicTrunc(Ty, X, Flags);

  auto Diff = MIRBuilder.buildFSub(Ty, X, T, Flags);
  auto AbsDiff = MIRBuilder.buildFAbs(Ty, Diff, Flags);
  auto Zero = MIRBuilder.buildFConstant(Ty, 0.0);
  auto One = MIRBuilder.buildFConstant(Ty, 1.0);
  auto Half = MIRBuilder.buildFConstant(Ty, 0.5);
  auto SignOne = MIRBuilder.buildFCopysign(Ty, One, X);

  auto Cmp = MIRBuilder.buildFCmp(CmpInst::FCMP_OGE, CondTy, AbsDiff, Half,
                                  Flags);
  auto Sel = MIRBuilder.buildSelect(Ty, Cmp, SignOne, Zero, Flags);

  MIRBuilder.buildFAdd(DstReg, T, Sel, Flags);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerFFloor(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  unsigned Flags = MI.getFlags();
  LLT Ty = MRI.getType(DstReg);
  const LLT CondTy = Ty.changeElementSize(1);

  // result = trunc(src);
  // if (src < 0.0 && src != result)
  //   result += -1.0.

  auto Trunc = MIRBuilder.buildIntrinsicTrunc(Ty, SrcReg, Flags);
  auto Zero = MIRBuilder.buildFConstant(Ty, 0.0);

  auto Lt0 = MIRBuilder.buildFCmp(CmpInst::FCMP_OLT, CondTy,
                                  SrcReg, Zero, Flags);
  auto NeTrunc = MIRBuilder.buildFCmp(CmpInst::FCMP_ONE, CondTy,
                                      SrcReg, Trunc, Flags);
  auto And = MIRBuilder.buildAnd(CondTy, Lt0, NeTrunc);
  auto AddVal = MIRBuilder.buildSITOFP(Ty, And);

  MIRBuilder.buildFAdd(DstReg, Trunc, AddVal, Flags);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerMergeValues(MachineInstr &MI) {
  const unsigned NumOps = MI.getNumOperands();
  Register DstReg = MI.getOperand(0).getReg();
  Register Src0Reg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(Src0Reg);
  unsigned PartSize = SrcTy.getSizeInBits();

  LLT WideTy = LLT::scalar(DstTy.getSizeInBits());
  Register ResultReg = MIRBuilder.buildZExt(WideTy, Src0Reg).getReg(0);

  for (unsigned I = 2; I != NumOps; ++I) {
    const unsigned Offset = (I - 1) * PartSize;

    Register SrcReg = MI.getOperand(I).getReg();
    auto ZextInput = MIRBuilder.buildZExt(WideTy, SrcReg);

    Register NextResult = I + 1 == NumOps && WideTy == DstTy ? DstReg :
      MRI.createGenericVirtualRegister(WideTy);

    auto ShiftAmt = MIRBuilder.buildConstant(WideTy, Offset);
    auto Shl = MIRBuilder.buildShl(WideTy, ZextInput, ShiftAmt);
    MIRBuilder.buildOr(NextResult, ResultReg, Shl);
    ResultReg = NextResult;
  }

  if (DstTy.isPointer()) {
    if (MIRBuilder.getDataLayout().isNonIntegralAddressSpace(
          DstTy.getAddressSpace())) {
      LLVM_DEBUG(dbgs() << "Not casting nonintegral address space\n");
      return UnableToLegalize;
    }

    MIRBuilder.buildIntToPtr(DstReg, ResultReg);
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerUnmergeValues(MachineInstr &MI) {
  const unsigned NumDst = MI.getNumOperands() - 1;
  Register SrcReg = MI.getOperand(NumDst).getReg();
  Register Dst0Reg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst0Reg);
  if (DstTy.isPointer())
    return UnableToLegalize; // TODO

  SrcReg = coerceToScalar(SrcReg);
  if (!SrcReg)
    return UnableToLegalize;

  // Expand scalarizing unmerge as bitcast to integer and shift.
  LLT IntTy = MRI.getType(SrcReg);

  MIRBuilder.buildTrunc(Dst0Reg, SrcReg);

  const unsigned DstSize = DstTy.getSizeInBits();
  unsigned Offset = DstSize;
  for (unsigned I = 1; I != NumDst; ++I, Offset += DstSize) {
    auto ShiftAmt = MIRBuilder.buildConstant(IntTy, Offset);
    auto Shift = MIRBuilder.buildLShr(IntTy, SrcReg, ShiftAmt);
    MIRBuilder.buildTrunc(MI.getOperand(I), Shift);
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerShuffleVector(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  Register Src0Reg = MI.getOperand(1).getReg();
  Register Src1Reg = MI.getOperand(2).getReg();
  LLT Src0Ty = MRI.getType(Src0Reg);
  LLT DstTy = MRI.getType(DstReg);
  LLT IdxTy = LLT::scalar(32);

  ArrayRef<int> Mask = MI.getOperand(3).getShuffleMask();

  if (DstTy.isScalar()) {
    if (Src0Ty.isVector())
      return UnableToLegalize;

    // This is just a SELECT.
    assert(Mask.size() == 1 && "Expected a single mask element");
    Register Val;
    if (Mask[0] < 0 || Mask[0] > 1)
      Val = MIRBuilder.buildUndef(DstTy).getReg(0);
    else
      Val = Mask[0] == 0 ? Src0Reg : Src1Reg;
    MIRBuilder.buildCopy(DstReg, Val);
    MI.eraseFromParent();
    return Legalized;
  }

  Register Undef;
  SmallVector<Register, 32> BuildVec;
  LLT EltTy = DstTy.getElementType();

  for (int Idx : Mask) {
    if (Idx < 0) {
      if (!Undef.isValid())
        Undef = MIRBuilder.buildUndef(EltTy).getReg(0);
      BuildVec.push_back(Undef);
      continue;
    }

    if (Src0Ty.isScalar()) {
      BuildVec.push_back(Idx == 0 ? Src0Reg : Src1Reg);
    } else {
      int NumElts = Src0Ty.getNumElements();
      Register SrcVec = Idx < NumElts ? Src0Reg : Src1Reg;
      int ExtractIdx = Idx < NumElts ? Idx : Idx - NumElts;
      auto IdxK = MIRBuilder.buildConstant(IdxTy, ExtractIdx);
      auto Extract = MIRBuilder.buildExtractVectorElement(EltTy, SrcVec, IdxK);
      BuildVec.push_back(Extract.getReg(0));
    }
  }

  MIRBuilder.buildBuildVector(DstReg, BuildVec);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerDynStackAlloc(MachineInstr &MI) {
  const auto &MF = *MI.getMF();
  const auto &TFI = *MF.getSubtarget().getFrameLowering();
  if (TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp)
    return UnableToLegalize;

  Register Dst = MI.getOperand(0).getReg();
  Register AllocSize = MI.getOperand(1).getReg();
  Align Alignment = assumeAligned(MI.getOperand(2).getImm());

  LLT PtrTy = MRI.getType(Dst);
  LLT IntPtrTy = LLT::scalar(PtrTy.getSizeInBits());

  const auto &TLI = *MF.getSubtarget().getTargetLowering();
  Register SPReg = TLI.getStackPointerRegisterToSaveRestore();
  auto SPTmp = MIRBuilder.buildCopy(PtrTy, SPReg);
  SPTmp = MIRBuilder.buildCast(IntPtrTy, SPTmp);

  // Subtract the final alloc from the SP. We use G_PTRTOINT here so we don't
  // have to generate an extra instruction to negate the alloc and then use
  // G_PTR_ADD to add the negative offset.
  auto Alloc = MIRBuilder.buildSub(IntPtrTy, SPTmp, AllocSize);
  if (Alignment > Align(1)) {
    APInt AlignMask(IntPtrTy.getSizeInBits(), Alignment.value(), true);
    AlignMask.negate();
    auto AlignCst = MIRBuilder.buildConstant(IntPtrTy, AlignMask);
    Alloc = MIRBuilder.buildAnd(IntPtrTy, Alloc, AlignCst);
  }

  SPTmp = MIRBuilder.buildCast(PtrTy, Alloc);
  MIRBuilder.buildCopy(SPReg, SPTmp);
  MIRBuilder.buildCopy(Dst, SPTmp);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerExtract(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  unsigned Offset = MI.getOperand(2).getImm();

  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (DstTy.isScalar() &&
      (SrcTy.isScalar() ||
       (SrcTy.isVector() && DstTy == SrcTy.getElementType()))) {
    LLT SrcIntTy = SrcTy;
    if (!SrcTy.isScalar()) {
      SrcIntTy = LLT::scalar(SrcTy.getSizeInBits());
      Src = MIRBuilder.buildBitcast(SrcIntTy, Src).getReg(0);
    }

    if (Offset == 0)
      MIRBuilder.buildTrunc(Dst, Src);
    else {
      auto ShiftAmt = MIRBuilder.buildConstant(SrcIntTy, Offset);
      auto Shr = MIRBuilder.buildLShr(SrcIntTy, Src, ShiftAmt);
      MIRBuilder.buildTrunc(Dst, Shr);
    }

    MI.eraseFromParent();
    return Legalized;
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult LegalizerHelper::lowerInsert(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  Register InsertSrc = MI.getOperand(2).getReg();
  uint64_t Offset = MI.getOperand(3).getImm();

  LLT DstTy = MRI.getType(Src);
  LLT InsertTy = MRI.getType(InsertSrc);

  if (InsertTy.isVector() ||
      (DstTy.isVector() && DstTy.getElementType() != InsertTy))
    return UnableToLegalize;

  const DataLayout &DL = MIRBuilder.getDataLayout();
  if ((DstTy.isPointer() &&
       DL.isNonIntegralAddressSpace(DstTy.getAddressSpace())) ||
      (InsertTy.isPointer() &&
       DL.isNonIntegralAddressSpace(InsertTy.getAddressSpace()))) {
    LLVM_DEBUG(dbgs() << "Not casting non-integral address space integer\n");
    return UnableToLegalize;
  }

  LLT IntDstTy = DstTy;

  if (!DstTy.isScalar()) {
    IntDstTy = LLT::scalar(DstTy.getSizeInBits());
    Src = MIRBuilder.buildCast(IntDstTy, Src).getReg(0);
  }

  if (!InsertTy.isScalar()) {
    const LLT IntInsertTy = LLT::scalar(InsertTy.getSizeInBits());
    InsertSrc = MIRBuilder.buildPtrToInt(IntInsertTy, InsertSrc).getReg(0);
  }

  Register ExtInsSrc = MIRBuilder.buildZExt(IntDstTy, InsertSrc).getReg(0);
  if (Offset != 0) {
    auto ShiftAmt = MIRBuilder.buildConstant(IntDstTy, Offset);
    ExtInsSrc = MIRBuilder.buildShl(IntDstTy, ExtInsSrc, ShiftAmt).getReg(0);
  }

  APInt MaskVal = APInt::getBitsSetWithWrap(
      DstTy.getSizeInBits(), Offset + InsertTy.getSizeInBits(), Offset);

  auto Mask = MIRBuilder.buildConstant(IntDstTy, MaskVal);
  auto MaskedSrc = MIRBuilder.buildAnd(IntDstTy, Src, Mask);
  auto Or = MIRBuilder.buildOr(IntDstTy, MaskedSrc, ExtInsSrc);

  MIRBuilder.buildCast(Dst, Or);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerSADDO_SSUBO(MachineInstr &MI) {
  Register Dst0 = MI.getOperand(0).getReg();
  Register Dst1 = MI.getOperand(1).getReg();
  Register LHS = MI.getOperand(2).getReg();
  Register RHS = MI.getOperand(3).getReg();
  const bool IsAdd = MI.getOpcode() == TargetOpcode::G_SADDO;

  LLT Ty = MRI.getType(Dst0);
  LLT BoolTy = MRI.getType(Dst1);

  if (IsAdd)
    MIRBuilder.buildAdd(Dst0, LHS, RHS);
  else
    MIRBuilder.buildSub(Dst0, LHS, RHS);

  // TODO: If SADDSAT/SSUBSAT is legal, compare results to detect overflow.

  auto Zero = MIRBuilder.buildConstant(Ty, 0);

  // For an addition, the result should be less than one of the operands (LHS)
  // if and only if the other operand (RHS) is negative, otherwise there will
  // be overflow.
  // For a subtraction, the result should be less than one of the operands
  // (LHS) if and only if the other operand (RHS) is (non-zero) positive,
  // otherwise there will be overflow.
  auto ResultLowerThanLHS =
      MIRBuilder.buildICmp(CmpInst::ICMP_SLT, BoolTy, Dst0, LHS);
  auto ConditionRHS = MIRBuilder.buildICmp(
      IsAdd ? CmpInst::ICMP_SLT : CmpInst::ICMP_SGT, BoolTy, RHS, Zero);

  MIRBuilder.buildXor(Dst1, ConditionRHS, ResultLowerThanLHS);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerBswap(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const LLT Ty = MRI.getType(Src);
  unsigned SizeInBytes = (Ty.getScalarSizeInBits() + 7) / 8;
  unsigned BaseShiftAmt = (SizeInBytes - 1) * 8;

  // Swap most and least significant byte, set remaining bytes in Res to zero.
  auto ShiftAmt = MIRBuilder.buildConstant(Ty, BaseShiftAmt);
  auto LSByteShiftedLeft = MIRBuilder.buildShl(Ty, Src, ShiftAmt);
  auto MSByteShiftedRight = MIRBuilder.buildLShr(Ty, Src, ShiftAmt);
  auto Res = MIRBuilder.buildOr(Ty, MSByteShiftedRight, LSByteShiftedLeft);

  // Set i-th high/low byte in Res to i-th low/high byte from Src.
  for (unsigned i = 1; i < SizeInBytes / 2; ++i) {
    // AND with Mask leaves byte i unchanged and sets remaining bytes to 0.
    APInt APMask(SizeInBytes * 8, 0xFF << (i * 8));
    auto Mask = MIRBuilder.buildConstant(Ty, APMask);
    auto ShiftAmt = MIRBuilder.buildConstant(Ty, BaseShiftAmt - 16 * i);
    // Low byte shifted left to place of high byte: (Src & Mask) << ShiftAmt.
    auto LoByte = MIRBuilder.buildAnd(Ty, Src, Mask);
    auto LoShiftedLeft = MIRBuilder.buildShl(Ty, LoByte, ShiftAmt);
    Res = MIRBuilder.buildOr(Ty, Res, LoShiftedLeft);
    // High byte shifted right to place of low byte: (Src >> ShiftAmt) & Mask.
    auto SrcShiftedRight = MIRBuilder.buildLShr(Ty, Src, ShiftAmt);
    auto HiShiftedRight = MIRBuilder.buildAnd(Ty, SrcShiftedRight, Mask);
    Res = MIRBuilder.buildOr(Ty, Res, HiShiftedRight);
  }
  Res.getInstr()->getOperand(0).setReg(Dst);

  MI.eraseFromParent();
  return Legalized;
}

//{ (Src & Mask) >> N } | { (Src << N) & Mask }
static MachineInstrBuilder SwapN(unsigned N, DstOp Dst, MachineIRBuilder &B,
                                 MachineInstrBuilder Src, APInt Mask) {
  const LLT Ty = Dst.getLLTTy(*B.getMRI());
  MachineInstrBuilder C_N = B.buildConstant(Ty, N);
  MachineInstrBuilder MaskLoNTo0 = B.buildConstant(Ty, Mask);
  auto LHS = B.buildLShr(Ty, B.buildAnd(Ty, Src, MaskLoNTo0), C_N);
  auto RHS = B.buildAnd(Ty, B.buildShl(Ty, Src, C_N), MaskLoNTo0);
  return B.buildOr(Dst, LHS, RHS);
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerBitreverse(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const LLT Ty = MRI.getType(Src);
  unsigned Size = Ty.getSizeInBits();

  MachineInstrBuilder BSWAP =
      MIRBuilder.buildInstr(TargetOpcode::G_BSWAP, {Ty}, {Src});

  // swap high and low 4 bits in 8 bit blocks 7654|3210 -> 3210|7654
  //    [(val & 0xF0F0F0F0) >> 4] | [(val & 0x0F0F0F0F) << 4]
  // -> [(val & 0xF0F0F0F0) >> 4] | [(val << 4) & 0xF0F0F0F0]
  MachineInstrBuilder Swap4 =
      SwapN(4, Ty, MIRBuilder, BSWAP, APInt::getSplat(Size, APInt(8, 0xF0)));

  // swap high and low 2 bits in 4 bit blocks 32|10 76|54 -> 10|32 54|76
  //    [(val & 0xCCCCCCCC) >> 2] & [(val & 0x33333333) << 2]
  // -> [(val & 0xCCCCCCCC) >> 2] & [(val << 2) & 0xCCCCCCCC]
  MachineInstrBuilder Swap2 =
      SwapN(2, Ty, MIRBuilder, Swap4, APInt::getSplat(Size, APInt(8, 0xCC)));

  // swap high and low 1 bit in 2 bit blocks 1|0 3|2 5|4 7|6 -> 0|1 2|3 4|5 6|7
  //    [(val & 0xAAAAAAAA) >> 1] & [(val & 0x55555555) << 1]
  // -> [(val & 0xAAAAAAAA) >> 1] & [(val << 1) & 0xAAAAAAAA]
  SwapN(1, Dst, MIRBuilder, Swap2, APInt::getSplat(Size, APInt(8, 0xAA)));

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerReadWriteRegister(MachineInstr &MI) {
  MachineFunction &MF = MIRBuilder.getMF();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetLowering *TLI = STI.getTargetLowering();

  bool IsRead = MI.getOpcode() == TargetOpcode::G_READ_REGISTER;
  int NameOpIdx = IsRead ? 1 : 0;
  int ValRegIndex = IsRead ? 0 : 1;

  Register ValReg = MI.getOperand(ValRegIndex).getReg();
  const LLT Ty = MRI.getType(ValReg);
  const MDString *RegStr = cast<MDString>(
    cast<MDNode>(MI.getOperand(NameOpIdx).getMetadata())->getOperand(0));

  Register PhysReg = TLI->getRegisterByName(RegStr->getString().data(), Ty, MF);
  if (!PhysReg.isValid())
    return UnableToLegalize;

  if (IsRead)
    MIRBuilder.buildCopy(ValReg, PhysReg);
  else
    MIRBuilder.buildCopy(PhysReg, ValReg);

  MI.eraseFromParent();
  return Legalized;
}
