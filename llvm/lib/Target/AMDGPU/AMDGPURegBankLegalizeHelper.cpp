//===-- AMDGPURegBankLegalizeHelper.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Implements actual lowering algorithms for each ID that can be used in
/// Rule.OperandMapping. Similar to legalizer helper but with register banks.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURegBankLegalizeHelper.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPURegBankLegalizeRules.h"
#include "AMDGPURegisterBankInfo.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

#define DEBUG_TYPE "amdgpu-regbanklegalize"

using namespace llvm;
using namespace AMDGPU;

RegBankLegalizeHelper::RegBankLegalizeHelper(
    MachineIRBuilder &B, const MachineUniformityInfo &MUI,
    const RegisterBankInfo &RBI, const RegBankLegalizeRules &RBLRules)
    : ST(B.getMF().getSubtarget<GCNSubtarget>()), B(B), MRI(*B.getMRI()),
      MUI(MUI), RBI(RBI), RBLRules(RBLRules),
      SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
      VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
      VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)) {}

void RegBankLegalizeHelper::findRuleAndApplyMapping(MachineInstr &MI) {
  const SetOfRulesForOpcode &RuleSet = RBLRules.getRulesForOpc(MI);
  const RegBankLLTMapping &Mapping = RuleSet.findMappingForMI(MI, MRI, MUI);

  SmallSet<Register, 4> WaterfallSgprs;
  unsigned OpIdx = 0;
  if (Mapping.DstOpMapping.size() > 0) {
    B.setInsertPt(*MI.getParent(), std::next(MI.getIterator()));
    applyMappingDst(MI, OpIdx, Mapping.DstOpMapping);
  }
  if (Mapping.SrcOpMapping.size() > 0) {
    B.setInstr(MI);
    applyMappingSrc(MI, OpIdx, Mapping.SrcOpMapping, WaterfallSgprs);
  }

  lower(MI, Mapping, WaterfallSgprs);
}

void RegBankLegalizeHelper::splitLoad(MachineInstr &MI,
                                      ArrayRef<LLT> LLTBreakdown, LLT MergeTy) {
  MachineFunction &MF = B.getMF();
  assert(MI.getNumMemOperands() == 1);
  MachineMemOperand &BaseMMO = **MI.memoperands_begin();
  Register Dst = MI.getOperand(0).getReg();
  const RegisterBank *DstRB = MRI.getRegBankOrNull(Dst);
  Register Base = MI.getOperand(1).getReg();
  LLT PtrTy = MRI.getType(Base);
  const RegisterBank *PtrRB = MRI.getRegBankOrNull(Base);
  LLT OffsetTy = LLT::scalar(PtrTy.getSizeInBits());
  SmallVector<Register, 4> LoadPartRegs;

  unsigned ByteOffset = 0;
  for (LLT PartTy : LLTBreakdown) {
    Register BasePlusOffset;
    if (ByteOffset == 0) {
      BasePlusOffset = Base;
    } else {
      auto Offset = B.buildConstant({PtrRB, OffsetTy}, ByteOffset);
      BasePlusOffset = B.buildPtrAdd({PtrRB, PtrTy}, Base, Offset).getReg(0);
    }
    auto *OffsetMMO = MF.getMachineMemOperand(&BaseMMO, ByteOffset, PartTy);
    auto LoadPart = B.buildLoad({DstRB, PartTy}, BasePlusOffset, *OffsetMMO);
    LoadPartRegs.push_back(LoadPart.getReg(0));
    ByteOffset += PartTy.getSizeInBytes();
  }

  if (!MergeTy.isValid()) {
    // Loads are of same size, concat or merge them together.
    B.buildMergeLikeInstr(Dst, LoadPartRegs);
  } else {
    // Loads are not all of same size, need to unmerge them to smaller pieces
    // of MergeTy type, then merge pieces to Dst.
    SmallVector<Register, 4> MergeTyParts;
    for (Register Reg : LoadPartRegs) {
      if (MRI.getType(Reg) == MergeTy) {
        MergeTyParts.push_back(Reg);
      } else {
        auto Unmerge = B.buildUnmerge({DstRB, MergeTy}, Reg);
        for (unsigned i = 0; i < Unmerge->getNumOperands() - 1; ++i)
          MergeTyParts.push_back(Unmerge.getReg(i));
      }
    }
    B.buildMergeLikeInstr(Dst, MergeTyParts);
  }
  MI.eraseFromParent();
}

void RegBankLegalizeHelper::widenLoad(MachineInstr &MI, LLT WideTy,
                                      LLT MergeTy) {
  MachineFunction &MF = B.getMF();
  assert(MI.getNumMemOperands() == 1);
  MachineMemOperand &BaseMMO = **MI.memoperands_begin();
  Register Dst = MI.getOperand(0).getReg();
  const RegisterBank *DstRB = MRI.getRegBankOrNull(Dst);
  Register Base = MI.getOperand(1).getReg();

  MachineMemOperand *WideMMO = MF.getMachineMemOperand(&BaseMMO, 0, WideTy);
  auto WideLoad = B.buildLoad({DstRB, WideTy}, Base, *WideMMO);

  if (WideTy.isScalar()) {
    B.buildTrunc(Dst, WideLoad);
  } else {
    SmallVector<Register, 4> MergeTyParts;
    auto Unmerge = B.buildUnmerge({DstRB, MergeTy}, WideLoad);

    LLT DstTy = MRI.getType(Dst);
    unsigned NumElts = DstTy.getSizeInBits() / MergeTy.getSizeInBits();
    for (unsigned i = 0; i < NumElts; ++i) {
      MergeTyParts.push_back(Unmerge.getReg(i));
    }
    B.buildMergeLikeInstr(Dst, MergeTyParts);
  }
  MI.eraseFromParent();
}

static bool isSignedBFE(MachineInstr &MI) {
  if (GIntrinsic *GI = dyn_cast<GIntrinsic>(&MI))
    return (GI->is(Intrinsic::amdgcn_sbfe));

  return MI.getOpcode() == AMDGPU::G_SBFX;
}

void RegBankLegalizeHelper::lowerV_BFE(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  assert(MRI.getType(Dst) == LLT::scalar(64));
  bool Signed = isSignedBFE(MI);
  unsigned FirstOpnd = isa<GIntrinsic>(MI) ? 2 : 1;
  // Extract bitfield from Src, LSBit is the least-significant bit for the
  // extraction (field offset) and Width is size of bitfield.
  Register Src = MI.getOperand(FirstOpnd).getReg();
  Register LSBit = MI.getOperand(FirstOpnd + 1).getReg();
  Register Width = MI.getOperand(FirstOpnd + 2).getReg();
  // Comments are for signed bitfield extract, similar for unsigned. x is sign
  // bit. s is sign, l is LSB and y are remaining bits of bitfield to extract.

  // Src >> LSBit Hi|Lo: x?????syyyyyyl??? -> xxxx?????syyyyyyl
  unsigned SHROpc = Signed ? AMDGPU::G_ASHR : AMDGPU::G_LSHR;
  auto SHRSrc = B.buildInstr(SHROpc, {{VgprRB, S64}}, {Src, LSBit});

  auto ConstWidth = getIConstantVRegValWithLookThrough(Width, MRI);

  // Expand to Src >> LSBit << (64 - Width) >> (64 - Width)
  // << (64 - Width): Hi|Lo: xxxx?????syyyyyyl -> syyyyyyl000000000
  // >> (64 - Width): Hi|Lo: syyyyyyl000000000 -> ssssssssssyyyyyyl
  if (!ConstWidth) {
    auto Amt = B.buildSub(VgprRB_S32, B.buildConstant(SgprRB_S32, 64), Width);
    auto SignBit = B.buildShl({VgprRB, S64}, SHRSrc, Amt);
    B.buildInstr(SHROpc, {Dst}, {SignBit, Amt});
    MI.eraseFromParent();
    return;
  }

  uint64_t WidthImm = ConstWidth->Value.getZExtValue();
  auto UnmergeSHRSrc = B.buildUnmerge(VgprRB_S32, SHRSrc);
  Register SHRSrcLo = UnmergeSHRSrc.getReg(0);
  Register SHRSrcHi = UnmergeSHRSrc.getReg(1);
  auto Zero = B.buildConstant({VgprRB, S32}, 0);
  unsigned BFXOpc = Signed ? AMDGPU::G_SBFX : AMDGPU::G_UBFX;

  if (WidthImm <= 32) {
    // SHRSrc Hi|Lo: ????????|???syyyl -> ????????|ssssyyyl
    auto Lo = B.buildInstr(BFXOpc, {VgprRB_S32}, {SHRSrcLo, Zero, Width});
    MachineInstrBuilder Hi;
    if (Signed) {
      // SHRSrc Hi|Lo: ????????|ssssyyyl -> ssssssss|ssssyyyl
      Hi = B.buildAShr(VgprRB_S32, Lo, B.buildConstant(VgprRB_S32, 31));
    } else {
      // SHRSrc Hi|Lo: ????????|000syyyl -> 00000000|000syyyl
      Hi = Zero;
    }
    B.buildMergeLikeInstr(Dst, {Lo, Hi});
  } else {
    auto Amt = B.buildConstant(VgprRB_S32, WidthImm - 32);
    // SHRSrc Hi|Lo: ??????sy|yyyyyyyl -> sssssssy|yyyyyyyl
    auto Hi = B.buildInstr(BFXOpc, {VgprRB_S32}, {SHRSrcHi, Zero, Amt});
    B.buildMergeLikeInstr(Dst, {SHRSrcLo, Hi});
  }

  MI.eraseFromParent();
}

void RegBankLegalizeHelper::lowerS_BFE(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(DstReg);
  bool Signed = isSignedBFE(MI);
  unsigned FirstOpnd = isa<GIntrinsic>(MI) ? 2 : 1;
  Register Src = MI.getOperand(FirstOpnd).getReg();
  Register LSBit = MI.getOperand(FirstOpnd + 1).getReg();
  Register Width = MI.getOperand(FirstOpnd + 2).getReg();
  // For uniform bit field extract there are 4 available instructions, but
  // LSBit(field offset) and Width(size of bitfield) need to be packed in S32,
  // field offset in low and size in high 16 bits.

  // Src1 Hi16|Lo16 = Size|FieldOffset
  auto Mask = B.buildConstant(SgprRB_S32, maskTrailingOnes<unsigned>(6));
  auto FieldOffset = B.buildAnd(SgprRB_S32, LSBit, Mask);
  auto Size = B.buildShl(SgprRB_S32, Width, B.buildConstant(SgprRB_S32, 16));
  auto Src1 = B.buildOr(SgprRB_S32, FieldOffset, Size);
  unsigned Opc32 = Signed ? AMDGPU::S_BFE_I32 : AMDGPU::S_BFE_U32;
  unsigned Opc64 = Signed ? AMDGPU::S_BFE_I64 : AMDGPU::S_BFE_U64;
  unsigned Opc = Ty == S32 ? Opc32 : Opc64;

  // Select machine instruction, because of reg class constraining, insert
  // copies from reg class to reg bank.
  auto S_BFE = B.buildInstr(Opc, {{SgprRB, Ty}},
                            {B.buildCopy(Ty, Src), B.buildCopy(S32, Src1)});
  if (!constrainSelectedInstRegOperands(*S_BFE, *ST.getInstrInfo(),
                                        *ST.getRegisterInfo(), RBI))
    llvm_unreachable("failed to constrain BFE");

  B.buildCopy(DstReg, S_BFE->getOperand(0).getReg());
  MI.eraseFromParent();
}

void RegBankLegalizeHelper::lowerSplitTo32(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst);
  assert(DstTy == V4S16 || DstTy == V2S32 || DstTy == S64);
  LLT Ty = DstTy == V4S16 ? V2S16 : S32;
  auto Op1 = B.buildUnmerge({VgprRB, Ty}, MI.getOperand(1).getReg());
  auto Op2 = B.buildUnmerge({VgprRB, Ty}, MI.getOperand(2).getReg());
  unsigned Opc = MI.getOpcode();
  auto Flags = MI.getFlags();
  auto Lo =
      B.buildInstr(Opc, {{VgprRB, Ty}}, {Op1.getReg(0), Op2.getReg(0)}, Flags);
  auto Hi =
      B.buildInstr(Opc, {{VgprRB, Ty}}, {Op1.getReg(1), Op2.getReg(1)}, Flags);
  B.buildMergeLikeInstr(Dst, {Lo, Hi});
  MI.eraseFromParent();
}

void RegBankLegalizeHelper::lower(MachineInstr &MI,
                                  const RegBankLLTMapping &Mapping,
                                  SmallSet<Register, 4> &WaterfallSgprs) {

  switch (Mapping.LoweringMethod) {
  case DoNotLower:
    return;
  case VccExtToSel: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    Register Src = MI.getOperand(1).getReg();
    unsigned Opc = MI.getOpcode();
    if (Ty == S32 || Ty == S16) {
      auto True = B.buildConstant({VgprRB, Ty}, Opc == G_SEXT ? -1 : 1);
      auto False = B.buildConstant({VgprRB, Ty}, 0);
      B.buildSelect(MI.getOperand(0).getReg(), Src, True, False);
    }
    if (Ty == S64) {
      auto True = B.buildConstant({VgprRB, S32}, Opc == G_SEXT ? -1 : 1);
      auto False = B.buildConstant({VgprRB, S32}, 0);
      auto Sel = B.buildSelect({VgprRB, S32}, Src, True, False);
      B.buildMergeValues(
          MI.getOperand(0).getReg(),
          {Sel.getReg(0), Opc == G_SEXT ? Sel.getReg(0) : False.getReg(0)});
    }
    MI.eraseFromParent();
    return;
  }
  case UniExtToSel: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    auto True = B.buildConstant({SgprRB, Ty},
                                MI.getOpcode() == AMDGPU::G_SEXT ? -1 : 1);
    auto False = B.buildConstant({SgprRB, Ty}, 0);
    // Input to G_{Z|S}EXT is 'Legalizer legal' S1. Most common case is compare.
    // We are making select here. S1 cond was already 'any-extended to S32' +
    // 'AND with 1 to clean high bits' by Sgpr32AExtBoolInReg.
    B.buildSelect(MI.getOperand(0).getReg(), MI.getOperand(1).getReg(), True,
                  False);
    MI.eraseFromParent();
    return;
  }
  case Ext32To64: {
    const RegisterBank *RB = MRI.getRegBank(MI.getOperand(0).getReg());
    MachineInstrBuilder Hi;

    if (MI.getOpcode() == AMDGPU::G_ZEXT) {
      Hi = B.buildConstant({RB, S32}, 0);
    } else {
      // Replicate sign bit from 32-bit extended part.
      auto ShiftAmt = B.buildConstant({RB, S32}, 31);
      Hi = B.buildAShr({RB, S32}, MI.getOperand(1).getReg(), ShiftAmt);
    }

    B.buildMergeLikeInstr(MI.getOperand(0).getReg(),
                          {MI.getOperand(1).getReg(), Hi});
    MI.eraseFromParent();
    return;
  }
  case UniCstExt: {
    uint64_t ConstVal = MI.getOperand(1).getCImm()->getZExtValue();
    B.buildConstant(MI.getOperand(0).getReg(), ConstVal);

    MI.eraseFromParent();
    return;
  }
  case VgprToVccCopy: {
    Register Src = MI.getOperand(1).getReg();
    LLT Ty = MRI.getType(Src);
    // Take lowest bit from each lane and put it in lane mask.
    // Lowering via compare, but we need to clean high bits first as compare
    // compares all bits in register.
    Register BoolSrc = MRI.createVirtualRegister({VgprRB, Ty});
    if (Ty == S64) {
      auto Src64 = B.buildUnmerge({VgprRB, Ty}, Src);
      auto One = B.buildConstant(VgprRB_S32, 1);
      auto AndLo = B.buildAnd(VgprRB_S32, Src64.getReg(0), One);
      auto Zero = B.buildConstant(VgprRB_S32, 0);
      auto AndHi = B.buildAnd(VgprRB_S32, Src64.getReg(1), Zero);
      B.buildMergeLikeInstr(BoolSrc, {AndLo, AndHi});
    } else {
      assert(Ty == S32 || Ty == S16);
      auto One = B.buildConstant({VgprRB, Ty}, 1);
      B.buildAnd(BoolSrc, Src, One);
    }
    auto Zero = B.buildConstant({VgprRB, Ty}, 0);
    B.buildICmp(CmpInst::ICMP_NE, MI.getOperand(0).getReg(), BoolSrc, Zero);
    MI.eraseFromParent();
    return;
  }
  case V_BFE:
    return lowerV_BFE(MI);
  case S_BFE:
    return lowerS_BFE(MI);
  case SplitTo32:
    return lowerSplitTo32(MI);
  case SplitLoad: {
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    unsigned Size = DstTy.getSizeInBits();
    // Even split to 128-bit loads
    if (Size > 128) {
      LLT B128;
      if (DstTy.isVector()) {
        LLT EltTy = DstTy.getElementType();
        B128 = LLT::fixed_vector(128 / EltTy.getSizeInBits(), EltTy);
      } else {
        B128 = LLT::scalar(128);
      }
      if (Size / 128 == 2)
        splitLoad(MI, {B128, B128});
      else if (Size / 128 == 4)
        splitLoad(MI, {B128, B128, B128, B128});
      else {
        LLVM_DEBUG(dbgs() << "MI: "; MI.dump(););
        llvm_unreachable("SplitLoad type not supported for MI");
      }
    }
    // 64 and 32 bit load
    else if (DstTy == S96)
      splitLoad(MI, {S64, S32}, S32);
    else if (DstTy == V3S32)
      splitLoad(MI, {V2S32, S32}, S32);
    else if (DstTy == V6S16)
      splitLoad(MI, {V4S16, V2S16}, V2S16);
    else {
      LLVM_DEBUG(dbgs() << "MI: "; MI.dump(););
      llvm_unreachable("SplitLoad type not supported for MI");
    }
    break;
  }
  case WidenLoad: {
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    if (DstTy == S96)
      widenLoad(MI, S128);
    else if (DstTy == V3S32)
      widenLoad(MI, V4S32, S32);
    else if (DstTy == V6S16)
      widenLoad(MI, V8S16, V2S16);
    else {
      LLVM_DEBUG(dbgs() << "MI: "; MI.dump(););
      llvm_unreachable("WidenLoad type not supported for MI");
    }
    break;
  }
  }

  // TODO: executeInWaterfallLoop(... WaterfallSgprs)
}

LLT RegBankLegalizeHelper::getTyFromID(RegBankLLTMappingApplyID ID) {
  switch (ID) {
  case Vcc:
  case UniInVcc:
    return LLT::scalar(1);
  case Sgpr16:
  case Vgpr16:
    return LLT::scalar(16);
  case Sgpr32:
  case Sgpr32Trunc:
  case Sgpr32AExt:
  case Sgpr32AExtBoolInReg:
  case Sgpr32SExt:
  case UniInVgprS32:
  case Vgpr32:
    return LLT::scalar(32);
  case Sgpr64:
  case Vgpr64:
    return LLT::scalar(64);
  case VgprP0:
    return LLT::pointer(0, 64);
  case SgprP1:
  case VgprP1:
    return LLT::pointer(1, 64);
  case SgprP3:
  case VgprP3:
    return LLT::pointer(3, 32);
  case SgprP4:
  case VgprP4:
    return LLT::pointer(4, 64);
  case SgprP5:
  case VgprP5:
    return LLT::pointer(5, 32);
  case SgprV4S32:
  case VgprV4S32:
  case UniInVgprV4S32:
    return LLT::fixed_vector(4, 32);
  default:
    return LLT();
  }
}

LLT RegBankLegalizeHelper::getBTyFromID(RegBankLLTMappingApplyID ID, LLT Ty) {
  switch (ID) {
  case SgprB32:
  case VgprB32:
  case UniInVgprB32:
    if (Ty == LLT::scalar(32) || Ty == LLT::fixed_vector(2, 16) ||
        Ty == LLT::pointer(3, 32) || Ty == LLT::pointer(5, 32) ||
        Ty == LLT::pointer(6, 32))
      return Ty;
    return LLT();
  case SgprB64:
  case VgprB64:
  case UniInVgprB64:
    if (Ty == LLT::scalar(64) || Ty == LLT::fixed_vector(2, 32) ||
        Ty == LLT::fixed_vector(4, 16) || Ty == LLT::pointer(0, 64) ||
        Ty == LLT::pointer(1, 64) || Ty == LLT::pointer(4, 64))
      return Ty;
    return LLT();
  case SgprB96:
  case VgprB96:
  case UniInVgprB96:
    if (Ty == LLT::scalar(96) || Ty == LLT::fixed_vector(3, 32) ||
        Ty == LLT::fixed_vector(6, 16))
      return Ty;
    return LLT();
  case SgprB128:
  case VgprB128:
  case UniInVgprB128:
    if (Ty == LLT::scalar(128) || Ty == LLT::fixed_vector(4, 32) ||
        Ty == LLT::fixed_vector(2, 64))
      return Ty;
    return LLT();
  case SgprB256:
  case VgprB256:
  case UniInVgprB256:
    if (Ty == LLT::scalar(256) || Ty == LLT::fixed_vector(8, 32) ||
        Ty == LLT::fixed_vector(4, 64) || Ty == LLT::fixed_vector(16, 16))
      return Ty;
    return LLT();
  case SgprB512:
  case VgprB512:
  case UniInVgprB512:
    if (Ty == LLT::scalar(512) || Ty == LLT::fixed_vector(16, 32) ||
        Ty == LLT::fixed_vector(8, 64))
      return Ty;
    return LLT();
  default:
    return LLT();
  }
}

const RegisterBank *
RegBankLegalizeHelper::getRegBankFromID(RegBankLLTMappingApplyID ID) {
  switch (ID) {
  case Vcc:
    return VccRB;
  case Sgpr16:
  case Sgpr32:
  case Sgpr64:
  case SgprP1:
  case SgprP3:
  case SgprP4:
  case SgprP5:
  case SgprV4S32:
  case SgprB32:
  case SgprB64:
  case SgprB96:
  case SgprB128:
  case SgprB256:
  case SgprB512:
  case UniInVcc:
  case UniInVgprS32:
  case UniInVgprV4S32:
  case UniInVgprB32:
  case UniInVgprB64:
  case UniInVgprB96:
  case UniInVgprB128:
  case UniInVgprB256:
  case UniInVgprB512:
  case Sgpr32Trunc:
  case Sgpr32AExt:
  case Sgpr32AExtBoolInReg:
  case Sgpr32SExt:
    return SgprRB;
  case Vgpr16:
  case Vgpr32:
  case Vgpr64:
  case VgprP0:
  case VgprP1:
  case VgprP3:
  case VgprP4:
  case VgprP5:
  case VgprV4S32:
  case VgprB32:
  case VgprB64:
  case VgprB96:
  case VgprB128:
  case VgprB256:
  case VgprB512:
    return VgprRB;
  default:
    return nullptr;
  }
}

void RegBankLegalizeHelper::applyMappingDst(
    MachineInstr &MI, unsigned &OpIdx,
    const SmallVectorImpl<RegBankLLTMappingApplyID> &MethodIDs) {
  // Defs start from operand 0
  for (; OpIdx < MethodIDs.size(); ++OpIdx) {
    if (MethodIDs[OpIdx] == None)
      continue;
    MachineOperand &Op = MI.getOperand(OpIdx);
    Register Reg = Op.getReg();
    LLT Ty = MRI.getType(Reg);
    [[maybe_unused]] const RegisterBank *RB = MRI.getRegBank(Reg);

    switch (MethodIDs[OpIdx]) {
    // vcc, sgpr and vgpr scalars, pointers and vectors
    case Vcc:
    case Sgpr16:
    case Sgpr32:
    case Sgpr64:
    case SgprP1:
    case SgprP3:
    case SgprP4:
    case SgprP5:
    case SgprV4S32:
    case Vgpr16:
    case Vgpr32:
    case Vgpr64:
    case VgprP0:
    case VgprP1:
    case VgprP3:
    case VgprP4:
    case VgprP5:
    case VgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[OpIdx]));
      assert(RB == getRegBankFromID(MethodIDs[OpIdx]));
      break;
    }
    // sgpr and vgpr B-types
    case SgprB32:
    case SgprB64:
    case SgprB96:
    case SgprB128:
    case SgprB256:
    case SgprB512:
    case VgprB32:
    case VgprB64:
    case VgprB96:
    case VgprB128:
    case VgprB256:
    case VgprB512: {
      assert(Ty == getBTyFromID(MethodIDs[OpIdx], Ty));
      assert(RB == getRegBankFromID(MethodIDs[OpIdx]));
      break;
    }
    // uniform in vcc/vgpr: scalars, vectors and B-types
    case UniInVcc: {
      assert(Ty == S1);
      assert(RB == SgprRB);
      Register NewDst = MRI.createVirtualRegister(VccRB_S1);
      Op.setReg(NewDst);
      auto CopyS32_Vcc =
          B.buildInstr(AMDGPU::G_AMDGPU_COPY_SCC_VCC, {SgprRB_S32}, {NewDst});
      B.buildTrunc(Reg, CopyS32_Vcc);
      break;
    }
    case UniInVgprS32:
    case UniInVgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[OpIdx]));
      assert(RB == SgprRB);
      Register NewVgprDst = MRI.createVirtualRegister({VgprRB, Ty});
      Op.setReg(NewVgprDst);
      buildReadAnyLane(B, Reg, NewVgprDst, RBI);
      break;
    }
    case UniInVgprB32:
    case UniInVgprB64:
    case UniInVgprB96:
    case UniInVgprB128:
    case UniInVgprB256:
    case UniInVgprB512: {
      assert(Ty == getBTyFromID(MethodIDs[OpIdx], Ty));
      assert(RB == SgprRB);
      Register NewVgprDst = MRI.createVirtualRegister({VgprRB, Ty});
      Op.setReg(NewVgprDst);
      AMDGPU::buildReadAnyLane(B, Reg, NewVgprDst, RBI);
      break;
    }
    // sgpr trunc
    case Sgpr32Trunc: {
      assert(Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      Register NewDst = MRI.createVirtualRegister(SgprRB_S32);
      Op.setReg(NewDst);
      B.buildTrunc(Reg, NewDst);
      break;
    }
    case InvalidMapping: {
      LLVM_DEBUG(dbgs() << "Instruction with Invalid mapping: "; MI.dump(););
      llvm_unreachable("missing fast rule for MI");
    }
    default:
      llvm_unreachable("ID not supported");
    }
  }
}

void RegBankLegalizeHelper::applyMappingSrc(
    MachineInstr &MI, unsigned &OpIdx,
    const SmallVectorImpl<RegBankLLTMappingApplyID> &MethodIDs,
    SmallSet<Register, 4> &SgprWaterfallOperandRegs) {
  for (unsigned i = 0; i < MethodIDs.size(); ++OpIdx, ++i) {
    if (MethodIDs[i] == None || MethodIDs[i] == IntrId || MethodIDs[i] == Imm)
      continue;

    MachineOperand &Op = MI.getOperand(OpIdx);
    Register Reg = Op.getReg();
    LLT Ty = MRI.getType(Reg);
    const RegisterBank *RB = MRI.getRegBank(Reg);

    switch (MethodIDs[i]) {
    case Vcc: {
      assert(Ty == S1);
      assert(RB == VccRB || RB == SgprRB);
      if (RB == SgprRB) {
        auto Aext = B.buildAnyExt(SgprRB_S32, Reg);
        auto CopyVcc_Scc =
            B.buildInstr(AMDGPU::G_AMDGPU_COPY_VCC_SCC, {VccRB_S1}, {Aext});
        Op.setReg(CopyVcc_Scc.getReg(0));
      }
      break;
    }
    // sgpr scalars, pointers and vectors
    case Sgpr16:
    case Sgpr32:
    case Sgpr64:
    case SgprP1:
    case SgprP3:
    case SgprP4:
    case SgprP5:
    case SgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[i]));
      assert(RB == getRegBankFromID(MethodIDs[i]));
      break;
    }
    // sgpr B-types
    case SgprB32:
    case SgprB64:
    case SgprB96:
    case SgprB128:
    case SgprB256:
    case SgprB512: {
      assert(Ty == getBTyFromID(MethodIDs[i], Ty));
      assert(RB == getRegBankFromID(MethodIDs[i]));
      break;
    }
    // vgpr scalars, pointers and vectors
    case Vgpr16:
    case Vgpr32:
    case Vgpr64:
    case VgprP0:
    case VgprP1:
    case VgprP3:
    case VgprP4:
    case VgprP5:
    case VgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[i]));
      if (RB != VgprRB) {
        auto CopyToVgpr = B.buildCopy({VgprRB, Ty}, Reg);
        Op.setReg(CopyToVgpr.getReg(0));
      }
      break;
    }
    // vgpr B-types
    case VgprB32:
    case VgprB64:
    case VgprB96:
    case VgprB128:
    case VgprB256:
    case VgprB512: {
      assert(Ty == getBTyFromID(MethodIDs[i], Ty));
      if (RB != VgprRB) {
        auto CopyToVgpr = B.buildCopy({VgprRB, Ty}, Reg);
        Op.setReg(CopyToVgpr.getReg(0));
      }
      break;
    }
    // sgpr and vgpr scalars with extend
    case Sgpr32AExt: {
      // Note: this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Aext = B.buildAnyExt(SgprRB_S32, Reg);
      Op.setReg(Aext.getReg(0));
      break;
    }
    case Sgpr32AExtBoolInReg: {
      // Note: this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() == 1);
      assert(RB == SgprRB);
      auto Aext = B.buildAnyExt(SgprRB_S32, Reg);
      // Zext SgprS1 is not legal, this instruction is most of times meant to be
      // combined away in RB combiner, so do not make AND with 1.
      auto Cst1 = B.buildConstant(SgprRB_S32, 1);
      auto BoolInReg = B.buildAnd(SgprRB_S32, Aext, Cst1);
      Op.setReg(BoolInReg.getReg(0));
      break;
    }
    case Sgpr32SExt: {
      assert(1 < Ty.getSizeInBits() && Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Sext = B.buildSExt(SgprRB_S32, Reg);
      Op.setReg(Sext.getReg(0));
      break;
    }
    default:
      llvm_unreachable("ID not supported");
    }
  }
}

void RegBankLegalizeHelper::applyMappingPHI(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);

  if (Ty == LLT::scalar(1) && MUI.isUniform(Dst)) {
    B.setInsertPt(*MI.getParent(), MI.getParent()->getFirstNonPHI());

    Register NewDst = MRI.createVirtualRegister(SgprRB_S32);
    MI.getOperand(0).setReg(NewDst);
    B.buildTrunc(Dst, NewDst);

    for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
      Register UseReg = MI.getOperand(i).getReg();

      auto DefMI = MRI.getVRegDef(UseReg)->getIterator();
      MachineBasicBlock *DefMBB = DefMI->getParent();

      B.setInsertPt(*DefMBB, DefMBB->SkipPHIsAndLabels(std::next(DefMI)));

      auto NewUse = B.buildAnyExt(SgprRB_S32, UseReg);
      MI.getOperand(i).setReg(NewUse.getReg(0));
    }

    return;
  }

  // ALL divergent i1 phis should be already lowered and inst-selected into PHI
  // with sgpr reg class and S1 LLT.
  // Note: this includes divergent phis that don't require lowering.
  if (Ty == LLT::scalar(1) && MUI.isDivergent(Dst)) {
    LLVM_DEBUG(dbgs() << "Divergent S1 G_PHI: "; MI.dump(););
    llvm_unreachable("Make sure to run AMDGPUGlobalISelDivergenceLowering "
                     "before RegBankLegalize to lower lane mask(vcc) phis");
  }

  // We accept all types that can fit in some register class.
  // Uniform G_PHIs have all sgpr registers.
  // Divergent G_PHIs have vgpr dst but inputs can be sgpr or vgpr.
  if (Ty == LLT::scalar(32) || Ty == LLT::pointer(1, 64) ||
      Ty == LLT::pointer(4, 64)) {
    return;
  }

  LLVM_DEBUG(dbgs() << "G_PHI not handled: "; MI.dump(););
  llvm_unreachable("type not supported");
}

[[maybe_unused]] static bool verifyRegBankOnOperands(MachineInstr &MI,
                                                     const RegisterBank *RB,
                                                     MachineRegisterInfo &MRI,
                                                     unsigned StartOpIdx,
                                                     unsigned EndOpIdx) {
  for (unsigned i = StartOpIdx; i <= EndOpIdx; ++i) {
    if (MRI.getRegBankOrNull(MI.getOperand(i).getReg()) != RB)
      return false;
  }
  return true;
}

void RegBankLegalizeHelper::applyMappingTrivial(MachineInstr &MI) {
  const RegisterBank *RB = MRI.getRegBank(MI.getOperand(0).getReg());
  // Put RB on all registers
  unsigned NumDefs = MI.getNumDefs();
  unsigned NumOperands = MI.getNumOperands();

  assert(verifyRegBankOnOperands(MI, RB, MRI, 0, NumDefs - 1));
  if (RB == SgprRB)
    assert(verifyRegBankOnOperands(MI, RB, MRI, NumDefs, NumOperands - 1));

  if (RB == VgprRB) {
    B.setInstr(MI);
    for (unsigned i = NumDefs; i < NumOperands; ++i) {
      Register Reg = MI.getOperand(i).getReg();
      if (MRI.getRegBank(Reg) != RB) {
        auto Copy = B.buildCopy({VgprRB, MRI.getType(Reg)}, Reg);
        MI.getOperand(i).setReg(Copy.getReg(0));
      }
    }
  }
}
