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
      MUI(MUI), RBI(RBI), RBLRules(RBLRules), IsWave32(ST.isWave32()),
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

bool RegBankLegalizeHelper::executeInWaterfallLoop(
    MachineIRBuilder &B, iterator_range<MachineBasicBlock::iterator> Range,
    SmallSet<Register, 4> &SGPROperandRegs) {
  // Track use registers which have already been expanded with a readfirstlane
  // sequence. This may have multiple uses if moving a sequence.
  DenseMap<Register, Register> WaterfalledRegMap;

  MachineBasicBlock &MBB = B.getMBB();
  MachineFunction &MF = B.getMF();

  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const TargetRegisterClass *WaveRC = TRI->getWaveMaskRegClass();
  unsigned MovExecOpc, MovExecTermOpc, XorTermOpc, AndSaveExecOpc, ExecReg;
  if (IsWave32) {
    MovExecOpc = AMDGPU::S_MOV_B32;
    MovExecTermOpc = AMDGPU::S_MOV_B32_term;
    XorTermOpc = AMDGPU::S_XOR_B32_term;
    AndSaveExecOpc = AMDGPU::S_AND_SAVEEXEC_B32;
    ExecReg = AMDGPU::EXEC_LO;
  } else {
    MovExecOpc = AMDGPU::S_MOV_B64;
    MovExecTermOpc = AMDGPU::S_MOV_B64_term;
    XorTermOpc = AMDGPU::S_XOR_B64_term;
    AndSaveExecOpc = AMDGPU::S_AND_SAVEEXEC_B64;
    ExecReg = AMDGPU::EXEC;
  }

#ifndef NDEBUG
  const int OrigRangeSize = std::distance(Range.begin(), Range.end());
#endif

  MachineRegisterInfo &MRI = *B.getMRI();
  Register SaveExecReg = MRI.createVirtualRegister(WaveRC);
  Register InitSaveExecReg = MRI.createVirtualRegister(WaveRC);

  // Don't bother using generic instructions/registers for the exec mask.
  B.buildInstr(TargetOpcode::IMPLICIT_DEF).addDef(InitSaveExecReg);

  Register SavedExec = MRI.createVirtualRegister(WaveRC);

  // To insert the loop we need to split the block. Move everything before
  // this point to a new block, and insert a new empty block before this
  // instruction.
  MachineBasicBlock *LoopBB = MF.CreateMachineBasicBlock();
  MachineBasicBlock *BodyBB = MF.CreateMachineBasicBlock();
  MachineBasicBlock *RestoreExecBB = MF.CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF.CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;
  MF.insert(MBBI, LoopBB);
  MF.insert(MBBI, BodyBB);
  MF.insert(MBBI, RestoreExecBB);
  MF.insert(MBBI, RemainderBB);

  LoopBB->addSuccessor(BodyBB);
  BodyBB->addSuccessor(RestoreExecBB);
  BodyBB->addSuccessor(LoopBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessorsAndUpdatePHIs(&MBB);
  RemainderBB->splice(RemainderBB->begin(), &MBB, Range.end(), MBB.end());

  MBB.addSuccessor(LoopBB);
  RestoreExecBB->addSuccessor(RemainderBB);

  B.setInsertPt(*LoopBB, LoopBB->end());

  // +-MBB:------------+
  // | ...             |
  // | %0 = G_INST_1   |
  // | %Dst = MI %Vgpr |
  // | %1 = G_INST_2   |
  // | ...             |
  // +-----------------+
  // ->
  // +-MBB-------------------------------+
  // | ...                               |
  // | %0 = G_INST_1                     |
  // | %SaveExecReg = S_MOV_B32 $exec_lo |
  // +----------------|------------------+
  //                  |                         /------------------------------|
  //                  V                        V                               |
  // +-LoopBB---------------------------------------------------------------+  |
  // | %CurrentLaneReg:sgpr(s32) = READFIRSTLANE %Vgpr                      |  |
  // |   instead of executing for each lane, see if other lanes had         |  |
  // |   same value for %Vgpr and execute for them also.                    |  |
  // | %CondReg:vcc(s1) = G_ICMP eq %CurrentLaneReg, %Vgpr                  |  |
  // | %CondRegLM:sreg_32 = ballot %CondReg // copy vcc to sreg32 lane mask |  |
  // | %SavedExec = S_AND_SAVEEXEC_B32 %CondRegLM                           |  |
  // |   exec is active for lanes with the same "CurrentLane value" in Vgpr |  |
  // +----------------|-----------------------------------------------------+  |
  //                  V                                                        |
  // +-BodyBB------------------------------------------------------------+     |
  // | %Dst = MI %CurrentLaneReg:sgpr(s32)                               |     |
  // |   executed only for active lanes and written to Dst               |     |
  // | $exec = S_XOR_B32 $exec, %SavedExec                               |     |
  // |   set active lanes to 0 in SavedExec, lanes that did not write to |     |
  // |   Dst yet, and set this as new exec (for READFIRSTLANE and ICMP)  |     |
  // | SI_WATERFALL_LOOP LoopBB                                          |-----|
  // +----------------|--------------------------------------------------+
  //                  V
  // +-RestoreExecBB--------------------------+
  // | $exec_lo = S_MOV_B32_term %SaveExecReg |
  // +----------------|-----------------------+
  //                  V
  // +-RemainderBB:----------------------+
  // | %1 = G_INST_2                     |
  // | ...                               |
  // +---------------------------------- +

  // Move the instruction into the loop body. Note we moved everything after
  // Range.end() already into a new block, so Range.end() is no longer valid.
  BodyBB->splice(BodyBB->end(), &MBB, Range.begin(), MBB.end());

  // Figure out the iterator range after splicing the instructions.
  MachineBasicBlock::iterator NewBegin = Range.begin()->getIterator();
  auto NewEnd = BodyBB->end();
  assert(std::distance(NewBegin, NewEnd) == OrigRangeSize);

  B.setMBB(*LoopBB);
  Register CondReg;

  for (MachineInstr &MI : make_range(NewBegin, NewEnd)) {
    for (MachineOperand &Op : MI.all_uses()) {
      Register OldReg = Op.getReg();
      if (!SGPROperandRegs.count(OldReg))
        continue;

      // See if we already processed this register in another instruction in
      // the sequence.
      auto OldVal = WaterfalledRegMap.find(OldReg);
      if (OldVal != WaterfalledRegMap.end()) {
        Op.setReg(OldVal->second);
        continue;
      }

      Register OpReg = Op.getReg();
      LLT OpTy = MRI.getType(OpReg);

      // TODO: support for agpr
      assert(MRI.getRegBank(OpReg) == VgprRB);
      Register CurrentLaneReg = MRI.createVirtualRegister({SgprRB, OpTy});
      buildReadFirstLane(B, CurrentLaneReg, OpReg, RBI);

      // Build the comparison(s), CurrentLaneReg == OpReg.
      unsigned OpSize = OpTy.getSizeInBits();
      unsigned PartSize = (OpSize % 64 == 0) ? 64 : 32;
      LLT PartTy = LLT::scalar(PartSize);
      unsigned NumParts = OpSize / PartSize;
      SmallVector<Register, 8> OpParts;
      SmallVector<Register, 8> CurrentLaneParts;

      if (NumParts == 1) {
        OpParts.push_back(OpReg);
        CurrentLaneParts.push_back(CurrentLaneReg);
      } else {
        auto UnmergeOp = B.buildUnmerge({VgprRB, PartTy}, OpReg);
        auto UnmergeCurrLane = B.buildUnmerge({SgprRB, PartTy}, CurrentLaneReg);
        for (unsigned i = 0; i < NumParts; ++i) {
          OpParts.push_back(UnmergeOp.getReg(i));
          CurrentLaneParts.push_back(UnmergeCurrLane.getReg(i));
        }
      }

      for (unsigned i = 0; i < NumParts; ++i) {
        Register CmpReg = MRI.createVirtualRegister(VccRB_S1);
        B.buildICmp(CmpInst::ICMP_EQ, CmpReg, CurrentLaneParts[i], OpParts[i]);

        if (!CondReg)
          CondReg = CmpReg;
        else
          CondReg = B.buildAnd(VccRB_S1, CondReg, CmpReg).getReg(0);
      }

      Op.setReg(CurrentLaneReg);

      // Make sure we don't re-process this register again.
      WaterfalledRegMap.insert(std::pair(OldReg, Op.getReg()));
    }
  }

  // Copy vcc to sgpr32/64, ballot becomes a no-op during instruction selection.
  Register CondRegLM =
      MRI.createVirtualRegister({WaveRC, LLT::scalar(IsWave32 ? 32 : 64)});
  B.buildIntrinsic(Intrinsic::amdgcn_ballot, CondRegLM).addReg(CondReg);

  // Update EXEC, save the original EXEC value to SavedExec.
  B.buildInstr(AndSaveExecOpc)
      .addDef(SavedExec)
      .addReg(CondRegLM, RegState::Kill);
  MRI.setSimpleHint(SavedExec, CondRegLM);

  B.setInsertPt(*BodyBB, BodyBB->end());

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  B.buildInstr(XorTermOpc).addDef(ExecReg).addReg(ExecReg).addReg(SavedExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  B.buildInstr(AMDGPU::SI_WATERFALL_LOOP).addMBB(LoopBB);

  // Save the EXEC mask before the loop.
  B.setInsertPt(MBB, MBB.end());
  B.buildInstr(MovExecOpc).addDef(SaveExecReg).addReg(ExecReg);

  // Restore the EXEC mask after the loop.
  B.setInsertPt(*RestoreExecBB, RestoreExecBB->begin());
  B.buildInstr(MovExecTermOpc).addDef(ExecReg).addReg(SaveExecReg);

  // Set the insert point after the original instruction, so any new
  // instructions will be in the remainder.
  B.setInsertPt(*RemainderBB, RemainderBB->begin());

  return true;
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
      BasePlusOffset =
          B.buildObjectPtrOffset({PtrRB, PtrTy}, Base, Offset).getReg(0);
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

void RegBankLegalizeHelper::widenMMOToS32(GAnyLoad &MI) const {
  Register Dst = MI.getDstReg();
  Register Ptr = MI.getPointerReg();
  MachineMemOperand &MMO = MI.getMMO();
  unsigned MemSize = 8 * MMO.getSize().getValue();

  MachineMemOperand *WideMMO = B.getMF().getMachineMemOperand(&MMO, 0, S32);

  if (MI.getOpcode() == G_LOAD) {
    B.buildLoad(Dst, Ptr, *WideMMO);
  } else {
    auto Load = B.buildLoad(SgprRB_S32, Ptr, *WideMMO);

    if (MI.getOpcode() == G_ZEXTLOAD) {
      APInt Mask = APInt::getLowBitsSet(S32.getSizeInBits(), MemSize);
      auto MaskCst = B.buildConstant(SgprRB_S32, Mask);
      B.buildAnd(Dst, Load, MaskCst);
    } else {
      assert(MI.getOpcode() == G_SEXTLOAD);
      B.buildSExtInReg(Dst, Load, MemSize);
    }
  }

  MI.eraseFromParent();
}

void RegBankLegalizeHelper::lowerVccExtToSel(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  Register Src = MI.getOperand(1).getReg();
  unsigned Opc = MI.getOpcode();
  int TrueExtCst = Opc == G_SEXT ? -1 : 1;
  if (Ty == S32 || Ty == S16) {
    auto True = B.buildConstant({VgprRB, Ty}, TrueExtCst);
    auto False = B.buildConstant({VgprRB, Ty}, 0);
    B.buildSelect(Dst, Src, True, False);
  } else if (Ty == S64) {
    auto True = B.buildConstant({VgprRB_S32}, TrueExtCst);
    auto False = B.buildConstant({VgprRB_S32}, 0);
    auto Lo = B.buildSelect({VgprRB_S32}, Src, True, False);
    MachineInstrBuilder Hi;
    switch (Opc) {
    case G_SEXT:
      Hi = Lo;
      break;
    case G_ZEXT:
      Hi = False;
      break;
    case G_ANYEXT:
      Hi = B.buildUndef({VgprRB_S32});
      break;
    default:
      llvm_unreachable("Opcode not supported");
    }

    B.buildMergeValues(Dst, {Lo.getReg(0), Hi.getReg(0)});
  } else {
    llvm_unreachable("Type not supported");
  }

  MI.eraseFromParent();
}

std::pair<Register, Register> RegBankLegalizeHelper::unpackZExt(Register Reg) {
  auto PackedS32 = B.buildBitcast(SgprRB_S32, Reg);
  auto Mask = B.buildConstant(SgprRB_S32, 0x0000ffff);
  auto Lo = B.buildAnd(SgprRB_S32, PackedS32, Mask);
  auto Hi = B.buildLShr(SgprRB_S32, PackedS32, B.buildConstant(SgprRB_S32, 16));
  return {Lo.getReg(0), Hi.getReg(0)};
}

std::pair<Register, Register> RegBankLegalizeHelper::unpackSExt(Register Reg) {
  auto PackedS32 = B.buildBitcast(SgprRB_S32, Reg);
  auto Lo = B.buildSExtInReg(SgprRB_S32, PackedS32, 16);
  auto Hi = B.buildAShr(SgprRB_S32, PackedS32, B.buildConstant(SgprRB_S32, 16));
  return {Lo.getReg(0), Hi.getReg(0)};
}

std::pair<Register, Register> RegBankLegalizeHelper::unpackAExt(Register Reg) {
  auto PackedS32 = B.buildBitcast(SgprRB_S32, Reg);
  auto Lo = PackedS32;
  auto Hi = B.buildLShr(SgprRB_S32, PackedS32, B.buildConstant(SgprRB_S32, 16));
  return {Lo.getReg(0), Hi.getReg(0)};
}

void RegBankLegalizeHelper::lowerUnpackBitShift(MachineInstr &MI) {
  Register Lo, Hi;
  switch (MI.getOpcode()) {
  case AMDGPU::G_SHL: {
    auto [Val0, Val1] = unpackAExt(MI.getOperand(1).getReg());
    auto [Amt0, Amt1] = unpackAExt(MI.getOperand(2).getReg());
    Lo = B.buildInstr(MI.getOpcode(), {SgprRB_S32}, {Val0, Amt0}).getReg(0);
    Hi = B.buildInstr(MI.getOpcode(), {SgprRB_S32}, {Val1, Amt1}).getReg(0);
    break;
  }
  case AMDGPU::G_LSHR: {
    auto [Val0, Val1] = unpackZExt(MI.getOperand(1).getReg());
    auto [Amt0, Amt1] = unpackZExt(MI.getOperand(2).getReg());
    Lo = B.buildInstr(MI.getOpcode(), {SgprRB_S32}, {Val0, Amt0}).getReg(0);
    Hi = B.buildInstr(MI.getOpcode(), {SgprRB_S32}, {Val1, Amt1}).getReg(0);
    break;
  }
  case AMDGPU::G_ASHR: {
    auto [Val0, Val1] = unpackSExt(MI.getOperand(1).getReg());
    auto [Amt0, Amt1] = unpackSExt(MI.getOperand(2).getReg());
    Lo = B.buildAShr(SgprRB_S32, Val0, Amt0).getReg(0);
    Hi = B.buildAShr(SgprRB_S32, Val1, Amt1).getReg(0);
    break;
  }
  default:
    llvm_unreachable("Unpack lowering not implemented");
  }
  B.buildBuildVectorTrunc(MI.getOperand(0).getReg(), {Lo, Hi});
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

void RegBankLegalizeHelper::lowerSplitTo32Select(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst);
  assert(DstTy == V4S16 || DstTy == V2S32 || DstTy == S64 ||
         (DstTy.isPointer() && DstTy.getSizeInBits() == 64));
  LLT Ty = DstTy == V4S16 ? V2S16 : S32;
  auto Op2 = B.buildUnmerge({VgprRB, Ty}, MI.getOperand(2).getReg());
  auto Op3 = B.buildUnmerge({VgprRB, Ty}, MI.getOperand(3).getReg());
  Register Cond = MI.getOperand(1).getReg();
  auto Flags = MI.getFlags();
  auto Lo =
      B.buildSelect({VgprRB, Ty}, Cond, Op2.getReg(0), Op3.getReg(0), Flags);
  auto Hi =
      B.buildSelect({VgprRB, Ty}, Cond, Op2.getReg(1), Op3.getReg(1), Flags);

  B.buildMergeLikeInstr(Dst, {Lo, Hi});
  MI.eraseFromParent();
}

void RegBankLegalizeHelper::lowerSplitTo32SExtInReg(MachineInstr &MI) {
  auto Op1 = B.buildUnmerge(VgprRB_S32, MI.getOperand(1).getReg());
  int Amt = MI.getOperand(2).getImm();
  Register Lo, Hi;
  // Hi|Lo: s sign bit, ?/x bits changed/not changed by sign-extend
  if (Amt <= 32) {
    auto Freeze = B.buildFreeze(VgprRB_S32, Op1.getReg(0));
    if (Amt == 32) {
      // Hi|Lo: ????????|sxxxxxxx -> ssssssss|sxxxxxxx
      Lo = Freeze.getReg(0);
    } else {
      // Hi|Lo: ????????|???sxxxx -> ssssssss|ssssxxxx
      Lo = B.buildSExtInReg(VgprRB_S32, Freeze, Amt).getReg(0);
    }

    auto SignExtCst = B.buildConstant(SgprRB_S32, 31);
    Hi = B.buildAShr(VgprRB_S32, Lo, SignExtCst).getReg(0);
  } else {
    // Hi|Lo: ?????sxx|xxxxxxxx -> ssssssxx|xxxxxxxx
    Lo = Op1.getReg(0);
    Hi = B.buildSExtInReg(VgprRB_S32, Op1.getReg(1), Amt - 32).getReg(0);
  }

  B.buildMergeLikeInstr(MI.getOperand(0).getReg(), {Lo, Hi});
  MI.eraseFromParent();
}

void RegBankLegalizeHelper::lower(MachineInstr &MI,
                                  const RegBankLLTMapping &Mapping,
                                  SmallSet<Register, 4> &WaterfallSgprs) {

  switch (Mapping.LoweringMethod) {
  case DoNotLower:
    break;
  case VccExtToSel:
    return lowerVccExtToSel(MI);
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
  case UnpackBitShift:
    return lowerUnpackBitShift(MI);
  case Ext32To64: {
    const RegisterBank *RB = MRI.getRegBank(MI.getOperand(0).getReg());
    MachineInstrBuilder Hi;
    switch (MI.getOpcode()) {
    case AMDGPU::G_ZEXT: {
      Hi = B.buildConstant({RB, S32}, 0);
      break;
    }
    case AMDGPU::G_SEXT: {
      // Replicate sign bit from 32-bit extended part.
      auto ShiftAmt = B.buildConstant({RB, S32}, 31);
      Hi = B.buildAShr({RB, S32}, MI.getOperand(1).getReg(), ShiftAmt);
      break;
    }
    case AMDGPU::G_ANYEXT: {
      Hi = B.buildUndef({RB, S32});
      break;
    }
    default:
      llvm_unreachable("Unsuported Opcode in Ext32To64");
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
      auto Src64 = B.buildUnmerge(VgprRB_S32, Src);
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
  case SplitTo32Select:
    return lowerSplitTo32Select(MI);
  case SplitTo32SExtInReg:
    return lowerSplitTo32SExtInReg(MI);
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
  case WidenMMOToS32:
    return widenMMOToS32(cast<GAnyLoad>(MI));
  }

  if (!WaterfallSgprs.empty()) {
    MachineBasicBlock::iterator I = MI.getIterator();
    executeInWaterfallLoop(B, make_range(I, std::next(I)), WaterfallSgprs);
  }
}

LLT RegBankLegalizeHelper::getTyFromID(RegBankLLTMappingApplyID ID) {
  switch (ID) {
  case Vcc:
  case UniInVcc:
    return LLT::scalar(1);
  case Sgpr16:
  case Vgpr16:
  case UniInVgprS16:
    return LLT::scalar(16);
  case Sgpr32:
  case Sgpr32_WF:
  case Sgpr32Trunc:
  case Sgpr32AExt:
  case Sgpr32AExtBoolInReg:
  case Sgpr32SExt:
  case Sgpr32ZExt:
  case UniInVgprS32:
  case Vgpr32:
  case Vgpr32SExt:
  case Vgpr32ZExt:
    return LLT::scalar(32);
  case Sgpr64:
  case Vgpr64:
    return LLT::scalar(64);
  case Sgpr128:
  case Vgpr128:
    return LLT::scalar(128);
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
  case SgprV2S16:
  case VgprV2S16:
  case UniInVgprV2S16:
    return LLT::fixed_vector(2, 16);
  case SgprV2S32:
  case VgprV2S32:
    return LLT::fixed_vector(2, 32);
  case SgprV4S32:
  case SgprV4S32_WF:
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
        isAnyPtr(Ty, 32))
      return Ty;
    return LLT();
  case SgprPtr32:
  case VgprPtr32:
    return isAnyPtr(Ty, 32) ? Ty : LLT();
  case SgprPtr64:
  case VgprPtr64:
    return isAnyPtr(Ty, 64) ? Ty : LLT();
  case SgprPtr128:
  case VgprPtr128:
    return isAnyPtr(Ty, 128) ? Ty : LLT();
  case SgprB64:
  case VgprB64:
  case UniInVgprB64:
    if (Ty == LLT::scalar(64) || Ty == LLT::fixed_vector(2, 32) ||
        Ty == LLT::fixed_vector(4, 16) || isAnyPtr(Ty, 64))
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
        Ty == LLT::fixed_vector(2, 64) || isAnyPtr(Ty, 128))
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
  case Sgpr32_WF:
  case Sgpr64:
  case Sgpr128:
  case SgprP1:
  case SgprP3:
  case SgprP4:
  case SgprP5:
  case SgprPtr32:
  case SgprPtr64:
  case SgprPtr128:
  case SgprV2S16:
  case SgprV2S32:
  case SgprV4S32:
  case SgprV4S32_WF:
  case SgprB32:
  case SgprB64:
  case SgprB96:
  case SgprB128:
  case SgprB256:
  case SgprB512:
  case UniInVcc:
  case UniInVgprS16:
  case UniInVgprS32:
  case UniInVgprV2S16:
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
  case Sgpr32ZExt:
    return SgprRB;
  case Vgpr16:
  case Vgpr32:
  case Vgpr64:
  case Vgpr128:
  case VgprP0:
  case VgprP1:
  case VgprP3:
  case VgprP4:
  case VgprP5:
  case VgprPtr32:
  case VgprPtr64:
  case VgprPtr128:
  case VgprV2S16:
  case VgprV2S32:
  case VgprV4S32:
  case VgprB32:
  case VgprB64:
  case VgprB96:
  case VgprB128:
  case VgprB256:
  case VgprB512:
  case Vgpr32SExt:
  case Vgpr32ZExt:
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
    case Sgpr128:
    case SgprP1:
    case SgprP3:
    case SgprP4:
    case SgprP5:
    case SgprV2S16:
    case SgprV2S32:
    case SgprV4S32:
    case Vgpr16:
    case Vgpr32:
    case Vgpr64:
    case Vgpr128:
    case VgprP0:
    case VgprP1:
    case VgprP3:
    case VgprP4:
    case VgprP5:
    case VgprV2S16:
    case VgprV2S32:
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
    case SgprPtr32:
    case SgprPtr64:
    case SgprPtr128:
    case VgprB32:
    case VgprB64:
    case VgprB96:
    case VgprB128:
    case VgprB256:
    case VgprB512:
    case VgprPtr32:
    case VgprPtr64:
    case VgprPtr128: {
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
    case UniInVgprS16: {
      assert(Ty == getTyFromID(MethodIDs[OpIdx]));
      assert(RB == SgprRB);
      Register NewVgprDstS16 = MRI.createVirtualRegister({VgprRB, S16});
      Register NewVgprDstS32 = MRI.createVirtualRegister({VgprRB, S32});
      Register NewSgprDstS32 = MRI.createVirtualRegister({SgprRB, S32});
      Op.setReg(NewVgprDstS16);
      B.buildAnyExt(NewVgprDstS32, NewVgprDstS16);
      buildReadAnyLane(B, NewSgprDstS32, NewVgprDstS32, RBI);
      B.buildTrunc(Reg, NewSgprDstS32);
      break;
    }
    case UniInVgprS32:
    case UniInVgprV2S16:
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
    case Sgpr128:
    case SgprP1:
    case SgprP3:
    case SgprP4:
    case SgprP5:
    case SgprV2S16:
    case SgprV2S32:
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
    case SgprB512:
    case SgprPtr32:
    case SgprPtr64:
    case SgprPtr128: {
      assert(Ty == getBTyFromID(MethodIDs[i], Ty));
      assert(RB == getRegBankFromID(MethodIDs[i]));
      break;
    }
    // vgpr scalars, pointers and vectors
    case Vgpr16:
    case Vgpr32:
    case Vgpr64:
    case Vgpr128:
    case VgprP0:
    case VgprP1:
    case VgprP3:
    case VgprP4:
    case VgprP5:
    case VgprV2S16:
    case VgprV2S32:
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
    case VgprB512:
    case VgprPtr32:
    case VgprPtr64:
    case VgprPtr128: {
      assert(Ty == getBTyFromID(MethodIDs[i], Ty));
      if (RB != VgprRB) {
        auto CopyToVgpr = B.buildCopy({VgprRB, Ty}, Reg);
        Op.setReg(CopyToVgpr.getReg(0));
      }
      break;
    }
    // sgpr waterfall, scalars and vectors
    case Sgpr32_WF:
    case SgprV4S32_WF: {
      assert(Ty == getTyFromID(MethodIDs[i]));
      if (RB != SgprRB)
        SgprWaterfallOperandRegs.insert(Reg);
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
      // Zext SgprS1 is not legal, make AND with 1 instead. This instruction is
      // most of times meant to be combined away in AMDGPURegBankCombiner.
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
    case Sgpr32ZExt: {
      assert(1 < Ty.getSizeInBits() && Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Zext = B.buildZExt({SgprRB, S32}, Reg);
      Op.setReg(Zext.getReg(0));
      break;
    }
    case Vgpr32SExt: {
      // Note this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() < 32);
      assert(RB == VgprRB);
      auto Sext = B.buildSExt({VgprRB, S32}, Reg);
      Op.setReg(Sext.getReg(0));
      break;
    }
    case Vgpr32ZExt: {
      // Note this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() < 32);
      assert(RB == VgprRB);
      auto Zext = B.buildZExt({VgprRB, S32}, Reg);
      Op.setReg(Zext.getReg(0));
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
