//===- SIPeepholeSDWA.cpp - Peephole optimization for SDWA instructions ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This pass tries to apply several peephole SDWA patterns.
///
/// E.g. original:
///   V_LSHRREV_B32_e32 %vreg0, 16, %vreg1
///   V_ADD_I32_e32 %vreg2, %vreg0, %vreg3
///   V_LSHLREV_B32_e32 %vreg4, 16, %vreg2
///
/// Replace:
///   V_ADD_I32_sdwa %vreg4, %vreg1, %vreg3
///       dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "si-peephole-sdwa"

STATISTIC(NumSDWAPatternsFound, "Number of SDWA patterns found.");
STATISTIC(NumSDWAInstructionsPeepholed,
          "Number of instruction converted to SDWA.");

namespace {

class SDWAOperand;

class SIPeepholeSDWA : public MachineFunctionPass {
public:
  using SDWAOperandsVector = SmallVector<SDWAOperand *, 4>;

private:
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;

  std::unordered_map<MachineInstr *, std::unique_ptr<SDWAOperand>> SDWAOperands;
  std::unordered_map<MachineInstr *, SDWAOperandsVector> PotentialMatches;
  SmallVector<MachineInstr *, 8> ConvertedInstructions;

  Optional<int64_t> foldToImm(const MachineOperand &Op) const;

public:
  static char ID;

  SIPeepholeSDWA() : MachineFunctionPass(ID) {
    initializeSIPeepholeSDWAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void matchSDWAOperands(MachineFunction &MF);
  bool isConvertibleToSDWA(const MachineInstr &MI, const SISubtarget &ST) const;
  bool convertToSDWA(MachineInstr &MI, const SDWAOperandsVector &SDWAOperands);
  void legalizeScalarOperands(MachineInstr &MI, const SISubtarget &ST) const;

  StringRef getPassName() const override { return "SI Peephole SDWA"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class SDWAOperand {
private:
  MachineOperand *Target; // Operand that would be used in converted instruction
  MachineOperand *Replaced; // Operand that would be replace by Target

public:
  SDWAOperand(MachineOperand *TargetOp, MachineOperand *ReplacedOp)
      : Target(TargetOp), Replaced(ReplacedOp) {
    assert(Target->isReg());
    assert(Replaced->isReg());
  }

  virtual ~SDWAOperand() = default;

  virtual MachineInstr *potentialToConvert(const SIInstrInfo *TII) = 0;
  virtual bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) = 0;

  MachineOperand *getTargetOperand() const { return Target; }
  MachineOperand *getReplacedOperand() const { return Replaced; }
  MachineInstr *getParentInst() const { return Target->getParent(); }

  MachineRegisterInfo *getMRI() const {
    return &getParentInst()->getParent()->getParent()->getRegInfo();
  }
};

using namespace AMDGPU::SDWA;

class SDWASrcOperand : public SDWAOperand {
private:
  SdwaSel SrcSel;
  bool Abs;
  bool Neg;
  bool Sext;

public:
  SDWASrcOperand(MachineOperand *TargetOp, MachineOperand *ReplacedOp,
                 SdwaSel SrcSel_ = DWORD, bool Abs_ = false, bool Neg_ = false,
                 bool Sext_ = false)
      : SDWAOperand(TargetOp, ReplacedOp), SrcSel(SrcSel_), Abs(Abs_),
        Neg(Neg_), Sext(Sext_) {}

  MachineInstr *potentialToConvert(const SIInstrInfo *TII) override;
  bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) override;

  SdwaSel getSrcSel() const { return SrcSel; }
  bool getAbs() const { return Abs; }
  bool getNeg() const { return Neg; }
  bool getSext() const { return Sext; }

  uint64_t getSrcMods(const SIInstrInfo *TII,
                      const MachineOperand *SrcOp) const;
};

class SDWADstOperand : public SDWAOperand {
private:
  SdwaSel DstSel;
  DstUnused DstUn;

public:
  SDWADstOperand(MachineOperand *TargetOp, MachineOperand *ReplacedOp,
                 SdwaSel DstSel_ = DWORD, DstUnused DstUn_ = UNUSED_PAD)
      : SDWAOperand(TargetOp, ReplacedOp), DstSel(DstSel_), DstUn(DstUn_) {}

  MachineInstr *potentialToConvert(const SIInstrInfo *TII) override;
  bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) override;

  SdwaSel getDstSel() const { return DstSel; }
  DstUnused getDstUnused() const { return DstUn; }
};

} // end anonymous namespace

INITIALIZE_PASS(SIPeepholeSDWA, DEBUG_TYPE, "SI Peephole SDWA", false, false)

char SIPeepholeSDWA::ID = 0;

char &llvm::SIPeepholeSDWAID = SIPeepholeSDWA::ID;

FunctionPass *llvm::createSIPeepholeSDWAPass() {
  return new SIPeepholeSDWA();
}

#ifndef NDEBUG
static raw_ostream& operator<<(raw_ostream &OS, const SdwaSel &Sel) {
  switch(Sel) {
  case BYTE_0: OS << "BYTE_0"; break;
  case BYTE_1: OS << "BYTE_1"; break;
  case BYTE_2: OS << "BYTE_2"; break;
  case BYTE_3: OS << "BYTE_3"; break;
  case WORD_0: OS << "WORD_0"; break;
  case WORD_1: OS << "WORD_1"; break;
  case DWORD:  OS << "DWORD"; break;
  }
  return OS;
}

static raw_ostream& operator<<(raw_ostream &OS, const DstUnused &Un) {
  switch(Un) {
  case UNUSED_PAD: OS << "UNUSED_PAD"; break;
  case UNUSED_SEXT: OS << "UNUSED_SEXT"; break;
  case UNUSED_PRESERVE: OS << "UNUSED_PRESERVE"; break;
  }
  return OS;
}

static raw_ostream& operator<<(raw_ostream &OS, const SDWASrcOperand &Src) {
  OS << "SDWA src: " << *Src.getTargetOperand()
     << " src_sel:" << Src.getSrcSel()
     << " abs:" << Src.getAbs() << " neg:" << Src.getNeg()
     << " sext:" << Src.getSext() << '\n';
  return OS;
}

static raw_ostream& operator<<(raw_ostream &OS, const SDWADstOperand &Dst) {
  OS << "SDWA dst: " << *Dst.getTargetOperand()
     << " dst_sel:" << Dst.getDstSel()
     << " dst_unused:" << Dst.getDstUnused() << '\n';
  return OS;
}
#endif

static void copyRegOperand(MachineOperand &To, const MachineOperand &From) {
  assert(To.isReg() && From.isReg());
  To.setReg(From.getReg());
  To.setSubReg(From.getSubReg());
  To.setIsUndef(From.isUndef());
  if (To.isUse()) {
    To.setIsKill(From.isKill());
  } else {
    To.setIsDead(From.isDead());
  }
}

static bool isSameReg(const MachineOperand &LHS, const MachineOperand &RHS) {
  return LHS.isReg() &&
         RHS.isReg() &&
         LHS.getReg() == RHS.getReg() &&
         LHS.getSubReg() == RHS.getSubReg();
}

static bool isSubregOf(const MachineOperand &SubReg,
                       const MachineOperand &SuperReg,
                       const TargetRegisterInfo *TRI) {

  if (!SuperReg.isReg() || !SubReg.isReg())
    return false;

  if (isSameReg(SuperReg, SubReg))
    return true;

  if (SuperReg.getReg() != SubReg.getReg())
    return false;

  LaneBitmask SuperMask = TRI->getSubRegIndexLaneMask(SuperReg.getSubReg());
  LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubReg.getSubReg());
  SuperMask |= ~SubMask;
  return SuperMask.all();
}

uint64_t SDWASrcOperand::getSrcMods(const SIInstrInfo *TII,
                                    const MachineOperand *SrcOp) const {
  uint64_t Mods = 0;
  const auto *MI = SrcOp->getParent();
  if (TII->getNamedOperand(*MI, AMDGPU::OpName::src0) == SrcOp) {
    if (auto *Mod = TII->getNamedOperand(*MI, AMDGPU::OpName::src0_modifiers)) {
      Mods = Mod->getImm();
    }
  } else if (TII->getNamedOperand(*MI, AMDGPU::OpName::src1) == SrcOp) {
    if (auto *Mod = TII->getNamedOperand(*MI, AMDGPU::OpName::src1_modifiers)) {
      Mods = Mod->getImm();
    }
  }
  if (Abs || Neg) {
    assert(!Sext &&
           "Float and integer src modifiers can't be set simulteniously");
    Mods |= Abs ? SISrcMods::ABS : 0;
    Mods ^= Neg ? SISrcMods::NEG : 0;
  } else if (Sext) {
    Mods |= SISrcMods::SEXT;
  }

  return Mods;
}

MachineInstr *SDWASrcOperand::potentialToConvert(const SIInstrInfo *TII) {
  // For SDWA src operand potential instruction is one that use register
  // defined by parent instruction
  MachineRegisterInfo *MRI = getMRI();
  MachineOperand *Replaced = getReplacedOperand();
  assert(Replaced->isReg());

  MachineInstr *PotentialMI = nullptr;
  for (MachineOperand &PotentialMO : MRI->use_operands(Replaced->getReg())) {
    // If this is use of another subreg of dst reg then do nothing
    if (!isSubregOf(*Replaced, PotentialMO, MRI->getTargetRegisterInfo()))
      continue;

    // If there exist use of superreg of dst then we should not combine this
    // opernad
    if (!isSameReg(PotentialMO, *Replaced))
      return nullptr;

    // Check that PotentialMI is only instruction that uses dst reg
    if (PotentialMI == nullptr) {
      PotentialMI = PotentialMO.getParent();
    } else if (PotentialMI != PotentialMO.getParent()) {
      return nullptr;
    }
  }

  return PotentialMI;
}

bool SDWASrcOperand::convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) {
  // Find operand in instruction that matches source operand and replace it with
  // target operand. Set corresponding src_sel

  MachineOperand *Src = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  MachineOperand *SrcSel = TII->getNamedOperand(MI, AMDGPU::OpName::src0_sel);
  MachineOperand *SrcMods =
      TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers);
  assert(Src && (Src->isReg() || Src->isImm()));
  if (!isSameReg(*Src, *getReplacedOperand())) {
    // If this is not src0 then it should be src1
    Src = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
    SrcSel = TII->getNamedOperand(MI, AMDGPU::OpName::src1_sel);
    SrcMods = TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers);

    assert(Src && Src->isReg());

    if ((MI.getOpcode() == AMDGPU::V_MAC_F16_sdwa ||
         MI.getOpcode() == AMDGPU::V_MAC_F32_sdwa) &&
        !isSameReg(*Src, *getReplacedOperand())) {
      // In case of v_mac_f16/32_sdwa this pass can try to apply src operand to
      // src2. This is not allowed.
      return false;
    }

    assert(isSameReg(*Src, *getReplacedOperand()) && SrcSel && SrcMods);
  }
  copyRegOperand(*Src, *getTargetOperand());
  SrcSel->setImm(getSrcSel());
  SrcMods->setImm(getSrcMods(TII, Src));
  getTargetOperand()->setIsKill(false);
  return true;
}

MachineInstr *SDWADstOperand::potentialToConvert(const SIInstrInfo *TII) {
  // For SDWA dst operand potential instruction is one that defines register
  // that this operand uses
  MachineRegisterInfo *MRI = getMRI();
  MachineInstr *ParentMI = getParentInst();
  MachineOperand *Replaced = getReplacedOperand();
  assert(Replaced->isReg());

  for (MachineOperand &PotentialMO : MRI->def_operands(Replaced->getReg())) {
    if (!isSubregOf(*Replaced, PotentialMO, MRI->getTargetRegisterInfo()))
      continue;

    if (!isSameReg(*Replaced, PotentialMO))
      return nullptr;

    // Check that ParentMI is the only instruction that uses replaced register
    for (MachineOperand &UseMO : MRI->use_operands(PotentialMO.getReg())) {
      if (isSubregOf(UseMO, PotentialMO, MRI->getTargetRegisterInfo()) &&
          UseMO.getParent() != ParentMI) {
        return nullptr;
      }
    }

    // Due to SSA this should be onle def of replaced register, so return it
    return PotentialMO.getParent();
  }

  return nullptr;
}

bool SDWADstOperand::convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) {
  // Replace vdst operand in MI with target operand. Set dst_sel and dst_unused

  if ((MI.getOpcode() == AMDGPU::V_MAC_F16_sdwa ||
       MI.getOpcode() == AMDGPU::V_MAC_F32_sdwa) &&
      getDstSel() != AMDGPU::SDWA::DWORD) {
    // v_mac_f16/32_sdwa allow dst_sel to be equal only to DWORD
    return false;
  }

  MachineOperand *Operand = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
  assert(Operand &&
         Operand->isReg() &&
         isSameReg(*Operand, *getReplacedOperand()));
  copyRegOperand(*Operand, *getTargetOperand());
  MachineOperand *DstSel= TII->getNamedOperand(MI, AMDGPU::OpName::dst_sel);
  assert(DstSel);
  DstSel->setImm(getDstSel());
  MachineOperand *DstUnused= TII->getNamedOperand(MI, AMDGPU::OpName::dst_unused);
  assert(DstUnused);
  DstUnused->setImm(getDstUnused());

  // Remove original instruction  because it would conflict with our new
  // instruction by register definition
  getParentInst()->eraseFromParent();
  return true;
}

Optional<int64_t> SIPeepholeSDWA::foldToImm(const MachineOperand &Op) const {
  if (Op.isImm()) {
    return Op.getImm();
  }

  // If this is not immediate then it can be copy of immediate value, e.g.:
  // %vreg1<def> = S_MOV_B32 255;
  if (Op.isReg()) {
    for (const MachineOperand &Def : MRI->def_operands(Op.getReg())) {
      if (!isSameReg(Op, Def))
        continue;

      const MachineInstr *DefInst = Def.getParent();
      if (!TII->isFoldableCopy(*DefInst))
        return None;

      const MachineOperand &Copied = DefInst->getOperand(1);
      if (!Copied.isImm())
        return None;

      return Copied.getImm();
    }
  }

  return None;
}

void SIPeepholeSDWA::matchSDWAOperands(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      switch (Opcode) {
      case AMDGPU::V_LSHRREV_B32_e32:
      case AMDGPU::V_ASHRREV_I32_e32:
      case AMDGPU::V_LSHLREV_B32_e32:
      case AMDGPU::V_LSHRREV_B32_e64:
      case AMDGPU::V_ASHRREV_I32_e64:
      case AMDGPU::V_LSHLREV_B32_e64: {
        // from: v_lshrrev_b32_e32 v1, 16/24, v0
        // to SDWA src:v0 src_sel:WORD_1/BYTE_3

        // from: v_ashrrev_i32_e32 v1, 16/24, v0
        // to SDWA src:v0 src_sel:WORD_1/BYTE_3 sext:1

        // from: v_lshlrev_b32_e32 v1, 16/24, v0
        // to SDWA dst:v1 dst_sel:WORD_1/BYTE_3 dst_unused:UNUSED_PAD
        MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
        auto Imm = foldToImm(*Src0);
        if (!Imm)
          break;

        if (*Imm != 16 && *Imm != 24)
          break;

        MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
        MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
        if (TRI->isPhysicalRegister(Src1->getReg()) ||
            TRI->isPhysicalRegister(Dst->getReg()))
          break;

        if (Opcode == AMDGPU::V_LSHLREV_B32_e32 ||
            Opcode == AMDGPU::V_LSHLREV_B32_e64) {
          auto SDWADst = make_unique<SDWADstOperand>(
              Dst, Src1, *Imm == 16 ? WORD_1 : BYTE_3, UNUSED_PAD);
          DEBUG(dbgs() << "Match: " << MI << "To: " << *SDWADst << '\n');
          SDWAOperands[&MI] = std::move(SDWADst);
          ++NumSDWAPatternsFound;
        } else {
          auto SDWASrc = make_unique<SDWASrcOperand>(
              Src1, Dst, *Imm == 16 ? WORD_1 : BYTE_3, false, false,
              Opcode != AMDGPU::V_LSHRREV_B32_e32 &&
              Opcode != AMDGPU::V_LSHRREV_B32_e64);
          DEBUG(dbgs() << "Match: " << MI << "To: " << *SDWASrc << '\n');
          SDWAOperands[&MI] = std::move(SDWASrc);
          ++NumSDWAPatternsFound;
        }
        break;
      }

      case AMDGPU::V_LSHRREV_B16_e32:
      case AMDGPU::V_ASHRREV_I16_e32:
      case AMDGPU::V_LSHLREV_B16_e32:
      case AMDGPU::V_LSHRREV_B16_e64:
      case AMDGPU::V_ASHRREV_I16_e64:
      case AMDGPU::V_LSHLREV_B16_e64: {
        // from: v_lshrrev_b16_e32 v1, 8, v0
        // to SDWA src:v0 src_sel:BYTE_1

        // from: v_ashrrev_i16_e32 v1, 8, v0
        // to SDWA src:v0 src_sel:BYTE_1 sext:1

        // from: v_lshlrev_b16_e32 v1, 8, v0
        // to SDWA dst:v1 dst_sel:BYTE_1 dst_unused:UNUSED_PAD
        MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
        auto Imm = foldToImm(*Src0);
        if (!Imm || *Imm != 8)
          break;

        MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
        MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);

        if (TRI->isPhysicalRegister(Src1->getReg()) ||
            TRI->isPhysicalRegister(Dst->getReg()))
          break;

        if (Opcode == AMDGPU::V_LSHLREV_B16_e32 ||
            Opcode == AMDGPU::V_LSHLREV_B16_e64) {
          auto SDWADst =
            make_unique<SDWADstOperand>(Dst, Src1, BYTE_1, UNUSED_PAD);
          DEBUG(dbgs() << "Match: " << MI << "To: " << *SDWADst << '\n');
          SDWAOperands[&MI] = std::move(SDWADst);
          ++NumSDWAPatternsFound;
        } else {
          auto SDWASrc = make_unique<SDWASrcOperand>(
              Src1, Dst, BYTE_1, false, false,
              Opcode != AMDGPU::V_LSHRREV_B16_e32 &&
              Opcode != AMDGPU::V_LSHRREV_B16_e64);
          DEBUG(dbgs() << "Match: " << MI << "To: " << *SDWASrc << '\n');
          SDWAOperands[&MI] = std::move(SDWASrc);
          ++NumSDWAPatternsFound;
        }
        break;
      }

      case AMDGPU::V_BFE_I32:
      case AMDGPU::V_BFE_U32: {
        // e.g.:
        // from: v_bfe_u32 v1, v0, 8, 8
        // to SDWA src:v0 src_sel:BYTE_1

        // offset | width | src_sel
        // ------------------------
        // 0      | 8     | BYTE_0
        // 0      | 16    | WORD_0
        // 0      | 32    | DWORD ?
        // 8      | 8     | BYTE_1
        // 16     | 8     | BYTE_2
        // 16     | 16    | WORD_1
        // 24     | 8     | BYTE_3

        MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
        auto Offset = foldToImm(*Src1);
        if (!Offset)
          break;

        MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
        auto Width = foldToImm(*Src2);
        if (!Width)
          break;

        SdwaSel SrcSel = DWORD;

        if (*Offset == 0 && *Width == 8)
          SrcSel = BYTE_0;
        else if (*Offset == 0 && *Width == 16)
          SrcSel = WORD_0;
        else if (*Offset == 0 && *Width == 32)
          SrcSel = DWORD;
        else if (*Offset == 8 && *Width == 8)
          SrcSel = BYTE_1;
        else if (*Offset == 16 && *Width == 8)
          SrcSel = BYTE_2;
        else if (*Offset == 16 && *Width == 16)
          SrcSel = WORD_1;
        else if (*Offset == 24 && *Width == 8)
          SrcSel = BYTE_3;
        else
          break;

        MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
        MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);

        if (TRI->isPhysicalRegister(Src0->getReg()) ||
            TRI->isPhysicalRegister(Dst->getReg()))
          break;

        auto SDWASrc = make_unique<SDWASrcOperand>(
            Src0, Dst, SrcSel, false, false,
            Opcode != AMDGPU::V_BFE_U32);
        DEBUG(dbgs() << "Match: " << MI << "To: " << *SDWASrc << '\n');
        SDWAOperands[&MI] = std::move(SDWASrc);
        ++NumSDWAPatternsFound;
        break;
      }
      case AMDGPU::V_AND_B32_e32:
      case AMDGPU::V_AND_B32_e64: {
        // e.g.:
        // from: v_and_b32_e32 v1, 0x0000ffff/0x000000ff, v0
        // to SDWA src:v0 src_sel:WORD_0/BYTE_0

        MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
        MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
        auto ValSrc = Src1;
        auto Imm = foldToImm(*Src0);

        if (!Imm) {
          Imm = foldToImm(*Src1);
          ValSrc = Src0;
        }

        if (!Imm || (*Imm != 0x0000ffff && *Imm != 0x000000ff))
          break;

        MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);

        if (TRI->isPhysicalRegister(Src1->getReg()) ||
            TRI->isPhysicalRegister(Dst->getReg()))
          break;

        auto SDWASrc = make_unique<SDWASrcOperand>(
            ValSrc, Dst, *Imm == 0x0000ffff ? WORD_0 : BYTE_0);
        DEBUG(dbgs() << "Match: " << MI << "To: " << *SDWASrc << '\n');
        SDWAOperands[&MI] = std::move(SDWASrc);
        ++NumSDWAPatternsFound;
        break;
      }
      }
    }
  }
}

bool SIPeepholeSDWA::isConvertibleToSDWA(const MachineInstr &MI,
                                         const SISubtarget &ST) const {
  // Check if this instruction has opcode that supports SDWA
  int Opc = MI.getOpcode();
  if (AMDGPU::getSDWAOp(Opc) == -1)
    Opc = AMDGPU::getVOPe32(Opc);

  if (Opc == -1 || AMDGPU::getSDWAOp(Opc) == -1)
    return false;

  if (!ST.hasSDWAOmod() && TII->hasModifiersSet(MI, AMDGPU::OpName::omod))
    return false;

  if (TII->isVOPC(Opc)) {
    if (!ST.hasSDWASdst()) {
      const MachineOperand *SDst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst);
      if (SDst && SDst->getReg() != AMDGPU::VCC)
        return false;
    }

    if (!ST.hasSDWAOutModsVOPC() &&
        (TII->hasModifiersSet(MI, AMDGPU::OpName::clamp) ||
         TII->hasModifiersSet(MI, AMDGPU::OpName::omod)))
      return false;

  } else if (TII->getNamedOperand(MI, AMDGPU::OpName::sdst) ||
             !TII->getNamedOperand(MI, AMDGPU::OpName::vdst)) {
    return false;
  }

  if (!ST.hasSDWAMac() && (Opc == AMDGPU::V_MAC_F16_e32 ||
                           Opc == AMDGPU::V_MAC_F32_e32))
    return false;

  return true;
}

bool SIPeepholeSDWA::convertToSDWA(MachineInstr &MI,
                                   const SDWAOperandsVector &SDWAOperands) {
  // Convert to sdwa
  int SDWAOpcode = AMDGPU::getSDWAOp(MI.getOpcode());
  if (SDWAOpcode == -1)
    SDWAOpcode = AMDGPU::getSDWAOp(AMDGPU::getVOPe32(MI.getOpcode()));
  assert(SDWAOpcode != -1);

  const MCInstrDesc &SDWADesc = TII->get(SDWAOpcode);

  // Create SDWA version of instruction MI and initialize its operands
  MachineInstrBuilder SDWAInst =
    BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), SDWADesc);

  // Copy dst, if it is present in original then should also be present in SDWA
  MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
  if (Dst) {
    assert(AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::vdst) != -1);
    SDWAInst.add(*Dst);
  } else if ((Dst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst))) {
    assert(Dst &&
           AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::sdst) != -1);
    SDWAInst.add(*Dst);
  } else {
    assert(AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::sdst) != -1);
    SDWAInst.addReg(AMDGPU::VCC, RegState::Define);
  }

  // Copy src0, initialize src0_modifiers. All sdwa instructions has src0 and
  // src0_modifiers (except for v_nop_sdwa, but it can't get here)
  MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  assert(
    Src0 &&
    AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::src0) != -1 &&
    AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::src0_modifiers) != -1);
  if (auto *Mod = TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers))
    SDWAInst.addImm(Mod->getImm());
  else
    SDWAInst.addImm(0);
  SDWAInst.add(*Src0);

  // Copy src1 if present, initialize src1_modifiers.
  MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  if (Src1) {
    assert(
      AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::src1) != -1 &&
      AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::src1_modifiers) != -1);
    if (auto *Mod = TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers))
      SDWAInst.addImm(Mod->getImm());
    else
      SDWAInst.addImm(0);
    SDWAInst.add(*Src1);
  }

  if (SDWAOpcode == AMDGPU::V_MAC_F16_sdwa ||
      SDWAOpcode == AMDGPU::V_MAC_F32_sdwa) {
    // v_mac_f16/32 has additional src2 operand tied to vdst
    MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
    assert(Src2);
    SDWAInst.add(*Src2);
  }

  // Copy clamp if present, initialize otherwise
  assert(AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::clamp) != -1);
  MachineOperand *Clamp = TII->getNamedOperand(MI, AMDGPU::OpName::clamp);
  if (Clamp) {
    SDWAInst.add(*Clamp);
  } else {
    SDWAInst.addImm(0);
  }

  // Copy omod if present, initialize otherwise if needed
  if (AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::omod) != -1) {
    MachineOperand *OMod = TII->getNamedOperand(MI, AMDGPU::OpName::omod);
    if (OMod) {
      SDWAInst.add(*OMod);
    } else {
      SDWAInst.addImm(0);
    }
  }

  // Initialize dst_sel if present
  if (AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::dst_sel) != -1) {
    SDWAInst.addImm(AMDGPU::SDWA::SdwaSel::DWORD);
  }

  // Initialize dst_unused if present
  if (AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::dst_unused) != -1) {
    SDWAInst.addImm(AMDGPU::SDWA::DstUnused::UNUSED_PAD);
  }

  // Initialize src0_sel
  assert(AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::src0_sel) != -1);
  SDWAInst.addImm(AMDGPU::SDWA::SdwaSel::DWORD);


  // Initialize src1_sel if present
  if (Src1) {
    assert(AMDGPU::getNamedOperandIdx(SDWAOpcode, AMDGPU::OpName::src1_sel) != -1);
    SDWAInst.addImm(AMDGPU::SDWA::SdwaSel::DWORD);
  }

  // Apply all sdwa operand pattenrs
  bool Converted = false;
  for (auto &Operand : SDWAOperands) {
    // There should be no intesection between SDWA operands and potential MIs
    // e.g.:
    // v_and_b32 v0, 0xff, v1 -> src:v1 sel:BYTE_0
    // v_and_b32 v2, 0xff, v0 -> src:v0 sel:BYTE_0
    // v_add_u32 v3, v4, v2
    //
    // In that example it is possible that we would fold 2nd instruction into 3rd
    // (v_add_u32_sdwa) and then try to fold 1st instruction into 2nd (that was
    // already destroyed). So if SDWAOperand is also a potential MI then do not
    // apply it.
    if (PotentialMatches.count(Operand->getParentInst()) == 0)
      Converted |= Operand->convertToSDWA(*SDWAInst, TII);
  }
  if (Converted) {
    ConvertedInstructions.push_back(SDWAInst);
  } else {
    SDWAInst->eraseFromParent();
    return false;
  }

  DEBUG(dbgs() << "Convert instruction:" << MI
               << "Into:" << *SDWAInst << '\n');
  ++NumSDWAInstructionsPeepholed;

  MI.eraseFromParent();
  return true;
}

// If an instruction was converted to SDWA it should not have immediates or SGPR
// operands (allowed one SGPR on GFX9). Copy its scalar operands into VGPRs.
void SIPeepholeSDWA::legalizeScalarOperands(MachineInstr &MI, const SISubtarget &ST) const {
  const MCInstrDesc &Desc = TII->get(MI.getOpcode());
  unsigned ConstantBusCount = 0;
  for (MachineOperand &Op: MI.explicit_uses()) {
    if (!Op.isImm() && !(Op.isReg() && !TRI->isVGPR(*MRI, Op.getReg())))
      continue;

    unsigned I = MI.getOperandNo(&Op);
    if (Desc.OpInfo[I].RegClass == -1 ||
       !TRI->hasVGPRs(TRI->getRegClass(Desc.OpInfo[I].RegClass)))
      continue;

    if (ST.hasSDWAScalar() && ConstantBusCount == 0 && Op.isReg() &&
        TRI->isSGPRReg(*MRI, Op.getReg())) {
      ++ConstantBusCount;
      continue;
    }

    unsigned VGPR = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    auto Copy = BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
                        TII->get(AMDGPU::V_MOV_B32_e32), VGPR);
    if (Op.isImm())
      Copy.addImm(Op.getImm());
    else if (Op.isReg())
      Copy.addReg(Op.getReg(), Op.isKill() ? RegState::Kill : 0,
                  Op.getSubReg());
    Op.ChangeToRegister(VGPR, false);
  }
}

bool SIPeepholeSDWA::runOnMachineFunction(MachineFunction &MF) {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();

  if (!ST.hasSDWA() || skipFunction(*MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();

  // Find all SDWA operands in MF.
  matchSDWAOperands(MF);

  for (const auto &OperandPair : SDWAOperands) {
    const auto &Operand = OperandPair.second;
    MachineInstr *PotentialMI = Operand->potentialToConvert(TII);
    if (PotentialMI && isConvertibleToSDWA(*PotentialMI, ST)) {
      PotentialMatches[PotentialMI].push_back(Operand.get());
    }
  }

  for (auto &PotentialPair : PotentialMatches) {
    MachineInstr &PotentialMI = *PotentialPair.first;
    convertToSDWA(PotentialMI, PotentialPair.second);
  }

  PotentialMatches.clear();
  SDWAOperands.clear();

  bool Ret = !ConvertedInstructions.empty();
  while (!ConvertedInstructions.empty())
    legalizeScalarOperands(*ConvertedInstructions.pop_back_val(), ST);

  return Ret;
}
