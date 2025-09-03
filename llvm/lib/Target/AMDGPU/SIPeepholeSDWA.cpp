//===- SIPeepholeSDWA.cpp - Peephole optimization for SDWA instructions ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass tries to apply several peephole SDWA patterns.
///
/// E.g. original:
///   V_LSHRREV_B32_e32 %0, 16, %1
///   V_ADD_CO_U32_e32 %2, %0, %3
///   V_LSHLREV_B32_e32 %4, 16, %2
///
/// Replace:
///   V_ADD_CO_U32_sdwa %4, %1, %3
///       dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
///
//===----------------------------------------------------------------------===//

#include "SIPeepholeSDWA.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "si-peephole-sdwa"

STATISTIC(NumSDWAPatternsFound, "Number of SDWA patterns found.");
STATISTIC(NumSDWAInstructionsPeepholed,
          "Number of instruction converted to SDWA.");

namespace {

bool isConvertibleToSDWA(MachineInstr &MI, const GCNSubtarget &ST,
                         const SIInstrInfo *TII);
class SDWAOperand;
class SDWADstOperand;

using SDWAOperandsVector = SmallVector<SDWAOperand *, 4>;
using SDWAOperandsMap = MapVector<MachineInstr *, SDWAOperandsVector>;

class SIPeepholeSDWA {
private:
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;

  MapVector<MachineInstr *, std::unique_ptr<SDWAOperand>> SDWAOperands;
  SDWAOperandsMap PotentialMatches;
  SmallVector<MachineInstr *, 8> ConvertedInstructions;

  std::optional<int64_t> foldToImm(const MachineOperand &Op) const;

  void matchSDWAOperands(MachineBasicBlock &MBB);
  std::unique_ptr<SDWAOperand> matchSDWAOperand(MachineInstr &MI);
  void pseudoOpConvertToVOP2(MachineInstr &MI,
                             const GCNSubtarget &ST) const;
  void convertVcndmaskToVOP2(MachineInstr &MI, const GCNSubtarget &ST) const;
  MachineInstr *createSDWAVersion(MachineInstr &MI);
  bool convertToSDWA(MachineInstr &MI, const SDWAOperandsVector &SDWAOperands);
  void legalizeScalarOperands(MachineInstr &MI, const GCNSubtarget &ST) const;

public:
  bool run(MachineFunction &MF);
};

class SIPeepholeSDWALegacy : public MachineFunctionPass {
public:
  static char ID;

  SIPeepholeSDWALegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SI Peephole SDWA"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

using namespace AMDGPU::SDWA;

class SDWAOperand {
private:
  MachineOperand *Target; // Operand that would be used in converted instruction
  MachineOperand *Replaced; // Operand that would be replace by Target

  /// Returns true iff the SDWA selection of this SDWAOperand can be combined
  /// with the SDWA selections of its uses in \p MI.
  virtual bool canCombineSelections(const MachineInstr &MI,
                                    const SIInstrInfo *TII) = 0;

public:
  SDWAOperand(MachineOperand *TargetOp, MachineOperand *ReplacedOp)
      : Target(TargetOp), Replaced(ReplacedOp) {
    assert(Target->isReg());
    assert(Replaced->isReg());
  }

  virtual ~SDWAOperand() = default;

  virtual MachineInstr *potentialToConvert(const SIInstrInfo *TII,
                                           const GCNSubtarget &ST,
                                           SDWAOperandsMap *PotentialMatches = nullptr) = 0;
  virtual bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) = 0;

  MachineOperand *getTargetOperand() const { return Target; }
  MachineOperand *getReplacedOperand() const { return Replaced; }
  MachineInstr *getParentInst() const { return Target->getParent(); }

  MachineRegisterInfo *getMRI() const {
    return &getParentInst()->getParent()->getParent()->getRegInfo();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  virtual void print(raw_ostream& OS) const = 0;
  void dump() const { print(dbgs()); }
#endif
};

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

  MachineInstr *potentialToConvert(const SIInstrInfo *TII,
                                   const GCNSubtarget &ST,
                                   SDWAOperandsMap *PotentialMatches = nullptr) override;
  bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) override;
  bool canCombineSelections(const MachineInstr &MI,
                            const SIInstrInfo *TII) override;

  SdwaSel getSrcSel() const { return SrcSel; }
  bool getAbs() const { return Abs; }
  bool getNeg() const { return Neg; }
  bool getSext() const { return Sext; }

  uint64_t getSrcMods(const SIInstrInfo *TII,
                      const MachineOperand *SrcOp) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream& OS) const override;
#endif
};

class SDWADstOperand : public SDWAOperand {
private:
  SdwaSel DstSel;
  DstUnused DstUn;

public:
  SDWADstOperand(MachineOperand *TargetOp, MachineOperand *ReplacedOp,
                 SdwaSel DstSel_ = DWORD, DstUnused DstUn_ = UNUSED_PAD)
      : SDWAOperand(TargetOp, ReplacedOp), DstSel(DstSel_), DstUn(DstUn_) {}

  MachineInstr *potentialToConvert(const SIInstrInfo *TII,
                                   const GCNSubtarget &ST,
                                   SDWAOperandsMap *PotentialMatches = nullptr) override;
  bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) override;
  bool canCombineSelections(const MachineInstr &MI,
                            const SIInstrInfo *TII) override;

  SdwaSel getDstSel() const { return DstSel; }
  DstUnused getDstUnused() const { return DstUn; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream& OS) const override;
#endif
};

class SDWADstPreserveOperand : public SDWADstOperand {
private:
  MachineOperand *Preserve;

public:
  SDWADstPreserveOperand(MachineOperand *TargetOp, MachineOperand *ReplacedOp,
                         MachineOperand *PreserveOp, SdwaSel DstSel_ = DWORD)
      : SDWADstOperand(TargetOp, ReplacedOp, DstSel_, UNUSED_PRESERVE),
        Preserve(PreserveOp) {}

  bool convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) override;
  bool canCombineSelections(const MachineInstr &MI,
                            const SIInstrInfo *TII) override;

  MachineOperand *getPreservedOperand() const { return Preserve; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream& OS) const override;
#endif
};

} // end anonymous namespace

INITIALIZE_PASS(SIPeepholeSDWALegacy, DEBUG_TYPE, "SI Peephole SDWA", false,
                false)

char SIPeepholeSDWALegacy::ID = 0;

char &llvm::SIPeepholeSDWALegacyID = SIPeepholeSDWALegacy::ID;

FunctionPass *llvm::createSIPeepholeSDWALegacyPass() {
  return new SIPeepholeSDWALegacy();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static raw_ostream& operator<<(raw_ostream &OS, SdwaSel Sel) {
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

LLVM_DUMP_METHOD
void SDWASrcOperand::print(raw_ostream& OS) const {
  OS << "SDWA src: " << *getTargetOperand()
    << " src_sel:" << getSrcSel()
    << " abs:" << getAbs() << " neg:" << getNeg()
    << " sext:" << getSext() << '\n';
}

LLVM_DUMP_METHOD
void SDWADstOperand::print(raw_ostream& OS) const {
  OS << "SDWA dst: " << *getTargetOperand()
    << " dst_sel:" << getDstSel()
    << " dst_unused:" << getDstUnused() << '\n';
}

LLVM_DUMP_METHOD
void SDWADstPreserveOperand::print(raw_ostream& OS) const {
  OS << "SDWA preserve dst: " << *getTargetOperand()
    << " dst_sel:" << getDstSel()
    << " preserve:" << *getPreservedOperand() << '\n';
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

static MachineOperand *findSingleRegUse(const MachineOperand *Reg,
                                        const MachineRegisterInfo *MRI) {
  if (!Reg->isReg() || !Reg->isDef())
    return nullptr;

  return MRI->getOneNonDBGUse(Reg->getReg());
}

static MachineOperand *findSingleRegDef(const MachineOperand *Reg,
                                        const MachineRegisterInfo *MRI) {
  if (!Reg->isReg())
    return nullptr;

  return MRI->getOneDef(Reg->getReg());
}

/// Combine an SDWA instruction's existing SDWA selection \p Sel with
/// the SDWA selection \p OperandSel of its operand. If the selections
/// are compatible, return the combined selection, otherwise return a
/// nullopt.
/// For example, if we have Sel = BYTE_0 Sel and OperandSel = WORD_1:
///     BYTE_0 Sel (WORD_1 Sel (%X)) -> BYTE_2 Sel (%X)
static std::optional<SdwaSel> combineSdwaSel(SdwaSel Sel, SdwaSel OperandSel) {
  if (Sel == SdwaSel::DWORD)
    return OperandSel;

  if (Sel == OperandSel || OperandSel == SdwaSel::DWORD)
    return Sel;

  if (Sel == SdwaSel::WORD_1 || Sel == SdwaSel::BYTE_2 ||
      Sel == SdwaSel::BYTE_3)
    return {};

  if (OperandSel == SdwaSel::WORD_0)
    return Sel;

  if (OperandSel == SdwaSel::WORD_1) {
    if (Sel == SdwaSel::BYTE_0)
      return SdwaSel::BYTE_2;
    if (Sel == SdwaSel::BYTE_1)
      return SdwaSel::BYTE_3;
    if (Sel == SdwaSel::WORD_0)
      return SdwaSel::WORD_1;
  }

  return {};
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
           "Float and integer src modifiers can't be set simultaneously");
    Mods |= Abs ? SISrcMods::ABS : 0u;
    Mods ^= Neg ? SISrcMods::NEG : 0u;
  } else if (Sext) {
    Mods |= SISrcMods::SEXT;
  }

  return Mods;
}

MachineInstr *SDWASrcOperand::potentialToConvert(const SIInstrInfo *TII,
                                                 const GCNSubtarget &ST,
                                                 SDWAOperandsMap *PotentialMatches) {
  if (PotentialMatches != nullptr) {
    // Fill out the map for all uses if all can be converted
    MachineOperand *Reg = getReplacedOperand();
    if (!Reg->isReg() || !Reg->isDef())
      return nullptr;

    for (MachineInstr &UseMI : getMRI()->use_nodbg_instructions(Reg->getReg()))
      // Check that all instructions that use Reg can be converted
      if (!isConvertibleToSDWA(UseMI, ST, TII) ||
          !canCombineSelections(UseMI, TII))
        return nullptr;

    // Now that it's guaranteed all uses are legal, iterate over the uses again
    // to add them for later conversion.
    for (MachineOperand &UseMO : getMRI()->use_nodbg_operands(Reg->getReg())) {
      // Should not get a subregister here
      assert(isSameReg(UseMO, *Reg));

      SDWAOperandsMap &potentialMatchesMap = *PotentialMatches;
      MachineInstr *UseMI = UseMO.getParent();
      potentialMatchesMap[UseMI].push_back(this);
    }
    return nullptr;
  }

  // For SDWA src operand potential instruction is one that use register
  // defined by parent instruction
  MachineOperand *PotentialMO = findSingleRegUse(getReplacedOperand(), getMRI());
  if (!PotentialMO)
    return nullptr;

  MachineInstr *Parent = PotentialMO->getParent();

  return canCombineSelections(*Parent, TII) ? Parent : nullptr;
}

bool SDWASrcOperand::convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) {
  switch (MI.getOpcode()) {
  case AMDGPU::V_CVT_F32_FP8_sdwa:
  case AMDGPU::V_CVT_F32_BF8_sdwa:
  case AMDGPU::V_CVT_PK_F32_FP8_sdwa:
  case AMDGPU::V_CVT_PK_F32_BF8_sdwa:
    // Does not support input modifiers: noabs, noneg, nosext.
    return false;
  case AMDGPU::V_CNDMASK_B32_sdwa:
    // SISrcMods uses the same bitmask for SEXT and NEG modifiers and
    // hence the compiler can only support one type of modifier for
    // each SDWA instruction.  For V_CNDMASK_B32_sdwa, this is NEG
    // since its operands get printed using
    // AMDGPUInstPrinter::printOperandAndFPInputMods which produces
    // the output intended for NEG if SEXT is set.
    //
    // The ISA does actually support both modifiers on most SDWA
    // instructions.
    //
    // FIXME Accept SEXT here after fixing this issue.
    if (Sext)
      return false;
    break;
  }

  // Find operand in instruction that matches source operand and replace it with
  // target operand. Set corresponding src_sel
  bool IsPreserveSrc = false;
  MachineOperand *Src = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  MachineOperand *SrcSel = TII->getNamedOperand(MI, AMDGPU::OpName::src0_sel);
  MachineOperand *SrcMods =
      TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers);
  assert(Src && (Src->isReg() || Src->isImm()));
  if (!isSameReg(*Src, *getReplacedOperand())) {
    // If this is not src0 then it could be src1
    Src = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
    SrcSel = TII->getNamedOperand(MI, AMDGPU::OpName::src1_sel);
    SrcMods = TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers);

    if (!Src ||
        !isSameReg(*Src, *getReplacedOperand())) {
      // It's possible this Src is a tied operand for
      // UNUSED_PRESERVE, in which case we can either
      // abandon the peephole attempt, or if legal we can
      // copy the target operand into the tied slot
      // if the preserve operation will effectively cause the same
      // result by overwriting the rest of the dst.
      MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
      MachineOperand *DstUnused =
        TII->getNamedOperand(MI, AMDGPU::OpName::dst_unused);

      if (Dst &&
          DstUnused->getImm() == AMDGPU::SDWA::DstUnused::UNUSED_PRESERVE) {
        // This will work if the tied src is accessing WORD_0, and the dst is
        // writing WORD_1. Modifiers don't matter because all the bits that
        // would be impacted are being overwritten by the dst.
        // Any other case will not work.
        SdwaSel DstSel = static_cast<SdwaSel>(
            TII->getNamedImmOperand(MI, AMDGPU::OpName::dst_sel));
        if (DstSel == AMDGPU::SDWA::SdwaSel::WORD_1 &&
            getSrcSel() == AMDGPU::SDWA::SdwaSel::WORD_0) {
          IsPreserveSrc = true;
          auto DstIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                                   AMDGPU::OpName::vdst);
          auto TiedIdx = MI.findTiedOperandIdx(DstIdx);
          Src = &MI.getOperand(TiedIdx);
          SrcSel = nullptr;
          SrcMods = nullptr;
        } else {
          // Not legal to convert this src
          return false;
        }
      }
    }
    assert(Src && Src->isReg());

    if ((MI.getOpcode() == AMDGPU::V_FMAC_F16_sdwa ||
         MI.getOpcode() == AMDGPU::V_FMAC_F32_sdwa ||
         MI.getOpcode() == AMDGPU::V_MAC_F16_sdwa ||
         MI.getOpcode() == AMDGPU::V_MAC_F32_sdwa) &&
         !isSameReg(*Src, *getReplacedOperand())) {
      // In case of v_mac_f16/32_sdwa this pass can try to apply src operand to
      // src2. This is not allowed.
      return false;
    }

    assert(isSameReg(*Src, *getReplacedOperand()) &&
           (IsPreserveSrc || (SrcSel && SrcMods)));
  }
  copyRegOperand(*Src, *getTargetOperand());
  if (!IsPreserveSrc) {
    SdwaSel ExistingSel = static_cast<SdwaSel>(SrcSel->getImm());
    SrcSel->setImm(*combineSdwaSel(ExistingSel, getSrcSel()));
    SrcMods->setImm(getSrcMods(TII, Src));
  }
  getTargetOperand()->setIsKill(false);
  return true;
}

/// Verify that the SDWA selection operand \p SrcSelOpName of the SDWA
/// instruction \p MI can be combined with the selection \p OpSel.
static bool canCombineOpSel(const MachineInstr &MI, const SIInstrInfo *TII,
                            AMDGPU::OpName SrcSelOpName, SdwaSel OpSel) {
  assert(TII->isSDWA(MI.getOpcode()));

  const MachineOperand *SrcSelOp = TII->getNamedOperand(MI, SrcSelOpName);
  SdwaSel SrcSel = static_cast<SdwaSel>(SrcSelOp->getImm());

  return combineSdwaSel(SrcSel, OpSel).has_value();
}

/// Verify that \p Op is the same register as the operand of the SDWA
/// instruction \p MI named by \p SrcOpName and that the SDWA
/// selection \p SrcSelOpName can be combined with the \p OpSel.
static bool canCombineOpSel(const MachineInstr &MI, const SIInstrInfo *TII,
                            AMDGPU::OpName SrcOpName,
                            AMDGPU::OpName SrcSelOpName, MachineOperand *Op,
                            SdwaSel OpSel) {
  assert(TII->isSDWA(MI.getOpcode()));

  const MachineOperand *Src = TII->getNamedOperand(MI, SrcOpName);
  if (!Src || !isSameReg(*Src, *Op))
    return true;

  return canCombineOpSel(MI, TII, SrcSelOpName, OpSel);
}

bool SDWASrcOperand::canCombineSelections(const MachineInstr &MI,
                                          const SIInstrInfo *TII) {
  if (!TII->isSDWA(MI.getOpcode()))
    return true;

  using namespace AMDGPU;

  return canCombineOpSel(MI, TII, OpName::src0, OpName::src0_sel,
                         getReplacedOperand(), getSrcSel()) &&
         canCombineOpSel(MI, TII, OpName::src1, OpName::src1_sel,
                         getReplacedOperand(), getSrcSel());
}

MachineInstr *SDWADstOperand::potentialToConvert(const SIInstrInfo *TII,
                                                 const GCNSubtarget &ST,
                                                 SDWAOperandsMap *PotentialMatches) {
  // For SDWA dst operand potential instruction is one that defines register
  // that this operand uses
  MachineRegisterInfo *MRI = getMRI();
  MachineInstr *ParentMI = getParentInst();

  MachineOperand *PotentialMO = findSingleRegDef(getReplacedOperand(), MRI);
  if (!PotentialMO)
    return nullptr;

  // Check that ParentMI is the only instruction that uses replaced register
  for (MachineInstr &UseInst : MRI->use_nodbg_instructions(PotentialMO->getReg())) {
    if (&UseInst != ParentMI)
      return nullptr;
  }

  MachineInstr *Parent = PotentialMO->getParent();
  return canCombineSelections(*Parent, TII) ? Parent : nullptr;
}

bool SDWADstOperand::convertToSDWA(MachineInstr &MI, const SIInstrInfo *TII) {
  // Replace vdst operand in MI with target operand. Set dst_sel and dst_unused

  if ((MI.getOpcode() == AMDGPU::V_FMAC_F16_sdwa ||
       MI.getOpcode() == AMDGPU::V_FMAC_F32_sdwa ||
       MI.getOpcode() == AMDGPU::V_MAC_F16_sdwa ||
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

  SdwaSel ExistingSel = static_cast<SdwaSel>(DstSel->getImm());
  DstSel->setImm(combineSdwaSel(ExistingSel, getDstSel()).value());

  MachineOperand *DstUnused= TII->getNamedOperand(MI, AMDGPU::OpName::dst_unused);
  assert(DstUnused);
  DstUnused->setImm(getDstUnused());

  // Remove original instruction  because it would conflict with our new
  // instruction by register definition
  getParentInst()->eraseFromParent();
  return true;
}

bool SDWADstOperand::canCombineSelections(const MachineInstr &MI,
                                          const SIInstrInfo *TII) {
  if (!TII->isSDWA(MI.getOpcode()))
    return true;

  return canCombineOpSel(MI, TII, AMDGPU::OpName::dst_sel, getDstSel());
}

bool SDWADstPreserveOperand::convertToSDWA(MachineInstr &MI,
                                           const SIInstrInfo *TII) {
  // MI should be moved right before v_or_b32.
  // For this we should clear all kill flags on uses of MI src-operands or else
  // we can encounter problem with use of killed operand.
  for (MachineOperand &MO : MI.uses()) {
    if (!MO.isReg())
      continue;
    getMRI()->clearKillFlags(MO.getReg());
  }

  // Move MI before v_or_b32
  MI.getParent()->remove(&MI);
  getParentInst()->getParent()->insert(getParentInst(), &MI);

  // Add Implicit use of preserved register
  MachineInstrBuilder MIB(*MI.getMF(), MI);
  MIB.addReg(getPreservedOperand()->getReg(),
             RegState::ImplicitKill,
             getPreservedOperand()->getSubReg());

  // Tie dst to implicit use
  MI.tieOperands(AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vdst),
                 MI.getNumOperands() - 1);

  // Convert MI as any other SDWADstOperand and remove v_or_b32
  return SDWADstOperand::convertToSDWA(MI, TII);
}

bool SDWADstPreserveOperand::canCombineSelections(const MachineInstr &MI,
                                                  const SIInstrInfo *TII) {
  return SDWADstOperand::canCombineSelections(MI, TII);
}

std::optional<int64_t>
SIPeepholeSDWA::foldToImm(const MachineOperand &Op) const {
  if (Op.isImm()) {
    return Op.getImm();
  }

  // If this is not immediate then it can be copy of immediate value, e.g.:
  // %1 = S_MOV_B32 255;
  if (Op.isReg()) {
    for (const MachineOperand &Def : MRI->def_operands(Op.getReg())) {
      if (!isSameReg(Op, Def))
        continue;

      const MachineInstr *DefInst = Def.getParent();
      if (!TII->isFoldableCopy(*DefInst))
        return std::nullopt;

      const MachineOperand &Copied = DefInst->getOperand(1);
      if (!Copied.isImm())
        return std::nullopt;

      return Copied.getImm();
    }
  }

  return std::nullopt;
}

std::unique_ptr<SDWAOperand>
SIPeepholeSDWA::matchSDWAOperand(MachineInstr &MI) {
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
    if (!Src1->isReg() || Src1->getReg().isPhysical() ||
        Dst->getReg().isPhysical())
      break;

    if (Opcode == AMDGPU::V_LSHLREV_B32_e32 ||
        Opcode == AMDGPU::V_LSHLREV_B32_e64) {
      return std::make_unique<SDWADstOperand>(
          Dst, Src1, *Imm == 16 ? WORD_1 : BYTE_3, UNUSED_PAD);
    }
    return std::make_unique<SDWASrcOperand>(
        Src1, Dst, *Imm == 16 ? WORD_1 : BYTE_3, false, false,
        Opcode != AMDGPU::V_LSHRREV_B32_e32 &&
            Opcode != AMDGPU::V_LSHRREV_B32_e64);
    break;
  }

  case AMDGPU::V_LSHRREV_B16_e32:
  case AMDGPU::V_ASHRREV_I16_e32:
  case AMDGPU::V_LSHLREV_B16_e32:
  case AMDGPU::V_LSHRREV_B16_e64:
  case AMDGPU::V_LSHRREV_B16_opsel_e64:
  case AMDGPU::V_ASHRREV_I16_e64:
  case AMDGPU::V_LSHLREV_B16_opsel_e64:
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

    if (!Src1->isReg() || Src1->getReg().isPhysical() ||
        Dst->getReg().isPhysical())
      break;

    if (Opcode == AMDGPU::V_LSHLREV_B16_e32 ||
        Opcode == AMDGPU::V_LSHLREV_B16_opsel_e64 ||
        Opcode == AMDGPU::V_LSHLREV_B16_e64)
      return std::make_unique<SDWADstOperand>(Dst, Src1, BYTE_1, UNUSED_PAD);
    return std::make_unique<SDWASrcOperand>(
        Src1, Dst, BYTE_1, false, false,
        Opcode != AMDGPU::V_LSHRREV_B16_e32 &&
            Opcode != AMDGPU::V_LSHRREV_B16_opsel_e64 &&
            Opcode != AMDGPU::V_LSHRREV_B16_e64);
    break;
  }

  case AMDGPU::V_BFE_I32_e64:
  case AMDGPU::V_BFE_U32_e64: {
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

    if (!Src0->isReg() || Src0->getReg().isPhysical() ||
        Dst->getReg().isPhysical())
      break;

    return std::make_unique<SDWASrcOperand>(
          Src0, Dst, SrcSel, false, false, Opcode != AMDGPU::V_BFE_U32_e64);
  }

  case AMDGPU::V_AND_B32_e32:
  case AMDGPU::V_AND_B32_e64: {
    // e.g.:
    // from: v_and_b32_e32 v1, 0x0000ffff/0x000000ff, v0
    // to SDWA src:v0 src_sel:WORD_0/BYTE_0

    MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
    MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
    auto *ValSrc = Src1;
    auto Imm = foldToImm(*Src0);

    if (!Imm) {
      Imm = foldToImm(*Src1);
      ValSrc = Src0;
    }

    if (!Imm || (*Imm != 0x0000ffff && *Imm != 0x000000ff))
      break;

    MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);

    if (!ValSrc->isReg() || ValSrc->getReg().isPhysical() ||
        Dst->getReg().isPhysical())
      break;

    return std::make_unique<SDWASrcOperand>(
        ValSrc, Dst, *Imm == 0x0000ffff ? WORD_0 : BYTE_0);
  }

  case AMDGPU::V_OR_B32_e32:
  case AMDGPU::V_OR_B32_e64: {
    // Patterns for dst_unused:UNUSED_PRESERVE.
    // e.g., from:
    // v_add_f16_sdwa v0, v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD
    //                           src1_sel:WORD_1 src2_sel:WORD1
    // v_add_f16_e32 v3, v1, v2
    // v_or_b32_e32 v4, v0, v3
    // to SDWA preserve dst:v4 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE preserve:v3

    // Check if one of operands of v_or_b32 is SDWA instruction
    using CheckRetType =
        std::optional<std::pair<MachineOperand *, MachineOperand *>>;
    auto CheckOROperandsForSDWA =
      [&](const MachineOperand *Op1, const MachineOperand *Op2) -> CheckRetType {
        if (!Op1 || !Op1->isReg() || !Op2 || !Op2->isReg())
          return CheckRetType(std::nullopt);

        MachineOperand *Op1Def = findSingleRegDef(Op1, MRI);
        if (!Op1Def)
          return CheckRetType(std::nullopt);

        MachineInstr *Op1Inst = Op1Def->getParent();
        if (!TII->isSDWA(*Op1Inst))
          return CheckRetType(std::nullopt);

        MachineOperand *Op2Def = findSingleRegDef(Op2, MRI);
        if (!Op2Def)
          return CheckRetType(std::nullopt);

        return CheckRetType(std::pair(Op1Def, Op2Def));
      };

    MachineOperand *OrSDWA = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
    MachineOperand *OrOther = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
    assert(OrSDWA && OrOther);
    auto Res = CheckOROperandsForSDWA(OrSDWA, OrOther);
    if (!Res) {
      OrSDWA = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
      OrOther = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
      assert(OrSDWA && OrOther);
      Res = CheckOROperandsForSDWA(OrSDWA, OrOther);
      if (!Res)
        break;
    }

    MachineOperand *OrSDWADef = Res->first;
    MachineOperand *OrOtherDef = Res->second;
    assert(OrSDWADef && OrOtherDef);

    MachineInstr *SDWAInst = OrSDWADef->getParent();
    MachineInstr *OtherInst = OrOtherDef->getParent();

    // Check that OtherInstr is actually bitwise compatible with SDWAInst = their
    // destination patterns don't overlap. Compatible instruction can be either
    // regular instruction with compatible bitness or SDWA instruction with
    // correct dst_sel
    // SDWAInst | OtherInst bitness / OtherInst dst_sel
    // -----------------------------------------------------
    // DWORD    | no                    / no
    // WORD_0   | no                    / BYTE_2/3, WORD_1
    // WORD_1   | 8/16-bit instructions / BYTE_0/1, WORD_0
    // BYTE_0   | no                    / BYTE_1/2/3, WORD_1
    // BYTE_1   | 8-bit                 / BYTE_0/2/3, WORD_1
    // BYTE_2   | 8/16-bit              / BYTE_0/1/3. WORD_0
    // BYTE_3   | 8/16/24-bit           / BYTE_0/1/2, WORD_0
    // E.g. if SDWAInst is v_add_f16_sdwa dst_sel:WORD_1 then v_add_f16 is OK
    // but v_add_f32 is not.

    // TODO: add support for non-SDWA instructions as OtherInst.
    // For now this only works with SDWA instructions. For regular instructions
    // there is no way to determine if the instruction writes only 8/16/24-bit
    // out of full register size and all registers are at min 32-bit wide.
    if (!TII->isSDWA(*OtherInst))
      break;

    SdwaSel DstSel = static_cast<SdwaSel>(
        TII->getNamedImmOperand(*SDWAInst, AMDGPU::OpName::dst_sel));
    SdwaSel OtherDstSel = static_cast<SdwaSel>(
      TII->getNamedImmOperand(*OtherInst, AMDGPU::OpName::dst_sel));

    bool DstSelAgree = false;
    switch (DstSel) {
    case WORD_0: DstSelAgree = ((OtherDstSel == BYTE_2) ||
                                (OtherDstSel == BYTE_3) ||
                                (OtherDstSel == WORD_1));
      break;
    case WORD_1: DstSelAgree = ((OtherDstSel == BYTE_0) ||
                                (OtherDstSel == BYTE_1) ||
                                (OtherDstSel == WORD_0));
      break;
    case BYTE_0: DstSelAgree = ((OtherDstSel == BYTE_1) ||
                                (OtherDstSel == BYTE_2) ||
                                (OtherDstSel == BYTE_3) ||
                                (OtherDstSel == WORD_1));
      break;
    case BYTE_1: DstSelAgree = ((OtherDstSel == BYTE_0) ||
                                (OtherDstSel == BYTE_2) ||
                                (OtherDstSel == BYTE_3) ||
                                (OtherDstSel == WORD_1));
      break;
    case BYTE_2: DstSelAgree = ((OtherDstSel == BYTE_0) ||
                                (OtherDstSel == BYTE_1) ||
                                (OtherDstSel == BYTE_3) ||
                                (OtherDstSel == WORD_0));
      break;
    case BYTE_3: DstSelAgree = ((OtherDstSel == BYTE_0) ||
                                (OtherDstSel == BYTE_1) ||
                                (OtherDstSel == BYTE_2) ||
                                (OtherDstSel == WORD_0));
      break;
    default: DstSelAgree = false;
    }

    if (!DstSelAgree)
      break;

    // Also OtherInst dst_unused should be UNUSED_PAD
    DstUnused OtherDstUnused = static_cast<DstUnused>(
      TII->getNamedImmOperand(*OtherInst, AMDGPU::OpName::dst_unused));
    if (OtherDstUnused != DstUnused::UNUSED_PAD)
      break;

    // Create DstPreserveOperand
    MachineOperand *OrDst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
    assert(OrDst && OrDst->isReg());

    return std::make_unique<SDWADstPreserveOperand>(
      OrDst, OrSDWADef, OrOtherDef, DstSel);

  }
  }

  return std::unique_ptr<SDWAOperand>(nullptr);
}

#if !defined(NDEBUG)
static raw_ostream& operator<<(raw_ostream &OS, const SDWAOperand &Operand) {
  Operand.print(OS);
  return OS;
}
#endif

void SIPeepholeSDWA::matchSDWAOperands(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    if (auto Operand = matchSDWAOperand(MI)) {
      LLVM_DEBUG(dbgs() << "Match: " << MI << "To: " << *Operand << '\n');
      SDWAOperands[&MI] = std::move(Operand);
      ++NumSDWAPatternsFound;
    }
  }
}

// Convert the V_ADD_CO_U32_e64 into V_ADD_CO_U32_e32. This allows
// isConvertibleToSDWA to perform its transformation on V_ADD_CO_U32_e32 into
// V_ADD_CO_U32_sdwa.
//
// We are transforming from a VOP3 into a VOP2 form of the instruction.
//   %19:vgpr_32 = V_AND_B32_e32 255,
//       killed %16:vgpr_32, implicit $exec
//   %47:vgpr_32, %49:sreg_64_xexec = V_ADD_CO_U32_e64
//       %26.sub0:vreg_64, %19:vgpr_32, implicit $exec
//  %48:vgpr_32, dead %50:sreg_64_xexec = V_ADDC_U32_e64
//       %26.sub1:vreg_64, %54:vgpr_32, killed %49:sreg_64_xexec, implicit $exec
//
// becomes
//   %47:vgpr_32 = V_ADD_CO_U32_sdwa
//       0, %26.sub0:vreg_64, 0, killed %16:vgpr_32, 0, 6, 0, 6, 0,
//       implicit-def $vcc, implicit $exec
//  %48:vgpr_32, dead %50:sreg_64_xexec = V_ADDC_U32_e64
//       %26.sub1:vreg_64, %54:vgpr_32, killed $vcc, implicit $exec
void SIPeepholeSDWA::pseudoOpConvertToVOP2(MachineInstr &MI,
                                           const GCNSubtarget &ST) const {
  int Opc = MI.getOpcode();
  assert((Opc == AMDGPU::V_ADD_CO_U32_e64 || Opc == AMDGPU::V_SUB_CO_U32_e64) &&
         "Currently only handles V_ADD_CO_U32_e64 or V_SUB_CO_U32_e64");

  // Can the candidate MI be shrunk?
  if (!TII->canShrink(MI, *MRI))
    return;
  Opc = AMDGPU::getVOPe32(Opc);
  // Find the related ADD instruction.
  const MachineOperand *Sdst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst);
  if (!Sdst)
    return;
  MachineOperand *NextOp = findSingleRegUse(Sdst, MRI);
  if (!NextOp)
    return;
  MachineInstr &MISucc = *NextOp->getParent();

  // Make sure the carry in/out are subsequently unused.
  MachineOperand *CarryIn = TII->getNamedOperand(MISucc, AMDGPU::OpName::src2);
  if (!CarryIn)
    return;
  MachineOperand *CarryOut = TII->getNamedOperand(MISucc, AMDGPU::OpName::sdst);
  if (!CarryOut)
    return;
  if (!MRI->hasOneUse(CarryIn->getReg()) || !MRI->use_empty(CarryOut->getReg()))
    return;
  // Make sure VCC or its subregs are dead before MI.
  MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::LivenessQueryResult Liveness =
      MBB.computeRegisterLiveness(TRI, AMDGPU::VCC, MI, 25);
  if (Liveness != MachineBasicBlock::LQR_Dead)
    return;
  // Check if VCC is referenced in range of (MI,MISucc].
  for (auto I = std::next(MI.getIterator()), E = MISucc.getIterator();
       I != E; ++I) {
    if (I->modifiesRegister(AMDGPU::VCC, TRI))
      return;
  }

  // Replace MI with V_{SUB|ADD}_I32_e32
  BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(Opc))
    .add(*TII->getNamedOperand(MI, AMDGPU::OpName::vdst))
    .add(*TII->getNamedOperand(MI, AMDGPU::OpName::src0))
    .add(*TII->getNamedOperand(MI, AMDGPU::OpName::src1))
    .setMIFlags(MI.getFlags());

  MI.eraseFromParent();

  // Since the carry output of MI is now VCC, update its use in MISucc.

  MISucc.substituteRegister(CarryIn->getReg(), TRI->getVCC(), 0, *TRI);
}

/// Try to convert an \p MI in VOP3 which takes an src2 carry-in
/// operand into the corresponding VOP2 form which expects the
/// argument in VCC. To this end, add an copy from the carry-in to
/// VCC.  The conversion will only be applied if \p MI can be shrunk
/// to VOP2 and if VCC can be proven to be dead before \p MI.
void SIPeepholeSDWA::convertVcndmaskToVOP2(MachineInstr &MI,
                                           const GCNSubtarget &ST) const {
  assert(MI.getOpcode() == AMDGPU::V_CNDMASK_B32_e64);

  LLVM_DEBUG(dbgs() << "Attempting VOP2 conversion: " << MI);
  if (!TII->canShrink(MI, *MRI)) {
    LLVM_DEBUG(dbgs() << "Cannot shrink instruction\n");
    return;
  }

  const MachineOperand &CarryIn =
      *TII->getNamedOperand(MI, AMDGPU::OpName::src2);
  Register CarryReg = CarryIn.getReg();
  MachineInstr *CarryDef = MRI->getVRegDef(CarryReg);
  if (!CarryDef) {
    LLVM_DEBUG(dbgs() << "Missing carry-in operand definition\n");
    return;
  }

  // Make sure VCC or its subregs are dead before MI.
  MCRegister Vcc = TRI->getVCC();
  MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::LivenessQueryResult Liveness =
      MBB.computeRegisterLiveness(TRI, Vcc, MI);
  if (Liveness != MachineBasicBlock::LQR_Dead) {
    LLVM_DEBUG(dbgs() << "VCC not known to be dead before instruction\n");
    return;
  }

  BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY), Vcc).add(CarryIn);

  auto Converted = BuildMI(MBB, MI, MI.getDebugLoc(),
                           TII->get(AMDGPU::getVOPe32(MI.getOpcode())))
                       .add(*TII->getNamedOperand(MI, AMDGPU::OpName::vdst))
                       .add(*TII->getNamedOperand(MI, AMDGPU::OpName::src0))
                       .add(*TII->getNamedOperand(MI, AMDGPU::OpName::src1))
                       .setMIFlags(MI.getFlags());
  TII->fixImplicitOperands(*Converted);
  LLVM_DEBUG(dbgs() << "Converted to VOP2: " << *Converted);
  (void)Converted;
  MI.eraseFromParent();
}

namespace {
bool isConvertibleToSDWA(MachineInstr &MI,
                         const GCNSubtarget &ST,
                         const SIInstrInfo* TII) {
  // Check if this is already an SDWA instruction
  unsigned Opc = MI.getOpcode();
  if (TII->isSDWA(Opc))
    return true;

  // Can only be handled after ealier conversion to
  // AMDGPU::V_CNDMASK_B32_e32 which is not always possible.
  if (Opc == AMDGPU::V_CNDMASK_B32_e64)
    return false;

  // Check if this instruction has opcode that supports SDWA
  if (AMDGPU::getSDWAOp(Opc) == -1)
    Opc = AMDGPU::getVOPe32(Opc);

  if (AMDGPU::getSDWAOp(Opc) == -1)
    return false;

  if (!ST.hasSDWAOmod() && TII->hasModifiersSet(MI, AMDGPU::OpName::omod))
    return false;

  if (TII->isVOPC(Opc)) {
    if (!ST.hasSDWASdst()) {
      const MachineOperand *SDst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst);
      if (SDst && (SDst->getReg() != AMDGPU::VCC &&
                   SDst->getReg() != AMDGPU::VCC_LO))
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

  if (!ST.hasSDWAMac() && (Opc == AMDGPU::V_FMAC_F16_e32 ||
                           Opc == AMDGPU::V_FMAC_F32_e32 ||
                           Opc == AMDGPU::V_MAC_F16_e32 ||
                           Opc == AMDGPU::V_MAC_F32_e32))
    return false;

  // Check if target supports this SDWA opcode
  if (TII->pseudoToMCOpcode(Opc) == -1)
    return false;

  if (MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0)) {
    if (!Src0->isReg() && !Src0->isImm())
      return false;
  }

  if (MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1)) {
    if (!Src1->isReg() && !Src1->isImm())
      return false;
  }

  return true;
}
} // namespace

MachineInstr *SIPeepholeSDWA::createSDWAVersion(MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  assert(!TII->isSDWA(Opcode));

  int SDWAOpcode = AMDGPU::getSDWAOp(Opcode);
  if (SDWAOpcode == -1)
    SDWAOpcode = AMDGPU::getSDWAOp(AMDGPU::getVOPe32(Opcode));
  assert(SDWAOpcode != -1);

  const MCInstrDesc &SDWADesc = TII->get(SDWAOpcode);

  // Create SDWA version of instruction MI and initialize its operands
  MachineInstrBuilder SDWAInst =
    BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), SDWADesc)
    .setMIFlags(MI.getFlags());

  // Copy dst, if it is present in original then should also be present in SDWA
  MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
  if (Dst) {
    assert(AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::vdst));
    SDWAInst.add(*Dst);
  } else if ((Dst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst))) {
    assert(Dst && AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::sdst));
    SDWAInst.add(*Dst);
  } else {
    assert(AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::sdst));
    SDWAInst.addReg(TRI->getVCC(), RegState::Define);
  }

  // Copy src0, initialize src0_modifiers. All sdwa instructions has src0 and
  // src0_modifiers (except for v_nop_sdwa, but it can't get here)
  MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  assert(Src0 && AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::src0) &&
         AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::src0_modifiers));
  if (auto *Mod = TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers))
    SDWAInst.addImm(Mod->getImm());
  else
    SDWAInst.addImm(0);
  SDWAInst.add(*Src0);

  // Copy src1 if present, initialize src1_modifiers.
  MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  if (Src1) {
    assert(AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::src1) &&
           AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::src1_modifiers));
    if (auto *Mod = TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers))
      SDWAInst.addImm(Mod->getImm());
    else
      SDWAInst.addImm(0);
    SDWAInst.add(*Src1);
  }

  if (SDWAOpcode == AMDGPU::V_FMAC_F16_sdwa ||
      SDWAOpcode == AMDGPU::V_FMAC_F32_sdwa ||
      SDWAOpcode == AMDGPU::V_MAC_F16_sdwa ||
      SDWAOpcode == AMDGPU::V_MAC_F32_sdwa) {
    // v_mac_f16/32 has additional src2 operand tied to vdst
    MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
    assert(Src2);
    SDWAInst.add(*Src2);
  }

  // Copy clamp if present, initialize otherwise
  assert(AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::clamp));
  MachineOperand *Clamp = TII->getNamedOperand(MI, AMDGPU::OpName::clamp);
  if (Clamp) {
    SDWAInst.add(*Clamp);
  } else {
    SDWAInst.addImm(0);
  }

  // Copy omod if present, initialize otherwise if needed
  if (AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::omod)) {
    MachineOperand *OMod = TII->getNamedOperand(MI, AMDGPU::OpName::omod);
    if (OMod) {
      SDWAInst.add(*OMod);
    } else {
      SDWAInst.addImm(0);
    }
  }

  // Initialize SDWA specific operands
  if (AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::dst_sel))
    SDWAInst.addImm(AMDGPU::SDWA::SdwaSel::DWORD);

  if (AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::dst_unused))
    SDWAInst.addImm(AMDGPU::SDWA::DstUnused::UNUSED_PAD);

  assert(AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::src0_sel));
  SDWAInst.addImm(AMDGPU::SDWA::SdwaSel::DWORD);

  if (Src1) {
    assert(AMDGPU::hasNamedOperand(SDWAOpcode, AMDGPU::OpName::src1_sel));
    SDWAInst.addImm(AMDGPU::SDWA::SdwaSel::DWORD);
  }

  // Check for a preserved register that needs to be copied.
  MachineInstr *Ret = SDWAInst.getInstr();
  TII->fixImplicitOperands(*Ret);
  return Ret;
}

bool SIPeepholeSDWA::convertToSDWA(MachineInstr &MI,
                                   const SDWAOperandsVector &SDWAOperands) {
  LLVM_DEBUG(dbgs() << "Convert instruction:" << MI);

  MachineInstr *SDWAInst;
  if (TII->isSDWA(MI.getOpcode())) {
    // Clone the instruction to allow revoking changes
    // made to MI during the processing of the operands
    // if the conversion fails.
    SDWAInst = MI.getParent()->getParent()->CloneMachineInstr(&MI);
    MI.getParent()->insert(MI.getIterator(), SDWAInst);
  } else {
    SDWAInst = createSDWAVersion(MI);
  }

  // Apply all sdwa operand patterns.
  bool Converted = false;
  for (auto &Operand : SDWAOperands) {
    LLVM_DEBUG(dbgs() << *SDWAInst << "\nOperand: " << *Operand);
    // There should be no intersection between SDWA operands and potential MIs
    // e.g.:
    // v_and_b32 v0, 0xff, v1 -> src:v1 sel:BYTE_0
    // v_and_b32 v2, 0xff, v0 -> src:v0 sel:BYTE_0
    // v_add_u32 v3, v4, v2
    //
    // In that example it is possible that we would fold 2nd instruction into
    // 3rd (v_add_u32_sdwa) and then try to fold 1st instruction into 2nd (that
    // was already destroyed). So if SDWAOperand is also a potential MI then do
    // not apply it.
    if (PotentialMatches.count(Operand->getParentInst()) == 0)
      Converted |= Operand->convertToSDWA(*SDWAInst, TII);
  }

  if (!Converted) {
    SDWAInst->eraseFromParent();
    return false;
  }

  ConvertedInstructions.push_back(SDWAInst);
  for (MachineOperand &MO : SDWAInst->uses()) {
    if (!MO.isReg())
      continue;

    MRI->clearKillFlags(MO.getReg());
  }
  LLVM_DEBUG(dbgs() << "\nInto:" << *SDWAInst << '\n');
  ++NumSDWAInstructionsPeepholed;

  MI.eraseFromParent();
  return true;
}

// If an instruction was converted to SDWA it should not have immediates or SGPR
// operands (allowed one SGPR on GFX9). Copy its scalar operands into VGPRs.
void SIPeepholeSDWA::legalizeScalarOperands(MachineInstr &MI,
                                            const GCNSubtarget &ST) const {
  const MCInstrDesc &Desc = TII->get(MI.getOpcode());
  unsigned ConstantBusCount = 0;
  for (MachineOperand &Op : MI.explicit_uses()) {
    if (!Op.isImm() && !(Op.isReg() && !TRI->isVGPR(*MRI, Op.getReg())))
      continue;

    unsigned I = Op.getOperandNo();
    if (Desc.operands()[I].RegClass == -1 ||
        !TRI->isVSSuperClass(TRI->getRegClass(Desc.operands()[I].RegClass)))
      continue;

    if (ST.hasSDWAScalar() && ConstantBusCount == 0 && Op.isReg() &&
        TRI->isSGPRReg(*MRI, Op.getReg())) {
      ++ConstantBusCount;
      continue;
    }

    Register VGPR = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
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

bool SIPeepholeSDWALegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return SIPeepholeSDWA().run(MF);
}

bool SIPeepholeSDWA::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasSDWA())
    return false;

  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();

  // Find all SDWA operands in MF.
  bool Ret = false;
  for (MachineBasicBlock &MBB : MF) {
    bool Changed = false;
    do {
      // Preprocess the ADD/SUB pairs so they could be SDWA'ed.
      // Look for a possible ADD or SUB that resulted from a previously lowered
      // V_{ADD|SUB}_U64_PSEUDO. The function pseudoOpConvertToVOP2
      // lowers the pair of instructions into e32 form.
      matchSDWAOperands(MBB);
      for (const auto &OperandPair : SDWAOperands) {
        const auto &Operand = OperandPair.second;
        MachineInstr *PotentialMI = Operand->potentialToConvert(TII, ST);
        if (!PotentialMI)
          continue;

        switch (PotentialMI->getOpcode()) {
        case AMDGPU::V_ADD_CO_U32_e64:
        case AMDGPU::V_SUB_CO_U32_e64:
          pseudoOpConvertToVOP2(*PotentialMI, ST);
          break;
        case AMDGPU::V_CNDMASK_B32_e64:
          convertVcndmaskToVOP2(*PotentialMI, ST);
          break;
        };
      }
      SDWAOperands.clear();

      // Generate potential match list.
      matchSDWAOperands(MBB);

      for (const auto &OperandPair : SDWAOperands) {
        const auto &Operand = OperandPair.second;
        MachineInstr *PotentialMI =
            Operand->potentialToConvert(TII, ST, &PotentialMatches);

        if (PotentialMI && isConvertibleToSDWA(*PotentialMI, ST, TII))
          PotentialMatches[PotentialMI].push_back(Operand.get());
      }

      for (auto &PotentialPair : PotentialMatches) {
        MachineInstr &PotentialMI = *PotentialPair.first;
        convertToSDWA(PotentialMI, PotentialPair.second);
      }

      PotentialMatches.clear();
      SDWAOperands.clear();

      Changed = !ConvertedInstructions.empty();

      if (Changed)
        Ret = true;
      while (!ConvertedInstructions.empty())
        legalizeScalarOperands(*ConvertedInstructions.pop_back_val(), ST);
    } while (Changed);
  }

  return Ret;
}

PreservedAnalyses SIPeepholeSDWAPass::run(MachineFunction &MF,
                                          MachineFunctionAnalysisManager &) {
  if (MF.getFunction().hasOptNone() || !SIPeepholeSDWA().run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
