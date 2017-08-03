//===----- HexagonMCChecker.cpp - Instruction bundle checking -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the checking of insns inside a bundle according to the
// packet constraint rules of the Hexagon ISA.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCChecker.h"

#include "HexagonBaseInfo.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
    RelaxNVChecks("relax-nv-checks", cl::init(false), cl::ZeroOrMore,
                  cl::Hidden, cl::desc("Relax checks of new-value validity"));

const HexagonMCChecker::PredSense
    HexagonMCChecker::Unconditional(Hexagon::NoRegister, false);

void HexagonMCChecker::init() {
  // Initialize read-only registers set.
  ReadOnly.insert(Hexagon::PC);
  ReadOnly.insert(Hexagon::C9_8);

  // Figure out the loop-registers definitions.
  if (HexagonMCInstrInfo::isInnerLoop(MCB)) {
    Defs[Hexagon::SA0].insert(Unconditional); // FIXME: define or change SA0?
    Defs[Hexagon::LC0].insert(Unconditional);
  }
  if (HexagonMCInstrInfo::isOuterLoop(MCB)) {
    Defs[Hexagon::SA1].insert(Unconditional); // FIXME: define or change SA0?
    Defs[Hexagon::LC1].insert(Unconditional);
  }

  if (HexagonMCInstrInfo::isBundle(MCB))
    // Unfurl a bundle.
    for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCB)) {
      MCInst const &Inst = *I.getInst();
      if (HexagonMCInstrInfo::isDuplex(MCII, Inst)) {
        init(*Inst.getOperand(0).getInst());
        init(*Inst.getOperand(1).getInst());
      } else
        init(Inst);
    }
  else
    init(MCB);
}

void HexagonMCChecker::initReg(MCInst const &MCI, unsigned R, unsigned &PredReg,
                               bool &isTrue) {
  if (HexagonMCInstrInfo::isPredicated(MCII, MCI) && isPredicateRegister(R)) {
    // Note an used predicate register.
    PredReg = R;
    isTrue = HexagonMCInstrInfo::isPredicatedTrue(MCII, MCI);

    // Note use of new predicate register.
    if (HexagonMCInstrInfo::isPredicatedNew(MCII, MCI))
      NewPreds.insert(PredReg);
  } else
    // Note register use.  Super-registers are not tracked directly,
    // but their components.
    for (MCRegAliasIterator SRI(R, &RI, !MCSubRegIterator(R, &RI).isValid());
         SRI.isValid(); ++SRI)
      if (!MCSubRegIterator(*SRI, &RI).isValid())
        // Skip super-registers used indirectly.
        Uses.insert(*SRI);
}

void HexagonMCChecker::init(MCInst const &MCI) {
  const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(MCII, MCI);
  unsigned PredReg = Hexagon::NoRegister;
  bool isTrue = false;

  // Get used registers.
  for (unsigned i = MCID.getNumDefs(); i < MCID.getNumOperands(); ++i)
    if (MCI.getOperand(i).isReg())
      initReg(MCI, MCI.getOperand(i).getReg(), PredReg, isTrue);
  for (unsigned i = 0; i < MCID.getNumImplicitUses(); ++i)
    initReg(MCI, MCID.getImplicitUses()[i], PredReg, isTrue);

  // Get implicit register definitions.
  if (const MCPhysReg *ImpDef = MCID.getImplicitDefs())
    for (; *ImpDef; ++ImpDef) {
      unsigned R = *ImpDef;

      if (Hexagon::R31 != R && MCID.isCall())
        // Any register other than the LR and the PC are actually volatile ones
        // as defined by the ABI, not modified implicitly by the call insn.
        continue;
      if (Hexagon::PC == R)
        // Branches are the only insns that can change the PC,
        // otherwise a read-only register.
        continue;

      if (Hexagon::USR_OVF == R)
        // Many insns change the USR implicitly, but only one or another flag.
        // The instruction table models the USR.OVF flag, which can be
        // implicitly modified more than once, but cannot be modified in the
        // same packet with an instruction that modifies is explicitly. Deal
        // with such situations individually.
        SoftDefs.insert(R);
      else if (isPredicateRegister(R) &&
               HexagonMCInstrInfo::isPredicateLate(MCII, MCI))
        // Include implicit late predicates.
        LatePreds.insert(R);
      else
        Defs[R].insert(PredSense(PredReg, isTrue));
    }

  // Figure out explicit register definitions.
  for (unsigned i = 0; i < MCID.getNumDefs(); ++i) {
    unsigned R = MCI.getOperand(i).getReg(), S = Hexagon::NoRegister;
    // USR has subregisters (while C8 does not for technical reasons), so
    // reset R to USR, since we know how to handle multiple defs of USR,
    // taking into account its subregisters.
    if (R == Hexagon::C8)
      R = Hexagon::USR;

    // Note register definitions, direct ones as well as indirect side-effects.
    // Super-registers are not tracked directly, but their components.
    for (MCRegAliasIterator SRI(R, &RI, !MCSubRegIterator(R, &RI).isValid());
         SRI.isValid(); ++SRI) {
      if (MCSubRegIterator(*SRI, &RI).isValid())
        // Skip super-registers defined indirectly.
        continue;

      if (R == *SRI) {
        if (S == R)
          // Avoid scoring the defined register multiple times.
          continue;
        else
          // Note that the defined register has already been scored.
          S = R;
      }

      if (Hexagon::P3_0 != R && Hexagon::P3_0 == *SRI)
        // P3:0 is a special case, since multiple predicate register definitions
        // in a packet is allowed as the equivalent of their logical "and".
        // Only an explicit definition of P3:0 is noted as such; if a
        // side-effect, then note as a soft definition.
        SoftDefs.insert(*SRI);
      else if (HexagonMCInstrInfo::isPredicateLate(MCII, MCI) &&
               isPredicateRegister(*SRI))
        // Some insns produce predicates too late to be used in the same packet.
        LatePreds.insert(*SRI);
      else if (i == 0 && llvm::HexagonMCInstrInfo::getType(MCII, MCI) ==
                             HexagonII::TypeCVI_VM_TMP_LD)
        // Temporary loads should be used in the same packet, but don't commit
        // results, so it should be disregarded if another insn changes the same
        // register.
        // TODO: relies on the impossibility of a current and a temporary loads
        // in the same packet.
        TmpDefs.insert(*SRI);
      else if (i <= 1 && llvm::HexagonMCInstrInfo::hasNewValue2(MCII, MCI))
        // vshuff(Vx, Vy, Rx) <- Vx(0) and Vy(1) are both source and
        // destination registers with this instruction. same for vdeal(Vx,Vy,Rx)
        Uses.insert(*SRI);
      else
        Defs[*SRI].insert(PredSense(PredReg, isTrue));
    }
  }

  // Figure out register definitions that produce new values.
  if (HexagonMCInstrInfo::hasNewValue(MCII, MCI)) {
    unsigned R = HexagonMCInstrInfo::getNewValueOperand(MCII, MCI).getReg();

    if (HexagonMCInstrInfo::isCompound(MCII, MCI))
      compoundRegisterMap(R); // Compound insns have a limited register range.

    for (MCRegAliasIterator SRI(R, &RI, !MCSubRegIterator(R, &RI).isValid());
         SRI.isValid(); ++SRI)
      if (!MCSubRegIterator(*SRI, &RI).isValid())
        // No super-registers defined indirectly.
        NewDefs[*SRI].push_back(NewSense::Def(
            PredReg, HexagonMCInstrInfo::isPredicatedTrue(MCII, MCI),
            HexagonMCInstrInfo::isFloat(MCII, MCI)));

    // For fairly unique 2-dot-new producers, example:
    // vdeal(V1, V9, R0) V1.new and V9.new can be used by consumers.
    if (HexagonMCInstrInfo::hasNewValue2(MCII, MCI)) {
      unsigned R2 = HexagonMCInstrInfo::getNewValueOperand2(MCII, MCI).getReg();

      bool HasSubRegs = MCSubRegIterator(R2, &RI).isValid();
      for (MCRegAliasIterator SRI(R2, &RI, !HasSubRegs); SRI.isValid(); ++SRI)
        if (!MCSubRegIterator(*SRI, &RI).isValid())
          NewDefs[*SRI].push_back(NewSense::Def(
              PredReg, HexagonMCInstrInfo::isPredicatedTrue(MCII, MCI),
              HexagonMCInstrInfo::isFloat(MCII, MCI)));
    }
  }

  // Figure out definitions of new predicate registers.
  if (HexagonMCInstrInfo::isPredicatedNew(MCII, MCI))
    for (unsigned i = MCID.getNumDefs(); i < MCID.getNumOperands(); ++i)
      if (MCI.getOperand(i).isReg()) {
        unsigned P = MCI.getOperand(i).getReg();

        if (isPredicateRegister(P))
          NewPreds.insert(P);
      }

  // Figure out uses of new values.
  if (HexagonMCInstrInfo::isNewValue(MCII, MCI)) {
    unsigned N = HexagonMCInstrInfo::getNewValueOperand(MCII, MCI).getReg();

    if (!MCSubRegIterator(N, &RI).isValid()) {
      // Super-registers cannot use new values.
      if (MCID.isBranch())
        NewUses[N] = NewSense::Jmp(
            llvm::HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypeNCJ);
      else
        NewUses[N] = NewSense::Use(
            PredReg, HexagonMCInstrInfo::isPredicatedTrue(MCII, MCI));
    }
  }
}

HexagonMCChecker::HexagonMCChecker(MCContext &Context, MCInstrInfo const &MCII,
                                   MCSubtargetInfo const &STI, MCInst &mcb,
                                   MCRegisterInfo const &ri, bool ReportErrors)
    : Context(Context), MCB(mcb), RI(ri), MCII(MCII), STI(STI),
      ReportErrors(ReportErrors) {
  init();
}

bool HexagonMCChecker::check(bool FullCheck) {
  bool chkB = checkBranches();
  bool chkP = checkPredicates();
  bool chkNV = checkNewValues();
  bool chkR = checkRegisters();
  bool chkRRO = checkRegistersReadOnly();
  bool chkELB = checkEndloopBranches();
  checkRegisterCurDefs();
  bool chkS = checkSolo();
  bool chkSh = true;
  if (FullCheck)
    chkSh = checkShuffle();
  bool chkSl = true;
  if (FullCheck)
    chkSl = checkSlots();
  bool chkAXOK = checkAXOK();
  bool chk = chkB && chkP && chkNV && chkR && chkRRO && chkELB && chkS &&
             chkSh && chkSl && chkAXOK;

  return chk;
}

bool HexagonMCChecker::checkEndloopBranches() {
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCII, MCB)) {
    MCInstrDesc const &Desc = HexagonMCInstrInfo::getDesc(MCII, I);
    if (Desc.isBranch() || Desc.isCall()) {
      auto Inner = HexagonMCInstrInfo::isInnerLoop(MCB);
      if (Inner || HexagonMCInstrInfo::isOuterLoop(MCB)) {
        reportError(I.getLoc(),
                    llvm::Twine("packet marked with `:endloop") +
                        (Inner ? "0" : "1") + "' " +
                        "cannot contain instructions that modify register " +
                        "`" + llvm::Twine(RI.getName(Hexagon::PC)) + "'");
        return false;
      }
    }
  }
  return true;
}

namespace {
bool isDuplexAGroup(unsigned Opcode) {
  switch (Opcode) {
  case Hexagon::SA1_addi:
  case Hexagon::SA1_addrx:
  case Hexagon::SA1_addsp:
  case Hexagon::SA1_and1:
  case Hexagon::SA1_clrf:
  case Hexagon::SA1_clrfnew:
  case Hexagon::SA1_clrt:
  case Hexagon::SA1_clrtnew:
  case Hexagon::SA1_cmpeqi:
  case Hexagon::SA1_combine0i:
  case Hexagon::SA1_combine1i:
  case Hexagon::SA1_combine2i:
  case Hexagon::SA1_combine3i:
  case Hexagon::SA1_combinerz:
  case Hexagon::SA1_combinezr:
  case Hexagon::SA1_dec:
  case Hexagon::SA1_inc:
  case Hexagon::SA1_seti:
  case Hexagon::SA1_setin1:
  case Hexagon::SA1_sxtb:
  case Hexagon::SA1_sxth:
  case Hexagon::SA1_tfr:
  case Hexagon::SA1_zxtb:
  case Hexagon::SA1_zxth:
    return true;
    break;
  default:
    return false;
  }
}

bool isNeitherAnorX(MCInstrInfo const &MCII, MCInst const &ID) {
  unsigned Result = 0;
  unsigned Type = HexagonMCInstrInfo::getType(MCII, ID);
  if (Type == HexagonII::TypeDUPLEX) {
    unsigned subInst0Opcode = ID.getOperand(0).getInst()->getOpcode();
    unsigned subInst1Opcode = ID.getOperand(1).getInst()->getOpcode();
    Result += !isDuplexAGroup(subInst0Opcode);
    Result += !isDuplexAGroup(subInst1Opcode);
  } else
    Result +=
        Type != HexagonII::TypeALU32_2op && Type != HexagonII::TypeALU32_3op &&
        Type != HexagonII::TypeALU32_ADDI && Type != HexagonII::TypeS_2op &&
        Type != HexagonII::TypeS_3op &&
        (Type != HexagonII::TypeALU64 || HexagonMCInstrInfo::isFloat(MCII, ID));
  return Result != 0;
}
} // namespace

bool HexagonMCChecker::checkAXOK() {
  MCInst const *HasSoloAXInst = nullptr;
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCII, MCB)) {
    if (HexagonMCInstrInfo::isSoloAX(MCII, I)) {
      HasSoloAXInst = &I;
    }
  }
  if (!HasSoloAXInst)
    return true;
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCII, MCB)) {
    if (&I != HasSoloAXInst && isNeitherAnorX(MCII, I)) {
      reportError(
          HasSoloAXInst->getLoc(),
          llvm::Twine("Instruction can only be in a packet with ALU or "
                      "non-FPU XTYPE instructions"));
      reportError(I.getLoc(),
                  llvm::Twine("Not an ALU or non-FPU XTYPE instruction"));
      return false;
    }
  }
  return true;
}

bool HexagonMCChecker::checkSlots() {
  unsigned slotsUsed = 0;
  for (auto HMI : HexagonMCInstrInfo::bundleInstructions(MCB)) {
    MCInst const &MCI = *HMI.getInst();
    if (HexagonMCInstrInfo::isImmext(MCI))
      continue;
    if (HexagonMCInstrInfo::isDuplex(MCII, MCI))
      slotsUsed += 2;
    else
      ++slotsUsed;
  }

  if (slotsUsed > HEXAGON_PACKET_SIZE) {
    reportError("invalid instruction packet: out of slots");
    return false;
  }
  return true;
}

// Check legal use of branches.
bool HexagonMCChecker::checkBranches() {
  if (HexagonMCInstrInfo::isBundle(MCB)) {
    bool hasConditional = false;
    unsigned Branches = 0, Conditional = HEXAGON_PRESHUFFLE_PACKET_SIZE,
             Unconditional = HEXAGON_PRESHUFFLE_PACKET_SIZE;

    for (unsigned i = HexagonMCInstrInfo::bundleInstructionsOffset;
         i < MCB.size(); ++i) {
      MCInst const &MCI = *MCB.begin()[i].getInst();

      if (HexagonMCInstrInfo::isImmext(MCI))
        continue;
      if (HexagonMCInstrInfo::getDesc(MCII, MCI).isBranch() ||
          HexagonMCInstrInfo::getDesc(MCII, MCI).isCall()) {
        ++Branches;
        if (HexagonMCInstrInfo::isPredicated(MCII, MCI) ||
            HexagonMCInstrInfo::isPredicatedNew(MCII, MCI)) {
          hasConditional = true;
          Conditional = i; // Record the position of the conditional branch.
        } else {
          Unconditional = i; // Record the position of the unconditional branch.
        }
      }
    }

    if (Branches > 1)
      if (!hasConditional || Conditional > Unconditional) {
        // Error out if more than one unconditional branch or
        // the conditional branch appears after the unconditional one.
        reportError(
            "unconditional branch cannot precede another branch in packet");
        return false;
      }
  }

  return true;
}

// Check legal use of predicate registers.
bool HexagonMCChecker::checkPredicates() {
  // Check for proper use of new predicate registers.
  for (const auto &I : NewPreds) {
    unsigned P = I;

    if (!Defs.count(P) || LatePreds.count(P)) {
      // Error out if the new predicate register is not defined,
      // or defined "late"
      // (e.g., "{ if (p3.new)... ; p3 = sp1loop0(#r7:2, Rs) }").
      reportErrorNewValue(P);
      return false;
    }
  }

  // Check for proper use of auto-anded of predicate registers.
  for (const auto &I : LatePreds) {
    unsigned P = I;

    if (LatePreds.count(P) > 1 || Defs.count(P)) {
      // Error out if predicate register defined "late" multiple times or
      // defined late and regularly defined
      // (e.g., "{ p3 = sp1loop0(...); p3 = cmp.eq(...) }".
      reportErrorRegisters(P);
      return false;
    }
  }

  return true;
}

// Check legal use of new values.
bool HexagonMCChecker::checkNewValues() {
  for (auto &I : NewUses) {
    unsigned R = I.first;
    NewSense &US = I.second;

    if (!hasValidNewValueDef(US, NewDefs[R])) {
      reportErrorNewValue(R);
      return false;
    }
  }

  return true;
}

bool HexagonMCChecker::checkRegistersReadOnly() {
  for (auto I : HexagonMCInstrInfo::bundleInstructions(MCB)) {
    MCInst const &Inst = *I.getInst();
    unsigned Defs = HexagonMCInstrInfo::getDesc(MCII, Inst).getNumDefs();
    for (unsigned j = 0; j < Defs; ++j) {
      MCOperand const &Operand = Inst.getOperand(j);
      assert(Operand.isReg() && "Def is not a register");
      unsigned Register = Operand.getReg();
      if (ReadOnly.find(Register) != ReadOnly.end()) {
        reportError(Inst.getLoc(), "Cannot write to read-only register `" +
                                       llvm::Twine(RI.getName(Register)) + "'");
        return false;
      }
    }
  }
  return true;
}

bool HexagonMCChecker::registerUsed(unsigned Register) {
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCII, MCB))
    for (unsigned j = HexagonMCInstrInfo::getDesc(MCII, I).getNumDefs(),
                  n = I.getNumOperands();
         j < n; ++j) {
      MCOperand const &Operand = I.getOperand(j);
      if (Operand.isReg() && Operand.getReg() == Register)
        return true;
    }
  return false;
}

void HexagonMCChecker::checkRegisterCurDefs() {
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCII, MCB)) {
    if (HexagonMCInstrInfo::isCVINew(MCII, I) &&
        HexagonMCInstrInfo::getDesc(MCII, I).mayLoad()) {
      unsigned Register = I.getOperand(0).getReg();
      if (!registerUsed(Register))
        reportWarning("Register `" + llvm::Twine(RI.getName(Register)) +
                      "' used with `.cur' "
                      "but not used in the same packet");
    }
  }
}

// Check for legal register uses and definitions.
bool HexagonMCChecker::checkRegisters() {
  // Check for proper register definitions.
  for (const auto &I : Defs) {
    unsigned R = I.first;

    if (isLoopRegister(R) && Defs.count(R) > 1 &&
        (HexagonMCInstrInfo::isInnerLoop(MCB) ||
         HexagonMCInstrInfo::isOuterLoop(MCB))) {
      // Error out for definitions of loop registers at the end of a loop.
      reportError("loop-setup and some branch instructions "
                  "cannot be in the same packet");
      return false;
    }
    if (SoftDefs.count(R)) {
      // Error out for explicit changes to registers also weakly defined
      // (e.g., "{ usr = r0; r0 = sfadd(...) }").
      unsigned UsrR = Hexagon::USR; // Silence warning about mixed types in ?:.
      unsigned BadR = RI.isSubRegister(Hexagon::USR, R) ? UsrR : R;
      reportErrorRegisters(BadR);
      return false;
    }
    if (!isPredicateRegister(R) && Defs[R].size() > 1) {
      // Check for multiple register definitions.
      PredSet &PM = Defs[R];

      // Check for multiple unconditional register definitions.
      if (PM.count(Unconditional)) {
        // Error out on an unconditional change when there are any other
        // changes, conditional or not.
        unsigned UsrR = Hexagon::USR;
        unsigned BadR = RI.isSubRegister(Hexagon::USR, R) ? UsrR : R;
        reportErrorRegisters(BadR);
        return false;
      }
      // Check for multiple conditional register definitions.
      for (const auto &J : PM) {
        PredSense P = J;

        // Check for multiple uses of the same condition.
        if (PM.count(P) > 1) {
          // Error out on conditional changes based on the same predicate
          // (e.g., "{ if (!p0) r0 =...; if (!p0) r0 =... }").
          reportErrorRegisters(R);
          return false;
        }
        // Check for the use of the complementary condition.
        P.second = !P.second;
        if (PM.count(P) && PM.size() > 2) {
          // Error out on conditional changes based on the same predicate
          // multiple times
          // (e.g., "if (p0) r0 =...; if (!p0) r0 =... }; if (!p0) r0 =...").
          reportErrorRegisters(R);
          return false;
        }
      }
    }
  }

  // Check for use of temporary definitions.
  for (const auto &I : TmpDefs) {
    unsigned R = I;

    if (!Uses.count(R)) {
      // special case for vhist
      bool vHistFound = false;
      for (auto const &HMI : HexagonMCInstrInfo::bundleInstructions(MCB)) {
        if (llvm::HexagonMCInstrInfo::getType(MCII, *HMI.getInst()) ==
            HexagonII::TypeCVI_HIST) {
          vHistFound = true; // vhist() implicitly uses ALL REGxx.tmp
          break;
        }
      }
      // Warn on an unused temporary definition.
      if (vHistFound == false) {
        reportWarning("register `" + llvm::Twine(RI.getName(R)) +
                      "' used with `.tmp' "
                      "but not used in the same packet");
        return true;
      }
    }
  }

  return true;
}

// Check for legal use of solo insns.
bool HexagonMCChecker::checkSolo() {
  if (HexagonMCInstrInfo::bundleSize(MCB) > 1)
    for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCII, MCB)) {
      if (llvm::HexagonMCInstrInfo::isSolo(MCII, I)) {
        reportError(I.getLoc(), "Instruction is marked `isSolo' and "
                                "cannot have other instructions in "
                                "the same packet");
        return false;
      }
    }

  return true;
}

bool HexagonMCChecker::checkShuffle() {
  HexagonMCShuffler MCSDX(Context, ReportErrors, MCII, STI, MCB);
  return MCSDX.check();
}

void HexagonMCChecker::compoundRegisterMap(unsigned &Register) {
  switch (Register) {
  default:
    break;
  case Hexagon::R15:
    Register = Hexagon::R23;
    break;
  case Hexagon::R14:
    Register = Hexagon::R22;
    break;
  case Hexagon::R13:
    Register = Hexagon::R21;
    break;
  case Hexagon::R12:
    Register = Hexagon::R20;
    break;
  case Hexagon::R11:
    Register = Hexagon::R19;
    break;
  case Hexagon::R10:
    Register = Hexagon::R18;
    break;
  case Hexagon::R9:
    Register = Hexagon::R17;
    break;
  case Hexagon::R8:
    Register = Hexagon::R16;
    break;
  }
}

bool HexagonMCChecker::hasValidNewValueDef(const NewSense &Use,
                                           const NewSenseList &Defs) const {
  bool Strict = !RelaxNVChecks;

  for (unsigned i = 0, n = Defs.size(); i < n; ++i) {
    const NewSense &Def = Defs[i];
    // NVJ cannot use a new FP value [7.6.1]
    if (Use.IsNVJ && (Def.IsFloat || Def.PredReg != 0))
      continue;
    // If the definition was not predicated, then it does not matter if
    // the use is.
    if (Def.PredReg == 0)
      return true;
    // With the strict checks, both the definition and the use must be
    // predicated on the same register and condition.
    if (Strict) {
      if (Def.PredReg == Use.PredReg && Def.Cond == Use.Cond)
        return true;
    } else {
      // With the relaxed checks, if the definition was predicated, the only
      // detectable violation is if the use is predicated on the opposing
      // condition, otherwise, it's ok.
      if (Def.PredReg != Use.PredReg || Def.Cond == Use.Cond)
        return true;
    }
  }
  return false;
}

void HexagonMCChecker::reportErrorRegisters(unsigned Register) {
  reportError("register `" + llvm::Twine(RI.getName(Register)) +
              "' modified more than once");
}

void HexagonMCChecker::reportErrorNewValue(unsigned Register) {
  reportError("register `" + llvm::Twine(RI.getName(Register)) +
              "' used with `.new' "
              "but not validly modified in the same packet");
}

void HexagonMCChecker::reportError(llvm::Twine const &Msg) {
  reportError(MCB.getLoc(), Msg);
}

void HexagonMCChecker::reportError(SMLoc Loc, llvm::Twine const &Msg) {
  if (ReportErrors)
    Context.reportError(Loc, Msg);
}

void HexagonMCChecker::reportWarning(llvm::Twine const &Msg) {
  if (ReportErrors) {
    auto SM = Context.getSourceManager();
    if (SM)
      SM->PrintMessage(MCB.getLoc(), SourceMgr::DK_Warning, Msg);
  }
}
