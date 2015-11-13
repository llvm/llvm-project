//===- HexagonMCInstrInfo.cpp - Hexagon sub-class of MCInst ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInstrInfo to allow Hexagon specific MCInstr queries
//
//===----------------------------------------------------------------------===//

#include "HexagonMCInstrInfo.h"

#include "Hexagon.h"
#include "HexagonBaseInfo.h"
#include "HexagonMCChecker.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
void HexagonMCInstrInfo::addConstant(MCInst &MI, uint64_t Value,
                                     MCContext &Context) {
  MI.addOperand(MCOperand::createExpr(MCConstantExpr::create(Value, Context)));
}

void HexagonMCInstrInfo::addConstExtender(MCContext &Context,
                                          MCInstrInfo const &MCII, MCInst &MCB,
                                          MCInst const &MCI) {
  assert(HexagonMCInstrInfo::isBundle(MCB));
  MCOperand const &exOp =
      MCI.getOperand(HexagonMCInstrInfo::getExtendableOp(MCII, MCI));

  // Create the extender.
  MCInst *XMCI =
      new (Context) MCInst(HexagonMCInstrInfo::deriveExtender(MCII, MCI, exOp));

  MCB.addOperand(MCOperand::createInst(XMCI));
}

iterator_range<MCInst::const_iterator>
HexagonMCInstrInfo::bundleInstructions(MCInst const &MCI) {
  assert(isBundle(MCI));
  return iterator_range<MCInst::const_iterator>(
      MCI.begin() + bundleInstructionsOffset, MCI.end());
}

size_t HexagonMCInstrInfo::bundleSize(MCInst const &MCI) {
  if (HexagonMCInstrInfo::isBundle(MCI))
    return (MCI.size() - bundleInstructionsOffset);
  else
    return (1);
}

bool HexagonMCInstrInfo::canonicalizePacket(MCInstrInfo const &MCII,
                                            MCSubtargetInfo const &STI,
                                            MCContext &Context, MCInst &MCB,
                                            HexagonMCChecker *Check) {
  // Examine the packet and convert pairs of instructions to compound
  // instructions when possible.
  if (!HexagonDisableCompound)
    HexagonMCInstrInfo::tryCompound(MCII, Context, MCB);
  // Check the bundle for errors.
  bool CheckOk = Check ? Check->check() : true;
  if (!CheckOk)
    return false;
  HexagonMCShuffle(MCII, STI, MCB);
  // Examine the packet and convert pairs of instructions to duplex
  // instructions when possible.
  MCInst InstBundlePreDuplex = MCInst(MCB);
  if (!HexagonDisableDuplex) {
    SmallVector<DuplexCandidate, 8> possibleDuplexes;
    possibleDuplexes = HexagonMCInstrInfo::getDuplexPossibilties(MCII, MCB);
    HexagonMCShuffle(MCII, STI, Context, MCB, possibleDuplexes);
  }
  // Examines packet and pad the packet, if needed, when an
  // end-loop is in the bundle.
  HexagonMCInstrInfo::padEndloop(Context, MCB);
  // If compounding and duplexing didn't reduce the size below
  // 4 or less we have a packet that is too big.
  if (HexagonMCInstrInfo::bundleSize(MCB) > HEXAGON_PACKET_SIZE)
    return false;
  HexagonMCShuffle(MCII, STI, MCB);
  return true;
}

void HexagonMCInstrInfo::clampExtended(MCInstrInfo const &MCII,
                                       MCContext &Context, MCInst &MCI) {
  assert(HexagonMCInstrInfo::isExtendable(MCII, MCI) ||
         HexagonMCInstrInfo::isExtended(MCII, MCI));
  MCOperand &exOp =
      MCI.getOperand(HexagonMCInstrInfo::getExtendableOp(MCII, MCI));
  // If the extended value is a constant, then use it for the extended and
  // for the extender instructions, masking off the lower 6 bits and
  // including the assumed bits.
  int64_t Value;
  if (exOp.getExpr()->evaluateAsAbsolute(Value)) {
    unsigned Shift = HexagonMCInstrInfo::getExtentAlignment(MCII, MCI);
    exOp.setExpr(MCConstantExpr::create((Value & 0x3f) << Shift, Context));
  }
}

MCInst HexagonMCInstrInfo::createBundle() {
  MCInst Result;
  Result.setOpcode(Hexagon::BUNDLE);
  Result.addOperand(MCOperand::createImm(0));
  return Result;
}

MCInst *HexagonMCInstrInfo::deriveDuplex(MCContext &Context, unsigned iClass,
                                         MCInst const &inst0,
                                         MCInst const &inst1) {
  assert((iClass <= 0xf) && "iClass must have range of 0 to 0xf");
  MCInst *duplexInst = new (Context) MCInst;
  duplexInst->setOpcode(Hexagon::DuplexIClass0 + iClass);

  MCInst *SubInst0 = new (Context) MCInst(deriveSubInst(inst0));
  MCInst *SubInst1 = new (Context) MCInst(deriveSubInst(inst1));
  duplexInst->addOperand(MCOperand::createInst(SubInst0));
  duplexInst->addOperand(MCOperand::createInst(SubInst1));
  return duplexInst;
}

MCInst HexagonMCInstrInfo::deriveExtender(MCInstrInfo const &MCII,
                                          MCInst const &Inst,
                                          MCOperand const &MO) {
  assert(HexagonMCInstrInfo::isExtendable(MCII, Inst) ||
         HexagonMCInstrInfo::isExtended(MCII, Inst));

  MCInstrDesc const &Desc = HexagonMCInstrInfo::getDesc(MCII, Inst);
  MCInst XMI;
  XMI.setOpcode((Desc.isBranch() || Desc.isCall() ||
                 HexagonMCInstrInfo::getType(MCII, Inst) == HexagonII::TypeCR)
                    ? Hexagon::A4_ext_b
                    : Hexagon::A4_ext);
  if (MO.isImm())
    XMI.addOperand(MCOperand::createImm(MO.getImm() & (~0x3f)));
  else if (MO.isExpr())
    XMI.addOperand(MCOperand::createExpr(MO.getExpr()));
  else
    llvm_unreachable("invalid extendable operand");
  return XMI;
}

MCInst const *HexagonMCInstrInfo::extenderForIndex(MCInst const &MCB,
                                                   size_t Index) {
  assert(Index <= bundleSize(MCB));
  if (Index == 0)
    return nullptr;
  MCInst const *Inst =
      MCB.getOperand(Index + bundleInstructionsOffset - 1).getInst();
  if (isImmext(*Inst))
    return Inst;
  return nullptr;
}

void HexagonMCInstrInfo::extendIfNeeded(MCContext &Context,
                                        MCInstrInfo const &MCII, MCInst &MCB,
                                        MCInst const &MCI, bool MustExtend) {
  if (isConstExtended(MCII, MCI) || MustExtend)
    addConstExtender(Context, MCII, MCB, MCI);
}

HexagonII::MemAccessSize
HexagonMCInstrInfo::getAccessSize(MCInstrInfo const &MCII, MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;

  return (HexagonII::MemAccessSize((F >> HexagonII::MemAccessSizePos) &
                                   HexagonII::MemAccesSizeMask));
}

unsigned HexagonMCInstrInfo::getBitCount(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask);
}

// Return constant extended operand number.
unsigned short HexagonMCInstrInfo::getCExtOpNum(MCInstrInfo const &MCII,
                                                MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

MCInstrDesc const &HexagonMCInstrInfo::getDesc(MCInstrInfo const &MCII,
                                               MCInst const &MCI) {
  return (MCII.get(MCI.getOpcode()));
}

unsigned short HexagonMCInstrInfo::getExtendableOp(MCInstrInfo const &MCII,
                                                   MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

MCOperand const &
HexagonMCInstrInfo::getExtendableOperand(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  unsigned O = HexagonMCInstrInfo::getExtendableOp(MCII, MCI);
  MCOperand const &MO = MCI.getOperand(O);

  assert((HexagonMCInstrInfo::isExtendable(MCII, MCI) ||
          HexagonMCInstrInfo::isExtended(MCII, MCI)) &&
         (MO.isImm() || MO.isExpr()));
  return (MO);
}

unsigned HexagonMCInstrInfo::getExtentAlignment(MCInstrInfo const &MCII,
                                                MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtentAlignPos) & HexagonII::ExtentAlignMask);
}

unsigned HexagonMCInstrInfo::getExtentBits(MCInstrInfo const &MCII,
                                           MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask);
}

// Return the max value that a constant extendable operand can have
// without being extended.
int HexagonMCInstrInfo::getMaxValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return ~(-1U << (bits - 1));
  else
    return ~(-1U << bits);
}

// Return the min value that a constant extendable operand can have
// without being extended.
int HexagonMCInstrInfo::getMinValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return -1U << (bits - 1);
  else
    return 0;
}

char const *HexagonMCInstrInfo::getName(MCInstrInfo const &MCII,
                                        MCInst const &MCI) {
  return MCII.getName(MCI.getOpcode());
}

unsigned short HexagonMCInstrInfo::getNewValueOp(MCInstrInfo const &MCII,
                                                 MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::NewValueOpPos) & HexagonII::NewValueOpMask);
}

MCOperand const &HexagonMCInstrInfo::getNewValueOperand(MCInstrInfo const &MCII,
                                                        MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned const O =
      (F >> HexagonII::NewValueOpPos) & HexagonII::NewValueOpMask;
  MCOperand const &MCO = MCI.getOperand(O);

  assert((HexagonMCInstrInfo::isNewValue(MCII, MCI) ||
          HexagonMCInstrInfo::hasNewValue(MCII, MCI)) &&
         MCO.isReg());
  return (MCO);
}

/// Return the new value or the newly produced value.
unsigned short HexagonMCInstrInfo::getNewValueOp2(MCInstrInfo const &MCII,
                                                  MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::NewValueOpPos2) & HexagonII::NewValueOpMask2);
}

MCOperand const &
HexagonMCInstrInfo::getNewValueOperand2(MCInstrInfo const &MCII,
                                        MCInst const &MCI) {
  unsigned O = HexagonMCInstrInfo::getNewValueOp2(MCII, MCI);
  MCOperand const &MCO = MCI.getOperand(O);

  assert((HexagonMCInstrInfo::isNewValue(MCII, MCI) ||
          HexagonMCInstrInfo::hasNewValue2(MCII, MCI)) &&
         MCO.isReg());
  return (MCO);
}

int HexagonMCInstrInfo::getSubTarget(MCInstrInfo const &MCII,
                                     MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;

  HexagonII::SubTarget Target = static_cast<HexagonII::SubTarget>(
      (F >> HexagonII::validSubTargetPos) & HexagonII::validSubTargetMask);

  switch (Target) {
  default:
    return Hexagon::ArchV4;
  case HexagonII::HasV5SubT:
    return Hexagon::ArchV5;
  }
}

// Return the Hexagon ISA class for the insn.
unsigned HexagonMCInstrInfo::getType(MCInstrInfo const &MCII,
                                     MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;

  return ((F >> HexagonII::TypePos) & HexagonII::TypeMask);
}

unsigned HexagonMCInstrInfo::getUnits(MCInstrInfo const &MCII,
                                      MCSubtargetInfo const &STI,
                                      MCInst const &MCI) {

  const InstrItinerary *II = STI.getSchedModel().InstrItineraries;
  int SchedClass = HexagonMCInstrInfo::getDesc(MCII, MCI).getSchedClass();
  return ((II[SchedClass].FirstStage + HexagonStages)->getUnits());
}

bool HexagonMCInstrInfo::hasImmExt(MCInst const &MCI) {
  if (!HexagonMCInstrInfo::isBundle(MCI))
    return false;

  for (const auto &I : HexagonMCInstrInfo::bundleInstructions(MCI)) {
    auto MI = I.getInst();
    if (isImmext(*MI))
      return true;
  }

  return false;
}

bool HexagonMCInstrInfo::hasExtenderForIndex(MCInst const &MCB, size_t Index) {
  return extenderForIndex(MCB, Index) != nullptr;
}

// Return whether the instruction is a legal new-value producer.
bool HexagonMCInstrInfo::hasNewValue(MCInstrInfo const &MCII,
                                     MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::hasNewValuePos) & HexagonII::hasNewValueMask);
}

/// Return whether the insn produces a second value.
bool HexagonMCInstrInfo::hasNewValue2(MCInstrInfo const &MCII,
                                      MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::hasNewValuePos2) & HexagonII::hasNewValueMask2);
}

MCInst const &HexagonMCInstrInfo::instruction(MCInst const &MCB, size_t Index) {
  assert(isBundle(MCB));
  assert(Index < HEXAGON_PACKET_SIZE);
  return *MCB.getOperand(bundleInstructionsOffset + Index).getInst();
}

bool HexagonMCInstrInfo::isBundle(MCInst const &MCI) {
  auto Result = Hexagon::BUNDLE == MCI.getOpcode();
  assert(!Result || (MCI.size() > 0 && MCI.getOperand(0).isImm()));
  return Result;
}

// Return whether the insn is an actual insn.
bool HexagonMCInstrInfo::isCanon(MCInstrInfo const &MCII, MCInst const &MCI) {
  return (!HexagonMCInstrInfo::getDesc(MCII, MCI).isPseudo() &&
          !HexagonMCInstrInfo::isPrefix(MCII, MCI) &&
          HexagonMCInstrInfo::getType(MCII, MCI) != HexagonII::TypeENDLOOP);
}

bool HexagonMCInstrInfo::isCompound(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  return (getType(MCII, MCI) == HexagonII::TypeCOMPOUND);
}

bool HexagonMCInstrInfo::isDblRegForSubInst(unsigned Reg) {
  return ((Reg >= Hexagon::D0 && Reg <= Hexagon::D3) ||
          (Reg >= Hexagon::D8 && Reg <= Hexagon::D11));
}

bool HexagonMCInstrInfo::isDuplex(MCInstrInfo const &MCII, MCInst const &MCI) {
  return HexagonII::TypeDUPLEX == HexagonMCInstrInfo::getType(MCII, MCI);
}

// Return whether the instruction needs to be constant extended.
// 1) Always return true if the instruction has 'isExtended' flag set.
//
// isExtendable:
// 2) For immediate extended operands, return true only if the value is
//    out-of-range.
// 3) For global address, always return true.

bool HexagonMCInstrInfo::isConstExtended(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  if (HexagonMCInstrInfo::isExtended(MCII, MCI))
    return true;
  // Branch insns are handled as necessary by relaxation.
  if ((HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypeJ) ||
      (HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypeCOMPOUND &&
       HexagonMCInstrInfo::getDesc(MCII, MCI).isBranch()) ||
      (HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypeNV &&
       HexagonMCInstrInfo::getDesc(MCII, MCI).isBranch()))
    return false;
  // Otherwise loop instructions and other CR insts are handled by relaxation
  else if ((HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypeCR) &&
           (MCI.getOpcode() != Hexagon::C4_addipc))
    return false;
  else if (!HexagonMCInstrInfo::isExtendable(MCII, MCI))
    return false;

  MCOperand const &MO = HexagonMCInstrInfo::getExtendableOperand(MCII, MCI);

  // We could be using an instruction with an extendable immediate and shoehorn
  // a global address into it. If it is a global address it will be constant
  // extended. We do this for COMBINE.
  // We currently only handle isGlobal() because it is the only kind of
  // object we are going to end up with here for now.
  // In the future we probably should add isSymbol(), etc.
  assert(!MO.isImm());
  int64_t Value;
  if (!MO.getExpr()->evaluateAsAbsolute(Value))
    return true;
  int MinValue = HexagonMCInstrInfo::getMinValue(MCII, MCI);
  int MaxValue = HexagonMCInstrInfo::getMaxValue(MCII, MCI);
  return (MinValue > Value || Value > MaxValue);
}

bool HexagonMCInstrInfo::isExtendable(MCInstrInfo const &MCII,
                                      MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (F >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask;
}

bool HexagonMCInstrInfo::isExtended(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
}

bool HexagonMCInstrInfo::isFloat(MCInstrInfo const &MCII, MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::FPPos) & HexagonII::FPMask);
}

bool HexagonMCInstrInfo::isImmext(MCInst const &MCI) {
  auto Op = MCI.getOpcode();
  return (Op == Hexagon::A4_ext_b || Op == Hexagon::A4_ext_c ||
          Op == Hexagon::A4_ext_g || Op == Hexagon::A4_ext);
}

bool HexagonMCInstrInfo::isInnerLoop(MCInst const &MCI) {
  assert(isBundle(MCI));
  int64_t Flags = MCI.getOperand(0).getImm();
  return (Flags & innerLoopMask) != 0;
}

bool HexagonMCInstrInfo::isIntReg(unsigned Reg) {
  return (Reg >= Hexagon::R0 && Reg <= Hexagon::R31);
}

bool HexagonMCInstrInfo::isIntRegForSubInst(unsigned Reg) {
  return ((Reg >= Hexagon::R0 && Reg <= Hexagon::R7) ||
          (Reg >= Hexagon::R16 && Reg <= Hexagon::R23));
}

// Return whether the insn is a new-value consumer.
bool HexagonMCInstrInfo::isNewValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::NewValuePos) & HexagonII::NewValueMask);
}

// Return whether the operand can be constant extended.
bool HexagonMCInstrInfo::isOperandExtended(MCInstrInfo const &MCII,
                                           MCInst const &MCI,
                                           unsigned short OperandNum) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask) ==
         OperandNum;
}

bool HexagonMCInstrInfo::isOuterLoop(MCInst const &MCI) {
  assert(isBundle(MCI));
  int64_t Flags = MCI.getOperand(0).getImm();
  return (Flags & outerLoopMask) != 0;
}

bool HexagonMCInstrInfo::isPredicated(MCInstrInfo const &MCII,
                                      MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool HexagonMCInstrInfo::isPredicateLate(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (F >> HexagonII::PredicateLatePos & HexagonII::PredicateLateMask);
}

/// Return whether the insn is newly predicated.
bool HexagonMCInstrInfo::isPredicatedNew(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::PredicatedNewPos) & HexagonII::PredicatedNewMask);
}

bool HexagonMCInstrInfo::isPredicatedTrue(MCInstrInfo const &MCII,
                                          MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (
      !((F >> HexagonII::PredicatedFalsePos) & HexagonII::PredicatedFalseMask));
}

bool HexagonMCInstrInfo::isPredReg(unsigned Reg) {
  return (Reg >= Hexagon::P0 && Reg <= Hexagon::P3_0);
}

bool HexagonMCInstrInfo::isPrefix(MCInstrInfo const &MCII, MCInst const &MCI) {
  return (HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypePREFIX);
}

bool HexagonMCInstrInfo::isSolo(MCInstrInfo const &MCII, MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::SoloPos) & HexagonII::SoloMask);
}

bool HexagonMCInstrInfo::isMemReorderDisabled(MCInst const &MCI) {
  assert(isBundle(MCI));
  auto Flags = MCI.getOperand(0).getImm();
  return (Flags & memReorderDisabledMask) != 0;
}

bool HexagonMCInstrInfo::isMemStoreReorderEnabled(MCInst const &MCI) {
  assert(isBundle(MCI));
  auto Flags = MCI.getOperand(0).getImm();
  return (Flags & memStoreReorderEnabledMask) != 0;
}

bool HexagonMCInstrInfo::isSoloAX(MCInstrInfo const &MCII, MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::SoloAXPos) & HexagonII::SoloAXMask);
}

bool HexagonMCInstrInfo::isSoloAin1(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::SoloAin1Pos) & HexagonII::SoloAin1Mask);
}

bool HexagonMCInstrInfo::isVector(MCInstrInfo const &MCII, MCInst const &MCI) {
  if ((getType(MCII, MCI) <= HexagonII::TypeCVI_LAST) &&
      (getType(MCII, MCI) >= HexagonII::TypeCVI_FIRST))
    return true;
  return false;
}

int64_t HexagonMCInstrInfo::minConstant(MCInst const &MCI, size_t Index) {
  auto Sentinal = static_cast<int64_t>(std::numeric_limits<uint32_t>::max())
                  << 8;
  if (MCI.size() <= Index)
    return Sentinal;
  MCOperand const &MCO = MCI.getOperand(Index);
  if (!MCO.isExpr())
    return Sentinal;
  int64_t Value;
  if (!MCO.getExpr()->evaluateAsAbsolute(Value))
    return Sentinal;
  return Value;
}

void HexagonMCInstrInfo::padEndloop(MCContext &Context, MCInst &MCB) {
  MCInst Nop;
  Nop.setOpcode(Hexagon::A2_nop);
  assert(isBundle(MCB));
  while ((HexagonMCInstrInfo::isInnerLoop(MCB) &&
          (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_INNER_SIZE)) ||
         ((HexagonMCInstrInfo::isOuterLoop(MCB) &&
           (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_OUTER_SIZE))))
    MCB.addOperand(MCOperand::createInst(new (Context) MCInst(Nop)));
}

bool HexagonMCInstrInfo::prefersSlot3(MCInstrInfo const &MCII,
                                      MCInst const &MCI) {
  if (HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypeCR)
    return false;

  unsigned SchedClass = HexagonMCInstrInfo::getDesc(MCII, MCI).getSchedClass();
  switch (SchedClass) {
  case Hexagon::Sched::ALU32_3op_tc_2_SLOT0123:
  case Hexagon::Sched::ALU64_tc_2_SLOT23:
  case Hexagon::Sched::ALU64_tc_3x_SLOT23:
  case Hexagon::Sched::M_tc_2_SLOT23:
  case Hexagon::Sched::M_tc_3x_SLOT23:
  case Hexagon::Sched::S_2op_tc_2_SLOT23:
  case Hexagon::Sched::S_3op_tc_2_SLOT23:
  case Hexagon::Sched::S_3op_tc_3x_SLOT23:
    return true;
  }
  return false;
}

void HexagonMCInstrInfo::replaceDuplex(MCContext &Context, MCInst &MCB,
                                       DuplexCandidate Candidate) {
  assert(Candidate.packetIndexI < MCB.size());
  assert(Candidate.packetIndexJ < MCB.size());
  assert(isBundle(MCB));
  MCInst *Duplex =
      deriveDuplex(Context, Candidate.iClass,
                   *MCB.getOperand(Candidate.packetIndexJ).getInst(),
                   *MCB.getOperand(Candidate.packetIndexI).getInst());
  assert(Duplex != nullptr);
  MCB.getOperand(Candidate.packetIndexI).setInst(Duplex);
  MCB.erase(MCB.begin() + Candidate.packetIndexJ);
}

void HexagonMCInstrInfo::setInnerLoop(MCInst &MCI) {
  assert(isBundle(MCI));
  MCOperand &Operand = MCI.getOperand(0);
  Operand.setImm(Operand.getImm() | innerLoopMask);
}

void HexagonMCInstrInfo::setMemReorderDisabled(MCInst &MCI) {
  assert(isBundle(MCI));
  MCOperand &Operand = MCI.getOperand(0);
  Operand.setImm(Operand.getImm() | memReorderDisabledMask);
  assert(isMemReorderDisabled(MCI));
}

void HexagonMCInstrInfo::setMemStoreReorderEnabled(MCInst &MCI) {
  assert(isBundle(MCI));
  MCOperand &Operand = MCI.getOperand(0);
  Operand.setImm(Operand.getImm() | memStoreReorderEnabledMask);
  assert(isMemStoreReorderEnabled(MCI));
}

void HexagonMCInstrInfo::setOuterLoop(MCInst &MCI) {
  assert(isBundle(MCI));
  MCOperand &Operand = MCI.getOperand(0);
  Operand.setImm(Operand.getImm() | outerLoopMask);
}
}
