//===----- HexagonMCShuffler.cpp - MC bundle shuffling --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the shuffling of insns inside a bundle according to the
// packet formation rules of the Hexagon ISA.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hexagon-shuffle"

#include "Hexagon.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCShuffler.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
    DisableShuffle("disable-hexagon-shuffle", cl::Hidden, cl::init(false),
                   cl::desc("Disable Hexagon instruction shuffling"));

void HexagonMCShuffler::init(MCInst &MCB) {
  if (HexagonMCInstrInfo::isBundle(MCB)) {
    MCInst const *Extender = nullptr;
    // Copy the bundle for the shuffling.
    for (const auto &I : HexagonMCInstrInfo::bundleInstructions(MCB)) {
      assert(!HexagonMCInstrInfo::getDesc(MCII, *I.getInst()).isPseudo());
      MCInst *MI = const_cast<MCInst *>(I.getInst());

      if (!HexagonMCInstrInfo::isImmext(*MI)) {
        append(MI, Extender, HexagonMCInstrInfo::getUnits(MCII, STI, *MI),
               false);
        Extender = nullptr;
      } else
        Extender = MI;
    }
  }

  BundleFlags = MCB.getOperand(0).getImm();
}

void HexagonMCShuffler::init(MCInst &MCB, MCInst const *AddMI,
                             bool bInsertAtFront) {
  if (HexagonMCInstrInfo::isBundle(MCB)) {
    if (bInsertAtFront && AddMI)
      append(AddMI, nullptr, HexagonMCInstrInfo::getUnits(MCII, STI, *AddMI),
             false);
    MCInst const *Extender = nullptr;
    // Copy the bundle for the shuffling.
    for (auto const &I : HexagonMCInstrInfo::bundleInstructions(MCB)) {
      assert(!HexagonMCInstrInfo::getDesc(MCII, *I.getInst()).isPseudo());
      MCInst *MI = const_cast<MCInst *>(I.getInst());
      if (!HexagonMCInstrInfo::isImmext(*MI)) {
        append(MI, Extender, HexagonMCInstrInfo::getUnits(MCII, STI, *MI),
               false);
        Extender = nullptr;
      } else
        Extender = MI;
    }
    if (!bInsertAtFront && AddMI)
      append(AddMI, nullptr, HexagonMCInstrInfo::getUnits(MCII, STI, *AddMI),
             false);
  }

  BundleFlags = MCB.getOperand(0).getImm();
}

void HexagonMCShuffler::copyTo(MCInst &MCB) {
  MCB.clear();
  MCB.addOperand(MCOperand::createImm(BundleFlags));
  // Copy the results into the bundle.
  for (HexagonShuffler::iterator I = begin(); I != end(); ++I) {

    MCInst const *MI = I->getDesc();
    MCInst const *Extender = I->getExtender();
    if (Extender)
      MCB.addOperand(MCOperand::createInst(Extender));
    MCB.addOperand(MCOperand::createInst(MI));
  }
}

bool HexagonMCShuffler::reshuffleTo(MCInst &MCB) {
  if (shuffle()) {
    // Copy the results into the bundle.
    copyTo(MCB);
  } else
    DEBUG(MCB.dump());

  return (!getError());
}

bool llvm::HexagonMCShuffle(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                            MCInst &MCB) {
  HexagonMCShuffler MCS(MCII, STI, MCB);

  if (DisableShuffle)
    // Ignore if user chose so.
    return false;

  if (!HexagonMCInstrInfo::bundleSize(MCB)) {
    // There once was a bundle:
    //    BUNDLE %D2<imp-def>, %R4<imp-def>, %R5<imp-def>, %D7<imp-def>, ...
    //      * %D2<def> = IMPLICIT_DEF; flags:
    //      * %D7<def> = IMPLICIT_DEF; flags:
    // After the IMPLICIT_DEFs were removed by the asm printer, the bundle
    // became empty.
    DEBUG(dbgs() << "Skipping empty bundle");
    return false;
  } else if (!HexagonMCInstrInfo::isBundle(MCB)) {
    DEBUG(dbgs() << "Skipping stand-alone insn");
    return false;
  }

  // Reorder the bundle and copy the result.
  if (!MCS.reshuffleTo(MCB)) {
    // Unless there is any error, which should not happen at this point.
    unsigned shuffleError = MCS.getError();
    switch (shuffleError) {
    default:
      llvm_unreachable("unknown error");
    case HexagonShuffler::SHUFFLE_ERROR_INVALID:
      llvm_unreachable("invalid packet");
    case HexagonShuffler::SHUFFLE_ERROR_STORES:
      llvm_unreachable("too many stores");
    case HexagonShuffler::SHUFFLE_ERROR_LOADS:
      llvm_unreachable("too many loads");
    case HexagonShuffler::SHUFFLE_ERROR_BRANCHES:
      llvm_unreachable("too many branches");
    case HexagonShuffler::SHUFFLE_ERROR_NOSLOTS:
      llvm_unreachable("no suitable slot");
    case HexagonShuffler::SHUFFLE_ERROR_SLOTS:
      llvm_unreachable("over-subscribed slots");
    case HexagonShuffler::SHUFFLE_SUCCESS: // Single instruction case.
      return true;
    }
  }

  return true;
}

unsigned
llvm::HexagonMCShuffle(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                       MCContext &Context, MCInst &MCB,
                       SmallVector<DuplexCandidate, 8> possibleDuplexes) {

  if (DisableShuffle)
    return HexagonShuffler::SHUFFLE_SUCCESS;

  if (!HexagonMCInstrInfo::bundleSize(MCB)) {
    // There once was a bundle:
    //    BUNDLE %D2<imp-def>, %R4<imp-def>, %R5<imp-def>, %D7<imp-def>, ...
    //      * %D2<def> = IMPLICIT_DEF; flags:
    //      * %D7<def> = IMPLICIT_DEF; flags:
    // After the IMPLICIT_DEFs were removed by the asm printer, the bundle
    // became empty.
    DEBUG(dbgs() << "Skipping empty bundle");
    return HexagonShuffler::SHUFFLE_SUCCESS;
  } else if (!HexagonMCInstrInfo::isBundle(MCB)) {
    DEBUG(dbgs() << "Skipping stand-alone insn");
    return HexagonShuffler::SHUFFLE_SUCCESS;
  }

  bool doneShuffling = false;
  unsigned shuffleError;
  while (possibleDuplexes.size() > 0 && (!doneShuffling)) {
    // case of Duplex Found
    DuplexCandidate duplexToTry = possibleDuplexes.pop_back_val();
    MCInst Attempt(MCB);
    HexagonMCInstrInfo::replaceDuplex(Context, Attempt, duplexToTry);
    HexagonMCShuffler MCS(MCII, STI, Attempt); // copy packet to the shuffler
    if (MCS.size() == 1) {                     // case of one duplex
      // copy the created duplex in the shuffler to the bundle
      MCS.copyTo(MCB);
      return HexagonShuffler::SHUFFLE_SUCCESS;
    }
    // try shuffle with this duplex
    doneShuffling = MCS.reshuffleTo(MCB);
    shuffleError = MCS.getError();

    if (doneShuffling)
      break;
  }

  if (doneShuffling == false) {
    HexagonMCShuffler MCS(MCII, STI, MCB);
    doneShuffling = MCS.reshuffleTo(MCB); // shuffle
    shuffleError = MCS.getError();
  }
  if (!doneShuffling)
    return shuffleError;

  return HexagonShuffler::SHUFFLE_SUCCESS;
}

bool llvm::HexagonMCShuffle(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                            MCInst &MCB, MCInst const *AddMI, int fixupCount) {
  if (!HexagonMCInstrInfo::isBundle(MCB) || !AddMI)
    return false;

  // if fixups present, make sure we don't insert too many nops that would
  // later prevent an extender from being inserted.
  unsigned int bundleSize = HexagonMCInstrInfo::bundleSize(MCB);
  if (bundleSize >= HEXAGON_PACKET_SIZE)
    return false;
  if (fixupCount >= 2) {
    return false;
  } else {
    if (bundleSize == HEXAGON_PACKET_SIZE - 1 && fixupCount)
      return false;
  }

  if (DisableShuffle)
    return false;

  HexagonMCShuffler MCS(MCII, STI, MCB, AddMI);
  if (!MCS.reshuffleTo(MCB)) {
    unsigned shuffleError = MCS.getError();
    switch (shuffleError) {
    default:
      return false;
    case HexagonShuffler::SHUFFLE_SUCCESS: // single instruction case
      return true;
    }
  }

  return true;
}
