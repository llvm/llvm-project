//===----- HexagonShuffler.cpp - Instruction bundle shuffling -------------===//
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

#include <algorithm>
#include <utility>
#include "Hexagon.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "HexagonShuffler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
// Insn shuffling priority.
class HexagonBid {
  // The priority is directly proportional to how restricted the insn is based
  // on its flexibility to run on the available slots.  So, the fewer slots it
  // may run on, the higher its priority.
  enum { MAX = 360360 }; // LCD of 1/2, 1/3, 1/4,... 1/15.
  unsigned Bid;

public:
  HexagonBid() : Bid(0){};
  HexagonBid(unsigned B) { Bid = B ? MAX / countPopulation(B) : 0; };

  // Check if the insn priority is overflowed.
  bool isSold() const { return (Bid >= MAX); };

  HexagonBid &operator+=(const HexagonBid &B) {
    Bid += B.Bid;
    return *this;
  };
};

// Slot shuffling allocation.
class HexagonUnitAuction {
  HexagonBid Scores[HEXAGON_PACKET_SIZE];
  // Mask indicating which slot is unavailable.
  unsigned isSold : HEXAGON_PACKET_SIZE;

public:
  HexagonUnitAuction() : isSold(0){};

  // Allocate slots.
  bool bid(unsigned B) {
    // Exclude already auctioned slots from the bid.
    unsigned b = B & ~isSold;
    if (b) {
      for (unsigned i = 0; i < HEXAGON_PACKET_SIZE; ++i)
        if (b & (1 << i)) {
          // Request candidate slots.
          Scores[i] += HexagonBid(b);
          isSold |= Scores[i].isSold() << i;
        }
      return true;
      ;
    } else
      // Error if the desired slots are already full.
      return false;
  };
};
} // end anonymous namespace

unsigned HexagonResource::setWeight(unsigned s) {
  const unsigned SlotWeight = 8;
  const unsigned MaskWeight = SlotWeight - 1;
  bool Key = (1 << s) & getUnits();

  // TODO: Improve this API so that we can prevent misuse statically.
  assert(SlotWeight * s < 32 && "Argument to setWeight too large.");

  // Calculate relative weight of the insn for the given slot, weighing it the
  // heavier the more restrictive the insn is and the lowest the slots that the
  // insn may be executed in.
  Weight =
      (Key << (SlotWeight * s)) * ((MaskWeight - countPopulation(getUnits()))
                                   << countTrailingZeros(getUnits()));
  return (Weight);
}

void HexagonCVIResource::SetupTUL(TypeUnitsAndLanes *TUL, StringRef CPU) {
  (*TUL)[HexagonII::TypeCVI_VA] =
      UnitsAndLanes(CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VA_DV] = UnitsAndLanes(CVI_XLANE | CVI_MPY0, 2);
  (*TUL)[HexagonII::TypeCVI_VX] = UnitsAndLanes(CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VX_DV] = UnitsAndLanes(CVI_MPY0, 2);
  (*TUL)[HexagonII::TypeCVI_VP] = UnitsAndLanes(CVI_XLANE, 1);
  (*TUL)[HexagonII::TypeCVI_VP_VS] = UnitsAndLanes(CVI_XLANE, 2);
  (*TUL)[HexagonII::TypeCVI_VS] = UnitsAndLanes(CVI_SHIFT, 1);
  (*TUL)[HexagonII::TypeCVI_VINLANESAT] = UnitsAndLanes(CVI_SHIFT, 1);
  (*TUL)[HexagonII::TypeCVI_VM_LD] =
      UnitsAndLanes(CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VM_TMP_LD] = UnitsAndLanes(CVI_NONE, 0);
  (*TUL)[HexagonII::TypeCVI_VM_CUR_LD] =
      UnitsAndLanes(CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VM_VP_LDU] = UnitsAndLanes(CVI_XLANE, 1);
  (*TUL)[HexagonII::TypeCVI_VM_ST] =
      UnitsAndLanes(CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VM_NEW_ST] = UnitsAndLanes(CVI_NONE, 0);
  (*TUL)[HexagonII::TypeCVI_VM_STU] = UnitsAndLanes(CVI_XLANE, 1);
  (*TUL)[HexagonII::TypeCVI_HIST] = UnitsAndLanes(CVI_XLANE, 4);
}

HexagonCVIResource::HexagonCVIResource(TypeUnitsAndLanes *TUL,
                                       MCInstrInfo const &MCII, unsigned s,
                                       MCInst const *id)
    : HexagonResource(s), TUL(TUL) {
  unsigned T = HexagonMCInstrInfo::getType(MCII, *id);

  if (TUL->count(T)) {
    // For an HVX insn.
    Valid = true;
    setUnits((*TUL)[T].first);
    setLanes((*TUL)[T].second);
    setLoad(HexagonMCInstrInfo::getDesc(MCII, *id).mayLoad());
    setStore(HexagonMCInstrInfo::getDesc(MCII, *id).mayStore());
  } else {
    // For core insns.
    Valid = false;
    setUnits(0);
    setLanes(0);
    setLoad(false);
    setStore(false);
  }
}

HexagonShuffler::HexagonShuffler(MCInstrInfo const &MCII,
                                 MCSubtargetInfo const &STI)
    : MCII(MCII), STI(STI) {
  reset();
  HexagonCVIResource::SetupTUL(&TUL, STI.getCPU());
}

void HexagonShuffler::reset() {
  Packet.clear();
  BundleFlags = 0;
  Error = SHUFFLE_SUCCESS;
}

void HexagonShuffler::append(MCInst const *ID, MCInst const *Extender,
                             unsigned S, bool X) {
  HexagonInstr PI(&TUL, MCII, ID, Extender, S, X);

  Packet.push_back(PI);
}

/// Check that the packet is legal and enforce relative insn order.
bool HexagonShuffler::check() {
  // Descriptive slot masks.
  const unsigned slotSingleLoad = 0x1, slotSingleStore = 0x1, slotOne = 0x2,
                 slotThree = 0x8, slotFirstJump = 0x8, slotLastJump = 0x4,
                 slotFirstLoadStore = 0x2, slotLastLoadStore = 0x1;
  // Highest slots for branches and stores used to keep their original order.
  unsigned slotJump = slotFirstJump;
  unsigned slotLoadStore = slotFirstLoadStore;
  // Number of branches, solo branches, indirect branches.
  unsigned jumps = 0, jump1 = 0, jumpr = 0;
  // Number of memory operations, loads, solo loads, stores, solo stores, single
  // stores.
  unsigned memory = 0, loads = 0, load0 = 0, stores = 0, store0 = 0, store1 = 0;
  // Number of HVX loads, HVX stores.
  unsigned CVIloads = 0, CVIstores = 0;
  // Number of duplex insns, solo insns.
  unsigned duplex = 0, solo = 0;
  // Number of insns restricting other insns in the packet to A and X types,
  // which is neither A or X types.
  unsigned onlyAX = 0, neitherAnorX = 0;
  // Number of insns restricting other insns in slot #1 to A type.
  unsigned onlyAin1 = 0;
  // Number of insns restricting any insn in slot #1, except A2_nop.
  unsigned onlyNo1 = 0;
  unsigned xtypeFloat = 0;
  unsigned pSlot3Cnt = 0;
  iterator slot3ISJ = end();

  // Collect information from the insns in the packet.
  for (iterator ISJ = begin(); ISJ != end(); ++ISJ) {
    MCInst const *ID = ISJ->getDesc();

    if (HexagonMCInstrInfo::isSolo(MCII, *ID))
      solo += !ISJ->isSoloException();
    else if (HexagonMCInstrInfo::isSoloAX(MCII, *ID))
      onlyAX += !ISJ->isSoloException();
    else if (HexagonMCInstrInfo::isSoloAin1(MCII, *ID))
      onlyAin1 += !ISJ->isSoloException();
    if (HexagonMCInstrInfo::getType(MCII, *ID) != HexagonII::TypeALU32 &&
        HexagonMCInstrInfo::getType(MCII, *ID) != HexagonII::TypeXTYPE)
      ++neitherAnorX;
    if (HexagonMCInstrInfo::prefersSlot3(MCII, *ID)) {
      ++pSlot3Cnt;
      slot3ISJ = ISJ;
    }

    switch (HexagonMCInstrInfo::getType(MCII, *ID)) {
    case HexagonII::TypeXTYPE:
      if (HexagonMCInstrInfo::isFloat(MCII, *ID))
        ++xtypeFloat;
      break;
    case HexagonII::TypeJR:
      ++jumpr;
    // Fall-through.
    case HexagonII::TypeJ:
      ++jumps;
      break;
    case HexagonII::TypeCVI_VM_VP_LDU:
      ++onlyNo1;
    case HexagonII::TypeCVI_VM_LD:
    case HexagonII::TypeCVI_VM_TMP_LD:
    case HexagonII::TypeCVI_VM_CUR_LD:
      ++CVIloads;
    case HexagonII::TypeLD:
      ++loads;
      ++memory;
      if (ISJ->Core.getUnits() == slotSingleLoad)
        ++load0;
      if (HexagonMCInstrInfo::getDesc(MCII, *ID).isReturn())
        ++jumps, ++jump1; // DEALLOC_RETURN is of type LD.
      break;
    case HexagonII::TypeCVI_VM_STU:
      ++onlyNo1;
    case HexagonII::TypeCVI_VM_ST:
    case HexagonII::TypeCVI_VM_NEW_ST:
      ++CVIstores;
    case HexagonII::TypeST:
      ++stores;
      ++memory;
      if (ISJ->Core.getUnits() == slotSingleStore)
        ++store0;
      break;
    case HexagonII::TypeMEMOP:
      ++loads;
      ++stores;
      ++store1;
      ++memory;
      break;
    case HexagonII::TypeNV:
      ++memory; // NV insns are memory-like.
      if (HexagonMCInstrInfo::getDesc(MCII, *ID).isBranch())
        ++jumps, ++jump1;
      break;
    case HexagonII::TypeCR:
    // Legacy conditional branch predicated on a register.
    case HexagonII::TypeSYSTEM:
      if (HexagonMCInstrInfo::getDesc(MCII, *ID).mayLoad())
        ++loads;
      break;
    }
  }

  // Check if the packet is legal.
  if ((load0 > 1 || store0 > 1 || CVIloads > 1 || CVIstores > 1) ||
      (duplex > 1 || (duplex && memory)) || (solo && size() > 1) ||
      (onlyAX && neitherAnorX > 1) || (onlyAX && xtypeFloat)) {
    Error = SHUFFLE_ERROR_INVALID;
    return false;
  }

  if (jump1 && jumps > 1) {
    // Error if single branch with another branch.
    Error = SHUFFLE_ERROR_BRANCHES;
    return false;
  }

  // Modify packet accordingly.
  // TODO: need to reserve slots #0 and #1 for duplex insns.
  bool bOnlySlot3 = false;
  for (iterator ISJ = begin(); ISJ != end(); ++ISJ) {
    MCInst const *ID = ISJ->getDesc();

    if (!ISJ->Core.getUnits()) {
      // Error if insn may not be executed in any slot.
      Error = SHUFFLE_ERROR_UNKNOWN;
      return false;
    }

    // Exclude from slot #1 any insn but A2_nop.
    if (HexagonMCInstrInfo::getDesc(MCII, *ID).getOpcode() != Hexagon::A2_nop)
      if (onlyNo1)
        ISJ->Core.setUnits(ISJ->Core.getUnits() & ~slotOne);

    // Exclude from slot #1 any insn but A-type.
    if (HexagonMCInstrInfo::getType(MCII, *ID) != HexagonII::TypeALU32)
      if (onlyAin1)
        ISJ->Core.setUnits(ISJ->Core.getUnits() & ~slotOne);

    // Branches must keep the original order.
    if (HexagonMCInstrInfo::getDesc(MCII, *ID).isBranch() ||
        HexagonMCInstrInfo::getDesc(MCII, *ID).isCall())
      if (jumps > 1) {
        if (jumpr || slotJump < slotLastJump) {
          // Error if indirect branch with another branch or
          // no more slots available for branches.
          Error = SHUFFLE_ERROR_BRANCHES;
          return false;
        }
        // Pin the branch to the highest slot available to it.
        ISJ->Core.setUnits(ISJ->Core.getUnits() & slotJump);
        // Update next highest slot available to branches.
        slotJump >>= 1;
      }

    // A single load must use slot #0.
    if (HexagonMCInstrInfo::getDesc(MCII, *ID).mayLoad()) {
      if (loads == 1 && loads == memory)
        // Pin the load to slot #0.
        ISJ->Core.setUnits(ISJ->Core.getUnits() & slotSingleLoad);
    }

    // A single store must use slot #0.
    if (HexagonMCInstrInfo::getDesc(MCII, *ID).mayStore()) {
      if (!store0) {
        if (stores == 1)
          ISJ->Core.setUnits(ISJ->Core.getUnits() & slotSingleStore);
        else if (stores > 1) {
          if (slotLoadStore < slotLastLoadStore) {
            // Error if no more slots available for stores.
            Error = SHUFFLE_ERROR_STORES;
            return false;
          }
          // Pin the store to the highest slot available to it.
          ISJ->Core.setUnits(ISJ->Core.getUnits() & slotLoadStore);
          // Update the next highest slot available to stores.
          slotLoadStore >>= 1;
        }
      }
      if (store1 && stores > 1) {
        // Error if a single store with another store.
        Error = SHUFFLE_ERROR_STORES;
        return false;
      }
    }

    // flag if an instruction can only be executed in slot 3
    if (ISJ->Core.getUnits() == slotThree)
      bOnlySlot3 = true;

    if (!ISJ->Core.getUnits()) {
      // Error if insn may not be executed in any slot.
      Error = SHUFFLE_ERROR_NOSLOTS;
      return false;
    }
  }

  bool validateSlots = true;
  if (bOnlySlot3 == false && pSlot3Cnt == 1 && slot3ISJ != end()) {
    // save off slot mask of instruction marked with A_PREFER_SLOT3
    // and then pin it to slot #3
    unsigned saveUnits = slot3ISJ->Core.getUnits();
    slot3ISJ->Core.setUnits(saveUnits & slotThree);

    HexagonUnitAuction AuctionCore;
    std::sort(begin(), end(), HexagonInstr::lessCore);

    // see if things ok with that instruction being pinned to slot #3
    bool bFail = false;
    for (iterator I = begin(); I != end() && bFail != true; ++I)
      if (!AuctionCore.bid(I->Core.getUnits()))
        bFail = true;

    // if yes, great, if not then restore original slot mask
    if (!bFail)
      validateSlots = false; // all good, no need to re-do auction
    else
      for (iterator ISJ = begin(); ISJ != end(); ++ISJ) {
        MCInst const *ID = ISJ->getDesc();
        if (HexagonMCInstrInfo::prefersSlot3(MCII, *ID))
          ISJ->Core.setUnits(saveUnits);
      }
  }

  // Check if any slot, core, is over-subscribed.
  // Verify the core slot subscriptions.
  if (validateSlots) {
    HexagonUnitAuction AuctionCore;

    std::sort(begin(), end(), HexagonInstr::lessCore);

    for (iterator I = begin(); I != end(); ++I)
      if (!AuctionCore.bid(I->Core.getUnits())) {
        Error = SHUFFLE_ERROR_SLOTS;
        return false;
      }
  }
  // Verify the CVI slot subscriptions.
  {
    HexagonUnitAuction AuctionCVI;

    std::sort(begin(), end(), HexagonInstr::lessCVI);

    for (iterator I = begin(); I != end(); ++I)
      for (unsigned i = 0; i < I->CVI.getLanes(); ++i) // TODO: I->CVI.isValid?
        if (!AuctionCVI.bid(I->CVI.getUnits() << i)) {
          Error = SHUFFLE_ERROR_SLOTS;
          return false;
        }
  }

  Error = SHUFFLE_SUCCESS;
  return true;
}

bool HexagonShuffler::shuffle() {
  if (size() > HEXAGON_PACKET_SIZE) {
    // Ignore a packet with with more than what a packet can hold
    // or with compound or duplex insns for now.
    Error = SHUFFLE_ERROR_INVALID;
    return false;
  }

  // Check and prepare packet.
  if (size() > 1 && check())
    // Reorder the handles for each slot.
    for (unsigned nSlot = 0, emptySlots = 0; nSlot < HEXAGON_PACKET_SIZE;
         ++nSlot) {
      iterator ISJ, ISK;
      unsigned slotSkip, slotWeight;

      // Prioritize the handles considering their restrictions.
      for (ISJ = ISK = Packet.begin(), slotSkip = slotWeight = 0;
           ISK != Packet.end(); ++ISK, ++slotSkip)
        if (slotSkip < nSlot - emptySlots)
          // Note which handle to begin at.
          ++ISJ;
        else
          // Calculate the weight of the slot.
          slotWeight += ISK->Core.setWeight(HEXAGON_PACKET_SIZE - nSlot - 1);

      if (slotWeight)
        // Sort the packet, favoring source order,
        // beginning after the previous slot.
        std::sort(ISJ, Packet.end());
      else
        // Skip unused slot.
        ++emptySlots;
    }

  for (iterator ISJ = begin(); ISJ != end(); ++ISJ)
    DEBUG(dbgs().write_hex(ISJ->Core.getUnits());
          dbgs() << ':'
                 << HexagonMCInstrInfo::getDesc(MCII, *ISJ->getDesc())
                        .getOpcode();
          dbgs() << '\n');
  DEBUG(dbgs() << '\n');

  return (!getError());
}
