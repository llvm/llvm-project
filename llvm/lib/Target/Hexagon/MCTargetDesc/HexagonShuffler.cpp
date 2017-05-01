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

#include "HexagonShuffler.h"
#include "Hexagon.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <utility>

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
  HexagonBid() : Bid(0) {}
  HexagonBid(unsigned B) { Bid = B ? MAX / countPopulation(B) : 0; }

  // Check if the insn priority is overflowed.
  bool isSold() const { return (Bid >= MAX); }

  HexagonBid &operator+=(const HexagonBid &B) {
    Bid += B.Bid;
    return *this;
  }
};

// Slot shuffling allocation.
class HexagonUnitAuction {
  HexagonBid Scores[HEXAGON_PACKET_SIZE];
  // Mask indicating which slot is unavailable.
  unsigned isSold : HEXAGON_PACKET_SIZE;

public:
  HexagonUnitAuction(unsigned cs = 0) : isSold(cs){};

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
    } else
      // Error if the desired slots are already full.
      return false;
  }
};
} // end anonymous namespace

unsigned HexagonResource::setWeight(unsigned s) {
  const unsigned SlotWeight = 8;
  const unsigned MaskWeight = SlotWeight - 1;
  unsigned Units = getUnits();
  unsigned Key = ((1u << s) & Units) != 0;

  // Calculate relative weight of the insn for the given slot, weighing it the
  // heavier the more restrictive the insn is and the lowest the slots that the
  // insn may be executed in.
  if (Key == 0 || Units == 0 || (SlotWeight * s >= 32))
    return Weight = 0;

  unsigned Ctpop = countPopulation(Units);
  unsigned Cttz = countTrailingZeros(Units);
  Weight = (1u << (SlotWeight * s)) * ((MaskWeight - Ctpop) << Cttz);
  return Weight;
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
  (*TUL)[HexagonII::TypeCVI_VINLANESAT] =
      (CPU == "hexagonv60" || CPU == "hexagonv61" || CPU == "hexagonv61v1")
          ? UnitsAndLanes(CVI_SHIFT, 1)
          : UnitsAndLanes(CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VM_LD] =
      UnitsAndLanes(CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1, 1);
  (*TUL)[HexagonII::TypeCVI_VM_TMP_LD] = UnitsAndLanes(CVI_NONE, 0);
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

struct CVIUnits {
  unsigned Units;
  unsigned Lanes;
};
typedef SmallVector<struct CVIUnits, 8> HVXInstsT;

static unsigned makeAllBits(unsigned startBit, unsigned Lanes)

{
  for (unsigned i = 1; i < Lanes; ++i)
    startBit = (startBit << 1) | startBit;
  return startBit;
}

static bool checkHVXPipes(const HVXInstsT &hvxInsts, unsigned startIdx,
                          unsigned usedUnits)

{
  if (startIdx < hvxInsts.size()) {
    if (!hvxInsts[startIdx].Units)
      return checkHVXPipes(hvxInsts, startIdx + 1, usedUnits);
    for (unsigned b = 0x1; b <= 0x8; b <<= 1) {
      if ((hvxInsts[startIdx].Units & b) == 0)
        continue;
      unsigned allBits = makeAllBits(b, hvxInsts[startIdx].Lanes);
      if ((allBits & usedUnits) == 0) {
        if (checkHVXPipes(hvxInsts, startIdx + 1, usedUnits | allBits))
          return true;
      }
    }
    return false;
  }
  return true;
}

HexagonShuffler::HexagonShuffler(MCContext &Context, bool ReportErrors,
                                 MCInstrInfo const &MCII,
                                 MCSubtargetInfo const &STI)
    : Context(Context), MCII(MCII), STI(STI), ReportErrors(ReportErrors) {
  reset();
  HexagonCVIResource::SetupTUL(&TUL, STI.getCPU());
}

void HexagonShuffler::reset() {
  Packet.clear();
  BundleFlags = 0;
}

void HexagonShuffler::append(MCInst const &ID, MCInst const *Extender,
                             unsigned S) {
  HexagonInstr PI(&TUL, MCII, &ID, Extender, S);

  Packet.push_back(PI);
}

static struct {
  unsigned first;
  unsigned second;
} jumpSlots[] = {{8, 4}, {8, 2}, {8, 1}, {4, 2}, {4, 1}, {2, 1}};
#define MAX_JUMP_SLOTS (sizeof(jumpSlots) / sizeof(jumpSlots[0]))

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

unsigned countNeitherAnorX(MCInstrInfo const &MCII, MCInst const &ID) {
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
        Type != HexagonII::TypeS_3op && Type != HexagonII::TypeALU64 &&
        (Type != HexagonII::TypeM || HexagonMCInstrInfo::isFloat(MCII, ID));
  return Result;
}
} // namespace

/// Check that the packet is legal and enforce relative insn order.
bool HexagonShuffler::check() {
  // Descriptive slot masks.
  const unsigned slotSingleLoad = 0x1, slotSingleStore = 0x1, slotOne = 0x2,
                 slotThree = 0x8, // slotFirstJump = 0x8,
      slotFirstLoadStore = 0x2, slotLastLoadStore = 0x1;
  // Highest slots for branches and stores used to keep their original order.
  // unsigned slotJump = slotFirstJump;
  unsigned slotLoadStore = slotFirstLoadStore;
  // Number of branches, solo branches, indirect branches.
  unsigned jumps = 0, jump1 = 0;
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
  unsigned nvstores = 0;
  unsigned memops = 0;
  unsigned deallocs = 0;
  iterator slot3ISJ = end();
  std::vector<iterator> foundBranches;
  unsigned reservedSlots = 0;

  // Collect information from the insns in the packet.
  for (iterator ISJ = begin(); ISJ != end(); ++ISJ) {
    MCInst const &ID = ISJ->getDesc();

    if (HexagonMCInstrInfo::isSolo(MCII, ID))
      solo++;
    else if (HexagonMCInstrInfo::isSoloAX(MCII, ID))
      onlyAX++;
    else if (HexagonMCInstrInfo::isSoloAin1(MCII, ID))
      onlyAin1++;
    neitherAnorX += countNeitherAnorX(MCII, ID);
    if (HexagonMCInstrInfo::prefersSlot3(MCII, ID)) {
      ++pSlot3Cnt;
      slot3ISJ = ISJ;
    }
    reservedSlots |= HexagonMCInstrInfo::getOtherReservedSlots(MCII, STI, ID);
    if (HexagonMCInstrInfo::isCofMax1(MCII, ID))
      ++jump1;

    switch (HexagonMCInstrInfo::getType(MCII, ID)) {
    case HexagonII::TypeS_2op:
    case HexagonII::TypeS_3op:
    case HexagonII::TypeALU64:
      if (HexagonMCInstrInfo::isFloat(MCII, ID))
        ++xtypeFloat;
      break;
    case HexagonII::TypeJ:
      ++jumps;
      foundBranches.push_back(ISJ);
      break;
    case HexagonII::TypeCVI_VM_VP_LDU:
      ++onlyNo1;
    case HexagonII::TypeCVI_VM_LD:
    case HexagonII::TypeCVI_VM_TMP_LD:
      ++CVIloads;
    case HexagonII::TypeLD:
      ++loads;
      ++memory;
      if (ISJ->Core.getUnits() == slotSingleLoad ||
          HexagonMCInstrInfo::getType(MCII, ID) == HexagonII::TypeCVI_VM_VP_LDU)
        ++load0;
      if (HexagonMCInstrInfo::getDesc(MCII, ID).isReturn()) {
        ++deallocs, ++jumps, ++jump1; // DEALLOC_RETURN is of type LD.
        foundBranches.push_back(ISJ);
      }
      break;
    case HexagonII::TypeCVI_VM_STU:
      ++onlyNo1;
    case HexagonII::TypeCVI_VM_ST:
    case HexagonII::TypeCVI_VM_NEW_ST:
      ++CVIstores;
    case HexagonII::TypeST:
      ++stores;
      ++memory;
      if (ISJ->Core.getUnits() == slotSingleStore ||
          HexagonMCInstrInfo::getType(MCII, ID) == HexagonII::TypeCVI_VM_STU)
        ++store0;
      break;
    case HexagonII::TypeV4LDST:
      ++loads;
      ++stores;
      ++store1;
      ++memops;
      ++memory;
      break;
    case HexagonII::TypeNCJ:
      ++memory; // NV insns are memory-like.
      if (HexagonMCInstrInfo::getDesc(MCII, ID).isBranch()) {
        ++jumps, ++jump1;
        foundBranches.push_back(ISJ);
      }
      break;
    case HexagonII::TypeV2LDST:
      if (HexagonMCInstrInfo::getDesc(MCII, ID).mayLoad()) {
        ++loads;
        ++memory;
        if (ISJ->Core.getUnits() == slotSingleLoad ||
            HexagonMCInstrInfo::getType(MCII, ID) ==
                HexagonII::TypeCVI_VM_VP_LDU)
          ++load0;
      } else {
        assert(HexagonMCInstrInfo::getDesc(MCII, ID).mayStore());
        ++memory;
        ++stores;
        if (HexagonMCInstrInfo::isNewValue(MCII, ID))
          ++nvstores;
      }
      break;
    case HexagonII::TypeCR:
    // Legacy conditional branch predicated on a register.
    case HexagonII::TypeCJ:
      if (HexagonMCInstrInfo::getDesc(MCII, ID).isBranch()) {
        ++jumps;
        foundBranches.push_back(ISJ);
      }
      break;
    case HexagonII::TypeDUPLEX: {
      ++duplex;
      MCInst const &Inst0 = *ID.getOperand(0).getInst();
      MCInst const &Inst1 = *ID.getOperand(1).getInst();
      if (HexagonMCInstrInfo::isCofMax1(MCII, Inst0))
        ++jump1;
      if (HexagonMCInstrInfo::isCofMax1(MCII, Inst1))
        ++jump1;
      if (HexagonMCInstrInfo::getDesc(MCII, Inst0).isBranch()) {
        ++jumps;
        foundBranches.push_back(ISJ);
      }
      if (HexagonMCInstrInfo::getDesc(MCII, Inst1).isBranch()) {
        ++jumps;
        foundBranches.push_back(ISJ);
      }
      break;
    }
    }
  }

  // Check if the packet is legal.
  if ((load0 > 1 || store0 > 1 || CVIloads > 1 || CVIstores > 1) ||
      (duplex > 1 || (duplex && memory)) || (solo && size() > 1) ||
      (onlyAX && neitherAnorX > 1) || (onlyAX && xtypeFloat)) {
    reportError(llvm::Twine("invalid instruction packet"));
    return false;
  }

  if (jump1 && jumps > 1) {
    // Error if single branch with another branch.
    reportError(llvm::Twine("too many branches in packet"));
    return false;
  }
  if ((nvstores || memops) && stores > 1) {
    reportError(llvm::Twine("slot 0 instruction does not allow slot 1 store"));
    return false;
  }
  if (deallocs && stores) {
    reportError(llvm::Twine("slot 0 instruction does not allow slot 1 store"));
    return false;
  }

  // Modify packet accordingly.
  // TODO: need to reserve slots #0 and #1 for duplex insns.
  bool bOnlySlot3 = false;
  for (iterator ISJ = begin(); ISJ != end(); ++ISJ) {
    MCInst const &ID = ISJ->getDesc();

    if (!ISJ->Core.getUnits()) {
      // Error if insn may not be executed in any slot.
      return false;
    }

    // Exclude from slot #1 any insn but A2_nop.
    if (HexagonMCInstrInfo::getDesc(MCII, ID).getOpcode() != Hexagon::A2_nop)
      if (onlyNo1)
        ISJ->Core.setUnits(ISJ->Core.getUnits() & ~slotOne);

    // Exclude from slot #1 any insn but A-type.
    if (HexagonMCInstrInfo::getType(MCII, ID) != HexagonII::TypeALU32_2op &&
        HexagonMCInstrInfo::getType(MCII, ID) != HexagonII::TypeALU32_3op &&
        HexagonMCInstrInfo::getType(MCII, ID) != HexagonII::TypeALU32_ADDI)
      if (onlyAin1)
        ISJ->Core.setUnits(ISJ->Core.getUnits() & ~slotOne);

    // A single load must use slot #0.
    if (HexagonMCInstrInfo::getDesc(MCII, ID).mayLoad()) {
      if (loads == 1 && loads == memory && memops == 0)
        // Pin the load to slot #0.
        ISJ->Core.setUnits(ISJ->Core.getUnits() & slotSingleLoad);
    }

    // A single store must use slot #0.
    if (HexagonMCInstrInfo::getDesc(MCII, ID).mayStore()) {
      if (!store0) {
        if (stores == 1)
          ISJ->Core.setUnits(ISJ->Core.getUnits() & slotSingleStore);
        else if (stores > 1) {
          if (slotLoadStore < slotLastLoadStore) {
            // Error if no more slots available for stores.
            reportError(
                llvm::Twine("invalid instruction packet: too many stores"));
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
        reportError(llvm::Twine("invalid instruction packet: too many stores"));
        return false;
      }
    }

    // flag if an instruction requires to be in slot 3
    if (ISJ->Core.getUnits() == slotThree)
      bOnlySlot3 = true;

    if (!ISJ->Core.getUnits()) {
      // Error if insn may not be executed in any slot.
      reportError(llvm::Twine("invalid instruction packet: out of slots"));
      return false;
    }
  }

  // preserve branch order
  bool validateSlots = true;
  if (jumps > 1) {
    if (foundBranches.size() > 2) {
      reportError(llvm::Twine("too many branches in packet"));
      return false;
    }

    // try all possible choices
    for (unsigned int i = 0; i < MAX_JUMP_SLOTS; ++i) {
      // validate first jump with this slot rule
      if (!(jumpSlots[i].first & foundBranches[0]->Core.getUnits()))
        continue;

      // validate second jump with this slot rule
      if (!(jumpSlots[i].second & foundBranches[1]->Core.getUnits()))
        continue;

      // both valid for this configuration, set new slot rules
      PacketSave = Packet;
      foundBranches[0]->Core.setUnits(jumpSlots[i].first);
      foundBranches[1]->Core.setUnits(jumpSlots[i].second);

      HexagonUnitAuction AuctionCore(reservedSlots);
      std::sort(begin(), end(), HexagonInstr::lessCore);

      // see if things ok with that instruction being pinned to slot "slotJump"
      bool bFail = false;
      for (iterator I = begin(); I != end() && bFail != true; ++I)
        if (!AuctionCore.bid(I->Core.getUnits()))
          bFail = true;

      // if yes, great, if not then restore original slot mask
      if (!bFail) {
        validateSlots = false; // all good, no need to re-do auction
        break;
      } else
        // restore original values
        Packet = PacketSave;
    }
    if (validateSlots == true) {
      reportError(llvm::Twine("invalid instruction packet: out of slots"));
      return false;
    }
  }

  if (jumps <= 1 && bOnlySlot3 == false && pSlot3Cnt == 1 &&
      slot3ISJ != end()) {
    validateSlots = true;
    // save off slot mask of instruction marked with A_PREFER_SLOT3
    // and then pin it to slot #3
    unsigned saveUnits = slot3ISJ->Core.getUnits();
    slot3ISJ->Core.setUnits(saveUnits & slotThree);

    HexagonUnitAuction AuctionCore(reservedSlots);
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
        MCInst const &ID = ISJ->getDesc();
        if (HexagonMCInstrInfo::prefersSlot3(MCII, ID))
          ISJ->Core.setUnits(saveUnits);
      }
  }

  // Check if any slot, core or CVI, is over-subscribed.
  // Verify the core slot subscriptions.
  if (validateSlots) {
    HexagonUnitAuction AuctionCore(reservedSlots);

    std::sort(begin(), end(), HexagonInstr::lessCore);

    for (iterator I = begin(); I != end(); ++I)
      if (!AuctionCore.bid(I->Core.getUnits())) {
        reportError(llvm::Twine("invalid instruction packet: slot error"));
        return false;
      }
  }
  // Verify the CVI slot subscriptions.
  std::sort(begin(), end(), HexagonInstr::lessCVI);
  // create vector of hvx instructions to check
  HVXInstsT hvxInsts;
  hvxInsts.clear();
  for (iterator I = begin(); I != end(); ++I) {
    struct CVIUnits inst;
    inst.Units = I->CVI.getUnits();
    inst.Lanes = I->CVI.getLanes();
    if (inst.Units == 0)
      continue; // not an hvx inst or an hvx inst that doesn't uses any pipes
    hvxInsts.push_back(inst);
  }
  // if there are any hvx instructions in this packet, check pipe usage
  if (hvxInsts.size() > 0) {
    unsigned startIdx, usedUnits;
    startIdx = usedUnits = 0x0;
    if (checkHVXPipes(hvxInsts, startIdx, usedUnits) == false) {
      // too many pipes used to be valid
      reportError(llvm::Twine("invalid instruction packet: slot error"));
      return false;
    }
  }

  return true;
}

bool HexagonShuffler::shuffle() {
  if (size() > HEXAGON_PACKET_SIZE) {
    // Ignore a packet with with more than what a packet can hold
    // or with compound or duplex insns for now.
    reportError(llvm::Twine("invalid instruction packet"));
    return false;
  }

  // Check and prepare packet.
  bool Ok = true;
  if (size() > 1 && (Ok = check()))
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
    DEBUG(dbgs().write_hex(ISJ->Core.getUnits()); if (ISJ->CVI.isValid()) {
      dbgs() << '/';
      dbgs().write_hex(ISJ->CVI.getUnits()) << '|';
      dbgs() << ISJ->CVI.getLanes();
    } dbgs() << ':'
             << HexagonMCInstrInfo::getDesc(MCII, ISJ->getDesc()).getOpcode();
          dbgs() << '\n');
  DEBUG(dbgs() << '\n');

  return Ok;
}

void HexagonShuffler::reportError(llvm::Twine const &Msg) {
  if (ReportErrors)
    Context.reportError(Loc, Msg);
}
