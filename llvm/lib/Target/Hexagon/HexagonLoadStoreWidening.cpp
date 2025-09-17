//===---HexagonLoadStoreWidening.cpp---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// HexagonStoreWidening:
// Replace sequences of "narrow" stores to adjacent memory locations with
// a fewer "wide" stores that have the same effect.
// For example, replace:
//   S4_storeirb_io  %100, 0, 0   ; store-immediate-byte
//   S4_storeirb_io  %100, 1, 0   ; store-immediate-byte
// with
//   S4_storeirh_io  %100, 0, 0   ; store-immediate-halfword
// The above is the general idea.  The actual cases handled by the code
// may be a bit more complex.
// The purpose of this pass is to reduce the number of outstanding stores,
// or as one could say, "reduce store queue pressure".  Also, wide stores
// mean fewer stores, and since there are only two memory instructions allowed
// per packet, it also means fewer packets, and ultimately fewer cycles.
//
// HexagonLoadWidening does the same thing as HexagonStoreWidening but
// for Loads. Here, we try to replace 4-byte Loads with register-pair loads.
// For example:
// Replace
//   %2:intregs = L2_loadri_io %1:intregs, 0 :: (load (s32) from %ptr1, align 8)
//   %3:intregs = L2_loadri_io %1:intregs, 4 :: (load (s32) from %ptr2)
// with
//   %4:doubleregs = L2_loadrd_io %1:intregs, 0 :: (load (s64) from %ptr1)
//   %2:intregs = COPY %4.isub_lo:doubleregs
//   %3:intregs = COPY %4.isub_hi:doubleregs
//
// LoadWidening for 8 and 16-bit loads is not useful as we end up generating 2N
// insts to replace N loads: 1 widened load, N bitwise and, N - 1 shifts

//===---------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "hexagon-load-store-widening"

static cl::opt<unsigned> MaxMBBSizeForLoadStoreWidening(
    "max-bb-size-for-load-store-widening", cl::Hidden, cl::init(1000),
    cl::desc("Limit block size to analyze in load/store widening pass"));

namespace {

struct HexagonLoadStoreWidening {
  enum WideningMode { Store, Load };
  const HexagonInstrInfo *TII;
  const HexagonRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  AliasAnalysis *AA;
  MachineFunction *MF;

public:
  HexagonLoadStoreWidening(const HexagonInstrInfo *TII,
                           const HexagonRegisterInfo *TRI,
                           MachineRegisterInfo *MRI, AliasAnalysis *AA,
                           MachineFunction *MF, bool StoreMode)
      : TII(TII), TRI(TRI), MRI(MRI), AA(AA), MF(MF),
        Mode(StoreMode ? WideningMode::Store : WideningMode::Load),
        HII(MF->getSubtarget<HexagonSubtarget>().getInstrInfo()) {}

  bool run();

private:
  const bool Mode;
  const unsigned MaxWideSize = 8;
  const HexagonInstrInfo *HII = nullptr;

  using InstrSet = SmallPtrSet<MachineInstr *, 16>;
  using InstrGroup = SmallVector<MachineInstr *, 8>;
  using InstrGroupList = SmallVector<InstrGroup, 8>;

  InstrSet ProcessedInsts;

  unsigned getBaseAddressRegister(const MachineInstr *MI);
  int64_t getOffset(const MachineInstr *MI);
  int64_t getPostIncrementValue(const MachineInstr *MI);
  bool handledInstType(const MachineInstr *MI);

  void createGroup(MachineInstr *BaseInst, InstrGroup &Group);
  void createGroups(MachineBasicBlock &MBB, InstrGroupList &StoreGroups);
  bool processBasicBlock(MachineBasicBlock &MBB);
  bool processGroup(InstrGroup &Group);
  bool selectInsts(InstrGroup::iterator Begin, InstrGroup::iterator End,
                   InstrGroup &OG, unsigned &TotalSize, unsigned MaxSize);
  bool createWideInsts(InstrGroup &OG, InstrGroup &NG, unsigned TotalSize);
  bool createWideStores(InstrGroup &OG, InstrGroup &NG, unsigned TotalSize);
  bool createWideLoads(InstrGroup &OG, InstrGroup &NG, unsigned TotalSize);
  bool replaceInsts(InstrGroup &OG, InstrGroup &NG);
  bool areAdjacent(const MachineInstr *S1, const MachineInstr *S2);
  bool canSwapInstructions(const MachineInstr *A, const MachineInstr *B);
};

struct HexagonStoreWidening : public MachineFunctionPass {
  static char ID;

  HexagonStoreWidening() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Hexagon Store Widening"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MFn) override {
    if (skipFunction(MFn.getFunction()))
      return false;

    auto &ST = MFn.getSubtarget<HexagonSubtarget>();
    const HexagonInstrInfo *TII = ST.getInstrInfo();
    const HexagonRegisterInfo *TRI = ST.getRegisterInfo();
    MachineRegisterInfo *MRI = &MFn.getRegInfo();
    AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

    return HexagonLoadStoreWidening(TII, TRI, MRI, AA, &MFn, true).run();
  }
};

struct HexagonLoadWidening : public MachineFunctionPass {
  static char ID;

  HexagonLoadWidening() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Hexagon Load Widening"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MFn) override {
    if (skipFunction(MFn.getFunction()))
      return false;

    auto &ST = MFn.getSubtarget<HexagonSubtarget>();
    const HexagonInstrInfo *TII = ST.getInstrInfo();
    const HexagonRegisterInfo *TRI = ST.getRegisterInfo();
    MachineRegisterInfo *MRI = &MFn.getRegInfo();
    AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
    return HexagonLoadStoreWidening(TII, TRI, MRI, AA, &MFn, false).run();
  }
};

char HexagonStoreWidening::ID = 0;
char HexagonLoadWidening::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(HexagonStoreWidening, "hexagon-widen-stores",
                      "Hexagon Store Widening", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(HexagonStoreWidening, "hexagon-widen-stores",
                    "Hexagon Store Widening", false, false)

INITIALIZE_PASS_BEGIN(HexagonLoadWidening, "hexagon-widen-loads",
                      "Hexagon Load Widening", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(HexagonLoadWidening, "hexagon-widen-loads",
                    "Hexagon Load Widening", false, false)

static const MachineMemOperand &getMemTarget(const MachineInstr *MI) {
  assert(!MI->memoperands_empty() && "Expecting memory operands");
  return **MI->memoperands_begin();
}

unsigned
HexagonLoadStoreWidening::getBaseAddressRegister(const MachineInstr *MI) {
  assert(HexagonLoadStoreWidening::handledInstType(MI) && "Unhandled opcode");
  unsigned Base, Offset;
  HII->getBaseAndOffsetPosition(*MI, Base, Offset);
  const MachineOperand &MO = MI->getOperand(Base);
  assert(MO.isReg() && "Expecting register operand");
  return MO.getReg();
}

int64_t HexagonLoadStoreWidening::getOffset(const MachineInstr *MI) {
  assert(HexagonLoadStoreWidening::handledInstType(MI) && "Unhandled opcode");

  // On Hexagon, post-incs always have an offset of 0
  // There is no Offset operand to post-incs
  if (HII->isPostIncrement(*MI))
    return 0;

  unsigned Base, Offset;

  HII->getBaseAndOffsetPosition(*MI, Base, Offset);
  const MachineOperand &MO = MI->getOperand(Offset);
  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    return MO.getImm();
  case MachineOperand::MO_GlobalAddress:
    return MO.getOffset();
  default:
    break;
  }
  llvm_unreachable("Expecting an immediate or global operand");
}

inline int64_t
HexagonLoadStoreWidening::getPostIncrementValue(const MachineInstr *MI) {
  unsigned Base, PostIncIdx;
  HII->getBaseAndOffsetPosition(*MI, Base, PostIncIdx);
  const MachineOperand &MO = MI->getOperand(PostIncIdx);
  return MO.getImm();
}

// Filtering function: any loads/stores whose opcodes are not "approved" of by
// this function will not be subjected to widening.
inline bool HexagonLoadStoreWidening::handledInstType(const MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();
  if (Mode == WideningMode::Store) {
    switch (Opc) {
    case Hexagon::S4_storeirb_io:
    case Hexagon::S4_storeirh_io:
    case Hexagon::S4_storeiri_io:
    case Hexagon::S2_storeri_io:
      // Base address must be a register. (Implement FI later.)
      return MI->getOperand(0).isReg();
    case Hexagon::S2_storeri_pi:
      return MI->getOperand(1).isReg();
    }
  } else {
    // LoadWidening for 8 and 16 bit loads needs 2x instructions to replace x
    // loads. So we only widen 32 bit loads as we don't need to select the
    // right bits with AND & SHIFT ops.
    switch (Opc) {
    case Hexagon::L2_loadri_io:
      // Base address must be a register and offset must be immediate.
      return !MI->memoperands_empty() && MI->getOperand(1).isReg() &&
             MI->getOperand(2).isImm();
    case Hexagon::L2_loadri_pi:
      return !MI->memoperands_empty() && MI->getOperand(2).isReg();
    }
  }
  return false;
}

static void addDefsUsesToList(const MachineInstr *MI,
                              DenseSet<Register> &RegDefs,
                              DenseSet<Register> &RegUses) {
  for (const auto &Op : MI->operands()) {
    if (!Op.isReg())
      continue;
    if (Op.isDef())
      RegDefs.insert(Op.getReg());
    if (Op.readsReg())
      RegUses.insert(Op.getReg());
  }
}

bool HexagonLoadStoreWidening::canSwapInstructions(const MachineInstr *A,
                                                   const MachineInstr *B) {
  DenseSet<Register> ARegDefs;
  DenseSet<Register> ARegUses;
  addDefsUsesToList(A, ARegDefs, ARegUses);
  if (A->mayLoadOrStore() && B->mayLoadOrStore() &&
      (A->mayStore() || B->mayStore()) && A->mayAlias(AA, *B, true))
    return false;
  for (const auto &BOp : B->operands()) {
    if (!BOp.isReg())
      continue;
    if ((BOp.isDef() || BOp.readsReg()) && ARegDefs.contains(BOp.getReg()))
      return false;
    if (BOp.isDef() && ARegUses.contains(BOp.getReg()))
      return false;
  }
  return true;
}

// Inspect a machine basic block, and generate groups out of loads/stores
// encountered in the block.
//
// A load/store group is a group of loads or stores that use the same base
// register, and which can be reordered within that group without altering the
// semantics of the program.  A single group could be widened as
// a whole, if there existed a single load/store instruction with the same
// semantics as the entire group.  In many cases, a single group may need more
// than one wide load or store.
void HexagonLoadStoreWidening::createGroups(MachineBasicBlock &MBB,
                                            InstrGroupList &StoreGroups) {
  // Traverse all instructions and if we encounter
  // a load/store, then try to create a group starting at that instruction
  // i.e. a sequence of independent loads/stores that can be widened.
  for (auto I = MBB.begin(); I != MBB.end(); ++I) {
    MachineInstr *MI = &(*I);
    if (!handledInstType(MI))
      continue;
    if (ProcessedInsts.count(MI))
      continue;

    // Found a store.  Try to create a store group.
    InstrGroup G;
    createGroup(MI, G);
    if (G.size() > 1)
      StoreGroups.push_back(G);
  }
}

// Create a single load/store group.  The insts need to be independent between
// themselves, and also there cannot be other instructions between them
// that could read or modify storage being read from or stored into.
void HexagonLoadStoreWidening::createGroup(MachineInstr *BaseInst,
                                           InstrGroup &Group) {
  assert(handledInstType(BaseInst) && "Unexpected instruction");
  unsigned BaseReg = getBaseAddressRegister(BaseInst);
  InstrGroup Other;

  Group.push_back(BaseInst);
  LLVM_DEBUG(dbgs() << "BaseInst: "; BaseInst->dump());
  auto End = BaseInst->getParent()->end();
  auto I = BaseInst->getIterator();

  while (true) {
    I = std::next(I);
    if (I == End)
      break;
    MachineInstr *MI = &(*I);

    // Assume calls are aliased to everything.
    if (MI->isCall() || MI->hasUnmodeledSideEffects() ||
        MI->hasOrderedMemoryRef())
      return;

    if (!handledInstType(MI)) {
      if (MI->mayLoadOrStore())
        Other.push_back(MI);
      continue;
    }

    // We have a handledInstType instruction
    // If this load/store instruction is aliased with anything already in the
    // group, terminate the group now.
    for (auto GI : Group)
      if (GI->mayAlias(AA, *MI, true))
        return;
    if (Mode == WideningMode::Load) {
      // Check if current load MI can be moved to the first load instruction
      // in Group. If any load instruction aliases with memory instructions in
      // Other, terminate the group.
      for (auto MemI : Other)
        if (!canSwapInstructions(MI, MemI))
          return;
    } else {
      // Check if store instructions in the group can be moved to current
      // store MI. If any store instruction aliases with memory instructions
      // in Other, terminate the group.
      for (auto MemI : Other) {
        if (std::distance(Group.back()->getIterator(), MemI->getIterator()) <=
            0)
          continue;
        for (auto GI : Group)
          if (!canSwapInstructions(MemI, GI))
            return;
      }
    }

    unsigned BR = getBaseAddressRegister(MI);
    if (BR == BaseReg) {
      LLVM_DEBUG(dbgs() << "Added MI to group: "; MI->dump());
      Group.push_back(MI);
      ProcessedInsts.insert(MI);
    }
  } // while
}

// Check if load/store instructions S1 and S2 are adjacent.  More precisely,
// S2 has to access memory immediately following that accessed by S1.
bool HexagonLoadStoreWidening::areAdjacent(const MachineInstr *S1,
                                           const MachineInstr *S2) {
  if (!handledInstType(S1) || !handledInstType(S2))
    return false;

  const MachineMemOperand &S1MO = getMemTarget(S1);

  // Currently only handling immediate stores.
  int Off1 = getOffset(S1);
  int Off2 = getOffset(S2);

  return (Off1 >= 0) ? Off1 + S1MO.getSize().getValue() == unsigned(Off2)
                     : int(Off1 + S1MO.getSize().getValue()) == Off2;
}

/// Given a sequence of adjacent loads/stores, and a maximum size of a single
/// wide inst, pick a group of insts that can be replaced by a single load/store
/// of size not exceeding MaxSize.  The selected sequence will be recorded
/// in OG ("old group" of instructions).
/// OG should be empty on entry, and should be left empty if the function
/// fails.
bool HexagonLoadStoreWidening::selectInsts(InstrGroup::iterator Begin,
                                           InstrGroup::iterator End,
                                           InstrGroup &OG, unsigned &TotalSize,
                                           unsigned MaxSize) {
  assert(Begin != End && "No instructions to analyze");
  assert(OG.empty() && "Old group not empty on entry");

  if (std::distance(Begin, End) <= 1)
    return false;

  MachineInstr *FirstMI = *Begin;
  assert(!FirstMI->memoperands_empty() && "Expecting some memory operands");
  const MachineMemOperand &FirstMMO = getMemTarget(FirstMI);
  if (!FirstMMO.getType().isValid())
    return false;

  unsigned Alignment = FirstMMO.getAlign().value();
  unsigned SizeAccum = FirstMMO.getSize().getValue();
  unsigned FirstOffset = getOffset(FirstMI);

  // The initial value of SizeAccum should always be a power of 2.
  assert(isPowerOf2_32(SizeAccum) && "First store size not a power of 2");

  // If the size of the first store equals to or exceeds the limit, do nothing.
  if (SizeAccum >= MaxSize)
    return false;

  // If the size of the first load/store is greater than or equal to the address
  // stored to, then the inst cannot be made any wider.
  if (SizeAccum >= Alignment) {
    LLVM_DEBUG(
        dbgs() << "Size of load/store greater than equal to its alignment\n");
    return false;
  }

  // The offset of a load/store will put restrictions on how wide the inst can
  // be.  Offsets in loads/stores of size 2^n bytes need to have the n lowest
  // bits be 0.  If the first inst already exhausts the offset limits, quit.
  // Test this by checking if the next wider size would exceed the limit.
  // For post-increment instructions, the increment amount needs to follow the
  // same rule.
  unsigned OffsetOrIncVal = 0;
  if (HII->isPostIncrement(*FirstMI))
    OffsetOrIncVal = getPostIncrementValue(FirstMI);
  else
    OffsetOrIncVal = FirstOffset;
  if ((2 * SizeAccum - 1) & OffsetOrIncVal) {
    LLVM_DEBUG(dbgs() << "Instruction cannot be widened as the offset/postinc"
                      << " value: " << getPostIncrementValue(FirstMI)
                      << " is invalid in the widened version\n");
    return false;
  }

  OG.push_back(FirstMI);
  MachineInstr *S1 = FirstMI;

  // Pow2Num will be the largest number of elements in OG such that the sum
  // of sizes of loads/stores 0...Pow2Num-1 will be a power of 2.
  unsigned Pow2Num = 1;
  unsigned Pow2Size = SizeAccum;
  bool HavePostInc = HII->isPostIncrement(*S1);

  // Be greedy: keep accumulating insts as long as they are to adjacent
  // memory locations, and as long as the total number of bytes stored
  // does not exceed the limit (MaxSize).
  // Keep track of when the total size covered is a power of 2, since
  // this is a size a single load/store can cover.
  for (InstrGroup::iterator I = Begin + 1; I != End; ++I) {
    MachineInstr *S2 = *I;
    // Insts are sorted, so if S1 and S2 are not adjacent, there won't be
    // any other store to fill the "hole".
    if (!areAdjacent(S1, S2))
      break;

    // Cannot widen two post increments, need to return two registers
    // with incremented values
    if (HavePostInc && HII->isPostIncrement(*S2))
      break;

    unsigned S2Size = getMemTarget(S2).getSize().getValue();
    if (SizeAccum + S2Size > std::min(MaxSize, Alignment))
      break;

    OG.push_back(S2);
    SizeAccum += S2Size;
    if (isPowerOf2_32(SizeAccum)) {
      Pow2Num = OG.size();
      Pow2Size = SizeAccum;
    }
    if ((2 * Pow2Size - 1) & FirstOffset)
      break;

    S1 = S2;
  }

  // The insts don't add up to anything that can be widened.  Clean up.
  if (Pow2Num <= 1) {
    OG.clear();
    return false;
  }

  // Only leave the loads/stores being widened.
  OG.resize(Pow2Num);
  TotalSize = Pow2Size;
  return true;
}

/// Given an "old group" OG of insts, create a "new group" NG of instructions
/// to replace them.
bool HexagonLoadStoreWidening::createWideInsts(InstrGroup &OG, InstrGroup &NG,
                                               unsigned TotalSize) {
  if (Mode == WideningMode::Store) {
    return createWideStores(OG, NG, TotalSize);
  }
  return createWideLoads(OG, NG, TotalSize);
}

/// Given an "old group" OG of stores, create a "new group" NG of instructions
/// to replace them.  Ideally, NG would only have a single instruction in it,
/// but that may only be possible for store-immediate.
bool HexagonLoadStoreWidening::createWideStores(InstrGroup &OG, InstrGroup &NG,
                                                unsigned TotalSize) {
  // XXX Current limitations:
  // - only handle a TotalSize of up to 8

  LLVM_DEBUG(dbgs() << "Creating wide stores\n");
  if (TotalSize > MaxWideSize)
    return false;

  uint64_t Acc = 0; // Value accumulator.
  unsigned Shift = 0;
  bool HaveImm = false;
  bool HaveReg = false;

  for (MachineInstr *MI : OG) {
    const MachineMemOperand &MMO = getMemTarget(MI);
    MachineOperand &SO = HII->isPostIncrement(*MI)
                             ? MI->getOperand(3)
                             : MI->getOperand(2); // Source.
    unsigned NBits;
    uint64_t Mask;
    uint64_t Val;

    switch (SO.getType()) {
    case MachineOperand::MO_Immediate:
      LLVM_DEBUG(dbgs() << "Have store immediate\n");
      HaveImm = true;

      NBits = MMO.getSizeInBits().toRaw();
      Mask = (0xFFFFFFFFFFFFFFFFU >> (64 - NBits));
      Val = (SO.getImm() & Mask) << Shift;
      Acc |= Val;
      Shift += NBits;
      break;
    case MachineOperand::MO_Register:
      HaveReg = true;
      break;
    default:
      LLVM_DEBUG(dbgs() << "Unhandled store\n");
      return false;
    }
  }

  if (HaveImm && HaveReg) {
    LLVM_DEBUG(dbgs() << "Cannot merge store register and store imm\n");
    return false;
  }

  MachineInstr *FirstSt = OG.front();
  DebugLoc DL = OG.back()->getDebugLoc();
  const MachineMemOperand &OldM = getMemTarget(FirstSt);
  MachineMemOperand *NewM =
      MF->getMachineMemOperand(OldM.getPointerInfo(), OldM.getFlags(),
                               TotalSize, OldM.getAlign(), OldM.getAAInfo());
  MachineInstr *StI;
  MachineOperand &MR =
      (HII->isPostIncrement(*FirstSt) ? FirstSt->getOperand(1)
                                      : FirstSt->getOperand(0));
  auto SecondSt = OG.back();
  if (HaveReg) {
    MachineOperand FReg =
        (HII->isPostIncrement(*FirstSt) ? FirstSt->getOperand(3)
                                        : FirstSt->getOperand(2));
    // Post increments appear first in the sorted group.
    // Cannot have a post increment for the second instruction
    assert(!HII->isPostIncrement(*SecondSt) && "Unexpected PostInc");
    MachineOperand SReg = SecondSt->getOperand(2);
    assert(FReg.isReg() && SReg.isReg() &&
           "Cannot merge store register and store imm");
    const MCInstrDesc &CombD = TII->get(Hexagon::A2_combinew);
    Register VReg =
        MF->getRegInfo().createVirtualRegister(&Hexagon::DoubleRegsRegClass);
    MachineInstr *CombI = BuildMI(*MF, DL, CombD, VReg).add(SReg).add(FReg);
    NG.push_back(CombI);

    if (FirstSt->getOpcode() == Hexagon::S2_storeri_pi) {
      const MCInstrDesc &StD = TII->get(Hexagon::S2_storerd_pi);
      auto IncDestMO = FirstSt->getOperand(0);
      auto IncMO = FirstSt->getOperand(2);
      StI =
          BuildMI(*MF, DL, StD).add(IncDestMO).add(MR).add(IncMO).addReg(VReg);
    } else {
      const MCInstrDesc &StD = TII->get(Hexagon::S2_storerd_io);
      auto OffMO = FirstSt->getOperand(1);
      StI = BuildMI(*MF, DL, StD).add(MR).add(OffMO).addReg(VReg);
    }
    StI->addMemOperand(*MF, NewM);
    NG.push_back(StI);
    return true;
  }

  // Handle store immediates
  // There are no post increment store immediates on Hexagon
  assert(!HII->isPostIncrement(*FirstSt) && "Unexpected PostInc");
  auto Off = FirstSt->getOperand(1).getImm();
  if (TotalSize == 8) {
    // Create vreg = A2_tfrsi #Acc; nreg = combine(#s32, vreg); memd = nreg
    uint64_t Mask = 0xFFFFFFFFU;
    int LowerAcc = int(Mask & Acc);
    int UpperAcc = Acc >> 32;
    Register DReg =
        MF->getRegInfo().createVirtualRegister(&Hexagon::DoubleRegsRegClass);
    MachineInstr *CombI;
    if (Acc != 0) {
      const MCInstrDesc &TfrD = TII->get(Hexagon::A2_tfrsi);
      const TargetRegisterClass *RC = TII->getRegClass(TfrD, 0, TRI);
      Register VReg = MF->getRegInfo().createVirtualRegister(RC);
      MachineInstr *TfrI = BuildMI(*MF, DL, TfrD, VReg).addImm(LowerAcc);
      NG.push_back(TfrI);
      const MCInstrDesc &CombD = TII->get(Hexagon::A4_combineir);
      CombI = BuildMI(*MF, DL, CombD, DReg)
                  .addImm(UpperAcc)
                  .addReg(VReg, RegState::Kill);
    }
    // If immediates are 0, we do not need A2_tfrsi
    else {
      const MCInstrDesc &CombD = TII->get(Hexagon::A4_combineii);
      CombI = BuildMI(*MF, DL, CombD, DReg).addImm(0).addImm(0);
    }
    NG.push_back(CombI);
    const MCInstrDesc &StD = TII->get(Hexagon::S2_storerd_io);
    StI =
        BuildMI(*MF, DL, StD).add(MR).addImm(Off).addReg(DReg, RegState::Kill);
  } else if (Acc < 0x10000) {
    // Create mem[hw] = #Acc
    unsigned WOpc = (TotalSize == 2)   ? Hexagon::S4_storeirh_io
                    : (TotalSize == 4) ? Hexagon::S4_storeiri_io
                                       : 0;
    assert(WOpc && "Unexpected size");

    int Val = (TotalSize == 2) ? int16_t(Acc) : int(Acc);
    const MCInstrDesc &StD = TII->get(WOpc);
    StI = BuildMI(*MF, DL, StD).add(MR).addImm(Off).addImm(Val);
  } else {
    // Create vreg = A2_tfrsi #Acc; mem[hw] = vreg
    const MCInstrDesc &TfrD = TII->get(Hexagon::A2_tfrsi);
    const TargetRegisterClass *RC = TII->getRegClass(TfrD, 0, TRI);
    Register VReg = MF->getRegInfo().createVirtualRegister(RC);
    MachineInstr *TfrI = BuildMI(*MF, DL, TfrD, VReg).addImm(int(Acc));
    NG.push_back(TfrI);

    unsigned WOpc = (TotalSize == 2)   ? Hexagon::S2_storerh_io
                    : (TotalSize == 4) ? Hexagon::S2_storeri_io
                                       : 0;
    assert(WOpc && "Unexpected size");

    const MCInstrDesc &StD = TII->get(WOpc);
    StI =
        BuildMI(*MF, DL, StD).add(MR).addImm(Off).addReg(VReg, RegState::Kill);
  }
  StI->addMemOperand(*MF, NewM);
  NG.push_back(StI);

  return true;
}

/// Given an "old group" OG of loads, create a "new group" NG of instructions
/// to replace them.  Ideally, NG would only have a single instruction in it,
/// but that may only be possible for double register loads.
bool HexagonLoadStoreWidening::createWideLoads(InstrGroup &OG, InstrGroup &NG,
                                               unsigned TotalSize) {
  LLVM_DEBUG(dbgs() << "Creating wide loads\n");
  // XXX Current limitations:
  // - only expect stores of immediate values in OG,
  // - only handle a TotalSize of up to 8
  if (TotalSize > MaxWideSize)
    return false;
  assert(OG.size() == 2 && "Expecting two elements in Instruction Group.");

  MachineInstr *FirstLd = OG.front();
  const MachineMemOperand &OldM = getMemTarget(FirstLd);
  MachineMemOperand *NewM =
      MF->getMachineMemOperand(OldM.getPointerInfo(), OldM.getFlags(),
                               TotalSize, OldM.getAlign(), OldM.getAAInfo());

  MachineOperand &MR = FirstLd->getOperand(0);
  MachineOperand &MRBase =
      (HII->isPostIncrement(*FirstLd) ? FirstLd->getOperand(2)
                                      : FirstLd->getOperand(1));
  DebugLoc DL = OG.back()->getDebugLoc();

  // Create the double register Load Instruction.
  Register NewMR = MRI->createVirtualRegister(&Hexagon::DoubleRegsRegClass);
  MachineInstr *LdI;

  // Post increments appear first in the sorted group
  if (FirstLd->getOpcode() == Hexagon::L2_loadri_pi) {
    auto IncDestMO = FirstLd->getOperand(1);
    auto IncMO = FirstLd->getOperand(3);
    LdI = BuildMI(*MF, DL, TII->get(Hexagon::L2_loadrd_pi))
              .addDef(NewMR, getKillRegState(MR.isKill()), MR.getSubReg())
              .add(IncDestMO)
              .add(MRBase)
              .add(IncMO);
    LdI->addMemOperand(*MF, NewM);
  } else {
    auto OffMO = FirstLd->getOperand(2);
    LdI = BuildMI(*MF, DL, TII->get(Hexagon::L2_loadrd_io))
              .addDef(NewMR, getKillRegState(MR.isKill()), MR.getSubReg())
              .add(MRBase)
              .add(OffMO);
    LdI->addMemOperand(*MF, NewM);
  }
  NG.push_back(LdI);

  auto getHalfReg = [&](MachineInstr *DoubleReg, unsigned SubReg,
                        MachineInstr *DstReg) {
    Register DestReg = DstReg->getOperand(0).getReg();
    return BuildMI(*MF, DL, TII->get(Hexagon::COPY), DestReg)
        .addReg(NewMR, getKillRegState(LdI->isKill()), SubReg);
  };

  MachineInstr *LdI_lo = getHalfReg(LdI, Hexagon::isub_lo, FirstLd);
  MachineInstr *LdI_hi = getHalfReg(LdI, Hexagon::isub_hi, OG.back());
  NG.push_back(LdI_lo);
  NG.push_back(LdI_hi);

  return true;
}

// Replace instructions from the old group OG with instructions from the
// new group NG.  Conceptually, remove all instructions in OG, and then
// insert all instructions in NG, starting at where the first instruction
// from OG was (in the order in which they appeared in the basic block).
// (The ordering in OG does not have to match the order in the basic block.)
bool HexagonLoadStoreWidening::replaceInsts(InstrGroup &OG, InstrGroup &NG) {
  LLVM_DEBUG({
    dbgs() << "Replacing:\n";
    for (auto I : OG)
      dbgs() << "  " << *I;
    dbgs() << "with\n";
    for (auto I : NG)
      dbgs() << "  " << *I;
  });

  MachineBasicBlock *MBB = OG.back()->getParent();
  MachineBasicBlock::iterator InsertAt = MBB->end();

  // Need to establish the insertion point.
  // For loads the best one is right before the first load in the OG,
  // but in the order in which the insts occur in the program list.
  // For stores the best point is right after the last store in the OG.
  // Since the ordering in OG does not correspond
  // to the order in the program list, we need to do some work to find
  // the insertion point.

  // Create a set of all instructions in OG (for quick lookup).
  InstrSet OldMemInsts(llvm::from_range, OG);

  if (Mode == WideningMode::Load) {
    // Find the first load instruction in the block that is present in OG.
    for (auto &I : *MBB) {
      if (OldMemInsts.count(&I)) {
        InsertAt = I;
        break;
      }
    }

    assert((InsertAt != MBB->end()) && "Cannot locate any load from the group");

    for (auto *I : NG)
      MBB->insert(InsertAt, I);
  } else {
    // Find the last store instruction in the block that is present in OG.
    auto I = MBB->rbegin();
    for (; I != MBB->rend(); ++I) {
      if (OldMemInsts.count(&(*I))) {
        InsertAt = (*I).getIterator();
        break;
      }
    }

    assert((I != MBB->rend()) && "Cannot locate any store from the group");

    for (auto I = NG.rbegin(); I != NG.rend(); ++I)
      MBB->insertAfter(InsertAt, *I);
  }

  for (auto *I : OG)
    I->eraseFromParent();

  return true;
}

// Break up the group into smaller groups, each of which can be replaced by
// a single wide load/store.  Widen each such smaller group and replace the old
// instructions with the widened ones.
bool HexagonLoadStoreWidening::processGroup(InstrGroup &Group) {
  bool Changed = false;
  InstrGroup::iterator I = Group.begin(), E = Group.end();
  InstrGroup OG, NG; // Old and new groups.
  unsigned CollectedSize;

  while (I != E) {
    OG.clear();
    NG.clear();

    bool Succ = selectInsts(I++, E, OG, CollectedSize, MaxWideSize) &&
                createWideInsts(OG, NG, CollectedSize) && replaceInsts(OG, NG);
    if (!Succ)
      continue;

    assert(OG.size() > 1 && "Created invalid group");
    assert(std::distance(I, E) + 1 >= int(OG.size()) && "Too many elements");
    I += OG.size() - 1;

    Changed = true;
  }

  return Changed;
}

// Process a single basic block: create the load/store groups, and replace them
// with the widened insts, if possible.  Processing of each basic block
// is independent from processing of any other basic block.  This transfor-
// mation could be stopped after having processed any basic block without
// any ill effects (other than not having performed widening in the unpro-
// cessed blocks).  Also, the basic blocks can be processed in any order.
bool HexagonLoadStoreWidening::processBasicBlock(MachineBasicBlock &MBB) {
  InstrGroupList SGs;
  bool Changed = false;

  // To prevent long compile time check for max BB size.
  if (MBB.size() > MaxMBBSizeForLoadStoreWidening)
    return false;

  createGroups(MBB, SGs);

  auto Less = [this](const MachineInstr *A, const MachineInstr *B) -> bool {
    return getOffset(A) < getOffset(B);
  };
  for (auto &G : SGs) {
    assert(G.size() > 1 && "Group with fewer than 2 elements");
    llvm::sort(G, Less);

    Changed |= processGroup(G);
  }

  return Changed;
}

bool HexagonLoadStoreWidening::run() {
  bool Changed = false;

  for (auto &B : *MF)
    Changed |= processBasicBlock(B);

  return Changed;
}

FunctionPass *llvm::createHexagonStoreWidening() {
  return new HexagonStoreWidening();
}

FunctionPass *llvm::createHexagonLoadWidening() {
  return new HexagonLoadWidening();
}
