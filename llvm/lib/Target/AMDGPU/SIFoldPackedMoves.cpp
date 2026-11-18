//===- SIFoldPackedMoves.cpp - Combine residual tuple copies
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIFoldPackedMoves.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "si-fold-packed-moves"

STATISTIC(NumPackedMoves, "Number of copy groups combined into packed moves");
STATISTIC(NumWideCopies, "Number of copy groups combined into wide copies");
STATISTIC(NumWidenedSources,
          "Number of packed-move sources widened without copies");

namespace {

constexpr unsigned NumPackedMoveLanes = 2;

/// Index of the V_PK_MOV_B32 operand that holds the source of \p Lane. Each
/// source follows its modifiers.
constexpr unsigned getSourceOpIdx(unsigned Lane) { return 2 + 2 * Lane; }

struct CopyLane {
  MachineInstr *MI = nullptr;
  Register Dst;
  Register Src;
  unsigned DstChannel;
  unsigned SrcChannel;
  /// Position of the copy in the scan of its block.
  unsigned Pos;
};

/// A read or write of some lanes of a virtual register.
struct RegAccess {
  unsigned Pos;
  LaneBitmask Lanes;
  bool IsDef;
};

/// The class each virtual register needs for a packed move. A class wider
/// than the current one means that the register has to be widened.
using RegClassMap = SmallMapVector<Register, const TargetRegisterClass *, 4>;

class SIFoldPackedMoves {
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals &LIS;
  SmallSet<Register, 16> ChangedRegs;
  SmallSet<Register, 32> MFMARegs;

  // State of the scan of a block. It only describes the instructions since
  // the last barrier, because copies never move across one.
  /// Accesses to each virtual register.
  DenseMap<Register, SmallVector<RegAccess, 4>> Accesses;
  /// Position of the last write to each register unit.
  DenseMap<MCRegUnit, unsigned> LastUnitDef;
  /// The copy still waiting for a partner, for each destination and the first
  /// channel of the lane pair it writes.
  DenseMap<std::pair<Register, unsigned>, CopyLane> PendingCopies;

  std::optional<CopyLane> getCopy(MachineInstr &MI, unsigned Pos) const;
  bool isBarrier(const MachineInstr &MI) const;
  void resetScan();
  void recordAccesses(const MachineInstr &MI, unsigned Pos);
  bool isUntouchedSince(const CopyLane &Copy, unsigned Pos) const;
  bool combineWithPending(const CopyLane &Copy);
  void updatePhysicalSources(ArrayRef<CopyLane> Copies,
                             MachineInstr &InsertBefore);
  unsigned getPairSubReg(const TargetRegisterClass &RC, unsigned Channel) const;
  MCRegister getPhysicalPair(const CopyLane &Copy, unsigned Lane) const;
  const TargetRegisterClass *planWidening(Register Reg, unsigned NumRegs) const;
  bool planPackedMove(ArrayRef<const CopyLane *> Lanes,
                      RegClassMap &NewClasses) const;
  void widenRegister(Register Reg, const TargetRegisterClass *WideRC);
  void addPackedOperands(MachineInstrBuilder &MIB,
                         ArrayRef<const CopyLane *> Lanes) const;
  MachineInstr *emitCombinedMove(ArrayRef<const CopyLane *> Lanes,
                                 MachineInstr &First,
                                 MachineInstr &InsertBefore, bool IsCopy) const;
  MachineInstr *combine(ArrayRef<CopyLane> Copies, MachineInstr &InsertBefore);
  void collectMFMARegisters(MachineFunction &MF);

public:
  explicit SIFoldPackedMoves(LiveIntervals &LIS) : LIS(LIS) {}
  bool run(MachineFunction &MF);
};

class SIFoldPackedMovesLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIFoldPackedMovesLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SI Fold Packed Moves"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return SIFoldPackedMoves(getAnalysis<LiveIntervalsWrapperPass>().getLIS())
        .run(MF);
  }
};

} // namespace

INITIALIZE_PASS_BEGIN(SIFoldPackedMovesLegacy, DEBUG_TYPE,
                      "SI Fold Packed Moves", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(SIFoldPackedMovesLegacy, DEBUG_TYPE, "SI Fold Packed Moves",
                    false, false)

char SIFoldPackedMovesLegacy::ID = 0;
char &llvm::SIFoldPackedMovesLegacyID = SIFoldPackedMovesLegacy::ID;

std::optional<CopyLane> SIFoldPackedMoves::getCopy(MachineInstr &MI,
                                                   unsigned Pos) const {
  if (!MI.isCopy() || MI.isBundled() || MI.getNumOperands() != 2)
    return std::nullopt;
  // An instruction-reference debug value names this narrow definition.
  if (MI.peekDebugInstrNum())
    return std::nullopt;
  const MachineOperand &Dst = MI.getOperand(0);
  const MachineOperand &Src = MI.getOperand(1);
  if (!Dst.getReg().isVirtual() || !Src.isReg() || Src.isUndef() ||
      Dst.isEarlyClobber() || !Dst.getSubReg() ||
      TRI->getSubRegIdxSize(Dst.getSubReg()) != 32 ||
      !TRI->hasVGPRs(MRI->getRegClass(Dst.getReg())))
    return std::nullopt;

  unsigned SrcChannel;
  if (Src.getReg().isPhysical()) {
    if (Src.getSubReg() || !AMDGPU::VGPR_32RegClass.contains(Src.getReg()))
      return std::nullopt;
    SrcChannel = TRI->getHWRegIndex(Src.getReg());
  } else {
    unsigned SrcSize =
        Src.getSubReg()
            ? TRI->getSubRegIdxSize(Src.getSubReg())
            : TRI->getRegSizeInBits(*MRI->getRegClass(Src.getReg()));
    if (SrcSize != 32)
      return std::nullopt;
    SrcChannel =
        Src.getSubReg() ? TRI->getChannelFromSubReg(Src.getSubReg()) : 0;
  }
  unsigned DstChannel = TRI->getChannelFromSubReg(Dst.getSubReg());
  return CopyLane{&MI, Dst.getReg(), Src.getReg(), DstChannel, SrcChannel, Pos};
}

/// Copies are neither combined across \p MI nor moved across it.
bool SIFoldPackedMoves::isBarrier(const MachineInstr &MI) const {
  // Register masks are not recorded as accesses.
  return MI.isBundled() || MI.isMetaInstruction() || MI.isCall() ||
         MI.hasUnmodeledSideEffects() ||
         TII->isSchedulingBoundary(MI, MI.getParent(), *MI.getMF()) ||
         llvm::any_of(MI.operands(),
                      [](const MachineOperand &MO) { return MO.isRegMask(); });
}

void SIFoldPackedMoves::resetScan() {
  Accesses.clear();
  LastUnitDef.clear();
  PendingCopies.clear();
}

void SIFoldPackedMoves::recordAccesses(const MachineInstr &MI, unsigned Pos) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MO.getReg().isPhysical()) {
      // A physical register can only be a copy source, which reads do not
      // disturb.
      if (!MO.isDef())
        continue;
      for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg())) {
        unsigned &LastDef = LastUnitDef[Unit];
        LastDef = std::max(LastDef, Pos);
      }
      continue;
    }
    if (MO.isUse() && MO.isUndef())
      continue;
    // A partial def preserves the other lanes without reading them, unless it
    // is undef, which makes them undefined.
    LaneBitmask Lanes = MO.isDef() && MO.isUndef()
                            ? LaneBitmask::getAll()
                            : TRI->getSubRegIndexLaneMask(MO.getSubReg());
    Accesses[MO.getReg()].push_back({Pos, Lanes, MO.isDef()});
  }
}

/// Returns true if no instruction after position \p Pos reads or writes the
/// destination of \p Copy, or writes its source. \p Copy can then be moved
/// across all of them.
bool SIFoldPackedMoves::isUntouchedSince(const CopyLane &Copy,
                                         unsigned Pos) const {
  auto IsAccessedSince = [&](Register Reg, LaneBitmask Lanes, bool DefsOnly) {
    auto It = Accesses.find(Reg);
    return It != Accesses.end() &&
           llvm::any_of(It->second, [&](const RegAccess &Access) {
             return Access.Pos > Pos && (Access.IsDef || !DefsOnly) &&
                    (Access.Lanes & Lanes).any();
           });
  };

  const MachineOperand &Dst = Copy.MI->getOperand(0);
  const MachineOperand &Src = Copy.MI->getOperand(1);
  // Moving an undef partial definition also moves the point at which the
  // other lanes become undefined. Do not cross accesses to those lanes.
  LaneBitmask DstLanes = Dst.isUndef()
                             ? LaneBitmask::getAll()
                             : TRI->getSubRegIndexLaneMask(Dst.getSubReg());
  if (IsAccessedSince(Copy.Dst, DstLanes, /*DefsOnly=*/false))
    return false;

  if (Copy.Src.isVirtual())
    return !IsAccessedSince(Copy.Src,
                            TRI->getSubRegIndexLaneMask(Src.getSubReg()),
                            /*DefsOnly=*/true);
  return llvm::none_of(TRI->regunits(Copy.Src.asMCReg()), [&](MCRegUnit Unit) {
    return LastUnitDef.lookup(Unit) > Pos;
  });
}

void SIFoldPackedMoves::updatePhysicalSources(ArrayRef<CopyLane> Copies,
                                              MachineInstr &InsertBefore) {
  SlotIndex UseIdx = LIS.getInstructionIndex(InsertBefore).getRegSlot();
  bool IsSink = &InsertBefore == Copies.back().MI;
  for (const CopyLane &Copy : Copies) {
    if (!Copy.Src.isPhysical())
      continue;
    // A sunk copy reads its source after the kills that it moves past.
    if (IsSink) {
      for (auto I = Copy.MI->getIterator(); I != InsertBefore.getIterator();
           ++I)
        I->clearRegisterKills(Copy.Src, TRI);
    }
    for (MCRegUnit Unit : TRI->regunits(Copy.Src.asMCReg())) {
      LiveRange &LR = LIS.getRegUnit(Unit);
      if (Copy.MI == &InsertBefore)
        continue;
      if (IsSink) {
        LIS.extendToIndices(LR, {UseIdx});
        continue;
      }

      // Hoisting can remove the last use of a physical source. Keep the
      // remaining uses, rather than leaving an endpoint on an erased COPY.
      SlotIndex OldUseIdx = LIS.getInstructionIndex(*Copy.MI).getRegSlot();
      auto Segment = LR.find(OldUseIdx.getPrevSlot());
      assert(Segment != LR.end() && Segment->start <= UseIdx &&
             Segment->end >= OldUseIdx && "hoisted source must be live");
      if (Segment->end != OldUseIdx)
        continue;
      SlotIndex LastUseIdx = UseIdx;
      for (auto I = std::next(InsertBefore.getIterator());
           I != Copy.MI->getIterator(); ++I) {
        if (I->isDebugInstr() ||
            llvm::any_of(
                Copies, [&](const CopyLane &Other) { return Other.MI == &*I; }))
          continue;
        if (llvm::any_of(I->operands(), [&](const MachineOperand &MO) {
              return MO.isReg() && MO.getReg().isPhysical() && MO.readsReg() &&
                     llvm::is_contained(TRI->regunits(MO.getReg().asMCReg()),
                                        Unit);
            }))
          LastUseIdx = LIS.getInstructionIndex(*I).getRegSlot();
      }
      Segment->end = LastUseIdx;
    }
  }
}

/// Returns the subregister index of the 64-bit tuple that contains \p Channel
/// in a register of class \p RC, or 0 if \p RC is itself 64 bits wide.
unsigned SIFoldPackedMoves::getPairSubReg(const TargetRegisterClass &RC,
                                          unsigned Channel) const {
  if (TRI->getRegSizeInBits(RC) == 32 * NumPackedMoveLanes)
    return 0;
  return SIRegisterInfo::getSubRegFromChannel(
      alignDown(Channel, NumPackedMoveLanes), NumPackedMoveLanes);
}

/// Returns the 64-bit register that V_PK_MOV_B32 can read for \p Lane and that
/// has the physical source of \p Copy as one of its halves, if there is one.
MCRegister SIFoldPackedMoves::getPhysicalPair(const CopyLane &Copy,
                                              unsigned Lane) const {
  const TargetRegisterClass *OpRC =
      TII->getRegClass(TII->get(AMDGPU::V_PK_MOV_B32), getSourceOpIdx(Lane));
  unsigned Half = SIRegisterInfo::getSubRegFromChannel(
      Copy.SrcChannel % NumPackedMoveLanes, 1);
  return TRI->getMatchingSuperReg(Copy.Src.asMCReg(), Half, OpRC);
}

/// Returns the class of a register with \p NumRegs channels that can replace
/// \p Reg everywhere while holding it in its low channels, or null.
const TargetRegisterClass *
SIFoldPackedMoves::planWidening(Register Reg, unsigned NumRegs) const {
  const TargetRegisterClass *OldRC = MRI->getRegClass(Reg);
  if (!TRI->hasVGPRs(OldRC))
    return nullptr;
  const TargetRegisterClass *RC = TRI->getVGPRClassForBitWidth(NumRegs * 32);
  if (!RC)
    return nullptr;
  unsigned OldNumRegs = TRI->getRegSizeInBits(*OldRC) / 32;
  unsigned EmbedSubReg = SIRegisterInfo::getSubRegFromChannel(0, OldNumRegs);
  RC = TRI->getMatchingSuperRegClass(RC, OldRC, EmbedSubReg);
  if (!RC)
    return nullptr;

  for (const MachineOperand &MO : MRI->reg_operands(Reg)) {
    const MachineInstr &MI = *MO.getParent();
    if (MI.isDebugInstr())
      continue;
    if (MO.isTied() || MO.isEarlyClobber() || MO.isInternalRead() ||
        MI.isBundled() || MI.isPHI())
      return nullptr;
    unsigned SubReg = TRI->composeSubRegIndices(EmbedSubReg, MO.getSubReg());
    if (!SubReg)
      return nullptr;
    const TargetRegisterClass *OpRC =
        MI.getRegClassConstraint(MO.getOperandNo(), TII, TRI);
    RC = OpRC ? TRI->getMatchingSuperRegClass(RC, OpRC, SubReg)
              : TRI->getSubClassWithSubReg(RC, SubReg);
    if (!RC)
      return nullptr;
  }
  return RC;
}

/// Returns true if the copies in \p Lanes can become one V_PK_MOV_B32, and
/// sets \p NewClasses to the classes that its registers need. Changes nothing.
bool SIFoldPackedMoves::planPackedMove(ArrayRef<const CopyLane *> Lanes,
                                       RegClassMap &NewClasses) const {
  if (llvm::any_of(Lanes, [&](const CopyLane *Copy) {
        return MFMARegs.contains(Copy->Dst) || MFMARegs.contains(Copy->Src);
      }))
    return false;

  // The instruction reads 64-bit tuples: widen sources that are too narrow.
  for (unsigned Lane = 0; Lane != NumPackedMoveLanes; ++Lane) {
    const CopyLane &Copy = *Lanes[Lane];
    if (Copy.Src.isPhysical()) {
      if (!getPhysicalPair(Copy, Lane))
        return false;
      continue;
    }
    const TargetRegisterClass *&RC =
        NewClasses.insert({Copy.Src, MRI->getRegClass(Copy.Src)}).first->second;
    unsigned NumRegs = alignTo(Copy.SrcChannel + 1, NumPackedMoveLanes);
    if (NumRegs > TRI->getRegSizeInBits(*RC) / 32) {
      RC = planWidening(Copy.Src, NumRegs);
      if (!RC)
        return false;
    }
  }
  Register Dst = Lanes[0]->Dst;
  NewClasses.insert({Dst, MRI->getRegClass(Dst)});

  // Every operand must be in the class that the instruction accepts.
  const MCInstrDesc &Desc = TII->get(AMDGPU::V_PK_MOV_B32);
  auto Constrain = [&](Register Reg, unsigned Channel, unsigned OpIdx) {
    const TargetRegisterClass *OpRC = TII->getRegClass(Desc, OpIdx);
    const TargetRegisterClass *&RC = NewClasses.find(Reg)->second;
    unsigned SubReg = getPairSubReg(*RC, Channel);
    RC = SubReg ? TRI->getMatchingSuperRegClass(RC, OpRC, SubReg)
                : TRI->getCommonSubClass(RC, OpRC);
    return RC != nullptr;
  };
  if (!Constrain(Dst, Lanes[0]->DstChannel, 0))
    return false;
  SmallSet<std::pair<Register, unsigned>, NumPackedMoveLanes> ScalarSources;
  for (unsigned Lane = 0; Lane != NumPackedMoveLanes; ++Lane) {
    const CopyLane &Copy = *Lanes[Lane];
    if (Copy.Src.isPhysical())
      continue;
    if (!Constrain(Copy.Src, Copy.SrcChannel, getSourceOpIdx(Lane)))
      return false;
    if (TRI->isSGPRClass(NewClasses.find(Copy.Src)->second))
      ScalarSources.insert({Copy.Src, Copy.SrcChannel / NumPackedMoveLanes});
  }
  return ScalarSources.size() <=
         TII->getSubtarget().getConstantBusLimit(AMDGPU::V_PK_MOV_B32);
}

/// Replaces \p Reg with a new register of the wider class \p WideRC that holds
/// \p Reg in its low channels.
void SIFoldPackedMoves::widenRegister(Register Reg,
                                      const TargetRegisterClass *WideRC) {
  unsigned NumRegs = TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) / 32;
  unsigned EmbedSubReg = SIRegisterInfo::getSubRegFromChannel(0, NumRegs);
  Register Wide = MRI->createVirtualRegister(WideRC);
  for (MachineOperand &MO : make_early_inc_range(MRI->reg_operands(Reg))) {
    bool WasFullDef = MO.isDef() && !MO.getSubReg();
    MO.substVirtReg(Wide, EmbedSubReg, *TRI);
    if (WasFullDef)
      MO.setIsUndef(true);
  }
  ChangedRegs.erase(Reg);
  LIS.removeInterval(Reg);
  ChangedRegs.insert(Wide);
  ++NumWidenedSources;
  // The recorded accesses and pending copies still refer to Reg.
  resetScan();
}

/// Adds the V_PK_MOV_B32 operands that read the sources of \p Lanes. Registers
/// come from the copy instructions, which widening may have rewritten.
void SIFoldPackedMoves::addPackedOperands(
    MachineInstrBuilder &MIB, ArrayRef<const CopyLane *> Lanes) const {
  for (unsigned Lane = 0; Lane != NumPackedMoveLanes; ++Lane) {
    const CopyLane &Copy = *Lanes[Lane];
    Register Reg = Copy.MI->getOperand(1).getReg();
    unsigned SubReg = 0;
    if (Reg.isPhysical())
      Reg = getPhysicalPair(Copy, Lane);
    else
      SubReg = getPairSubReg(*MRI->getRegClass(Reg), Copy.SrcChannel);
    bool ReadsHighHalf = Copy.SrcChannel % NumPackedMoveLanes;
    unsigned Mods =
        SISrcMods::OP_SEL_1 | (ReadsHighHalf ? SISrcMods::OP_SEL_0 : 0);
    MIB.addImm(Mods).addReg(Reg, RegState::Undef, SubReg);
  }
  for (unsigned I = 0; I != 5; ++I)
    MIB.addImm(0);
  // The full source tuple is needed for encoding, but only selected lanes
  // are read. Preserve that precision for both virtual and physical sources.
  for (const CopyLane *Copy : Lanes) {
    const MachineOperand &Src = Copy->MI->getOperand(1);
    MIB.addReg(Src.getReg(), RegState::Implicit, Src.getSubReg());
  }
}

/// Emits the instruction replacing \p Lanes. \p First is the earlier copy.
MachineInstr *SIFoldPackedMoves::emitCombinedMove(
    ArrayRef<const CopyLane *> Lanes, MachineInstr &First,
    MachineInstr &InsertBefore, bool IsCopy) const {
  const CopyLane &Lo = *Lanes[0];
  const CopyLane &Hi = *Lanes[1];
  // Widening rewrites the copies, so read the destination from the copy.
  const MachineOperand &DstMO = First.getOperand(0);
  Register Dst = DstMO.getReg();
  unsigned DstSubReg = getPairSubReg(*MRI->getRegClass(Dst), Lo.DstChannel);
  bool Dead = Lo.MI->getOperand(0).isDead() && Hi.MI->getOperand(0).isDead();
  RegState DstFlags = RegState::Define | getDeadRegState(Dead) |
                      getUndefRegState(DstSubReg && DstMO.isUndef());
  MachineInstrBuilder NewMI =
      BuildMI(*First.getParent(), InsertBefore, First.getDebugLoc(),
              TII->get(IsCopy ? TargetOpcode::COPY : AMDGPU::V_PK_MOV_B32))
          .addReg(Dst, DstFlags, DstSubReg)
          .setMIFlags(First.getFlags());
  if (IsCopy) {
    unsigned SrcSubReg =
        getPairSubReg(*MRI->getRegClass(Lo.Src), Lo.SrcChannel);
    NewMI.addReg(Lo.Src, {}, SrcSubReg);
    ++NumWideCopies;
  } else {
    addPackedOperands(NewMI, Lanes);
    ++NumPackedMoves;
  }
  LLVM_DEBUG(dbgs() << "Combined " << Lanes.size() << " copies into "
                    << *NewMI);
  return NewMI;
}

/// Replaces \p Copies with a single instruction placed at \p InsertBefore,
/// which is the first or the last of them. Returns the new instruction, or
/// null if the copies cannot be combined.
MachineInstr *SIFoldPackedMoves::combine(ArrayRef<CopyLane> Copies,
                                         MachineInstr &InsertBefore) {
  assert(Copies.size() == NumPackedMoveLanes);
  MachineInstr &First = *Copies.front().MI;
  assert((&InsertBefore == &First || &InsertBefore == Copies.back().MI) &&
         "combine at a group endpoint");
  const CopyLane *Lanes[NumPackedMoveLanes] = {};
  for (const CopyLane &Copy : Copies)
    Lanes[Copy.DstChannel % NumPackedMoveLanes] = &Copy;
  const CopyLane &Lo = *Lanes[0];
  const CopyLane &Hi = *Lanes[1];

  // Both halves of one 64-bit tuple, in order, form a plain wider copy.
  bool InOrder = Lo.SrcChannel % NumPackedMoveLanes == 0 &&
                 Hi.SrcChannel == Lo.SrcChannel + 1;
  // Keep ABI copies available to the allocator's physical-register hints.
  if (InOrder && Lo.Src.isPhysical() && Hi.Src.isPhysical())
    return nullptr;
  bool IsCopy = InOrder && Lo.Src == Hi.Src;
  RegClassMap NewClasses;
  if (!IsCopy && !planPackedMove(Lanes, NewClasses))
    return nullptr;

  // The combination is legal, so registers can be changed from here on.
  for (auto [Reg, RC] : NewClasses) {
    if (MRI->getRegClass(Reg)->hasSubClassEq(RC))
      MRI->setRegClass(Reg, RC);
    else
      widenRegister(Reg, RC);
  }
  updatePhysicalSources(Copies, InsertBefore);
  MachineInstr *NewMI = emitCombinedMove(Lanes, First, InsertBefore, IsCopy);

  LIS.ReplaceMachineInstrInMaps(InsertBefore, *NewMI);
  for (const MachineOperand &MO : NewMI->operands()) {
    if (MO.isReg() && MO.getReg().isVirtual())
      ChangedRegs.insert(MO.getReg());
  }
  for (const CopyLane &Copy : Copies) {
    if (Copy.MI != &InsertBefore)
      LIS.RemoveMachineInstrFromMaps(*Copy.MI);
    Copy.MI->eraseFromParent();
  }
  return NewMI;
}

void SIFoldPackedMoves::collectMFMARegisters(MachineFunction &MF) {
  SmallVector<Register, 16> Worklist;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (TII->isMFMA(MI)) {
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.getReg().isVirtual() &&
              MFMARegs.insert(MO.getReg()).second)
            Worklist.push_back(MO.getReg());
        }
      }
    }
  }
  while (!Worklist.empty()) {
    Register Reg = Worklist.pop_back_val();
    for (const MachineInstr &MI : MRI->reg_nodbg_instructions(Reg)) {
      if (!MI.isCopy())
        continue;
      for (unsigned I : {0u, 1u}) {
        Register Other = MI.getOperand(I).getReg();
        if (Other.isVirtual() && MFMARegs.insert(Other).second)
          Worklist.push_back(Other);
      }
    }
  }
}

/// Combines \p Copy with the pending copy to the other lane of its destination
/// pair, if there is one. Returns true if \p Copy was replaced.
bool SIFoldPackedMoves::combineWithPending(const CopyLane &Copy) {
  unsigned FirstChannel =
      Copy.DstChannel / NumPackedMoveLanes * NumPackedMoveLanes;
  if (FirstChannel + NumPackedMoveLanes >
      TRI->getRegSizeInBits(*MRI->getRegClass(Copy.Dst)) / 32)
    return false;

  // Whatever happens to the pending copy, only Copy can pair with a later one.
  std::pair<Register, unsigned> Key(Copy.Dst, FirstChannel);
  CopyLane First = std::exchange(PendingCopies[Key], Copy);
  if (!First.MI || First.DstChannel == Copy.DstChannel ||
      First.MI->getFlags() != Copy.MI->getFlags())
    return false;
  if (Copy.MI->getOperand(0).isUndef())
    return false;
  // Copy must not read what First defines; one instruction replaces both.
  if (Copy.Src == First.Dst && (First.MI->getOperand(0).isUndef() ||
                                First.DstChannel == Copy.SrcChannel))
    return false;

  // The combined instruction replaces Copy if First can sink to it, otherwise
  // First if Copy can be hoisted to it.
  const CopyLane *Anchor;
  if (isUntouchedSince(First, First.Pos))
    Anchor = &Copy;
  else if (isUntouchedSince(Copy, First.Pos))
    Anchor = &First;
  else
    return false;
  MachineInstr *Combined = combine({First, Copy}, *Anchor->MI);
  if (!Combined)
    return false;
  PendingCopies.erase(Key);
  recordAccesses(*Combined, Anchor->Pos);
  return true;
}

bool SIFoldPackedMoves::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasPkMovB32() || MF.getProperties().hasNoVRegs())
    return false;
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  collectMFMARegisters(MF);

  for (MachineBasicBlock &MBB : MF) {
    resetScan();
    unsigned Pos = 0;
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      // Attribute debug accesses to the next non-debug instruction.
      if (MI.isDebugInstr()) {
        recordAccesses(MI, Pos + 1);
        continue;
      }
      ++Pos;
      if (isBarrier(MI)) {
        resetScan();
        continue;
      }
      std::optional<CopyLane> Copy = getCopy(MI, Pos);
      if (Copy && combineWithPending(*Copy))
        continue;
      recordAccesses(MI, Pos);
    }
  }
  for (Register Reg : ChangedRegs) {
    MRI->clearKillFlags(Reg);
    if (LIS.hasInterval(Reg))
      LIS.removeInterval(Reg);
    LiveInterval &LI = LIS.createAndComputeVirtRegInterval(Reg);
    // Replacing partial COPYs with a full definition can disconnect the old
    // and new values when an intervening instruction still uses the old value.
    SmallVector<LiveInterval *, 4> SplitLIs;
    LIS.splitSeparateComponents(LI, SplitLIs);
  }
  return !ChangedRegs.empty();
}

PreservedAnalyses
SIFoldPackedMovesPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  if (MF.getFunction().hasOptNone() ||
      !SIFoldPackedMoves(MFAM.getResult<LiveIntervalsAnalysis>(MF)).run(MF))
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserve<SlotIndexesAnalysis>();
  return PA;
}
