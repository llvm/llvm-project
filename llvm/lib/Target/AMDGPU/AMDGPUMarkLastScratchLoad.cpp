//===-- AMDGPUMarkLastScratchLoad.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark scratch load/spill instructions which are guaranteed to be the last time
// this scratch slot is used so it can be evicted from caches.
//
// Liveness is computed from the spill code's memory operands, so every spill
// slot access needs one. Must run after anything that reorders spill code
// (e.g. the post-RA scheduler) and before SIMemoryLegalizer.
//
// TODO: Handle general stack accesses not just spilling.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/PseudoSourceValue.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-last-scratch-load"

static cl::opt<unsigned> MarkLastScratchLoadUnitSize(
    "amdgpu-mark-last-scratch-load-unit-size",
    cl::desc("Granularity in bytes of spill slot liveness when marking last "
             "scratch loads"),
    cl::init(4));

namespace {

class AMDGPUMarkLastScratchLoad {
private:
  const unsigned UnitSize = MarkLastScratchLoadUnitSize;

  struct SlotAccess {
    const MachineMemOperand *MMO;
    unsigned Begin, End; // Liveness units [Begin, End), not bytes.
  };

  const SIInstrInfo *SII = nullptr;
  // Liveness units [first, last) of each register allocator spill slot.
  DenseMap<int, std::pair<unsigned, unsigned>> SlotUnits;
  unsigned NumUnits = 0;

  void collectSpillSlots(const MachineFunction &MF);
  void getSlotAccesses(const MachineInstr &MI,
                       SmallVectorImpl<SlotAccess> &Accesses) const;
  void markLastUse(MachineInstr &MI, const MachineMemOperand *MMO) const;

public:
  bool run(MachineFunction &MF);
};

class AMDGPUMarkLastScratchLoadLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUMarkLastScratchLoadLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "AMDGPU Mark Last Scratch Load";
  }
};

} // end anonymous namespace

[[maybe_unused]] static raw_ostream &operator<<(raw_ostream &OS,
                                                const BitVector &BV) {
  OS << '{';
  ListSeparator LS(" ");
  for (unsigned U : BV.set_bits())
    OS << LS << U;
  return OS << '}';
}

bool AMDGPUMarkLastScratchLoadLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return AMDGPUMarkLastScratchLoad().run(MF);
}

PreservedAnalyses
AMDGPUMarkLastScratchLoadPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  AMDGPUMarkLastScratchLoad().run(MF);
  return PreservedAnalyses::all();
}

void AMDGPUMarkLastScratchLoad::markLastUse(
    MachineInstr &MI, const MachineMemOperand *MMO) const {
  MachineFunction &MF = *MI.getMF();
  MachineMemOperand *NewMMO =
      MF.getMachineMemOperand(MMO, MMO->getFlags() | MOLastUse);
  SmallVector<MachineMemOperand *, 2> MMOs;
  for (MachineMemOperand *Op : MI.memoperands())
    MMOs.push_back(Op == MMO ? NewMMO : Op);
  MI.setMemRefs(MF, MMOs);

  // Spill pseudos get cpol from the MMO when they are lowered.
  if (MachineOperand *CPol = SII->getNamedOperand(MI, AMDGPU::OpName::cpol))
    CPol->setImm((CPol->getImm() & ~AMDGPU::CPol::TH) | AMDGPU::CPol::TH_LU);

  if (MI.isBundled()) {
    MachineBasicBlock::instr_iterator Start = getBundleStart(MI.getIterator());
    SmallVector<const MachineInstr *, 8> Members;
    for (const MachineInstr &I :
         make_range(std::next(Start), getBundleEnd(MI.getIterator())))
      Members.push_back(&I);
    Start->cloneMergedMemRefs(MF, Members);
  }
  LLVM_DEBUG(dbgs() << "  Found last load: " << MI);
}

void AMDGPUMarkLastScratchLoad::collectSpillSlots(const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SIMachineFunctionInfo &FuncInfo = *MF.getInfo<SIMachineFunctionInfo>();

  // Prolog/epilog spill code is created by frame lowering, not by the register
  // allocator.
  SmallDenseSet<int, 8> FrameLoweringSlots;
  if (MFI.isCalleeSavedInfoValid())
    for (const CalleeSavedInfo &CS : MFI.getCalleeSavedInfo())
      if (!CS.isSpilledToReg())
        FrameLoweringSlots.insert(CS.getFrameIdx());
  for (const auto &[Reg, FI] : FuncInfo.getWWMSpills())
    FrameLoweringSlots.insert(FI);
  for (const auto &[Reg, Info] : FuncInfo.getPrologEpilogSGPRSpills())
    if (Info.getKind() == SGPRSaveKind::SPILL_TO_MEM)
      FrameLoweringSlots.insert(Info.getIndex());
  if (std::optional<int> FI = FuncInfo.getOptionalScavengeFI())
    FrameLoweringSlots.insert(*FI);

  for (int FI = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd(); FI != E;
       ++FI) {
    if (!MFI.isSpillSlotObjectIndex(FI) || MFI.isDeadObjectIndex(FI) ||
        FrameLoweringSlots.contains(FI))
      continue;
    unsigned First = NumUnits;
    NumUnits += divideCeil(MFI.getObjectSize(FI), UnitSize);
    SlotUnits[FI] = {First, NumUnits};
    LLVM_DEBUG(dbgs() << "  fi#" << FI << ": units [" << First << ", "
                      << NumUnits << ")\n");
  }
}

// Get the liveness units of tracked spill slots that \p MI accesses.
// For loads report every unit they touch. For stores only the units they
// completely overwrite.
void AMDGPUMarkLastScratchLoad::getSlotAccesses(
    const MachineInstr &MI, SmallVectorImpl<SlotAccess> &Accesses) const {
  Accesses.clear();
  if (MI.isBundle() || !MI.mayLoadOrStore())
    return;
  for (const MachineMemOperand *MMO : MI.memoperands()) {
    const auto *FS =
        dyn_cast_if_present<FixedStackPseudoSourceValue>(MMO->getPseudoValue());
    if (!FS || !MMO->getSize().hasValue() || MMO->getOffset() < 0)
      continue;
    auto It = SlotUnits.find(FS->getFrameIndex());
    if (It == SlotUnits.end())
      continue;
    auto [First, Last] = It->second;
    uint64_t Begin = MMO->getOffset();
    uint64_t End = Begin + MMO->getSize().getValue().getKnownMinValue();
    uint64_t I =
        MMO->isStore() ? divideCeil(Begin, UnitSize) : Begin / UnitSize;
    uint64_t E = MMO->isStore() ? End / UnitSize : divideCeil(End, UnitSize);
    I = std::min<uint64_t>(First + I, Last);
    E = std::min<uint64_t>(First + E, Last);
    if (I < E)
      Accesses.push_back({MMO, unsigned(I), unsigned(E)});
  }
}

bool AMDGPUMarkLastScratchLoad::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.getGeneration() < AMDGPUSubtarget::GFX12)
    return false;
  if (UnitSize == 0)
    reportFatalUsageError("-amdgpu-mark-last-scratch-load-unit-size must be "
                          "non-zero");

  SII = ST.getInstrInfo();

  // Assign a range of liveness units to each tracked spill slot.
  collectSpillSlots(MF);
  if (NumUnits == 0) {
    LLVM_DEBUG(dbgs() << "No spill slots, skipping\n");
    return false;
  }

  // Per block:
  // Gen = units read before being overwritten
  // Kill = units fully overwritten
  const unsigned NumBlocks = MF.getNumBlockIDs();
  SmallVector<BitVector> Gen(NumBlocks, BitVector(NumUnits));
  SmallVector<BitVector> Kill(NumBlocks, BitVector(NumUnits));
  SmallVector<SlotAccess, 2> Accesses;
  for (const MachineBasicBlock &MBB : MF) {
    BitVector &G = Gen[MBB.getNumber()];
    BitVector &K = Kill[MBB.getNumber()];
    for (const MachineInstr &MI : reverse(MBB.instrs())) {
      getSlotAccesses(MI, Accesses);
      for (const SlotAccess &A : Accesses) {
        if (A.MMO->isStore()) {
          G.reset(A.Begin, A.End);
          K.set(A.Begin, A.End);
        } else {
          G.set(A.Begin, A.End);
        }
      }
    }
    LLVM_DEBUG(dbgs() << "  " << printMBBReference(MBB) << ": gen " << G
                      << " kill " << K << '\n');
  }

  SmallVector<BitVector> LiveIn(NumBlocks, BitVector(NumUnits));
  auto LiveOut = [&](const MachineBasicBlock &MBB) {
    BitVector Live(NumUnits);
    for (const MachineBasicBlock *Succ : MBB.successors())
      Live |= LiveIn[Succ->getNumber()];
    return Live;
  };

  // LiveIn[N] = Gen[N] | (LiveOut(N) & ~Kill[N])
  // Worklist in post order, so successors are usually computed first. When a
  // block's LiveIn changes its predecessors are queued again. This revisits
  // the blocks that saw incomplete LiveIn across a loop back edge.
  SmallVector<const MachineBasicBlock *> PO;
  SmallVector<int> POIndex(NumBlocks, -1);
  for (const MachineBasicBlock *MBB : post_order(&MF)) {
    POIndex[MBB->getNumber()] = PO.size();
    PO.push_back(MBB);
  }
  BitVector Pending(PO.size(), true);
  for (int I = Pending.find_first(); I >= 0;) {
    Pending.reset(I);
    const MachineBasicBlock *MBB = PO[I];
    unsigned N = MBB->getNumber();
    BitVector Live = LiveOut(*MBB);
    Live.reset(Kill[N]);
    Live |= Gen[N];
    // If change was made, update LiveIn and mark predecessors as pending.
    if (Live != LiveIn[N]) {
      LLVM_DEBUG(dbgs() << "  " << printMBBReference(*MBB) << ": live-in "
                        << Live << '\n');
      LiveIn[N] = std::move(Live);
      for (const MachineBasicBlock *Pred : MBB->predecessors()) {
        // Predecessors unreachable from the entry have no post order index.
        int P = POIndex[Pred->getNumber()];
        if (P >= 0)
          Pending.set(P);
      }
    }
    // Loop back edge sources come earlier in post order, so wrap around.
    I = Pending.find_next(I);
    if (I < 0)
      I = Pending.find_first();
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    BitVector Live = LiveOut(MBB);
    for (MachineInstr &MI : reverse(MBB.instrs())) {
      getSlotAccesses(MI, Accesses);
      for (const SlotAccess &A : Accesses) {
        if (A.MMO->isStore()) {
          Live.reset(A.Begin, A.End);
          continue;
        }
        if (!(A.MMO->getFlags() & MOLastUse) &&
            Live.find_first_in(A.Begin, A.End) < 0) {
          markLastUse(MI, A.MMO);
          Changed = true;
        }
        Live.set(A.Begin, A.End);
      }
    }
  }
  return Changed;
}

char AMDGPUMarkLastScratchLoadLegacy::ID = 0;

char &llvm::AMDGPUMarkLastScratchLoadID = AMDGPUMarkLastScratchLoadLegacy::ID;

INITIALIZE_PASS(AMDGPUMarkLastScratchLoadLegacy, DEBUG_TYPE,
                "AMDGPU Mark last scratch load", false, false)
