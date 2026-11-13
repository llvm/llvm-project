//===- AMDGPUFetchWindowPlacement.cpp - Place loops in fetch windows ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// GFX950 fetches instructions in aligned 32-byte windows and does not predict
/// branches. Where an innermost loop sits within the window affects:
///  - A branch target whose first instruction crosses a window. It waits for
///    two windows, which at occupancy 1 can also trigger power throttling.
///  - Runs of consecutive VMEM instructions. The LSU coalesces only within the
///    current window and the next one, so a run that does not start on a window
///    boundary can spill into a third window and stall on issue.
///
/// A crossing inside a long run of 8-byte instructions is not charged. Where
/// that term asked for a shift, the new alignment was slower.
///
/// This pass runs after all code size changes. It computes exact byte offsets,
/// scores the eight 4-byte shifts of each innermost loop by the frequency
/// weighted cost above, and inserts the chosen number of s_nop in the coldest
/// block that precedes the loop in layout. A shift that would move a block
/// hotter than the loop being placed is refused when that block is not part of
/// an innermost loop scored later. The 32-byte header alignment from
/// SITargetLowering::getPrefLoopAlignment is dropped for loops handled here.
/// Loops with a user requested alignment (llvm.loop.align, -align-loops) are
/// left alone, and no shift may break a later user requested alignment.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-fetch-window-placement"

static cl::opt<bool>
    EnablePlacement("amdgpu-enable-fetch-window-placement",
                    cl::desc("Place innermost loops within the 32-byte "
                             "instruction fetch window (GFX950)"),
                    cl::init(true), cl::Hidden);

static cl::opt<unsigned>
    MaxLoopBytes("amdgpu-fetch-window-max-loop-bytes",
                 cl::desc("Largest innermost loop considered for placement"),
                 cl::init(4096), cl::Hidden);

static cl::opt<bool>
    PrintDecisions("amdgpu-fetch-window-print",
                   cl::desc("Print fetch window placement decisions"),
                   cl::init(false), cl::Hidden);

static cl::opt<unsigned> EntryPadForTesting(
    "amdgpu-fetch-window-entry-pad-for-testing",
    cl::desc("Insert this many s_nop at kernel entry before placement"),
    cl::init(0), cl::Hidden);

namespace {

constexpr unsigned FetchWindow = 32;
constexpr unsigned NopBytes = 4;
constexpr double TargetCrossCost = 8.0;
constexpr double VMEMSpillCost = 2.0;
constexpr double CostSlack = 1.0;

class AMDGPUFetchWindowPlacement : public MachineFunctionPass {
public:
  static char ID;
  AMDGPUFetchWindowPlacement() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  const SIInstrInfo *TII = nullptr;
  const MachineBlockFrequencyInfo *MBFI = nullptr;
  DenseMap<const MachineBasicBlock *, unsigned> Offset;
  SmallPtrSet<const MachineBasicBlock *, 8> UserAligned;
  SmallPtrSet<const MachineBasicBlock *, 32> InnerBlocks;

  unsigned size(const MachineInstr &MI) const;
  void computeOffsets(MachineFunction &MF);
  unsigned endOfLayoutPred(const MachineBasicBlock &MBB) const;
  double cost(const MachineLoop &L, MachineBasicBlock *First,
              MachineBasicBlock *Last, unsigned Shift, bool KeepAlign) const;
  const MachineBasicBlock *hotCodeMoved(const MachineLoop &L,
                                        const MachineBasicBlock &PadMBB) const;
  bool placeLoop(MachineLoop &L, MachineBasicBlock *&Fence);
};

} // end anonymous namespace

StringRef AMDGPUFetchWindowPlacement::getPassName() const {
  return "AMDGPU Fetch Window Placement";
}

void AMDGPUFetchWindowPlacement::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

unsigned AMDGPUFetchWindowPlacement::size(const MachineInstr &MI) const {
  return MI.isBundle() ? 0 : TII->getInstSizeInBytes(MI);
}

unsigned AMDGPUFetchWindowPlacement::endOfLayoutPred(
    const MachineBasicBlock &MBB) const {
  if (MBB.getIterator() == MBB.getParent()->begin())
    return 0;
  const MachineBasicBlock &Prev = *std::prev(MBB.getIterator());
  unsigned Off = Offset.lookup(&Prev);
  for (const MachineInstr &MI : Prev.instrs())
    Off += size(MI);
  return Off;
}

static unsigned alignPad(unsigned Off, const MachineBasicBlock &MBB) {
  unsigned A = MBB.getAlignment().value();
  unsigned Pad = alignTo(Off, A) - Off;
  unsigned Max = MBB.getMaxBytesForAlignment();
  return (Max && Pad > Max) ? 0 : Pad;
}

void AMDGPUFetchWindowPlacement::computeOffsets(MachineFunction &MF) {
  unsigned Off = 0;
  for (MachineBasicBlock &MBB : MF) {
    Off += alignPad(Off, MBB);
    Offset[&MBB] = Off;
    for (const MachineInstr &MI : MBB.instrs())
      Off += size(MI);
  }
}

/// The only loop alignment the compiler itself sets on GFX950 is the 32-byte
/// fetch window alignment from SITargetLowering::getPrefLoopAlignment, capped
/// at one s_nop. Any other alignment was requested by the user.
static bool hasUserAlignment(const MachineBasicBlock &MBB) {
  if (MBB.getAlignment() == Align(1))
    return false;
  return MBB.getAlignment() != Align(FetchWindow) ||
         MBB.getMaxBytesForAlignment() != NopBytes;
}

/// The llvm.loop.align value of \p L, or 0 if there is none.
static unsigned getLoopAlignMetadata(const MachineLoop &L) {
  MDNode *LoopID = L.getLoopID();
  if (!LoopID)
    return 0;
  for (const MDOperand &MDO : drop_begin(LoopID->operands()))
    if (auto *MD = dyn_cast<MDNode>(MDO))
      if (auto *S = dyn_cast<MDString>(MD->getOperand(0)))
        if (S->getString() == "llvm.loop.align")
          return mdconst::extract<ConstantInt>(MD->getOperand(1))
              ->getZExtValue();
  return 0;
}

static bool isBranchTarget(const MachineBasicBlock &MBB) {
  const MachineBasicBlock *Prev = MBB.getIterator() == MBB.getParent()->begin()
                                      ? nullptr
                                      : &*std::prev(MBB.getIterator());
  for (const MachineBasicBlock *P : MBB.predecessors()) {
    if (P != Prev)
      return true;
    for (const MachineInstr &T : P->terminators())
      if (T.isBranch())
        for (const MachineOperand &MO : T.operands())
          if (MO.isMBB() && MO.getMBB() == &MBB)
            return true;
  }
  return false;
}

/// Cost of the layout range [First, Last] when shifted by \p Shift bytes.
/// Block alignment inside the range is re-evaluated at the new offsets; the
/// alignment of \p First is ignored unless \p KeepAlign is set.
double AMDGPUFetchWindowPlacement::cost(const MachineLoop &L,
                                        MachineBasicBlock *First,
                                        MachineBasicBlock *Last, unsigned Shift,
                                        bool KeepAlign) const {
  double HeaderFreq = MBFI->getBlockFreq(L.getHeader()).getFrequency();
  if (HeaderFreq == 0)
    HeaderFreq = 1;
  double C = 0;
  unsigned Off = endOfLayoutPred(*First) + Shift;
  for (auto I = First->getIterator();; ++I) {
    MachineBasicBlock &MBB = *I;
    if (&MBB != First || KeepAlign)
      Off += alignPad(Off, MBB);
    double W = L.contains(&MBB)
                   ? MBFI->getBlockFreq(&MBB).getFrequency() / HeaderFreq
                   : 0;
    bool Target = isBranchTarget(MBB);
    SmallVector<const MachineInstr *, 64> Insts;
    for (const MachineInstr &MI : MBB.instrs())
      if (size(MI))
        Insts.push_back(&MI);
    // A crossing starts in one 32-byte fetch window and ends in the next.
    // Only the first instruction of a branch target is charged. A long run of
    // 8-byte instructions can cross too, but charging those crossings asked
    // for shifts that were slower, so they add nothing here.
    bool InRun = false;
    unsigned RunLen = 0, RunWindow = 0;
    for (auto [Idx, MIp] : enumerate(Insts)) {
      const MachineInstr &MI = *MIp;
      unsigned Sz = size(MI);
      if (Idx == 0 && Target && (Off % FetchWindow) + Sz > FetchWindow)
        C += W * TargetCrossCost;
      // The load/store unit combines one consecutive VMEM run across the fetch
      // window where the run starts and the next window. Those two windows are
      // 64 bytes. A run that starts at the first byte can use all 64. A run
      // that starts later still has only those two windows, so it reaches a
      // third window sooner, and each instruction from there on stalls on
      // issue. A single VMEM has nothing to combine with, so RunLen > 1 leaves
      // it free.
      if (SIInstrInfo::isVMEM(MI)) {
        if (!InRun) {
          InRun = true;
          RunLen = 0;
          RunWindow = Off / FetchWindow;
        }
        ++RunLen;
        if (RunLen > 1 && (Off + Sz - 1) / FetchWindow > RunWindow + 1)
          C += W * VMEMSpillCost;
      } else {
        InRun = false;
      }
      Off += Sz;
    }
    if (&MBB == Last)
      break;
  }
  return C;
}

/// The first block after \p PadMBB that padding there moves and that is
/// hotter than the header of \p L, or null. Blocks of innermost loops are
/// re-scored when those loops are placed, so they do not count.
const MachineBasicBlock *
AMDGPUFetchWindowPlacement::hotCodeMoved(const MachineLoop &L,
                                         const MachineBasicBlock &PadMBB) const {
  BlockFrequency HeaderFreq = MBFI->getBlockFreq(L.getHeader());
  for (const MachineBasicBlock &MBB :
       make_range(std::next(PadMBB.getIterator()), PadMBB.getParent()->end())) {
    if (MBB.getAlignment() >= Align(FetchWindow) &&
        !MBB.getMaxBytesForAlignment())
      return nullptr;
    if (!InnerBlocks.contains(&MBB) && MBFI->getBlockFreq(&MBB) > HeaderFreq)
      return &MBB;
  }
  return nullptr;
}

bool AMDGPUFetchWindowPlacement::placeLoop(MachineLoop &L,
                                           MachineBasicBlock *&Fence) {
  MachineFunction &MF = *L.getHeader()->getParent();
  MachineBasicBlock *First = nullptr, *Last = nullptr;
  for (MachineBasicBlock &MBB : MF) {
    if (!L.contains(&MBB))
      continue;
    if (!First)
      First = &MBB;
    Last = &MBB;
  }
  if (Fence && Offset.lookup(First) <= Offset.lookup(Fence))
    return false;
  unsigned Bytes = Offset.lookup(Last) - Offset.lookup(First);
  for (const MachineInstr &MI : Last->instrs())
    Bytes += size(MI);

  // Padding goes into the coldest block between the previous placed loop and
  // this one, so earlier placements are not disturbed. An aligned block in
  // between would absorb the padding, so do not look past one.
  MachineBasicBlock *PadMBB = nullptr;
  auto Stop = Fence ? Fence->getIterator() : MF.end();
  for (auto I = First->getIterator(); I != MF.begin();) {
    --I;
    if (I == Stop)
      break;
    if (!PadMBB || MBFI->getBlockFreq(&*I) < MBFI->getBlockFreq(PadMBB))
      PadMBB = &*I;
    if (I->getAlignment() > Align(1))
      break;
  }
  Fence = Last;
  if (!PadMBB || Bytes > MaxLoopBytes)
    return false;

  // Never move a loop whose alignment the user asked for.
  if (getLoopAlignMetadata(L) || UserAligned.contains(First)) {
    if (PrintDecisions)
      errs() << "fetch-window: " << MF.getName() << " loop "
             << printMBBReference(*L.getHeader()) << " user-aligned, skip\n";
    return false;
  }

  // The model does not resolve differences below CostSlack, so act only on a
  // larger gain and then take the least padding within CostSlack of the best.
  double Base = cost(L, First, Last, 0, /*KeepAlign=*/true);
  SmallVector<double, 8> Costs;
  for (unsigned S = 0; S < FetchWindow; S += NopBytes)
    Costs.push_back(cost(L, First, Last, S, /*KeepAlign=*/false));
  double Best = *llvm::min_element(Costs);
  unsigned BestShift = ~0u;
  if (Base > Best + CostSlack)
    for (auto [Idx, C] : enumerate(Costs))
      if (C <= Best + CostSlack) {
        BestShift = Idx * NopBytes;
        break;
      }

  if (PrintDecisions) {
    errs() << "fetch-window: " << MF.getName() << " loop "
           << printMBBReference(*L.getHeader()) << " first "
           << printMBBReference(*First) << " @" << Offset.lookup(First)
           << " bytes=" << Bytes << " base=" << Base << " cost=[";
    for (double C : Costs)
      errs() << ' ' << C;
    errs() << " ] shift="
           << (BestShift == ~0u ? std::string("keep")
                                : std::to_string(BestShift))
           << " pad-in " << printMBBReference(*PadMBB) << '\n';
  }
  if (BestShift == ~0u)
    return false;

  // The score only sees this loop. Padding also moves the code after it,
  // which can be hotter and is not scored again.
  if (const MachineBasicBlock *Hot = hotCodeMoved(L, *PadMBB)) {
    if (PrintDecisions)
      errs() << "fetch-window: " << MF.getName() << " loop "
             << printMBBReference(*L.getHeader()) << " moves hotter "
             << printMBBReference(*Hot) << ", skip\n";
    return false;
  }

  // With -max-bytes-for-alignment the shift can make a later user-aligned
  // block miss its alignment. Undo the shift if it does.
  SmallVector<const MachineBasicBlock *, 4> Capped;
  for (const MachineBasicBlock &MBB :
       make_range(std::next(PadMBB->getIterator()), MF.end()))
    if (UserAligned.contains(&MBB) && MBB.getMaxBytesForAlignment() &&
        isAligned(MBB.getAlignment(), Offset.lookup(&MBB)))
      Capped.push_back(&MBB);

  Align OldAlign = First->getAlignment();
  unsigned OldMaxBytes = First->getMaxBytesForAlignment();
  First->setAlignment(Align(1), 0);
  DebugLoc DL;
  auto InsertPt = PadMBB->getFirstTerminator();
  SmallVector<MachineInstr *, 8> Nops;
  for (unsigned N = 0; N < BestShift / NopBytes; ++N)
    Nops.push_back(
        BuildMI(*PadMBB, InsertPt, DL, TII->get(AMDGPU::S_NOP)).addImm(0));
  computeOffsets(MF);

  if (any_of(Capped, [&](const MachineBasicBlock *MBB) {
        return !isAligned(MBB->getAlignment(), Offset.lookup(MBB));
      })) {
    if (PrintDecisions)
      errs() << "fetch-window: " << MF.getName() << " loop "
             << printMBBReference(*L.getHeader())
             << " shift breaks a user alignment, undo\n";
    for (MachineInstr *MI : Nops)
      MI->eraseFromParent();
    First->setAlignment(OldAlign, OldMaxBytes);
    computeOffsets(MF);
    return false;
  }
  return true;
}

bool AMDGPUFetchWindowPlacement::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (EntryPadForTesting && MFI->isEntryFunction()) {
    MachineBasicBlock &Entry = MF.front();
    for (unsigned N = 0; N < EntryPadForTesting; ++N)
      BuildMI(Entry, Entry.begin(), DebugLoc(),
              ST.getInstrInfo()->get(AMDGPU::S_NOP))
          .addImm(0);
  }

  // -align-loops is a user request for every loop.
  if (!EnablePlacement || !ST.hasLoopHeadInstSplitSensitivity() ||
      MF.getTarget().Options.LoopAlignment || skipFunction(MF.getFunction()))
    return EntryPadForTesting != 0;

  // Offsets are only known modulo the function alignment, and the kernarg
  // preload prologue is inserted after this pass.
  if (!MFI->isEntryFunction() || MFI->getNumKernargPreloadedSGPRs())
    return false;

  TII = ST.getInstrInfo();
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB.instrs())
      if (MI.isInlineAsm())
        return false;

  MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  // MachineBlockPlacement takes the larger of llvm.loop.align and the fetch
  // window alignment, but the latter caps the padding at one s_nop. Lift the
  // cap so the user request always holds; any alignment works for 4 bytes.
  bool Changed = false;
  UserAligned.clear();
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.getAlignment() == Align(1))
      continue;
    const MachineLoop *L = MLI.getLoopFor(&MBB);
    if (L && getLoopAlignMetadata(*L) > NopBytes && !hasUserAlignment(MBB)) {
      MBB.setMaxBytesForAlignment(0);
      Changed = true;
    }
    if (hasUserAlignment(MBB))
      UserAligned.insert(&MBB);
  }

  SmallVector<MachineLoop *, 8> Inner;
  for (MachineLoop *L : MLI.getLoopsInPreorder())
    if (L->isInnermost())
      Inner.push_back(L);
  if (Inner.empty())
    return Changed;

  computeOffsets(MF);
  auto LayoutStart = [&](MachineLoop *L) {
    unsigned Min = ~0u;
    for (MachineBasicBlock *MBB : L->blocks())
      Min = std::min(Min, Offset.lookup(MBB));
    return Min;
  };
  llvm::sort(Inner, [&](MachineLoop *A, MachineLoop *B) {
    return LayoutStart(A) < LayoutStart(B);
  });

  InnerBlocks.clear();
  for (MachineLoop *L : Inner)
    InnerBlocks.insert_range(L->blocks());

  MachineBasicBlock *Fence = nullptr;
  for (MachineLoop *L : Inner)
    Changed |= placeLoop(*L, Fence);
  Offset.clear();
  return Changed;
}

char AMDGPUFetchWindowPlacement::ID = 0;
char &llvm::AMDGPUFetchWindowPlacementID = AMDGPUFetchWindowPlacement::ID;

INITIALIZE_PASS_BEGIN(AMDGPUFetchWindowPlacement, DEBUG_TYPE,
                      "AMDGPU Fetch Window Placement", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUFetchWindowPlacement, DEBUG_TYPE,
                    "AMDGPU Fetch Window Placement", false, false)
