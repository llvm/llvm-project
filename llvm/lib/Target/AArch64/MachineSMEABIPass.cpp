//===- MachineSMEABIPass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the SME ABI requirements for ZA state. This includes
// implementing the lazy (and agnostic) ZA state save schemes around calls.
//
//===----------------------------------------------------------------------===//
//
// This pass works by collecting instructions that require ZA to be in a
// specific state (e.g., "ACTIVE" or "SAVED") and inserting the necessary state
// transitions to ensure ZA is in the required state before instructions. State
// transitions represent actions such as setting up or restoring a lazy save.
// Certain points within a function may also have predefined states independent
// of any instructions, for example, a "shared_za" function is always entered
// and exited in the "ACTIVE" state.
//
// To handle ZA state across control flow, we make use of edge bundling. This
// assigns each block an "incoming" and "outgoing" edge bundle (representing
// incoming and outgoing edges). Initially, these are unique to each block;
// then, in the process of forming bundles, the outgoing bundle of a block is
// joined with the incoming bundle of all successors. The result is that each
// bundle can be assigned a single ZA state, which ensures the state required by
// all a blocks' successors is the same, and that each basic block will always
// be entered with the same ZA state. This eliminates the need for splitting
// edges to insert state transitions or "phi" nodes for ZA states.
//
// See below for a simple example of edge bundling.
//
// The following shows a conditionally executed basic block (BB1):
//
// if (cond)
//   BB1
// BB2
//
// Initial Bundles         Joined Bundles
//
//   ┌──0──┐                ┌──0──┐
//   │ BB0 │                │ BB0 │
//   └──1──┘                └──1──┘
//      ├───────┐              ├───────┐
//      ▼       │              ▼       │
//   ┌──2──┐    │   ─────►  ┌──1──┐    │
//   │ BB1 │    ▼           │ BB1 │    ▼
//   └──3──┘ ┌──4──┐        └──1──┘ ┌──1──┐
//      └───►4 BB2 │           └───►1 BB2 │
//           └──5──┘                └──2──┘
//
// On the left are the initial per-block bundles, and on the right are the
// joined bundles (which are the result of the EdgeBundles analysis).

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-machine-sme-abi"

namespace {

enum ZAState {
  // Any/unknown state (not valid)
  ANY = 0,

  // ZA is in use and active (i.e. within the accumulator)
  ACTIVE,

  // A ZA save has been set up or committed (i.e. ZA is dormant or off)
  LOCAL_SAVED,

  // ZA is off or a lazy save has been set up by the caller
  CALLER_DORMANT,

  // ZA is off
  OFF,

  // The number of ZA states (not a valid state)
  NUM_ZA_STATE
};

/// A bitmask enum to record live physical registers that the "emit*" routines
/// may need to preserve. Note: This only tracks registers we may clobber.
enum LiveRegs : uint8_t {
  None = 0,
  NZCV = 1 << 0,
  W0 = 1 << 1,
  W0_HI = 1 << 2,
  X0 = W0 | W0_HI,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ W0_HI)
};

/// Holds the virtual registers live physical registers have been saved to.
struct PhysRegSave {
  LiveRegs PhysLiveRegs;
  Register StatusFlags = AArch64::NoRegister;
  Register X0Save = AArch64::NoRegister;
};

/// Contains the needed ZA state (and live registers) at an instruction. That is
/// the state ZA must be in _before_ "InsertPt".
struct InstInfo {
  ZAState NeededState{ZAState::ANY};
  MachineBasicBlock::iterator InsertPt;
  LiveRegs PhysLiveRegs = LiveRegs::None;
};

/// Contains the needed ZA state for each instruction in a block. Instructions
/// that do not require a ZA state are not recorded.
struct BlockInfo {
  ZAState FixedEntryState{ZAState::ANY};
  SmallVector<InstInfo> Insts;
  LiveRegs PhysLiveRegsAtEntry = LiveRegs::None;
  LiveRegs PhysLiveRegsAtExit = LiveRegs::None;
};

/// Contains the needed ZA state information for all blocks within a function.
struct FunctionInfo {
  SmallVector<BlockInfo> Blocks;
  std::optional<MachineBasicBlock::iterator> AfterSMEProloguePt;
  LiveRegs PhysLiveRegsAfterSMEPrologue = LiveRegs::None;
};

/// State/helpers that is only needed when emitting code to handle
/// saving/restoring ZA.
class EmitContext {
public:
  EmitContext() = default;

  /// Get or create a TPIDR2 block in \p MF.
  int getTPIDR2Block(MachineFunction &MF) {
    if (TPIDR2BlockFI)
      return *TPIDR2BlockFI;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    TPIDR2BlockFI = MFI.CreateStackObject(16, Align(16), false);
    return *TPIDR2BlockFI;
  }

  /// Get or create agnostic ZA buffer pointer in \p MF.
  Register getAgnosticZABufferPtr(MachineFunction &MF) {
    if (AgnosticZABufferPtr != AArch64::NoRegister)
      return AgnosticZABufferPtr;
    Register BufferPtr =
        MF.getInfo<AArch64FunctionInfo>()->getEarlyAllocSMESaveBuffer();
    AgnosticZABufferPtr =
        BufferPtr != AArch64::NoRegister
            ? BufferPtr
            : MF.getRegInfo().createVirtualRegister(&AArch64::GPR64RegClass);
    return AgnosticZABufferPtr;
  }

  /// Returns true if the function must allocate a ZA save buffer on entry. This
  /// will be the case if, at any point in the function, a ZA save was emitted.
  bool needsSaveBuffer() const {
    assert(!(TPIDR2BlockFI && AgnosticZABufferPtr) &&
           "Cannot have both a TPIDR2 block and agnostic ZA buffer");
    return TPIDR2BlockFI || AgnosticZABufferPtr != AArch64::NoRegister;
  }

private:
  std::optional<int> TPIDR2BlockFI;
  Register AgnosticZABufferPtr = AArch64::NoRegister;
};

static bool isLegalEdgeBundleZAState(ZAState State) {
  switch (State) {
  case ZAState::ACTIVE:
  case ZAState::LOCAL_SAVED:
    return true;
  default:
    return false;
  }
}

StringRef getZAStateString(ZAState State) {
#define MAKE_CASE(V)                                                           \
  case V:                                                                      \
    return #V;
  switch (State) {
    MAKE_CASE(ZAState::ANY)
    MAKE_CASE(ZAState::ACTIVE)
    MAKE_CASE(ZAState::LOCAL_SAVED)
    MAKE_CASE(ZAState::CALLER_DORMANT)
    MAKE_CASE(ZAState::OFF)
  default:
    llvm_unreachable("Unexpected ZAState");
  }
#undef MAKE_CASE
}

static bool isZAorZTRegOp(const TargetRegisterInfo &TRI,
                          const MachineOperand &MO) {
  if (!MO.isReg() || !MO.getReg().isPhysical())
    return false;
  return any_of(TRI.subregs_inclusive(MO.getReg()), [](const MCPhysReg &SR) {
    return AArch64::MPR128RegClass.contains(SR) ||
           AArch64::ZTRRegClass.contains(SR);
  });
}

/// Returns the required ZA state needed before \p MI and an iterator pointing
/// to where any code required to change the ZA state should be inserted.
static std::pair<ZAState, MachineBasicBlock::iterator>
getZAStateBeforeInst(const TargetRegisterInfo &TRI, MachineInstr &MI,
                     bool ZAOffAtReturn) {
  MachineBasicBlock::iterator InsertPt(MI);

  if (MI.getOpcode() == AArch64::InOutZAUsePseudo)
    return {ZAState::ACTIVE, std::prev(InsertPt)};

  if (MI.getOpcode() == AArch64::RequiresZASavePseudo)
    return {ZAState::LOCAL_SAVED, std::prev(InsertPt)};

  if (MI.isReturn())
    return {ZAOffAtReturn ? ZAState::OFF : ZAState::ACTIVE, InsertPt};

  for (auto &MO : MI.operands()) {
    if (isZAorZTRegOp(TRI, MO))
      return {ZAState::ACTIVE, InsertPt};
  }

  return {ZAState::ANY, InsertPt};
}

struct MachineSMEABI : public MachineFunctionPass {
  inline static char ID = 0;

  MachineSMEABI() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Machine SME ABI pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<EdgeBundlesWrapperLegacy>();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  /// Collects the needed ZA state (and live registers) before each instruction
  /// within the machine function.
  FunctionInfo collectNeededZAStates(SMEAttrs SMEFnAttrs);

  /// Assigns each edge bundle a ZA state based on the needed states of blocks
  /// that have incoming or outgoing edges in that bundle.
  SmallVector<ZAState> assignBundleZAStates(const EdgeBundles &Bundles,
                                            const FunctionInfo &FnInfo);

  /// Inserts code to handle changes between ZA states within the function.
  /// E.g., ACTIVE -> LOCAL_SAVED will insert code required to save ZA.
  void insertStateChanges(EmitContext &, const FunctionInfo &FnInfo,
                          const EdgeBundles &Bundles,
                          ArrayRef<ZAState> BundleStates);

  // Emission routines for private and shared ZA functions (using lazy saves).
  void emitNewZAPrologue(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  void emitRestoreLazySave(EmitContext &, MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           LiveRegs PhysLiveRegs);
  void emitSetupLazySave(EmitContext &, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  void emitAllocateLazySaveBuffer(EmitContext &, MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI);
  void emitZAOff(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                 bool ClearTPIDR2);

  // Emission routines for agnostic ZA functions.
  void emitSetupFullZASave(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           LiveRegs PhysLiveRegs);
  // Emit a "full" ZA save or restore. It is "full" in the sense that this
  // function will emit a call to __arm_sme_save or __arm_sme_restore, which
  // handles saving and restoring both ZA and ZT0.
  void emitFullZASaveRestore(EmitContext &, MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI,
                             LiveRegs PhysLiveRegs, bool IsSave);
  void emitAllocateFullZASaveBuffer(EmitContext &, MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    LiveRegs PhysLiveRegs);

  /// Attempts to find an insertion point before \p Inst where the status flags
  /// are not live. If \p Inst is `Block.Insts.end()` a point before the end of
  /// the block is found.
  std::pair<MachineBasicBlock::iterator, LiveRegs>
  findStateChangeInsertionPoint(MachineBasicBlock &MBB, const BlockInfo &Block,
                                SmallVectorImpl<InstInfo>::const_iterator Inst);
  void emitStateChange(EmitContext &, MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator MBBI, ZAState From,
                       ZAState To, LiveRegs PhysLiveRegs);

  // Helpers for switching between lazy/full ZA save/restore routines.
  void emitZASave(EmitContext &Context, MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MBBI, LiveRegs PhysLiveRegs) {
    if (AFI->getSMEFnAttrs().hasAgnosticZAInterface())
      return emitFullZASaveRestore(Context, MBB, MBBI, PhysLiveRegs,
                                   /*IsSave=*/true);
    return emitSetupLazySave(Context, MBB, MBBI);
  }
  void emitZARestore(EmitContext &Context, MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator MBBI, LiveRegs PhysLiveRegs) {
    if (AFI->getSMEFnAttrs().hasAgnosticZAInterface())
      return emitFullZASaveRestore(Context, MBB, MBBI, PhysLiveRegs,
                                   /*IsSave=*/false);
    return emitRestoreLazySave(Context, MBB, MBBI, PhysLiveRegs);
  }
  void emitAllocateZASaveBuffer(EmitContext &Context, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                LiveRegs PhysLiveRegs) {
    if (AFI->getSMEFnAttrs().hasAgnosticZAInterface())
      return emitAllocateFullZASaveBuffer(Context, MBB, MBBI, PhysLiveRegs);
    return emitAllocateLazySaveBuffer(Context, MBB, MBBI);
  }

  /// Save live physical registers to virtual registers.
  PhysRegSave createPhysRegSave(LiveRegs PhysLiveRegs, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI, DebugLoc DL);
  /// Restore physical registers from a save of their previous values.
  void restorePhyRegSave(const PhysRegSave &RegSave, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, DebugLoc DL);

private:
  MachineFunction *MF = nullptr;
  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64RegisterInfo *TRI = nullptr;
  const AArch64FunctionInfo *AFI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
};

static LiveRegs getPhysLiveRegs(LiveRegUnits const &LiveUnits) {
  LiveRegs PhysLiveRegs = LiveRegs::None;
  if (!LiveUnits.available(AArch64::NZCV))
    PhysLiveRegs |= LiveRegs::NZCV;
  // We have to track W0 and X0 separately as otherwise things can get
  // confused if we attempt to preserve X0 but only W0 was defined.
  if (!LiveUnits.available(AArch64::W0))
    PhysLiveRegs |= LiveRegs::W0;
  if (!LiveUnits.available(AArch64::W0_HI))
    PhysLiveRegs |= LiveRegs::W0_HI;
  return PhysLiveRegs;
}

static void setPhysLiveRegs(LiveRegUnits &LiveUnits, LiveRegs PhysLiveRegs) {
  if (PhysLiveRegs & LiveRegs::NZCV)
    LiveUnits.addReg(AArch64::NZCV);
  if (PhysLiveRegs & LiveRegs::W0)
    LiveUnits.addReg(AArch64::W0);
  if (PhysLiveRegs & LiveRegs::W0_HI)
    LiveUnits.addReg(AArch64::W0_HI);
}

FunctionInfo MachineSMEABI::collectNeededZAStates(SMEAttrs SMEFnAttrs) {
  assert((SMEFnAttrs.hasAgnosticZAInterface() || SMEFnAttrs.hasZT0State() ||
          SMEFnAttrs.hasZAState()) &&
         "Expected function to have ZA/ZT0 state!");

  SmallVector<BlockInfo> Blocks(MF->getNumBlockIDs());
  LiveRegs PhysLiveRegsAfterSMEPrologue = LiveRegs::None;
  std::optional<MachineBasicBlock::iterator> AfterSMEProloguePt;

  for (MachineBasicBlock &MBB : *MF) {
    BlockInfo &Block = Blocks[MBB.getNumber()];

    if (MBB.isEntryBlock()) {
      // Entry block:
      Block.FixedEntryState = SMEFnAttrs.hasPrivateZAInterface()
                                  ? ZAState::CALLER_DORMANT
                                  : ZAState::ACTIVE;
    } else if (MBB.isEHPad()) {
      // EH entry block:
      Block.FixedEntryState = ZAState::LOCAL_SAVED;
    }

    LiveRegUnits LiveUnits(*TRI);
    LiveUnits.addLiveOuts(MBB);

    Block.PhysLiveRegsAtExit = getPhysLiveRegs(LiveUnits);
    auto FirstTerminatorInsertPt = MBB.getFirstTerminator();
    auto FirstNonPhiInsertPt = MBB.getFirstNonPHI();
    for (MachineInstr &MI : reverse(MBB)) {
      MachineBasicBlock::iterator MBBI(MI);
      LiveUnits.stepBackward(MI);
      LiveRegs PhysLiveRegs = getPhysLiveRegs(LiveUnits);
      // The SMEStateAllocPseudo marker is added to a function if the save
      // buffer was allocated in SelectionDAG. It marks the end of the
      // allocation -- which is a safe point for this pass to insert any TPIDR2
      // block setup.
      if (MI.getOpcode() == AArch64::SMEStateAllocPseudo) {
        AfterSMEProloguePt = MBBI;
        PhysLiveRegsAfterSMEPrologue = PhysLiveRegs;
      }
      // Note: We treat Agnostic ZA as inout_za with an alternate save/restore.
      auto [NeededState, InsertPt] = getZAStateBeforeInst(
          *TRI, MI, /*ZAOffAtReturn=*/SMEFnAttrs.hasPrivateZAInterface());
      assert((InsertPt == MBBI ||
              InsertPt->getOpcode() == AArch64::ADJCALLSTACKDOWN) &&
             "Unexpected state change insertion point!");
      // TODO: Do something to avoid state changes where NZCV is live.
      if (MBBI == FirstTerminatorInsertPt)
        Block.PhysLiveRegsAtExit = PhysLiveRegs;
      if (MBBI == FirstNonPhiInsertPt)
        Block.PhysLiveRegsAtEntry = PhysLiveRegs;
      if (NeededState != ZAState::ANY)
        Block.Insts.push_back({NeededState, InsertPt, PhysLiveRegs});
    }

    // Reverse vector (as we had to iterate backwards for liveness).
    std::reverse(Block.Insts.begin(), Block.Insts.end());
  }

  return FunctionInfo{std::move(Blocks), AfterSMEProloguePt,
                      PhysLiveRegsAfterSMEPrologue};
}

/// Assigns each edge bundle a ZA state based on the needed states of blocks
/// that have incoming or outgoing edges in that bundle.
SmallVector<ZAState>
MachineSMEABI::assignBundleZAStates(const EdgeBundles &Bundles,
                                    const FunctionInfo &FnInfo) {
  SmallVector<ZAState> BundleStates(Bundles.getNumBundles());
  for (unsigned I = 0, E = Bundles.getNumBundles(); I != E; ++I) {
    LLVM_DEBUG(dbgs() << "Assigning ZA state for edge bundle: " << I << '\n');

    // Attempt to assign a ZA state for this bundle that minimizes state
    // transitions. Edges within loops are given a higher weight as we assume
    // they will be executed more than once.
    // TODO: We should propagate desired incoming/outgoing states through blocks
    // that have the "ANY" state first to make better global decisions.
    int EdgeStateCounts[ZAState::NUM_ZA_STATE] = {0};
    for (unsigned BlockID : Bundles.getBlocks(I)) {
      LLVM_DEBUG(dbgs() << "- bb." << BlockID);

      const BlockInfo &Block = FnInfo.Blocks[BlockID];
      if (Block.Insts.empty()) {
        LLVM_DEBUG(dbgs() << " (no state preference)\n");
        continue;
      }
      bool InEdge = Bundles.getBundle(BlockID, /*Out=*/false) == I;
      bool OutEdge = Bundles.getBundle(BlockID, /*Out=*/true) == I;

      ZAState DesiredIncomingState = Block.Insts.front().NeededState;
      if (InEdge && isLegalEdgeBundleZAState(DesiredIncomingState)) {
        EdgeStateCounts[DesiredIncomingState]++;
        LLVM_DEBUG(dbgs() << " DesiredIncomingState: "
                          << getZAStateString(DesiredIncomingState));
      }
      ZAState DesiredOutgoingState = Block.Insts.back().NeededState;
      if (OutEdge && isLegalEdgeBundleZAState(DesiredOutgoingState)) {
        EdgeStateCounts[DesiredOutgoingState]++;
        LLVM_DEBUG(dbgs() << " DesiredOutgoingState: "
                          << getZAStateString(DesiredOutgoingState));
      }
      LLVM_DEBUG(dbgs() << '\n');
    }

    ZAState BundleState =
        ZAState(max_element(EdgeStateCounts) - EdgeStateCounts);

    // Force ZA to be active in bundles that don't have a preferred state.
    // TODO: Something better here (to avoid extra mode switches).
    if (BundleState == ZAState::ANY)
      BundleState = ZAState::ACTIVE;

    LLVM_DEBUG({
      dbgs() << "Chosen ZA state: " << getZAStateString(BundleState) << '\n'
             << "Edge counts:";
      for (auto [State, Count] : enumerate(EdgeStateCounts))
        dbgs() << " " << getZAStateString(ZAState(State)) << ": " << Count;
      dbgs() << "\n\n";
    });

    BundleStates[I] = BundleState;
  }

  return BundleStates;
}

std::pair<MachineBasicBlock::iterator, LiveRegs>
MachineSMEABI::findStateChangeInsertionPoint(
    MachineBasicBlock &MBB, const BlockInfo &Block,
    SmallVectorImpl<InstInfo>::const_iterator Inst) {
  LiveRegs PhysLiveRegs;
  MachineBasicBlock::iterator InsertPt;
  if (Inst != Block.Insts.end()) {
    InsertPt = Inst->InsertPt;
    PhysLiveRegs = Inst->PhysLiveRegs;
  } else {
    InsertPt = MBB.getFirstTerminator();
    PhysLiveRegs = Block.PhysLiveRegsAtExit;
  }

  if (!(PhysLiveRegs & LiveRegs::NZCV))
    return {InsertPt, PhysLiveRegs}; // Nothing to do (no live flags).

  // Find the previous state change. We can not move before this point.
  MachineBasicBlock::iterator PrevStateChangeI;
  if (Inst == Block.Insts.begin()) {
    PrevStateChangeI = MBB.begin();
  } else {
    // Note: `std::prev(Inst)` is the previous InstInfo. We only create an
    // InstInfo object for instructions that require a specific ZA state, so the
    // InstInfo is the site of the previous state change in the block (which can
    // be several MIs earlier).
    PrevStateChangeI = std::prev(Inst)->InsertPt;
  }

  // Note: LiveUnits will only accurately track X0 and NZCV.
  LiveRegUnits LiveUnits(*TRI);
  setPhysLiveRegs(LiveUnits, PhysLiveRegs);
  for (MachineBasicBlock::iterator I = InsertPt; I != PrevStateChangeI; --I) {
    // Don't move before/into a call (which may have a state change before it).
    if (I->getOpcode() == TII->getCallFrameDestroyOpcode() || I->isCall())
      break;
    LiveUnits.stepBackward(*I);
    if (LiveUnits.available(AArch64::NZCV))
      return {I, getPhysLiveRegs(LiveUnits)};
  }
  return {InsertPt, PhysLiveRegs};
}

void MachineSMEABI::insertStateChanges(EmitContext &Context,
                                       const FunctionInfo &FnInfo,
                                       const EdgeBundles &Bundles,
                                       ArrayRef<ZAState> BundleStates) {
  for (MachineBasicBlock &MBB : *MF) {
    const BlockInfo &Block = FnInfo.Blocks[MBB.getNumber()];
    ZAState InState = BundleStates[Bundles.getBundle(MBB.getNumber(),
                                                     /*Out=*/false)];

    ZAState CurrentState = Block.FixedEntryState;
    if (CurrentState == ZAState::ANY)
      CurrentState = InState;

    for (auto &Inst : Block.Insts) {
      if (CurrentState != Inst.NeededState) {
        auto [InsertPt, PhysLiveRegs] =
            findStateChangeInsertionPoint(MBB, Block, &Inst);
        emitStateChange(Context, MBB, InsertPt, CurrentState, Inst.NeededState,
                        PhysLiveRegs);
        CurrentState = Inst.NeededState;
      }
    }

    if (MBB.succ_empty())
      continue;

    ZAState OutState =
        BundleStates[Bundles.getBundle(MBB.getNumber(), /*Out=*/true)];
    if (CurrentState != OutState) {
      auto [InsertPt, PhysLiveRegs] =
          findStateChangeInsertionPoint(MBB, Block, Block.Insts.end());
      emitStateChange(Context, MBB, InsertPt, CurrentState, OutState,
                      PhysLiveRegs);
    }
  }
}

static DebugLoc getDebugLoc(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI) {
  if (MBBI != MBB.end())
    return MBBI->getDebugLoc();
  return DebugLoc();
}

void MachineSMEABI::emitSetupLazySave(EmitContext &Context,
                                      MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Get pointer to TPIDR2 block.
  Register TPIDR2 = MRI->createVirtualRegister(&AArch64::GPR64spRegClass);
  Register TPIDR2Ptr = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXri), TPIDR2)
      .addFrameIndex(Context.getTPIDR2Block(*MF))
      .addImm(0)
      .addImm(0);
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), TPIDR2Ptr)
      .addReg(TPIDR2);
  // Set TPIDR2_EL0 to point to TPIDR2 block.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(TPIDR2Ptr);
}

PhysRegSave MachineSMEABI::createPhysRegSave(LiveRegs PhysLiveRegs,
                                             MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MBBI,
                                             DebugLoc DL) {
  PhysRegSave RegSave{PhysLiveRegs};
  if (PhysLiveRegs & LiveRegs::NZCV) {
    RegSave.StatusFlags = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MRS), RegSave.StatusFlags)
        .addImm(AArch64SysReg::NZCV)
        .addReg(AArch64::NZCV, RegState::Implicit);
  }
  // Note: Preserving X0 is "free" as this is before register allocation, so
  // the register allocator is still able to optimize these copies.
  if (PhysLiveRegs & LiveRegs::W0) {
    RegSave.X0Save = MRI->createVirtualRegister(PhysLiveRegs & LiveRegs::W0_HI
                                                    ? &AArch64::GPR64RegClass
                                                    : &AArch64::GPR32RegClass);
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), RegSave.X0Save)
        .addReg(PhysLiveRegs & LiveRegs::W0_HI ? AArch64::X0 : AArch64::W0);
  }
  return RegSave;
}

void MachineSMEABI::restorePhyRegSave(const PhysRegSave &RegSave,
                                      MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      DebugLoc DL) {
  if (RegSave.StatusFlags != AArch64::NoRegister)
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSR))
        .addImm(AArch64SysReg::NZCV)
        .addReg(RegSave.StatusFlags)
        .addReg(AArch64::NZCV, RegState::ImplicitDefine);

  if (RegSave.X0Save != AArch64::NoRegister)
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY),
            RegSave.PhysLiveRegs & LiveRegs::W0_HI ? AArch64::X0 : AArch64::W0)
        .addReg(RegSave.X0Save);
}

void MachineSMEABI::emitRestoreLazySave(EmitContext &Context,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        LiveRegs PhysLiveRegs) {
  auto *TLI = Subtarget->getTargetLowering();
  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register TPIDR2EL0 = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  Register TPIDR2 = AArch64::X0;

  // TODO: Emit these within the restore MBB to prevent unnecessary saves.
  PhysRegSave RegSave = createPhysRegSave(PhysLiveRegs, MBB, MBBI, DL);

  // Enable ZA.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(1);
  // Get current TPIDR2_EL0.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MRS), TPIDR2EL0)
      .addImm(AArch64SysReg::TPIDR2_EL0);
  // Get pointer to TPIDR2 block.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXri), TPIDR2)
      .addFrameIndex(Context.getTPIDR2Block(*MF))
      .addImm(0)
      .addImm(0);
  // (Conditionally) restore ZA state.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::RestoreZAPseudo))
      .addReg(TPIDR2EL0)
      .addReg(TPIDR2)
      .addExternalSymbol(TLI->getLibcallName(RTLIB::SMEABI_TPIDR2_RESTORE))
      .addRegMask(TRI->SMEABISupportRoutinesCallPreservedMaskFromX0());
  // Zero TPIDR2_EL0.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(AArch64::XZR);

  restorePhyRegSave(RegSave, MBB, MBBI, DL);
}

void MachineSMEABI::emitZAOff(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              bool ClearTPIDR2) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  if (ClearTPIDR2)
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSR))
        .addImm(AArch64SysReg::TPIDR2_EL0)
        .addReg(AArch64::XZR);

  // Disable ZA.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(0);
}

void MachineSMEABI::emitAllocateLazySaveBuffer(
    EmitContext &Context, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBI) {
  MachineFrameInfo &MFI = MF->getFrameInfo();
  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register SP = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  Register SVL = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  Register Buffer = AFI->getEarlyAllocSMESaveBuffer();

  // Calculate SVL.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::RDSVLI_XI), SVL).addImm(1);

  // 1. Allocate the lazy save buffer.
  if (Buffer == AArch64::NoRegister) {
    // TODO: On Windows, we allocate the lazy save buffer in SelectionDAG (so
    // Buffer != AArch64::NoRegister). This is done to reuse the existing
    // expansions (which can insert stack checks). This works, but it means we
    // will always allocate the lazy save buffer (even if the function contains
    // no lazy saves). If we want to handle Windows here, we'll need to
    // implement something similar to LowerWindowsDYNAMIC_STACKALLOC.
    assert(!Subtarget->isTargetWindows() &&
           "Lazy ZA save is not yet supported on Windows");
    Buffer = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
    // Get original stack pointer.
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), SP)
        .addReg(AArch64::SP);
    // Allocate a lazy-save buffer object of the size given, normally SVL * SVL
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSUBXrrr), Buffer)
        .addReg(SVL)
        .addReg(SVL)
        .addReg(SP);
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), AArch64::SP)
        .addReg(Buffer);
    // We have just allocated a variable sized object, tell this to PEI.
    MFI.CreateVariableSizedObject(Align(16), nullptr);
  }

  // 2. Setup the TPIDR2 block.
  {
    // Note: This case just needs to do `SVL << 48`. It is not implemented as we
    // generally don't support big-endian SVE/SME.
    if (!Subtarget->isLittleEndian())
      reportFatalInternalError(
          "TPIDR2 block initialization is not supported on big-endian targets");

    // Store buffer pointer and num_za_save_slices.
    // Bytes 10-15 are implicitly zeroed.
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::STPXi))
        .addReg(Buffer)
        .addReg(SVL)
        .addFrameIndex(Context.getTPIDR2Block(*MF))
        .addImm(0);
  }
}

void MachineSMEABI::emitNewZAPrologue(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  auto *TLI = Subtarget->getTargetLowering();
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Get current TPIDR2_EL0.
  Register TPIDR2EL0 = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MRS))
      .addReg(TPIDR2EL0, RegState::Define)
      .addImm(AArch64SysReg::TPIDR2_EL0);
  // If TPIDR2_EL0 is non-zero, commit the lazy save.
  // NOTE: Functions that only use ZT0 don't need to zero ZA.
  bool ZeroZA = AFI->getSMEFnAttrs().hasZAState();
  auto CommitZASave =
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::CommitZASavePseudo))
          .addReg(TPIDR2EL0)
          .addImm(ZeroZA ? 1 : 0)
          .addExternalSymbol(TLI->getLibcallName(RTLIB::SMEABI_TPIDR2_SAVE))
          .addRegMask(TRI->SMEABISupportRoutinesCallPreservedMaskFromX0());
  if (ZeroZA)
    CommitZASave.addDef(AArch64::ZAB0, RegState::ImplicitDefine);
  // Enable ZA (as ZA could have previously been in the OFF state).
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(1);
}

void MachineSMEABI::emitFullZASaveRestore(EmitContext &Context,
                                          MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          LiveRegs PhysLiveRegs, bool IsSave) {
  auto *TLI = Subtarget->getTargetLowering();
  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register BufferPtr = AArch64::X0;

  PhysRegSave RegSave = createPhysRegSave(PhysLiveRegs, MBB, MBBI, DL);

  // Copy the buffer pointer into X0.
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), BufferPtr)
      .addReg(Context.getAgnosticZABufferPtr(*MF));

  // Call __arm_sme_save/__arm_sme_restore.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::BL))
      .addReg(BufferPtr, RegState::Implicit)
      .addExternalSymbol(TLI->getLibcallName(
          IsSave ? RTLIB::SMEABI_SME_SAVE : RTLIB::SMEABI_SME_RESTORE))
      .addRegMask(TRI->getCallPreservedMask(
          *MF,
          CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1));

  restorePhyRegSave(RegSave, MBB, MBBI, DL);
}

void MachineSMEABI::emitAllocateFullZASaveBuffer(
    EmitContext &Context, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBI, LiveRegs PhysLiveRegs) {
  // Buffer already allocated in SelectionDAG.
  if (AFI->getEarlyAllocSMESaveBuffer())
    return;

  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register BufferPtr = Context.getAgnosticZABufferPtr(*MF);
  Register BufferSize = MRI->createVirtualRegister(&AArch64::GPR64RegClass);

  PhysRegSave RegSave = createPhysRegSave(PhysLiveRegs, MBB, MBBI, DL);

  // Calculate the SME state size.
  {
    auto *TLI = Subtarget->getTargetLowering();
    const AArch64RegisterInfo *TRI = Subtarget->getRegisterInfo();
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::BL))
        .addExternalSymbol(TLI->getLibcallName(RTLIB::SMEABI_SME_STATE_SIZE))
        .addReg(AArch64::X0, RegState::ImplicitDefine)
        .addRegMask(TRI->getCallPreservedMask(
            *MF, CallingConv::
                     AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), BufferSize)
        .addReg(AArch64::X0);
  }

  // Allocate a buffer object of the size given __arm_sme_state_size.
  {
    MachineFrameInfo &MFI = MF->getFrameInfo();
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SUBXrx64), AArch64::SP)
        .addReg(AArch64::SP)
        .addReg(BufferSize)
        .addImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTX, 0));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), BufferPtr)
        .addReg(AArch64::SP);

    // We have just allocated a variable sized object, tell this to PEI.
    MFI.CreateVariableSizedObject(Align(16), nullptr);
  }

  restorePhyRegSave(RegSave, MBB, MBBI, DL);
}

void MachineSMEABI::emitStateChange(EmitContext &Context,
                                    MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertPt,
                                    ZAState From, ZAState To,
                                    LiveRegs PhysLiveRegs) {
  // ZA not used.
  if (From == ZAState::ANY || To == ZAState::ANY)
    return;

  // If we're exiting from the CALLER_DORMANT state that means this new ZA
  // function did not touch ZA (so ZA was never turned on).
  if (From == ZAState::CALLER_DORMANT && To == ZAState::OFF)
    return;

  // TODO: Avoid setting up the save buffer if there's no transition to
  // LOCAL_SAVED.
  if (From == ZAState::CALLER_DORMANT) {
    assert(AFI->getSMEFnAttrs().hasPrivateZAInterface() &&
           "CALLER_DORMANT state requires private ZA interface");
    assert(&MBB == &MBB.getParent()->front() &&
           "CALLER_DORMANT state only valid in entry block");
    emitNewZAPrologue(MBB, MBB.getFirstNonPHI());
    if (To == ZAState::ACTIVE)
      return; // Nothing more to do (ZA is active after the prologue).

    // Note: "emitNewZAPrologue" zeros ZA, so we may need to setup a lazy save
    // if "To" is "ZAState::LOCAL_SAVED". It may be possible to improve this
    // case by changing the placement of the zero instruction.
    From = ZAState::ACTIVE;
  }

  if (From == ZAState::ACTIVE && To == ZAState::LOCAL_SAVED)
    emitZASave(Context, MBB, InsertPt, PhysLiveRegs);
  else if (From == ZAState::LOCAL_SAVED && To == ZAState::ACTIVE)
    emitZARestore(Context, MBB, InsertPt, PhysLiveRegs);
  else if (To == ZAState::OFF) {
    assert(From != ZAState::CALLER_DORMANT &&
           "CALLER_DORMANT to OFF should have already been handled");
    assert(!AFI->getSMEFnAttrs().hasAgnosticZAInterface() &&
           "Should not turn ZA off in agnostic ZA function");
    emitZAOff(MBB, InsertPt, /*ClearTPIDR2=*/From == ZAState::LOCAL_SAVED);
  } else {
    dbgs() << "Error: Transition from " << getZAStateString(From) << " to "
           << getZAStateString(To) << '\n';
    llvm_unreachable("Unimplemented state transition");
  }
}

} // end anonymous namespace

INITIALIZE_PASS(MachineSMEABI, "aarch64-machine-sme-abi", "Machine SME ABI",
                false, false)

bool MachineSMEABI::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getSubtarget<AArch64Subtarget>().hasSME())
    return false;

  AFI = MF.getInfo<AArch64FunctionInfo>();
  SMEAttrs SMEFnAttrs = AFI->getSMEFnAttrs();
  if (!SMEFnAttrs.hasZAState() && !SMEFnAttrs.hasZT0State() &&
      !SMEFnAttrs.hasAgnosticZAInterface())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  this->MF = &MF;
  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();
  MRI = &MF.getRegInfo();

  const EdgeBundles &Bundles =
      getAnalysis<EdgeBundlesWrapperLegacy>().getEdgeBundles();

  FunctionInfo FnInfo = collectNeededZAStates(SMEFnAttrs);
  SmallVector<ZAState> BundleStates = assignBundleZAStates(Bundles, FnInfo);

  EmitContext Context;
  insertStateChanges(Context, FnInfo, Bundles, BundleStates);

  if (Context.needsSaveBuffer()) {
    if (FnInfo.AfterSMEProloguePt) {
      // Note: With inline stack probes the AfterSMEProloguePt may not be in the
      // entry block (due to the probing loop).
      MachineBasicBlock::iterator MBBI = *FnInfo.AfterSMEProloguePt;
      emitAllocateZASaveBuffer(Context, *MBBI->getParent(), MBBI,
                               FnInfo.PhysLiveRegsAfterSMEPrologue);
    } else {
      MachineBasicBlock &EntryBlock = MF.front();
      emitAllocateZASaveBuffer(
          Context, EntryBlock, EntryBlock.getFirstNonPHI(),
          FnInfo.Blocks[EntryBlock.getNumber()].PhysLiveRegsAtEntry);
    }
  }

  return true;
}

FunctionPass *llvm::createMachineSMEABIPass() { return new MachineSMEABI(); }
