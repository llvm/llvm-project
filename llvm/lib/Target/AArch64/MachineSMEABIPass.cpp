//===- MachineSMEABIPass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the SME ABI requirements for ZA state. This includes
// implementing the lazy ZA state save schemes around calls.
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
// then, in the process of forming bundles, the outgoing block of a block is
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

static bool isLegalEdgeBundleZAState(ZAState State) {
  switch (State) {
  case ZAState::ACTIVE:
  case ZAState::LOCAL_SAVED:
    return true;
  default:
    return false;
  }
}
struct TPIDR2State {
  int FrameIndex = -1;
};

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

static bool isZAorZT0RegOp(const TargetRegisterInfo &TRI,
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
    if (isZAorZT0RegOp(TRI, MO))
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
  void collectNeededZAStates(SMEAttrs);

  /// Assigns each edge bundle a ZA state based on the needed states of blocks
  /// that have incoming or outgoing edges in that bundle.
  void assignBundleZAStates();

  /// Inserts code to handle changes between ZA states within the function.
  /// E.g., ACTIVE -> LOCAL_SAVED will insert code required to save ZA.
  void insertStateChanges();

  // Emission routines for private and shared ZA functions (using lazy saves).
  void emitNewZAPrologue(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  void emitRestoreLazySave(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           LiveRegs PhysLiveRegs);
  void emitSetupLazySave(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  void emitAllocateLazySaveBuffer(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI);
  void emitZAOff(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                 bool ClearTPIDR2);

  void emitStateChange(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       ZAState From, ZAState To, LiveRegs PhysLiveRegs);

  /// Save live physical registers to virtual registers.
  PhysRegSave createPhysRegSave(LiveRegs PhysLiveRegs, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI, DebugLoc DL);
  /// Restore physical registers from a save of their previous values.
  void restorePhyRegSave(PhysRegSave const &RegSave, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, DebugLoc DL);

  /// Get or create a TPIDR2 block in this function.
  TPIDR2State getTPIDR2Block();

private:
  /// Contains the needed ZA state (and live registers) at an instruction.
  struct InstInfo {
    ZAState NeededState{ZAState::ANY};
    MachineBasicBlock::iterator InsertPt;
    LiveRegs PhysLiveRegs = LiveRegs::None;
  };

  /// Contains the needed ZA state for each instruction in a block.
  /// Instructions that do not require a ZA state are not recorded.
  struct BlockInfo {
    ZAState FixedEntryState{ZAState::ANY};
    SmallVector<InstInfo> Insts;
    LiveRegs PhysLiveRegsAtExit = LiveRegs::None;
  };

  // All pass state that must be cleared between functions.
  struct PassState {
    SmallVector<BlockInfo> Blocks;
    SmallVector<ZAState> BundleStates;
    std::optional<TPIDR2State> TPIDR2Block;
  } State;

  MachineFunction *MF = nullptr;
  EdgeBundles *EdgeBundles = nullptr;
  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64RegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
};

void MachineSMEABI::collectNeededZAStates(SMEAttrs SMEFnAttrs) {
  assert((SMEFnAttrs.hasZT0State() || SMEFnAttrs.hasZAState()) &&
         "Expected function to have ZA/ZT0 state!");

  State.Blocks.resize(MF->getNumBlockIDs());
  for (MachineBasicBlock &MBB : *MF) {
    BlockInfo &Block = State.Blocks[MBB.getNumber()];
    if (&MBB == &MF->front()) {
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

    auto GetPhysLiveRegs = [&] {
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
    };

    Block.PhysLiveRegsAtExit = GetPhysLiveRegs();
    auto FirstTerminatorInsertPt = MBB.getFirstTerminator();
    for (MachineInstr &MI : reverse(MBB)) {
      MachineBasicBlock::iterator MBBI(MI);
      LiveUnits.stepBackward(MI);
      LiveRegs PhysLiveRegs = GetPhysLiveRegs();
      auto [NeededState, InsertPt] = getZAStateBeforeInst(
          *TRI, MI, /*ZAOffAtReturn=*/SMEFnAttrs.hasPrivateZAInterface());
      assert((InsertPt == MBBI ||
              InsertPt->getOpcode() == AArch64::ADJCALLSTACKDOWN) &&
             "Unexpected state change insertion point!");
      // TODO: Do something to avoid state changes where NZCV is live.
      if (MBBI == FirstTerminatorInsertPt)
        Block.PhysLiveRegsAtExit = PhysLiveRegs;
      if (NeededState != ZAState::ANY)
        Block.Insts.push_back({NeededState, InsertPt, PhysLiveRegs});
    }

    // Reverse vector (as we had to iterate backwards for liveness).
    std::reverse(Block.Insts.begin(), Block.Insts.end());
  }
}

void MachineSMEABI::assignBundleZAStates() {
  State.BundleStates.resize(EdgeBundles->getNumBundles());
  for (unsigned I = 0, E = EdgeBundles->getNumBundles(); I != E; ++I) {
    LLVM_DEBUG(dbgs() << "Assigning ZA state for edge bundle: " << I << '\n');

    // Attempt to assign a ZA state for this bundle that minimizes state
    // transitions. Edges within loops are given a higher weight as we assume
    // they will be executed more than once.
    // TODO: We should propagate desired incoming/outgoing states through blocks
    // that have the "ANY" state first to make better global decisions.
    int EdgeStateCounts[ZAState::NUM_ZA_STATE] = {0};
    for (unsigned BlockID : EdgeBundles->getBlocks(I)) {
      LLVM_DEBUG(dbgs() << "- bb." << BlockID);

      const BlockInfo &Block = State.Blocks[BlockID];
      if (Block.Insts.empty()) {
        LLVM_DEBUG(dbgs() << " (no state preference)\n");
        continue;
      }
      bool InEdge = EdgeBundles->getBundle(BlockID, /*Out=*/false) == I;
      bool OutEdge = EdgeBundles->getBundle(BlockID, /*Out=*/true) == I;

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

    State.BundleStates[I] = BundleState;
  }
}

void MachineSMEABI::insertStateChanges() {
  for (MachineBasicBlock &MBB : *MF) {
    const BlockInfo &Block = State.Blocks[MBB.getNumber()];
    ZAState InState = State.BundleStates[EdgeBundles->getBundle(MBB.getNumber(),
                                                                /*Out=*/false)];

    ZAState CurrentState = Block.FixedEntryState;
    if (CurrentState == ZAState::ANY)
      CurrentState = InState;

    for (auto &Inst : Block.Insts) {
      if (CurrentState != Inst.NeededState)
        emitStateChange(MBB, Inst.InsertPt, CurrentState, Inst.NeededState,
                        Inst.PhysLiveRegs);
      CurrentState = Inst.NeededState;
    }

    if (MBB.succ_empty())
      continue;

    ZAState OutState = State.BundleStates[EdgeBundles->getBundle(
        MBB.getNumber(), /*Out=*/true)];
    if (CurrentState != OutState)
      emitStateChange(MBB, MBB.getFirstTerminator(), CurrentState, OutState,
                      Block.PhysLiveRegsAtExit);
  }
}

TPIDR2State MachineSMEABI::getTPIDR2Block() {
  if (State.TPIDR2Block)
    return *State.TPIDR2Block;
  MachineFrameInfo &MFI = MF->getFrameInfo();
  State.TPIDR2Block = TPIDR2State{MFI.CreateStackObject(16, Align(16), false)};
  return *State.TPIDR2Block;
}

static DebugLoc getDebugLoc(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI) {
  if (MBBI != MBB.end())
    return MBBI->getDebugLoc();
  return DebugLoc();
}

void MachineSMEABI::emitSetupLazySave(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Get pointer to TPIDR2 block.
  Register TPIDR2 = MRI->createVirtualRegister(&AArch64::GPR64spRegClass);
  Register TPIDR2Ptr = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXri), TPIDR2)
      .addFrameIndex(getTPIDR2Block().FrameIndex)
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

void MachineSMEABI::restorePhyRegSave(PhysRegSave const &RegSave,
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

void MachineSMEABI::emitRestoreLazySave(MachineBasicBlock &MBB,
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
      .addFrameIndex(getTPIDR2Block().FrameIndex)
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
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  MachineFrameInfo &MFI = MF->getFrameInfo();

  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register SP = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  Register SVL = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  Register Buffer = MRI->createVirtualRegister(&AArch64::GPR64RegClass);

  // Calculate SVL.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::RDSVLI_XI), SVL).addImm(1);

  // 1. Allocate the lazy save buffer.
  {
    // TODO This function grows the stack with a subtraction, which doesn't work
    // on Windows. Some refactoring to share the functionality in
    // LowerWindowsDYNAMIC_STACKALLOC will be required once the Windows ABI
    // supports SME
    assert(!Subtarget->isTargetWindows() &&
           "Lazy ZA save is not yet supported on Windows");
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
        .addFrameIndex(getTPIDR2Block().FrameIndex)
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
  bool ZeroZA =
      MF->getInfo<AArch64FunctionInfo>()->getSMEFnAttrs().hasZAState();
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

void MachineSMEABI::emitStateChange(MachineBasicBlock &MBB,
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
    assert(MBB.getParent()
               ->getInfo<AArch64FunctionInfo>()
               ->getSMEFnAttrs()
               .hasPrivateZAInterface() &&
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
    emitSetupLazySave(MBB, InsertPt);
  else if (From == ZAState::LOCAL_SAVED && To == ZAState::ACTIVE)
    emitRestoreLazySave(MBB, InsertPt, PhysLiveRegs);
  else if (To == ZAState::OFF) {
    assert(From != ZAState::CALLER_DORMANT &&
           "CALLER_DORMANT to OFF should have already been handled");
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

  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  SMEAttrs SMEFnAttrs = AFI->getSMEFnAttrs();
  if (!SMEFnAttrs.hasZAState() && !SMEFnAttrs.hasZT0State())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  // Reset pass state.
  State = PassState{};
  this->MF = &MF;
  EdgeBundles = &getAnalysis<EdgeBundlesWrapperLegacy>().getEdgeBundles();
  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();
  MRI = &MF.getRegInfo();

  collectNeededZAStates(SMEFnAttrs);
  assignBundleZAStates();
  insertStateChanges();

  // Allocate save buffer (if needed).
  if (State.TPIDR2Block) {
    MachineBasicBlock &EntryBlock = MF.front();
    emitAllocateLazySaveBuffer(EntryBlock, EntryBlock.getFirstNonPHI());
  }

  return true;
}

FunctionPass *llvm::createMachineSMEABIPass() { return new MachineSMEABI(); }
