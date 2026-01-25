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
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-machine-sme-abi"

namespace {

// Note: For agnostic ZA, we assume the function is always entered/exited in the
// "ACTIVE" state -- this _may_ not be the case (since OFF is also a
// possibility, but for the purpose of placing ZA saves/restores, that does not
// matter).
enum ZAState : uint8_t {
  // Any/unknown state (not valid)
  ANY = 0,

  // ZA is in use and active (i.e. within the accumulator)
  ACTIVE,

  // ZA is active, but ZT0 has been saved.
  // This handles the edge case of sharedZA && !sharesZT0.
  ACTIVE_ZT0_SAVED,

  // A ZA save has been set up or committed (i.e. ZA is dormant or off)
  // If the function uses ZT0 it must also be saved.
  LOCAL_SAVED,

  // ZA has been committed to the lazy save buffer of the current function.
  // If the function uses ZT0 it must also be saved.
  // ZA is off.
  LOCAL_COMMITTED,

  // The ZA/ZT0 state on entry to the function.
  ENTRY,

  // ZA is off.
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
  SmallVector<InstInfo> Insts;
  ZAState FixedEntryState{ZAState::ANY};
  ZAState DesiredIncomingState{ZAState::ANY};
  ZAState DesiredOutgoingState{ZAState::ANY};
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

  int getZT0SaveSlot(MachineFunction &MF) {
    if (ZT0SaveFI)
      return *ZT0SaveFI;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    ZT0SaveFI = MFI.CreateSpillStackObject(64, Align(16));
    return *ZT0SaveFI;
  }

  /// Returns true if the function must allocate a ZA save buffer on entry. This
  /// will be the case if, at any point in the function, a ZA save was emitted.
  bool needsSaveBuffer() const {
    assert(!(TPIDR2BlockFI && AgnosticZABufferPtr) &&
           "Cannot have both a TPIDR2 block and agnostic ZA buffer");
    return TPIDR2BlockFI || AgnosticZABufferPtr != AArch64::NoRegister;
  }

private:
  std::optional<int> ZT0SaveFI;
  std::optional<int> TPIDR2BlockFI;
  Register AgnosticZABufferPtr = AArch64::NoRegister;
};

/// Checks if \p State is a legal edge bundle state. For a state to be a legal
/// bundle state, it must be possible to transition from it to any other bundle
/// state without losing any ZA state. This is the case for ACTIVE/LOCAL_SAVED,
/// as you can transition between those states by saving/restoring ZA. The OFF
/// state would not be legal, as transitioning to it drops the content of ZA.
static bool isLegalEdgeBundleZAState(ZAState State) {
  switch (State) {
  case ZAState::ACTIVE:           // ZA state within the accumulator/ZT0.
  case ZAState::ACTIVE_ZT0_SAVED: // ZT0 is saved (ZA is active).
  case ZAState::LOCAL_SAVED:      // ZA state may be saved on the stack.
  case ZAState::LOCAL_COMMITTED:  // ZA state is saved on the stack.
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
    MAKE_CASE(ZAState::ACTIVE_ZT0_SAVED)
    MAKE_CASE(ZAState::LOCAL_SAVED)
    MAKE_CASE(ZAState::LOCAL_COMMITTED)
    MAKE_CASE(ZAState::ENTRY)
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
getInstNeededZAState(const TargetRegisterInfo &TRI, MachineInstr &MI,
                     SMEAttrs SMEFnAttrs) {
  MachineBasicBlock::iterator InsertPt(MI);

  // Note: InOutZAUsePseudo, RequiresZASavePseudo, and RequiresZT0SavePseudo are
  // intended to mark the position immediately before a call. Due to
  // SelectionDAG constraints, these markers occur after the ADJCALLSTACKDOWN,
  // so we use std::prev(InsertPt) to get the position before the call.

  if (MI.getOpcode() == AArch64::InOutZAUsePseudo)
    return {ZAState::ACTIVE, std::prev(InsertPt)};

  // Note: If we need to save both ZA and ZT0 we use RequiresZASavePseudo.
  if (MI.getOpcode() == AArch64::RequiresZASavePseudo)
    return {ZAState::LOCAL_SAVED, std::prev(InsertPt)};

  // If we only need to save ZT0 there's two cases to consider:
  //   1. The function has ZA state (that we don't need to save).
  //      - In this case we switch to the "ACTIVE_ZT0_SAVED" state.
  //        This only saves ZT0.
  //   2. The function does not have ZA state
  //      - In this case we switch to "LOCAL_COMMITTED" state.
  //        This saves ZT0 and turns ZA off.
  if (MI.getOpcode() == AArch64::RequiresZT0SavePseudo) {
    return {SMEFnAttrs.hasZAState() ? ZAState::ACTIVE_ZT0_SAVED
                                    : ZAState::LOCAL_COMMITTED,
            std::prev(InsertPt)};
  }

  if (MI.isReturn()) {
    bool ZAOffAtReturn = SMEFnAttrs.hasPrivateZAInterface();
    return {ZAOffAtReturn ? ZAState::OFF : ZAState::ACTIVE, InsertPt};
  }

  for (auto &MO : MI.operands()) {
    if (isZAorZTRegOp(TRI, MO))
      return {ZAState::ACTIVE, InsertPt};
  }

  return {ZAState::ANY, InsertPt};
}

struct MachineSMEABI : public MachineFunctionPass {
  inline static char ID = 0;

  MachineSMEABI(CodeGenOptLevel OptLevel = CodeGenOptLevel::Default)
      : MachineFunctionPass(ID), OptLevel(OptLevel) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Machine SME ABI pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<EdgeBundlesWrapperLegacy>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    AU.addRequired<LibcallLoweringInfoWrapper>();
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

  /// Propagates desired states forwards (from predecessors -> successors) if
  /// \p Forwards, otherwise, propagates backwards (from successors ->
  /// predecessors).
  void propagateDesiredStates(FunctionInfo &FnInfo, bool Forwards = true);

  void addSMELibCall(MachineInstrBuilder &MIB, RTLIB::Libcall LC,
                     CallingConv::ID ExpectedCC);

  void emitZT0SaveRestore(EmitContext &, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI, bool IsSave);

  // Emission routines for private and shared ZA functions (using lazy saves).
  void emitSMEPrologue(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator MBBI);
  void emitRestoreLazySave(EmitContext &, MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           LiveRegs PhysLiveRegs);
  void emitSetupLazySave(EmitContext &, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  void emitAllocateLazySaveBuffer(EmitContext &, MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI);
  void emitZAMode(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  bool ClearTPIDR2, bool On);

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

  /// Collects the reachable calls from \p MBBI marked with \p Marker. This is
  /// intended to be used to emit lazy save remarks. Note: This stops at the
  /// first marked call along any path.
  void collectReachableMarkedCalls(const MachineBasicBlock &MBB,
                                   MachineBasicBlock::const_iterator MBBI,
                                   SmallVectorImpl<const MachineInstr *> &Calls,
                                   unsigned Marker) const;

  void emitCallSaveRemarks(const MachineBasicBlock &MBB,
                           MachineBasicBlock::const_iterator MBBI, DebugLoc DL,
                           unsigned Marker, StringRef RemarkName,
                           StringRef SaveName) const;

  void emitError(const Twine &Message) {
    LLVMContext &Context = MF->getFunction().getContext();
    Context.emitError(MF->getName() + ": " + Message);
  }

  /// Save live physical registers to virtual registers.
  PhysRegSave createPhysRegSave(LiveRegs PhysLiveRegs, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI, DebugLoc DL);
  /// Restore physical registers from a save of their previous values.
  void restorePhyRegSave(const PhysRegSave &RegSave, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, DebugLoc DL);

private:
  CodeGenOptLevel OptLevel = CodeGenOptLevel::Default;

  MachineFunction *MF = nullptr;
  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64RegisterInfo *TRI = nullptr;
  const AArch64FunctionInfo *AFI = nullptr;
  const AArch64InstrInfo *TII = nullptr;
  const LibcallLoweringInfo *LLI = nullptr;

  MachineOptimizationRemarkEmitter *ORE = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *MLI = nullptr;
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

[[maybe_unused]] bool isCallStartOpcode(unsigned Opc) {
  switch (Opc) {
  case AArch64::TLSDESC_CALLSEQ:
  case AArch64::TLSDESC_AUTH_CALLSEQ:
  case AArch64::ADJCALLSTACKDOWN:
    return true;
  default:
    return false;
  }
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
      Block.FixedEntryState = ZAState::ENTRY;
    } else if (MBB.isEHPad()) {
      // EH entry block:
      Block.FixedEntryState = ZAState::LOCAL_COMMITTED;
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
      auto [NeededState, InsertPt] = getInstNeededZAState(*TRI, MI, SMEFnAttrs);
      assert((InsertPt == MBBI || isCallStartOpcode(InsertPt->getOpcode())) &&
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

    // Record the desired states on entry/exit of this block. These are the
    // states that would not incur a state transition.
    if (!Block.Insts.empty()) {
      Block.DesiredIncomingState = Block.Insts.front().NeededState;
      Block.DesiredOutgoingState = Block.Insts.back().NeededState;
    }
  }

  return FunctionInfo{std::move(Blocks), AfterSMEProloguePt,
                      PhysLiveRegsAfterSMEPrologue};
}

void MachineSMEABI::propagateDesiredStates(FunctionInfo &FnInfo,
                                           bool Forwards) {
  // If `Forwards`, this propagates desired states from predecessors to
  // successors, otherwise, this propagates states from successors to
  // predecessors.
  auto GetBlockState = [](BlockInfo &Block, bool Incoming) -> ZAState & {
    return Incoming ? Block.DesiredIncomingState : Block.DesiredOutgoingState;
  };

  SmallVector<MachineBasicBlock *> Worklist;
  for (auto [BlockID, BlockInfo] : enumerate(FnInfo.Blocks)) {
    if (!isLegalEdgeBundleZAState(GetBlockState(BlockInfo, Forwards)))
      Worklist.push_back(MF->getBlockNumbered(BlockID));
  }

  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();
    BlockInfo &Block = FnInfo.Blocks[MBB->getNumber()];

    // Pick a legal edge bundle state that matches the majority of
    // predecessors/successors.
    int StateCounts[ZAState::NUM_ZA_STATE] = {0};
    for (MachineBasicBlock *PredOrSucc :
         Forwards ? predecessors(MBB) : successors(MBB)) {
      BlockInfo &PredOrSuccBlock = FnInfo.Blocks[PredOrSucc->getNumber()];
      ZAState ZAState = GetBlockState(PredOrSuccBlock, !Forwards);
      if (isLegalEdgeBundleZAState(ZAState))
        StateCounts[ZAState]++;
    }

    ZAState PropagatedState = ZAState(max_element(StateCounts) - StateCounts);
    ZAState &CurrentState = GetBlockState(Block, Forwards);
    if (PropagatedState != CurrentState) {
      CurrentState = PropagatedState;
      ZAState &OtherState = GetBlockState(Block, !Forwards);
      // Propagate to the incoming/outgoing state if that is also "ANY".
      if (OtherState == ZAState::ANY)
        OtherState = PropagatedState;
      // Push any successors/predecessors that may need updating to the
      // worklist.
      for (MachineBasicBlock *SuccOrPred :
           Forwards ? successors(MBB) : predecessors(MBB)) {
        BlockInfo &SuccOrPredBlock = FnInfo.Blocks[SuccOrPred->getNumber()];
        if (!isLegalEdgeBundleZAState(GetBlockState(SuccOrPredBlock, Forwards)))
          Worklist.push_back(SuccOrPred);
      }
    }
  }
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
    int EdgeStateCounts[ZAState::NUM_ZA_STATE] = {0};
    for (unsigned BlockID : Bundles.getBlocks(I)) {
      LLVM_DEBUG(dbgs() << "- bb." << BlockID);

      const BlockInfo &Block = FnInfo.Blocks[BlockID];
      bool InEdge = Bundles.getBundle(BlockID, /*Out=*/false) == I;
      bool OutEdge = Bundles.getBundle(BlockID, /*Out=*/true) == I;

      bool LegalInEdge =
          InEdge && isLegalEdgeBundleZAState(Block.DesiredIncomingState);
      bool LegalOutEgde =
          OutEdge && isLegalEdgeBundleZAState(Block.DesiredOutgoingState);
      if (LegalInEdge) {
        LLVM_DEBUG(dbgs() << " DesiredIncomingState: "
                          << getZAStateString(Block.DesiredIncomingState));
        EdgeStateCounts[Block.DesiredIncomingState]++;
      }
      if (LegalOutEgde) {
        LLVM_DEBUG(dbgs() << " DesiredOutgoingState: "
                          << getZAStateString(Block.DesiredOutgoingState));
        EdgeStateCounts[Block.DesiredOutgoingState]++;
      }
      if (!LegalInEdge && !LegalOutEgde)
        LLVM_DEBUG(dbgs() << " (no state preference)");
      LLVM_DEBUG(dbgs() << '\n');
    }

    ZAState BundleState =
        ZAState(max_element(EdgeStateCounts) - EdgeStateCounts);

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

  if (PhysLiveRegs == LiveRegs::None)
    return {InsertPt, PhysLiveRegs}; // Nothing to do (no live regs).

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
  auto BestCandidate = std::make_pair(InsertPt, PhysLiveRegs);
  for (MachineBasicBlock::iterator I = InsertPt; I != PrevStateChangeI; --I) {
    // Don't move before/into a call (which may have a state change before it).
    if (I->getOpcode() == TII->getCallFrameDestroyOpcode() || I->isCall())
      break;
    LiveUnits.stepBackward(*I);
    LiveRegs CurrentPhysLiveRegs = getPhysLiveRegs(LiveUnits);
    // Find places where NZCV is available, but keep looking for locations where
    // both NZCV and X0 are available, which can avoid some copies.
    if (!(CurrentPhysLiveRegs & LiveRegs::NZCV))
      BestCandidate = {I, CurrentPhysLiveRegs};
    if (CurrentPhysLiveRegs == LiveRegs::None)
      break;
  }
  return BestCandidate;
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
  if (MBB.empty())
    return DebugLoc();
  return MBBI != MBB.end() ? MBBI->getDebugLoc() : MBB.back().getDebugLoc();
}

/// Finds the first call (as determined by MachineInstr::isCall()) starting from
/// \p MBBI in \p MBB marked with \p Marker (which is a marker opcode such as
/// RequiresZASavePseudo). If a marked call is found, it is pushed to \p Calls
/// and the function returns true.
static bool findMarkedCall(const MachineBasicBlock &MBB,
                           MachineBasicBlock::const_iterator MBBI,
                           SmallVectorImpl<const MachineInstr *> &Calls,
                           unsigned Marker, unsigned CallDestroyOpcode) {
  auto IsMarker = [&](auto &MI) { return MI.getOpcode() == Marker; };
  auto MarkerInst = std::find_if(MBBI, MBB.end(), IsMarker);
  if (MarkerInst == MBB.end())
    return false;
  MachineBasicBlock::const_iterator I = MarkerInst;
  while (++I != MBB.end()) {
    if (I->isCall() || I->getOpcode() == CallDestroyOpcode)
      break;
  }
  if (I != MBB.end() && I->isCall())
    Calls.push_back(&*I);
  // Note: This function always returns true if a "Marker" was found.
  return true;
}

void MachineSMEABI::collectReachableMarkedCalls(
    const MachineBasicBlock &StartMBB,
    MachineBasicBlock::const_iterator StartInst,
    SmallVectorImpl<const MachineInstr *> &Calls, unsigned Marker) const {
  assert(Marker == AArch64::InOutZAUsePseudo ||
         Marker == AArch64::RequiresZASavePseudo ||
         Marker == AArch64::RequiresZT0SavePseudo);
  unsigned CallDestroyOpcode = TII->getCallFrameDestroyOpcode();
  if (findMarkedCall(StartMBB, StartInst, Calls, Marker, CallDestroyOpcode))
    return;

  SmallPtrSet<const MachineBasicBlock *, 4> Visited;
  SmallVector<const MachineBasicBlock *> Worklist(StartMBB.succ_rbegin(),
                                                  StartMBB.succ_rend());
  while (!Worklist.empty()) {
    const MachineBasicBlock *MBB = Worklist.pop_back_val();
    auto [_, Inserted] = Visited.insert(MBB);
    if (!Inserted)
      continue;

    if (!findMarkedCall(*MBB, MBB->begin(), Calls, Marker, CallDestroyOpcode))
      Worklist.append(MBB->succ_rbegin(), MBB->succ_rend());
  }
}

static StringRef getCalleeName(const MachineInstr &CallInst) {
  assert(CallInst.isCall() && "expected a call");
  for (const MachineOperand &MO : CallInst.operands()) {
    if (MO.isSymbol())
      return MO.getSymbolName();
    if (MO.isGlobal())
      return MO.getGlobal()->getName();
  }
  return {};
}

void MachineSMEABI::emitCallSaveRemarks(const MachineBasicBlock &MBB,
                                        MachineBasicBlock::const_iterator MBBI,
                                        DebugLoc DL, unsigned Marker,
                                        StringRef RemarkName,
                                        StringRef SaveName) const {
  auto SaveRemark = [&](DebugLoc DL, const MachineBasicBlock &MBB) {
    return MachineOptimizationRemarkAnalysis("sme", RemarkName, DL, &MBB);
  };
  StringRef StateName = Marker == AArch64::RequiresZT0SavePseudo ? "ZT0" : "ZA";
  ORE->emit([&] {
    return SaveRemark(DL, MBB) << SaveName << " of " << StateName
                               << " emitted in '" << MF->getName() << "'";
  });
  if (!ORE->allowExtraAnalysis("sme"))
    return;
  SmallVector<const MachineInstr *> CallsRequiringSaves;
  collectReachableMarkedCalls(MBB, MBBI, CallsRequiringSaves, Marker);
  for (const MachineInstr *CallInst : CallsRequiringSaves) {
    auto R = SaveRemark(CallInst->getDebugLoc(), *CallInst->getParent());
    R << "call";
    if (StringRef CalleeName = getCalleeName(*CallInst); !CalleeName.empty())
      R << " to '" << CalleeName << "'";
    R << " requires " << StateName << " save";
    ORE->emit(R);
  }
}

void MachineSMEABI::emitSetupLazySave(EmitContext &Context,
                                      MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  emitCallSaveRemarks(MBB, MBBI, DL, AArch64::RequiresZASavePseudo,
                      "SMELazySaveZA", "lazy save");

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

void MachineSMEABI::addSMELibCall(MachineInstrBuilder &MIB, RTLIB::Libcall LC,
                                  CallingConv::ID ExpectedCC) {
  RTLIB::LibcallImpl LCImpl = LLI->getLibcallImpl(LC);
  if (LCImpl == RTLIB::Unsupported)
    emitError("cannot lower SME ABI (SME routines unsupported)");
  CallingConv::ID CC = LLI->getLibcallImplCallingConv(LCImpl);
  StringRef ImplName = RTLIB::RuntimeLibcallsInfo::getLibcallImplName(LCImpl);
  if (CC != ExpectedCC)
    emitError("invalid calling convention for SME routine: '" + ImplName + "'");
  // FIXME: This assumes the ImplName StringRef is null-terminated.
  MIB.addExternalSymbol(ImplName.data());
  MIB.addRegMask(TRI->getCallPreservedMask(*MF, CC));
}

void MachineSMEABI::emitRestoreLazySave(EmitContext &Context,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        LiveRegs PhysLiveRegs) {
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
  auto RestoreZA = BuildMI(MBB, MBBI, DL, TII->get(AArch64::RestoreZAPseudo))
                       .addReg(TPIDR2EL0)
                       .addReg(TPIDR2);
  addSMELibCall(
      RestoreZA, RTLIB::SMEABI_TPIDR2_RESTORE,
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0);
  // Zero TPIDR2_EL0.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(AArch64::XZR);

  restorePhyRegSave(RegSave, MBB, MBBI, DL);
}

void MachineSMEABI::emitZAMode(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               bool ClearTPIDR2, bool On) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  if (ClearTPIDR2)
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSR))
        .addImm(AArch64SysReg::TPIDR2_EL0)
        .addReg(AArch64::XZR);

  // Disable ZA.
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(On ? 1 : 0);
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

static constexpr unsigned ZERO_ALL_ZA_MASK = 0b11111111;

void MachineSMEABI::emitSMEPrologue(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  bool ZeroZA = AFI->getSMEFnAttrs().isNewZA();
  bool ZeroZT0 = AFI->getSMEFnAttrs().isNewZT0();
  if (AFI->getSMEFnAttrs().hasPrivateZAInterface()) {
    // Get current TPIDR2_EL0.
    Register TPIDR2EL0 = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MRS))
        .addReg(TPIDR2EL0, RegState::Define)
        .addImm(AArch64SysReg::TPIDR2_EL0);
    // If TPIDR2_EL0 is non-zero, commit the lazy save.
    // NOTE: Functions that only use ZT0 don't need to zero ZA.
    auto CommitZASave =
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::CommitZASavePseudo))
            .addReg(TPIDR2EL0)
            .addImm(ZeroZA)
            .addImm(ZeroZT0);
    addSMELibCall(
        CommitZASave, RTLIB::SMEABI_TPIDR2_SAVE,
        CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0);
    if (ZeroZA)
      CommitZASave.addDef(AArch64::ZAB0, RegState::ImplicitDefine);
    if (ZeroZT0)
      CommitZASave.addDef(AArch64::ZT0, RegState::ImplicitDefine);
    // Enable ZA (as ZA could have previously been in the OFF state).
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MSRpstatesvcrImm1))
        .addImm(AArch64SVCR::SVCRZA)
        .addImm(1);
  } else if (AFI->getSMEFnAttrs().hasSharedZAInterface()) {
    if (ZeroZA)
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ZERO_M))
          .addImm(ZERO_ALL_ZA_MASK)
          .addDef(AArch64::ZAB0, RegState::ImplicitDefine);
    if (ZeroZT0)
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ZERO_T)).addDef(AArch64::ZT0);
  }
}

void MachineSMEABI::emitFullZASaveRestore(EmitContext &Context,
                                          MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          LiveRegs PhysLiveRegs, bool IsSave) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  if (IsSave)
    emitCallSaveRemarks(MBB, MBBI, DL, AArch64::RequiresZASavePseudo,
                        "SMEFullZASave", "full save");

  PhysRegSave RegSave = createPhysRegSave(PhysLiveRegs, MBB, MBBI, DL);

  // Copy the buffer pointer into X0.
  Register BufferPtr = AArch64::X0;
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), BufferPtr)
      .addReg(Context.getAgnosticZABufferPtr(*MF));

  // Call __arm_sme_save/__arm_sme_restore.
  auto SaveRestoreZA = BuildMI(MBB, MBBI, DL, TII->get(AArch64::BL))
                           .addReg(BufferPtr, RegState::Implicit);
  addSMELibCall(
      SaveRestoreZA,
      IsSave ? RTLIB::SMEABI_SME_SAVE : RTLIB::SMEABI_SME_RESTORE,
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1);

  restorePhyRegSave(RegSave, MBB, MBBI, DL);
}

void MachineSMEABI::emitZT0SaveRestore(EmitContext &Context,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       bool IsSave) {
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Note: This will report calls that _only_ need ZT0 saved. Call that save
  // both ZA and ZT0 will be under the SMELazySaveZA remark. This prevents
  // reporting the same calls twice.
  if (IsSave)
    emitCallSaveRemarks(MBB, MBBI, DL, AArch64::RequiresZT0SavePseudo,
                        "SMEZT0Save", "spill");

  Register ZT0Save = MRI->createVirtualRegister(&AArch64::GPR64spRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXri), ZT0Save)
      .addFrameIndex(Context.getZT0SaveSlot(*MF))
      .addImm(0)
      .addImm(0);

  if (IsSave) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::STR_TX))
        .addReg(AArch64::ZT0)
        .addReg(ZT0Save);
  } else {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDR_TX), AArch64::ZT0)
        .addReg(ZT0Save);
  }
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
    auto SMEStateSize = BuildMI(MBB, MBBI, DL, TII->get(AArch64::BL))
                            .addReg(AArch64::X0, RegState::ImplicitDefine);
    addSMELibCall(
        SMEStateSize, RTLIB::SMEABI_SME_STATE_SIZE,
        CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1);
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

struct FromState {
  ZAState From;

  constexpr uint8_t to(ZAState To) const {
    static_assert(NUM_ZA_STATE < 16, "expected ZAState to fit in 4-bits");
    return uint8_t(From) << 4 | uint8_t(To);
  }
};

constexpr FromState transitionFrom(ZAState From) { return FromState{From}; }

void MachineSMEABI::emitStateChange(EmitContext &Context,
                                    MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertPt,
                                    ZAState From, ZAState To,
                                    LiveRegs PhysLiveRegs) {
  // ZA not used.
  if (From == ZAState::ANY || To == ZAState::ANY)
    return;

  // If we're exiting from the ENTRY state that means that the function has not
  // used ZA, so in the case of private ZA/ZT0 functions we can omit any set up.
  if (From == ZAState::ENTRY && To == ZAState::OFF)
    return;

  // TODO: Avoid setting up the save buffer if there's no transition to
  // LOCAL_SAVED.
  if (From == ZAState::ENTRY) {
    assert(&MBB == &MBB.getParent()->front() &&
           "ENTRY state only valid in entry block");
    emitSMEPrologue(MBB, MBB.getFirstNonPHI());
    if (To == ZAState::ACTIVE)
      return; // Nothing more to do (ZA is active after the prologue).

    // Note: "emitNewZAPrologue" zeros ZA, so we may need to setup a lazy save
    // if "To" is "ZAState::LOCAL_SAVED". It may be possible to improve this
    // case by changing the placement of the zero instruction.
    From = ZAState::ACTIVE;
  }

  SMEAttrs SMEFnAttrs = AFI->getSMEFnAttrs();
  bool IsAgnosticZA = SMEFnAttrs.hasAgnosticZAInterface();
  bool HasZT0State = SMEFnAttrs.hasZT0State();
  bool HasZAState = IsAgnosticZA || SMEFnAttrs.hasZAState();

  switch (transitionFrom(From).to(To)) {
  // This section handles: ACTIVE <-> ACTIVE_ZT0_SAVED
  case transitionFrom(ZAState::ACTIVE).to(ZAState::ACTIVE_ZT0_SAVED):
    emitZT0SaveRestore(Context, MBB, InsertPt, /*IsSave=*/true);
    break;
  case transitionFrom(ZAState::ACTIVE_ZT0_SAVED).to(ZAState::ACTIVE):
    emitZT0SaveRestore(Context, MBB, InsertPt, /*IsSave=*/false);
    break;

  // This section handles: ACTIVE[_ZT0_SAVED] -> LOCAL_SAVED
  case transitionFrom(ZAState::ACTIVE).to(ZAState::LOCAL_SAVED):
  case transitionFrom(ZAState::ACTIVE_ZT0_SAVED).to(ZAState::LOCAL_SAVED):
    if (HasZT0State && From == ZAState::ACTIVE)
      emitZT0SaveRestore(Context, MBB, InsertPt, /*IsSave=*/true);
    if (HasZAState)
      emitZASave(Context, MBB, InsertPt, PhysLiveRegs);
    break;

  // This section handles: ACTIVE -> LOCAL_COMMITTED
  case transitionFrom(ZAState::ACTIVE).to(ZAState::LOCAL_COMMITTED):
    // TODO: We could support ZA state here, but this transition is currently
    // only possible when we _don't_ have ZA state.
    assert(HasZT0State && !HasZAState && "Expect to only have ZT0 state.");
    emitZT0SaveRestore(Context, MBB, InsertPt, /*IsSave=*/true);
    emitZAMode(MBB, InsertPt, /*ClearTPIDR2=*/false, /*On=*/false);
    break;

  // This section handles: LOCAL_COMMITTED -> (OFF|LOCAL_SAVED)
  case transitionFrom(ZAState::LOCAL_COMMITTED).to(ZAState::OFF):
  case transitionFrom(ZAState::LOCAL_COMMITTED).to(ZAState::LOCAL_SAVED):
    // These transistions are a no-op.
    break;

  // This section handles: LOCAL_(SAVED|COMMITTED) -> ACTIVE[_ZT0_SAVED]
  case transitionFrom(ZAState::LOCAL_COMMITTED).to(ZAState::ACTIVE):
  case transitionFrom(ZAState::LOCAL_COMMITTED).to(ZAState::ACTIVE_ZT0_SAVED):
  case transitionFrom(ZAState::LOCAL_SAVED).to(ZAState::ACTIVE):
    if (HasZAState)
      emitZARestore(Context, MBB, InsertPt, PhysLiveRegs);
    else
      emitZAMode(MBB, InsertPt, /*ClearTPIDR2=*/false, /*On=*/true);
    if (HasZT0State && To == ZAState::ACTIVE)
      emitZT0SaveRestore(Context, MBB, InsertPt, /*IsSave=*/false);
    break;

  // This section handles transistions to OFF (not previously covered)
  case transitionFrom(ZAState::ACTIVE).to(ZAState::OFF):
  case transitionFrom(ZAState::ACTIVE_ZT0_SAVED).to(ZAState::OFF):
  case transitionFrom(ZAState::LOCAL_SAVED).to(ZAState::OFF):
    assert(SMEFnAttrs.hasPrivateZAInterface() &&
           "Did not expect to turn ZA off in shared/agnostic ZA function");
    emitZAMode(MBB, InsertPt, /*ClearTPIDR2=*/From == ZAState::LOCAL_SAVED,
               /*On=*/false);
    break;

  default:
    dbgs() << "Error: Transition from " << getZAStateString(From) << " to "
           << getZAStateString(To) << '\n';
    llvm_unreachable("Unimplemented state transition");
  }
}

} // end anonymous namespace

INITIALIZE_PASS(MachineSMEABI, "aarch64-machine-sme-abi", "Machine SME ABI",
                false, false)

bool MachineSMEABI::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  if (!Subtarget->hasSME())
    return false;

  AFI = MF.getInfo<AArch64FunctionInfo>();
  SMEAttrs SMEFnAttrs = AFI->getSMEFnAttrs();
  if (!SMEFnAttrs.hasZAState() && !SMEFnAttrs.hasZT0State() &&
      !SMEFnAttrs.hasAgnosticZAInterface())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  this->MF = &MF;
  ORE = &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  LLI = &getAnalysis<LibcallLoweringInfoWrapper>().getLibcallLowering(
      *MF.getFunction().getParent(), *Subtarget);
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();
  MRI = &MF.getRegInfo();

  const EdgeBundles &Bundles =
      getAnalysis<EdgeBundlesWrapperLegacy>().getEdgeBundles();

  FunctionInfo FnInfo = collectNeededZAStates(SMEFnAttrs);

  if (OptLevel != CodeGenOptLevel::None) {
    // Propagate desired states forward, then backwards. Most of the propagation
    // should be done in the forward step, and backwards propagation is then
    // used to fill in the gaps. Note: Doing both in one step can give poor
    // results. For example, consider this subgraph:
    //
    //    ┌─────┐
    //  ┌─┤ BB0 ◄───┐
    //  │ └─┬───┘   │
    //  │ ┌─▼───◄──┐│
    //  │ │ BB1 │  ││
    //  │ └─┬┬──┘  ││
    //  │   │└─────┘│
    //  │ ┌─▼───┐   │
    //  │ │ BB2 ├───┘
    //  │ └─┬───┘
    //  │ ┌─▼───┐
    //  └─► BB3 │
    //    └─────┘
    //
    // If:
    // - "BB0" and "BB2" (outer loop) has no state preference
    // - "BB1" (inner loop) desires the ACTIVE state on entry/exit
    // - "BB3" desires the LOCAL_SAVED state on entry
    //
    // If we propagate forwards first, ACTIVE is propagated from BB1 to BB2,
    // then from BB2 to BB0. Which results in the inner and outer loops having
    // the "ACTIVE" state. This avoids any state changes in the loops.
    //
    // If we propagate backwards first, we _could_ propagate LOCAL_SAVED from
    // BB3 to BB0, which would result in a transition from ACTIVE -> LOCAL_SAVED
    // in the outer loop.
    for (bool Forwards : {true, false})
      propagateDesiredStates(FnInfo, Forwards);
  }

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

FunctionPass *llvm::createMachineSMEABIPass(CodeGenOptLevel OptLevel) {
  return new MachineSMEABI(OptLevel);
}
