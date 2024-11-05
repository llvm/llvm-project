//===- AMDGPUWaitSGPRHazards.cpp - Mitigate SGPR read hazards -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert s_wait_alu instructions to mitigate SGPR read hazards on GFX12.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SetVector.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-wait-sgpr-hazards"

static cl::opt<bool> EnableSGPRHazardWaits(
    "amdgpu-sgpr-hazard-wait", cl::init(true), cl::Hidden,
    cl::desc("Enable required s_wait_alu on SGPR hazards"));

static cl::opt<bool> CullSGPRHazardsOnFunctionBoundary(
    "amdgpu-sgpr-hazard-boundary-cull", cl::init(false), cl::Hidden,
    cl::desc("Cull hazards on function boundaries"));

static cl::opt<bool>
    CullSGPRHazardsAtMemWait("amdgpu-sgpr-hazard-mem-wait-cull",
                             cl::init(false), cl::Hidden,
                             cl::desc("Cull hazards on memory waits"));

static cl::opt<unsigned> CullSGPRHazardsMemWaitThreshold(
    "amdgpu-sgpr-hazard-mem-wait-cull-threshold", cl::init(1), cl::Hidden,
    cl::desc("Number of tracked SGPRs before initiating hazard cull on memory "
             "wait"));

namespace {

class AMDGPUWaitSGPRHazards : public MachineFunctionPass {
public:
  static char ID;

  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  bool Wave64;

  AMDGPUWaitSGPRHazards() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Return the numeric ID 0-127 for a given SGPR.
  static std::optional<unsigned> sgprNumber(Register Reg,
                                            const SIRegisterInfo &TRI) {
    switch (Reg) {
    case AMDGPU::M0:
    case AMDGPU::EXEC:
    case AMDGPU::EXEC_LO:
    case AMDGPU::EXEC_HI:
    case AMDGPU::SGPR_NULL:
    case AMDGPU::SGPR_NULL64:
      return {};
    default:
      break;
    }
    unsigned RegN = TRI.getEncodingValue(Reg);
    if (RegN > 127)
      return {};
    return RegN;
  }

  static inline bool IsVCC(Register Reg) {
    return (Reg == AMDGPU::VCC || Reg == AMDGPU::VCC_LO ||
            Reg == AMDGPU::VCC_HI);
  }

  // Adjust global offsets for instructions bundled with S_GETPC_B64 after
  // insertion of a new instruction.
  static void updateGetPCBundle(MachineInstr *NewMI) {
    if (!NewMI->isBundled())
      return;

    // Find start of bundle.
    auto I = NewMI->getIterator();
    while (I->isBundledWithPred())
      I--;
    if (I->isBundle())
      I++;

    // Bail if this is not an S_GETPC bundle.
    if (I->getOpcode() != AMDGPU::S_GETPC_B64)
      return;

    // Update offsets of any references in the bundle.
    const unsigned NewBytes = 4;
    assert(NewMI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR &&
           "Unexpected instruction insertion in bundle");
    auto NextMI = std::next(NewMI->getIterator());
    auto End = NewMI->getParent()->end();
    while (NextMI != End && NextMI->isBundledWithPred()) {
      for (auto &Operand : NextMI->operands()) {
        if (Operand.isGlobal())
          Operand.setOffset(Operand.getOffset() + NewBytes);
      }
      NextMI++;
    }
  }

  // Total order instructions, and use this for VA_SDST?
  struct HazardState {
    static constexpr unsigned None = 0;
    static constexpr unsigned SALU = (1 << 0);
    static constexpr unsigned VALU = (1 << 1);

    std::bitset<64> Tracked;      // SGPR banks ever read by VALU
    std::bitset<128> SALUHazards; // SGPRs with uncommitted values from SALU
    std::bitset<128> VALUHazards; // SGPRs with uncommitted values from VALU
    unsigned VCCHazard = None;    // Source of current VCC writes
    bool ActiveFlat = false;      // Has unwaited flat instructions

    bool merge(const HazardState &RHS) {
      HazardState Orig(*this);

      Tracked |= RHS.Tracked;
      SALUHazards |= RHS.SALUHazards;
      VALUHazards |= RHS.VALUHazards;
      VCCHazard |= RHS.VCCHazard;
      ActiveFlat |= RHS.ActiveFlat;

      return (*this != Orig);
    }

    bool operator==(const HazardState &RHS) const {
      return Tracked == RHS.Tracked && SALUHazards == RHS.SALUHazards &&
             VALUHazards == RHS.VALUHazards && VCCHazard == RHS.VCCHazard &&
             ActiveFlat == RHS.ActiveFlat;
    }
    bool operator!=(const HazardState &RHS) const { return !(*this == RHS); }
  };

  struct BlockHazardState {
    HazardState In;
    HazardState Out;
  };

  DenseMap<const MachineBasicBlock *, BlockHazardState> BlockState;

  static constexpr unsigned WAVE32_NOPS = 4;
  static constexpr unsigned WAVE64_NOPS = 8;

  void insertHazardCull(MachineBasicBlock &MBB,
                        MachineBasicBlock::instr_iterator &MI) {
    assert(!MI->isBundled());
    unsigned Count = Wave64 ? WAVE64_NOPS : WAVE32_NOPS;
    while (Count--)
      BuildMI(MBB, MI, MI->getDebugLoc(), TII->get(AMDGPU::DS_NOP));
  }

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB, bool Emit) {
    enum { WA_VALU = 0x1, WA_SALU = 0x2, WA_VCC = 0x4 };

    HazardState State = BlockState[&MBB].In;
    SmallSet<Register, 8> SeenRegs;
    bool Emitted = false;
    unsigned DsNops = 0;

    for (MachineBasicBlock::instr_iterator MI = MBB.instr_begin(),
                                           E = MBB.instr_end();
         MI != E; ++MI) {
      // Clear tracked SGPRs if sufficient DS_NOPs occur
      if (MI->getOpcode() == AMDGPU::DS_NOP) {
        if (++DsNops >= (Wave64 ? WAVE64_NOPS : WAVE32_NOPS))
          State.Tracked.reset();
        continue;
      }
      DsNops = 0;

      // Snoop FLAT instructions to avoid adding culls before scratch/lds loads.
      // Culls could be disproportionate in cost to load time.
      if (SIInstrInfo::isFLAT(*MI) && !SIInstrInfo::isFLATGlobal(*MI))
        State.ActiveFlat = true;

      // SMEM or VMEM clears hazards
      if (SIInstrInfo::isVMEM(*MI) || SIInstrInfo::isSMRD(*MI)) {
        State.VCCHazard = HazardState::None;
        State.SALUHazards.reset();
        State.VALUHazards.reset();
        continue;
      }

      // Existing S_WAITALU can clear hazards
      if (MI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR) {
        unsigned int Mask = MI->getOperand(0).getImm();
        if (AMDGPU::DepCtr::decodeFieldVaVcc(Mask) == 0)
          State.VCCHazard = HazardState::None;
        if (AMDGPU::DepCtr::decodeFieldSaSdst(Mask) == 0)
          State.SALUHazards.reset();
        if (AMDGPU::DepCtr::decodeFieldVaSdst(Mask) == 0)
          State.VALUHazards.reset();
        continue;
      }

      // Snoop counter waits to insert culls
      if (CullSGPRHazardsAtMemWait &&
          (MI->getOpcode() == AMDGPU::S_WAIT_LOADCNT ||
           MI->getOpcode() == AMDGPU::S_WAIT_SAMPLECNT ||
           MI->getOpcode() == AMDGPU::S_WAIT_BVHCNT) &&
          (MI->getOperand(0).isImm() && MI->getOperand(0).getImm() == 0) &&
          (State.Tracked.count() >= CullSGPRHazardsMemWaitThreshold)) {
        if (MI->getOpcode() == AMDGPU::S_WAIT_LOADCNT && State.ActiveFlat) {
          State.ActiveFlat = false;
        } else {
          State.Tracked.reset();
          if (Emit)
            insertHazardCull(MBB, MI);
          continue;
        }
      }

      // Process only VALUs and SALUs
      bool IsVALU = SIInstrInfo::isVALU(*MI);
      bool IsSALU = SIInstrInfo::isSALU(*MI);
      if (!IsVALU && !IsSALU)
        continue;

      unsigned Wait = 0;

      auto processOperand = [&](const MachineOperand &Op, bool IsUse) {
        if (!Op.isReg())
          return;
        Register Reg = Op.getReg();
        assert(!Op.getSubReg());
        // Only consider implicit operands of VCC.
        if (Op.isImplicit() && !IsVCC(Reg))
          return;
        if (!TRI->isSGPRReg(*MRI, Reg))
          return;

        // Only visit each register once
        if (!SeenRegs.insert(Reg).second)
          return;

        auto RegNumber = sgprNumber(Reg, *TRI);
        if (!RegNumber)
          return;

        // Track SGPRs by pair -- numeric ID of an 64b SGPR pair.
        // i.e. SGPR0 = SGPR0_SGPR1 = 0, SGPR3 = SGPR2_SGPR3 = 1, etc
        unsigned RegN = *RegNumber;
        unsigned PairN = (RegN >> 1) & 0x3f;

        // Read/write of untracked register is safe; but must record any new
        // reads.
        if (!State.Tracked[PairN]) {
          if (IsVALU && IsUse)
            State.Tracked.set(PairN);
          return;
        }

        uint8_t SGPRCount =
            AMDGPU::getRegBitWidth(*TRI->getRegClassForReg(*MRI, Reg)) / 32;

        if (IsUse) {
          // SALU reading SGPR clears VALU hazards
          if (IsSALU) {
            if (IsVCC(Reg)) {
              if (State.VCCHazard & HazardState::VALU)
                State.VCCHazard = HazardState::None;
            } else {
              State.VALUHazards.reset();
            }
          }
          // Compute required waits
          for (uint8_t RegIdx = 0; RegIdx < SGPRCount; ++RegIdx) {
            Wait |= State.SALUHazards[RegN + RegIdx] ? WA_SALU : 0;
            Wait |= IsVALU && State.VALUHazards[RegN + RegIdx] ? WA_VALU : 0;
          }
          if (IsVCC(Reg) && State.VCCHazard)
            Wait |= WA_VCC;
        } else {
          // Update hazards
          if (IsVCC(Reg)) {
            State.VCCHazard = IsSALU ? HazardState::SALU : HazardState::VALU;
          } else {
            for (uint8_t RegIdx = 0; RegIdx < SGPRCount; ++RegIdx) {
              if (IsSALU)
                State.SALUHazards.set(RegN + RegIdx);
              else
                State.VALUHazards.set(RegN + RegIdx);
            }
          }
        }
      };

      const bool IsSetPC = (MI->isCall() || MI->isReturn() ||
                            MI->getOpcode() == AMDGPU::S_SETPC_B64) &&
                           !(MI->getOpcode() == AMDGPU::S_ENDPGM ||
                             MI->getOpcode() == AMDGPU::S_ENDPGM_SAVED);

      if (IsSetPC) {
        // All SGPR writes before a call/return must be flushed as the
        // callee/caller will not will not see the hazard chain.
        if (State.VCCHazard)
          Wait |= WA_VCC;
        if (State.SALUHazards.any())
          Wait |= WA_SALU;
        if (State.VALUHazards.any())
          Wait |= WA_VALU;
        if (CullSGPRHazardsOnFunctionBoundary && State.Tracked.any()) {
          State.Tracked.reset();
          if (Emit)
            insertHazardCull(MBB, MI);
        }
      } else {
        // Process uses to determine required wait.
        SeenRegs.clear();
        for (const MachineOperand &Op : MI->all_uses())
          processOperand(Op, true);
      }

      // Apply wait
      if (Wait) {
        unsigned Mask = 0xffff;
        if (Wait & WA_VCC) {
          State.VCCHazard = HazardState::None;
          Mask = AMDGPU::DepCtr::encodeFieldVaVcc(Mask, 0);
        }
        if (Wait & WA_SALU) {
          State.SALUHazards.reset();
          Mask = AMDGPU::DepCtr::encodeFieldSaSdst(Mask, 0);
        }
        if (Wait & WA_VALU) {
          State.VALUHazards.reset();
          Mask = AMDGPU::DepCtr::encodeFieldVaSdst(Mask, 0);
        }
        if (Emit) {
          auto NewMI = BuildMI(MBB, MI, MI->getDebugLoc(),
                               TII->get(AMDGPU::S_WAITCNT_DEPCTR))
                           .addImm(Mask);
          updateGetPCBundle(NewMI);
          Emitted = true;
        }
      }

      // On return from a call SGPR state is unknown, so all potential hazards.
      if (MI->isCall() && !CullSGPRHazardsOnFunctionBoundary)
        State.Tracked.set();

      // Update hazards based on defs.
      SeenRegs.clear();
      for (const MachineOperand &Op : MI->all_defs())
        processOperand(Op, false);
    }

    bool Changed = State != BlockState[&MBB].Out;
    if (Emit) {
      assert(!Changed && "Hazard state should not change on emit pass");
      return Emitted;
    }
    if (Changed)
      BlockState[&MBB].Out = State;
    return Changed;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    if (!ST.hasVALUReadSGPRHazard())
      return false;

    if (!EnableSGPRHazardWaits)
      return false;

    LLVM_DEBUG(dbgs() << "AMDGPUWaitSGPRHazards running on " << MF.getName()
                      << "\n");

    TII = ST.getInstrInfo();
    TRI = ST.getRegisterInfo();
    MRI = &(MF.getRegInfo());
    Wave64 = ST.isWave64();

    auto CallingConv = MF.getFunction().getCallingConv();
    if (!AMDGPU::isEntryFunctionCC(CallingConv) && !MF.empty() &&
        !CullSGPRHazardsOnFunctionBoundary) {
      // Callee must consider all SGPRs as tracked.
      LLVM_DEBUG(dbgs() << "Is called function, track all SGPRs.\n");
      MachineBasicBlock &EntryBlock = MF.front();
      BlockState[&EntryBlock].In.Tracked.set();
    }

    // Calculate the hazard state for each basic block.
    // Iterate until a fixed point is reached.
    // Fixed point is guaranteed as merge function only ever increases
    // the hazard set, and all backedges will cause a merge.
    //
    // Note: we have to take care of the entry block as this technically
    // has an edge from outside the function. Failure to treat this as
    // a merge could prevent fixed point being reached.
    SetVector<MachineBasicBlock *> Worklist;
    for (auto &MBB : reverse(MF))
      Worklist.insert(&MBB);
    while (!Worklist.empty()) {
      auto &MBB = *Worklist.pop_back_val();
      bool Changed = runOnMachineBasicBlock(MBB, false);
      if (Changed) {
        // Note: take a copy of state here in case it is reallocated by map
        HazardState NewState = BlockState[&MBB].Out;
        // Propagate to all successor blocks
        for (auto Succ : MBB.successors()) {
          // We only need to merge hazards at CFG merge points.
          if (Succ->getSinglePredecessor() && !Succ->isEntryBlock()) {
            if (BlockState[Succ].In != NewState) {
              BlockState[Succ].In = NewState;
              Worklist.insert(Succ);
            }
          } else if (BlockState[Succ].In.merge(NewState)) {
            Worklist.insert(Succ);
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << "Emit s_wait_alu instructions\n");

    // Final to emit wait instructions.
    bool Changed = false;
    for (auto &MBB : MF)
      Changed |= runOnMachineBasicBlock(MBB, true);

    BlockState.clear();
    return Changed;
  }
};

} // namespace

char AMDGPUWaitSGPRHazards::ID = 0;

char &llvm::AMDGPUWaitSGPRHazardsID = AMDGPUWaitSGPRHazards::ID;

INITIALIZE_PASS(AMDGPUWaitSGPRHazards, DEBUG_TYPE,
                "AMDGPU Mitigate SGPR Hazards", false, false)
