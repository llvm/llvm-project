//===- AMDGPUSim/Simulator.cpp - Single-wave simulation logic ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements the representation independent single wave simulation core,
/// including issue timing, stall accounting, and state updates.
//
//===----------------------------------------------------------------------===//

#include "Simulator.h"
#include <algorithm>

namespace llvm {
namespace AMDGPUSim {

namespace {

struct StallSources {
  StallBreakdown Breakdown;
  /// WMMA start cycle used only for scaled WMMA state recording because its
  /// computed issue cycle represents the preceding scale read.
  unsigned WMMAStartCycle = 0;
};

static void applyStall(unsigned &IssueCycle, unsigned StallUntil) {
  IssueCycle = std::max(IssueCycle, StallUntil);
}

// Break ties by the order of these checks so reporting remains deterministic.
static StallReason getDominantStallReason(const StallBreakdown &Breakdown) {
  unsigned Max = 0;
  StallReason Reason = StallReason::NONE;
  auto Check = [&](unsigned Value, StallReason Candidate) {
    if (Value > Max) {
      Max = Value;
      Reason = Candidate;
    }
  };

  Check(Breakdown.WaitCnt, StallReason::WAITCNT);
  Check(Breakdown.DelayAlu, StallReason::DELAY_ALU);
  Check(Breakdown.LongLatVALU, StallReason::LONG_LAT_VALU);
  Check(Breakdown.LOLVALUTRANSHazard, StallReason::LOLVALU_TRANS_HAZARD);
  Check(Breakdown.CoExec, StallReason::COEXEC_BLOCKED);
  Check(Breakdown.VALUSlot, StallReason::COEXEC_BLOCKED);
  Check(Breakdown.MemFIFO, StallReason::MEM_FIFO);
  Check(Breakdown.FU, StallReason::FU_BUSY);
  Check(Breakdown.SSRC, StallReason::VA_SSRC_STALL);
  Check(Breakdown.VaVdst, StallReason::VA_VDST_WAIT);
  return Reason;
}

static bool canMSBSetFuse(InstClass PreviousClass) {
  switch (PreviousClass) {
  case InstClass::VALU:
  case InstClass::TRANS:
  case InstClass::SALU:
  case InstClass::WMMA:
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
  case InstClass::SMEM:
  case InstClass::TDM:
    return true;
  default:
    return false;
  }
}

} // namespace

class Simulator::Impl {
  const SimInstInfo &InstInfo;
  const HWModel &Model;
  GPUSimState State;
  SimulatorConfig Config;
  raw_ostream *Log = nullptr;

  // Dependency values 1 through 4 select recent VALU instructions, 5 through
  // 7 select recent TRANS instructions, and 9 through 12 select SALU delays.
  unsigned decodeDelayDependency(unsigned Dependency) const {
    if (Dependency >= 1 && Dependency <= 4) {
      unsigned Index = Dependency - 1;
      if (Index < State.RecentVALU.size()) {
        const GPUSimState::RecentInst &Recent =
            State.RecentVALU[State.RecentVALU.size() - 1 - Index];
        unsigned Elapsed = State.CurrentCycle - Recent.IssueCycle;
        return Elapsed < Recent.Latency ? Recent.Latency - Elapsed : 0;
      }
    } else if (Dependency >= 5 && Dependency <= 7) {
      unsigned Index = Dependency - 5;
      if (Index < State.RecentTRANS.size()) {
        const GPUSimState::RecentInst &Recent =
            State.RecentTRANS[State.RecentTRANS.size() - 1 - Index];
        unsigned Elapsed = State.CurrentCycle - Recent.IssueCycle;
        return Elapsed < Recent.Latency ? Recent.Latency - Elapsed : 0;
      }
    } else if (Dependency >= 9 && Dependency <= 12) {
      unsigned WaitCycles = Dependency - 8;
      unsigned Elapsed = State.CurrentCycle - State.LastSALUCycle;
      return Elapsed < WaitCycles ? WaitCycles - Elapsed : 0;
    }
    return 0;
  }

  // Decrement the deferred skip count or apply the second dependency.
  // SkipApply leaves a ready dependency pending across S_SET_VGPR_MSB.
  unsigned checkPendingDelayAlu(bool SkipApply = false) {
    if (!State.PendingInstId1)
      return 0;
    GPUSimState::PendingDelayAlu &Pending = *State.PendingInstId1;
    if (Pending.InstructionsLeft > 0) {
      --Pending.InstructionsLeft;
      return 0;
    }
    if (SkipApply)
      return 0;
    unsigned Stall = decodeDelayDependency(Pending.DepType);
    State.PendingInstId1.reset();
    return Stall;
  }

  // Decode dependency 0 from bits 3 through 0, the skip count from bits 6
  // through 4, and dependency 1 from bits 10 through 7.
  unsigned parseDelayAlu(const SimInst &Inst) {
    unsigned Immediate = InstInfo.getDelayAluImm(Inst);
    if (Immediate == 0)
      return 0;

    unsigned FirstDependency = Immediate & 0xF;
    unsigned Skip = (Immediate >> 4) & 0x7;
    unsigned SecondDependency = (Immediate >> 7) & 0xF;
    if (SecondDependency != 0)
      State.PendingInstId1 =
          GPUSimState::PendingDelayAlu{SecondDependency, Skip};
    return decodeDelayDependency(FirstDependency);
  }

  unsigned computeWaitStall(const WaitRequirement &Wait) const {
    switch (Wait.Type) {
    case WaitType::DS:
      return State.computeWaitStall(State.PendingDS, Wait.Count);
    case WaitType::VMEMLoad:
      return State.computeWaitStall(State.PendingVMEMLoad, Wait.Count);
    case WaitType::VMEMStore:
      return State.computeWaitStall(State.PendingVMEMStore, Wait.Count);
    case WaitType::SMEM:
      return State.computeWaitStall(State.PendingSMEM, Wait.Count);
    case WaitType::Tensor:
      return State.computeWaitStall(State.PendingTDM, Wait.Count);
    case WaitType::XCnt:
      return State.computeWaitStall(State.PendingXACK, Wait.Count);
    case WaitType::DepCtr:
      // VaVdst waits use their target value in computeStallSources.
      return 0;
    }
    return 0;
  }

  unsigned computeWaitStall(ArrayRef<WaitRequirement> Waits) const {
    unsigned Stall = 0;
    for (const WaitRequirement &Wait : Waits)
      Stall = std::max(Stall, computeWaitStall(Wait));
    return Stall;
  }

  void applyWait(const WaitRequirement &Wait) {
    switch (Wait.Type) {
    case WaitType::DS:
      State.waitDS(Wait.Count);
      break;
    case WaitType::VMEMLoad:
      State.waitVMEMLoad(Wait.Count);
      break;
    case WaitType::VMEMStore:
      State.waitVMEMStore(Wait.Count);
      break;
    case WaitType::SMEM:
      State.waitSMEM(Wait.Count);
      break;
    case WaitType::Tensor:
      State.waitTensor(Wait.Count);
      break;
    case WaitType::XCnt:
      State.waitXCnt(Wait.Count);
      break;
    case WaitType::DepCtr:
      // VaVdst retirement is driven by cycle advancement.
      break;
    }
  }

  void applyWait(ArrayRef<WaitRequirement> Waits) {
    for (const WaitRequirement &Wait : Waits)
      applyWait(Wait);
  }

  // VALU always observes this reservation. TRANS observes it only while a
  // WMMA coexecution window is active.
  unsigned computeVALUResourceStall(InstClass IC, unsigned IssueCycle) const {
    if ((IC == InstClass::VALU ||
         (IC == InstClass::TRANS && State.inWMMAWindow())) &&
        State.VALUResourceBusyUntil > IssueCycle)
      return State.VALUResourceBusyUntil - IssueCycle;
    return 0;
  }

  // Place LD_SCALE in an allowed VALU slot and start matrix execution in the
  // immediately following cycle.
  void computeScaledWMMAStall(unsigned &IssueCycle,
                              StallSources &Sources) const {
    unsigned ScaleReadCycle = State.resolveScaledWMMAAbsorbCycle(
        std::max(IssueCycle, State.VALUResourceBusyUntil));
    unsigned MatrixStartCycle = ScaleReadCycle + 1;
    unsigned XDLFreeCycle = State.getUnitBusyUntil(FunctionalUnit::XDL);

    if (MatrixStartCycle < XDLFreeCycle) {
      unsigned DesiredScaleCycle = XDLFreeCycle - 1;
      if (State.inWMMAWindow() && XDLFreeCycle >= State.ActiveWMMA.StartCycle &&
          XDLFreeCycle < State.ActiveWMMA.EndCycle) {
        if (State.canAbsorbScaledWMMAAt(DesiredScaleCycle)) {
          ScaleReadCycle = DesiredScaleCycle;
          MatrixStartCycle = XDLFreeCycle;
        } else {
          ScaleReadCycle = State.ActiveWMMA.EndCycle;
          MatrixStartCycle = ScaleReadCycle + 1;
        }
      } else {
        ScaleReadCycle =
            std::max(DesiredScaleCycle, State.VALUResourceBusyUntil);
        ScaleReadCycle = State.resolveScaledWMMAAbsorbCycle(ScaleReadCycle);
        MatrixStartCycle = ScaleReadCycle + 1;
      }
    }

    IssueCycle = ScaleReadCycle;
    Sources.WMMAStartCycle = MatrixStartCycle;
    Sources.Breakdown.VALUSlot = ScaleReadCycle - State.CurrentCycle;
    if (XDLFreeCycle > State.CurrentCycle + 1)
      Sources.Breakdown.FU = XDLFreeCycle - State.CurrentCycle - 1;
  }

  // Issue when XDL is free. If that cycle is still in an active window, only
  // a vacant matrix stage can accept the next WMMA.
  void computeUnscaledWMMAStall(unsigned &IssueCycle,
                                StallSources &Sources) const {
    unsigned XDLFreeCycle = State.getUnitBusyUntil(FunctionalUnit::XDL);
    if (IssueCycle < XDLFreeCycle) {
      if (std::optional<unsigned> Stage =
              State.ActiveWMMA.getCurrentStage(XDLFreeCycle)) {
        const AMDGPU::CoExecInfo &CoExec = State.ActiveWMMA.Info;
        IssueCycle = CoExec.getType(*Stage) == AMDGPU::CoExecStageType::V
                         ? XDLFreeCycle
                         : State.ActiveWMMA.EndCycle;
      } else {
        IssueCycle = XDLFreeCycle;
      }
      Sources.Breakdown.FU = IssueCycle - State.CurrentCycle;
    }
    Sources.WMMAStartCycle = IssueCycle;
  }

  unsigned computeMemFIFOStall(InstClass IC) const {
    switch (IC) {
    case InstClass::DS_READ:
    case InstClass::DS_WRITE:
      return State.getFIFOStall(State.PendingDS, Model.MaxDSInFlight);
    case InstClass::VMEM_READ:
    case InstClass::VMEM_WRITE:
      return State.getVMEMBufferStall(Model.MaxVMEMInFlight);
    case InstClass::TDM:
      return State.getFIFOStall(State.PendingTDM, Model.MaxTDMInFlight);
    default:
      return 0;
    }
  }

  // Compute delays before advancing CurrentCycle or recording resource
  // effects. EffectiveStall composes sources whose delays can overlap.
  StallSources computeStallSources(const SimInst &Inst,
                                   ArrayRef<WaitRequirement> Waits) {
    StallSources Sources;
    StallBreakdown &Breakdown = Sources.Breakdown;
    InstClass IC = Inst.Class;
    unsigned IssueCycle = State.CurrentCycle;
    bool IsLOLVALU = InstInfo.isLOLVALU(Inst);

    Breakdown.DelayAlu = checkPendingDelayAlu();
    applyStall(IssueCycle, State.CurrentCycle + Breakdown.DelayAlu);

    if (IC != InstClass::WMMA) {
      unsigned BusyUntil = State.getUnitBusyUntil(Inst.Unit);
      if (BusyUntil > IssueCycle) {
        Breakdown.FU = BusyUntil - State.CurrentCycle;
        IssueCycle = BusyUntil;
      }
    }

    unsigned VALUResourceStall = computeVALUResourceStall(IC, IssueCycle);
    if (VALUResourceStall > 0) {
      Breakdown.FU = std::max(Breakdown.FU, VALUResourceStall);
      IssueCycle = State.VALUResourceBusyUntil;
    }

    if ((IC == InstClass::TRANS || IsLOLVALU) &&
        State.LOLVALUTRANSHazardUntil > IssueCycle) {
      Breakdown.LOLVALUTRANSHazard = State.LOLVALUTRANSHazardUntil - IssueCycle;
      IssueCycle = State.LOLVALUTRANSHazardUntil;
    }

    if (IC == InstClass::WMMA) {
      WMMAProperties Properties = InstInfo.getWMMAProperties(Inst);
      if (Properties.HasScaling) {
        applyStall(IssueCycle, State.CurrentCycle + State.getWMMATRANSStall());
        computeScaledWMMAStall(IssueCycle, Sources);
      } else {
        computeUnscaledWMMAStall(IssueCycle, Sources);
      }
    }

    if (IC == InstClass::SALU && State.VaSSRCBusyUntil > IssueCycle) {
      Breakdown.SSRC = State.VaSSRCBusyUntil - IssueCycle;
      IssueCycle = State.VaSSRCBusyUntil;
    }

    for (const WaitRequirement &Wait : Waits) {
      if (Wait.Type != WaitType::DepCtr)
        continue;
      unsigned Target = InstInfo.getVaVdstTarget(Inst);
      if (Target < 15) {
        unsigned ReadyCycle = State.getVaVdstReadyCycle(Target);
        if (ReadyCycle > State.CurrentCycle) {
          Breakdown.VaVdst = ReadyCycle - State.CurrentCycle;
          applyStall(IssueCycle, ReadyCycle);
        }
      }
    }

    if (State.inWMMAWindow() && IC != InstClass::WMMA) {
      // Long latency VALU waits for the whole window instead of searching for
      // the next compatible slot.
      if (IsLOLVALU && State.ActiveWMMA.EndCycle > IssueCycle) {
        Breakdown.LongLatVALU = State.ActiveWMMA.EndCycle - IssueCycle;
        IssueCycle = State.ActiveWMMA.EndCycle;
      } else {
        unsigned CoExecStall = State.getCoExecStallAt(IC, IssueCycle);
        if (CoExecStall > 0) {
          Breakdown.CoExec = CoExecStall;
          Breakdown.CoExecFromEffective = CoExecStall;
          IssueCycle += CoExecStall;
        }
      }
    }

    if (IC == InstClass::DELAY_ALU) {
      unsigned Stall = parseDelayAlu(Inst);
      Breakdown.DelayAlu = std::max(Breakdown.DelayAlu, Stall);
      applyStall(IssueCycle, State.CurrentCycle + Stall);
    }

    if (IC == InstClass::WAITCNT) {
      Breakdown.WaitCnt = computeWaitStall(Waits);
      applyStall(IssueCycle, State.CurrentCycle + Breakdown.WaitCnt);
    }

    Breakdown.MemFIFO = computeMemFIFOStall(IC);
    applyStall(IssueCycle, State.CurrentCycle + Breakdown.MemFIFO);
    Breakdown.EffectiveStall = IssueCycle - State.CurrentCycle;
    return Sources;
  }

  void recordInstruction(const SimInst &Inst, unsigned WMMAStartCycle) {
    InstClass IC = Inst.Class;
    switch (IC) {
    case InstClass::VALU:
      State.trackVALU(Inst.Latency);
      State.trackVALUForWMMA(IC);
      if (Log)
        *Log << "  -> LastVALUCycle = " << State.LastVALUCycle << "\n";
      State.trackVaVdst(Inst.Latency, Model.VaVdstMultiplier);
      if (InstInfo.hasSGPROperands(Inst))
        State.VaSSRCBusyUntil =
            std::max(State.VaSSRCBusyUntil, State.CurrentCycle + Inst.Latency);
      if (InstInfo.isLOLVALU(Inst) && State.inWMMAWindow())
        State.holdVALUResourceInWindow(InstInfo.getRepeatRate(Inst));
      if (InstInfo.isLOLVALU(Inst))
        State.LOLVALUTRANSHazardUntil =
            std::max(State.LOLVALUTRANSHazardUntil, State.CurrentCycle + 2);
      break;
    case InstClass::SALU:
      State.LastSALUCycle = State.CurrentCycle;
      break;
    case InstClass::TRANS:
      State.trackTRANS(Inst.Latency);
      State.trackVALUForWMMA(IC);
      State.trackVaVdst(Inst.Latency, Model.VaVdstMultiplier);
      State.holdVALUResourceInWindow(InstInfo.getResourceCycles(Inst));
      State.LOLVALUTRANSHazardUntil =
          std::max(State.LOLVALUTRANSHazardUntil, State.CurrentCycle + 2);
      break;
    case InstClass::WMMA: {
      State.trackTRANS(Inst.Latency);
      WMMAProperties Properties = InstInfo.getWMMAProperties(Inst);
      unsigned EffectiveStart =
          Properties.HasScaling ? WMMAStartCycle : State.CurrentCycle;
      unsigned Occupancy = State.startWMMAWindow(Properties, EffectiveStart);
      if (Properties.HasScaling)
        State.reserveScaledWMMAScaleRead(State.CurrentCycle);
      if (InstInfo.hasSGPROperands(Inst))
        State.VaSSRCBusyUntil =
            std::max(State.VaSSRCBusyUntil, EffectiveStart + Occupancy);
      State.PendingVaVdst.push_back(
          {EffectiveStart + Occupancy * Model.VaVdstMultiplier});
      break;
    }
    case InstClass::DS_READ:
      State.issueDS(Inst.Latency);
      break;
    case InstClass::DS_WRITE:
      State.issueDS(Inst.Latency);
      break;
    case InstClass::VMEM_READ:
      State.issueVMEM(Inst.Latency, true);
      break;
    case InstClass::VMEM_WRITE:
      State.issueVMEM(Inst.Latency, false);
      // Store acknowledgment is tracked separately for S_WAIT_XCNT.
      State.issueXACK(GFX1250::XACKLatency);
      break;
    case InstClass::SMEM:
      State.issueSMEM(Inst.Latency);
      break;
    case InstClass::TDM:
      State.issueTDM(Inst.Latency);
      break;
    default:
      break;
    }

    if (IC != InstClass::WMMA)
      State.setUnitBusyUntil(Inst.Unit, State.CurrentCycle +
                                            InstInfo.getResourceCycles(Inst));
  }

  void populateInfo(InstrSimInfo &Info, const StallSources &Sources,
                    InstClass IC) const {
    Info.StallCycles = Sources.Breakdown.total();
    Info.Reason = getDominantStallReason(Sources.Breakdown);
    Info.Breakdown = Sources.Breakdown;

    if (IC == InstClass::DELAY_ALU)
      Info.WasFused = true;
    // Snapshot the active stage before stall cycles advance CurrentCycle.
    if (State.inWMMAWindow() && IC != InstClass::WMMA) {
      Info.InWMMAWindow = true;
      Info.WMMATotalWindow = State.ActiveWMMA.Info.TotalWindow;
      if (std::optional<unsigned> Stage = State.getWMMAStage()) {
        Info.WMMAStage = *Stage;
        Info.StageType = State.ActiveWMMA.Info.getType(*Stage);
      }
      Info.CoExecuted =
          Sources.Breakdown.CoExec == 0 && Sources.Breakdown.LongLatVALU == 0;
    }
  }

public:
  Impl(const SimInstInfo &II, const HWModel &M, SimulatorConfig C)
      : InstInfo(II), Model(M), Config(C), Log(C.Verbose ? C.Log : nullptr) {
    State.reset();
  }

  InstrSimInfo simulateInst(const SimInst &Inst, ArrayRef<SimInst> Lookahead) {
    InstrSimInfo Info;
    if (Inst.Class == InstClass::MSB_SET) {
      // Fused MSB sets consume no cycle. Exposed sets consume one unless the
      // successor already has an unavoidable coexecution delay.
      checkPendingDelayAlu(true);
      if (canMSBSetFuse(State.PreviousInstClass)) {
        Info.WasFused = true;
      } else {
        Info.WasExposed = true;
        if (State.inWMMAWindow() && !Lookahead.empty() &&
            State.getCoExecStall(Lookahead.front().Class) > 0) {
          Info.WasMasked = true;
        } else {
          Info.StallCycles = 1;
          Info.Reason = StallReason::MSB_SET_EXPOSED;
          State.advanceCycle();
        }
      }
      // Treat the MSB set as the previous SALU class instruction for
      // subsequent S_SET_VGPR_MSB fusion.
      State.PreviousInstClass = InstClass::SALU;
      return Info;
    }

    WaitRequirements Waits = InstInfo.getWaitInfo(Inst);
    StallSources Sources = computeStallSources(Inst, Waits);
    populateInfo(Info, Sources, Inst.Class);
    if (Log)
      *Log << "AMDGPU static simulator instruction " << Inst.InstIndex
           << " stall " << Info.StallCycles << "\n";

    // Apply preissue delay, update wait state at issue, record instruction
    // side effects, then consume one issue cycle.
    State.advanceCycle(Sources.Breakdown.total());
    if (Inst.Class == InstClass::WAITCNT)
      applyWait(Waits);

    recordInstruction(Inst, Sources.WMMAStartCycle);
    State.advanceCycle();
    State.PreviousInstClass = Inst.Class;

    if (Inst.Class == InstClass::WMMA) {
      Info.IsWMMA = true;
      Info.WMMAPattern = State.ActiveWMMA.Info.Pattern;
    }
    return Info;
  }

  void advanceCycles(unsigned Count) { State.advanceCycle(Count); }

  const GPUSimState &getState() const { return State; }
  const SimulatorConfig &getConfig() const { return Config; }
  const HWModel &getModel() const { return Model; }
};

Simulator::Simulator(const SimInstInfo &II, const HWModel &Model,
                     SimulatorConfig Config)
    : PImpl(std::make_unique<Impl>(II, Model, Config)) {}

Simulator::~Simulator() = default;

InstrSimInfo Simulator::simulateInst(const SimInst &Inst,
                                     ArrayRef<SimInst> Lookahead) {
  return PImpl->simulateInst(Inst, Lookahead);
}

void Simulator::advanceCycles(unsigned Count) { PImpl->advanceCycles(Count); }

const GPUSimState &Simulator::getState() const { return PImpl->getState(); }

const SimulatorConfig &Simulator::getConfig() const {
  return PImpl->getConfig();
}

const HWModel &Simulator::getModel() const { return PImpl->getModel(); }

} // namespace AMDGPUSim
} // namespace llvm
