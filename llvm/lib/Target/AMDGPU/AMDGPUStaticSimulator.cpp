//===- AMDGPUStaticSimulator.cpp - Static performance simulator ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Runs AMDGPUSim over late MachineInstr streams and reports static metrics.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUStaticSimulator.h"
#include "AMDGPU.h"
#include "AMDGPUSim/AMDGPUSim.h"
#include "AMDGPUSim/MIRAdapter.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <iterator>
#include <memory>

using namespace llvm;
using namespace llvm::AMDGPU;
using namespace llvm::AMDGPUSim;

#define DEBUG_TYPE "amdgpu-static-simulator"

static cl::opt<bool> VerboseSimulation(
    "amdgpu-static-sim-verbose",
    cl::desc("Enable verbose per-instruction logging in static simulator"),
    cl::init(false), cl::Hidden);

namespace {

// Verbose output is available only when LLVM debugging is active for this
// DEBUG_TYPE.
static bool isVerboseLoggingEnabled() {
  bool Enabled = false;
  if (VerboseSimulation)
    LLVM_DEBUG(Enabled = true);
  return Enabled;
}

static const MachineInstr *findNextSimulatedInstr(const MachineInstr &MI) {
  const MachineBasicBlock &MBB = *MI.getParent();
  for (auto I = std::next(MI.getIterator()), E = MBB.instr_end(); I != E; ++I)
    if (!I->isBundle() && !I->isMetaInstruction())
      return &*I;
  return nullptr;
}

static void attributeCoExecStall(unsigned StallCycles, InstClass IC,
                                 StaticSimulatorBlockMetrics &Metrics) {
  Metrics.StallCoExec += StallCycles;
  switch (IC) {
  case InstClass::VALU:
    Metrics.CoExecMissVALU += StallCycles;
    break;
  case InstClass::TRANS:
    Metrics.CoExecMissTRANS += StallCycles;
    break;
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
  case InstClass::SMEM:
  case InstClass::TDM:
    Metrics.CoExecMissMemory += StallCycles;
    break;
  default:
    Metrics.CoExecMissOther += StallCycles;
    break;
  }
}

static void attributeStall(const InstrSimInfo &Info, InstClass IC,
                           bool InWMMAWindow,
                           StaticSimulatorBlockMetrics &Metrics) {
  // InWMMAWindow is sampled after simulation and drives MSB opportunity
  // attribution after the instruction cycle has advanced.
  if (Info.WasExposed) {
    if (Info.WasMasked)
      ++Metrics.NumMSBSetMasked;
    else {
      ++Metrics.NumMSBSetExposed;
      ++Metrics.TotalStallCycles;
    }
  }

  unsigned StallCycles = Info.StallCycles;
  if (StallCycles == 0)
    return;

  if (Info.Reason == StallReason::MSB_SET_EXPOSED) {
    if (InWMMAWindow) {
      // Preserve additive reporting by counting the exposed cycle again as a
      // coexecution miss.
      Metrics.TotalStallCycles += StallCycles;
      attributeCoExecStall(StallCycles, IC, Metrics);
    }
    return;
  }

  // Individual stall sources remain available in StallBreakdown for
  // diagnostics.
  // Block totals other than MSB assign the full effective stall to one
  // dominant reason so overlapping or cumulative sources cannot be omitted or
  // double counted.
  Metrics.TotalStallCycles += StallCycles;
  switch (Info.Reason) {
  case StallReason::NONE:
    Metrics.StallOther += StallCycles;
    break;
  case StallReason::FU_BUSY:
    Metrics.StallFunctionalUnit += StallCycles;
    break;
  case StallReason::COEXEC_BLOCKED:
    attributeCoExecStall(StallCycles, IC, Metrics);
    break;
  case StallReason::LONG_LAT_VALU:
    Metrics.StallLongLatVALU += StallCycles;
    break;
  case StallReason::LOLVALU_TRANS_HAZARD:
    Metrics.StallLOLVALUTRANS += StallCycles;
    break;
  case StallReason::VA_SSRC_STALL:
    Metrics.StallVaSSRC += StallCycles;
    break;
  case StallReason::VA_VDST_WAIT:
    Metrics.StallVaVdst += StallCycles;
    break;
  case StallReason::WAITCNT:
    Metrics.StallWaitCnt += StallCycles;
    break;
  case StallReason::DELAY_ALU:
    Metrics.StallDelayAlu += StallCycles;
    break;
  case StallReason::MEM_FIFO:
    Metrics.StallMemFIFO += StallCycles;
    break;
  case StallReason::MSB_SET_EXPOSED:
    llvm_unreachable("MSB_SET handled above");
  }
}

// Count \p MI in the instruction categories used by the summary report.
static void countInstruction(const MachineInstr &MI, const SimInst &SI,
                             StaticSimulatorBlockMetrics &Metrics) {
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case AMDGPU::S_NOP:
  case AMDGPU::V_NOP_e32:
  case AMDGPU::V_NOP_e64:
  case AMDGPU::V_NOP_sdwa:
  case AMDGPU::V_NOP_dpp:
  case AMDGPU::V_NOP_dpp8:
    ++Metrics.NumNop;
    return;
  default:
    break;
  }

  switch (SI.Class) {
  case InstClass::VALU:
    ++Metrics.NumVALU;
    if (AMDGPU::isVOPD(MI.getOpcode())) {
      ++Metrics.NumVOPD;
      ++Metrics.NumVALU;
    } else if (SIInstrInfo::isPacked(MI)) {
      ++Metrics.NumPacked;
      ++Metrics.NumVALU;
    }
    break;
  case InstClass::SALU:
    ++Metrics.NumSALU;
    break;
  case InstClass::TRANS:
    ++Metrics.NumTRANS;
    break;
  case InstClass::WMMA:
    ++Metrics.NumWMMA;
    break;
  case InstClass::DS_READ:
    ++Metrics.NumDSRead;
    break;
  case InstClass::DS_WRITE:
    ++Metrics.NumDSWrite;
    break;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    ++Metrics.NumVMEM;
    break;
  case InstClass::SMEM:
    ++Metrics.NumSMEM;
    break;
  case InstClass::TDM:
    ++Metrics.NumTDM;
    break;
  case InstClass::BARRIER:
  case InstClass::BARRIER_SIGNAL:
  case InstClass::BARRIER_WAIT:
    ++Metrics.NumBarrier;
    break;
  case InstClass::WAITCNT:
    ++Metrics.NumWaitcnt;
    break;
  case InstClass::DELAY_ALU:
    ++Metrics.NumDelayAlu;
    break;
  case InstClass::MSB_SET:
    ++Metrics.NumMSBSet;
    break;
  case InstClass::NOP:
    ++Metrics.NumNop;
    break;
  case InstClass::BRANCH:
    ++Metrics.NumBranch;
    break;
  case InstClass::OTHER:
    break;
  }

  if (Opc == AMDGPU::V_WRITELANE_B32)
    ++Metrics.NumSGPRToVGPR;
  else if (Opc == AMDGPU::V_READLANE_B32)
    ++Metrics.NumVGPRToSGPR;

  if (SIInstrInfo::isSpill(MI) || SIInstrInfo::isFLATScratch(MI)) {
    if (MI.mayStore())
      ++Metrics.NumSpill;
    if (MI.mayLoad())
      ++Metrics.NumReload;
  }
}

// Count coexecution and internal slot use from the retained WMMA snapshot.
static void trackWMMACoExec(const InstrSimInfo &Info, InstClass IC,
                            const GPUSimState &State,
                            StaticSimulatorBlockMetrics &Metrics) {
  if (Info.IsWMMA) {
    Metrics.WMMAOccupancyCycles += State.ActiveWMMA.Info.UnitOccupancy;
    return;
  }
  if (!Info.InWMMAWindow)
    return;

  if (Info.CoExecuted)
    ++Metrics.WMMACoExecUsed;
  else
    ++Metrics.WMMACoExecBlocked;

  bool IsISlot = Info.StageType == AMDGPU::CoExecStageType::I ||
                 Info.StageType == AMDGPU::CoExecStageType::IS;
  if (!Info.CoExecuted || !IsISlot)
    return;

  ++Metrics.ISlotTotal;
  if (IC == InstClass::VALU || IC == InstClass::TRANS)
    ++Metrics.ISlotUsedByVALU;
  else
    ++Metrics.ISlotWastedOnNonVALU;
}

// Print the MIR instruction and its modeled timing properties.
static void logInstruction(unsigned Cycle, const MachineInstr &MI,
                           const SimInst &SI, MachineInstrInfo &MII) {
  dbgs() << "\n[Cycle " << Cycle << "] ";
  MI.print(dbgs(), true, false, true, false);
  dbgs() << "\n";

  if (SI.Class == InstClass::WMMA) {
    AMDGPU::CoExecInfo Info =
        AMDGPU::getKnownCoExecInfo(MII.getWMMAProperties(SI))
            .value_or(AMDGPU::CoExecInfo());
    dbgs() << "  Class: WMMA | Unit: XDL | Occupancy: " << Info.UnitOccupancy
           << " | Window: " << Info.TotalWindow << "\n";
    return;
  }

  dbgs() << "  Class: " << getInstClassName(SI.Class)
         << " | Unit: " << getUnitName(SI.Unit) << " | Latency: " << SI.Latency
         << " | ResourceCycles: " << MII.getResourceCycles(SI)
         << " | Size: " << MII.getInstBytes(SI) << " bytes\n";
}

// Print the nonzero stall sources and their effective total.
static void logStalls(const StallBreakdown &B) {
  dbgs() << "  Stalls: ";
  if (B.total() == 0) {
    dbgs() << "(none)";
  } else {
    bool First = true;
    auto Print = [&](const char *Name, unsigned Value) {
      if (Value == 0)
        return;
      if (!First)
        dbgs() << ", ";
      dbgs() << Name << "=" << Value;
      First = false;
    };
    Print("FU", B.FU);
    Print("VALUSlot", B.VALUSlot);
    Print("WMMACoExecMiss", B.CoExecFromEffective);
    Print("LongLatVALU", B.LongLatVALU);
    Print("LOLVALUxTRANS", B.LOLVALUTRANSHazard);
    Print("SSRC", B.SSRC);
    Print("VaVdst", B.VaVdst);
    Print("DelayALU", B.DelayAlu);
    Print("WaitCnt", B.WaitCnt);
    Print("MemFIFO", B.MemFIFO);
    if (First)
      dbgs() << "Other=" << B.total();
  }
  dbgs() << " -> Total: " << B.total();
  dbgs() << "\n";
}

// Print the issue result and resulting simulator cycle for \p SI.
static void logSimulationResult(unsigned EntryCycle, const SimInst &SI,
                                const InstrSimInfo &Info,
                                const GPUSimState &State,
                                bool WasInWMMAWindow) {
  if (SI.Class == InstClass::MSB_SET) {
    dbgs() << "  -> MSB_SET ";
    if (Info.WasFused)
      dbgs() << "fused with prev (free)";
    else if (Info.WasMasked)
      dbgs() << "exposed but MASKED (next instr stalls anyway)";
    else {
      dbgs() << "EXPOSED (+1 cycle)";
      if (State.inWMMAWindow())
        dbgs() << " [in WMMA window]";
    }
    dbgs() << "\n";
    return;
  }

  if (Info.IsWMMA && !Info.WMMAPattern.empty())
    dbgs() << "  -> WMMA pattern: " << Info.WMMAPattern << "\n";

  if (Info.InWMMAWindow) {
    dbgs() << "  -> WMMA[" << unsigned(Info.WMMAStage) << "/"
           << unsigned(Info.WMMATotalWindow) << "] "
           << AMDGPU::getStageTypeName(Info.StageType);
    if (Info.CoExecuted)
      dbgs() << " OK";
    else
      dbgs() << " BLOCKED";
    dbgs() << "\n";
  }

  logStalls(Info.Breakdown);
  if (Info.StallCycles > 0)
    dbgs() << "  -> Advancing cycle: " << EntryCycle << " -> "
           << EntryCycle + Info.StallCycles << "\n";

  if (SI.Class == InstClass::WMMA) {
    dbgs() << "  -> ActiveWMMA: cycles " << State.ActiveWMMA.StartCycle << "-"
           << State.ActiveWMMA.EndCycle;
    if (WasInWMMAWindow)
      dbgs() << " [back-to-back]";
    dbgs() << "\n";
  }
  dbgs() << "  -> NextCycle: " << State.CurrentCycle << "\n";
}

static StaticSimulatorBlockMetrics analyzeBlock(MachineBasicBlock &MBB,
                                                Simulator &Sim,
                                                MachineInstrInfo &MII,
                                                bool Verbose) {
  if (Verbose)
    dbgs() << "\n=== BB#" << MBB.getNumber() << " [Cycle "
           << Sim.getState().CurrentCycle << "] ===\n";

  StaticSimulatorBlockMetrics Metrics;
  unsigned StartCycle = Sim.getState().CurrentCycle;

  for (MachineInstr &MI : MBB.instrs()) {
    if (MI.isBundle() || MI.isMetaInstruction())
      continue;

    SimInst SI = MII.createSimInst(MI);
    SI.InstIndex = Metrics.NumInstructions;
    unsigned EntryCycle = Sim.getState().CurrentCycle;
    bool WasInWMMAWindow = Sim.getState().inWMMAWindow();

    if (Verbose)
      logInstruction(EntryCycle, MI, SI, MII);

    SmallVector<SimInst, 1> Lookahead;
    // The next retained instruction in this block determines whether an
    // exposed MSB set is hidden by an unavoidable coexecution stall.
    if (SI.Class == InstClass::MSB_SET)
      if (const MachineInstr *Next = findNextSimulatedInstr(MI))
        Lookahead.push_back(MII.createSimInst(*Next));

    InstrSimInfo Info = Sim.simulateInst(SI, Lookahead);
    ++Metrics.NumInstructions;
    Metrics.NumBytes += MII.getInstBytes(SI);
    attributeStall(Info, SI.Class, Sim.getState().inWMMAWindow(), Metrics);
    countInstruction(MI, SI, Metrics);
    trackWMMACoExec(Info, SI.Class, Sim.getState(), Metrics);

    if (Verbose)
      logSimulationResult(EntryCycle, SI, Info, Sim.getState(),
                          WasInWMMAWindow);
  }

  Metrics.TotalCycles = Sim.getState().CurrentCycle - StartCycle;
  if (Verbose)
    dbgs() << "=== End BB#" << MBB.getNumber() << ": "
           << Metrics.NumInstructions << " insts, " << Metrics.TotalCycles
           << " cycles, " << Metrics.TotalStallCycles << " stalls ===\n";
  return Metrics;
}

// Traverse blocks in layout order with one simulator state. CFG path selection
// and frequency weighting are not modeled.
static StaticSimulatorReport
analyzeFunction(MachineFunction &MF, const SIInstrInfo &TII, bool Verbose) {
  StaticSimulatorReport Report;
  MachineInstrInfo MII(TII, TII.getRegisterInfo());
  std::unique_ptr<HWModel> Model = createHWModel(GPUTarget::GFX1250);

  SimulatorConfig Config;
  Config.Verbose = Verbose;
  Config.Log = Verbose ? &dbgs() : nullptr;
  Simulator Sim(MII, *Model, Config);

  for (MachineBasicBlock &MBB : MF) {
    StaticSimulatorBlockMetrics Metrics = analyzeBlock(MBB, Sim, MII, Verbose);
    Report.PerBlock[&MBB] = Metrics;
    Report.Total.add(Metrics);
  }
  return Report;
}

static void runStaticSimulator(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX1250Insts())
    return;

  const SIInstrInfo *TII = ST.getInstrInfo();

  if (!AMDGPU::isExpertSchedulingMode(ST, MF.getFunction())) {
    Function &F = MF.getFunction();
    F.getContext().diagnose(DiagnosticInfoUnsupported(
        F,
        "AMDGPU static simulator does not model non-expert scheduling mode, "
        "performance estimates may be overly optimistic",
        DiagnosticLocation(), DS_Warning));
  }

  LLVM_DEBUG(dbgs() << "Running Static Simulator on: " << MF.getName() << "\n");
  bool Verbose = isVerboseLoggingEnabled();
  if (Verbose)
    dbgs() << "\n=== Function: " << MF.getName() << " ===\n";

  StaticSimulatorReport Report = analyzeFunction(MF, *TII, Verbose);
  Report.print(dbgs(), MF);
}

class AMDGPUStaticSimulatorLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUStaticSimulatorLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUStaticSimulatorLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    runStaticSimulator(MF);
    return false;
  }

  StringRef getPassName() const override {
    return "AMDGPU Static Performance Simulator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

void StaticSimulatorBlockMetrics::add(
    const StaticSimulatorBlockMetrics &Other) {
#define ADD(Field) Field += Other.Field
  ADD(NumInstructions);
  ADD(NumVALU);
  ADD(NumSALU);
  ADD(NumTRANS);
  ADD(NumWMMA);
  ADD(NumVOPD);
  ADD(NumPacked);
  ADD(NumDSRead);
  ADD(NumDSWrite);
  ADD(NumVMEM);
  ADD(NumSMEM);
  ADD(NumTDM);
  ADD(NumBranch);
  ADD(NumBarrier);
  ADD(NumNop);
  ADD(NumDelayAlu);
  ADD(NumMSBSet);
  ADD(NumSpill);
  ADD(NumReload);
  ADD(NumSGPRToVGPR);
  ADD(NumVGPRToSGPR);
  ADD(NumWaitcnt);
  ADD(NumBytes);
  ADD(TotalCycles);
  ADD(TotalStallCycles);
  ADD(StallFunctionalUnit);
  ADD(StallCoExec);
  ADD(StallDelayAlu);
  ADD(StallMemFIFO);
  ADD(StallWaitCnt);
  ADD(StallLongLatVALU);
  ADD(StallLOLVALUTRANS);
  ADD(StallVaSSRC);
  ADD(StallVaVdst);
  ADD(StallOther);
  ADD(NumMSBSetExposed);
  ADD(NumMSBSetMasked);
  ADD(CoExecMissVALU);
  ADD(CoExecMissTRANS);
  ADD(CoExecMissMemory);
  ADD(CoExecMissOther);
  ADD(WMMAOccupancyCycles);
  ADD(WMMACoExecUsed);
  ADD(WMMACoExecBlocked);
  ADD(ISlotTotal);
  ADD(ISlotUsedByVALU);
  ADD(ISlotWastedOnNonVALU);
#undef ADD
}

void StaticSimulatorBlockMetrics::printStallBreakdown(raw_ostream &OS) const {
  bool First = true;
  auto Print = [&](const char *Name, unsigned Value) {
    if (Value == 0)
      return;
    if (!First)
      OS << " | ";
    OS << Name << ":" << Value;
    First = false;
  };

  Print("FU", StallFunctionalUnit);
  if (StallCoExec) {
    if (!First)
      OS << " | ";
    OS << "WMMACoExec:" << StallCoExec;
    if (CoExecMissVALU || CoExecMissTRANS || CoExecMissMemory ||
        CoExecMissOther) {
      OS << "(";
      bool SubFirst = true;
      auto PrintSub = [&](const char *Name, unsigned Value) {
        if (Value == 0)
          return;
        if (!SubFirst)
          OS << "+";
        OS << Name << ":" << Value;
        SubFirst = false;
      };
      PrintSub("VALU", CoExecMissVALU);
      PrintSub("TRANS", CoExecMissTRANS);
      PrintSub("MEM", CoExecMissMemory);
      PrintSub("Other", CoExecMissOther);
      OS << ")";
    }
    First = false;
  }
  Print("DelayAlu", StallDelayAlu);
  Print("MemFIFO", StallMemFIFO);
  Print("Wait", StallWaitCnt);
  Print("LongLatVALU", StallLongLatVALU);
  Print("LOLVALUxTRANS", StallLOLVALUTRANS);
  Print("VaSSRC", StallVaSSRC);
  Print("VaVdst", StallVaVdst);
  Print("Other", StallOther);
  if (NumMSBSetExposed || NumMSBSetMasked) {
    if (!First)
      OS << " | ";
    OS << "MSBExposed:" << NumMSBSetExposed;
    if (NumMSBSetMasked)
      OS << " (+" << NumMSBSetMasked << " masked)";
    First = false;
  }
  if (First)
    OS << "(none)";
}

// Print instruction category counters from \p Metrics using \p Indent.
static void
printInstructionBreakdown(raw_ostream &OS,
                          const StaticSimulatorBlockMetrics &Metrics,
                          unsigned Indent) {
  auto PrintIndent = [&]() {
    OS << ";";
    OS.indent(Indent);
  };

  unsigned NumVALUInsts = Metrics.NumVALU - Metrics.NumVOPD - Metrics.NumPacked;
  PrintIndent();
  OS << "VALU: " << NumVALUInsts;
  if (Metrics.NumVOPD || Metrics.NumPacked) {
    OS << " (";
    if (Metrics.NumVOPD)
      OS << "VOPD:" << Metrics.NumVOPD;
    if (Metrics.NumVOPD && Metrics.NumPacked)
      OS << "+";
    if (Metrics.NumPacked)
      OS << "PK:" << Metrics.NumPacked;
    OS << ")";
  }
  OS << " | SALU: " << Metrics.NumSALU << " | TRANS: " << Metrics.NumTRANS
     << " | WMMA: " << Metrics.NumWMMA << "\n";
  PrintIndent();
  OS << "DS_RD: " << Metrics.NumDSRead << " | DS_WR: " << Metrics.NumDSWrite
     << " | VMEM: " << Metrics.NumVMEM << " | SMEM: " << Metrics.NumSMEM
     << " | TDM: " << Metrics.NumTDM << "\n";
  PrintIndent();
  OS << "Branch: " << Metrics.NumBranch << " | Barrier: " << Metrics.NumBarrier
     << " | Waitcnt: " << Metrics.NumWaitcnt << " | NOP: " << Metrics.NumNop
     << "\n";
  PrintIndent();
  OS << "delay_alu: " << Metrics.NumDelayAlu
     << " | MSB_set: " << Metrics.NumMSBSet
     << " (exposed: " << Metrics.NumMSBSetExposed
     << ", masked: " << Metrics.NumMSBSetMasked << ")\n";
  if (Metrics.NumSpill || Metrics.NumReload) {
    PrintIndent();
    OS << "Spill: " << Metrics.NumSpill << " | Reload: " << Metrics.NumReload
       << "\n";
  }
  if (Metrics.NumSGPRToVGPR || Metrics.NumVGPRToSGPR) {
    PrintIndent();
    OS << "SGPR->Lane: " << Metrics.NumSGPRToVGPR
       << " | Lane->SGPR: " << Metrics.NumVGPRToSGPR << "\n";
  }
}

void StaticSimulatorReport::print(raw_ostream &OS,
                                  const MachineFunction &MF) const {
  OS << "; ============================================================\n";
  OS << "; " << MF.getName() << " - STATIC PERFORMANCE ESTIMATE (gfx1250)\n";
  OS << "; ============================================================\n";
  OS << ";\n";
  OS << "; === Raw Metrics (single layout-order traversal) ===\n";
  OS << ";   Instructions: " << Total.NumInstructions << "\n";
  OS << ";   Bytes:        " << Total.NumBytes << "\n";
  OS << ";   Cycles:       " << Total.TotalCycles << "\n";
  OS << ";   Stall:        " << Total.TotalStallCycles << "\n";
  OS << ";   ";
  Total.printStallBreakdown(OS);
  OS << "\n";
  OS << ";   Waitcnts:     " << Total.NumWaitcnt << "\n";
  if (Total.NumWMMA) {
    float CoExecEfficiency =
        Total.WMMAOccupancyCycles
            ? 100.0f * Total.WMMACoExecUsed / Total.WMMAOccupancyCycles
            : 0.0f;
    OS << formatv(";   WMMA occupancy cycles: {0} | Co-executed: {1} ({2:F0}%)"
                  " | Blocked: {3}\n",
                  Total.WMMAOccupancyCycles, Total.WMMACoExecUsed,
                  CoExecEfficiency, Total.WMMACoExecBlocked);
  }
  if (Total.ISlotTotal)
    OS << ";   I-slots: " << Total.ISlotUsedByVALU << " used by VALU/TRANS | "
       << Total.ISlotWastedOnNonVALU << " used by other instructions\n";
  OS << ";\n";

  OS << "; === Instruction Breakdown ===\n";
  printInstructionBreakdown(OS, Total, 3);
  OS << ";\n";

  unsigned ComputeOps =
      Total.NumVALU + Total.NumSALU + Total.NumTRANS + Total.NumWMMA;
  float ComputeIPC = Total.TotalCycles
                         ? static_cast<float>(ComputeOps) / Total.TotalCycles
                         : 0.0f;
  float StallRatio = Total.TotalCycles
                         ? 100.0f * Total.TotalStallCycles / Total.TotalCycles
                         : 0.0f;
  OS << "; === Derived Metrics ===\n";
  OS << formatv(";   Compute IPC: {0:F2} | Stall ratio: {1:F1}%\n", ComputeIPC,
                StallRatio);
  OS << ";\n";
  OS << "; ============================================================\n";
  OS << ";\n";

  OS << "; === Block Metrics (layout order) ===\n";
  for (const MachineBasicBlock &MBB : MF) {
    auto It = PerBlock.find(&MBB);
    if (It == PerBlock.end())
      continue;
    const StaticSimulatorBlockMetrics &Metrics = It->second;
    OS << ";   bb." << MBB.getNumber() << ": " << Metrics.TotalCycles
       << " cycles\n";
    printInstructionBreakdown(OS, Metrics, 5);
    if (Metrics.TotalStallCycles > 0) {
      float StallPct = 100.0f * Metrics.TotalStallCycles / Metrics.TotalCycles;
      OS << formatv(";     Stall: {0} cycles ({1:F0}%)\n",
                    Metrics.TotalStallCycles, StallPct);
      OS << ";       ";
      Metrics.printStallBreakdown(OS);
      OS << "\n";
    }
  }
  OS << ";\n";
}

char AMDGPUStaticSimulatorLegacy::ID = 0;
char &llvm::AMDGPUStaticSimulatorLegacyID = AMDGPUStaticSimulatorLegacy::ID;

INITIALIZE_PASS(AMDGPUStaticSimulatorLegacy, DEBUG_TYPE,
                "AMDGPU Static Performance Simulator", false, false)

FunctionPass *llvm::createAMDGPUStaticSimulatorPass() {
  return new AMDGPUStaticSimulatorLegacy();
}
