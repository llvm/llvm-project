//===-- GCNHazardRecognizers.h - GCN Hazard Recognizers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling on GCN processors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPUHAZARDRECOGNIZERS_H
#define LLVM_LIB_TARGET_AMDGPUHAZARDRECOGNIZERS_H

#include "AMDGPUCoExecInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include <array>
#include <list>
#include <optional>

namespace llvm {

class MachineFunction;
class MachineInstr;
class MachineOperand;
class MachineRegisterInfo;
class SIInstrInfo;
class SIRegisterInfo;
class GCNSubtarget;

class GCNHazardRecognizer final : public ScheduleHazardRecognizer {
public:
  typedef function_ref<bool(const MachineInstr &)> IsHazardFn;
  typedef function_ref<bool(const MachineInstr &, int WaitStates)> IsExpiredFn;
  typedef function_ref<unsigned int(const MachineInstr &)> GetNumWaitStatesFn;

  /// Operating mode for the hazard recognizer.
  /// - PreRA: Used during pre-RA scheduling (virtual regs, co-exec slot
  /// tracking)
  /// - PostRA: Used during post-RA scheduling (physical regs, full hazard
  /// checking)
  /// - HazardRecognizerMode: Used by standalone hazard recognizer pass (inserts
  /// NOPs)
  enum class OperatingMode { PreRA, PostRA, HazardRecognizerMode };

private:
  // Operating mode determines which hazards are checked.
  OperatingMode Mode;

  // This variable stores the instruction that has been emitted this cycle. It
  // will be added to EmittedInstrs, when AdvanceCycle() or RecedeCycle() is
  // called.
  MachineInstr *CurrCycleInstr;
  std::list<MachineInstr*> EmittedInstrs;
  const MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const TargetSchedModel &TSchedModel;

  // Loop info for V_NOP hoisting, passed from the pass manager.
  MachineLoopInfo *MLI = nullptr;

  bool RunLdsBranchVmemWARHazardFixup;

  //===--------------------------------------------------------------------===//
  // Pre-RA WMMA Co-execution Window State
  //===--------------------------------------------------------------------===//

  /// Active WMMA co-execution info (slot masks, preferences).
  AMDGPU::CoExecInfo ActiveCoExecInfo;

  /// Current stage within the WMMA co-execution window (0-based).
  /// nullopt when not in a WMMA window.
  std::optional<unsigned> CurrentCoExecStage;

  /// Cycle when the current WMMA window started.
  unsigned CoExecWindowStartCycle = 0;

  /// Tracks cycles until TRANS can be issued again (back-to-back TRANS hazard).
  unsigned CyclesUntilTRANS = 0;

  /// Tracks cycles until next VALU after multi-cycle VALU (CVT hazard).
  unsigned CyclesUntilVALU = 0;

  /// Debug: log of what was scheduled at each stage of the co-exec window.
  /// '.' = not yet reached, '-' = stall, else CoExecMask short char.
  std::array<char, AMDGPU::MaxCoExecStages> CoExecWindowLog;

  /// Debug: print the co-exec window visual summary.
  void dumpCoExecWindow() const;

  /// Check WMMA co-execution slot hazard for pre-RA scheduling.
  /// Returns stall cycles needed before MI can be issued in the current slot.
  unsigned checkWMMACoexecSlot(const MachineInstr &MI) const;

  /// Check TRANS-after-TRANS hazard. Returns stall cycles if MI is TRANS
  /// or multi-cycle VALU and a previous TRANS shadow is still active.
  unsigned checkTRANSHazard(const MachineInstr &MI) const;

  /// Check multi-cycle VALU hazard. Returns stall cycles if MI is a VALU
  /// that would conflict with an active multi-cycle VALU pipeline.
  unsigned checkMultiCycleVALUHazard(const MachineInstr &MI) const;

  /// Check if we have both a TRANS and WMMA window active. If so, for VALU
  /// instructions, return the number of stall cycles until one shadow clears.
  unsigned checkMultiShadowHazard(const MachineInstr &MI) const;

  /// Update WMMA window state when a WMMA instruction is emitted.
  void updateWMMAWindowState(const MachineInstr &MI);

  /// Update TRANS state when an instruction is emitted.
  void updateTRANSState(const MachineInstr &MI);

  /// Update multi-cycle VALU state when an instruction is emitted.
  void updateMultiCycleVALUState(const MachineInstr &MI);

  /// Pre-RA wrapper for EmitInstruction - updates pre-RA specific state.
  void preRAEmitInstruction(MachineInstr *MI);

  /// Pre-RA wrapper for AdvanceCycle - advances pipeline state.
  void preRAAdvanceCycle();

  /// Pre-RA wrapper for Reset - clears pre-RA specific state.
  void preRAReset();

  /// RegUnits of uses in the current soft memory clause.
  mutable BitVector ClauseUses;

  /// RegUnits of defs in the current soft memory clause.
  mutable BitVector ClauseDefs;

  void resetClause() const {
    ClauseUses.reset();
    ClauseDefs.reset();
  }

  void addClauseInst(const MachineInstr &MI) const;

  /// \returns the number of wait states before another MFMA instruction can be
  /// issued after \p MI.
  unsigned getMFMAPipelineWaitStates(const MachineInstr &MI) const;

  // Advance over a MachineInstr bundle. Look for hazards in the bundled
  // instructions.
  void processBundle();

  // Run on an individual instruction in hazard recognizer mode. This can be
  // used on a newly inserted instruction before returning from PreEmitNoops.
  void runOnInstruction(MachineInstr *MI);

  int getWaitStatesSince(IsHazardFn IsHazard, int Limit,
                         GetNumWaitStatesFn GetNumWaitStates) const;
  int getWaitStatesSince(IsHazardFn IsHazard, int Limit) const;
  int getWaitStatesSinceDef(unsigned Reg, IsHazardFn IsHazardDef,
                            int Limit) const;
  int getWaitStatesSinceSetReg(IsHazardFn IsHazard, int Limit) const;

  int checkSoftClauseHazards(MachineInstr *SMEM) const;
  int checkSMRDHazards(MachineInstr *SMRD) const;
  int checkVMEMHazards(MachineInstr *VMEM) const;
  int checkDPPHazards(MachineInstr *DPP) const;
  int checkDivFMasHazards(MachineInstr *DivFMas) const;
  int checkGetRegHazards(MachineInstr *GetRegInstr) const;
  int checkSetRegHazards(MachineInstr *SetRegInstr) const;
  int createsVALUHazard(const MachineInstr &MI) const;
  int checkVALUHazards(MachineInstr *VALU) const;
  int checkVALUHazardsHelper(const MachineOperand &Def,
                             const MachineRegisterInfo &MRI) const;
  int checkRWLaneHazards(MachineInstr *RWLane) const;
  int checkRFEHazards(MachineInstr *RFE) const;
  int checkInlineAsmHazards(MachineInstr *IA) const;
  int checkReadM0Hazards(MachineInstr *SMovRel) const;
  int checkNSAtoVMEMHazard(MachineInstr *MI) const;
  int checkFPAtomicToDenormModeHazard(MachineInstr *MI) const;
  // Emit \p WaitStatesNeeded V_NOP instructions before \p InsertPt.
  // If IsHoisting is true, uses empty DebugLoc for compiler-inserted NOPs.
  void emitVNops(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                 int WaitStatesNeeded, bool IsHoisting = false);
  void fixHazards(MachineInstr *MI);
  bool fixVcmpxPermlaneHazards(MachineInstr *MI);
  bool fixVMEMtoScalarWriteHazards(MachineInstr *MI);
  bool fixSMEMtoVectorWriteHazards(MachineInstr *MI);
  bool fixVcmpxExecWARHazard(MachineInstr *MI);
  bool fixLdsBranchVmemWARHazard(MachineInstr *MI);
  bool fixLdsDirectVALUHazard(MachineInstr *MI);
  bool fixLdsDirectVMEMHazard(MachineInstr *MI);
  bool fixVALUPartialForwardingHazard(MachineInstr *MI);
  bool fixVALUTransUseHazard(MachineInstr *MI);
  bool fixVALUTransCoexecutionHazards(MachineInstr *MI);
  bool fixWMMAHazards(MachineInstr *MI);
  int checkWMMACoexecutionHazards(MachineInstr *MI) const;
  bool fixWMMACoexecutionHazards(MachineInstr *MI);
  bool tryHoistWMMAVnopsFromLoop(MachineInstr *MI, int WaitStatesNeeded);
  bool hasWMMAHazardInLoop(MachineLoop *L, MachineInstr *MI,
                           bool IncludeSubloops = true);
  bool hasWMMAToWMMARegOverlap(const MachineInstr &WMMA,
                               const MachineInstr &MI) const;
  bool hasWMMAToVALURegOverlap(const MachineInstr &WMMA,
                               const MachineInstr &MI) const;
  bool isCoexecutionHazardFor(const MachineInstr &I,
                              const MachineInstr &MI) const;
  bool fixShift64HighRegBug(MachineInstr *MI);
  bool fixVALUMaskWriteHazard(MachineInstr *MI);
  bool fixRequiredExportPriority(MachineInstr *MI);
  bool fixGetRegWaitIdle(MachineInstr *MI);
  bool fixDsAtomicAsyncBarrierArriveB64(MachineInstr *MI);
  bool fixScratchBaseForwardingHazard(MachineInstr *MI);
  bool fixSetRegMode(MachineInstr *MI);

  int checkMAIHazards(MachineInstr *MI) const;
  int checkMAIHazards908(MachineInstr *MI) const;
  int checkMAIHazards90A(MachineInstr *MI) const;
  /// Pad the latency between neighboring MFMA instructions with s_nops. The
  /// percentage of wait states to fill with s_nops is specified by the command
  /// line option '-amdgpu-mfma-padding-ratio'.
  ///
  /// For example, with '-amdgpu-mfma-padding-ratio=100':
  ///
  /// 2 pass MFMA instructions have a latency of 2 wait states. Therefore, a
  /// 'S_NOP 1' will be added between sequential MFMA instructions.
  ///
  /// V_MFMA_F32_4X4X1F32
  /// V_MFMA_F32_4X4X1F32
  ///-->
  /// V_MFMA_F32_4X4X1F32
  /// S_NOP 1
  /// V_MFMA_F32_4X4X1F32
  int checkMFMAPadding(MachineInstr *MI) const;
  int checkMAIVALUHazards(MachineInstr *MI) const;
  int checkMAILdStHazards(MachineInstr *MI) const;
  int checkPermlaneHazards(MachineInstr *MI) const;

public:
  /// Construct with explicit operating mode.
  GCNHazardRecognizer(const MachineFunction &MF, OperatingMode Mode,
                      MachineLoopInfo *MLI = nullptr);

  /// Legacy constructor - defaults to PostRA mode.
  GCNHazardRecognizer(const MachineFunction &MF,
                      MachineLoopInfo *MLI = nullptr);

  ~GCNHazardRecognizer();

  /// Returns the current operating mode.
  OperatingMode getOperatingMode() const { return Mode; }

  /// Returns true if running in pre-RA scheduling mode.
  bool isPreRA() const { return Mode == OperatingMode::PreRA; }

  /// Returns true if running in post-RA scheduling mode.
  bool isPostRA() const { return Mode == OperatingMode::PostRA; }

  /// Returns true if running as a scheduler (pre-RA or post-RA).
  bool isSchedulerMode() const { return isPreRA() || isPostRA(); }

  /// Returns true if running as the standalone hazard recognizer pass.
  bool isHazardRecognizerMode() const {
    return Mode == OperatingMode::HazardRecognizerMode;
  }

  //===--------------------------------------------------------------------===//
  // Pre-RA Co-execution Window Queries
  //===--------------------------------------------------------------------===//

  /// Returns true if currently inside a WMMA co-execution window.
  bool inCoExecWindow() const { return CurrentCoExecStage.has_value(); }

  /// Returns the current stage within the co-execution window, or nullopt.
  std::optional<unsigned> getCurrentCoExecStage() const {
    return CurrentCoExecStage;
  }

  /// Returns the active co-execution info (slot masks, preferences).
  const AMDGPU::CoExecInfo &getActiveCoExecInfo() const {
    return ActiveCoExecInfo;
  }

  /// Get the CoExecMask for a given instruction.
  static AMDGPU::CoExecMaskT getCoExecMaskForMI(const MachineInstr &MI,
                                                const SIInstrInfo &TII);
  // We can only issue one instruction per cycle.
  bool atIssueLimit() const override { return true; }
  void EmitInstruction(SUnit *SU) override;
  void EmitInstruction(MachineInstr *MI) override;
  HazardType getHazardType(SUnit *SU, int Stalls) override;

  /// Returns the number of wait states until all hazards for \p MI are
  /// resolved. This is useful for scheduling heuristics that want
  /// cycle-accurate hazard information rather than just a boolean.  Unlike
  /// PreEmitNoops, this does not modify state or fix hazards.
  unsigned getHazardWaitStates(MachineInstr *MI) const;
  void EmitNoop() override;
  unsigned PreEmitNoops(MachineInstr *) override;
  unsigned PreEmitNoopsCommon(MachineInstr *) const;
  void AdvanceCycle() override;
  void RecedeCycle() override;
  bool ShouldPreferAnother(SUnit *SU) const override;
  void Reset() override;
};

} // end namespace llvm

#endif //LLVM_LIB_TARGET_AMDGPUHAZARDRECOGNIZERS_H
