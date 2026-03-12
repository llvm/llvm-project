//===-- AMDGPUCacheCapacityHazardRecognizer.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a hazard recognizer for modeling L1 data cache capacity
// pressure during instruction scheduling on AMDGPU processors.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCacheCapacityHazardRecognizer.h"
#include "AMDGPUWaitcntUtils.h"
#include "SIInstrInfo.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>
#include <optional>

#define DEBUG_TYPE "amdgpu-l1-hazard"

namespace llvm {
namespace AMDGPU {

/// Cap on the user-controlled L1 latency. The internal IssueHistory ring
/// buffer is sized in cycles, so an unclamped cl::opt could request a
/// pathologically large allocation per scheduling region.
static constexpr unsigned MaxLatencyClamp = 10000;

namespace {
unsigned getEffectiveCacheBytes(MachineInstr *MI) {
  unsigned Res = 0;
  for (auto *MMO : MI->memoperands()) {
    auto AS = MMO->getAddrSpace();
    if (AS == AMDGPUAS::FLAT_ADDRESS || AS == AMDGPUAS::GLOBAL_ADDRESS ||
        AS == AMDGPUAS::BUFFER_RESOURCE) {
      auto MMOSize = MMO->getSize();
      if (MMOSize.hasValue() && MMOSize.isPrecise() && !MMOSize.isScalable())
        Res += (unsigned)MMOSize.getValue().getFixedValue();
    }
  }
  return Res;
}
std::optional<unsigned> getInstrVMCNT(MachineInstr *MI, AMDGPU::IsaVersion IV) {
  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(MI->getOpcode());
  if (Opcode == AMDGPU::S_WAITCNT) {
    unsigned IEnc = MI->getOperand(0).getImm();
    AMDGPU::Waitcnt Waitcnt = AMDGPU::decodeWaitcnt(IV, IEnc);
    return Waitcnt.get(InstCounterType::LOAD_CNT);
  }
  return std::nullopt;
}
} // namespace

CacheCapacityHazardRecognizer::CacheCapacityHazardRecognizer(
    unsigned CapacityBytes, BytesPerCycle DrainRate, unsigned LatencyIn,
    AMDGPU::IsaVersion IsaVer)
    : IsaVer(IsaVer), Latency(std::min(LatencyIn, MaxLatencyClamp)) {
  if (LatencyIn > MaxLatencyClamp)
    LLVM_DEBUG(dbgs() << "L1Hazard: clamping latency from " << LatencyIn
                      << " to " << MaxLatencyClamp << "\n");
  assert(DrainRate.Dividend > 0 && DrainRate.Divisor > 0 &&
         "BytesPerCycle dividend and divisor must be positive");
  unsigned GCD = std::gcd(DrainRate.Dividend, DrainRate.Divisor);
  ScaledDrainPerCycle = DrainRate.Dividend / GCD;
  BytesScaleFactor = DrainRate.Divisor / GCD;
  ScaledCapacityBytes = CapacityBytes * BytesScaleFactor;
  unsigned MaxCycles =
      Latency +
      (ScaledCapacityBytes + ScaledDrainPerCycle - 1) / ScaledDrainPerCycle;
  // Buffer must accommodate both the time horizon (MaxCycles) and the worst-
  // case slot count: with multi-issue and small per-load footprints, up to
  // ScaledCapacityBytes entries can pile up before the byte cap blocks further
  // issues. Sizing only by MaxCycles undercounts when ScaledDrainPerCycle > 1.
  IssueHistory.resize(std::max(MaxCycles, ScaledCapacityBytes) + 1);
  MaxLookAhead = MaxCycles;
}

void CacheCapacityHazardRecognizer::Reset() {
  ScaledUsedBytes = 0;
  CurrentCycle = 0;
  IssueHead = IssueTail = 0;
}

void CacheCapacityHazardRecognizer::waitLessThanInstr(unsigned InstrCount) {
  while (historySize() > InstrCount) {
    AdvanceCycle();
  }
}

void CacheCapacityHazardRecognizer::issueBytes(unsigned Bytes) {
  if (Bytes == 0) {
    return;
  }
  unsigned ScaledBytes = Bytes * BytesScaleFactor;
  // Clamp to ScaledCapacityBytes so an instruction whose footprint exceeds the
  // modeled cache cannot make the drain loop below spin forever. Real-world
  // load groups larger than capacity will still drain the FIFO between issues.
  if (ScaledBytes > ScaledCapacityBytes) {
    LLVM_DEBUG(dbgs() << "L1Hazard: clamping issue footprint from " << Bytes
                      << " to capacity ("
                      << ScaledCapacityBytes / BytesScaleFactor
                      << ") bytes\n");
    ScaledBytes = ScaledCapacityBytes;
  }
  while (ScaledUsedBytes + ScaledBytes > ScaledCapacityBytes) {
    AdvanceCycle();
  }
  ScaledUsedBytes += ScaledBytes;
  IssueHistory[IssueHead] = {ScaledBytes, CurrentCycle};
  IssueHead += 1;
  if (IssueHead == IssueHistory.size())
    IssueHead = 0;
  assert(IssueHead != IssueTail && "Circular buffer overflow");
}

bool CacheCapacityHazardRecognizer::canWaitLessThanInstr(unsigned InstrCount,
                                                         int Stalls) const {
  unsigned RemBytesScaled, RemInstrs;
  simulateAdvanceCycles(Stalls, RemBytesScaled, RemInstrs);
  return RemInstrs <= InstrCount;
}

bool CacheCapacityHazardRecognizer::canIssueBytes(unsigned Bytes,
                                                  int Stalls) const {
  // Scale incoming bytes to match the internal scaled representation.
  unsigned ScaledBytes = Bytes * BytesScaleFactor;
  // Clamp to ScaledCapacityBytes. Without this, an instruction whose footprint
  // exceeds the cache would report a hazard that can never be resolved.
  if (ScaledBytes > ScaledCapacityBytes) {
    LLVM_DEBUG(dbgs() << "L1Hazard: clamping probe footprint from " << Bytes
                      << " to capacity ("
                      << ScaledCapacityBytes / BytesScaleFactor
                      << ") bytes\n");
    ScaledBytes = ScaledCapacityBytes;
  }
  unsigned RemBytesScaled, RemInstrs;
  simulateAdvanceCycles(Stalls, RemBytesScaled, RemInstrs);
  return RemBytesScaled + ScaledBytes <= ScaledCapacityBytes;
}

ScheduleHazardRecognizer::HazardType
CacheCapacityHazardRecognizer::getHazardType(MachineInstr *MI, int Stalls) {
  assert(Stalls >= 0 && "Only top-down is implemented");
  if (SIInstrInfo::isWaitcnt(MI->getOpcode())) {
    auto VMCNT = getInstrVMCNT(MI, IsaVer);
    return !VMCNT || canWaitLessThanInstr(*VMCNT, Stalls) ? NoHazard
                                                          : NoopHazard;
  }
  unsigned CacheBytes = getEffectiveCacheBytes(MI);
  return canIssueBytes(CacheBytes, Stalls) ? NoHazard : NoopHazard;
}

ScheduleHazardRecognizer::HazardType
CacheCapacityHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  return getHazardType(SU->getInstr(), Stalls);
}

void CacheCapacityHazardRecognizer::EmitInstruction(SUnit *SU) {
  return EmitInstruction(SU->getInstr());
}

void CacheCapacityHazardRecognizer::EmitInstruction(MachineInstr *MI) {
  if (SIInstrInfo::isWaitcnt(MI->getOpcode())) {
    auto VMCNT = getInstrVMCNT(MI, IsaVer);
    if (VMCNT)
      waitLessThanInstr(*VMCNT);
    return;
  }
  return issueBytes(getEffectiveCacheBytes(MI));
}

void CacheCapacityHazardRecognizer::drainOneCycle(
    unsigned ScaledDrainPerCycle, unsigned CurrCycle, unsigned Latency,
    unsigned &Tail, unsigned Head, ArrayRef<IssueEntry> History,
    unsigned &UsedBytes, unsigned &Remain) {
  unsigned Budget = ScaledDrainPerCycle;
  while (Budget && Tail != Head) {
    // Only drain entries that have matured (age >= Latency).
    if (CurrCycle - History[Tail].Cycle < Latency)
      break;
    unsigned Finished = std::min(Budget, Remain);
    Budget -= Finished;
    assert(UsedBytes >= Finished && "ScaledUsedBytes underflow");
    UsedBytes -= Finished;
    if (Finished == Remain) {
      if (++Tail == History.size())
        Tail = 0;
      Remain = Tail != Head ? History[Tail].ScaledBytes : 0;
    } else {
      Remain -= Finished;
      break;
    }
  }
}

void CacheCapacityHazardRecognizer::simulateAdvanceCycles(
    int Stalls, unsigned &RemBytesScaled, unsigned &RemInstrs) const {
  unsigned SimHead = IssueHead, SimTail = IssueTail;
  unsigned SimUsed = ScaledUsedBytes;
  unsigned SimCycle = CurrentCycle;
  unsigned Remain = SimTail != SimHead ? IssueHistory[SimTail].ScaledBytes : 0;
  while (Stalls > 0 && SimTail != SimHead) {
    drainOneCycle(ScaledDrainPerCycle, SimCycle, Latency, SimTail, SimHead,
                  IssueHistory, SimUsed, Remain);
    ++SimCycle;
    --Stalls;
  }
  RemBytesScaled = SimUsed;
  RemInstrs = circularDistance(SimHead, SimTail, IssueHistory.size());
}

void CacheCapacityHazardRecognizer::AdvanceCycle() {
  // Each cycle drains exactly ScaledDrainPerCycle scaled-units.
  // Unused drain capacity does not carry over to future cycles.
  unsigned Remain =
      IssueTail != IssueHead ? IssueHistory[IssueTail].ScaledBytes : 0;
  drainOneCycle(ScaledDrainPerCycle, CurrentCycle, Latency, IssueTail,
                IssueHead, IssueHistory, ScaledUsedBytes, Remain);
  // Persist any partial drain on the entry we stopped at.
  if (IssueTail != IssueHead)
    IssueHistory[IssueTail].ScaledBytes = Remain;
  ++CurrentCycle;
}

void CacheCapacityHazardRecognizer::RecedeCycle() {
  llvm_unreachable("Bottom-Up scheduling not implemented");
}

} // namespace AMDGPU
} // namespace llvm
