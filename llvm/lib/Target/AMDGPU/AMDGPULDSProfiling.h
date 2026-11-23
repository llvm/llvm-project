//===- AMDGPULDSProfiling.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// LDS bank-conflict profiling clones.
///
/// For each kernel that accesses LDS, AMDGPUCreateProfilingClonesPass creates
/// a second kernel "<name>.prof" with an identical body. AMDGPULDSProfilingPass
/// then numbers the LDS instructions of both copies identically, records the
/// address of each one in a PC section, and in the clone inserts code which
/// computes the per-wave N-way bank conflict of each supported LDS access and
/// accumulates it in "<name>.prof.counters".
///
/// Counter layout: one record of five 32-bit words per site, at
/// counters + site * 20:
///   [0] sum of N-way over samples
///   [1] number of samples
///   [2] sum of total bank rounds
///   [3] sum of baseline bank rounds
///   [4] number of invalid samples (currently never written)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULDSPROFILING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULDSPROFILING_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class GCNSubtarget;
class TargetMachine;

namespace AMDGPU::LDSProfiling {

/// Set on the clone. The value is the name of the kernel it was cloned from.
inline constexpr StringLiteral CloneAttr("amdgpu-lds-prof-clone");
/// Set on a kernel which has a profiling clone.
inline constexpr StringLiteral OriginalAttr("amdgpu-lds-prof-original");
/// Set by a frontend to suppress the clone for one kernel.
inline constexpr StringLiteral OptOutAttr("amdgpu-no-profiling-clone");

inline constexpr StringLiteral CloneSuffix(".prof");
inline constexpr StringLiteral CountersSuffix(".counters");

/// PC section which holds one entry per LDS instruction:
/// { i32 pc-relative address, i32 site id, i32 kind }.
inline constexpr StringLiteral SitesSection("amdgpu_lds_prof_sites");
/// Non-allocated section which holds one entry per kernel:
/// { i32 number of sites, i32 hash of the site sequence }, in a symbol named
/// "<kernel>.ldsprof".
inline constexpr StringLiteral InfoSection(".AMDGPU.ldsprof");

/// Size in bytes of the counter record of one site.
inline constexpr unsigned SiteRecordSize = 20;

/// Bank-conflict model of an LDS instruction.
enum SiteKind : unsigned {
  /// Counted as a site, but not instrumented.
  Unsupported = 0,
  /// One dword per lane, one service group: DS_READ_B32 and DS_WRITE_B32.
  OrdinaryB32 = 1,
};

/// \returns true if profiling clones are implemented for \p ST.
bool isSupportedSubtarget(const GCNSubtarget &ST);

/// \returns the name of the kernel whose LDS layout \p F uses: the kernel a
/// profiling clone was created from, or the name of \p F itself.
StringRef getLDSKernelName(const Function &F);

} // namespace AMDGPU::LDSProfiling

class AMDGPUCreateProfilingClonesPass
    : public RequiredPassInfoMixin<AMDGPUCreateProfilingClonesPass> {
  const TargetMachine &TM;

public:
  AMDGPUCreateProfilingClonesPass(const TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

class AMDGPULDSProfilingPass
    : public RequiredPassInfoMixin<AMDGPULDSProfilingPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULDSPROFILING_H
