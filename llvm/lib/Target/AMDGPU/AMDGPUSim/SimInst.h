//===- AMDGPUSim/SimInst.h - Abstract instruction type ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines instruction classes, resources, wait requirements, and opaque
/// instruction data used by the representation independent simulator core.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINST_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINST_H

#include "AMDGPUCoExecInfo.h"
#include <cstdint>

namespace llvm {
namespace AMDGPUSim {

using AMDGPU::WMMAProperties;
using AMDGPU::WMMAVariant;

/// Instruction classes used to select simulator timing behavior.
enum class InstClass : uint8_t {
  VALU,
  SALU,
  TRANS,
  WMMA,
  DS_READ,
  DS_WRITE,
  VMEM_READ,
  VMEM_WRITE,
  SMEM,
  TDM,
  BARRIER,
  BARRIER_SIGNAL,
  BARRIER_WAIT,
  WAITCNT,
  DELAY_ALU,
  MSB_SET,
  NOP,
  BRANCH,
  OTHER
};

/// Structural execution units tracked for resource occupancy and issue stalls.
enum class FunctionalUnit : uint8_t {
  NONE = 0,
  XDL,
  VALU,
  SALU,
  TRANS,
  LDS,
  VMEM,
  SMEM,
  BRANCH,
  NUM_UNITS
};

/// Return the textual representation of \p IC.
inline const char *getInstClassName(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:
    return "VALU";
  case InstClass::SALU:
    return "SALU";
  case InstClass::TRANS:
    return "TRANS";
  case InstClass::WMMA:
    return "WMMA";
  case InstClass::DS_READ:
    return "DS_READ";
  case InstClass::DS_WRITE:
    return "DS_WRITE";
  case InstClass::VMEM_READ:
    return "VMEM_READ";
  case InstClass::VMEM_WRITE:
    return "VMEM_WRITE";
  case InstClass::SMEM:
    return "SMEM";
  case InstClass::TDM:
    return "TDM";
  case InstClass::BARRIER:
    return "BARRIER";
  case InstClass::BARRIER_SIGNAL:
    return "BARRIER_SIGNAL";
  case InstClass::BARRIER_WAIT:
    return "BARRIER_WAIT";
  case InstClass::WAITCNT:
    return "WAITCNT";
  case InstClass::DELAY_ALU:
    return "DELAY_ALU";
  case InstClass::MSB_SET:
    return "MSB_SET";
  case InstClass::NOP:
    return "NOP";
  case InstClass::BRANCH:
    return "BRANCH";
  case InstClass::OTHER:
    return "OTHER";
  }
  return "UNKNOWN";
}

/// Return the textual representation of \p Unit.
inline const char *getUnitName(FunctionalUnit Unit) {
  switch (Unit) {
  case FunctionalUnit::NONE:
    return "NONE";
  case FunctionalUnit::XDL:
    return "XDL";
  case FunctionalUnit::VALU:
    return "VALU";
  case FunctionalUnit::SALU:
    return "SALU";
  case FunctionalUnit::TRANS:
    return "TRANS";
  case FunctionalUnit::LDS:
    return "LDS";
  case FunctionalUnit::VMEM:
    return "VMEM";
  case FunctionalUnit::SMEM:
    return "SMEM";
  case FunctionalUnit::BRANCH:
    return "BRANCH";
  case FunctionalUnit::NUM_UNITS:
    return "NUM_UNITS";
  }
  return "UNKNOWN";
}

/// Return the structural unit occupied by \p IC.
inline FunctionalUnit getUnitForClass(InstClass IC) {
  switch (IC) {
  case InstClass::WMMA:
    return FunctionalUnit::XDL;
  case InstClass::VALU:
    return FunctionalUnit::VALU;
  case InstClass::TRANS:
    return FunctionalUnit::TRANS;
  case InstClass::SALU:
  case InstClass::DELAY_ALU:
  case InstClass::MSB_SET:
    return FunctionalUnit::SALU;
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::TDM:
    return FunctionalUnit::LDS;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return FunctionalUnit::VMEM;
  case InstClass::SMEM:
    return FunctionalUnit::SMEM;
  case InstClass::BRANCH:
    return FunctionalUnit::BRANCH;
  case InstClass::BARRIER:
  case InstClass::BARRIER_SIGNAL:
  case InstClass::BARRIER_WAIT:
  case InstClass::WAITCNT:
  case InstClass::NOP:
  case InstClass::OTHER:
    return FunctionalUnit::NONE;
  }
  return FunctionalUnit::NONE;
}

/// Wait counter domains modeled by the simulator.
/// DepCtr represents VaVdst dependency waits rather than a memory queue.
enum class WaitType : uint8_t {
  DS = 0,
  VMEMLoad,
  VMEMStore,
  SMEM,
  Tensor,
  XCnt,
  DepCtr
};

/// Stores counter information used to evaluate a modeled wait.
struct WaitRequirement {
  WaitType Type;
  unsigned Count;

  WaitRequirement(WaitType Type, unsigned Count) : Type(Type), Count(Count) {}
};

/// Representation independent instruction data produced by an adapter and
/// consumed by the simulator core.
struct SimInst {
  /// Opaque source instruction pointer supplied by the adapter.
  void *InstPtr = nullptr;
  /// Instruction class used to select simulator behavior.
  InstClass Class = InstClass::OTHER;
  /// Modeled result latency in cycles.
  unsigned Latency = 1;
  /// Structural execution unit occupied by the instruction.
  FunctionalUnit Unit = FunctionalUnit::NONE;
  /// Caller assigned sequence number used for logging.
  unsigned InstIndex = 0;

  SimInst() = default;
  SimInst(void *Ptr, InstClass C, unsigned Lat, FunctionalUnit U,
          unsigned Index = 0)
      : InstPtr(Ptr), Class(C), Latency(Lat), Unit(U), InstIndex(Index) {}

  /// Cast InstPtr to the source instruction type expected by the adapter.
  template <typename T> T *getAs() const { return static_cast<T *>(InstPtr); }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
