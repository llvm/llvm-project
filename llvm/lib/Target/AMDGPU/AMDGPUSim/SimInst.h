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
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

namespace llvm {
namespace AMDGPUSim {

/// Instruction classes used to select simulator timing behavior.
enum class InstClass : uint8_t {
#define AMDGPUSIM_INST_CLASS(Name, Unit, Flavor) Name,
#include "SimEnums.def"
};

/// Structural execution units tracked for resource occupancy and issue stalls.
enum class FunctionalUnit : uint8_t {
#define AMDGPUSIM_FUNCTIONAL_UNIT(Name) Name,
#include "SimEnums.def"
  NUM_UNITS
};

/// Return the textual representation of \p IC.
inline const char *getInstClassName(InstClass IC) {
  switch (IC) {
#define AMDGPUSIM_INST_CLASS(Name, Unit, Flavor)                               \
  case InstClass::Name:                                                        \
    return #Name;
#include "SimEnums.def"
  }
  llvm_unreachable("unknown instruction class");
}

/// Return the textual representation of \p Unit.
inline const char *getUnitName(FunctionalUnit Unit) {
  switch (Unit) {
#define AMDGPUSIM_FUNCTIONAL_UNIT(Name)                                        \
  case FunctionalUnit::Name:                                                   \
    return #Name;
#include "SimEnums.def"
  case FunctionalUnit::NUM_UNITS:
    break;
  }
  llvm_unreachable("unknown functional unit");
}

/// Return the structural unit occupied by \p IC.
inline FunctionalUnit getUnitForClass(InstClass IC) {
  switch (IC) {
#define AMDGPUSIM_INST_CLASS(Name, Unit, Flavor)                               \
  case InstClass::Name:                                                        \
    return FunctionalUnit::Unit;
#include "SimEnums.def"
  }
  llvm_unreachable("unknown instruction class");
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
