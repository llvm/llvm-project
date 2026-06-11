//===-- EJitRuntimeState.h - Activate/Deactivate State --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITRUNTIMESTATE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITRUNTIMESTATE_H

#include "llvm/ExecutionEngine/EJIT/EJitBareMetal.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#ifndef EJIT_FREESTANDING
#include <mutex>
#endif

namespace llvm {
namespace ejit {

struct PeriodArrayInfo {
  std::string varName;
  std::string periodName;
  void *baseAddr;
  size_t arraySize;
};

struct StaticVarInfo {
  std::string varName;
  void *varAddr;
};

/// Registry of period arrays and static variables. Initialized once
/// at ejit_init time from StoredData, then read-only at runtime.
class PeriodArrayRegistry {
public:
  void registerArray(const std::string &periodName,
                     const std::string &varName,
                     void *baseAddr, size_t size);
  void registerStaticVar(const std::string &varName, void *varAddr);

  const std::vector<PeriodArrayInfo> *getArrays(const std::string &periodName) const;
  const std::vector<StaticVarInfo> &getStaticVars() const { return staticVars_; }
  const PeriodArrayInfo *getArrayInfo(const std::string &varName) const;
  void *getStaticVarAddr(const std::string &varName) const;

  const PeriodArrayInfo *getArrayByBaseAddr(void *addr) const;

private:
  std::unordered_map<std::string, std::vector<PeriodArrayInfo>> arraysByPeriod_;
  std::vector<StaticVarInfo> staticVars_;
  // Indexed by value for stable element addresses (vector push_back would
  // invalidate pointers to vector elements on reallocation).
  std::unordered_map<std::string, PeriodArrayInfo> varNameIndex_;
  std::unordered_map<std::string, void *> staticVarIndex_;
  std::unordered_map<uintptr_t, PeriodArrayInfo> baseAddrIndex_;
};

enum class PeriodState { Inactive, Active };

/// Manages activate/deactivate state of time-window period instances.
///
/// Two activation granularities:
///   - period-level: ejit_activate("cell", 3) → activates ALL arrays under "cell"
///     at index 3 (fans out to registered arrays).
///   - array-level:  ejit_activate_array("cell", &g_cellCfg, 3) → activates only
///     g_cellCfg at index 3 (other arrays under "cell" unaffected).
///
/// isActive() returns true if ANY array under the period name at the given
/// cellIdx is active (period-level semantics for JIT compile decision).
class EJitRuntimeState {
public:
#ifdef EJIT_FREESTANDING
  using MutexType = BareMetalMutex;
#else
  using MutexType = std::mutex;
#endif
  /// Period-level activation: activates all arrays registered under periodName.
  void activate(const std::string &periodName, uint8_t cellIdx);
  void deactivate(const std::string &periodName, uint8_t cellIdx);

  /// Array-level activation: activates only the specific array.
  void activateArray(void *arrayPtr, uint8_t cellIdx);
  void deactivateArray(void *arrayPtr, uint8_t cellIdx);

  void activateAll(const std::string &periodName);
  void deactivateAll(const std::string &periodName);
  bool isActive(const std::string &periodName, uint8_t cellIdx) const;

  PeriodArrayRegistry &getRegistry() { return registry_; }
  const PeriodArrayRegistry &getRegistry() const { return registry_; }

private:
  PeriodArrayRegistry registry_;
#ifndef EJIT_FREESTANDING
  mutable MutexType mutex_;
#endif
  /// Array-level state: (baseAddr, cellIdx) → Active/Inactive.
  /// Period-level activate/deactivate fans out to all arrays under that period.
  std::unordered_map<uintptr_t, std::unordered_map<uint8_t, PeriodState>>
      arrayStates_;
};

} // namespace ejit
} // namespace llvm

#endif
