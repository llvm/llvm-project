//===-- EJitLifecycleRegistry.h - Global lifecycle -> dimType map ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Process-global, name-keyed assignment of lifecycle (period) names to a dense
// dimType slot in [0, kEJitMaxDimTypes) (spec §5.1 MAX_DIM_TYPES).
//
// A dimType must be (a) identical for one lifecycle across every independently
// compiled module and (b) distinct for different lifecycles, while fitting in
// the 8-slot SwitchController. No per-module-independent rule can satisfy that
// (a name hash folded into [0,8) aliases unrelated names, e.g.
// fnv("cell")%8 == fnv("tenant")%8), so the slot is assigned once, here, the
// first time a lifecycle name is registered, and every other site reads the
// SAME slot back by name. The wrapper does not re-derive it: it loads the slot
// from a per-lifecycle global this registry fills at registration; the module
// loader reads it back with lookup(). The SwitchController therefore keeps its
// strict, direct [dimType][instanceId] indexing with no runtime probing.
//
// Determinism / ordering: a lifecycle's slot is fixed by the FIRST registration
// and is never reassigned, so registering more modules never shifts an existing
// lifecycle's slot. Distinct names always receive distinct slots (no collision)
// until the 8 slots are exhausted; the 9th distinct lifecycle is rejected
// (kEJitInvalidDimType) — a clean failure, never a silent alias or default
// enable.
//
// Concurrency: resolveAssign() (the only mutator) runs solely at registration,
// which completes single-threaded at process/EJit startup before any wrapped
// function is called; the compile worker only ever calls the read-only
// lookup(). Registration therefore happens-before every lookup and no lock is
// required.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITLIFECYCLEREGISTRY_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITLIFECYCLEREGISTRY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace ejit {

class EJitLifecycleRegistry {
public:
  /// The single process-global instance.
  static EJitLifecycleRegistry &instance() {
    static EJitLifecycleRegistry Reg;
    return Reg;
  }

  /// Return the dense dimType slot for \p Name, assigning the next free slot on
  /// first sight. Idempotent: the same name always maps to the same slot.
  /// Returns kEJitInvalidDimType once all kEJitMaxDimTypes slots are taken (the
  /// 9th distinct lifecycle is rejected, never aliased). Registration-only.
  uint32_t resolveAssign(StringRef Name) {
    for (uint32_t I = 0; I < count_; ++I)
      if (names_[I] == Name)
        return I;
    if (count_ >= kEJitMaxDimTypes)
      return kEJitInvalidDimType;
    names_[count_] = Name.str();
    return count_++;
  }

  /// Return the slot previously assigned to \p Name, or kEJitInvalidDimType if
  /// the lifecycle was never registered. Read-only (never assigns).
  uint32_t lookup(StringRef Name) const {
    for (uint32_t I = 0; I < count_; ++I)
      if (names_[I] == Name)
        return I;
    return kEJitInvalidDimType;
  }

  /// Number of distinct lifecycles assigned so far.
  uint32_t count() const { return count_; }

  /// Deterministic digest of the (name -> dimType slot) mapping. Order MATTERS
  /// here (slot == position), so this folds names sequentially. Used by the
  /// cross-core shared taskpool to reject a peer whose dimType mapping diverges
  /// from the owner's.
  uint64_t fingerprint() const {
    uint64_t acc = 0xcbf29ce484222325ULL ^
                   (static_cast<uint64_t>(count_) * 0x100000001b3ULL);
    for (uint32_t I = 0; I < count_; ++I) {
      acc ^= static_cast<uint64_t>(I) + 0x9e3779b9ULL;
      acc *= 0x100000001b3ULL;
      for (unsigned char c : names_[I]) {
        acc ^= c;
        acc *= 0x100000001b3ULL;
      }
    }
    return acc;
  }

  /// Drop all assignments. For tests only (each test starts from empty state).
  void reset() {
    for (uint32_t I = 0; I < count_; ++I)
      names_[I].clear();
    count_ = 0;
  }

private:
  EJitLifecycleRegistry() = default;
  EJitLifecycleRegistry(const EJitLifecycleRegistry &) = delete;
  EJitLifecycleRegistry &operator=(const EJitLifecycleRegistry &) = delete;

  std::string names_[kEJitMaxDimTypes];
  uint32_t count_ = 0;
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITLIFECYCLEREGISTRY_H
