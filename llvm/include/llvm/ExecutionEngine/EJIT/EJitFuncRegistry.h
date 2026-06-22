//===-- EJitFuncRegistry.h - Global function -> dense funcIndex map -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Process-global, name-keyed assignment of entry-function names to a dense
// funcIndex in [0, kEJitMaxFuncIndex) (spec §3.5 inFlight_[funcIndex]).
//
// A funcIndex must be (a) identical for one function across every
// independently compiled module and registration order and (b) distinct for
// different functions, while indexing the flat dedup table directly. A modulo
// name hash cannot guarantee (b) (collisions are common at 4096 slots), and no
// AOT pass sees every final module, so the index is assigned ONCE, here, the
// first time a function name is registered, and read back everywhere else by
// name. The wrapper does not compute it: it loads the value this registry
// backfills into @__ejit_funcidx_<name>; the module loader keys its bitcode
// table by the same value.
//
// Determinism: a function's index is fixed by its FIRST registration and never
// reassigned, so registering more modules never shifts an existing function's
// index. Distinct names always receive distinct dense indices (a monotonic
// counter), so two functions can never alias one slot. When all
// kEJitMaxFuncIndex indices are taken the next new name is rejected
// (kEJitInvalidFuncIndex) — a clean failure the caller propagates (ejit_init
// fails; the wrapper global stays invalid and that call site falls back),
// never a silent alias.
//
// Concurrency: resolveAssign() (the only mutator) runs solely at registration,
// which completes single-threaded at process/EJit startup before any wrapped
// function is called or any compile worker runs; lookups are read-only.
// Registration therefore happens-before every lookup and no lock is required.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITFUNCREGISTRY_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITFUNCREGISTRY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace llvm {
namespace ejit {

class EJitFuncRegistry {
public:
  /// The single process-global instance.
  static EJitFuncRegistry &instance() {
    static EJitFuncRegistry Reg;
    return Reg;
  }

  /// Return the dense funcIndex for \p Name, assigning the next free index on
  /// first sight. Idempotent: the same name always maps to the same index.
  /// Returns kEJitInvalidFuncIndex once all kEJitMaxFuncIndex indices are taken
  /// (the new name is rejected, never aliased onto an existing index).
  /// Registration-only.
  uint32_t resolveAssign(StringRef Name) {
    auto It = byName_.find(Name.str());
    if (It != byName_.end())
      return It->second;
    if (next_ >= kEJitMaxFuncIndex)
      return kEJitInvalidFuncIndex;
    uint32_t Idx = next_++;
    byName_.emplace(Name.str(), Idx);
    return Idx;
  }

  /// Return the index previously assigned to \p Name, or kEJitInvalidFuncIndex
  /// if the function was never registered. Read-only (never assigns).
  uint32_t lookup(StringRef Name) const {
    auto It = byName_.find(Name.str());
    return It == byName_.end() ? kEJitInvalidFuncIndex : It->second;
  }

  /// Number of distinct functions assigned so far.
  uint32_t count() const { return next_; }

  /// Deterministic digest of the whole (name -> funcIndex) mapping. Used by the
  /// cross-core shared taskpool to reject a peer core whose registration
  /// diverges from the owner's (different mapping => different indices). FNV-1a
  /// over each name+index, XOR-accumulated so it is independent of the
  /// unordered_map's iteration order; two cores with an identical mapping (same
  /// single binary image, same registration set) compute the same value.
  uint64_t fingerprint() const {
    uint64_t acc = 0xcbf29ce484222325ULL ^
                   (static_cast<uint64_t>(next_) * 0x100000001b3ULL);
    for (const auto &KV : byName_) {
      uint64_t h = 0xcbf29ce484222325ULL;
      for (unsigned char c : KV.first) {
        h ^= c;
        h *= 0x100000001b3ULL;
      }
      h ^= static_cast<uint64_t>(KV.second) + 0x9e3779b97f4a7c15ULL;
      h *= 0x100000001b3ULL;
      acc ^= h; // XOR-accumulate: order-independent
    }
    return acc;
  }

  /// Drop all assignments. For tests only (each test starts from empty state).
  void reset() {
    byName_.clear();
    next_ = 0;
  }

private:
  EJitFuncRegistry() = default;
  EJitFuncRegistry(const EJitFuncRegistry &) = delete;
  EJitFuncRegistry &operator=(const EJitFuncRegistry &) = delete;

  std::unordered_map<std::string, uint32_t> byName_;
  uint32_t next_ = 0;
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITFUNCREGISTRY_H
