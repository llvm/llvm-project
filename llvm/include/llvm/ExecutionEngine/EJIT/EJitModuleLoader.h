//===-- EJitModuleLoader.h - Bitcode Lookup -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITMODULELOADER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITMODULELOADER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>

namespace llvm {
namespace ejit {

/// Maintains funcIndex → bitcode data in a fixed table keyed by the dense
/// funcIndex EJitFuncRegistry assigns to each entry-function name at
/// registration (EJitFuncRegistry.h). The wrapper backfills the SAME registry
/// index into its per-function global, so the index it requests always selects
/// the bitcode the loader registered for that exact name — no name hashing, so
/// distinct functions never alias. registerBitcode rejects a null/zero payload
/// or funcIndex-capacity exhaustion; same-name re-registration is idempotent
/// only when the payload (data ptr + size) is identical, otherwise rejected.
class EJitModuleLoader {
public:
  /// Register bitcode for \p funcName. Returns false (registering nothing) on a
  /// null/zero payload, funcIndex capacity exhaustion, or a same-name
  /// re-registration with a DIFFERENT payload (the original is kept); true on a
  /// fresh insert or an idempotent same-name + same-payload registration.
  bool registerBitcode(const std::string &funcName, const uint8_t *data,
                       size_t size);

  /// Dense-funcIndex bitcode lookup for the cache-miss path.
  Expected<StringRef> getBitcodeByFuncIdx(uint32_t funcIdx) const;

  /// Dense funcIndex → function name for the cache-miss path.
  const std::string &getFuncNameByFuncIdx(uint32_t funcIdx) const;

  /// Period metadata cached on first cache miss to avoid re-parsing bitcode.
  /// dimTypes[i] is the explicit lifecycle dimType for periodNames[i], read
  /// back from the process-global EJitLifecycleRegistry (the SAME slot the
  /// wrapper baked into its per-lifecycle global), never re-derived here.
  struct FuncMeta {
    unsigned dimCount = 0;
    std::string periodNames[4];
    uint32_t dimTypes[4] = {0, 0, 0, 0};
  };
  /// Parse bitcode once per funcIdx, cache the result.
  const FuncMeta &getOrCacheFuncMeta(uint32_t funcIdx);

private:
  struct Entry {
    std::string funcName;
    const uint8_t *data = nullptr;
    size_t size = 0;
  };
  /// Fixed table: EJitFuncRegistry's dense funcIndex(funcName) → bitcode entry.
  /// The key is assigned once per name in the process-global registry, so it is
  /// stable regardless of registration order or count and matches the value the
  /// wrapper backfills into its per-function global.
  std::unordered_map<uint32_t, Entry> entriesByFuncIdx_;
  mutable std::unordered_map<uint32_t, FuncMeta> funcMetaCache_;
};

} // namespace ejit
} // namespace llvm

#endif
