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
#include <string>
#include <unordered_map>

namespace llvm {
namespace ejit {

/// Maintains funcName → bitcode data mapping, indexed by deterministic
/// FNV-1a hash of the function name (funcIdx) for O(1) lookup.
class EJitModuleLoader {
public:
  void registerBitcode(const std::string &funcName,
                       const uint8_t *data, size_t size);

  /// O(1) funcIdx-based bitcode lookup for the cache-miss path.
  Expected<StringRef> getBitcodeByFuncIdx(uint32_t funcIdx) const;

  /// funcIdx → function name for the cache-miss path (O(1)).
  const std::string &getFuncNameByFuncIdx(uint32_t funcIdx) const;

  /// Period metadata cached on first cache miss to avoid re-parsing bitcode.
  struct FuncMeta {
    unsigned dimCount = 0;
    std::string periodNames[4];
  };
  /// Parse bitcode once per funcIdx, cache the result. Hash collision
  /// precondition: detectHashCollisions() passed at init time.
  const FuncMeta &getOrCacheFuncMeta(uint32_t funcIdx);

private:
  struct Entry {
    std::string funcName;
    const uint8_t *data;
    size_t size;
  };
  std::unordered_map<uint32_t, Entry> entriesByFuncIdx_;
  mutable std::unordered_map<uint32_t, FuncMeta> funcMetaCache_;
};

} // namespace ejit
} // namespace llvm

#endif
