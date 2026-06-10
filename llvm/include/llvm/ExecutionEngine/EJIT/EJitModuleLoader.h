//===-- EJitModuleLoader.h - Bitcode Lookup -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITMODULELOADER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITMODULELOADER_H

#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace llvm {
namespace ejit {

/// Maintains funcName → bitcode data mapping. Bitcode is lazily loaded
/// (first JIT compilation triggers module parsing via LLVMContext).
/// Internally indexed by deterministic 32-bit hash of the function name
/// (FNV-1a) to enable O(1) lookup via funcIdx in the hot path.
class EJitModuleLoader {
public:
  void registerBitcode(const std::string &funcName,
                       const uint8_t *data, size_t size);

  /// Returns a view of the bitcode data. The data is stable for the
  /// lifetime of the process (pointing to embedded .ejit.bitcode section).
  Expected<StringRef> getBitcode(const std::string &funcName) const;

  /// O(1) funcIdx-based bitcode lookup for the cache-miss path.
  Expected<StringRef> getBitcodeByFuncIdx(uint32_t funcIdx) const;

  /// Returns a unique uint32_t index for a function name.
  /// Assigns a new index on first call; stable thereafter.
  uint32_t getFuncIndex(const std::string &funcName);

  /// Reverse lookup: index → function name. Returns "" if unknown.
  const std::string &getFuncName(uint32_t index) const;

  /// funcIdx → function name for the cache-miss path (O(1)).
  const std::string &getFuncNameByFuncIdx(uint32_t funcIdx) const;

  /// Run collision detection for funcIdx hashes (call once after all
  /// registrations). Returns true if a collision is detected.
  bool detectHashCollisions() const;

  size_t getEntryCount() const;
  size_t getTotalBitcodeSize() const;

private:
  struct Entry {
    std::string funcName;
    const uint8_t *data;
    size_t size;
  };
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_map<uint32_t, Entry> entriesByFuncIdx_;
  std::unordered_map<std::string, uint32_t> funcToIndex_;
  std::vector<std::string> indexToFunc_;
  size_t totalSize_ = 0;
};

} // namespace ejit
} // namespace llvm

#endif
