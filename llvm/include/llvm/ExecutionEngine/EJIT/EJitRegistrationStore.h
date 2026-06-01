//===-- EJitRegistrationStore.h - Registration Staging Area ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITREGISTRATIONSTORE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITREGISTRATIONSTORE_H

#include "llvm/ExecutionEngine/EJIT/EJitBareMetal.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#ifndef EJIT_BARE_METAL
#include <mutex>
#endif

namespace llvm {
namespace ejit {

struct BitcodeEntry {
  std::string funcName;
  const uint8_t *data;
  size_t size;
};

struct PeriodArrayEntry {
  std::string periodName;
  std::string varName;
  void *baseAddr;
  uint64_t arraySize;
};

struct StaticVarEntry {
  std::string varName;
  void *varAddr;
};

struct SymbolEntry {
  std::string name;
  void *addr;
};

struct StoredData {
  std::vector<BitcodeEntry> bitcodes;
  std::vector<PeriodArrayEntry> periodArrays;
  std::vector<StaticVarEntry> staticVars;
  std::vector<SymbolEntry> userSymbols;

  bool empty() const {
    return bitcodes.empty() && periodArrays.empty() &&
           staticVars.empty() && userSymbols.empty();
  }
};

/// Process-global singleton staging area for data passed from
/// constructor-phase registration callbacks to ejit_init.
class EJitRegistrationStore {
public:
#ifdef EJIT_BARE_METAL
  using MutexType = BareMetalMutex;
#else
  using MutexType = std::mutex;
#endif
  static EJitRegistrationStore &instance();

  void registerBitcode(const std::string &funcName,
                       const uint8_t *data, size_t size);
  void registerPeriodArray(const std::string &periodName,
                           const std::string &varName,
                           void *baseAddr, uint64_t arraySize);
  void registerStaticVar(const std::string &varName, void *varAddr);
  void registerSymbol(const std::string &name, void *addr);

  /// Consume and clear all stored registration data.
  StoredData consume();

private:
  EJitRegistrationStore() = default;

  MutexType mutex_;
  std::vector<BitcodeEntry> bitcodes_;
  std::vector<PeriodArrayEntry> periodArrays_;
  std::vector<StaticVarEntry> staticVars_;
  std::vector<SymbolEntry> userSymbols_;
};

} // namespace ejit
} // namespace llvm

#endif
