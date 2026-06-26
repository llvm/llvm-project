//===-- EJitRegistrationStore.h - Registration Staging Area
//----------------===//
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
#ifndef EJIT_FREESTANDING
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

/// A registration failure recorded before/at ejit_init (e.g. funcIndex or
/// lifecycle capacity exhausted in a constructor-phase callback). Surfaced so
/// ejit_init can fail instead of silently building a half-registered taskpool.
struct RegistrationError {
  int code = 0; // 0 == no error; otherwise an ejit_status_t value.
  std::string message;
  std::string funcName;

  bool ok() const { return code == 0; }
};

struct StoredData {
  std::vector<BitcodeEntry> bitcodes;
  std::vector<PeriodArrayEntry> periodArrays;
  std::vector<StaticVarEntry> staticVars;
  std::vector<SymbolEntry> userSymbols;

  bool empty() const {
    return bitcodes.empty() && periodArrays.empty() && staticVars.empty() &&
           userSymbols.empty();
  }
};

/// Process-global singleton staging area for data passed from
/// constructor-phase registration callbacks to ejit_init.
class EJitRegistrationStore {
public:
#ifdef EJIT_FREESTANDING
  using MutexType = BareMetalMutex;
#else
  using MutexType = std::mutex;
#endif
  static EJitRegistrationStore &instance();

  void registerBitcode(const std::string &funcName, const uint8_t *data,
                       size_t size);
  void registerPeriodArray(const std::string &periodName,
                           const std::string &varName, void *baseAddr,
                           uint64_t arraySize);
  void registerStaticVar(const std::string &varName, void *varAddr);
  void registerSymbol(const std::string &name, void *addr);

  /// Record the FIRST registration failure (subsequent calls are ignored so the
  /// earliest cause is preserved). Used by constructor-phase callbacks that
  /// cannot return a status (the void ejit_register_* C ABI).
  void recordError(int code, const std::string &message,
                   const std::string &funcName);
  /// True if a registration error has been recorded and not yet consumed.
  bool hasError() const;
  /// Return and clear the recorded error (code 0 when none). ejit_init consumes
  /// it so each init cycle starts from a clean error state.
  RegistrationError consumeError();

  /// Consume and clear all stored registration data.
  StoredData consume();

private:
  EJitRegistrationStore() = default;

  MutexType mutex_;
  std::vector<BitcodeEntry> bitcodes_;
  std::vector<PeriodArrayEntry> periodArrays_;
  std::vector<StaticVarEntry> staticVars_;
  std::vector<SymbolEntry> userSymbols_;
  RegistrationError error_;
};

} // namespace ejit
} // namespace llvm

#endif
