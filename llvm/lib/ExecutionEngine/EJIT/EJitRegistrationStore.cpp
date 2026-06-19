//===-- EJitRegistrationStore.cpp - Registration Staging Area
//--------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include <mutex>

using namespace llvm::ejit;

EJitRegistrationStore &EJitRegistrationStore::instance() {
  static EJitRegistrationStore store;
  return store;
}

void EJitRegistrationStore::registerBitcode(const std::string &funcName,
                                            const uint8_t *data, size_t size) {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  bitcodes_.push_back({funcName, data, size});
}

void EJitRegistrationStore::registerPeriodArray(const std::string &periodName,
                                                const std::string &varName,
                                                void *baseAddr,
                                                uint64_t arraySize) {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  periodArrays_.push_back({periodName, varName, baseAddr, arraySize});
}

void EJitRegistrationStore::registerStaticVar(const std::string &varName,
                                              void *varAddr) {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  staticVars_.push_back({varName, varAddr});
}

void EJitRegistrationStore::registerSymbol(const std::string &name,
                                           void *addr) {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  userSymbols_.push_back({name, addr});
}

void EJitRegistrationStore::recordError(int code, const std::string &message,
                                        const std::string &funcName) {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  // Preserve the first error so the earliest root cause is reported.
  if (error_.code != 0)
    return;
  error_.code = code;
  error_.message = message;
  error_.funcName = funcName;
}

bool EJitRegistrationStore::hasError() const {
  // error_ is only written under mutex_; a relaxed read of the code is fine for
  // the single-threaded startup registration phase.
  return error_.code != 0;
}

RegistrationError EJitRegistrationStore::consumeError() {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  RegistrationError taken = error_;
  error_ = RegistrationError{};
  return taken;
}

StoredData EJitRegistrationStore::consume() {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  StoredData data;
  data.bitcodes = std::move(bitcodes_);
  data.periodArrays = std::move(periodArrays_);
  data.staticVars = std::move(staticVars_);
  data.userSymbols = std::move(userSymbols_);
  bitcodes_.clear();
  periodArrays_.clear();
  staticVars_.clear();
  userSymbols_.clear();
  return data;
}
