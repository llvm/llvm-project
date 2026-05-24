//===-- EJitRegistrationStore.cpp - Registration Staging Area --------------===//

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

void EJitRegistrationStore::registerPeriodArray(
    const std::string &periodName, const std::string &varName,
    void *baseAddr, uint64_t arraySize) {
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
