//===-- EJitRegistrationStore.cpp - Registration Staging Area --------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"

using namespace llvm::ejit;

EJitRegistrationStore &EJitRegistrationStore::instance() {
  static EJitRegistrationStore store;
  return store;
}

void EJitRegistrationStore::registerBitcode(const std::string &funcName,
                                            const uint8_t *data, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  bitcodes_.push_back({funcName, data, size});
}

void EJitRegistrationStore::registerPeriodArray(
    const std::string &periodName, const std::string &varName,
    void *baseAddr, uint64_t arraySize) {
  std::lock_guard<std::mutex> lock(mutex_);
  periodArrays_.push_back({periodName, varName, baseAddr, arraySize});
}

void EJitRegistrationStore::registerStaticVar(const std::string &varName,
                                              void *varAddr) {
  std::lock_guard<std::mutex> lock(mutex_);
  staticVars_.push_back({varName, varAddr});
}

StoredData EJitRegistrationStore::consume() {
  std::lock_guard<std::mutex> lock(mutex_);
  StoredData data;
  data.bitcodes = std::move(bitcodes_);
  data.periodArrays = std::move(periodArrays_);
  data.staticVars = std::move(staticVars_);
  bitcodes_.clear();
  periodArrays_.clear();
  staticVars_.clear();
  return data;
}
