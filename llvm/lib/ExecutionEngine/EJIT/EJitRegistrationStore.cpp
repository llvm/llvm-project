//===-- EJitRegistrationStore.cpp - Registration Staging Area
//--------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include <mutex>

using namespace llvm::ejit;

EJitRegistrationStore &EJitRegistrationStore::instance() {
  static EJitRegistrationStore store;
  return store;
}

void EJitRegistrationStore::registerBitcode(const std::string &funcName,
                                            const uint8_t *data, size_t size) {
  EJIT_DIAG("regstore stage bitcode name=%s data=%p size=%zu",
            funcName.c_str(), data, size);
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  bitcodes_.push_back({funcName, data, size});
}

void EJitRegistrationStore::registerPeriodArray(const std::string &periodName,
                                                const std::string &varName,
                                                void *baseAddr,
                                                uint64_t arraySize) {
  EJIT_DIAG("regstore stage periodArray period=%s var=%s base=%p size=%llu",
            periodName.c_str(), varName.c_str(), baseAddr,
            static_cast<unsigned long long>(arraySize));
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  periodArrays_.push_back({periodName, varName, baseAddr, arraySize});
}

void EJitRegistrationStore::registerStaticVar(const std::string &varName,
                                              void *varAddr) {
  EJIT_DIAG("regstore stage staticVar var=%s addr=%p", varName.c_str(),
            varAddr);
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  staticVars_.push_back({varName, varAddr});
}

void EJitRegistrationStore::registerSymbol(const std::string &name,
                                           void *addr) {
  EJIT_DIAG("regstore stage symbol name=%s addr=%p", name.c_str(), addr);
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  userSymbols_.push_back({name, addr});
}

void EJitRegistrationStore::recordError(int code, const std::string &message,
                                        const std::string &funcName) {
  std::lock_guard<decltype(mutex_)> lock(mutex_);
  // Preserve the first error so the earliest root cause is reported.
  if (error_.code != 0) {
    EJIT_DIAG("regstore recordError suppressed (first error kept): code=%d "
              "func=%s msg=%s",
              code, funcName.c_str(), message.c_str());
    return;
  }
  error_.code = code;
  error_.message = message;
  error_.funcName = funcName;
  EJIT_DIAG("regstore recordError: code=%d func=%s msg=%s", code,
            funcName.c_str(), message.c_str());
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
  EJIT_DIAG("regstore consumeError: code=%d func=%s", taken.code,
            taken.funcName.c_str());
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
  EJIT_DIAG("regstore consume: bitcodes=%zu periodArrays=%zu staticVars=%zu "
            "symbols=%zu",
            data.bitcodes.size(), data.periodArrays.size(),
            data.staticVars.size(), data.userSymbols.size());
  return data;
}
