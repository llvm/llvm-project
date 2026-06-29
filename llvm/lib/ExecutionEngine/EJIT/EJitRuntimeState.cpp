//===-- EJitRuntimeState.cpp - Activate/Deactivate State ------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include <cassert>
#include <mutex>

using namespace llvm::ejit;

void PeriodArrayRegistry::registerArray(const std::string &periodName,
                                        const std::string &varName,
                                        void *baseAddr, size_t size) {
  EJIT_DIAG("registry registerArray period=%s var=%s base=%p size=%zu",
            periodName.c_str(), varName.c_str(), baseAddr, size);
  PeriodArrayInfo info{varName, periodName, baseAddr, size};
  arraysByPeriod_[periodName].push_back(info);
  varNameIndex_[varName] = info;
  baseAddrIndex_[reinterpret_cast<uintptr_t>(baseAddr)] = info;
}

void PeriodArrayRegistry::registerStaticVar(const std::string &varName,
                                            void *varAddr) {
  EJIT_DIAG("registry registerStaticVar var=%s addr=%p", varName.c_str(),
            varAddr);
  staticVars_.push_back({varName, varAddr});
  staticVarIndex_[varName] = varAddr;
}

const std::vector<PeriodArrayInfo> *
PeriodArrayRegistry::getArrays(const std::string &periodName) const {
  auto it = arraysByPeriod_.find(periodName);
  if (it == arraysByPeriod_.end())
    return nullptr;
  return &it->second;
}

const PeriodArrayInfo *
PeriodArrayRegistry::getArrayInfo(const std::string &varName) const {
  auto it = varNameIndex_.find(varName);
  if (it == varNameIndex_.end())
    return nullptr;
  return &it->second;
}

void *
PeriodArrayRegistry::getStaticVarAddr(const std::string &varName) const {
  auto it = staticVarIndex_.find(varName);
  if (it == staticVarIndex_.end())
    return nullptr;
  return it->second;
}

const PeriodArrayInfo *
PeriodArrayRegistry::getArrayByBaseAddr(void *addr) const {
  auto it = baseAddrIndex_.find(reinterpret_cast<uintptr_t>(addr));
  if (it == baseAddrIndex_.end())
    return nullptr;
  return &it->second;
}

void EJitRuntimeState::activate(const std::string &periodName,
                                uint8_t cellIdx) {
  EJIT_DIAG("runtimeState activate period=%s cellIdx=%u", periodName.c_str(),
            cellIdx);
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  // Period-level: fan out to all arrays registered under this period name.
  const auto *arrs = registry_.getArrays(periodName);
  if (arrs) {
    for (const auto &info : *arrs)
      arrayStates_[reinterpret_cast<uintptr_t>(info.baseAddr)][cellIdx] =
          PeriodState::Active;
  }
}

void EJitRuntimeState::deactivate(const std::string &periodName,
                                  uint8_t cellIdx) {
  EJIT_DIAG("runtimeState deactivate period=%s cellIdx=%u", periodName.c_str(),
            cellIdx);
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  const auto *arrs = registry_.getArrays(periodName);
  if (arrs) {
    for (const auto &info : *arrs)
      arrayStates_[reinterpret_cast<uintptr_t>(info.baseAddr)][cellIdx] =
          PeriodState::Inactive;
  }
}

void EJitRuntimeState::activateArray(void *arrayPtr, uint8_t cellIdx) {
  if (!registry_.getArrayByBaseAddr(arrayPtr)) {
    EJIT_DIAG("runtimeState activateArray FAIL: arrayPtr=%p not registered",
              arrayPtr);
    assert(registry_.getArrayByBaseAddr(arrayPtr) &&
           "activateArray: arrayPtr is not a registered period array");
  }
  EJIT_DIAG("runtimeState activateArray ptr=%p cellIdx=%u", arrayPtr, cellIdx);
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  arrayStates_[reinterpret_cast<uintptr_t>(arrayPtr)][cellIdx] =
      PeriodState::Active;
}

void EJitRuntimeState::deactivateArray(void *arrayPtr, uint8_t cellIdx) {
  if (!registry_.getArrayByBaseAddr(arrayPtr)) {
    EJIT_DIAG("runtimeState deactivateArray FAIL: arrayPtr=%p not registered",
              arrayPtr);
    assert(registry_.getArrayByBaseAddr(arrayPtr) &&
           "deactivateArray: arrayPtr is not a registered period array");
  }
  EJIT_DIAG("runtimeState deactivateArray ptr=%p cellIdx=%u", arrayPtr,
            cellIdx);
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  arrayStates_[reinterpret_cast<uintptr_t>(arrayPtr)][cellIdx] =
      PeriodState::Inactive;
}

void EJitRuntimeState::activateAll(const std::string &periodName) {
  EJIT_DIAG("runtimeState activateAll period=%s", periodName.c_str());
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  const auto *arrs = registry_.getArrays(periodName);
  if (arrs) {
    for (const auto &info : *arrs) {
      auto &inner = arrayStates_[reinterpret_cast<uintptr_t>(info.baseAddr)];
      for (size_t i = 0; i < info.arraySize; i++)
        inner[static_cast<uint8_t>(i)] = PeriodState::Active;
    }
  }
}

void EJitRuntimeState::deactivateAll(const std::string &periodName) {
  EJIT_DIAG("runtimeState deactivateAll period=%s", periodName.c_str());
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  const auto *arrs = registry_.getArrays(periodName);
  if (arrs) {
    for (const auto &info : *arrs) {
      auto &inner = arrayStates_[reinterpret_cast<uintptr_t>(info.baseAddr)];
      for (size_t i = 0; i < info.arraySize; i++)
        inner[static_cast<uint8_t>(i)] = PeriodState::Inactive;
    }
  }
}

bool EJitRuntimeState::isActive(const std::string &periodName,
                                uint8_t cellIdx) const {
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  // The built-in "static" time window is always active (SPEC4 §2.1).
  // static variables are registered via ejit_register_static_var, not
  // listed in arraysByPeriod_, so getArrays("static") returns nullptr.
  if (periodName == "static")
    return true;

  // Return true if ANY array registered under this period name is active
  // at the given cellIdx (period-level semantics for JIT compile decisions).
  const auto *arrs = registry_.getArrays(periodName);
  if (!arrs)
    return false;
  for (const auto &info : *arrs) {
    auto pit = arrayStates_.find(reinterpret_cast<uintptr_t>(info.baseAddr));
    if (pit == arrayStates_.end())
      continue;
    auto cit = pit->second.find(cellIdx);
    if (cit != pit->second.end() && cit->second == PeriodState::Active)
      return true;
  }
  return false;
}
