//===-- EJitRuntimeState.cpp - Activate/Deactivate State ------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include <mutex>

using namespace llvm::ejit;

void PeriodArrayRegistry::registerArray(const std::string &periodName,
                                        const std::string &varName,
                                        void *baseAddr, size_t size) {
  PeriodArrayInfo info{varName, periodName, baseAddr, size};
  arraysByPeriod_[periodName].push_back(info);
  varNameIndex_[varName] = info;
  baseAddrIndex_[reinterpret_cast<uintptr_t>(baseAddr)] = info;
}

void PeriodArrayRegistry::registerStaticVar(const std::string &varName,
                                            void *varAddr) {
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
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  states_[periodName][cellIdx] = PeriodState::Active;
}

void EJitRuntimeState::deactivate(const std::string &periodName,
                                  uint8_t cellIdx) {
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  states_[periodName][cellIdx] = PeriodState::Inactive;
}

void EJitRuntimeState::activateAll(const std::string &periodName) {
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  // Activate all cells of all arrays registered under this period name.
  const auto *arrs = registry_.getArrays(periodName);
  if (arrs) {
    for (const auto &info : *arrs) {
      for (size_t i = 0; i < info.arraySize; i++) {
        states_[periodName][static_cast<uint8_t>(i)] = PeriodState::Active;
      }
    }
  }
}

void EJitRuntimeState::deactivateAll(const std::string &periodName) {
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  const auto *arrs = registry_.getArrays(periodName);
  if (arrs) {
    for (const auto &info : *arrs) {
      for (size_t i = 0; i < info.arraySize; i++) {
        states_[periodName][static_cast<uint8_t>(i)] = PeriodState::Inactive;
      }
    }
  }
}

bool EJitRuntimeState::isActive(const std::string &periodName,
                                uint8_t cellIdx) const {
#ifndef EJIT_FREESTANDING
  std::lock_guard<decltype(mutex_)> lock(mutex_);
#endif
  auto pit = states_.find(periodName);
  if (pit == states_.end())
    return false;
  auto cit = pit->second.find(cellIdx);
  if (cit == pit->second.end())
    return false;
  return cit->second == PeriodState::Active;
}
