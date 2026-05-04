//===-- EJitRuntimeState.cpp - Activate/Deactivate State ------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"

using namespace llvm::ejit;

void PeriodArrayRegistry::registerArray(const std::string &periodName,
                                        const std::string &varName,
                                        void *baseAddr, size_t size) {
  PeriodArrayInfo info{varName, periodName, baseAddr, size};
  arraysByPeriod_[periodName].push_back(info);
  varNameIndex_[varName] = &arraysByPeriod_[periodName].back();
  baseAddrIndex_[reinterpret_cast<uintptr_t>(baseAddr)] =
      &arraysByPeriod_[periodName].back();
}

void PeriodArrayRegistry::registerStaticVar(const std::string &varName,
                                            void *varAddr) {
  staticVars_.push_back({varName, varAddr});
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
  return it->second;
}

const PeriodArrayInfo *
PeriodArrayRegistry::getArrayByBaseAddr(void *addr) const {
  auto it = baseAddrIndex_.find(reinterpret_cast<uintptr_t>(addr));
  if (it == baseAddrIndex_.end())
    return nullptr;
  return it->second;
}

void EJitRuntimeState::activate(const std::string &periodName,
                                uint8_t cellIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  states_[periodName][cellIdx] = PeriodState::Active;
}

void EJitRuntimeState::deactivate(const std::string &periodName,
                                  uint8_t cellIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  states_[periodName][cellIdx] = PeriodState::Inactive;
}

void EJitRuntimeState::activateAll(const std::string &periodName) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &cells = states_[periodName];
  for (auto &[idx, state] : cells)
    state = PeriodState::Active;
}

void EJitRuntimeState::deactivateAll(const std::string &periodName) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &cells = states_[periodName];
  for (auto &[idx, state] : cells)
    state = PeriodState::Inactive;
}

bool EJitRuntimeState::isActive(const std::string &periodName,
                                uint8_t cellIdx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pit = states_.find(periodName);
  if (pit == states_.end())
    return false;
  auto cit = pit->second.find(cellIdx);
  if (cit == pit->second.end())
    return false;
  return cit->second == PeriodState::Active;
}
