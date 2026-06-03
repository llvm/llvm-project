//===-- EJitLogger.cpp - EmbeddedJIT Error Logger -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"

#ifndef EJIT_FREESTANDING

#include <chrono>
#include <mutex>

using namespace llvm::ejit;

void EJitLogger::log(ErrorCode code, const std::string &message,
                     const std::string &funcName, const std::string &cacheKey,
                     size_t attemptedMemUsage) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  errors_[writeIdx_] = {code, message, funcName, cacheKey, ts,
                        attemptedMemUsage};
  writeIdx_ = (writeIdx_ + 1) % kMaxErrors;
  if (count_ < kMaxErrors)
    count_++;
}

const EJitError *EJitLogger::getLastError() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (count_ == 0)
    return nullptr;
  size_t idx = (writeIdx_ == 0) ? kMaxErrors - 1 : writeIdx_ - 1;
  return &errors_[idx];
}

bool EJitLogger::copyLastError(EJitError &out) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (count_ == 0)
    return false;
  size_t idx = (writeIdx_ == 0) ? kMaxErrors - 1 : writeIdx_ - 1;
  out = errors_[idx];
  return true;
}

std::vector<EJitError> EJitLogger::getErrors(size_t limit) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<EJitError> result;
  if (count_ == 0)
    return result;

  size_t n = std::min(limit, count_);
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    size_t idx = (writeIdx_ + kMaxErrors - count_ + i) % kMaxErrors;
    result.push_back(errors_[idx]);
  }
  return result;
}

void EJitLogger::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  writeIdx_ = 0;
  count_ = 0;
}

#else // EJIT_FREESTANDING — no-op stubs

using namespace llvm::ejit;

void EJitLogger::log(ErrorCode, const std::string &,
                     const std::string &, const std::string &, size_t) {}
const EJitError *EJitLogger::getLastError() const { return nullptr; }
bool EJitLogger::copyLastError(EJitError &) const { return false; }
std::vector<EJitError> EJitLogger::getErrors(size_t) const { return {}; }
void EJitLogger::clear() {}

#endif // EJIT_FREESTANDING
