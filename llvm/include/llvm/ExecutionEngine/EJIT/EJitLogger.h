//===-- EJitLogger.h - EmbeddedJIT Error Logger ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITLOGGER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITLOGGER_H

#include "llvm/ExecutionEngine/EJIT/EJitError.h"
#include <cstdint>
#include <vector>
#ifndef EJIT_FREESTANDING
#include <mutex>
#endif

namespace llvm {
namespace ejit {

/// Ring-buffer error logger. Pre-allocated, no dynamic allocation at log time.
class EJitLogger {
public:
  static constexpr size_t kMaxErrors = 32;

  void log(ErrorCode code, const std::string &message,
           const std::string &funcName = {},
           const std::string &cacheKey = {},
           size_t attemptedMemUsage = 0);

  const EJitError *getLastError() const;
  /// Copy the last error into the provided buffer. Thread-safe.
  bool copyLastError(EJitError &out) const;
  std::vector<EJitError> getErrors(size_t limit = kMaxErrors) const;
  void clear();

private:
#ifndef EJIT_FREESTANDING
  mutable std::mutex mutex_;
#endif
  EJitError errors_[kMaxErrors];
  size_t writeIdx_ = 0;
  size_t count_ = 0;
};

} // namespace ejit
} // namespace llvm

#endif
