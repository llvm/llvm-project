//===- Logger.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ACC_OFFLOAD_LOGGER_H_
#define LLVM_ACC_OFFLOAD_LOGGER_H_

#include "Shared/Debug.h"
#include "Shared/SourceInfo.h"
#include <optional>

namespace llvm::acc::target::debug {
inline std::string formatLoc(ident_t *Loc) {
  SourceInfo SI(Loc);
  return std::string(SI.getFilename()) + ":" + std::to_string(SI.getLine());
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, ident_t *Loc) {
  return OS << formatLoc(Loc);
}
struct ScopeLoggerTy {
  const char *ScopeName;
  std::optional<ident_t *> Loc = std::nullopt;
  ScopeLoggerTy(const char *ScopeName, ident_t *Loc)
      : ScopeName(ScopeName), Loc(Loc) {
    ODBG() << "> " << ScopeName << "(" << Loc << ")";
  }
  ScopeLoggerTy(const char *ScopeName) : ScopeName(ScopeName) {
    ODBG() << "> " << ScopeName;
  }
  ~ScopeLoggerTy() {
    if (Loc)
      ODBG() << "< " << ScopeName << "(" << *Loc << ")";
    else
      ODBG() << "< " << ScopeName;
  }
};
} // namespace llvm::acc::target::debug

#define FUNC_LOGGER(...)                                                       \
  ScopeLoggerTy FunctionScopeLogger(__FUNCTION__, ##__VA_ARGS__)

#endif // LLVM_ACC_OFFLOAD_LOGGER_H_
