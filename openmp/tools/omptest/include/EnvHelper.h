//===- EnvHelper.h - General logging class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides environment helpers shared between a couple of places.
///
//===----------------------------------------------------------------------===//

#include <optional>

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_ENVHELPER_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_ENVHELPER_H

namespace omptest {
/// Load the value of a given boolean environmental variable. Return
/// std::nullopt if not specified in the environment.
inline std::optional<bool>
getBoolEnvironmentVariable(const char *VariableName) {
  if (VariableName == nullptr)
    return std::nullopt;
  if (const char *EnvValue = std::getenv(VariableName)) {
    std::string S{EnvValue};
    for (auto &C : S)
      C = (char)std::tolower(C);
    if (S == "1" || S == "on" || S == "true" || S == "yes")
      return true;
    if (S == "0" || S == "off" || S == "false" || S == "no")
      return false;
  }
  return std::nullopt;
}
} // namespace omptest
#endif // OPENMP_TOOLS_OMPTEST_INCLUDE_ENVHELPER_H
