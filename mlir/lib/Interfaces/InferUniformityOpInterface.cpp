//===- InferUniformityOpInterface.cpp - Uniformity inference --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferUniformityOpInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#include "mlir/Interfaces/InferUniformityOpInterface.cpp.inc"

StringRef mlir::stringifyUniformityScope(UniformityScope scope) {
  switch (scope) {
  case UniformityScope::Divergent:
    return "divergent";
  case UniformityScope::Subgroup:
    return "subgroup";
  case UniformityScope::Workgroup:
    return "workgroup";
  case UniformityScope::Cluster:
    return "cluster";
  case UniformityScope::Uniform:
    return "uniform";
  }
  llvm_unreachable("unknown uniformity scope");
}

std::optional<UniformityScope> mlir::symbolizeUniformityScope(StringRef name) {
  return llvm::StringSwitch<std::optional<UniformityScope>>(name)
      .Case("divergent", UniformityScope::Divergent)
      .Case("subgroup", UniformityScope::Subgroup)
      .Case("workgroup", UniformityScope::Workgroup)
      .Case("cluster", UniformityScope::Cluster)
      .Case("uniform", UniformityScope::Uniform)
      .Default(std::nullopt);
}

raw_ostream &mlir::operator<<(raw_ostream &os, UniformityScope scope) {
  return os << stringifyUniformityScope(scope);
}

void Uniformity::print(raw_ostream &os) const {
  if (isUninitialized())
    os << "<uninitialized>";
  else
    os << *scope;
}
