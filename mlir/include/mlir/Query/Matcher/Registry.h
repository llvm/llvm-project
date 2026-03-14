//===--- Registry.h - Matcher Registry --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry class to manage the registry of matchers using a map.
//
// This class provides a convenient interface for registering and accessing
// matcher constructors using a string-based map.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRY_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRY_H

#include "Marshallers.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace mlir::query::matcher {

using ConstructorMap =
    llvm::StringMap<std::unique_ptr<const internal::MatcherDescriptor>>;

class Registry {
public:
  Registry() = default;
  ~Registry() = default;

  const ConstructorMap &constructors() const { return constructorMap; }

  template <typename MatcherType>
  void registerMatcher(const std::string &name, MatcherType matcher) {
    registerMatcherDescriptor(name,
                              internal::makeMatcherAutoMarshall(matcher, name));
  }

private:
  void registerMatcherDescriptor(
      llvm::StringRef matcherName,
      std::unique_ptr<internal::MatcherDescriptor> callback);

  ConstructorMap constructorMap;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRY_H
