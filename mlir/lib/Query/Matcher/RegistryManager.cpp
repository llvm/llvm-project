//===- RegistryManager.cpp - Matcher registry -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry map populated at static initialization time.
//
//===----------------------------------------------------------------------===//

#include "RegistryManager.h"
#include "mlir/Query/Matcher/Registry.h"

#include <set>
#include <utility>

namespace mlir::query::matcher {
namespace {

// Enum to string for autocomplete.
static std::string asArgString(ArgKind kind) {
  switch (kind) {
  case ArgKind::Boolean:
    return "Boolean";
  case ArgKind::Matcher:
    return "Matcher";
  case ArgKind::Signed:
    return "Signed";
  case ArgKind::String:
    return "String";
  }
  llvm_unreachable("Unhandled ArgKind");
}

} // namespace

void Registry::registerMatcherDescriptor(
    llvm::StringRef matcherName,
    std::unique_ptr<internal::MatcherDescriptor> callback) {
  assert(!constructorMap.contains(matcherName));
  constructorMap[matcherName] = std::move(callback);
}

std::optional<MatcherCtor>
RegistryManager::lookupMatcherCtor(llvm::StringRef matcherName,
                                   const Registry &matcherRegistry) {
  auto it = matcherRegistry.constructors().find(matcherName);
  return it == matcherRegistry.constructors().end()
             ? std::optional<MatcherCtor>()
             : it->second.get();
}

std::vector<ArgKind> RegistryManager::getAcceptedCompletionTypes(
    llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context) {
  // Starting with the above seed of acceptable top-level matcher types, compute
  // the acceptable type set for the argument indicated by each context element.
  std::set<ArgKind> typeSet;
  typeSet.insert(ArgKind::Matcher);

  for (const auto &ctxEntry : context) {
    MatcherCtor ctor = ctxEntry.first;
    unsigned argNumber = ctxEntry.second;
    std::vector<ArgKind> nextTypeSet;

    if (ctor->isVariadic() || argNumber < ctor->getNumArgs())
      ctor->getArgKinds(argNumber, nextTypeSet);

    typeSet.insert(nextTypeSet.begin(), nextTypeSet.end());
  }

  return std::vector<ArgKind>(typeSet.begin(), typeSet.end());
}

std::vector<MatcherCompletion>
RegistryManager::getMatcherCompletions(llvm::ArrayRef<ArgKind> acceptedTypes,
                                       const Registry &matcherRegistry) {
  std::vector<MatcherCompletion> completions;

  // Search the registry for acceptable matchers.
  for (const auto &m : matcherRegistry.constructors()) {
    const internal::MatcherDescriptor &matcher = *m.getValue();
    llvm::StringRef name = m.getKey();

    unsigned numArgs = matcher.isVariadic() ? 1 : matcher.getNumArgs();
    std::vector<std::vector<ArgKind>> argKinds(numArgs);

    for (const ArgKind &kind : acceptedTypes) {
      if (kind != ArgKind::Matcher)
        continue;

      for (unsigned arg = 0; arg != numArgs; ++arg)
        matcher.getArgKinds(arg, argKinds[arg]);
    }

    std::string decl;
    llvm::raw_string_ostream os(decl);

    std::string typedText = std::string(name);
    os << "Matcher: " << name << "(";

    for (const std::vector<ArgKind> &arg : argKinds) {
      if (&arg != &argKinds[0])
        os << ", ";

      bool firstArgKind = true;
      // Two steps. First all non-matchers, then matchers only.
      for (const ArgKind &argKind : arg) {
        if (!firstArgKind)
          os << "|";

        firstArgKind = false;
        os << asArgString(argKind);
      }
    }

    if (matcher.isVariadic())
      os << ",...";

    os << ")";
    typedText += "(";

    if (argKinds.empty())
      typedText += ")";
    else if (argKinds[0][0] == ArgKind::String)
      typedText += "\"";

    completions.emplace_back(typedText, decl);
  }

  return completions;
}

VariantMatcher RegistryManager::constructMatcher(
    MatcherCtor ctor, internal::SourceRange nameRange,
    llvm::StringRef functionName, llvm::ArrayRef<ParserValue> args,
    internal::Diagnostics *error) {
  VariantMatcher out = ctor->create(nameRange, args, error);
  if (functionName.empty() || out.isNull())
    return out;

  if (std::optional<DynMatcher> result = out.getDynMatcher()) {
    result->setFunctionName(functionName);
    return VariantMatcher::SingleMatcher(*result);
  }

  error->addError(nameRange, internal::ErrorType::RegistryNotBindable);
  return {};
}

} // namespace mlir::query::matcher
