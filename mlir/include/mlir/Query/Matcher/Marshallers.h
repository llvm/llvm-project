//===--- Marshallers.h - Generic matcher function marshallers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains function templates and classes to wrap matcher construct
// functions. It provides a collection of template function and classes that
// present a generic marshalling layer on top of matcher construct functions.
// The registry uses these to export all marshaller constructors with a uniform
// interface. This mechanism takes inspiration from clang-query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_MARSHALLERS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_MARSHALLERS_H

#include "ErrorBuilder.h"
#include "VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::query::matcher::internal {

// Helper template class for jumping from argument type to the correct is/get
// functions in VariantValue. This is used for verifying and extracting the
// matcher arguments.
template <class T>
struct ArgTypeTraits;
template <class T>
struct ArgTypeTraits<const T &> : public ArgTypeTraits<T> {};

template <>
struct ArgTypeTraits<llvm::StringRef> {

  static bool hasCorrectType(const VariantValue &value) {
    return value.isString();
  }

  static const llvm::StringRef &get(const VariantValue &value) {
    return value.getString();
  }

  static ArgKind getKind() { return ArgKind::String; }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<DynMatcher> {

  static bool hasCorrectType(const VariantValue &value) {
    return value.isMatcher();
  }

  static DynMatcher get(const VariantValue &value) {
    return *value.getMatcher().getDynMatcher();
  }

  static ArgKind getKind() { return ArgKind::Matcher; }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

// Interface for generic matcher descriptor.
// Offers a create() method that constructs the matcher from the provided
// arguments.
class MatcherDescriptor {
public:
  virtual ~MatcherDescriptor() = default;
  virtual VariantMatcher create(SourceRange nameRange,
                                const llvm::ArrayRef<ParserValue> args,
                                Diagnostics *error) const = 0;

  // Returns the number of arguments accepted by the matcher.
  virtual unsigned getNumArgs() const = 0;

  // Append the set of argument types accepted for argument 'argNo' to
  // 'argKinds'.
  virtual void getArgKinds(unsigned argNo,
                           std::vector<ArgKind> &argKinds) const = 0;
};

class FixedArgCountMatcherDescriptor : public MatcherDescriptor {
public:
  using MarshallerType = VariantMatcher (*)(void (*matcherFunc)(),
                                            llvm::StringRef matcherName,
                                            SourceRange nameRange,
                                            llvm::ArrayRef<ParserValue> args,
                                            Diagnostics *error);

  // Marshaller Function to unpack the arguments and call Func. Func is the
  // Matcher construct function. This is the function that the matcher
  // expressions would use to create the matcher.
  FixedArgCountMatcherDescriptor(MarshallerType marshaller,
                                 void (*matcherFunc)(),
                                 llvm::StringRef matcherName,
                                 llvm::ArrayRef<ArgKind> argKinds)
      : marshaller(marshaller), matcherFunc(matcherFunc),
        matcherName(matcherName), argKinds(argKinds.begin(), argKinds.end()) {}

  VariantMatcher create(SourceRange nameRange, llvm::ArrayRef<ParserValue> args,
                        Diagnostics *error) const override {
    return marshaller(matcherFunc, matcherName, nameRange, args, error);
  }

  unsigned getNumArgs() const override { return argKinds.size(); }

  void getArgKinds(unsigned argNo, std::vector<ArgKind> &kinds) const override {
    kinds.push_back(argKinds[argNo]);
  }

private:
  const MarshallerType marshaller;
  void (*const matcherFunc)();
  const llvm::StringRef matcherName;
  const std::vector<ArgKind> argKinds;
};

// Helper function to check if argument count matches expected count
inline bool checkArgCount(SourceRange nameRange, size_t expectedArgCount,
                          llvm::ArrayRef<ParserValue> args,
                          Diagnostics *error) {
  if (args.size() != expectedArgCount) {
    addError(error, nameRange, ErrorType::RegistryWrongArgCount,
             {llvm::Twine(expectedArgCount), llvm::Twine(args.size())});
    return false;
  }
  return true;
}

// Helper function for checking argument type
template <typename ArgType, size_t Index>
inline bool checkArgTypeAtIndex(llvm::StringRef matcherName,
                                llvm::ArrayRef<ParserValue> args,
                                Diagnostics *error) {
  if (!ArgTypeTraits<ArgType>::hasCorrectType(args[Index].value)) {
    addError(error, args[Index].range, ErrorType::RegistryWrongArgType,
             {llvm::Twine(matcherName), llvm::Twine(Index + 1)});
    return false;
  }
  return true;
}

// Marshaller function for fixed number of arguments
template <typename ReturnType, typename... ArgTypes, size_t... Is>
static VariantMatcher
matcherMarshallFixedImpl(void (*matcherFunc)(), llvm::StringRef matcherName,
                         SourceRange nameRange,
                         llvm::ArrayRef<ParserValue> args, Diagnostics *error,
                         std::index_sequence<Is...>) {
  using FuncType = ReturnType (*)(ArgTypes...);

  // Check if the argument count matches the expected count
  if (!checkArgCount(nameRange, sizeof...(ArgTypes), args, error))
    return VariantMatcher();

  // Check if each argument at the corresponding index has the correct type
  if ((... && checkArgTypeAtIndex<ArgTypes, Is>(matcherName, args, error))) {
    ReturnType fnPointer = reinterpret_cast<FuncType>(matcherFunc)(
        ArgTypeTraits<ArgTypes>::get(args[Is].value)...);
    return VariantMatcher::SingleMatcher(
        *DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
  }

  return VariantMatcher();
}

template <typename ReturnType, typename... ArgTypes>
static VariantMatcher
matcherMarshallFixed(void (*matcherFunc)(), llvm::StringRef matcherName,
                     SourceRange nameRange, llvm::ArrayRef<ParserValue> args,
                     Diagnostics *error) {
  return matcherMarshallFixedImpl<ReturnType, ArgTypes...>(
      matcherFunc, matcherName, nameRange, args, error,
      std::index_sequence_for<ArgTypes...>{});
}

// Fixed number of arguments overload
template <typename ReturnType, typename... ArgTypes>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(ReturnType (*matcherFunc)(ArgTypes...),
                        llvm::StringRef matcherName) {
  // Create a vector of argument kinds
  std::vector<ArgKind> argKinds = {ArgTypeTraits<ArgTypes>::getKind()...};
  return std::make_unique<FixedArgCountMatcherDescriptor>(
      matcherMarshallFixed<ReturnType, ArgTypes...>,
      reinterpret_cast<void (*)()>(matcherFunc), matcherName, argKinds);
}

} // namespace mlir::query::matcher::internal

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MARSHALLERS_H
