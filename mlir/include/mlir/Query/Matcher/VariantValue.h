//===--- VariantValue.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supports all the types required for dynamic Matcher construction.
// Used by the registry to construct matchers in a generic way.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_VARIANTVALUE_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_VARIANTVALUE_H

#include "ErrorBuilder.h"
#include "MatchersInternal.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::query::matcher {

// All types that VariantValue can contain.
enum class ArgKind { Matcher, String };

// A variant matcher object to abstract simple and complex matchers into a
// single object type.
class VariantMatcher {
  class MatcherOps;

  // Payload interface to be specialized by each matcher type. It follows a
  // similar interface as VariantMatcher itself.
  class Payload {
  public:
    virtual ~Payload();
    virtual std::optional<DynMatcher> getDynMatcher() const = 0;
    virtual std::string getTypeAsString() const = 0;
  };

public:
  // A null matcher.
  VariantMatcher();

  // Clones the provided matcher.
  static VariantMatcher SingleMatcher(DynMatcher matcher);

  // Makes the matcher the "null" matcher.
  void reset();

  // Checks if the matcher is null.
  bool isNull() const { return !value; }

  // Returns the matcher
  std::optional<DynMatcher> getDynMatcher() const;

  // String representation of the type of the value.
  std::string getTypeAsString() const;

private:
  explicit VariantMatcher(std::shared_ptr<Payload> value)
      : value(std::move(value)) {}

  class SinglePayload;

  std::shared_ptr<const Payload> value;
};

// Variant value class with a tagged union with value type semantics. It is used
// by the registry as the return value and argument type for the matcher factory
// methods. It can be constructed from any of the supported types:
//  - StringRef
//  - VariantMatcher
class VariantValue {
public:
  VariantValue() : type(ValueType::Nothing) {}

  VariantValue(const VariantValue &other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &other);

  // Specific constructors for each supported type.
  VariantValue(const llvm::StringRef string);
  VariantValue(const VariantMatcher &matcher);

  // String value functions.
  bool isString() const;
  const llvm::StringRef &getString() const;
  void setString(const llvm::StringRef &string);

  // Matcher value functions.
  bool isMatcher() const;
  const VariantMatcher &getMatcher() const;
  void setMatcher(const VariantMatcher &matcher);

  // String representation of the type of the value.
  std::string getTypeAsString() const;

private:
  void reset();

  // All supported value types.
  enum class ValueType {
    Nothing,
    String,
    Matcher,
  };

  // All supported value types.
  union AllValues {
    llvm::StringRef *String;
    VariantMatcher *Matcher;
  };

  ValueType type;
  AllValues value;
};

// A VariantValue instance annotated with its parser context.
struct ParserValue {
  ParserValue() {}
  llvm::StringRef text;
  internal::SourceRange range;
  VariantValue value;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_VARIANTVALUE_H
