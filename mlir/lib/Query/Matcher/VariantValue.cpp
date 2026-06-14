//===--- Variantvalue.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Matcher/VariantValue.h"

namespace mlir::query::matcher {

VariantMatcher::Payload::~Payload() = default;

class VariantMatcher::SinglePayload : public VariantMatcher::Payload {
public:
  explicit SinglePayload(DynMatcher matcher) : matcher(std::move(matcher)) {}

  std::optional<DynMatcher> getDynMatcher() const override { return matcher; }

  std::string getTypeAsString() const override { return "Matcher"; }

private:
  DynMatcher matcher;
};

class VariantMatcher::VariadicOpPayload : public VariantMatcher::Payload {
public:
  VariadicOpPayload(DynMatcher::VariadicOperator varOp,
                    std::vector<VariantMatcher> args)
      : varOp(varOp), args(std::move(args)) {}

  std::optional<DynMatcher> getDynMatcher() const override {
    std::vector<DynMatcher> dynMatchers;
    for (const auto &variantMatcher : args) {
      std::optional<DynMatcher> dynMatcher = variantMatcher.getDynMatcher();
      if (dynMatcher)
        dynMatchers.push_back(dynMatcher.value());
    }
    auto result = DynMatcher::constructVariadic(varOp, dynMatchers);
    return *result;
  }

  std::string getTypeAsString() const override {
    std::string inner;
    llvm::interleave(
        args, [&](auto const &arg) { inner += arg.getTypeAsString(); },
        [&] { inner += " & "; });
    return inner;
  }

private:
  const DynMatcher::VariadicOperator varOp;
  const std::vector<VariantMatcher> args;
};

VariantMatcher::VariantMatcher() = default;

VariantMatcher VariantMatcher::SingleMatcher(DynMatcher matcher) {
  return VariantMatcher(std::make_shared<SinglePayload>(std::move(matcher)));
}

VariantMatcher
VariantMatcher::VariadicOperatorMatcher(DynMatcher::VariadicOperator varOp,
                                        ArrayRef<VariantMatcher> args) {
  return VariantMatcher(std::make_shared<VariadicOpPayload>(varOp, args));
}

std::optional<DynMatcher> VariantMatcher::MatcherOps::constructVariadicOperator(
    DynMatcher::VariadicOperator varOp,
    ArrayRef<VariantMatcher> innerMatchers) const {
  std::vector<DynMatcher> dynMatchers;
  for (const auto &innerMatcher : innerMatchers) {
    if (!innerMatcher.value)
      return std::nullopt;
    std::optional<DynMatcher> inner = innerMatcher.value->getDynMatcher();
    if (!inner)
      return std::nullopt;
    dynMatchers.push_back(*inner);
  }
  return *DynMatcher::constructVariadic(varOp, dynMatchers);
}

std::optional<DynMatcher> VariantMatcher::getDynMatcher() const {
  return value ? value->getDynMatcher() : std::nullopt;
}

void VariantMatcher::reset() { value.reset(); }

std::string VariantMatcher::getTypeAsString() const { return "<Nothing>"; }

VariantValue::VariantValue(const VariantValue &other)
    : type(ValueType::Nothing) {
  *this = other;
}

VariantValue::VariantValue(const llvm::StringRef string)
    : type(ValueType::String) {
  value.String = new llvm::StringRef(string);
}

VariantValue::VariantValue(const VariantMatcher &matcher)
    : type(ValueType::Matcher) {
  value.Matcher = new VariantMatcher(matcher);
}

VariantValue::VariantValue(int64_t signedValue) : type(ValueType::Signed) {
  value.Signed = signedValue;
}

VariantValue::VariantValue(bool setBoolean) : type(ValueType::Boolean) {
  value.Boolean = setBoolean;
}

VariantValue::~VariantValue() { reset(); }

VariantValue &VariantValue::operator=(const VariantValue &other) {
  if (this == &other)
    return *this;
  reset();
  switch (other.type) {
  case ValueType::String:
    setString(other.getString());
    break;
  case ValueType::Matcher:
    setMatcher(other.getMatcher());
    break;
  case ValueType::Signed:
    setSigned(other.getSigned());
    break;
  case ValueType::Boolean:
    setBoolean(other.getBoolean());
    break;
  case ValueType::Nothing:
    type = ValueType::Nothing;
    break;
  }
  return *this;
}

void VariantValue::reset() {
  switch (type) {
  case ValueType::String:
    delete value.String;
    break;
  case ValueType::Matcher:
    delete value.Matcher;
    break;
  // Cases that do nothing.
  case ValueType::Signed:
  case ValueType::Boolean:
  case ValueType::Nothing:
    break;
  }
  type = ValueType::Nothing;
}

// Signed
bool VariantValue::isSigned() const { return type == ValueType::Signed; }

int64_t VariantValue::getSigned() const { return value.Signed; }

void VariantValue::setSigned(int64_t newValue) {
  type = ValueType::Signed;
  value.Signed = newValue;
}

// Boolean
bool VariantValue::isBoolean() const { return type == ValueType::Boolean; }

bool VariantValue::getBoolean() const { return value.Boolean; }

void VariantValue::setBoolean(bool newValue) {
  type = ValueType::Boolean;
  value.Boolean = newValue;
}

bool VariantValue::isString() const { return type == ValueType::String; }

const llvm::StringRef &VariantValue::getString() const {
  assert(isString());
  return *value.String;
}

void VariantValue::setString(const llvm::StringRef &newValue) {
  reset();
  type = ValueType::String;
  value.String = new llvm::StringRef(newValue);
}

bool VariantValue::isMatcher() const { return type == ValueType::Matcher; }

const VariantMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *value.Matcher;
}

void VariantValue::setMatcher(const VariantMatcher &newValue) {
  reset();
  type = ValueType::Matcher;
  value.Matcher = new VariantMatcher(newValue);
}

std::string VariantValue::getTypeAsString() const {
  switch (type) {
  case ValueType::String:
    return "String";
  case ValueType::Matcher:
    return "Matcher";
  case ValueType::Signed:
    return "Signed";
  case ValueType::Boolean:
    return "Boolean";
  case ValueType::Nothing:
    return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // namespace mlir::query::matcher
