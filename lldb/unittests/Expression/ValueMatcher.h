//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the definition of the ValueMatcher class which is a used
/// to match lldb_private::Value in gtest assert/expect macros. It also contains
/// several helper functions to create matchers for common Value types.
///
/// The ValueMatcher class was created using the gtest guide found here:
//  https://google.github.io/googletest/gmock_cook_book.html#writing-new-monomorphic-matchers
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_EXPRESSION_VALUEMATCHER_H
#define LLDB_UNITTESTS_EXPRESSION_VALUEMATCHER_H

#include "lldb/Core/Value.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <vector>

namespace lldb_private {

/// Custom printer for Value objects to make test failures more readable.
void PrintTo(const Value &val, std::ostream *os);

/// Custom matcher for Value.
///
/// It matches against an expected value_type, and context_type.
/// For HostAddress value types it will match the expected contents of
/// the host buffer. For other value types it matches against an expected
/// scalar value.
class ValueMatcher {
public:
  ValueMatcher(Value::ValueType value_type, const Scalar &expected_scalar,
               Value::ContextType context_type)
      : m_value_type(value_type), m_context_type(context_type),
        m_expected_scalar(expected_scalar) {
    assert(value_type == Value::ValueType::Scalar ||
           value_type == Value::ValueType::FileAddress ||
           value_type == Value::ValueType::LoadAddress);
  }

  ValueMatcher(Value::ValueType value_type,
               const std::vector<uint8_t> &expected_bytes,
               Value::ContextType context_type)
      : m_value_type(value_type), m_context_type(context_type),
        m_expected_bytes(expected_bytes) {
    assert(value_type == Value::ValueType::HostAddress);
  }

  // Typedef to hook into the gtest matcher machinery.
  using is_gtest_matcher = void;

  bool MatchAndExplain(const Value &val, std::ostream *os) const;

  void DescribeTo(std::ostream *os) const;

  void DescribeNegationTo(std::ostream *os) const;

private:
  Value::ValueType m_value_type = Value::ValueType::Invalid;
  Value::ContextType m_context_type = Value::ContextType::Invalid;
  Scalar m_expected_scalar;
  std::vector<uint8_t> m_expected_bytes;

  bool MatchAndExplainImpl(const Value &val, llvm::raw_ostream &os) const;
};

/// Matcher for Value with Scalar, FileAddress, or LoadAddress types.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(MatchScalarValue(...)));
testing::Matcher<Value> MatchScalarValue(Value::ValueType value_type,
                                         const Scalar &expected_scalar,
                                         Value::ContextType context_type);

/// Matcher for Value with HostAddress type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(MatchHostValue(...)));
testing::Matcher<Value>
MatchHostValue(Value::ValueType value_type,
               const std::vector<uint8_t> &expected_bytes,
               Value::ContextType context_type);

/// Helper to match a Scalar value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsScalar(42)));
testing::Matcher<Value> IsScalar(const Scalar &expected_scalar,
                                 Value::ContextType context_type);

/// Helper to match a LoadAddress value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsLoadAddress(0x1000)));
testing::Matcher<Value> IsLoadAddress(const Scalar &expected_address,
                                      Value::ContextType context_type);

/// Helper to match a FileAddress value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsFileAddress(Scalar(0x1000))));
testing::Matcher<Value> IsFileAddress(const Scalar &expected_address,
                                      Value::ContextType context_type);

/// Helper to match a HostAddress value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsHostValue({0x11, 0x22})));
testing::Matcher<Value> IsHostValue(const std::vector<uint8_t> &expected_bytes,
                                    Value::ContextType context_type);

/// Helper to create a scalar because Scalar's operator==() is really picky.
Scalar GetScalar(unsigned bits, uint64_t value, bool sign);

/// Helper that combines IsScalar with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectScalar(42));
llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
ExpectScalar(const Scalar &expected_scalar,
             Value::ContextType context_type = Value::ContextType::Invalid);

/// Helper that combines GetScalar with ExpectScalar to get a precise scalar.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectScalar(8, 42, true));
llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
ExpectScalar(unsigned bits, uint64_t value, bool sign,
             Value::ContextType context_type = Value::ContextType::Invalid);

/// Helper that combines IsLoadAddress with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectLoadAddress(0x1000));
llvm::detail::ValueMatchesPoly<testing::Matcher<Value>> ExpectLoadAddress(
    const Scalar &expected_address,
    Value::ContextType context_type = Value::ContextType::Invalid);

/// Helper that combines IsFileAddress with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectFileAddress(Scalar(0x2000)));
llvm::detail::ValueMatchesPoly<testing::Matcher<Value>> ExpectFileAddress(
    const Scalar &expected_address,
    Value::ContextType context_type = Value::ContextType::Invalid);

/// Helper that combines IsHostValue with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectHostAddress({0x11, 0x22}));
llvm::detail::ValueMatchesPoly<testing::Matcher<Value>> ExpectHostAddress(
    const std::vector<uint8_t> &expected_bytes,
    Value::ContextType context_type = Value::ContextType::Invalid);

} // namespace lldb_private

#endif // LLDB_UNITTESTS_EXPRESSION_VALUEMATCHER_H
