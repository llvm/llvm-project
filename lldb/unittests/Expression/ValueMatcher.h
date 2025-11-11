//===-- ValueMatcher.h ----------------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_EXPRESSION_VALUEMATCHER_H
#define LLDB_UNITTESTS_EXPRESSION_VALUEMATCHER_H

#include "lldb/Core/Value.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <iomanip>
#include <vector>

namespace lldb_private {

/// Helper function to format Value details to an ostream.
///
/// This is used for format the details coming from either a Value directly
/// or the parts stored in the ValueMatcher object.
inline void FormatValueDetails(std::ostream &os, Value::ValueType value_type,
                               Value::ContextType context_type,
                               const Scalar &scalar,
                               llvm::ArrayRef<uint8_t> buffer_data) {
  os << "Value(";
  os << "value_type=" << Value::GetValueTypeAsCString(value_type);
  os << ", context_type=" << Value::GetContextTypeAsCString(context_type);

  if (value_type == Value::ValueType::HostAddress) {
    os << ", buffer=[";
    for (size_t i = 0; i < std::min(buffer_data.size(), size_t(16)); ++i) {
      if (i > 0)
        os << " ";
      os << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(buffer_data[i]);
    }
    if (buffer_data.size() > 16) {
      os << " ...";
    }
    os << std::dec << "] (" << buffer_data.size() << " bytes)";
  } else {
    std::string scalar_str;
    llvm::raw_string_ostream scalar_os(scalar_str);
    scalar_os << scalar;
    os << ", value=" << scalar_os.str();
  }
  os << ")";
}

/// Custom printer for Value objects to make test failures more readable.
inline void PrintTo(const Value &val, std::ostream *os) {
  if (!os)
    return;

  FormatValueDetails(*os, val.GetValueType(), val.GetContextType(),
                     val.GetScalar(), val.GetBuffer().GetData());
}

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

  bool MatchAndExplain(const Value &val,
                       testing::MatchResultListener *listener) const {
    if (val.GetValueType() != m_value_type) {
      *listener << "value_type mismatch: expected "
                << Value::GetValueTypeAsCString(m_value_type) << ", got "
                << Value::GetValueTypeAsCString(val.GetValueType()) << " ";
      return false;
    }

    if (val.GetContextType() != m_context_type) {
      *listener << "context_type mismatch: expected "
                << Value::GetContextTypeAsCString(m_context_type) << ", got "
                << Value::GetContextTypeAsCString(val.GetContextType()) << " ";
      return false;
    }

    if (m_value_type == Value::ValueType::HostAddress) {
      const DataBufferHeap &buffer = val.GetBuffer();
      const size_t buffer_size = buffer.GetByteSize();
      if (buffer_size != m_expected_bytes.size()) {
        *listener << "buffer size mismatch: expected "
                  << m_expected_bytes.size() << ", got " << buffer_size << " ";
        return false;
      }

      const uint8_t *data = buffer.GetBytes();
      for (size_t i = 0; i < buffer_size; ++i) {
        if (data[i] != m_expected_bytes[i]) {
          *listener << "byte mismatch at index " << i << ": expected "
                    << "0x" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(m_expected_bytes[i]) << ", got "
                    << "0x" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(data[i]) << " ";
          return false;
        }
      }
    } else {
      // For Scalar, FileAddress, and LoadAddress - compare m_value
      const Scalar &actual_scalar = val.GetScalar();
      if (actual_scalar != m_expected_scalar) {
        std::string expected_str, actual_str;
        llvm::raw_string_ostream expected_os(expected_str);
        llvm::raw_string_ostream actual_os(actual_str);
        expected_os << m_expected_scalar;
        actual_os << actual_scalar;
        *listener << "scalar value mismatch: expected " << expected_os.str()
                  << ", got " << actual_os.str() << " ";
        return false;
      }
    }

    return true;
  }

  void DescribeTo(std::ostream *os) const {
    if (!os)
      return;
    FormatValueDetails(*os, m_value_type, m_context_type, m_expected_scalar,
                       m_expected_bytes);
  }

  void DescribeNegationTo(std::ostream *os) const {
    if (!os)
      return;
    *os << "value does not match";
  }

private:
  Value::ValueType m_value_type = Value::ValueType::Invalid;
  Value::ContextType m_context_type = Value::ContextType::Invalid;
  Scalar m_expected_scalar{};
  std::vector<uint8_t> m_expected_bytes{};
};

/// Matcher for Value with Scalar, FileAddress, or LoadAddress types.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(MatchScalarValue(...)));
inline testing::Matcher<Value>
MatchScalarValue(Value::ValueType value_type, const Scalar &expected_scalar,
                 Value::ContextType context_type) {
  return ValueMatcher(value_type, expected_scalar, context_type);
}

/// Matcher for Value with HostAddress type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(MatchHostValue(...)));
inline testing::Matcher<Value>
MatchHostValue(Value::ValueType value_type,
               const std::vector<uint8_t> &expected_bytes,
               Value::ContextType context_type) {
  return ValueMatcher(value_type, expected_bytes, context_type);
}

/// Helper to match a Scalar value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsScalar(42)));
inline testing::Matcher<Value> IsScalar(const Scalar &expected_scalar,
                                        Value::ContextType context_type) {
  return MatchScalarValue(Value::ValueType::Scalar, expected_scalar,
                          context_type);
}

/// Helper to match a LoadAddress value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsLoadAddress(0x1000)));
inline testing::Matcher<Value> IsLoadAddress(const Scalar &expected_address,
                                             Value::ContextType context_type) {
  return MatchScalarValue(Value::ValueType::LoadAddress, expected_address,
                          context_type);
}

/// Helper to match a FileAddress value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsFileAddress(Scalar(0x1000))));
inline testing::Matcher<Value> IsFileAddress(const Scalar &expected_address,
                                             Value::ContextType context_type) {
  return MatchScalarValue(Value::ValueType::FileAddress, expected_address,
                          context_type);
}

/// Helper to match a HostAddress value and context type.
/// Use with llvm::HasValue() to match Expected<Value>:
/// EXPECT_THAT_EXPECTED(result, llvm::HasValue(IsHostValue({0x11, 0x22})));
inline testing::Matcher<Value>
IsHostValue(const std::vector<uint8_t> &expected_bytes,
            Value::ContextType context_type) {
  return MatchHostValue(Value::ValueType::HostAddress, expected_bytes,
                        context_type);
}

/// Helper to create a scalar because Scalar's operator==() is really picky.
inline Scalar GetScalar(unsigned bits, uint64_t value, bool sign) {
  Scalar scalar(value);
  scalar.TruncOrExtendTo(bits, sign);
  return scalar;
}

/// Helper that combines IsScalar with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectScalar(42));
inline auto
ExpectScalar(const Scalar &expected_scalar,
             Value::ContextType context_type = Value::ContextType::Invalid) {
  return llvm::HasValue(IsScalar(expected_scalar, context_type));
}

/// Helper that combines GetScalar with ExpectScalar to get a precise scalar.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectScalar(8, 42, true));
inline auto
ExpectScalar(unsigned bits, uint64_t value, bool sign,
             Value::ContextType context_type = Value::ContextType::Invalid) {
  return ExpectScalar(GetScalar(bits, value, sign), context_type);
}

/// Helper that combines IsLoadAddress with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectLoadAddress(0x1000));
inline auto ExpectLoadAddress(
    const Scalar &expected_address,
    Value::ContextType context_type = Value::ContextType::Invalid) {
  return llvm::HasValue(IsLoadAddress(expected_address, context_type));
}

/// Helper that combines IsFileAddress with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectFileAddress(Scalar(0x2000)));
inline auto ExpectFileAddress(
    const Scalar &expected_address,
    Value::ContextType context_type = Value::ContextType::Invalid) {
  return llvm::HasValue(IsFileAddress(expected_address, context_type));
}

/// Helper that combines IsHostValue with llvm::HasValue for Expected<Value>.
/// Use it on an Expected<Value> like this:
/// EXPECT_THAT_EXPECTED(result, ExpectHostAddress({0x11, 0x22}));
inline auto ExpectHostAddress(
    const std::vector<uint8_t> &expected_bytes,
    Value::ContextType context_type = Value::ContextType::Invalid) {
  return llvm::HasValue(IsHostValue(expected_bytes, context_type));
}

} // namespace lldb_private

#endif // LLDB_UNITTESTS_EXPRESSION_VALUEMATCHER_H
