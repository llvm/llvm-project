//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueMatcher.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;

static void FormatValueDetails(llvm::raw_ostream &os,
                               Value::ValueType value_type,
                               Value::ContextType context_type,
                               const Scalar &scalar,
                               llvm::ArrayRef<uint8_t> buffer_data) {
  os << "Value(";
  os << "value_type=" << Value::GetValueTypeAsCString(value_type);
  os << ", context_type=" << Value::GetContextTypeAsCString(context_type);

  if (value_type == Value::ValueType::HostAddress) {
    auto bytes_to_print = buffer_data.take_front(16);
    os << ", buffer=[";
    llvm::interleave(
        bytes_to_print,
        [&](uint8_t byte) {
          os << llvm::format("%02x", static_cast<unsigned>(byte));
        },
        [&]() { os << " "; });
    if (buffer_data.size() > 16)
      os << " ...";
    os << "] (" << buffer_data.size() << " bytes)";
  } else {
    os << ", value=" << scalar;
  }
  os << ")";
}

void lldb_private::PrintTo(const Value &val, std::ostream *os) {
  if (!os)
    return;

  llvm::raw_os_ostream raw_os(*os);
  FormatValueDetails(raw_os, val.GetValueType(), val.GetContextType(),
                     val.GetScalar(), val.GetBuffer().GetData());
}

bool ValueMatcher::MatchAndExplain(const Value &val,
                                   std::ostream *stream) const {
  if (stream) {
    llvm::raw_os_ostream os(*stream);
    return MatchAndExplainImpl(val, os);
  }

  llvm::raw_null_ostream os;
  return MatchAndExplainImpl(val, os);
}

// Match the provided value and explain any mismatches using
// the raw_ostream. We use the llvm::raw_ostream here to simplify the formatting
// of Scalar values which already know how to print themselves to that stream.
bool ValueMatcher::MatchAndExplainImpl(const Value &val,
                                       llvm::raw_ostream &os) const {
  if (val.GetValueType() != m_value_type) {
    os << "value_type mismatch: expected "
       << Value::GetValueTypeAsCString(m_value_type) << ", got "
       << Value::GetValueTypeAsCString(val.GetValueType()) << " ";
    return false;
  }

  if (val.GetContextType() != m_context_type) {
    os << "context_type mismatch: expected "
       << Value::GetContextTypeAsCString(m_context_type) << ", got "
       << Value::GetContextTypeAsCString(val.GetContextType()) << " ";
    return false;
  }

  if (m_value_type == Value::ValueType::HostAddress) {
    const DataBufferHeap &buffer = val.GetBuffer();
    const size_t buffer_size = buffer.GetByteSize();
    if (buffer_size != m_expected_bytes.size()) {
      os << "buffer size mismatch: expected " << m_expected_bytes.size()
         << ", got " << buffer_size << " ";
      return false;
    }

    const uint8_t *data = buffer.GetBytes();
    for (size_t i = 0; i < buffer_size; ++i) {
      if (data[i] != m_expected_bytes[i]) {
        os << "byte mismatch at index " << i << ": expected "
           << llvm::format("0x%02x", static_cast<unsigned>(m_expected_bytes[i]))
           << ", got " << llvm::format("0x%02x", static_cast<unsigned>(data[i]))
           << " ";
        return false;
      }
    }
  } else {
    // For Scalar, FileAddress, and LoadAddress compare m_value.
    const Scalar &actual_scalar = val.GetScalar();
    if (actual_scalar != m_expected_scalar) {
      os << "scalar value mismatch: expected " << m_expected_scalar << ", got "
         << actual_scalar;
      return false;
    }
  }

  return true;
}

void ValueMatcher::DescribeTo(std::ostream *os) const {
  if (!os)
    return;
  llvm::raw_os_ostream raw_os(*os);
  FormatValueDetails(raw_os, m_value_type, m_context_type, m_expected_scalar,
                     m_expected_bytes);
}

void ValueMatcher::DescribeNegationTo(std::ostream *os) const {
  if (!os)
    return;
  *os << "value does not match";
}

testing::Matcher<Value>
lldb_private::MatchScalarValue(Value::ValueType value_type,
                               const Scalar &expected_scalar,
                               Value::ContextType context_type) {
  return ValueMatcher(value_type, expected_scalar, context_type);
}

testing::Matcher<Value>
lldb_private::MatchHostValue(Value::ValueType value_type,
                             const std::vector<uint8_t> &expected_bytes,
                             Value::ContextType context_type) {
  return ValueMatcher(value_type, expected_bytes, context_type);
}

testing::Matcher<Value>
lldb_private::IsScalar(const Scalar &expected_scalar,
                       Value::ContextType context_type) {
  return MatchScalarValue(Value::ValueType::Scalar, expected_scalar,
                          context_type);
}

testing::Matcher<Value>
lldb_private::IsLoadAddress(const Scalar &expected_address,
                            Value::ContextType context_type) {
  return MatchScalarValue(Value::ValueType::LoadAddress, expected_address,
                          context_type);
}

testing::Matcher<Value>
lldb_private::IsFileAddress(const Scalar &expected_address,
                            Value::ContextType context_type) {
  return MatchScalarValue(Value::ValueType::FileAddress, expected_address,
                          context_type);
}

testing::Matcher<Value>
lldb_private::IsHostValue(const std::vector<uint8_t> &expected_bytes,
                          Value::ContextType context_type) {
  return MatchHostValue(Value::ValueType::HostAddress, expected_bytes,
                        context_type);
}

Scalar lldb_private::GetScalar(unsigned bits, uint64_t value, bool sign) {
  Scalar scalar(value);
  scalar.TruncOrExtendTo(bits, sign);
  return scalar;
}

llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
lldb_private::ExpectScalar(const Scalar &expected_scalar,
                           Value::ContextType context_type) {
  return llvm::HasValue(IsScalar(expected_scalar, context_type));
}

llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
lldb_private::ExpectScalar(unsigned bits, uint64_t value, bool sign,
                           Value::ContextType context_type) {
  return ExpectScalar(GetScalar(bits, value, sign), context_type);
}

llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
lldb_private::ExpectLoadAddress(const Scalar &expected_address,
                                Value::ContextType context_type) {
  return llvm::HasValue(IsLoadAddress(expected_address, context_type));
}

llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
lldb_private::ExpectFileAddress(const Scalar &expected_address,
                                Value::ContextType context_type) {
  return llvm::HasValue(IsFileAddress(expected_address, context_type));
}

llvm::detail::ValueMatchesPoly<testing::Matcher<Value>>
lldb_private::ExpectHostAddress(const std::vector<uint8_t> &expected_bytes,
                                Value::ContextType context_type) {
  return llvm::HasValue(IsHostValue(expected_bytes, context_type));
}
