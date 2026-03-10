//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

/// Helpers for gtest printing.
namespace lldb_dap::protocol {

inline void PrintTo(const Request &req, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(req)).str();
}

inline void PrintTo(const Response &resp, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(resp)).str();
}

inline void PrintTo(const Event &evt, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(evt)).str();
}

inline void PrintTo(const Message &message, std::ostream *os) {
  return std::visit([os](auto &&message) { return PrintTo(message, os); },
                    message);
}

} // namespace lldb_dap::protocol

namespace lldb_dap_tests {

static constexpr llvm::StringLiteral k_linux_x86_64_binary =
    "linux-x86_64.out.yaml";
static constexpr llvm::StringLiteral k_linux_x86_64_core =
    "linux-x86_64.core.yaml";

/// A matcher for a DAP event.
template <typename EventMatcher, typename BodyMatcher>
inline testing::Matcher<const lldb_dap::protocol::Event &>
IsEvent(const EventMatcher &event_matcher, const BodyMatcher &body_matcher) {
  return testing::AllOf(
      testing::Field(&lldb_dap::protocol::Event::event, event_matcher),
      testing::Field(&lldb_dap::protocol::Event::body, body_matcher));
}

template <typename EventMatcher>
inline testing::Matcher<const lldb_dap::protocol::Event &>
IsEvent(const EventMatcher &event_matcher) {
  return testing::AllOf(
      testing::Field(&lldb_dap::protocol::Event::event, event_matcher),
      testing::Field(&lldb_dap::protocol::Event::body, std::nullopt));
}

/// Matches an "output" event.
inline auto Output(llvm::StringRef o, llvm::StringRef cat = "console") {
  return IsEvent("output",
                 testing::Optional(llvm::json::Value(
                     llvm::json::Object{{"category", cat}, {"output", o}})));
}

} // namespace lldb_dap_tests
