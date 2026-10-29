//===- TCPTestUtils.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fixtures for tests of the TCP primitives, which skip on machines that cannot
// run them. The probes are defined once per system under sys/<system>/.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_CONNECT_TCP_TCPTESTUTILS_H
#define ORC_RT_UNITTEST_CONNECT_TCP_TCPTESTUTILS_H

#include "gtest/gtest.h"

#include <optional>
#include <string>

namespace orc_rt::test {

enum class IPFamily { IPv4, IPv6 };

/// Returns why TCP over F's loopback address can't be tested on this machine,
/// or nullopt if it can.
///
/// Probes by listening on the loopback address with raw system calls rather
/// than the code under test, so a bug in that code fails its tests instead of
/// skipping them. Only failures that mean the machine does not offer loopback
/// TCP for F count: an unsupported address family, an unavailable address, or
/// a sandbox that forbids the bind. Any other failure is left for the tests to
/// hit and report.
std::optional<std::string> whyLoopbackUnavailable(IPFamily F);

/// Returns true if "localhost" resolves to at least one address of family F.
bool localhostResolvesTo(IPFamily F);

/// A fixture whose suite is skipped if TCP over F's loopback address is
/// unavailable (see whyLoopbackUnavailable). The probe runs once per suite.
template <IPFamily F> class LoopbackTCPTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (auto Why = whyLoopbackUnavailable(F))
      GTEST_SKIP() << *Why;
  }
};

} // namespace orc_rt::test

#endif // ORC_RT_UNITTEST_CONNECT_TCP_TCPTESTUTILS_H
