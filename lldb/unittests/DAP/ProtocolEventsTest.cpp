//===-- ProtocolTypesTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
#include "Protocol/ProtocolTypes.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::pp;

TEST(ProtocolEventsTest, CapabilitiesEventBody) {
  Capabilities capabilities;
  capabilities.supportedFeatures = {
      eAdapterFeatureANSIStyling,
      eAdapterFeatureBreakpointLocationsRequest,
  };
  CapabilitiesEventBody body;
  body.capabilities = capabilities;
  StringRef json = R"({
  "capabilities": {
    "supportsANSIStyling": true,
    "supportsBreakpointLocationsRequest": true
  }
})";
  // Validate toJSON
  EXPECT_EQ(json, pp(body));
}
