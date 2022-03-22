//===-- TraceGDBRemotePacketsTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"

#include "gtest/gtest.h"

#include <limits>

using namespace lldb_private;
using namespace llvm;

// Test serialization and deserialization of a non-empty
// TraceIntelPTGetStateResponse.
TEST(TraceGDBRemotePacketsTest, IntelPTGetStateResponse) {
  // This test works as follows:
  //  1. Create a non-empty TraceIntelPTGetStateResponse
  //  2. Serialize to JSON
  //  3. Deserialize the serialized JSON value
  //  4. Ensure the original value and the deserialized value are equivalent
  //
  //  Notes:
  //    - We intentionally set an integer value out of its signed range
  //      to ensure the serialization/deserialization isn't lossy since JSON
  //      operates on signed values

  // Choose arbitrary values for time_mult and time_shift
  uint32_t test_time_mult = 1076264588;
  uint16_t test_time_shift = 31;
  // Intentionally set time_zero value out of the signed type's range.
  uint64_t test_time_zero =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;

  // Create TraceIntelPTGetStateResponse.
  TraceIntelPTGetStateResponse response;
  response.tsc_conversion = std::make_unique<LinuxPerfZeroTscConversion>(
      test_time_mult, test_time_shift, test_time_zero);

  // Serialize then deserialize.
  Expected<TraceIntelPTGetStateResponse> deserialized_response =
      json::parse<TraceIntelPTGetStateResponse>(
          llvm::formatv("{0}", toJSON(response)).str(),
          "TraceIntelPTGetStateResponse");
  if (!deserialized_response)
    FAIL() << toString(deserialized_response.takeError());

  // Choose arbitrary TSC value to test the Convert function.
  const uint64_t TSC = std::numeric_limits<uint32_t>::max();
  // Expected nanosecond value pre calculated using the TSC to wall time
  // conversion formula located in the time_zero section of
  // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  const uint64_t EXPECTED_NANOS = 9223372039007304983u;

  uint64_t pre_serialization_conversion =
      response.tsc_conversion->Convert(TSC).count();
  uint64_t post_serialization_conversion =
      deserialized_response->tsc_conversion->Convert(TSC).count();

  // Check equality:
  // Ensure that both the TraceGetStateResponse and TraceIntelPTGetStateResponse
  // portions of the JSON representation are unchanged.
  ASSERT_EQ(toJSON(response), toJSON(*deserialized_response));
  // Ensure the result of the Convert function is unchanged.
  ASSERT_EQ(EXPECTED_NANOS, pre_serialization_conversion);
  ASSERT_EQ(EXPECTED_NANOS, post_serialization_conversion);
}

// Test serialization and deserialization of an empty
// TraceIntelPTGetStateResponse.
TEST(TraceGDBRemotePacketsTest, IntelPTGetStateResponseEmpty) {
  // This test works as follows:
  //  1. Create an empty TraceIntelPTGetStateResponse
  //  2. Serialize to JSON
  //  3. Deserialize the serialized JSON value
  //  4. Ensure the original value and the deserialized value are equivalent

  // Create TraceIntelPTGetStateResponse.
  TraceIntelPTGetStateResponse response;

  // Serialize then deserialize.
  Expected<TraceIntelPTGetStateResponse> deserialized_response =
      json::parse<TraceIntelPTGetStateResponse>(
          llvm::formatv("{0}", toJSON(response)).str(),
          "TraceIntelPTGetStateResponse");
  if (!deserialized_response)
    FAIL() << toString(deserialized_response.takeError());

  // Check equality:
  // Ensure that both the TraceGetStateResponse and TraceIntelPTGetStateResponse
  // portions of the JSON representation are unchanged.
  ASSERT_EQ(toJSON(response), toJSON(*deserialized_response));
  // Ensure that the tsc_conversion's are nullptr.
  ASSERT_EQ(response.tsc_conversion.get(), nullptr);
  ASSERT_EQ(response.tsc_conversion.get(),
            deserialized_response->tsc_conversion.get());
}
