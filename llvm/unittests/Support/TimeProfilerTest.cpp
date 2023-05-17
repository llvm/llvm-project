//===- unittests/TimeProfilerTest.cpp - TimeProfiler tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// These are bare-minimum 'smoke' tests of the time profiler. Not tested:
//  - multi-threading
//  - 'Total' entries
//  - elision of short or ill-formed entries
//  - detail callback
//  - no calls to now() if profiling is disabled
//  - suppression of contributions to total entries for nested entries
//===----------------------------------------------------------------------===//

#include "llvm/Support/TimeProfiler.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

void setupProfiler() {
  timeTraceProfilerInitialize(/*TimeTraceGranularity=*/0, "test");
}

std::string teardownProfiler() {
  SmallVector<char, 1024> smallVector;
  raw_svector_ostream os(smallVector);
  timeTraceProfilerWrite(os);
  timeTraceProfilerCleanup();
  return os.str().str();
}

TEST(TimeProfiler, Scope_Smoke) {
  setupProfiler();

  { TimeTraceScope scope("event", "detail"); }

  std::string json = teardownProfiler();
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Begin_End_Smoke) {
  setupProfiler();

  timeTraceProfilerBegin("event", "detail");
  timeTraceProfilerEnd();

  std::string json = teardownProfiler();
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Begin_End_Disabled) {
  // Nothing should be observable here. The test is really just making sure
  // we've not got a stray nullptr deref.
  timeTraceProfilerBegin("event", "detail");
  timeTraceProfilerEnd();
}

} // namespace
