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
#include "llvm/Support/JSON.h"
#include "gtest/gtest.h"
#include <chrono>
#include <thread>

using namespace llvm;

namespace {

void setupProfiler(unsigned Granularity = 0) {
  timeTraceProfilerInitialize(Granularity, "test");
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

TEST(TimeProfiler, Async_Begin_End_Smoke) {
  setupProfiler();

  auto *Profiler = timeTraceAsyncProfilerBegin("event", "detail");
  timeTraceProfilerEnd(Profiler);

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

TEST(TimeProfiler, Instant_Add_Smoke) {
  setupProfiler();

  timeTraceProfilerBegin("sync event", "sync detail");
  timeTraceAddInstantEvent("instant event", [&] { return "instant detail"; });
  timeTraceProfilerEnd();

  std::string json = teardownProfiler();
  ASSERT_TRUE(json.find(R"("name":"sync event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"sync detail")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("name":"instant event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"instant detail")") != std::string::npos);
}

TEST(TimeProfiler, Instant_Not_Added_Smoke) {
  setupProfiler();

  timeTraceAddInstantEvent("instant event", [&] { return "instant detail"; });

  std::string json = teardownProfiler();
  ASSERT_TRUE(json.find(R"("name":"instant event")") == std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"instant detail")") == std::string::npos);
}

TEST(TimeProfiler, Child_Clamped_Within_Parent) {
  setupProfiler(/*Granularity=*/0);

  for (int I = 0; I < 50; ++I) {
    TimeTraceScope Outer("outer", "");
    for (int J = 0; J < 5; ++J) {
      TimeTraceScope Inner("inner", "");
      timeTraceAddInstantEvent("instant", [&] { return ""; });
    }
  }

  std::string Json = teardownProfiler();
  Expected<json::Value> Root = json::parse(Json);
  ASSERT_TRUE(static_cast<bool>(Root));
  json::Array *TraceEvents = Root->getAsObject()->getArray("traceEvents");
  ASSERT_NE(TraceEvents, nullptr);

  struct Span {
    int64_t Start;
    int64_t End;
  };
  SmallVector<Span, 8> Children;
  for (json::Value &Val : *TraceEvents) {
    json::Object *Obj = Val.getAsObject();
    StringRef Ph = Obj->getString("ph").value_or("");
    StringRef Name = Obj->getString("name").value_or("");
    int64_t Ts = Obj->getInteger("ts").value_or(0);
    int64_t Dur = Obj->getInteger("dur").value_or(0);
    if (Ph == "i" && Name == "instant") {
      ASSERT_FALSE(Children.empty());
      EXPECT_GE(Ts, Children.back().Start);
      EXPECT_LE(Ts, Children.back().End);
    } else if (Ph == "X" && Name == "inner") {
      Children.push_back({Ts, Ts + Dur});
    } else if (Ph == "X" && Name == "outer") {
      int64_t PrevEnd = Ts;
      for (const Span &C : Children) {
        EXPECT_GE(C.Start, PrevEnd);
        EXPECT_LE(C.End, Ts + Dur);
        PrevEnd = C.End;
      }
      Children.clear();
    }
  }
}

} // namespace
