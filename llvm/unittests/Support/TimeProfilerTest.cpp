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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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

TEST(TimeProfiler, Compression) {
  EXPECT_EQ(inferTimeTraceCompressionFromPath("foo.json"),
            TimeTraceCompression::None);
  EXPECT_EQ(inferTimeTraceCompressionFromPath("foo.json.zst"),
            TimeTraceCompression::Zstd);
  EXPECT_EQ(inferTimeTraceCompressionFromPath("foo.ZSTD"),
            TimeTraceCompression::Zstd);

  if (!compression::zstd::isAvailable())
    return;

  timeTraceProfilerInitialize(/*TimeTraceGranularity=*/0, "test",
                              /*TimeTraceVerbose=*/false,
                              TimeTraceCompression::Zstd);
  {
    TimeTraceScope Scope("compressed_event", "compressed_detail");
  }

  SmallVector<char, 0> CompressedChars;
  raw_svector_ostream OS(CompressedChars);
  timeTraceProfilerWrite(OS);
  timeTraceProfilerCleanup();

  ASSERT_FALSE(CompressedChars.empty());
  // Compressed output must not start with '{'.
  EXPECT_NE(CompressedChars.front(), '{');

  SmallString<128> TempPath;
  sys::fs::createUniquePath("time-trace-%%%%%%%.json.zst", TempPath, true);
  llvm::scope_exit CleanupFile([&]() { sys::fs::remove(TempPath); });

  // Default (TimeTraceCompression::Infer) infers Zstd from the .zst extension.
  timeTraceProfilerInitialize(/*TimeTraceGranularity=*/0, "test");
  {
    TimeTraceScope Scope("inferred_event", "inferred_detail");
  }
  ASSERT_FALSE(errorToBool(timeTraceProfilerWrite(TempPath, "fallback")));
  timeTraceProfilerCleanup();
  auto BufOrErr = MemoryBuffer::getFile(TempPath);
  ASSERT_TRUE(static_cast<bool>(BufOrErr));
  ASSERT_FALSE((*BufOrErr)->getBuffer().empty());
  EXPECT_NE((*BufOrErr)->getBuffer().front(), '{');

  // Explicit TimeTraceCompression::None overrides the .zst extension.
  timeTraceProfilerInitialize(/*TimeTraceGranularity=*/0, "test",
                              /*TimeTraceVerbose=*/false,
                              TimeTraceCompression::None);
  {
    TimeTraceScope Scope("uncompressed_event", "uncompressed_detail");
  }
  ASSERT_FALSE(errorToBool(timeTraceProfilerWrite(TempPath, "fallback")));
  timeTraceProfilerCleanup();
  BufOrErr = MemoryBuffer::getFile(TempPath);
  ASSERT_TRUE(static_cast<bool>(BufOrErr));
  ASSERT_FALSE((*BufOrErr)->getBuffer().empty());
  EXPECT_EQ((*BufOrErr)->getBuffer().front(), '{');
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

} // namespace
