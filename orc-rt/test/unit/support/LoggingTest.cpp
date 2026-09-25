//===- LoggingTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the logging-backend-independent parts of
// orc-rt-c/support/Logging.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/support/Logging.h"

#include "gtest/gtest.h"

#include <string>
#include <unordered_set>

namespace {

// Every level and category must compile and be usable as a plain statement,
// with any number of printf-style arguments. This holds for all backends.
TEST(LoggingTest, CompilesAtEveryLevel) {
  ORC_RT_LOG(Error, General, "no format args");
  ORC_RT_LOG(Warning, General, "one arg: %d", 42);
  ORC_RT_LOG(Info, General, "two args: %s = %d", "answer", 42);
  ORC_RT_LOG(Debug, General, "wide arg: %llu", (unsigned long long)1 << 40);

  // ORC_RT_LOG_PUB_S must concatenate into the literal format and type-check as
  // a "%s" conversion against a runtime string on every backend.
  const char *RuntimeStr = "runtime";
  ORC_RT_LOG(Info, General, "public string: " ORC_RT_LOG_PUB_S, RuntimeStr);
  SUCCEED();
}

TEST(LoggingTest, LevelGetName) {
  EXPECT_STREQ("DEBUG", orc_rt_log_Level_getName(ORC_RT_LOG_LEVEL_DEBUG));
  EXPECT_STREQ("INFO", orc_rt_log_Level_getName(ORC_RT_LOG_LEVEL_INFO));
  EXPECT_STREQ("WARNING", orc_rt_log_Level_getName(ORC_RT_LOG_LEVEL_WARNING));
  EXPECT_STREQ("ERROR", orc_rt_log_Level_getName(ORC_RT_LOG_LEVEL_ERROR));
  EXPECT_STREQ("OFF", orc_rt_log_Level_getName(ORC_RT_LOG_LEVEL_OFF));

  // Out-of-range levels have no name.
  EXPECT_EQ(nullptr, orc_rt_log_Level_getName(-1));
  EXPECT_EQ(nullptr, orc_rt_log_Level_getName(ORC_RT_LOG_LEVEL_COUNT));
  EXPECT_EQ(nullptr, orc_rt_log_Level_getName(100));
}

TEST(LoggingTest, LevelParse) {
  EXPECT_EQ(ORC_RT_LOG_LEVEL_DEBUG, orc_rt_log_Level_parse("debug"));
  EXPECT_EQ(ORC_RT_LOG_LEVEL_INFO, orc_rt_log_Level_parse("info"));
  EXPECT_EQ(ORC_RT_LOG_LEVEL_WARNING, orc_rt_log_Level_parse("warning"));
  EXPECT_EQ(ORC_RT_LOG_LEVEL_ERROR, orc_rt_log_Level_parse("error"));
  EXPECT_EQ(ORC_RT_LOG_LEVEL_OFF, orc_rt_log_Level_parse("off"));

  // Parsing is case-insensitive.
  EXPECT_EQ(ORC_RT_LOG_LEVEL_INFO, orc_rt_log_Level_parse("INFO"));
  EXPECT_EQ(ORC_RT_LOG_LEVEL_WARNING, orc_rt_log_Level_parse("WaRnInG"));

  // Unrecognized names, including prefixes/superstrings of valid ones and the
  // empty string, return -1.
  EXPECT_EQ(-1, orc_rt_log_Level_parse(""));
  EXPECT_EQ(-1, orc_rt_log_Level_parse("inf"));
  EXPECT_EQ(-1, orc_rt_log_Level_parse("infox"));
  EXPECT_EQ(-1, orc_rt_log_Level_parse("bogus"));
}

TEST(LoggingTest, LevelParseGetNameRoundTrip) {
  // Every level's canonical name parses back to that level.
  for (orc_rt_log_Level Level = ORC_RT_LOG_LEVEL_DEBUG;
       Level != ORC_RT_LOG_LEVEL_COUNT; ++Level) {
    const char *Name = orc_rt_log_Level_getName(Level);
    ASSERT_NE(nullptr, Name) << "level " << Level << " has no name";
    EXPECT_EQ(Level, orc_rt_log_Level_parse(Name)) << "round trip for " << Name;
  }
}

TEST(LoggingTest, CategoryNamesAreUniqueAndNonNull) {
  // Check that every category has a unique, non-null name.
  std::unordered_set<std::string> Seen;
  for (int C = orc_rt_log_Category_General; C != orc_rt_log_Category_Count;
       ++C) {
    const char *Name =
        orc_rt_log_Category_getName(static_cast<orc_rt_log_Category>(C));
    ASSERT_NE(Name, nullptr) << "category " << C << " has no name";
    EXPECT_TRUE(Seen.insert(Name).second)
        << "category " << C << " has a duplicate name: " << Name;
  }
}

TEST(LoggingTest, OutOfRangeCategoryNamesAreNull) {
  // The Count sentinel is not a real category, and out-of-range values have no
  // name.
  EXPECT_EQ(nullptr, orc_rt_log_Category_getName(orc_rt_log_Category_Count));
  EXPECT_EQ(nullptr, orc_rt_log_Category_getName((orc_rt_log_Category)-1));
}

// ORC_RT_LOG_ENABLED must be usable in preprocessor conditionals, and must
// follow the level ordering: if a level is compiled in, so is every level
// above it.
#if ORC_RT_LOG_ENABLED(Debug) && !ORC_RT_LOG_ENABLED(Info)
#error "ORC_RT_LOG_ENABLED(Debug) implies ORC_RT_LOG_ENABLED(Info)"
#endif
#if ORC_RT_LOG_ENABLED(Info) && !ORC_RT_LOG_ENABLED(Warning)
#error "ORC_RT_LOG_ENABLED(Info) implies ORC_RT_LOG_ENABLED(Warning)"
#endif
#if ORC_RT_LOG_ENABLED(Warning) && !ORC_RT_LOG_ENABLED(Error)
#error "ORC_RT_LOG_ENABLED(Warning) implies ORC_RT_LOG_ENABLED(Error)"
#endif
#if ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_NONE && ORC_RT_LOG_ENABLED(Error)
#error "The none backend compiles every level out"
#endif

#if ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_NONE ||                           \
    ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_PRINTF

// Increments Count and returns a printable value, so a test can observe whether
// a log macro evaluated it.
static const char *bumpAndReturn(int &Count) {
  ++Count;
  return "arg";
}

// ORC_RT_LOG_ENABLED must agree with what ORC_RT_LOG actually does: a
// compiled-in log site evaluates its arguments, a compiled-out one doesn't.
// (Not checked for os_log: os_log_with_type only evaluates its arguments if
// the type is enabled at runtime, which depends on system configuration.)
TEST(LoggingTest, EnabledMatchesCompiledInLevels) {
  int NumEvals = 0;
  ORC_RT_LOG(Error, General, "%s", bumpAndReturn(NumEvals));
  EXPECT_EQ(NumEvals, ORC_RT_LOG_ENABLED(Error));

  NumEvals = 0;
  ORC_RT_LOG(Warning, General, "%s", bumpAndReturn(NumEvals));
  EXPECT_EQ(NumEvals, ORC_RT_LOG_ENABLED(Warning));

  NumEvals = 0;
  ORC_RT_LOG(Info, General, "%s", bumpAndReturn(NumEvals));
  EXPECT_EQ(NumEvals, ORC_RT_LOG_ENABLED(Info));

  NumEvals = 0;
  ORC_RT_LOG(Debug, General, "%s", bumpAndReturn(NumEvals));
  EXPECT_EQ(NumEvals, ORC_RT_LOG_ENABLED(Debug));
}

#endif

#if ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_NONE

// The none backend compiles log sites out entirely, so their arguments must
// never be evaluated.
TEST(LoggingTest, NoneBackendDoesNotEvaluateArguments) {
  int NumEvals = 0;

  // Sanity check: a direct call is evaluated and bumps the counter.
  bumpAndReturn(NumEvals);
  ASSERT_EQ(NumEvals, 1);

  // The log argument, by contrast, must not run under the none backend.
  NumEvals = 0;
  ORC_RT_LOG(Error, General, "%s", bumpAndReturn(NumEvals));
  EXPECT_EQ(NumEvals, 0) << "the none backend must not evaluate log arguments";
}

#endif

} // namespace
