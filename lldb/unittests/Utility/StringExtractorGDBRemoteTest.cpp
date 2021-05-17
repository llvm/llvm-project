#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <limits.h>

#include "lldb/Utility/StringExtractorGDBRemote.h"
#include "lldb/lldb-defines.h"

TEST(StringExtractorGDBRemoteTest, GetPidTid) {
  StringExtractorGDBRemote ex("");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  // invalid/short values

  ex.Reset("narf");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset(";1234");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset(".1234");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("pnarf");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p;1234");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p.1234");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p1234.");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p1234.;1234");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("-2");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p1234.-2");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p-2");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p-2.1234");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  // overflow

  ex.Reset("p10000000000000000");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p10000000000000000.0");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("10000000000000000");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p0.10000000000000000");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  ex.Reset("p10000000000000000.10000000000000000");
  EXPECT_EQ(ex.GetPidTid(0), llvm::None);

  // invalid: all processes but specific thread

  ex.Reset("p-1.0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p-1.1234");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p-1.123456789ABCDEF0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  // unsupported: pid/tid 0

  ex.Reset("0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p0.0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p0.-1");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p0.1234");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p0.123456789ABCDEF0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p1234.0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  ex.Reset("p123456789ABCDEF0.0");
  EXPECT_EQ(ex.GetPidTid(100), llvm::None);

  // pure thread id

  ex.Reset("-1");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(100, StringExtractorGDBRemote::AllThreads));

  ex.Reset("1234");
  EXPECT_THAT(ex.GetPidTid(100).getValue(), ::testing::Pair(100, 0x1234ULL));

  ex.Reset("123456789ABCDEF0");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(100, 0x123456789ABCDEF0ULL));

  // pure process id

  ex.Reset("p-1");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(StringExtractorGDBRemote::AllProcesses,
                              StringExtractorGDBRemote::AllThreads));

  ex.Reset("p1234");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x1234ULL, StringExtractorGDBRemote::AllThreads));

  ex.Reset("p123456789ABCDEF0");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x123456789ABCDEF0ULL,
                              StringExtractorGDBRemote::AllThreads));

  ex.Reset("pFFFFFFFFFFFFFFFF");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(StringExtractorGDBRemote::AllProcesses,
                              StringExtractorGDBRemote::AllThreads));

  // combined thread id + process id

  ex.Reset("p-1.-1");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(StringExtractorGDBRemote::AllProcesses,
                              StringExtractorGDBRemote::AllThreads));

  ex.Reset("p1234.-1");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x1234ULL, StringExtractorGDBRemote::AllThreads));

  ex.Reset("p1234.123456789ABCDEF0");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x1234ULL, 0x123456789ABCDEF0ULL));

  ex.Reset("p123456789ABCDEF0.-1");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x123456789ABCDEF0ULL,
                              StringExtractorGDBRemote::AllThreads));

  ex.Reset("p123456789ABCDEF0.1234");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x123456789ABCDEF0ULL, 0x1234ULL));

  ex.Reset("p123456789ABCDEF0.123456789ABCDEF0");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x123456789ABCDEF0ULL, 0x123456789ABCDEF0ULL));

  ex.Reset("p123456789ABCDEF0.123456789ABCDEF0");
  EXPECT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x123456789ABCDEF0ULL, 0x123456789ABCDEF0ULL));
}

TEST(StringExtractorGDBRemoteTest, GetPidTidMultipleValues) {
  StringExtractorGDBRemote ex("1234;p12;p1234.-1");
  ASSERT_THAT(ex.GetPidTid(100).getValue(), ::testing::Pair(100, 0x1234ULL));
  ASSERT_EQ(ex.GetChar(), ';');
  ASSERT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x12ULL, StringExtractorGDBRemote::AllThreads));
  ASSERT_EQ(ex.GetChar(), ';');
  ASSERT_THAT(ex.GetPidTid(100).getValue(),
              ::testing::Pair(0x1234ULL, StringExtractorGDBRemote::AllThreads));
}
