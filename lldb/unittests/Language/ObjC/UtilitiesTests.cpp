//===-- UtilitiesTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/ObjC/Utilities.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <string>

using namespace lldb_private;

// Try to format the date_value field of a NSDate.
static llvm::Optional<std::string> formatDateValue(double date_value) {
  StreamString s;
  bool succcess = formatters::NSDate::FormatDateValue(date_value, s);
  if (succcess)
    return std::string(s.GetData());
  return llvm::None;
}

TEST(DataFormatterMockTest, NSDate) {
  EXPECT_EQ(formatDateValue(-63114076800),
            std::string("0001-12-30 00:00:00 +0000"));

  // Can't convert the date_value to a time_t.
  EXPECT_EQ(formatDateValue((double)(std::numeric_limits<time_t>::max()) + 1),
            llvm::None);
  EXPECT_EQ(formatDateValue((double)(std::numeric_limits<time_t>::min()) - 1),
            llvm::None);

  // Can't add the macOS epoch to the converted date_value (the add overflows).
  EXPECT_EQ(formatDateValue((double)std::numeric_limits<time_t>::max()),
            llvm::None);
  EXPECT_EQ(formatDateValue((double)std::numeric_limits<time_t>::min()),
            llvm::None);

  // FIXME: The formatting result is wrong on Windows because we adjust the
  // epoch when _WIN32 is defined (see GetOSXEpoch).
#ifndef _WIN32
  EXPECT_TRUE(
      llvm::StringRef(*formatDateValue(0)).startswith("2001-01-01 00:00:00"));
#endif
}
