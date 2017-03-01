//===-- VASprintfTest.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/VASPrintf.h"
#include "llvm/ADT/SmallString.h"

#include "gtest/gtest.h"

#include <locale.h>

using namespace lldb_private;
using namespace llvm;

static bool Sprintf(llvm::SmallVectorImpl<char> &Buffer, const char *Fmt, ...) {
  va_list args;
  va_start(args, Fmt);
  bool Result = VASprintf(Buffer, Fmt, args);
  va_end(args);
  return Result;
}

TEST(VASprintfTest, NoBufferResize) {
  std::string TestStr("small");

  llvm::SmallString<32> BigBuffer;
  ASSERT_TRUE(Sprintf(BigBuffer, "%s", TestStr.c_str()));
  EXPECT_STREQ(TestStr.c_str(), BigBuffer.c_str());
  EXPECT_EQ(TestStr.size(), BigBuffer.size());
}

TEST(VASprintfTest, BufferResize) {
  std::string TestStr("bigger");
  llvm::SmallString<4> SmallBuffer;
  ASSERT_TRUE(Sprintf(SmallBuffer, "%s", TestStr.c_str()));
  EXPECT_STREQ(TestStr.c_str(), SmallBuffer.c_str());
  EXPECT_EQ(TestStr.size(), SmallBuffer.size());
}

TEST(VASprintfTest, EncodingError) {
  // Save the current locale first.
  std::string Current(::setlocale(LC_ALL, nullptr));

  setlocale(LC_ALL, ".932");

  wchar_t Invalid[2];
  Invalid[0] = 0x100;
  Invalid[1] = 0;
  llvm::SmallString<32> Buffer;
  EXPECT_FALSE(Sprintf(Buffer, "%ls", Invalid));
  EXPECT_EQ("<Encoding error>", Buffer);

  setlocale(LC_ALL, Current.c_str());
}
