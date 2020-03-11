//===- ArchiveTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Archive.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace object;
using namespace testing;

static const char ArchiveWithMember[] = "!<arch>\n"        // Global header
                                        "test/           " // Member name
                                        "0           "     // Timestamp
                                        "0     "           // Owner ID
                                        "0     "           // Group ID
                                        "0       "         // File mode
                                        "9999999999"       // Size
                                        "`\n";

static const char ThinArchiveWithMember[] = "!<thin>\n"        // Global header
                                            "test/           " // Member name
                                            "0           "     // Timestamp
                                            "0     "           // Owner ID
                                            "0     "           // Group ID
                                            "0       "         // File mode
                                            "9999999999"       // Size
                                            "`\n";

struct ArchiveTestsFixture : Test {
  Expected<Archive::child_iterator> createChild(StringRef Src) {
    MemoryBufferRef Source(Src, "archive");
    Expected<std::unique_ptr<Archive>> AOrErr = Archive::create(Source);
    if (!AOrErr)
      return AOrErr.takeError();
    A = std::move(*AOrErr);

    Error ChildErr = Error::success();
    auto Child = A->child_begin(ChildErr);
    if (ChildErr)
      return std::move(ChildErr);
    return Child;
  }

  std::unique_ptr<Archive> A;
};

TEST_F(ArchiveTestsFixture, ArchiveChildGetSizeRegularArchive) {
  auto Child = createChild(ArchiveWithMember);
  ASSERT_THAT_EXPECTED(Child, Succeeded());

  Expected<uint64_t> Size = (*Child)->getSize();
  ASSERT_THAT_EXPECTED(Size, Succeeded());
  EXPECT_EQ(9999999999u, *Size);
}

TEST_F(ArchiveTestsFixture, ArchiveChildGetSizeThinArchive) {
  auto Child = createChild(ThinArchiveWithMember);
  ASSERT_THAT_EXPECTED(Child, Succeeded());

  Expected<uint64_t> Size = (*Child)->getSize();
  ASSERT_THAT_EXPECTED(Size, Succeeded());
  EXPECT_EQ(9999999999u, *Size);
}

TEST_F(ArchiveTestsFixture, ArchiveChildGetBuffer) {
  auto Child = createChild(ArchiveWithMember);
  ASSERT_THAT_EXPECTED(Child, Succeeded());

  Expected<StringRef> Buffer = (*Child)->getBuffer();
  // Cannot use ASSERT_THAT_EXPECTED, as that will attempt to print the
  // StringRef (which has an invalid size).
  ASSERT_TRUE((bool)Buffer);
  EXPECT_EQ(9999999999u, Buffer->size());
  EXPECT_EQ(ArchiveWithMember + sizeof(ArchiveWithMember) - 1, Buffer->data());
}
