//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/CAS/OnDiskCommon.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

static std::error_code listDir(const Twine &Dir,
                               SmallVectorImpl<std::string> &Files) {
  std::error_code EC;
  for (sys::fs::directory_iterator I(Dir, EC), E; I != E && !EC;
       I.increment(EC)) {
    // Only regular files (skip subdirectories, etc.)
    ErrorOr<sys::fs::basic_file_status> Status = I->status();
    if (!Status)
      continue;
    if (Status->type() == sys::fs::file_type::regular_file)
      Files.push_back(sys::path::filename(I->path()).str());
  }
  return EC;
}

TEST(UniqueTempFileTest, Basic) {
  unittest::TempFile TmpFile("somefile.o", /*Suffix=*/"", /*Contents=*/"hello",
                             /*Unique=*/true);
  unittest::TempDir TmpDir("UniqueTempFileTest", /*Unique=*/true);

  {
    {
      ondisk::UniqueTempFile UniqueTmp;
      StringRef TmpPath;
      ASSERT_THAT_ERROR(
          UniqueTmp.createAndCopyFrom(TmpDir.path(), TmpFile.path())
              .moveInto(TmpPath),
          Succeeded());
    }
    SmallVector<std::string, 2> DirFiles;
    std::error_code EC = listDir(TmpDir.path(), DirFiles);
    ASSERT_FALSE(EC) << EC.message();
    EXPECT_THAT(DirFiles, testing::IsEmpty());
  }

  {
    {
      ondisk::UniqueTempFile UniqueTmp;
      StringRef TmpPath;
      ASSERT_THAT_ERROR(
          UniqueTmp.createAndCopyFrom(TmpDir.path(), TmpFile.path())
              .moveInto(TmpPath),
          Succeeded());
      SmallString<256> NewPath(TmpDir.path());
      sys::path::append(NewPath, "otherfile.o");
      ASSERT_THAT_ERROR(UniqueTmp.renameTo(NewPath.str()), Succeeded());
    }
    SmallVector<std::string, 2> DirFiles;
    std::error_code EC = listDir(TmpDir.path(), DirFiles);
    ASSERT_FALSE(EC) << EC.message();
    EXPECT_THAT(DirFiles, testing::ElementsAre("otherfile.o"));
  }
}
