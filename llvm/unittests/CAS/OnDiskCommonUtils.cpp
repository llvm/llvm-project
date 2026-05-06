//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OnDiskCommonUtils.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Process.h"

using namespace llvm;

// Create a file with small size and and controlling the initial byte so
// caller can create different contents.
std::unique_ptr<unittest::TempFile>
unittest::cas::createSmallFile(char initChar) {
  auto TmpFile = std::make_unique<unittest::TempFile>(
      "smallfile.o", /*Suffix=*/"", /*Contents=*/"", /*Unique=*/true);
  StringRef Path = TmpFile->path();
  std::error_code EC;
  raw_fd_stream Out(Path, EC);
  EXPECT_FALSE(EC);
  SmallString<200> Data;
  Data += initChar;
  for (unsigned i = 1; i != 200; ++i) {
    Data += i;
  }
  Out.write(Data.data(), Data.size());
  return TmpFile;
}

std::unique_ptr<unittest::TempFile>
unittest::cas::createLargeFile(char initChar) {
  auto TmpFile = std::make_unique<unittest::TempFile>(
      "largefile.o", /*Suffix=*/"", /*Contents=*/"",
      /*Unique=*/true);
  StringRef Path = TmpFile->path();
  std::error_code EC;
  raw_fd_stream Out(Path, EC);
  EXPECT_FALSE(EC);
  SmallString<200> Data;
  Data += initChar;
  for (unsigned i = 1; i != 200; ++i) {
    Data += i;
  }
  for (unsigned i = 0; i != 1000; ++i) {
    Out.write(Data.data(), Data.size());
  }
  return TmpFile;
}

std::unique_ptr<unittest::TempFile>
unittest::cas::createLargePageAlignedFile(char initChar) {
  auto TmpFile = std::make_unique<unittest::TempFile>(
      "largepagealignedfile.o", /*Suffix=*/"", /*Contents=*/"",
      /*Unique=*/true);
  StringRef Path = TmpFile->path();
  std::error_code EC;
  raw_fd_stream Out(Path, EC);
  EXPECT_FALSE(EC);
  SmallString<256> Data;
  Data += initChar;
  for (unsigned i = 1; i != sys::Process::getPageSizeEstimate(); ++i) {
    Data += char(i);
  }
  for (unsigned i = 0; i != 64; ++i) {
    Out.write(Data.data(), Data.size());
  }
  Out.close();
  uint64_t FileSize;
  EC = sys::fs::file_size(Path, FileSize);
  EXPECT_FALSE(EC);
  assert(isAligned(Align(sys::Process::getPageSizeEstimate()), FileSize));
  return TmpFile;
}
