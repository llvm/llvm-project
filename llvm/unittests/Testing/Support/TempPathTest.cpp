//===- TempPathTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using llvm::unittest::TempDir;
using llvm::unittest::TempFile;
using llvm::unittest::TempLink;

namespace {

TEST(TempPathTest, TempDir) {
  Optional<TempDir> Dir1, Dir2;
  StringRef Prefix = "temp-path-test";
  Dir1.emplace(Prefix, /*Unique=*/true);
  EXPECT_EQ(Prefix,
            sys::path::filename(Dir1->path()).take_front(Prefix.size()));
  EXPECT_NE(Prefix, sys::path::filename(Dir1->path()));

  std::string Path = Dir1->path().str();
  ASSERT_TRUE(sys::fs::exists(Path));

  Dir2 = std::move(*Dir1);
  ASSERT_EQ(Path, Dir2->path());
  ASSERT_TRUE(sys::fs::exists(Path));

  Dir1 = std::nullopt;
  ASSERT_TRUE(sys::fs::exists(Path));

  Dir2 = std::nullopt;
  ASSERT_FALSE(sys::fs::exists(Path));
}

TEST(TempPathTest, TempFile) {
  TempDir D("temp-path-test", /*Unique=*/true);
  ASSERT_TRUE(sys::fs::exists(D.path()));

  Optional<TempFile> File1, File2;
  File1.emplace(D.path("file"), "suffix", "content");
  EXPECT_EQ("file.suffix", sys::path::filename(File1->path()));
  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        MemoryBuffer::getFile(File1->path());
    ASSERT_TRUE(Buffer);
    ASSERT_EQ("content", (*Buffer)->getBuffer());
  }

  std::string Path = File1->path().str();
  ASSERT_TRUE(sys::fs::exists(Path));

  File2 = std::move(*File1);
  ASSERT_EQ(Path, File2->path());
  ASSERT_TRUE(sys::fs::exists(Path));

  File1 = std::nullopt;
  ASSERT_TRUE(sys::fs::exists(Path));

  File2 = std::nullopt;
  ASSERT_FALSE(sys::fs::exists(Path));
}

TEST(TempPathTest, TempLink) {
  TempDir D("temp-path-test", /*Unique=*/true);
  ASSERT_TRUE(sys::fs::exists(D.path()));
  TempFile File(D.path("file"), "suffix", "content");

  Optional<TempLink> Link1, Link2;
  Link1.emplace(File.path(), D.path("link"));
  EXPECT_EQ("link", sys::path::filename(Link1->path()));
  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        MemoryBuffer::getFile(Link1->path());
    ASSERT_TRUE(Buffer);
    ASSERT_EQ("content", (*Buffer)->getBuffer());
  }

  std::string Path = Link1->path().str();
  ASSERT_TRUE(sys::fs::exists(Path));

  Link2 = std::move(*Link1);
  ASSERT_EQ(Path, Link2->path());
  ASSERT_TRUE(sys::fs::exists(Path));

  Link1 = std::nullopt;
  ASSERT_TRUE(sys::fs::exists(Path));

  Link2 = std::nullopt;
  ASSERT_FALSE(sys::fs::exists(Path));
}

} // namespace
