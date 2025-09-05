//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputFile.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::vfs;

namespace {

struct MockOutputFileData {
  int Kept = 0;
  int Discarded = 0;
  int Handled = 0;
  unique_function<Error()> Keeper;
  unique_function<Error()> Discarder;

  void handler(Error E) {
    consumeError(std::move(E));
    ++Handled;
  }
  unique_function<void(Error)> getHandler() {
    return [this](Error E) { handler(std::move(E)); };
  }

  SmallString<128> V;
  std::optional<raw_svector_ostream> VOS;
  raw_pwrite_stream *OS = nullptr;

  MockOutputFileData() : VOS(std::in_place, V), OS(&*VOS) {}
  MockOutputFileData(raw_pwrite_stream &OS) : OS(&OS) {}
};

struct MockOutputFile final : public OutputFileImpl {
  Error keep() override {
    ++Data.Kept;
    if (Data.Keeper)
      return Data.Keeper();
    return Error::success();
  }

  Error discard() override {
    ++Data.Discarded;
    if (Data.Discarder)
      return Data.Discarder();
    return Error::success();
  }

  raw_pwrite_stream &getOS() override {
    if (!Data.OS)
      report_fatal_error("missing stream in MockOutputFile::getOS");
    return *Data.OS;
  }

  MockOutputFile(MockOutputFileData &Data) : Data(Data) {}
  MockOutputFileData &Data;
};

static std::unique_ptr<MockOutputFile>
createMockOutput(MockOutputFileData &Data) {
  return std::make_unique<MockOutputFile>(Data);
}

static Error createCustomError() {
  return createStringError(inconvertibleErrorCode(), "custom error");
}

TEST(VirtualOutputFileTest, construct) {
  OutputFile F;
  EXPECT_EQ("", F.getPath());
  EXPECT_FALSE(F);
  EXPECT_FALSE(F.isOpen());

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  EXPECT_DEATH(F.getOS(), "Expected open output stream");
#endif
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
TEST(VirtualOutputFileTest, constructNull) {
  EXPECT_DEATH(OutputFile("some/file/path", nullptr),
               "Expected open output file");
}
#endif

TEST(VirtualOutputFileTest, destroy) {
  MockOutputFileData Data;
  StringRef FilePath = "some/file/path";

  // Check behaviour when destroying, first without a handler and then with
  // one. The handler shouldn't be called.
  std::optional<OutputFile> F(std::in_place, FilePath, createMockOutput(Data));
  EXPECT_TRUE(F->isOpen());
  EXPECT_EQ(FilePath, F->getPath());
  EXPECT_EQ(Data.OS, &F->getOS());
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(F.reset(), "output not closed");
#endif
  F->discardOnDestroy(Data.getHandler());
  EXPECT_EQ(0, Data.Discarded);
  EXPECT_EQ(0, Data.Handled);
  F.reset();
  EXPECT_EQ(1, Data.Discarded);
  EXPECT_EQ(0, Data.Handled);

  // Try again when discard returns an error. This time the handler should be
  // called.
  Data.Discarder = createCustomError;
  F.emplace("some/file/path", createMockOutput(Data));
  F->discardOnDestroy(Data.getHandler());
  F.reset();
  EXPECT_EQ(2, Data.Discarded);
  EXPECT_EQ(1, Data.Handled);
}

TEST(VirtualOutputFileTest, destroyProxy) {
  MockOutputFileData Data;

  std::optional<OutputFile> F(std::in_place, "some/file/path",
                              createMockOutput(Data));
  F->discardOnDestroy(Data.getHandler());
  std::unique_ptr<raw_pwrite_stream> Proxy;
  EXPECT_THAT_ERROR(F->createProxy().moveInto(Proxy), Succeeded());
  F.reset();
#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  EXPECT_DEATH(*Proxy << "data", "use after reset");
#endif
  Proxy.reset();
}

TEST(VirtualOutputFileTest, discard) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  {
    OutputFile F("some/file/path", createMockOutput(Data));
    F.discardOnDestroy(Data.getHandler());
    F << Content;
    EXPECT_EQ(Content, Data.V);

    EXPECT_THAT_ERROR(F.discard(), Succeeded());
    EXPECT_FALSE(F.isOpen());
    EXPECT_EQ(0, Data.Kept);
    EXPECT_EQ(1, Data.Discarded);

#if GTEST_HAS_DEATH_TEST
    EXPECT_DEATH(consumeError(F.keep()),
                 "some/file/path: output already closed");
    EXPECT_DEATH(consumeError(F.discard()),
                 "some/file/path: output already closed");
#endif
  }
  EXPECT_EQ(0, Data.Kept);
  EXPECT_EQ(1, Data.Discarded);
}

TEST(VirtualOutputFileTest, discardError) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  Data.Discarder = createCustomError;
  {
    OutputFile F("some/file/path", createMockOutput(Data));
    F.discardOnDestroy(Data.getHandler());
    F << Content;
    EXPECT_EQ(Content, Data.V);
    EXPECT_THAT_ERROR(F.discard(), FailedWithMessage("custom error"));
    EXPECT_FALSE(F.isOpen());
    EXPECT_EQ(0, Data.Kept);
    EXPECT_EQ(1, Data.Discarded);
    EXPECT_EQ(0, Data.Handled);
  }
  EXPECT_EQ(0, Data.Kept);
  EXPECT_EQ(1, Data.Discarded);
  EXPECT_EQ(0, Data.Handled);
}

TEST(VirtualOutputFileTest, discardProxy) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  OutputFile F("some/file/path", createMockOutput(Data));
  F.discardOnDestroy(Data.getHandler());

  std::unique_ptr<raw_pwrite_stream> Proxy;
  EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
  *Proxy << Content;
  EXPECT_EQ(Content, Data.V);

  EXPECT_THAT_ERROR(F.discard(), Succeeded());
  EXPECT_FALSE(F.isOpen());
  EXPECT_EQ(0, Data.Kept);
  EXPECT_EQ(1, Data.Discarded);
}

TEST(VirtualOutputFileTest, discardProxyFlush) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  OutputFile F("some/file/path", createMockOutput(Data));
  F.discardOnDestroy(Data.getHandler());
  F.getOS().SetBufferSize(Content.size() * 2);

  std::unique_ptr<raw_pwrite_stream> Proxy;
  EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
  *Proxy << Content;
  EXPECT_EQ("", Data.V);
  EXPECT_THAT_ERROR(F.discard(), Succeeded());
  EXPECT_EQ(Content, Data.V);
  EXPECT_FALSE(F.isOpen());
  EXPECT_EQ(0, Data.Kept);
  EXPECT_EQ(1, Data.Discarded);
}

TEST(VirtualOutputFileTest, keep) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  {
    OutputFile F("some/file/path", createMockOutput(Data));
    F.discardOnDestroy(Data.getHandler());
    F << Content;
    EXPECT_EQ(Content, Data.V);

    EXPECT_THAT_ERROR(F.keep(), Succeeded());
    EXPECT_FALSE(F.isOpen());
    EXPECT_EQ(1, Data.Kept);
    EXPECT_EQ(0, Data.Discarded);

#if GTEST_HAS_DEATH_TEST
    EXPECT_DEATH(consumeError(F.keep()),
                 "some/file/path: output already closed");
    EXPECT_DEATH(consumeError(F.discard()),
                 "some/file/path: output already closed");
#endif
  }
  EXPECT_EQ(1, Data.Kept);
  EXPECT_EQ(0, Data.Discarded);
}

TEST(VirtualOutputFileTest, keepError) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  Data.Keeper = createCustomError;
  {
    OutputFile F("some/file/path", createMockOutput(Data));
    F.discardOnDestroy(Data.getHandler());
    F << Content;
    EXPECT_EQ(Content, Data.V);

    EXPECT_THAT_ERROR(F.keep(), FailedWithMessage("custom error"));
    EXPECT_FALSE(F.isOpen());
    EXPECT_EQ(1, Data.Kept);
    EXPECT_EQ(0, Data.Discarded);
    EXPECT_EQ(0, Data.Handled);
  }
  EXPECT_EQ(1, Data.Kept);
  EXPECT_EQ(0, Data.Discarded);
  EXPECT_EQ(0, Data.Handled);
}

TEST(VirtualOutputFileTest, keepProxy) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  OutputFile F("some/file/path", createMockOutput(Data));
  F.discardOnDestroy(Data.getHandler());

  std::unique_ptr<raw_pwrite_stream> Proxy;
  EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
  *Proxy << Content;
  EXPECT_EQ(Content, Data.V);
  Proxy.reset();
  EXPECT_THAT_ERROR(F.keep(), Succeeded());
  EXPECT_FALSE(F.isOpen());
  EXPECT_EQ(1, Data.Kept);
  EXPECT_EQ(0, Data.Discarded);
}

#if GTEST_HAS_DEATH_TEST
TEST(VirtualOutputFileTest, keepProxyStillOpen) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  OutputFile F("some/file/path", createMockOutput(Data));
  F.discardOnDestroy(Data.getHandler());

  std::unique_ptr<raw_pwrite_stream> Proxy;
  EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
  *Proxy << Content;
  EXPECT_EQ(Content, Data.V);
  EXPECT_DEATH(consumeError(F.keep()), "some/file/path: output has open proxy");
}
#endif

TEST(VirtualOutputFileTest, keepProxyFlush) {
  StringRef Content = "some data";
  MockOutputFileData Data;
  OutputFile F("some/file/path", createMockOutput(Data));
  F.discardOnDestroy(Data.getHandler());
  F.getOS().SetBufferSize(Content.size() * 2);

  std::unique_ptr<raw_pwrite_stream> Proxy;
  EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
  *Proxy << Content;
  EXPECT_EQ("", Data.V);
  Proxy.reset();
  EXPECT_THAT_ERROR(F.keep(), Succeeded());
  EXPECT_EQ(Content, Data.V);
  EXPECT_FALSE(F.isOpen());
  EXPECT_EQ(1, Data.Kept);
  EXPECT_EQ(0, Data.Discarded);
}

TEST(VirtualOutputFileTest, TwoProxies) {
  StringRef Content = "some data";
  MockOutputFileData Data;

  OutputFile F("some/file/path", createMockOutput(Data));
  F.discardOnDestroy(Data.getHandler());

  // Can't have two open proxies at once.
  {
    std::unique_ptr<raw_pwrite_stream> Proxy;
    EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
    EXPECT_THAT_ERROR(
        F.createProxy().takeError(),
        FailedWithMessage("some/file/path: output has open proxy"));
  }
  EXPECT_EQ(0, Data.Kept);
  EXPECT_EQ(0, Data.Discarded);

  // A second proxy after the first closes should work...
  {
    std::unique_ptr<raw_pwrite_stream> Proxy;
    EXPECT_THAT_ERROR(F.createProxy().moveInto(Proxy), Succeeded());
    *Proxy << Content;
    EXPECT_EQ(Content, Data.V);
  }
}

} // end namespace
