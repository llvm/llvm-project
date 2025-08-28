//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputBackend.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::vfs;

namespace {

struct MockOutputBackendData {
  int Cloned = 0;
  int FilesCreated = 0;
  std::optional<OutputConfig> LastConfig;
  unique_function<Error()> FileCreator;
};

struct MockOutputBackend final : public OutputBackend {
  struct MockFile final : public OutputFileImpl {
    Error keep() override { return Error::success(); }
    Error discard() override { return Error::success(); }
    raw_pwrite_stream &getOS() override { return OS; }
    raw_null_ostream OS;
  };

  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    ++Data.Cloned;
    return const_cast<MockOutputBackend *>(this);
  }

  Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef, std::optional<OutputConfig> Config) override {
    ++Data.FilesCreated;
    Data.LastConfig = Config;
    if (Data.FileCreator)
      return Data.FileCreator();
    return std::make_unique<MockFile>();
  }

  Expected<OutputFile>
  createAutoDiscardFile(const Twine &OutputPath,
                        std::optional<OutputConfig> Config = std::nullopt) {
    return consumeDiscardOnDestroy(createFile(OutputPath, Config));
  }

  MockOutputBackend(MockOutputBackendData &Data) : Data(Data) {}
  MockOutputBackendData &Data;
};

static IntrusiveRefCntPtr<MockOutputBackend>
createMockBackend(MockOutputBackendData &Data) {
  return makeIntrusiveRefCnt<MockOutputBackend>(Data);
}

static Error createCustomError() {
  return createStringError(inconvertibleErrorCode(), "custom error");
}

TEST(VirtualOutputBackendTest, construct) {
  MockOutputBackendData Data;
  auto B = createMockBackend(Data);
  EXPECT_EQ(0, Data.Cloned);
  EXPECT_EQ(0, Data.FilesCreated);
}

TEST(VirtualOutputBackendTest, clone) {
  MockOutputBackendData Data;
  auto Backend = createMockBackend(Data);
  auto Clone = Backend->clone();
  EXPECT_EQ(1, Data.Cloned);

  // Confirm the clone matches what the mock's cloneImpl() does.
  EXPECT_EQ(Backend.get(), Clone.get());

  // Make another clone.
  Backend->clone();
  EXPECT_EQ(2, Data.Cloned);
}

TEST(VirtualOutputBackendTest, createFile) {
  MockOutputBackendData Data;
  auto Backend = createMockBackend(Data);

  StringRef FilePath = "dir/file";
  OutputFile F;
  EXPECT_THAT_ERROR(Backend->createFile(Twine(FilePath)).moveInto(F),
                    Succeeded());
  EXPECT_EQ(1, Data.FilesCreated);
  EXPECT_EQ(FilePath, F.getPath());
  EXPECT_EQ(std::nullopt, Data.LastConfig);

  // Confirm OutputBackend has not installed a discard handler.
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(F = OutputFile(), "output not closed");
#endif
  consumeError(F.discard());

  // Create more files and specify configs.
  for (OutputConfig Config : {
           OutputConfig(),
           OutputConfig().setNoAtomicWrite().setDiscardOnSignal(),
           OutputConfig().setAtomicWrite().setNoDiscardOnSignal(),
           OutputConfig().setText(),
           OutputConfig().setTextWithCRLF(),
       }) {
    int CreatedAlready = Data.FilesCreated;
    EXPECT_THAT_ERROR(
        Backend->createAutoDiscardFile(Twine(FilePath), Config).takeError(),
        Succeeded());
    EXPECT_EQ(Config, Data.LastConfig);
    EXPECT_EQ(1 + CreatedAlready, Data.FilesCreated);
  }
}

TEST(VirtualOutputBackendTest, createFileInvalidConfigCRLF) {
  MockOutputBackendData Data;
  auto Backend = createMockBackend(Data);

  // Check that invalid configs don't make it to the backend.
  EXPECT_THAT_ERROR(
      Backend
          ->createAutoDiscardFile(Twine("dir/file"), OutputConfig().setCRLF())
          .takeError(),
      FailedWithMessage("dir/file: invalid config: {CRLF}"));
  EXPECT_EQ(0, Data.FilesCreated);
}

TEST(VirtualOutputBackendTest, createFileError) {
  MockOutputBackendData Data;
  Data.FileCreator = createCustomError;
  auto Backend = createMockBackend(Data);

  // Check that invalid configs don't make it to the backend.
  EXPECT_THAT_ERROR(
      Backend->createAutoDiscardFile(Twine("dir/file")).takeError(),
      FailedWithMessage("custom error"));
  EXPECT_EQ(1, Data.FilesCreated);
}

} // end namespace
