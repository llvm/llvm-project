//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputBackends.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/HashingOutputBackend.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::vfs;

namespace {

class OutputBackendProvider {
public:
  virtual bool rejectsMissingDirectories() = 0;

  virtual IntrusiveRefCntPtr<OutputBackend> createBackend() = 0;
  virtual std::string getFilePathToCreate() = 0;
  virtual std::string getFilePathToCreateUnder(StringRef Parent1,
                                               StringRef Parent2 = "") = 0;
  virtual Error checkCreated(StringRef FilePath,
                             OutputConfig Config = OutputConfig()) = 0;
  virtual Error checkWrote(StringRef FilePath, StringRef Data) = 0;
  virtual Error checkFlushed(StringRef FilePath, StringRef Data) = 0;
  virtual Error checkKept(StringRef FilePath, StringRef Data) = 0;
  virtual Error checkDiscarded(StringRef FilePath) = 0;

  virtual ~OutputBackendProvider() = default;

  struct Generator {
    std::string Name;
    std::function<std::unique_ptr<OutputBackendProvider>()> Generate;

    std::unique_ptr<OutputBackendProvider> operator()() const {
      return Generate();
    }
  };
};

struct BackendTest
    : public ::testing::TestWithParam<OutputBackendProvider::Generator> {
  std::unique_ptr<OutputBackendProvider> Provider;

  void SetUp() override { Provider = GetParam()(); }
  void TearDown() override { Provider = nullptr; }

  IntrusiveRefCntPtr<OutputBackend> createBackend() {
    return Provider->createBackend();
  }
};

TEST_P(BackendTest, Discard) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";
  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());

  O << Data;
  EXPECT_THAT_ERROR(O.discard(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkDiscarded(FilePath), Succeeded());
  EXPECT_FALSE(O.isOpen());
}

TEST_P(BackendTest, DiscardNoAtomicWrite) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";
  OutputConfig Config = OutputConfig().setNoAtomicWrite();

  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O),
                    Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath, Config), Succeeded());

  O << Data;
  EXPECT_THAT_ERROR(O.discard(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkDiscarded(FilePath), Succeeded());
  EXPECT_FALSE(O.isOpen());
}

TEST_P(BackendTest, Keep) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";

  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());
  ASSERT_TRUE(O.isOpen());

  O << Data;
  EXPECT_THAT_ERROR(Provider->checkWrote(FilePath, Data), Succeeded());

  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
  EXPECT_FALSE(O.isOpen());
}

TEST_P(BackendTest, KeepFlush) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";
  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());

  O << Data;
  EXPECT_THAT_ERROR(Provider->checkWrote(FilePath, Data), Succeeded());

  O.getOS().flush();
  EXPECT_THAT_ERROR(Provider->checkFlushed(FilePath, Data), Succeeded());

  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
}

TEST_P(BackendTest, KeepFlushProxy) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";
  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());
  {
    std::unique_ptr<raw_pwrite_stream> Proxy;
    EXPECT_THAT_ERROR(O.createProxy().moveInto(Proxy), Succeeded());
    *Proxy << Data;
    EXPECT_THAT_ERROR(Provider->checkWrote(FilePath, Data), Succeeded());

    Proxy->flush();
    EXPECT_THAT_ERROR(Provider->checkFlushed(FilePath, Data), Succeeded());
  }
  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
}

TEST_P(BackendTest, KeepEmpty) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());
  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, ""), Succeeded());
}

TEST_P(BackendTest, KeepMissingDirectory) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreateUnder("missing");
  StringRef Data = "some data";

  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());

  O << Data;
  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
}

TEST_P(BackendTest, KeepMissingDirectoryNested) {
  auto Backend = createBackend();
  std::string FilePath =
      Provider->getFilePathToCreateUnder("missing", "nested");
  StringRef Data = "some data";

  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath).moveInto(O), Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath), Succeeded());

  O << Data;
  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
}

TEST_P(BackendTest, KeepNoAtomicWrite) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";
  OutputConfig Config = OutputConfig().setNoAtomicWrite();

  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O),
                    Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath, Config), Succeeded());
  O << Data;
  EXPECT_THAT_ERROR(Provider->checkWrote(FilePath, Data), Succeeded());

  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
  EXPECT_FALSE(O.isOpen());
}

TEST_P(BackendTest, KeepNoAtomicWriteMissingDirectory) {
  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreate();
  StringRef Data = "some data";
  OutputConfig Config = OutputConfig().setNoAtomicWrite();

  OutputFile O;
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O),
                    Succeeded());
  consumeDiscardOnDestroy(O);
  ASSERT_THAT_ERROR(Provider->checkCreated(FilePath, Config), Succeeded());

  O << Data;
  EXPECT_THAT_ERROR(Provider->checkWrote(FilePath, Data), Succeeded());

  EXPECT_THAT_ERROR(O.keep(), Succeeded());
  EXPECT_THAT_ERROR(Provider->checkKept(FilePath, Data), Succeeded());
  EXPECT_FALSE(O.isOpen());
}

TEST_P(BackendTest, KeepMissingDirectoryNoImply) {
  // Skip this test if the backend doesn't have a concept of missing
  // directories.
  if (!Provider->rejectsMissingDirectories())
    return;

  auto Backend = createBackend();
  std::string FilePath = Provider->getFilePathToCreateUnder("missing");
  std::error_code EC = errorToErrorCode(
      consumeDiscardOnDestroy(
          Backend->createFile(FilePath,
                              OutputConfig().setNoImplyCreateDirectories()))
          .takeError());
  EXPECT_EQ(int(std::errc::no_such_file_or_directory), EC.value());
}

class NullOutputBackendProvider : public OutputBackendProvider {
public:
  bool rejectsMissingDirectories() override { return false; }

  IntrusiveRefCntPtr<OutputBackend> createBackend() override {
    return makeNullOutputBackend();
  }
  std::string getFilePathToCreate() override { return "ignored.data"; }
  std::string getFilePathToCreateUnder(StringRef Parent1,
                                       StringRef Parent2) override {
    SmallString<128> Path;
    sys::path::append(Path, Parent1, Parent2, getFilePathToCreate());
    return Path.str().str();
  }
  Error checkCreated(StringRef, OutputConfig) override {
    return Error::success();
  }
  Error checkWrote(StringRef, StringRef) override { return Error::success(); }
  Error checkFlushed(StringRef, StringRef) override { return Error::success(); }
  Error checkKept(StringRef, StringRef) override { return Error::success(); }
  Error checkDiscarded(StringRef) override { return Error::success(); }
};

struct OnDiskFile {
  const unittest::TempDir &D;
  SmallString<128> Path;
  StringRef ParentPath;
  StringRef Filename;
  StringRef Stem;
  StringRef Extension;
  std::unique_ptr<MemoryBuffer> LastBuffer;

  OnDiskFile(const unittest::TempDir &D, const Twine &InputPath) : D(D) {
    if (sys::path::is_absolute(InputPath))
      InputPath.toVector(Path);
    else
      sys::path::append(Path, D.path(), InputPath);
    ParentPath = sys::path::parent_path(Path);
    Filename = sys::path::filename(Path);
    Stem = sys::path::stem(Filename);
    Extension = sys::path::extension(Filename);
  }

  std::optional<OnDiskFile> findTemp() const;

  std::optional<sys::fs::UniqueID> getCurrentUniqueID();

  bool hasUniqueID(sys::fs::UniqueID ID) {
    auto CurrentID = getCurrentUniqueID();
    if (!CurrentID)
      return false;
    return *CurrentID == ID;
  }

  std::optional<StringRef> getCurrentContent() {
    auto OnDiskOrErr = MemoryBuffer::getFile(Path);
    if (!OnDiskOrErr)
      return std::nullopt;
    LastBuffer = std::move(*OnDiskOrErr);
    return LastBuffer->getBuffer();
  }

  bool equalsCurrentContent(StringRef Data) {
    auto CurrentContent = getCurrentContent();
    if (!CurrentContent)
      return false;
    return *CurrentContent == Data;
  }

  bool equalsCurrentContent(std::nullopt_t) {
    return getCurrentContent() == std::nullopt;
  }
};

class OnDiskOutputBackendProvider : public OutputBackendProvider {
public:
  bool rejectsMissingDirectories() override { return true; }

  std::optional<unittest::TempDir> D;

  IntrusiveRefCntPtr<OutputBackend> createBackend() override {
    auto Backend = makeIntrusiveRefCnt<OnDiskOutputBackend>();
    Backend->Settings = Settings;
    return Backend;
  }
  void init() {
    if (!D)
      D.emplace("OutputBackendTest.d", /*Unique=*/true);
  }
  std::string getFilePathToCreate() override {
    init();
    return OnDiskFile(*D, "file.data").Path.str().str();
  }
  std::string getFilePathToCreateUnder(StringRef Parent1,
                                       StringRef Parent2) override {
    init();
    SmallString<128> Path;
    sys::path::append(Path, D->path(), Parent1, Parent2, "file.data");
    return Path.str().str();
  }

  Error checkCreated(StringRef FilePath, OutputConfig Config) override;
  Error checkWrote(StringRef FilePath, StringRef Data) override;
  Error checkFlushed(StringRef FilePath, StringRef Data) override;
  Error checkKept(StringRef FilePath, StringRef Data) override;
  Error checkDiscarded(StringRef FilePath) override;

  struct FileInfo {
    OutputConfig Config;
    std::optional<OnDiskFile> F;
    std::optional<OnDiskFile> Temp;
    std::optional<sys::fs::UniqueID> UID;
    std::optional<sys::fs::UniqueID> TempUID;
  };
  Error checkOpen(FileInfo &Info);
  bool shouldUseTemporaries(const FileInfo &Info) const;

  OnDiskOutputBackendProvider() = default;
  explicit OnDiskOutputBackendProvider(
      const OnDiskOutputBackend::OutputSettings &Settings)
      : Settings(Settings) {}
  OnDiskOutputBackend::OutputSettings Settings;

  StringMap<FileInfo> Files;
  Error lookupFileInfo(StringRef FilePath, FileInfo *&Info);
};

bool OnDiskOutputBackendProvider::shouldUseTemporaries(
    const FileInfo &Info) const {
  return Info.Config.getAtomicWrite() && Settings.UseTemporaries;
}

struct ProviderGeneratorList {
  std::vector<OutputBackendProvider::Generator> Generators;
  ProviderGeneratorList(
      std::initializer_list<OutputBackendProvider::Generator> IL)
      : Generators(IL) {}

  std::string operator()(
      const ::testing::TestParamInfo<OutputBackendProvider::Generator> &Info) {
    return Info.param.Name;
  }
};

ProviderGeneratorList BackendGenerators = {
    {"Null", []() { return std::make_unique<NullOutputBackendProvider>(); }},
    {"OnDisk",
     []() { return std::make_unique<OnDiskOutputBackendProvider>(); }},
    {"OnDisk_DisableRemoveOnSignal",
     []() {
       OnDiskOutputBackend::OutputSettings Settings;
       Settings.RemoveOnSignal = false;
       return std::make_unique<OnDiskOutputBackendProvider>(Settings);
     }},
    {"OnDisk_DisableTemporaries",
     []() {
       OnDiskOutputBackend::OutputSettings Settings;
       Settings.UseTemporaries = false;
       return std::make_unique<OnDiskOutputBackendProvider>(Settings);
     }},
};

INSTANTIATE_TEST_SUITE_P(VirtualOutput, BackendTest,
                         ::testing::ValuesIn(BackendGenerators.Generators),
                         BackendGenerators);

std::optional<sys::fs::UniqueID> OnDiskFile::getCurrentUniqueID() {
  sys::fs::file_status Status;
  sys::fs::status(Path, Status, /*follow=*/false);
  if (!sys::fs::is_regular_file(Status))
    return std::nullopt;
  return Status.getUniqueID();
}

std::optional<OnDiskFile> OnDiskFile::findTemp() const {
  std::error_code EC;
  for (sys::fs::directory_iterator I(ParentPath, EC), E; !EC && I != E;
       I.increment(EC)) {
    StringRef TempPath = I->path();
    if (!TempPath.starts_with(D.path()))
      continue;

    // Look for "<stem>-*.<extension>.tmp".
    if (sys::path::extension(TempPath) != ".tmp")
      continue;

    // Drop the ".tmp" and check the extension and stem.
    StringRef TempStem = sys::path::stem(TempPath);
    if (sys::path::extension(TempStem) != Extension)
      continue;
    StringRef OriginalStem = sys::path::stem(TempStem);
    if (!OriginalStem.starts_with(Stem))
      continue;
    if (!OriginalStem.drop_front(Stem.size()).starts_with("-"))
      continue;

    // Found it.
    return OnDiskFile(D, TempPath.drop_front(D.path().size() + 1));
  }
  return std::nullopt;
}

Error OnDiskOutputBackendProvider::lookupFileInfo(StringRef FilePath,
                                                  FileInfo *&Info) {
  auto I = Files.find(FilePath);
  if (Files.find(FilePath) == Files.end())
    return createStringError(inconvertibleErrorCode(),
                             "Missing call to checkCreated()");
  Info = &I->second;
  assert(Info->F && "Expected OnDiskFile to be initialized");
  return Error::success();
}

Error OnDiskOutputBackendProvider::checkOpen(FileInfo &Info) {
  // Collect info about filesystem state.
  assert(Info.F);
  std::optional<sys::fs::UniqueID> UID = Info.F->getCurrentUniqueID();
  std::optional<OnDiskFile> Temp = Info.F->findTemp();
  std::optional<sys::fs::UniqueID> TempUID;
  if (Temp)
    TempUID = Temp->getCurrentUniqueID();

  // Check if it's correct.
  if (shouldUseTemporaries(Info)) {
    if (!Temp)
      return createStringError(inconvertibleErrorCode(),
                               "Missing temporary file");
    if (!TempUID)
      return createStringError(inconvertibleErrorCode(),
                               "Missing UID for temporary");
    if (UID)
      return createStringError(
          inconvertibleErrorCode(),
          "Unexpected final UID when temporaries should be used");

    // Check previous data.
    if (Info.Temp)
      if (Temp->Path != Info.Temp->Path)
        return createStringError(inconvertibleErrorCode(),
                                 "Temporary path changed");
    if (Info.TempUID)
      if (*TempUID != *Info.TempUID)
        return createStringError(inconvertibleErrorCode(),
                                 "Temporary UID changed");
  } else {
    if (Temp)
      return createStringError(inconvertibleErrorCode(),
                               "Unexpected temporary file");
    if (!UID)
      return createStringError(inconvertibleErrorCode(),
                               "Missing UID for temporary");

    // Check previous data.
    if (Info.UID)
      if (*UID != *Info.UID)
        return createStringError(inconvertibleErrorCode(), "UID changed");
  }

  Info.UID = UID;
  if (Temp)
    Info.Temp.emplace(*D, Temp->Path);
  else
    Info.Temp.reset();
  Info.TempUID = TempUID;
  return Error::success();
}

Error OnDiskOutputBackendProvider::checkCreated(StringRef FilePath,
                                                OutputConfig Config) {
  auto &Info = Files[FilePath];
  if (Info.F) {
    assert(OnDiskFile(*D, FilePath).Path == Info.F->Path);
    Info.UID = std::nullopt;
    Info.Temp.reset();
    Info.TempUID = std::nullopt;
  } else {
    Info.F.emplace(*D, FilePath);
  }
  Info.Config = Config;
  return checkOpen(Info);
}

Error OnDiskOutputBackendProvider::checkWrote(StringRef FilePath,
                                              StringRef Data) {
  FileInfo *Info = nullptr;
  if (Error E = lookupFileInfo(FilePath, Info))
    return E;
  return checkOpen(*Info);
}

Error OnDiskOutputBackendProvider::checkFlushed(StringRef FilePath,
                                                StringRef Data) {
  FileInfo *Info = nullptr;
  if (Error E = lookupFileInfo(FilePath, Info))
    return E;
  if (Error E = checkOpen(*Info))
    return E;

  OnDiskFile &F = shouldUseTemporaries(*Info) ? *Info->Temp : *Info->F;
  if (!F.equalsCurrentContent(Data))
    return createStringError(inconvertibleErrorCode(), "content not flushed");
  return Error::success();
}

Error OnDiskOutputBackendProvider::checkKept(StringRef FilePath,
                                             StringRef Data) {
  FileInfo *Info = nullptr;
  if (Error E = lookupFileInfo(FilePath, Info))
    return E;

#ifndef _WIN32
  sys::fs::UniqueID UID =
      shouldUseTemporaries(*Info) ? *Info->TempUID : *Info->UID;
  if (!Info->F->hasUniqueID(UID))
    return createStringError(inconvertibleErrorCode(),
                             "File not created by keep or changed UID");
#else
  // On Windows, the UID changes in a rename that happens in keep()
  // because it's based on hash of file paths. Instead check that the
  // file contents are the same.
  if (!Info->F->equalsCurrentContent(Data))
    return createStringError(inconvertibleErrorCode(),
                             "File not created by keep");
#endif

  if (std::optional<OnDiskFile> Temp = Info->F->findTemp())
    return createStringError(inconvertibleErrorCode(),
                             "Temporary not removed by keep");

  return Error::success();
}

Error OnDiskOutputBackendProvider::checkDiscarded(StringRef FilePath) {
  FileInfo *Info = nullptr;
  if (Error E = lookupFileInfo(FilePath, Info))
    return E;

  if (std::optional<sys::fs::UniqueID> UID = Info->F->getCurrentUniqueID())
    return createStringError(inconvertibleErrorCode(),
                             "File not removed by discard");

  if (std::optional<OnDiskFile> Temp = Info->F->findTemp())
    return createStringError(inconvertibleErrorCode(),
                             "Temporary not removed by discard");

  return Error::success();
}

TEST(VirtualOutputBackendAdaptors, makeFilteringOutputBackend) {
  bool ShouldCreate = false;
  auto Backend = makeFilteringOutputBackend(
      makeIntrusiveRefCnt<OnDiskOutputBackend>(),
      [&ShouldCreate](StringRef, std::optional<OutputConfig>) {
        return ShouldCreate;
      });

  int Count = 0;
  unittest::TempDir D("FilteringOutputBackendTest.d", /*Unique=*/true);
  for (bool ShouldCreateVal : {false, true, true, false}) {
    ShouldCreate = ShouldCreateVal;
    OnDiskFile OnDisk(D, "file." + Twine(Count++) + "." + Twine(ShouldCreate));
    OutputFile Output;
    ASSERT_THAT_ERROR(consumeDiscardOnDestroy(Backend->createFile(OnDisk.Path))
                          .moveInto(Output),
                      Succeeded());
    EXPECT_NE(ShouldCreate, Output.isNull());
    Output << "content";
    EXPECT_THAT_ERROR(Output.keep(), Succeeded());

    if (ShouldCreate) {
      EXPECT_EQ(StringRef("content"), OnDisk.getCurrentContent());
    } else {
      EXPECT_FALSE(OnDisk.getCurrentUniqueID());
    }
  }
  SmallString<128> Path;
}

class AbsolutePathBackend : public ProxyOutputBackend {
  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    llvm_unreachable("unimplemented");
  }

  Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig> Config) override {
    assert(!sys::path::is_absolute(Path) &&
           "Expected tests to pass all relative paths");
    SmallString<256> AbsPath;
    sys::path::append(AbsPath, CWD, Path);
    return ProxyOutputBackend::createFileImpl(AbsPath, Config);
  }

public:
  AbsolutePathBackend(const Twine &CWD,
                      IntrusiveRefCntPtr<OutputBackend> Backend)
      : ProxyOutputBackend(std::move(Backend)), CWD(CWD.str()) {
    assert(sys::path::is_absolute(this->CWD) &&
           "Expected tests to pass a relative path");
  }

private:
  std::string CWD;
};

TEST(VirtualOutputBackendAdaptors, makeMirroringOutputBackend) {
  unittest::TempDir D1("MirroringOutputBackendTest.1.d", /*Unique=*/true);
  unittest::TempDir D2("MirroringOutputBackendTest.2.d", /*Unique=*/true);

  IntrusiveRefCntPtr<OutputBackend> Backend;
  {
    auto OnDisk = makeIntrusiveRefCnt<OnDiskOutputBackend>();
    Backend = makeMirroringOutputBackend(
        makeIntrusiveRefCnt<AbsolutePathBackend>(D1.path(), OnDisk),
        makeIntrusiveRefCnt<AbsolutePathBackend>(D2.path(), OnDisk));
  }

  OnDiskFile OnDisk1(D1, "file");
  OnDiskFile OnDisk2(D2, "file");
  OutputFile Output;
  ASSERT_THAT_ERROR(
      consumeDiscardOnDestroy(Backend->createFile("file")).moveInto(Output),
      Succeeded());
  EXPECT_TRUE(OnDisk1.findTemp());
  EXPECT_TRUE(OnDisk2.findTemp());

  Output << "content";
  Output.getOS().pwrite("ON", /*Size=*/2, /*Offset=*/1);
  EXPECT_THAT_ERROR(Output.keep(), Succeeded());
  EXPECT_EQ(StringRef("cONtent"), OnDisk1.getCurrentContent());
  EXPECT_EQ(StringRef("cONtent"), OnDisk2.getCurrentContent());
  EXPECT_NE(OnDisk1.getCurrentUniqueID(), OnDisk2.getCurrentUniqueID());
}

/// Behaves like NullOutputFileImpl, but doesn't match the RTTI (so OutputFile
/// cannot tell).
class LikeNullOutputFile final : public OutputFileImpl {
  Error keep() final { return Error::success(); }
  Error discard() final { return Error::success(); }
  raw_pwrite_stream &getOS() final { return OS; }

public:
  LikeNullOutputFile(raw_null_ostream &OS) : OS(OS) {}
  raw_null_ostream &OS;
};
class LikeNullOutputBackend final : public OutputBackend {
  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    llvm_unreachable("not implemented");
  }

  Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig> Config) override {
    return std::make_unique<LikeNullOutputFile>(OS);
  }

public:
  raw_null_ostream OS;
};

TEST(VirtualOutputBackendAdaptors, makeMirroringOutputBackendNull) {
  // Check that null outputs are skipped by seeing that LikeNull->OS is passed
  // through directly (without a mirroring proxy stream) to Output.
  auto LikeNull = makeIntrusiveRefCnt<LikeNullOutputBackend>();
  auto Null1 = makeNullOutputBackend();
  auto Mirror = makeMirroringOutputBackend(Null1, LikeNull);
  OutputFile Output;
  ASSERT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).moveInto(Output),
      Succeeded());
  EXPECT_TRUE(!Output.isNull());
  EXPECT_EQ(&Output.getOS(), &LikeNull->OS);

  // Check the other direction.
  Mirror = makeMirroringOutputBackend(LikeNull, Null1);
  ASSERT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).moveInto(Output),
      Succeeded());
  EXPECT_TRUE(!Output.isNull());
  EXPECT_EQ(&Output.getOS(), &LikeNull->OS);

  // Same null backend, twice.
  Mirror = makeMirroringOutputBackend(Null1, Null1);
  ASSERT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).moveInto(Output),
      Succeeded());
  EXPECT_TRUE(Output.isNull());

  // Two null backends.
  auto Null2 = makeNullOutputBackend();
  Mirror = makeMirroringOutputBackend(Null1, Null2);
  ASSERT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).moveInto(Output),
      Succeeded());
  EXPECT_TRUE(Output.isNull());
}

class StringErrorBackend final : public OutputBackend {
  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    llvm_unreachable("not implemented");
  }

  Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig> Config) override {
    return createStringError(inconvertibleErrorCode(), Msg);
  }

public:
  StringErrorBackend(const Twine &Msg) : Msg(Msg.str()) {}
  std::string Msg;
};

TEST(VirtualOutputBackendAdaptors, makeMirroringOutputBackendCreateError) {
  auto Error1 = makeIntrusiveRefCnt<StringErrorBackend>("error-backend-1");
  auto Null = makeNullOutputBackend();

  auto Mirror = makeMirroringOutputBackend(Null, Error1);
  EXPECT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).takeError(),
      FailedWithMessage(Error1->Msg));

  Mirror = makeMirroringOutputBackend(Error1, Null);
  EXPECT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).takeError(),
      FailedWithMessage(Error1->Msg));

  auto Error2 = makeIntrusiveRefCnt<StringErrorBackend>("error-backend-2");
  Mirror = makeMirroringOutputBackend(Error1, Error2);
  EXPECT_THAT_ERROR(
      consumeDiscardOnDestroy(Mirror->createFile("file")).takeError(),
      FailedWithMessage(Error1->Msg));
}

TEST(OnDiskBackendTest, OnlyIfDifferent) {
  OnDiskOutputBackendProvider Provider;
  auto Backend = Provider.createBackend();
  std::string FilePath = Provider.getFilePathToCreate();
  StringRef Data = "some data";
  OutputConfig Config = OutputConfig().setOnlyIfDifferent();

  OutputFile O1, O2, O3;
  sys::fs::file_status Status1, Status2, Status3;
  // Write first file.
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O1),
                    Succeeded());
  O1 << Data;
  EXPECT_THAT_ERROR(O1.keep(), Succeeded());
  EXPECT_FALSE(O1.isOpen());
  EXPECT_FALSE(sys::fs::status(FilePath, Status1, /*follow=*/false));

  // Write second with same content.
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O2),
                    Succeeded());
  O2 << Data;
  EXPECT_THAT_ERROR(O2.keep(), Succeeded());
  EXPECT_FALSE(O2.isOpen());
  EXPECT_FALSE(sys::fs::status(FilePath, Status2, /*follow=*/false));

#ifndef _WIN32
  // Make sure the output path file is not modified with same content.
  EXPECT_EQ(Status1.getUniqueID(), Status2.getUniqueID());
#else
  // On Windows, UniqueIDs are currently hash of file paths and don't
  // change on overwrites or depend on the file content. Check that
  // the file content is what is expected instead.
  auto EqualsCurrentContent = [](StringRef FilePath, StringRef Data) -> bool {
    auto BufOrErr = MemoryBuffer::getFile(FilePath);
    if (!BufOrErr)
      return false;
    return (*BufOrErr)->getBuffer() == Data;
  };
  EXPECT_TRUE(EqualsCurrentContent(FilePath, Data));
#endif

  // Write third with different content.
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O3),
                    Succeeded());
  O3 << Data << "\n";
  EXPECT_THAT_ERROR(O3.keep(), Succeeded());
  EXPECT_FALSE(O3.isOpen());
  EXPECT_FALSE(sys::fs::status(FilePath, Status3, /*follow=*/false));

#ifndef _WIN32
  // This should overwrite the file and create a different UniqueID.
  EXPECT_NE(Status1.getUniqueID(), Status3.getUniqueID());
#else
  // On Windows, UniqueIDs are currently hash of file paths and don't
  // change on overwrites or depend on the file content. Check that
  // the file content is what is expected instead.
  EXPECT_TRUE(EqualsCurrentContent(FilePath, (Data + "\n").str()));
#endif
}

TEST(OnDiskBackendTest, Append) {
  OnDiskOutputBackendProvider Provider;
  auto Backend = Provider.createBackend();
  std::string FilePath = Provider.getFilePathToCreate();
  OutputConfig Config = OutputConfig().setAppend();

  OutputFile O1, O2, O3;
  // Write first file.
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O1),
                    Succeeded());
  O1 << "some data\n";
  EXPECT_THAT_ERROR(O1.keep(), Succeeded());
  EXPECT_FALSE(O1.isOpen());

  OnDiskFile File1(*Provider.D, FilePath);
  EXPECT_TRUE(File1.equalsCurrentContent("some data\n"));

  // Append same data.
  EXPECT_THAT_ERROR(Backend->createFile(FilePath, Config).moveInto(O2),
                    Succeeded());
  O2 << "more data\n";
  EXPECT_THAT_ERROR(O2.keep(), Succeeded());
  EXPECT_FALSE(O2.isOpen());

  // Check data is appended.
  OnDiskFile File2(*Provider.D, FilePath);
  EXPECT_TRUE(File2.equalsCurrentContent("some data\nmore data\n"));

  // Non atomic append.
  EXPECT_THAT_ERROR(
      Backend->createFile(FilePath, Config.setNoAtomicWrite()).moveInto(O3),
      Succeeded());
  O3 << "more more\n";
  EXPECT_THAT_ERROR(O3.keep(), Succeeded());
  EXPECT_FALSE(O3.isOpen());

  // Check data is appended.
  OnDiskFile File3(*Provider.D, FilePath);
  EXPECT_TRUE(File3.equalsCurrentContent("some data\nmore data\nmore more\n"));
}

TEST(HashingBackendTest, HashOutput) {
  HashingOutputBackend<BLAKE3> Backend;
  OutputFile O1, O2, O3, O4, O5;
  EXPECT_THAT_ERROR(Backend.createFile("file1").moveInto(O1), Succeeded());
  O1 << "some data";
  EXPECT_THAT_ERROR(O1.keep(), Succeeded());
  EXPECT_THAT_ERROR(Backend.createFile("file2").moveInto(O2), Succeeded());
  O2 << "some data";
  EXPECT_THAT_ERROR(O2.keep(), Succeeded());
  EXPECT_EQ(Backend.getHashValueForFile("file1"),
            Backend.getHashValueForFile("file2"));

  EXPECT_THAT_ERROR(Backend.createFile("file3").moveInto(O3), Succeeded());
  O3 << "some ";
  O3 << "data";
  EXPECT_THAT_ERROR(O3.keep(), Succeeded());
  EXPECT_EQ(Backend.getHashValueForFile("file1"),
            Backend.getHashValueForFile("file3"));

  EXPECT_THAT_ERROR(Backend.createFile("file4").moveInto(O4), Succeeded());
  O4 << "same data";
  O4.getOS().pwrite("o", 1, 1);
  EXPECT_THAT_ERROR(O4.keep(), Succeeded());
  EXPECT_EQ(Backend.getHashValueForFile("file1"),
            Backend.getHashValueForFile("file4"));

  EXPECT_THAT_ERROR(Backend.createFile("file5").moveInto(O5), Succeeded());
  O5 << "different data";
  EXPECT_THAT_ERROR(O5.keep(), Succeeded());
  EXPECT_NE(Backend.getHashValueForFile("file1"),
            Backend.getHashValueForFile("file5"));
}

TEST(HashingBackendTest, ParallelHashOutput) {
  const unsigned NumFile = 20;
  DefaultThreadPool Pool;
  HashingOutputBackend<BLAKE3> Backend;
  auto getFileName = [](unsigned Idx) -> std::string {
    std::string Name;
    raw_string_ostream OS(Name);
    OS << "file" << Idx;
    return Name;
  };
  for (unsigned I = 0; I < NumFile; ++I) {
    Pool.async(
        [&](unsigned Idx) {
          OutputFile O;
          auto Name = getFileName(Idx);
          EXPECT_THAT_ERROR(Backend.createFile(Name).moveInto(O), Succeeded());
          O << "some data" << Idx;
          EXPECT_THAT_ERROR(O.keep(), Succeeded());
        },
        I);
  }
  Pool.wait();

  for (unsigned I = 0; I < NumFile; ++I) {
    auto Name = getFileName(I);
    auto Hash = Backend.getHashValueForFile(Name);
    EXPECT_TRUE(Hash);
  }
}
} // end namespace
