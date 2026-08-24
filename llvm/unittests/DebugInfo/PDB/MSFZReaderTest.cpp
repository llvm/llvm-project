//===- MSFZReaderTest.cpp - MSFZ native PDB reader tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/CodeView/DebugStringTableSubsection.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFZ.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptorBuilder.h"
#include "llvm/DebugInfo/PDB/Native/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <system_error>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

extern const char *TestMainArgv0;

static std::string getInputPath(StringRef FileName) {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  sys::path::append(InputsDir, FileName);
  return std::string(InputsDir);
}

static std::string getPdbPath() { return getInputPath("SimpleTest.pdb"); }

TEST(MSFZReaderTest, NativeSessionReadsPdbRsFixture) {
  // Generated from SimpleTest.pdb by pdb-rs pdbtool commit
  // 44ff1d3543f1811481c06727baa70e99158c1f98:
  // pdbtool pdz-encode SimpleTest.pdb SimpleTest.pdz --verify --pad16k
  std::string PdzPath = getInputPath("SimpleTest.pdz");
  auto BufferOrErr = MemoryBuffer::getFile(PdzPath, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(BufferOrErr);

  std::unique_ptr<IPDBSession> Session;
  ASSERT_THAT_ERROR(
      NativeSession::createFromPdb(std::move(*BufferOrErr), Session),
      Succeeded());

  auto &Native = static_cast<NativeSession &>(*Session);
  PDBFile &MSFZ = Native.getPDBFile();
  EXPECT_TRUE(MSFZ.isMSFZ());
  EXPECT_EQ(MSFZ.getNumStreams(), 16u);

  auto Tpi = MSFZ.getPDBTpiStream();
  ASSERT_THAT_EXPECTED(Tpi, Succeeded());
  EXPECT_EQ(Tpi->getType(TypeIndex(0x1000)).kind(), LF_ARGLIST);
}

TEST(MSFZReaderTest, NativeSessionReadsLogicalStreams) {
  std::string PdbPath = getPdbPath();
  auto BufferOrErr = MemoryBuffer::getFile(PdbPath, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(BufferOrErr);

  BumpPtrAllocator Allocator;
  auto Input = std::make_unique<MemoryBufferByteStream>(
      std::move(*BufferOrErr), llvm::endianness::little);
  auto Classic =
      std::make_unique<PDBFile>(PdbPath, std::move(Input), Allocator);
  ASSERT_THAT_ERROR(Classic->parseFileHeaders(), Succeeded());
  ASSERT_THAT_ERROR(Classic->parseStreamData(), Succeeded());

  ArrayRef<uint8_t> ClassicData;
  ASSERT_THAT_ERROR(
      Classic->getMsfBuffer().readBytes(0, Classic->getFileSize(), ClassicData),
      Succeeded());

  unittest::TempDir TestDirectory("msfz-native-reader", /*Unique=*/true);
  SmallString<128> MSFZPath = TestDirectory.path("SimpleTest.pdz");
  ASSERT_THAT_ERROR(
      msf::writeMSFZ(MSFZPath, Classic->getMsfLayout(), ClassicData),
      Succeeded());

  auto MSFZBufferOrErr =
      MemoryBuffer::getFile(MSFZPath, /*IsText=*/false,
                            /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(MSFZBufferOrErr);
  std::unique_ptr<IPDBSession> Session;
  ASSERT_THAT_ERROR(
      NativeSession::createFromPdb(std::move(*MSFZBufferOrErr), Session),
      Succeeded());

  auto &Native = static_cast<NativeSession &>(*Session);
  PDBFile &MSFZ = Native.getPDBFile();
  EXPECT_TRUE(MSFZ.isMSFZ());
  EXPECT_EQ(MSFZ.getNumStreams(), Classic->getNumStreams());

  auto Tpi = MSFZ.getPDBTpiStream();
  ASSERT_THAT_EXPECTED(Tpi, Succeeded());
  EXPECT_EQ(Tpi->getType(TypeIndex(0x1000)).kind(), LF_ARGLIST);
}

static Error writeDbiMSFZ(StringRef Path, unsigned ThreadCount,
                          bool FailProduction = false) {
  parallel::strategy = hardware_concurrency(ThreadCount);

  BumpPtrAllocator Allocator;
  PDBFileBuilder Builder(Allocator);
  if (Error E = Builder.initialize(4096))
    return E;
  for (unsigned I = 0; I != kSpecialStreamCount; ++I) {
    Expected<uint32_t> StreamOrErr = Builder.getMsfBuilder().addStream(0);
    if (!StreamOrErr)
      return StreamOrErr.takeError();
  }

  InfoStreamBuilder &Info = Builder.getInfoBuilder();
  Info.setVersion(PdbRaw_ImplVer::PdbImplVC70);
  Info.setAge(1);
  Info.setSignature(0x12345678);

  DbiStreamBuilder &Dbi = Builder.getDbiBuilder();
  Dbi.setVersionHeader(PdbDbiV70);
  Dbi.setAge(1);

  constexpr uint8_t Symbols[] = {0, 1, 2, 3, 4, 5, 6, 7};
  for (unsigned I = 0; I != 3; ++I) {
    std::string ModuleName = (Twine("module-") + Twine(I)).str();
    Expected<DbiModuleDescriptorBuilder &> ModuleOrErr =
        Dbi.addModuleInfo(ModuleName);
    if (!ModuleOrErr)
      return ModuleOrErr.takeError();
    DbiModuleDescriptorBuilder &Module = *ModuleOrErr;
    Module.setObjFileName(ModuleName + ".obj");
    if (I == 2) {
      Module.addUnmergedSymbols(const_cast<uint8_t *>(Symbols),
                                static_cast<uint32_t>(sizeof(Symbols)));
      Module.setMergeSymbolsCallback(
          nullptr, [](void *, void *SymbolData, BinaryStreamWriter &Writer) {
            return Writer.writeBytes(
                ArrayRef(static_cast<const uint8_t *>(SymbolData), 8));
          });
    } else {
      Module.addSymbolsInBulk(Symbols);
    }
    Module.setStringTableFixups({{0x10203040 + I, sizeof(uint32_t)}});

    auto Strings = std::make_shared<DebugStringTableSubsection>();
    Strings->insert((Twine("string-") + Twine(I)).str());
    Module.addDebugSubsection(std::move(Strings));
  }

  Expected<DbiModuleDescriptorBuilder &> EmptyModuleOrErr =
      Dbi.addModuleInfo("empty-module");
  if (!EmptyModuleOrErr)
    return EmptyModuleOrErr.takeError();
  EmptyModuleOrErr->setObjFileName("empty-module.obj");

  if (FailProduction) {
    Expected<DbiModuleDescriptorBuilder &> ModuleOrErr =
        Dbi.addModuleInfo("failing-module");
    if (!ModuleOrErr)
      return ModuleOrErr.takeError();
    DbiModuleDescriptorBuilder &Module = *ModuleOrErr;
    Module.addUnmergedSymbols(const_cast<uint8_t *>(Symbols),
                              static_cast<uint32_t>(sizeof(Symbols)));
    Module.setMergeSymbolsCallback(
        nullptr, [](void *, void *, BinaryStreamWriter &) {
          return createStringError(std::errc::invalid_argument,
                                   "module production failed");
        });
  }

  codeview::GUID Guid = {};
  return Builder.commitMSFZ(Path, &Guid);
}

TEST(MSFZReaderTest, PdbBuilderPlacesInfoStreamInInitialRead) {
  ThreadPoolStrategy SavedStrategy = parallel::strategy;
  scope_exit RestoreStrategy([&] { parallel::strategy = SavedStrategy; });

  unittest::TempDir TestDirectory("msfz-pdbi-layout", /*Unique=*/true);
  SmallString<128> Path = TestDirectory.path("pdbi.pdz");
  ASSERT_THAT_ERROR(writeDbiMSFZ(Path, 1), Succeeded());

  auto BufferOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false);
  ASSERT_TRUE(BufferOrErr);
  StringRef MSFZBytes = (*BufferOrErr)->getBuffer();
  msf::MSFZHeader Header = {};
  std::memcpy(&Header, MSFZBytes.data(), sizeof(Header));
  ASSERT_EQ(Header.StreamDirectoryCompression,
            static_cast<uint32_t>(msf::MSFZCompression::None));
  StringRef Directory = MSFZBytes.substr(
      Header.StreamDirectoryOffset, Header.StreamDirectorySizeUncompressed);
  support::ulittle32_t StreamSize;
  support::ulittle64_t StreamLocation;
  std::memcpy(&StreamSize, Directory.data() + sizeof(uint32_t),
              sizeof(StreamSize));
  std::memcpy(&StreamLocation, Directory.data() + 2 * sizeof(uint32_t),
              sizeof(StreamLocation));
  EXPECT_GT(uint32_t(StreamSize), 0u);
  EXPECT_EQ(StreamLocation, sizeof(msf::MSFZHeader));
  EXPECT_LE(uint64_t(StreamLocation) + uint32_t(StreamSize), 4096u);
}

TEST(MSFZReaderTest, ParallelDbiMatchesDirectSerialization) {
  if (hardware_concurrency(2).compute_thread_count() < 2)
    GTEST_SKIP();

  ThreadPoolStrategy SavedStrategy = parallel::strategy;
  scope_exit RestoreStrategy([&] { parallel::strategy = SavedStrategy; });

  unittest::TempDir TestDirectory("msfz-dbi-parallel", /*Unique=*/true);
  SmallString<128> DirectPath = TestDirectory.path("direct.pdz");
  ASSERT_THAT_ERROR(writeDbiMSFZ(DirectPath, 1), Succeeded());
  SmallString<128> ParallelPath = TestDirectory.path("parallel.pdz");
  ASSERT_THAT_ERROR(writeDbiMSFZ(ParallelPath, 2), Succeeded());

  auto DirectOrErr = MemoryBuffer::getFile(DirectPath, /*IsText=*/false);
  ASSERT_TRUE(DirectOrErr);
  auto ParallelOrErr = MemoryBuffer::getFile(ParallelPath, /*IsText=*/false);
  ASSERT_TRUE(ParallelOrErr);
  EXPECT_EQ((*DirectOrErr)->getBuffer(), (*ParallelOrErr)->getBuffer());
}

TEST(MSFZReaderTest, ParallelDbiProductionErrorDiscardsOutput) {
  if (hardware_concurrency(4).compute_thread_count() < 2)
    GTEST_SKIP();

  ThreadPoolStrategy SavedStrategy = parallel::strategy;
  scope_exit RestoreStrategy([&] { parallel::strategy = SavedStrategy; });

  unittest::TempDir TestDirectory("msfz-dbi-error", /*Unique=*/true);
  SmallString<128> Path = TestDirectory.path("error.pdz");
  EXPECT_THAT_ERROR(writeDbiMSFZ(Path, 4, /*FailProduction=*/true),
                    FailedWithMessage("module production failed"));
  EXPECT_FALSE(sys::fs::exists(Path));
}
