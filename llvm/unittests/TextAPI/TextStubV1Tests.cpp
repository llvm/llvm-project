//===-- TextStubV1Tests.cpp - TBD V1 File Test ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "TextStubHelpers.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/TextAPIReader.h"
#include "llvm/TextAPI/TextAPIWriter.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::MachO;

static ExportedSymbol TBDv1Symbols[] = {
    {EncodeKind::GlobalSymbol, "$ld$hide$os9.0$_sym1", false, false},
    {EncodeKind::GlobalSymbol, "_sym1", false, false},
    {EncodeKind::GlobalSymbol, "_sym2", false, false},
    {EncodeKind::GlobalSymbol, "_sym3", false, false},
    {EncodeKind::GlobalSymbol, "_sym4", false, false},
    {EncodeKind::GlobalSymbol, "_sym5", false, false},
    {EncodeKind::GlobalSymbol, "_tlv1", false, true},
    {EncodeKind::GlobalSymbol, "_tlv2", false, true},
    {EncodeKind::GlobalSymbol, "_tlv3", false, true},
    {EncodeKind::GlobalSymbol, "_weak1", true, false},
    {EncodeKind::GlobalSymbol, "_weak2", true, false},
    {EncodeKind::GlobalSymbol, "_weak3", true, false},
    {EncodeKind::ObjectiveCClass, "class1", false, false},
    {EncodeKind::ObjectiveCClass, "class2", false, false},
    {EncodeKind::ObjectiveCClass, "class3", false, false},
    {EncodeKind::ObjectiveCInstanceVariable, "class1._ivar1", false, false},
    {EncodeKind::ObjectiveCInstanceVariable, "class1._ivar2", false, false},
    {EncodeKind::ObjectiveCInstanceVariable, "class1._ivar3", false, false},
};

namespace TBDv1 {

TEST(TBDv1, ReadFile) {
  static const char TBDv1File1[] =
      "---\n"
      "archs: [ armv7, armv7s, armv7k, arm64 ]\n"
      "platform: ios\n"
      "install-name: Test.dylib\n"
      "current-version: 2.3.4\n"
      "compatibility-version: 1.0\n"
      "swift-version: 1.1\n"
      "exports:\n"
      "  - archs: [ armv7, armv7s, armv7k, arm64 ]\n"
      "    allowed-clients: [ clientA ]\n"
      "    re-exports: [ /usr/lib/libfoo.dylib ]\n"
      "    symbols: [ _sym1, _sym2, _sym3, _sym4, $ld$hide$os9.0$_sym1 ]\n"
      "    objc-classes: [ _class1, _class2 ]\n"
      "    objc-ivars: [ _class1._ivar1, _class1._ivar2 ]\n"
      "    weak-def-symbols: [ _weak1, _weak2 ]\n"
      "    thread-local-symbols: [ _tlv1, _tlv2 ]\n"
      "  - archs: [ armv7, armv7s, armv7k ]\n"
      "    symbols: [ _sym5 ]\n"
      "    objc-classes: [ _class3 ]\n"
      "    objc-ivars: [ _class1._ivar3 ]\n"
      "    weak-def-symbols: [ _weak3 ]\n"
      "    thread-local-symbols: [ _tlv3 ]\n"
      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1File1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  auto Archs = AK_armv7 | AK_armv7s | AK_armv7k | AK_arm64;
  auto Platform = PLATFORM_IOS;
  TargetList Targets;
  for (auto &&arch : Archs)
    Targets.emplace_back(Target(arch, Platform));
  EXPECT_EQ(Archs, File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
  EXPECT_EQ(std::string("Test.dylib"), File->getInstallName());
  EXPECT_EQ(PackedVersion(2, 3, 4), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCompatibilityVersion());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
  EXPECT_EQ(ObjCConstraintType::None, File->getObjCConstraint());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  InterfaceFileRef client("clientA", Targets);
  InterfaceFileRef reexport("/usr/lib/libfoo.dylib", Targets);
  EXPECT_EQ(1U, File->allowableClients().size());
  EXPECT_EQ(client, File->allowableClients().front());
  EXPECT_EQ(1U, File->reexportedLibraries().size());
  EXPECT_EQ(reexport, File->reexportedLibraries().front());

  ExportedSymbolSeq Exports;
  for (const auto *Sym : File->symbols()) {
    EXPECT_FALSE(Sym->isWeakReferenced());
    EXPECT_FALSE(Sym->isUndefined());
    Exports.emplace_back(
        ExportedSymbol{Sym->getKind(), std::string(Sym->getName()),
                       Sym->isWeakDefined(), Sym->isThreadLocalValue()});
  }
  llvm::sort(Exports);

  EXPECT_EQ(std::size(TBDv1Symbols), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(TBDv1Symbols)));

  File->addSymbol(EncodeKind::ObjectiveCClassEHType, "Class1", {Targets[1]});
  File->addSymbol(EncodeKind::ObjectiveCInstanceVariable, "Class1._ivar1",
                  {Targets[1]});
}

TEST(TBDv1, ReadFile2) {
  static const char TBDv1File2[] = "--- !tapi-tbd-v1\n"
                                   "archs: [ armv7, armv7s, armv7k, arm64 ]\n"
                                   "platform: ios\n"
                                   "install-name: Test.dylib\n"
                                   "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1File2, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  auto Archs = AK_armv7 | AK_armv7s | AK_armv7k | AK_arm64;
  auto Platform = PLATFORM_IOS;
  TargetList Targets;
  for (auto &&arch : Archs)
    Targets.emplace_back(Target(arch, Platform));
  EXPECT_EQ(Archs, File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
  EXPECT_EQ(std::string("Test.dylib"), File->getInstallName());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCompatibilityVersion());
  EXPECT_EQ(0U, File->getSwiftABIVersion());
  EXPECT_EQ(ObjCConstraintType::None, File->getObjCConstraint());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_EQ(0U, File->allowableClients().size());
  EXPECT_EQ(0U, File->reexportedLibraries().size());
}

TEST(TBDv1, WriteFile) {
  static const char TBDv1File3[] =
      "---\n"
      "archs:           [ i386, x86_64 ]\n"
      "platform:        macosx\n"
      "install-name:    '/usr/lib/libfoo.dylib'\n"
      "current-version: 1.2.3\n"
      "compatibility-version: 0\n"
      "swift-version:   5\n"
      "objc-constraint: retain_release\n"
      "exports:\n"
      "  - archs:           [ i386 ]\n"
      "    symbols:         [ _sym1 ]\n"
      "    weak-def-symbols: [ _sym2 ]\n"
      "    thread-local-symbols: [ _sym3 ]\n"
      "  - archs:           [ x86_64 ]\n"
      "    allowed-clients: [ clientA ]\n"
      "    re-exports:      [ '/usr/lib/libfoo.dylib' ]\n"
      "    symbols:         [ '_OBJC_EHTYPE_$_Class1' ]\n"
      "    objc-classes:    [ _Class1 ]\n"
      "    objc-ivars:      [ _Class1._ivar1 ]\n"
      "...\n";

  InterfaceFile File;
  TargetList Targets;
  for (auto &&arch : AK_i386 | AK_x86_64)
    Targets.emplace_back(Target(arch, PLATFORM_MACOS));
  File.setPath("libfoo.dylib");
  File.setInstallName("/usr/lib/libfoo.dylib");
  File.setFileType(FileType::TBD_V1);
  File.addTargets(Targets);
  File.setCurrentVersion(PackedVersion(1, 2, 3));
  File.setSwiftABIVersion(5);
  File.setObjCConstraint(ObjCConstraintType::Retain_Release);
  File.addAllowableClient("clientA", Targets[1]);
  File.addReexportedLibrary("/usr/lib/libfoo.dylib", Targets[1]);
  File.addSymbol(EncodeKind::GlobalSymbol, "_sym1", {Targets[0]});
  File.addSymbol(EncodeKind::GlobalSymbol, "_sym2", {Targets[0]},
                 SymbolFlags::WeakDefined);
  File.addSymbol(EncodeKind::GlobalSymbol, "_sym3", {Targets[0]},
                 SymbolFlags::ThreadLocalValue);
  File.addSymbol(EncodeKind::ObjectiveCClass, "Class1", {Targets[1]});
  File.addSymbol(EncodeKind::ObjectiveCClassEHType, "Class1", {Targets[1]});
  File.addSymbol(EncodeKind::ObjectiveCInstanceVariable, "Class1._ivar1",
                 {Targets[1]});

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error Result = TextAPIWriter::writeToStream(OS, File);
  EXPECT_FALSE(Result);
  EXPECT_STREQ(TBDv1File3, Buffer.c_str());
}

TEST(TBDv1, Platform_macOS) {
  static const char TBDv1PlatformMacOS[] = "---\n"
                                           "archs: [ x86_64 ]\n"
                                           "platform: macosx\n"
                                           "install-name: Test.dylib\n"
                                           "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1PlatformMacOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_MACOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_iOS) {
  static const char TBDv1PlatformiOS[] = "---\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1PlatformiOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_IOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_watchOS) {
  static const char TBDv1PlatformWatchOS[] = "---\n"
                                             "archs: [ armv7k ]\n"
                                             "platform: watchos\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1PlatformWatchOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_WATCHOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_tvOS) {
  static const char TBDv1PlatformtvOS[] = "---\n"
                                          "archs: [ arm64 ]\n"
                                          "platform: tvos\n"
                                          "install-name: Test.dylib\n"
                                          "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1PlatformtvOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_TVOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_bridgeOS) {
  static const char TBDv1BridgeOS[] = "---\n"
                                      "archs: [ armv7k ]\n"
                                      "platform: bridgeos\n"
                                      "install-name: Test.dylib\n"
                                      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1BridgeOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_BRIDGEOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Swift_1_0) {
  static const char TBDv1Swift1[] = "---\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 1.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(1U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_1_1) {
  static const char TBDv1Swift1dot[] = "---\n"
                                       "archs: [ arm64 ]\n"
                                       "platform: ios\n"
                                       "install-name: Test.dylib\n"
                                       "swift-version: 1.1\n"
                                       "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift1dot, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_2_0) {
  static const char TBDv1Swift2[] = "---\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 2.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift2, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(3U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_3_0) {
  static const char TBDv1Swift3[] = "---\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 3.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift3, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(4U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_4_0) {
  static const char TBDv1Swift4[] = "---\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 4.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift4, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:5:16: error: invalid Swift ABI "
            "version.\nswift-version: 4.0\n               ^~~\n",
            ErrorMessage);
}

TEST(TBDv1, Swift_5) {
  static const char TBDv1Swift5[] = "---\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 5\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift5, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(5U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_99) {
  static const char TBDv1Swift99[] = "---\n"
                                     "archs: [ arm64 ]\n"
                                     "platform: ios\n"
                                     "install-name: Test.dylib\n"
                                     "swift-version: 99\n"
                                     "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1Swift99, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(99U, File->getSwiftABIVersion());
}

TEST(TBDv1, UnknownArchitecture) {
  static const char TBDv1FileUnknownArch[] = "---\n"
                                             "archs: [ foo ]\n"
                                             "platform: macosx\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1FileUnknownArch, "Test.tbd"));
  EXPECT_TRUE(!!Result);
}

TEST(TBDv1, UnknownPlatform) {
  static const char TBDv1FileUnknownPlatform[] = "---\n"
                                                 "archs: [ i386 ]\n"
                                                 "platform: newOS\n"
                                                 "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1FileUnknownPlatform, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:11: error: unknown platform\nplatform: "
            "newOS\n          ^~~~~\n",
            ErrorMessage);
}

TEST(TBDv1, MalformedFile1) {
  static const char TBDv1FileMalformed1[] = "---\n"
                                            "archs: [ arm64 ]\n"
                                            "foobar: \"Unsupported key\"\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1FileMalformed1, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  ASSERT_EQ("malformed file\nTest.tbd:2:1: error: missing required key "
            "'platform'\narchs: [ arm64 ]\n^\n",
            ErrorMessage);
}

TEST(TBDv1, MalformedFile2) {
  static const char TBDv1FileMalformed2[] = "---\n"
                                            "archs: [ arm64 ]\n"
                                            "platform: ios\n"
                                            "install-name: Test.dylib\n"
                                            "foobar: \"Unsupported key\"\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv1FileMalformed2, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  ASSERT_EQ(
      "malformed file\nTest.tbd:5:1: error: unknown key 'foobar'\nfoobar: "
      "\"Unsupported key\"\n^~~~~~\n",
      ErrorMessage);
}

} // end namespace TBDv1.
