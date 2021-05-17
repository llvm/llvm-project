//===-- TextStubV2Tests.cpp - TBD V2 File Test ----------------------------===//
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

static ExportedSymbol TBDv2Symbols[] = {
    {SymbolKind::GlobalSymbol, "$ld$hide$os9.0$_sym1", false, false},
    {SymbolKind::GlobalSymbol, "_sym1", false, false},
    {SymbolKind::GlobalSymbol, "_sym2", false, false},
    {SymbolKind::GlobalSymbol, "_sym3", false, false},
    {SymbolKind::GlobalSymbol, "_sym4", false, false},
    {SymbolKind::GlobalSymbol, "_sym5", false, false},
    {SymbolKind::GlobalSymbol, "_tlv1", false, true},
    {SymbolKind::GlobalSymbol, "_tlv2", false, true},
    {SymbolKind::GlobalSymbol, "_tlv3", false, true},
    {SymbolKind::GlobalSymbol, "_weak1", true, false},
    {SymbolKind::GlobalSymbol, "_weak2", true, false},
    {SymbolKind::GlobalSymbol, "_weak3", true, false},
    {SymbolKind::ObjectiveCClass, "class1", false, false},
    {SymbolKind::ObjectiveCClass, "class2", false, false},
    {SymbolKind::ObjectiveCClass, "class3", false, false},
    {SymbolKind::ObjectiveCInstanceVariable, "class1._ivar1", false, false},
    {SymbolKind::ObjectiveCInstanceVariable, "class1._ivar2", false, false},
    {SymbolKind::ObjectiveCInstanceVariable, "class1._ivar3", false, false},
};

namespace TBDv2 {

TEST(TBDv2, ReadFile) {
  static const char TBDv2File1[] =
      "--- !tapi-tbd-v2\n"
      "archs: [ armv7, armv7s, armv7k, arm64 ]\n"
      "platform: ios\n"
      "flags: [ installapi ]\n"
      "install-name: Test.dylib\n"
      "current-version: 2.3.4\n"
      "compatibility-version: 1.0\n"
      "swift-version: 1.1\n"
      "parent-umbrella: Umbrella.dylib\n"
      "exports:\n"
      "  - archs: [ armv7, armv7s, armv7k, arm64 ]\n"
      "    allowable-clients: [ clientA ]\n"
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
      TextAPIReader::get(MemoryBufferRef(TBDv2File1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  auto Archs = AK_armv7 | AK_armv7s | AK_armv7k | AK_arm64;
  auto Platform = PlatformKind::iOS;
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
  EXPECT_EQ(ObjCConstraintType::Retain_Release, File->getObjCConstraint());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_TRUE(File->isInstallAPI());
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
  llvm::sort(Exports.begin(), Exports.end());

  EXPECT_EQ(sizeof(TBDv2Symbols) / sizeof(ExportedSymbol), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(TBDv2Symbols)));
}

TEST(TBDv2, ReadFile2) {
  static const char TBDv2File2[] =
      "--- !tapi-tbd-v2\n"
      "archs: [ armv7, armv7s, armv7k, arm64 ]\n"
      "platform: ios\n"
      "flags: [ flat_namespace, not_app_extension_safe ]\n"
      "install-name: Test.dylib\n"
      "swift-version: 1.1\n"
      "exports:\n"
      "  - archs: [ armv7, armv7s, armv7k, arm64 ]\n"
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
      "undefineds:\n"
      "  - archs: [ armv7, armv7s, armv7k, arm64 ]\n"
      "    symbols: [ _undefSym1, _undefSym2, _undefSym3 ]\n"
      "    objc-classes: [ _undefClass1, _undefClass2 ]\n"
      "    objc-ivars: [ _undefClass1._ivar1, _undefClass1._ivar2 ]\n"
      "    weak-ref-symbols: [ _undefWeak1, _undefWeak2 ]\n"
      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2File2, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  auto Archs = AK_armv7 | AK_armv7s | AK_armv7k | AK_arm64;
  auto Platform = PlatformKind::iOS;
  TargetList Targets;
  for (auto &&arch : Archs)
    Targets.emplace_back(Target(arch, Platform));
  EXPECT_EQ(Archs, File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
  EXPECT_EQ(std::string("Test.dylib"), File->getInstallName());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCompatibilityVersion());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
  EXPECT_EQ(ObjCConstraintType::Retain_Release, File->getObjCConstraint());
  EXPECT_FALSE(File->isTwoLevelNamespace());
  EXPECT_FALSE(File->isApplicationExtensionSafe());
  EXPECT_FALSE(File->isInstallAPI());
  EXPECT_EQ(0U, File->allowableClients().size());
  EXPECT_EQ(0U, File->reexportedLibraries().size());
}

TEST(TBDv2, WriteFile) {
  static const char TBDv2File3[] =
      "--- !tapi-tbd-v2\n"
      "archs:           [ i386, x86_64 ]\n"
      "platform:        macosx\n"
      "install-name:    '/usr/lib/libfoo.dylib'\n"
      "current-version: 1.2.3\n"
      "compatibility-version: 0\n"
      "swift-version:   5\n"
      "exports:\n"
      "  - archs:           [ i386 ]\n"
      "    symbols:         [ _sym1 ]\n"
      "    weak-def-symbols: [ _sym2 ]\n"
      "    thread-local-symbols: [ _sym3 ]\n"
      "  - archs:           [ x86_64 ]\n"
      "    allowable-clients: [ clientA ]\n"
      "    re-exports:      [ '/usr/lib/libfoo.dylib' ]\n"
      "    symbols:         [ '_OBJC_EHTYPE_$_Class1' ]\n"
      "    objc-classes:    [ _Class1 ]\n"
      "    objc-ivars:      [ _Class1._ivar1 ]\n"
      "...\n";

  InterfaceFile File;
  TargetList Targets;
  for (auto &&arch : AK_i386 | AK_x86_64)
    Targets.emplace_back(Target(arch, PlatformKind::macOS));
  File.setPath("libfoo.dylib");
  File.setInstallName("/usr/lib/libfoo.dylib");
  File.setFileType(FileType::TBD_V2);
  File.addTargets(Targets);
  File.setCurrentVersion(PackedVersion(1, 2, 3));
  File.setTwoLevelNamespace();
  File.setApplicationExtensionSafe();
  File.setSwiftABIVersion(5);
  File.setObjCConstraint(ObjCConstraintType::Retain_Release);
  File.addAllowableClient("clientA", Targets[1]);
  File.addReexportedLibrary("/usr/lib/libfoo.dylib", Targets[1]);
  File.addSymbol(SymbolKind::GlobalSymbol, "_sym1", {Targets[0]});
  File.addSymbol(SymbolKind::GlobalSymbol, "_sym2", {Targets[0]},
                 SymbolFlags::WeakDefined);
  File.addSymbol(SymbolKind::GlobalSymbol, "_sym3", {Targets[0]},
                 SymbolFlags::ThreadLocalValue);
  File.addSymbol(SymbolKind::ObjectiveCClass, "Class1", {Targets[1]});
  File.addSymbol(SymbolKind::ObjectiveCClassEHType, "Class1", {Targets[1]});
  File.addSymbol(SymbolKind::ObjectiveCInstanceVariable, "Class1._ivar1",
                 {Targets[1]});

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error Result = TextAPIWriter::writeToStream(OS, File);
  EXPECT_FALSE(Result);
  EXPECT_STREQ(TBDv2File3, Buffer.c_str());
}

TEST(TBDv2, Platform_macOS) {
  static const char TBDv2PlatformMacOS[] = "--- !tapi-tbd-v2\n"
                                           "archs: [ x86_64 ]\n"
                                           "platform: macosx\n"
                                           "install-name: Test.dylib\n"
                                           "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2PlatformMacOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  auto Platform = PlatformKind::macOS;
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv2, Platform_iOS) {
  static const char TBDv2PlatformiOS[] = "--- !tapi-tbd-v2\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2PlatformiOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::iOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv2, Platform_watchOS) {
  static const char TBDv2PlatformWatchOS[] = "--- !tapi-tbd-v2\n"
                                             "archs: [ armv7k ]\n"
                                             "platform: watchos\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2PlatformWatchOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::watchOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv2, Platform_tvOS) {
  static const char TBDv2PlatformtvOS[] = "--- !tapi-tbd-v2\n"
                                          "archs: [ arm64 ]\n"
                                          "platform: tvos\n"
                                          "install-name: Test.dylib\n"
                                          "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2PlatformtvOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::tvOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv2, Platform_bridgeOS) {
  static const char TBDv2BridgeOS[] = "--- !tapi-tbd-v2\n"
                                      "archs: [ armv7k ]\n"
                                      "platform: bridgeos\n"
                                      "install-name: Test.dylib\n"
                                      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2BridgeOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::bridgeOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv2, Swift_1_0) {
  static const char TBDv2Swift1[] = "--- !tapi-tbd-v2\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 1.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2Swift1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(1U, File->getSwiftABIVersion());
}

TEST(TBDv2, Swift_1_1) {
  static const char TBDv2Swift1dot[] = "--- !tapi-tbd-v2\n"
                                       "archs: [ arm64 ]\n"
                                       "platform: ios\n"
                                       "install-name: Test.dylib\n"
                                       "swift-version: 1.1\n"
                                       "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2Swift1dot, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
}

TEST(TBDv2, Swift_2_0) {
  static const char tbd_v2_swift_2_0[] = "--- !tapi-tbd-v2\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "swift-version: 2.0\n"
                                         "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v2_swift_2_0, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(3U, File->getSwiftABIVersion());
}

TEST(TBDv2, Swift_3_0) {
  static const char TBDv2Swift3[] = "--- !tapi-tbd-v2\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 3.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2Swift3, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(4U, File->getSwiftABIVersion());
}

TEST(TBDv2, Swift_4_0) {
  static const char TBDv2Swift4[] = "--- !tapi-tbd-v2\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 4.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2Swift4, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:5:16: error: invalid Swift ABI "
            "version.\nswift-version: 4.0\n               ^~~\n",
            ErrorMessage);
}

TEST(TBDv2, Swift_5) {
  static const char TBDv2Swift5[] = "--- !tapi-tbd-v2\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-version: 5\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2Swift5, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(5U, File->getSwiftABIVersion());
}

TEST(TBDv2, Swift_99) {
  static const char TBDv2Swift99[] = "--- !tapi-tbd-v2\n"
                                     "archs: [ arm64 ]\n"
                                     "platform: ios\n"
                                     "install-name: Test.dylib\n"
                                     "swift-version: 99\n"
                                     "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2Swift99, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V2, File->getFileType());
  EXPECT_EQ(99U, File->getSwiftABIVersion());
}

TEST(TBDv2, UnknownArchitecture) {
  static const char TBDv2FileUnknownArch[] = "--- !tapi-tbd-v2\n"
                                             "archs: [ foo ]\n"
                                             "platform: macosx\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";
  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2FileUnknownArch, "Test.tbd"));
  EXPECT_TRUE(!!Result);
}

TEST(TBDv2, UnknownPlatform) {
  static const char TBDv2FileUnknownPlatform[] = "--- !tapi-tbd-v2\n"
                                                 "archs: [ i386 ]\n"
                                                 "platform: newOS\n"
                                                 "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2FileUnknownPlatform, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:11: error: unknown platform\nplatform: "
            "newOS\n          ^~~~~\n",
            ErrorMessage);
}

TEST(TBDv2, InvalidPlatform) {
  static const char TBDv2FileInvalidPlatform[] = "--- !tapi-tbd-v2\n"
                                                 "archs: [ i386 ]\n"
                                                 "platform: iosmac\n"
                                                 "install-name: Test.dylib\n"
                                                 "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2FileInvalidPlatform, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:11: error: invalid platform\nplatform: "
            "iosmac\n          ^~~~~~\n",
            ErrorMessage);
}

TEST(TBDv2, MalformedFile1) {
  static const char TBDv2FileMalformed1[] = "--- !tapi-tbd-v2\n"
                                            "archs: [ arm64 ]\n"
                                            "foobar: \"Unsupported key\"\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2FileMalformed1, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  ASSERT_EQ("malformed file\nTest.tbd:2:1: error: missing required key "
            "'platform'\narchs: [ arm64 ]\n^\n",
            ErrorMessage);
}

TEST(TBDv2, MalformedFile2) {
  static const char TBDv2FileMalformed2[] = "--- !tapi-tbd-v2\n"
                                            "archs: [ arm64 ]\n"
                                            "platform: ios\n"
                                            "install-name: Test.dylib\n"
                                            "foobar: \"Unsupported key\"\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv2FileMalformed2, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  ASSERT_EQ(
      "malformed file\nTest.tbd:5:1: error: unknown key 'foobar'\nfoobar: "
      "\"Unsupported key\"\n^~~~~~\n",
      ErrorMessage);
}

} // namespace TBDv2
