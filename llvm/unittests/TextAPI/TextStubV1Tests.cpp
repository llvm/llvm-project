//===-- TextStubV1Tests.cpp - TBD V1 File Test ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"
#include "llvm/TextAPI/MachO/TextAPIWriter.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::MachO;

struct ExportedSymbol {
  SymbolKind Kind;
  std::string Name;
  bool WeakDefined;
  bool ThreadLocalValue;
};
using ExportedSymbolSeq = std::vector<ExportedSymbol>;

inline bool operator<(const ExportedSymbol &lhs, const ExportedSymbol &rhs) {
  return std::tie(lhs.Kind, lhs.Name) < std::tie(rhs.Kind, rhs.Name);
}

inline bool operator==(const ExportedSymbol &lhs, const ExportedSymbol &rhs) {
  return std::tie(lhs.Kind, lhs.Name, lhs.WeakDefined, lhs.ThreadLocalValue) ==
         std::tie(rhs.Kind, rhs.Name, rhs.WeakDefined, rhs.ThreadLocalValue);
}

static ExportedSymbol TBDv1Symbols[] = {
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

namespace TBDv1 {

TEST(TBDv1, ReadFile) {
  static const char tbd_v1_file1[] =
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

  auto Result = TextAPIReader::get(MemoryBufferRef(tbd_v1_file1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
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
  EXPECT_EQ(ObjCConstraintType::None, File->getObjCConstraint());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_FALSE(File->isInstallAPI());
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
    Exports.emplace_back(ExportedSymbol{Sym->getKind(), Sym->getName(),
                                        Sym->isWeakDefined(),
                                        Sym->isThreadLocalValue()});
  }
  llvm::sort(Exports.begin(), Exports.end());

  EXPECT_EQ(sizeof(TBDv1Symbols) / sizeof(ExportedSymbol), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(TBDv1Symbols)));

  File->addSymbol(SymbolKind::ObjectiveCClassEHType, "Class1", {Targets[1]});
  File->addSymbol(SymbolKind::ObjectiveCInstanceVariable, "Class1._ivar1",
                  {Targets[1]});
}

TEST(TBDv1, ReadFile2) {
  static const char tbd_v1_file2[] = "--- !tapi-tbd-v1\n"
                                     "archs: [ armv7, armv7s, armv7k, arm64 ]\n"
                                     "platform: ios\n"
                                     "install-name: Test.dylib\n"
                                     "...\n";

  auto Result = TextAPIReader::get(MemoryBufferRef(tbd_v1_file2, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
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
  EXPECT_EQ(0U, File->getSwiftABIVersion());
  EXPECT_EQ(ObjCConstraintType::None, File->getObjCConstraint());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_FALSE(File->isInstallAPI());
  EXPECT_EQ(0U, File->allowableClients().size());
  EXPECT_EQ(0U, File->reexportedLibraries().size());
}

TEST(TBDv1, WriteFile) {
  static const char tbd_v1_file3[] =
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
    Targets.emplace_back(Target(arch, PlatformKind::macOS));
  File.setPath("libfoo.dylib");
  File.setInstallName("/usr/lib/libfoo.dylib");
  File.setFileType(FileType::TBD_V1);
  File.addTargets(Targets);
  File.setCurrentVersion(PackedVersion(1, 2, 3));
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
  auto Result = TextAPIWriter::writeToStream(OS, File);
  EXPECT_FALSE(Result);
  EXPECT_STREQ(tbd_v1_file3, Buffer.c_str());
}

TEST(TBDv1, Platform_macOS) {
  static const char tbd_v1_platform_macos[] = "---\n"
                                              "archs: [ x86_64 ]\n"
                                              "platform: macosx\n"
                                              "install-name: Test.dylib\n"
                                              "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_platform_macos, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::macOS;
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_iOS) {
  static const char tbd_v1_platform_ios[] = "---\n"
                                            "archs: [ arm64 ]\n"
                                            "platform: ios\n"
                                            "install-name: Test.dylib\n"
                                            "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_platform_ios, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::iOS;
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_watchOS) {
  static const char tbd_v1_platform_watchos[] = "---\n"
                                                "archs: [ armv7k ]\n"
                                                "platform: watchos\n"
                                                "install-name: Test.dylib\n"
                                                "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_platform_watchos, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::watchOS;
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_tvOS) {
  static const char tbd_v1_platform_tvos[] = "---\n"
                                             "archs: [ arm64 ]\n"
                                             "platform: tvos\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_platform_tvos, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::tvOS;
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Platform_bridgeOS) {
  static const char tbd_v1_platform_bridgeos[] = "---\n"
                                                 "archs: [ armv7k ]\n"
                                                 "platform: bridgeos\n"
                                                 "install-name: Test.dylib\n"
                                                 "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_platform_bridgeos, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PlatformKind::bridgeOS;
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
}

TEST(TBDv1, Swift_1_0) {
  static const char tbd_v1_swift_1_0[] = "---\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "swift-version: 1.0\n"
                                         "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_1_0, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(1U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_1_1) {
  static const char tbd_v1_swift_1_1[] = "---\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "swift-version: 1.1\n"
                                         "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_1_1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_2_0) {
  static const char tbd_v1_swift_2_0[] = "---\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "swift-version: 2.0\n"
                                         "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_2_0, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(3U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_3_0) {
  static const char tbd_v1_swift_3_0[] = "---\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "swift-version: 3.0\n"
                                         "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_3_0, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(4U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_4_0) {
  static const char tbd_v1_swift_4_0[] = "---\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "swift-version: 4.0\n"
                                         "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_4_0, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:5:16: error: invalid Swift ABI "
            "version.\nswift-version: 4.0\n               ^~~\n",
            errorMessage);
}

TEST(TBDv1, Swift_5) {
  static const char tbd_v1_swift_5[] = "---\n"
                                       "archs: [ arm64 ]\n"
                                       "platform: ios\n"
                                       "install-name: Test.dylib\n"
                                       "swift-version: 5\n"
                                       "...\n";

  auto Result = TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_5, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(5U, File->getSwiftABIVersion());
}

TEST(TBDv1, Swift_99) {
  static const char tbd_v1_swift_99[] = "---\n"
                                        "archs: [ arm64 ]\n"
                                        "platform: ios\n"
                                        "install-name: Test.dylib\n"
                                        "swift-version: 99\n"
                                        "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(tbd_v1_swift_99, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V1, File->getFileType());
  EXPECT_EQ(99U, File->getSwiftABIVersion());
}

TEST(TBDv1, UnknownArchitecture) {
  static const char tbd_v1_file_unknown_architecture[] =
      "---\n"
      "archs: [ foo ]\n"
      "platform: macosx\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result = TextAPIReader::get(
      MemoryBufferRef(tbd_v1_file_unknown_architecture, "Test.tbd"));
  EXPECT_TRUE(!!Result);
}

TEST(TBDv1, UnknownPlatform) {
  static const char tbd_v1_file_unknown_platform[] = "---\n"
                                                     "archs: [ i386 ]\n"
                                                     "platform: newOS\n"
                                                     "...\n";

  auto Result = TextAPIReader::get(
      MemoryBufferRef(tbd_v1_file_unknown_platform, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:11: error: unknown platform\nplatform: "
            "newOS\n          ^~~~~\n",
            errorMessage);
}

TEST(TBDv1, MalformedFile1) {
  static const char malformed_file1[] = "---\n"
                                        "archs: [ arm64 ]\n"
                                        "foobar: \"Unsupported key\"\n"
                                        "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(malformed_file1, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  ASSERT_EQ("malformed file\nTest.tbd:2:1: error: missing required key "
            "'platform'\narchs: [ arm64 ]\n^\n",
            errorMessage);
}

TEST(TBDv1, MalformedFile2) {
  static const char malformed_file2[] = "---\n"
                                        "archs: [ arm64 ]\n"
                                        "platform: ios\n"
                                        "install-name: Test.dylib\n"
                                        "foobar: \"Unsupported key\"\n"
                                        "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(malformed_file2, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  ASSERT_EQ(
      "malformed file\nTest.tbd:5:9: error: unknown key 'foobar'\nfoobar: "
      "\"Unsupported key\"\n        ^~~~~~~~~~~~~~~~~\n",
      errorMessage);
}

} // end namespace TBDv1.
