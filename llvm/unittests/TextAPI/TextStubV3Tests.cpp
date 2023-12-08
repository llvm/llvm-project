//===-- TextStubV3Tests.cpp - TBD V3 File Test ----------------------------===//
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

static ExportedSymbol TBDv3Symbols[] = {
    {SymbolKind::GlobalSymbol, "$ld$hide$os9.0$_sym1", false, false},
    {SymbolKind::GlobalSymbol, "_sym1", false, false},
    {SymbolKind::GlobalSymbol, "_sym2", false, false},
    {SymbolKind::GlobalSymbol, "_sym3", false, false},
    {SymbolKind::GlobalSymbol, "_sym4", false, false},
    {SymbolKind::GlobalSymbol, "_sym5", false, false},
    {SymbolKind::GlobalSymbol, "_tlv1", false, true},
    {SymbolKind::GlobalSymbol, "_tlv3", false, true},
    {SymbolKind::GlobalSymbol, "_weak1", true, false},
    {SymbolKind::GlobalSymbol, "_weak2", true, false},
    {SymbolKind::GlobalSymbol, "_weak3", true, false},
    {SymbolKind::ObjectiveCClass, "class1", false, false},
    {SymbolKind::ObjectiveCClass, "class2", false, false},
    {SymbolKind::ObjectiveCClass, "class3", false, false},
    {SymbolKind::ObjectiveCClassEHType, "class1", false, false},
    {SymbolKind::ObjectiveCInstanceVariable, "class1._ivar1", false, false},
    {SymbolKind::ObjectiveCInstanceVariable, "class1._ivar2", false, false},
    {SymbolKind::ObjectiveCInstanceVariable, "class1._ivar3", false, false},
};

namespace TBDv3 {

TEST(TBDv3, ReadFile) {
  static const char TBDv3File1[] =
      "--- !tapi-tbd-v3\n"
      "archs: [ armv7, arm64 ]\n"
      "uuids: [ 'armv7: 00000000-0000-0000-0000-000000000000',\n"
      "         'arm64: 11111111-1111-1111-1111-111111111111']\n"
      "platform: ios\n"
      "flags: [ installapi ]\n"
      "install-name: Test.dylib\n"
      "current-version: 2.3.4\n"
      "compatibility-version: 1.0\n"
      "swift-abi-version: 1.1\n"
      "parent-umbrella: Umbrella.dylib\n"
      "exports:\n"
      "  - archs: [ armv7, arm64 ]\n"
      "    allowable-clients: [ clientA ]\n"
      "    re-exports: [ /usr/lib/libfoo.dylib ]\n"
      "    symbols: [ _sym1, _sym2, _sym3, _sym4, $ld$hide$os9.0$_sym1 ]\n"
      "    objc-classes: [ class1, class2 ]\n"
      "    objc-eh-types: [ class1 ]\n"
      "    objc-ivars: [ class1._ivar1, class1._ivar2 ]\n"
      "    weak-def-symbols: [ _weak1, _weak2 ]\n"
      "    thread-local-symbols: [ _tlv1, _tlv3 ]\n"
      "  - archs: [ armv7 ]\n"
      "    symbols: [ _sym5 ]\n"
      "    objc-classes: [ class3 ]\n"
      "    objc-ivars: [ class1._ivar3 ]\n"
      "    weak-def-symbols: [ _weak3 ]\n"
      "    thread-local-symbols: [ _tlv3 ]\n"
      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3File1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  auto Archs = AK_armv7 | AK_arm64;
  auto Platform = PLATFORM_IOS;
  TargetList Targets;
  for (auto &&arch : Archs)
    Targets.emplace_back(Target(arch, Platform));
  EXPECT_EQ(Archs, File->getArchitectures());
  TargetToAttr Uuids = {{Target(AK_armv7, PLATFORM_UNKNOWN),
                         "00000000-0000-0000-0000-000000000000"},
                        {Target(AK_arm64, PLATFORM_UNKNOWN),
                         "11111111-1111-1111-1111-111111111111"}};
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
  EXPECT_EQ(std::string("Test.dylib"), File->getInstallName());
  EXPECT_EQ(PackedVersion(2, 3, 4), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCompatibilityVersion());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
  EXPECT_EQ(ObjCConstraintType::Retain_Release, File->getObjCConstraint());
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

  EXPECT_EQ(std::size(TBDv3Symbols), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(TBDv3Symbols)));
}

TEST(TBDv3, ReadMultipleDocuments) {
  static const char TBDv3Inlines[] =
      "--- !tapi-tbd-v3\n"
      "archs: [ armv7, arm64 ]\n"
      "platform: ios\n"
      "install-name: Test.dylib\n"
      "current-version: 2.3.4\n"
      "compatibility-version: 1.0\n"
      "swift-abi-version: 1.1\n"
      "parent-umbrella: Umbrella.dylib\n"
      "exports:\n"
      "  - archs: [ armv7, arm64 ]\n"
      "    allowable-clients: [ clientA ]\n"
      "    re-exports: [ /usr/lib/libfoo.dylib,\n"
      "                  TestInline.dylib ]\n"
      "    symbols: [ _sym1, _sym2, _sym3, _sym4, $ld$hide$os9.0$_sym1 ]\n"
      "    objc-classes: [ class1, class2 ]\n"
      "    objc-eh-types: [ class1 ]\n"
      "    objc-ivars: [ class1._ivar1, class1._ivar2 ]\n"
      "    weak-def-symbols: [ _weak1, _weak2 ]\n"
      "    thread-local-symbols: [ _tlv1, _tlv3 ]\n"
      "  - archs: [ armv7 ]\n"
      "    symbols: [ _sym5 ]\n"
      "    objc-classes: [ class3 ]\n"
      "    objc-ivars: [ class1._ivar3 ]\n"
      "    weak-def-symbols: [ _weak3 ]\n"
      "    thread-local-symbols: [ _tlv3 ]\n"
      "--- !tapi-tbd-v3\n"
      "archs: [ armv7, arm64 ]\n"
      "platform: ios\n"
      "install-name: TestInline.dylib\n"
      "swift-abi-version: 1.1\n"
      "exports:\n"
      "  - archs: [ armv7, arm64 ]\n"
      "    symbols: [ _sym5, _sym6 ]\n"
      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Inlines, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(File->documents().size(), 1U);
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  auto Archs = AK_armv7 | AK_arm64;
  auto Platform = PLATFORM_IOS;
  TargetList Targets;
  for (auto &&arch : Archs)
    Targets.emplace_back(Target(arch, Platform));
  EXPECT_EQ(Archs, File->getArchitectures());
  TargetToAttr Uuids = {{Target(AK_armv7, PLATFORM_UNKNOWN),
                         "00000000-0000-0000-0000-000000000000"},
                        {Target(AK_arm64, PLATFORM_UNKNOWN),
                         "11111111-1111-1111-1111-111111111111"}};
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
  EXPECT_EQ(std::string("Test.dylib"), File->getInstallName());
  EXPECT_EQ(PackedVersion(2, 3, 4), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCompatibilityVersion());
  EXPECT_EQ(2U, File->getSwiftABIVersion());
  EXPECT_EQ(ObjCConstraintType::Retain_Release, File->getObjCConstraint());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  InterfaceFileRef Client("clientA", Targets);
  const std::vector<InterfaceFileRef> Reexports = {
      InterfaceFileRef("/usr/lib/libfoo.dylib", Targets),
      InterfaceFileRef("TestInline.dylib", Targets)};
  EXPECT_EQ(1U, File->allowableClients().size());
  EXPECT_EQ(Client, File->allowableClients().front());
  EXPECT_EQ(2U, File->reexportedLibraries().size());
  EXPECT_EQ(Reexports, File->reexportedLibraries());

  ExportedSymbolSeq Exports;
  for (const auto *Sym : File->symbols()) {
    EXPECT_FALSE(Sym->isWeakReferenced());
    EXPECT_FALSE(Sym->isUndefined());
    Exports.emplace_back(ExportedSymbol{Sym->getKind(), Sym->getName().str(),
                                        Sym->isWeakDefined(),
                                        Sym->isThreadLocalValue()});
  }
  llvm::sort(Exports);

  EXPECT_EQ(std::size(TBDv3Symbols), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(TBDv3Symbols)));

  // Check Second Document
  Exports.clear();
  TBDReexportFile Document = File->documents().front();
  EXPECT_EQ(FileType::TBD_V3, Document->getFileType());
  EXPECT_EQ(Archs, Document->getArchitectures());
  EXPECT_EQ(Platform, *Document->getPlatforms().begin());
  EXPECT_EQ(std::string("TestInline.dylib"), Document->getInstallName());
  EXPECT_EQ(PackedVersion(1, 0, 0), Document->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), Document->getCompatibilityVersion());
  EXPECT_EQ(2U, Document->getSwiftABIVersion());

  for (const auto *Sym : Document->symbols()) {
    EXPECT_FALSE(Sym->isWeakReferenced());
    EXPECT_FALSE(Sym->isUndefined());
    Exports.emplace_back(ExportedSymbol{Sym->getKind(), Sym->getName().str(),
                                        Sym->isWeakDefined(),
                                        Sym->isThreadLocalValue()});
  }
  llvm::sort(Exports);

  ExportedSymbolSeq DocumentSymbols{
      {SymbolKind::GlobalSymbol, "_sym5", false, false},
      {SymbolKind::GlobalSymbol, "_sym6", false, false},
  };

  EXPECT_EQ(DocumentSymbols.size(), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), DocumentSymbols.begin()));
}

TEST(TBDv3, WriteFile) {
  static const char TBDv3File3[] =
      "--- !tapi-tbd-v3\n"
      "archs:           [ i386, x86_64 ]\n"
      "platform:        macosx\n"
      "install-name:    '/usr/lib/libfoo.dylib'\n"
      "current-version: 1.2.3\n"
      "compatibility-version: 0\n"
      "swift-abi-version: 5\n"
      "exports:\n"
      "  - archs:           [ i386 ]\n"
      "    symbols:         [ _sym1 ]\n"
      "    weak-def-symbols: [ _sym2 ]\n"
      "    thread-local-symbols: [ _sym3 ]\n"
      "  - archs:           [ x86_64 ]\n"
      "    allowable-clients: [ clientA ]\n"
      "    re-exports:      [ '/usr/lib/libfoo.dylib' ]\n"
      "    objc-classes:    [ Class1 ]\n"
      "    objc-eh-types:   [ Class1 ]\n"
      "    objc-ivars:      [ Class1._ivar1 ]\n"
      "...\n";

  InterfaceFile File;
  TargetList Targets;
  for (auto &&arch : AK_i386 | AK_x86_64)
    Targets.emplace_back(Target(arch, PLATFORM_MACOS));
  File.setPath("libfoo.dylib");
  File.setInstallName("/usr/lib/libfoo.dylib");
  File.setFileType(FileType::TBD_V3);
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
  EXPECT_STREQ(TBDv3File3, Buffer.c_str());
}

TEST(TBDv3, WriteMultipleDocuments) {
  static const char TBDv3Inlines[] =
      "--- !tapi-tbd-v3\n"
      "archs:           [ i386, x86_64 ]\n"
      "platform:        zippered\n"
      "install-name:    '/usr/lib/libfoo.dylib'\n"
      "current-version: 1.2.3\n"
      "compatibility-version: 0\n"
      "swift-abi-version: 5\n"
      "exports:\n"
      "  - archs:           [ x86_64 ]\n"
      "    allowable-clients: [ clientA ]\n"
      "    re-exports:      [ '/usr/lib/libbar.dylib' ]\n"
      "  - archs:           [ i386, x86_64 ]\n"
      "    symbols:         [ _sym1 ]\n"
      "    objc-classes:    [ Class1 ]\n"
      "    objc-eh-types:   [ Class1 ]\n"
      "    objc-ivars:      [ Class1._ivar1 ]\n"
      "    weak-def-symbols: [ _sym2 ]\n"
      "    thread-local-symbols: [ _symA ]\n"
      "--- !tapi-tbd-v3\n"
      "archs:           [ i386 ]\n"
      "platform:        macosx\n"
      "install-name:    '/usr/lib/libbar.dylib'\n"
      "current-version: 0\n"
      "compatibility-version: 0\n"
      "swift-abi-version: 5\n"
      "objc-constraint: none\n"
      "exports:\n"
      "  - archs:           [ i386 ]\n"
      "    symbols:         [ _sym3, _sym4 ]\n"
      "...\n";

  InterfaceFile File;
  TargetList Targets;
  for (auto &&arch : AK_i386 | AK_x86_64) {
    Targets.emplace_back(Target(arch, PLATFORM_MACOS));
    Targets.emplace_back(Target(arch, PLATFORM_MACCATALYST));
  }
  File.addTargets(Targets);
  File.setPath("libfoo.dylib");
  File.setInstallName("/usr/lib/libfoo.dylib");
  File.setFileType(FileType::TBD_V3);
  File.setCurrentVersion(PackedVersion(1, 2, 3));
  File.setTwoLevelNamespace();
  File.setApplicationExtensionSafe();
  File.setSwiftABIVersion(5);
  File.setObjCConstraint(ObjCConstraintType::Retain_Release);
  File.addAllowableClient("clientA", Targets[2]);
  File.addReexportedLibrary("/usr/lib/libbar.dylib", Targets[2]);
  File.addSymbol(SymbolKind::GlobalSymbol, "_sym1", Targets);
  File.addSymbol(SymbolKind::GlobalSymbol, "_sym2", Targets,
                 SymbolFlags::WeakDefined);
  File.addSymbol(SymbolKind::GlobalSymbol, "_symA", Targets,
                 SymbolFlags::ThreadLocalValue);
  File.addSymbol(SymbolKind::ObjectiveCClass, "Class1", Targets);
  File.addSymbol(SymbolKind::ObjectiveCClassEHType, "Class1", Targets);
  File.addSymbol(SymbolKind::ObjectiveCInstanceVariable, "Class1._ivar1",
                 Targets);

  // Inline document
  InterfaceFile Document;
  Targets = {Target(AK_i386, PLATFORM_MACOS)};
  Document.addTargets(Targets);
  Document.setPath("libbar.dylib");
  Document.setInstallName("/usr/lib/libbar.dylib");
  Document.setFileType(FileType::TBD_V3);
  Document.setTwoLevelNamespace();
  Document.setApplicationExtensionSafe();
  Document.setSwiftABIVersion(5);
  Document.addSymbol(SymbolKind::GlobalSymbol, "_sym3", Targets);
  Document.addSymbol(SymbolKind::GlobalSymbol, "_sym4", Targets);
  File.addDocument(std::make_shared<InterfaceFile>(std::move(Document)));
  
  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error Result = TextAPIWriter::writeToStream(OS, File);
  EXPECT_FALSE(Result);
  EXPECT_STREQ(TBDv3Inlines, Buffer.c_str());
}

TEST(TBDv3, Platform_macOS) {
  static const char TBDv3PlatformMacOS[] = "--- !tapi-tbd-v3\n"
                                           "archs: [ x86_64 ]\n"
                                           "platform: macosx\n"
                                           "install-name: Test.dylib\n"
                                           "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformMacOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_MACOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformMacOS),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_iOS) {
  static const char TBDv3PlatformiOS[] = "--- !tapi-tbd-v3\n"
                                         "archs: [ arm64 ]\n"
                                         "platform: ios\n"
                                         "install-name: Test.dylib\n"
                                         "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformiOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_IOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformiOS), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_watchOS) {
  static const char TBDv3watchOS[] = "--- !tapi-tbd-v3\n"
                                     "archs: [ armv7k ]\n"
                                     "platform: watchos\n"
                                     "install-name: Test.dylib\n"
                                     "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3watchOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_WATCHOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3watchOS), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_tvOS) {
  static const char TBDv3PlatformtvOS[] = "--- !tapi-tbd-v3\n"
                                          "archs: [ arm64 ]\n"
                                          "platform: tvos\n"
                                          "install-name: Test.dylib\n"
                                          "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformtvOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  auto Platform = PLATFORM_TVOS;
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_FALSE(WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformtvOS),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_bridgeOS) {
  static const char TBDv3BridgeOS[] = "--- !tapi-tbd-v3\n"
                                      "archs: [ armv7k ]\n"
                                      "platform: bridgeos\n"
                                      "install-name: Test.dylib\n"
                                      "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3BridgeOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_BRIDGEOS;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3BridgeOS), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_macCatalyst) {
  static const char TBDv3PlatformiOSmac[] = "--- !tapi-tbd-v3\n"
                                            "archs: [ armv7k ]\n"
                                            "platform: maccatalyst\n"
                                            "install-name: Test.dylib\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformiOSmac, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_MACCATALYST;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformiOSmac),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_zippered) {
  static const char TBDv3PlatformZippered[] = "--- !tapi-tbd-v3\n"
                                              "archs: [ armv7k ]\n"
                                              "platform: zippered\n"
                                              "install-name: Test.dylib\n"
                                              "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformZippered, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());

  PlatformSet Platforms;
  Platforms.insert(PLATFORM_MACOS);
  Platforms.insert(PLATFORM_MACCATALYST);
  EXPECT_EQ(Platforms.size(), File->getPlatforms().size());
  for (auto Platform : File->getPlatforms())
	    EXPECT_EQ(Platforms.count(Platform), 1U);

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformZippered),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_iOSSim) {
  static const char TBDv3PlatformiOSsim[] = "--- !tapi-tbd-v3\n"
                                            "archs: [ x86_64 ]\n"
                                            "platform: ios\n"
                                            "install-name: Test.dylib\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformiOSsim, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_IOSSIMULATOR;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformiOSsim),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_watchOSSim) {
  static const char TBDv3watchOSsim[] = "--- !tapi-tbd-v3\n"
                                        "archs: [ x86_64 ]\n"
                                        "platform: watchos\n"
                                        "install-name: Test.dylib\n"
                                        "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3watchOSsim, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto Platform = PLATFORM_WATCHOSSIMULATOR;
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3watchOSsim), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Platform_tvOSSim) {
  static const char TBDv3PlatformtvOSsim[] = "--- !tapi-tbd-v3\n"
                                             "archs: [ x86_64 ]\n"
                                             "platform: tvos\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3PlatformtvOSsim, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  auto Platform = PLATFORM_TVOSSIMULATOR;
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3PlatformtvOSsim),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Arch_arm64e) {
  static const char TBDv3ArchArm64e[] = "--- !tapi-tbd-v3\n"
                                        "archs: [ arm64, arm64e ]\n"
                                        "platform: ios\n"
                                        "install-name: Test.dylib\n"
                                        "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3ArchArm64e, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  auto Platform = PLATFORM_IOS;
  auto Archs = AK_arm64 | AK_arm64e;
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(Platform, *File->getPlatforms().begin());
  EXPECT_EQ(Archs, File->getArchitectures());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3ArchArm64e), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Swift_1_0) {
  static const char TBDv3Swift1[] = "--- !tapi-tbd-v3\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-abi-version: 1.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(1U, File->getSwiftABIVersion());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3Swift1), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Swift_1_1) {
  static const char TBDv3Swift1Dot[] = "--- !tapi-tbd-v3\n"
                                       "archs: [ arm64 ]\n"
                                       "platform: ios\n"
                                       "install-name: Test.dylib\n"
                                       "swift-abi-version: 1.1\n"
                                       "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift1Dot, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(2U, File->getSwiftABIVersion());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3Swift1Dot), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Swift_2_0) {
  static const char TBDv3Swift2[] = "--- !tapi-tbd-v3\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-abi-version: 2.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift2, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(3U, File->getSwiftABIVersion());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3Swift2), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Swift_3_0) {
  static const char TBDv3Swift3[] = "--- !tapi-tbd-v3\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-abi-version: 3.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift3, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(4U, File->getSwiftABIVersion());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv3Swift3), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv3, Swift_4_0) {
  static const char TBDv3Swift4[] = "--- !tapi-tbd-v3\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-abi-version: 4.0\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift4, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:5:20: error: invalid Swift ABI "
            "version.\nswift-abi-version: 4.0\n                   ^~~\n",
            ErrorMessage);
}

TEST(TBDv3, Swift_5) {
  static const char TBDv3Swift5[] = "--- !tapi-tbd-v3\n"
                                    "archs: [ arm64 ]\n"
                                    "platform: ios\n"
                                    "install-name: Test.dylib\n"
                                    "swift-abi-version: 5\n"
                                    "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift5, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(5U, File->getSwiftABIVersion());
}

TEST(TBDv3, Swift_99) {
  static const char TBDv3Swift99[] = "--- !tapi-tbd-v3\n"
                                     "archs: [ arm64 ]\n"
                                     "platform: ios\n"
                                     "install-name: Test.dylib\n"
                                     "swift-abi-version: 99\n"
                                     "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3Swift99, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V3, File->getFileType());
  EXPECT_EQ(99U, File->getSwiftABIVersion());
}

TEST(TBDv3, UnknownArchitecture) {
  static const char TBDv3FileUnknownArch[] = "--- !tapi-tbd-v3\n"
                                             "archs: [ foo ]\n"
                                             "platform: macosx\n"
                                             "install-name: Test.dylib\n"
                                             "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3FileUnknownArch, "Test.tbd"));
  EXPECT_TRUE(!!Result);
}

TEST(TBDv3, UnknownPlatform) {
  static const char TBDv3FileUnknownPlatform[] = "--- !tapi-tbd-v3\n"
                                                 "archs: [ i386 ]\n"
                                                 "platform: newOS\n"
                                                 "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3FileUnknownPlatform, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:11: error: unknown platform\nplatform: "
            "newOS\n          ^~~~~\n",
            ErrorMessage);
}

TEST(TBDv3, MalformedFile1) {
  static const char TBDv3FileMalformed1[] = "--- !tapi-tbd-v3\n"
                                            "archs: [ arm64 ]\n"
                                            "foobar: \"Unsupported key\"\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3FileMalformed1, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  ASSERT_EQ("malformed file\nTest.tbd:2:1: error: missing required key "
            "'platform'\narchs: [ arm64 ]\n^\n",
            ErrorMessage);
}

TEST(TBDv3, MalformedFile2) {
  static const char TBDv3FileMalformed2[] = "--- !tapi-tbd-v3\n"
                                            "archs: [ arm64 ]\n"
                                            "platform: ios\n"
                                            "install-name: Test.dylib\n"
                                            "foobar: \"Unsupported key\"\n"
                                            "...\n";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv3FileMalformed2, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  ASSERT_EQ(
      "malformed file\nTest.tbd:5:1: error: unknown key 'foobar'\nfoobar: "
      "\"Unsupported key\"\n^~~~~~\n",
      ErrorMessage);
}

TEST(TBDv3, InterfaceEquality) {
  static const char TBDv3File[] =
      "--- !tapi-tbd-v3\n"
      "archs: [ armv7, arm64 ]\n"
      "uuids: [ 'armv7: 00000000-0000-0000-0000-000000000000',\n"
      "         'arm64: 11111111-1111-1111-1111-111111111111']\n"
      "platform: ios\n"
      "flags: [ installapi ]\n"
      "install-name: Test.dylib\n"
      "current-version: 2.3.4\n"
      "compatibility-version: 1.0\n"
      "swift-abi-version: 1.1\n"
      "parent-umbrella: Umbrella.dylib\n"
      "exports:\n"
      "  - archs: [ armv7, arm64 ]\n"
      "    allowable-clients: [ clientA ]\n"
      "    re-exports: [ /usr/lib/libfoo.dylib ]\n"
      "    symbols: [ _sym1, _sym2, _sym3, _sym4, $ld$hide$os9.0$_sym1 ]\n"
      "    objc-classes: [ class1, class2 ]\n"
      "    objc-eh-types: [ class1 ]\n"
      "    objc-ivars: [ class1._ivar1, class1._ivar2 ]\n"
      "    weak-def-symbols: [ _weak1, _weak2 ]\n"
      "    thread-local-symbols: [ _tlv1, _tlv3 ]\n"
      "  - archs: [ armv7 ]\n"
      "    symbols: [ _sym5 ]\n"
      "    objc-classes: [ class3 ]\n"
      "    objc-ivars: [ class1._ivar3 ]\n"
      "    weak-def-symbols: [ _weak3 ]\n"
      "    thread-local-symbols: [ _tlv3 ]\n"
      "--- !tapi-tbd-v3\n"
      "archs:           [ i386 ]\n"
      "platform:        macosx\n"
      "install-name:    '/usr/lib/libbar.dylib'\n"
      "current-version: 0\n"
      "compatibility-version: 0\n"
      "swift-abi-version: 5\n"
      "objc-constraint: none\n"
      "exports:\n"
      "  - archs:           [ i386 ]\n"
      "    symbols:         [ _sym3, _sym4 ]\n"
      "...\n";
  Expected<TBDFile> ResultA =
      TextAPIReader::get(MemoryBufferRef(TBDv3File, "TestA.tbd"));
  EXPECT_TRUE(!!ResultA);
  InterfaceFile FileA = std::move(*ResultA.get());
  Expected<TBDFile> ResultB =
      TextAPIReader::get(MemoryBufferRef(TBDv3File, "TestB.tbd"));
  EXPECT_TRUE(!!ResultB);
  InterfaceFile FileB = std::move(*ResultB.get());
  EXPECT_FALSE(FileA.getPath() == FileB.getPath());
  EXPECT_TRUE(FileA == FileB);
}



TEST(TBDv3, InterfaceInequality) {
  static const char TBDv3File[] = "--- !tapi-tbd-v3\n"
                                  "archs: [ armv7, arm64 ]\n"
                                  "platform: ios\n"
                                  "install-name: Test.dylib\n"
                                  "...\n";

  Expected<TBDFile> ResultA =
      TextAPIReader::get(MemoryBufferRef(TBDv3File, "TestA.tbd"));
  EXPECT_TRUE(!!ResultA);
  InterfaceFile FileA = std::move(*ResultA.get());
  Expected<TBDFile> ResultB =
      TextAPIReader::get(MemoryBufferRef(TBDv3File, "TestB.tbd"));
  EXPECT_TRUE(!!ResultB);
  InterfaceFile FileB = std::move(*ResultB.get());

  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->addTarget(Target(AK_x86_64, PLATFORM_IOS));
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->setCurrentVersion(PackedVersion(1, 2, 3));
    File->setCompatibilityVersion(PackedVersion(1, 0, 0));
  }));
  EXPECT_TRUE(checkEqualityOnTransform(
      FileA, FileB, [](InterfaceFile *File) { File->setSwiftABIVersion(5); }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->setTwoLevelNamespace(false);
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->setApplicationExtensionSafe(false);
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->addParentUmbrella(Target(AK_armv7, PLATFORM_IOS), "Umbrella.dylib");
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->addAllowableClient("ClientA", Target(AK_armv7, PLATFORM_IOS));
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->addReexportedLibrary("/System/Library/Frameworks/A.framework/A",
                               Target(AK_armv7, PLATFORM_IOS));
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    File->addSymbol(SymbolKind::GlobalSymbol, "_symA",
                    {Target(AK_arm64, PLATFORM_IOS)});
  }));
  EXPECT_TRUE(checkEqualityOnTransform(FileA, FileB, [](InterfaceFile *File) {
    InterfaceFile Document;
    Document.setFileType(FileType::TBD_V3);
    Document.addTargets(TargetList{Target(AK_armv7, PLATFORM_IOS),
                                   Target(AK_arm64, PLATFORM_IOS)});
    Document.setInstallName("/System/Library/Frameworks/A.framework/A");
    File->addDocument(std::make_shared<InterfaceFile>(std::move(Document)));
  }));
}

} // namespace TBDv3
