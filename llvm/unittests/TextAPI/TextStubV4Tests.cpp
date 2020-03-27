//===-- TextStubV4Tests.cpp - TBD V4 File Test ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "TextStubHelpers.h"
#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"
#include "llvm/TextAPI/MachO/TextAPIWriter.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::MachO;

static ExportedSymbol TBDv4ExportedSymbols[] = {
    {SymbolKind::GlobalSymbol, "_symA", false, false},
    {SymbolKind::GlobalSymbol, "_symAB", false, false},
    {SymbolKind::GlobalSymbol, "_symB", false, false},
};

static ExportedSymbol TBDv4ReexportedSymbols[] = {
    {SymbolKind::GlobalSymbol, "_symC", false, false},
};

static ExportedSymbol TBDv4UndefinedSymbols[] = {
    {SymbolKind::GlobalSymbol, "_symD", false, false},
};

namespace TBDv4 {

TEST(TBDv4, ReadFile) {
  static const char TBDv4File[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets:  [ i386-macos, x86_64-macos, x86_64-ios ]\n"
      "uuids:\n"
      "  - target: i386-macos\n"
      "    value: 00000000-0000-0000-0000-000000000000\n"
      "  - target: x86_64-macos\n"
      "    value: 11111111-1111-1111-1111-111111111111\n"
      "  - target: x86_64-ios\n"
      "    value: 11111111-1111-1111-1111-111111111111\n"
      "flags: [ flat_namespace, installapi ]\n"
      "install-name: Umbrella.framework/Umbrella\n"
      "current-version: 1.2.3\n"
      "compatibility-version: 1.2\n"
      "swift-abi-version: 5\n"
      "parent-umbrella:\n"
      "  - targets: [ i386-macos, x86_64-macos, x86_64-ios ]\n"
      "    umbrella: System\n"
      "allowable-clients:\n"
      "  - targets: [ i386-macos, x86_64-macos, x86_64-ios ]\n"
      "    clients: [ ClientA ]\n"
      "reexported-libraries:\n"
      "  - targets: [ i386-macos ]\n"
      "    libraries: [ /System/Library/Frameworks/A.framework/A ]\n"
      "exports:\n"
      "  - targets: [ i386-macos ]\n"
      "    symbols: [ _symA ]\n"
      "    objc-classes: []\n"
      "    objc-eh-types: []\n"
      "    objc-ivars: []\n"
      "    weak-symbols: []\n"
      "    thread-local-symbols: []\n"
      "  - targets: [ x86_64-ios ]\n"
      "    symbols: [_symB]\n"
      "  - targets: [ x86_64-macos, x86_64-ios ]\n"
      "    symbols: [_symAB]\n"
      "reexports:\n"
      "  - targets: [ i386-macos ]\n"
      "    symbols: [_symC]\n"
      "    objc-classes: []\n"
      "    objc-eh-types: []\n"
      "    objc-ivars: []\n"
      "    weak-symbols: []\n"
      "    thread-local-symbols: []\n"
      "undefineds:\n"
      "  - targets: [ i386-macos ]\n"
      "    symbols: [ _symD ]\n"
      "    objc-classes: []\n"
      "    objc-eh-types: []\n"
      "    objc-ivars: []\n"
      "    weak-symbols: []\n"
      "    thread-local-symbols: []\n"
      "...\n";

  auto Result = TextAPIReader::get(MemoryBufferRef(TBDv4File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  PlatformSet Platforms;
  Platforms.insert(PlatformKind::macOS);
  Platforms.insert(PlatformKind::iOS);
  auto Archs = AK_i386 | AK_x86_64;
  TargetList Targets = {
      Target(AK_i386, PlatformKind::macOS),
      Target(AK_x86_64, PlatformKind::macOS),
      Target(AK_x86_64, PlatformKind::iOS),
  };
  UUIDs uuids = {{Targets[0], "00000000-0000-0000-0000-000000000000"},
                 {Targets[1], "11111111-1111-1111-1111-111111111111"},
                 {Targets[2], "11111111-1111-1111-1111-111111111111"}};
  EXPECT_EQ(Archs, File->getArchitectures());
  EXPECT_EQ(uuids, File->uuids());
  EXPECT_EQ(Platforms.size(), File->getPlatforms().size());
  for (auto Platform : File->getPlatforms())
    EXPECT_EQ(Platforms.count(Platform), 1U);
  EXPECT_EQ(std::string("Umbrella.framework/Umbrella"), File->getInstallName());
  EXPECT_EQ(PackedVersion(1, 2, 3), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 2, 0), File->getCompatibilityVersion());
  EXPECT_EQ(5U, File->getSwiftABIVersion());
  EXPECT_FALSE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_TRUE(File->isInstallAPI());
  InterfaceFileRef client("ClientA", Targets);
  InterfaceFileRef reexport("/System/Library/Frameworks/A.framework/A",
                            {Targets[0]});
  EXPECT_EQ(1U, File->allowableClients().size());
  EXPECT_EQ(client, File->allowableClients().front());
  EXPECT_EQ(1U, File->reexportedLibraries().size());
  EXPECT_EQ(reexport, File->reexportedLibraries().front());

  ExportedSymbolSeq Exports, Reexports, Undefineds;
  ExportedSymbol temp;
  for (const auto *Sym : File->symbols()) {
    temp = ExportedSymbol{Sym->getKind(), std::string(Sym->getName()),
                          Sym->isWeakDefined(), Sym->isThreadLocalValue()};
    EXPECT_FALSE(Sym->isWeakReferenced());
    if (Sym->isUndefined())
      Undefineds.emplace_back(std::move(temp));
    else
      Sym->isReexported() ? Reexports.emplace_back(std::move(temp))
                          : Exports.emplace_back(std::move(temp));
  }
  llvm::sort(Exports.begin(), Exports.end());
  llvm::sort(Reexports.begin(), Reexports.end());
  llvm::sort(Undefineds.begin(), Undefineds.end());

  EXPECT_EQ(sizeof(TBDv4ExportedSymbols) / sizeof(ExportedSymbol),
            Exports.size());
  EXPECT_EQ(sizeof(TBDv4ReexportedSymbols) / sizeof(ExportedSymbol),
            Reexports.size());
  EXPECT_EQ(sizeof(TBDv4UndefinedSymbols) / sizeof(ExportedSymbol),
            Undefineds.size());
  EXPECT_TRUE(std::equal(Exports.begin(), Exports.end(),
                         std::begin(TBDv4ExportedSymbols)));
  EXPECT_TRUE(std::equal(Reexports.begin(), Reexports.end(),
                         std::begin(TBDv4ReexportedSymbols)));
  EXPECT_TRUE(std::equal(Undefineds.begin(), Undefineds.end(),
                         std::begin(TBDv4UndefinedSymbols)));
}

TEST(TBDv4, WriteFile) {
  static const char TBDv4File[] =
      "--- !tapi-tbd\n"
      "tbd-version:     4\n"
      "targets:         [ i386-macos, x86_64-ios-simulator ]\n"
      "uuids:\n"
      "  - target:          i386-macos\n"
      "    value:           00000000-0000-0000-0000-000000000000\n"
      "  - target:          x86_64-ios-simulator\n"
      "    value:           11111111-1111-1111-1111-111111111111\n"
      "flags:           [ installapi ]\n"
      "install-name:    'Umbrella.framework/Umbrella'\n"
      "current-version: 1.2.3\n"
      "compatibility-version: 0\n"
      "swift-abi-version: 5\n"
      "parent-umbrella:\n"
      "  - targets:         [ i386-macos, x86_64-ios-simulator ]\n"
      "    umbrella:        System\n"
      "allowable-clients:\n"
      "  - targets:         [ i386-macos ]\n"
      "    clients:         [ ClientA ]\n"
      "exports:\n"
      "  - targets:         [ i386-macos ]\n"
      "    symbols:         [ _symA ]\n"
      "    objc-classes:    [ Class1 ]\n"
      "    weak-symbols:    [ _symC ]\n"
      "  - targets:         [ x86_64-ios-simulator ]\n"
      "    symbols:         [ _symB ]\n"
      "...\n";

  InterfaceFile File;
  TargetList Targets = {
      Target(AK_i386, PlatformKind::macOS),
      Target(AK_x86_64, PlatformKind::iOSSimulator),
  };
  UUIDs uuids = {{Targets[0], "00000000-0000-0000-0000-000000000000"},
                 {Targets[1], "11111111-1111-1111-1111-111111111111"}};
  File.setInstallName("Umbrella.framework/Umbrella");
  File.setFileType(FileType::TBD_V4);
  File.addTargets(Targets);
  File.addUUID(uuids[0].first, uuids[0].second);
  File.addUUID(uuids[1].first, uuids[1].second);
  File.setCurrentVersion(PackedVersion(1, 2, 3));
  File.setTwoLevelNamespace();
  File.setInstallAPI(true);
  File.setApplicationExtensionSafe(true);
  File.setSwiftABIVersion(5);
  File.addAllowableClient("ClientA", Targets[0]);
  File.addParentUmbrella(Targets[0], "System");
  File.addParentUmbrella(Targets[1], "System");
  File.addSymbol(SymbolKind::GlobalSymbol, "_symA", {Targets[0]});
  File.addSymbol(SymbolKind::GlobalSymbol, "_symB", {Targets[1]});
  File.addSymbol(SymbolKind::GlobalSymbol, "_symC", {Targets[0]},
                 SymbolFlags::WeakDefined);
  File.addSymbol(SymbolKind::ObjectiveCClass, "Class1", {Targets[0]});

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto Result = TextAPIWriter::writeToStream(OS, File);
  EXPECT_FALSE(Result);
  EXPECT_STREQ(TBDv4File, Buffer.c_str());
}

TEST(TBDv4, MultipleTargets) {
  static const char TBDv4MultipleTargets[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets: [ i386-maccatalyst, x86_64-tvos, arm64-ios ]\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4MultipleTargets, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  PlatformSet Platforms;
  Platforms.insert(PlatformKind::macCatalyst);
  Platforms.insert(PlatformKind::tvOS);
  Platforms.insert(PlatformKind::iOS);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(AK_x86_64 | AK_arm64 | AK_i386, File->getArchitectures());
  EXPECT_EQ(Platforms.size(), File->getPlatforms().size());
  for (auto Platform : File->getPlatforms())
    EXPECT_EQ(Platforms.count(Platform), 1U);

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4MultipleTargets),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, MultipleTargetsSameArch) {
  static const char TBDv4TargetsSameArch[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets: [ x86_64-tvos , x86_64-maccatalyst ]\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4TargetsSameArch, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  PlatformSet Platforms;
  Platforms.insert(PlatformKind::tvOS);
  Platforms.insert(PlatformKind::macCatalyst);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_x86_64), File->getArchitectures());
  EXPECT_EQ(Platforms.size(), File->getPlatforms().size());
  for (auto Platform : File->getPlatforms())
    EXPECT_EQ(Platforms.count(Platform), 1U);

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4TargetsSameArch),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, MultipleTargetsSamePlatform) {
  static const char TBDv4MultipleTargetsSamePlatform[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets: [ armv7k-ios , arm64-ios]\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result = TextAPIReader::get(
      MemoryBufferRef(TBDv4MultipleTargetsSamePlatform, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(AK_arm64 | AK_armv7k, File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::iOS, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4MultipleTargetsSamePlatform),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_maccatalyst) {
  static const char TBDv4TargetMacCatalyst[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets: [  x86_64-maccatalyst ]\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4TargetMacCatalyst, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_x86_64), File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::macCatalyst, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4TargetMacCatalyst),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_x86_ios) {
  static const char TBDv4Targetx86iOS[] = "--- !tapi-tbd\n"
                                          "tbd-version: 4\n"
                                          "targets: [  x86_64-ios ]\n"
                                          "install-name: Test.dylib\n"
                                          "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4Targetx86iOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_x86_64), File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::iOS, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4Targetx86iOS),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_arm_bridgeOS) {
  static const char TBDv4PlatformBridgeOS[] = "--- !tapi-tbd\n"
                                              "tbd-version: 4\n"
                                              "targets: [  armv7k-bridgeos ]\n"
                                              "install-name: Test.dylib\n"
                                              "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4PlatformBridgeOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::bridgeOS, *File->getPlatforms().begin());
  EXPECT_EQ(ArchitectureSet(AK_armv7k), File->getArchitectures());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4PlatformBridgeOS),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_arm_iOS) {
  static const char TBDv4ArchArm64e[] = "--- !tapi-tbd\n"
                                        "tbd-version: 4\n"
                                        "targets: [  arm64e-ios ]\n"
                                        "install-name: Test.dylib\n"
                                        "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4ArchArm64e, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::iOS, *File->getPlatforms().begin());
  EXPECT_EQ(ArchitectureSet(AK_arm64e), File->getArchitectures());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4ArchArm64e), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_x86_macos) {
  static const char TBDv4Targetx86MacOS[] = "--- !tapi-tbd\n"
                                            "tbd-version: 4\n"
                                            "targets: [  x86_64-macos ]\n"
                                            "install-name: Test.dylib\n"
                                            "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4Targetx86MacOS, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_x86_64), File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::macOS, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4Targetx86MacOS),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_x86_ios_simulator) {
  static const char TBDv4Targetx86iOSSim[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets: [  x86_64-ios-simulator  ]\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4Targetx86iOSSim, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_x86_64), File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::iOSSimulator, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4Targetx86iOSSim),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_x86_tvos_simulator) {
  static const char TBDv4x86tvOSSim[] = "--- !tapi-tbd\n"
                                        "tbd-version: 4\n"
                                        "targets: [  x86_64-tvos-simulator  ]\n"
                                        "install-name: Test.dylib\n"
                                        "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4x86tvOSSim, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_x86_64), File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::tvOSSimulator, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4x86tvOSSim), stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Target_i386_watchos_simulator) {
  static const char TBDv4i386watchOSSim[] =
      "--- !tapi-tbd\n"
      "tbd-version: 4\n"
      "targets: [  i386-watchos-simulator  ]\n"
      "install-name: Test.dylib\n"
      "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4i386watchOSSim, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(ArchitectureSet(AK_i386), File->getArchitectures());
  EXPECT_EQ(File->getPlatforms().size(), 1U);
  EXPECT_EQ(PlatformKind::watchOSSimulator, *File->getPlatforms().begin());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4i386watchOSSim),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Swift_1) {
  static const char TBDv4SwiftVersion1[] = "--- !tapi-tbd\n"
                                           "tbd-version: 4\n"
                                           "targets: [  x86_64-macos ]\n"
                                           "install-name: Test.dylib\n"
                                           "swift-abi-version: 1\n"
                                           "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4SwiftVersion1, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(1U, File->getSwiftABIVersion());

  // No writer test because we emit "swift-abi-version:1.0".
}

TEST(TBDv4, Swift_2) {
  static const char TBDv4Swift2[] = "--- !tapi-tbd\n"
                                    "tbd-version: 4\n"
                                    "targets: [  x86_64-macos ]\n"
                                    "install-name: Test.dylib\n"
                                    "swift-abi-version: 2\n"
                                    "...\n";

  auto Result = TextAPIReader::get(MemoryBufferRef(TBDv4Swift2, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(2U, File->getSwiftABIVersion());

  // No writer test because we emit "swift-abi-version:2.0".
}

TEST(TBDv4, Swift_5) {
  static const char TBDv4SwiftVersion5[] = "--- !tapi-tbd\n"
                                           "tbd-version: 4\n"
                                           "targets: [  x86_64-macos ]\n"
                                           "install-name: Test.dylib\n"
                                           "swift-abi-version: 5\n"
                                           "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4SwiftVersion5, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(5U, File->getSwiftABIVersion());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4SwiftVersion5),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, Swift_99) {
  static const char TBDv4SwiftVersion99[] = "--- !tapi-tbd\n"
                                            "tbd-version: 4\n"
                                            "targets: [  x86_64-macos ]\n"
                                            "install-name: Test.dylib\n"
                                            "swift-abi-version: 99\n"
                                            "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4SwiftVersion99, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  auto File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V4, File->getFileType());
  EXPECT_EQ(99U, File->getSwiftABIVersion());

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  auto WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);
  EXPECT_EQ(stripWhitespace(TBDv4SwiftVersion99),
            stripWhitespace(Buffer.c_str()));
}

TEST(TBDv4, InvalidArchitecture) {
  static const char TBDv4UnknownArch[] = "--- !tapi-tbd\n"
                                         "tbd-version: 4\n"
                                         "targets: [ foo-macos ]\n"
                                         "install-name: Test.dylib\n"
                                         "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4UnknownArch, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:12: error: unknown "
            "architecture\ntargets: [ foo-macos ]\n"
            "           ^~~~~~~~~~\n",
            errorMessage);
}

TEST(TBDv4, InvalidPlatform) {
  static const char TBDv4FInvalidPlatform[] = "--- !tapi-tbd\n"
                                              "tbd-version: 4\n"
                                              "targets: [ x86_64-maos ]\n"
                                              "install-name: Test.dylib\n"
                                              "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4FInvalidPlatform, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:3:12: error: unknown platform\ntargets: "
            "[ x86_64-maos ]\n"
            "           ^~~~~~~~~~~~\n",
            errorMessage);
}

TEST(TBDv4, MalformedFile1) {
  static const char TBDv4MalformedFile1[] = "--- !tapi-tbd\n"
                                            "tbd-version: 4\n"
                                            "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4MalformedFile1, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  ASSERT_EQ("malformed file\nTest.tbd:2:1: error: missing required key "
            "'targets'\ntbd-version: 4\n^\n",
            errorMessage);
}

TEST(TBDv4, MalformedFile2) {
  static const char TBDv4MalformedFile2[] = "--- !tapi-tbd\n"
                                            "tbd-version: 4\n"
                                            "targets: [ x86_64-macos ]\n"
                                            "install-name: Test.dylib\n"
                                            "foobar: \"unsupported key\"\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4MalformedFile2, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  ASSERT_EQ(
      "malformed file\nTest.tbd:5:9: error: unknown key 'foobar'\nfoobar: "
      "\"unsupported key\"\n        ^~~~~~~~~~~~~~~~~\n",
      errorMessage);
}

TEST(TBDv4, MalformedFile3) {
  static const char TBDv4MalformedSwift[] = "--- !tapi-tbd\n"
                                            "tbd-version: 4\n"
                                            "targets: [  x86_64-macos ]\n"
                                            "install-name: Test.dylib\n"
                                            "swift-abi-version: 1.1\n"
                                            "...\n";

  auto Result =
      TextAPIReader::get(MemoryBufferRef(TBDv4MalformedSwift, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  auto errorMessage = toString(Result.takeError());
  EXPECT_EQ("malformed file\nTest.tbd:5:20: error: invalid Swift ABI "
            "version.\nswift-abi-version: 1.1\n                   ^~~\n",
            errorMessage);
}

} // end namespace TBDv4
