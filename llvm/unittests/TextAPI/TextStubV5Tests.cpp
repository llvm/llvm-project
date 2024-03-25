//===-- TextStubV5Tests.cpp - TBD V5 File Test ----------------------------===//
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

namespace TBDv5 {

TEST(TBDv5, ReadFile) {
  static const char TBDv5File[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }
  ],
  "flags": [
    {
      "targets": [
            "x86_64-macos"
        ],
      "attributes": [
            "flat_namespace"
        ]
    }
  ],
  "install_names": [
    {
        "name": "/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "compatibility_versions": [
    { "version": "1.1" }
  ],
  "rpaths": [
    {
      "targets": [
          "x86_64-macos"
      ],
      "paths": [
          "@executable_path/.../Frameworks"
      ]
    }
  ],
  "parent_umbrellas": [
    {
      "umbrella": "System"
    }
  ],
  "allowable_clients": [
    {
        "clients": [
            "ClientA",
            "ClientB"
        ]
    }
  ],
  "reexported_libraries": [
    {
        "names": [
            "/u/l/l/libfoo.dylib",
            "/u/l/l/libbar.dylib"
        ]
    }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_global"
            ],
            "objc_class": [
                "ClassA"
            ],
            "weak": [],
            "thread_local": []
        },
        "text": {
            "global": [
                "_func"
            ],
            "weak": [],
            "thread_local": []
        }
    },
    {
      "targets": [
          "x86_64-macos"
      ],
      "data": {
          "global": [
              "_globalVar"
          ],
          "objc_class": [
              "ClassA",
              "ClassB",
              "ClassData"
          ],
          "objc_eh_type": [
              "ClassA",
              "ClassB"
          ],
          "objc_ivar": [
              "ClassA.ivar1",
              "ClassA.ivar2",
              "ClassC.ivar1"
          ]
      },
      "text": {
          "global": [
              "_funcFoo"
          ]
      }
    }
  ],
  "reexported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_globalRe"
            ],
            "objc_class": [
                "ClassRexport"
            ]
        },
        "text": {
            "global": [
                "_funcA"
            ]
        }
    }
  ],
  "undefined_symbols": [
    {
        "targets": [
            "x86_64-macos"
        ],
        "data": {
            "global": [
                "_globalBind"
            ],
            "weak": [
                "referenced_sym"
            ]
        }
    }
  ]
},
"libraries": []
})";

  MemoryBufferRef InputBuf = MemoryBufferRef(TBDv5File, "Test.tbd");
  Expected<FileType> ExpectedFT = TextAPIReader::canRead(InputBuf);
  EXPECT_TRUE(!!ExpectedFT);

  Expected<TBDFile> Result = TextAPIReader::get(InputBuf);
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  EXPECT_EQ(*ExpectedFT, File->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"), File->getInstallName());

  TargetList AllTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0, 0)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(14, 0)),
  };
  std::set<Target> FileTargets{File->targets().begin(), File->targets().end()};
  EXPECT_EQ(mapToPlatformSet(AllTargets), File->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets), File->getArchitectures());
  EXPECT_EQ(FileTargets.size(), AllTargets.size());
  for (const auto &Targ : AllTargets) {
    auto FileTarg = FileTargets.find(Targ);
    EXPECT_FALSE(FileTarg == FileTargets.end());
    EXPECT_EQ(*FileTarg, Targ);
    PackedVersion MD = Targ.MinDeployment;
    PackedVersion FileMD = FileTarg->MinDeployment;
    EXPECT_EQ(MD, FileMD);
  }

  EXPECT_EQ(PackedVersion(1, 2, 0), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 1, 0), File->getCompatibilityVersion());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_FALSE(File->isTwoLevelNamespace());
  EXPECT_FALSE(File->isOSLibNotForSharedCache());
  EXPECT_EQ(0U, File->documents().size());

  InterfaceFileRef ClientA("ClientA", AllTargets);
  InterfaceFileRef ClientB("ClientB", AllTargets);
  EXPECT_EQ(2U, File->allowableClients().size());
  EXPECT_EQ(ClientA, File->allowableClients().at(0));
  EXPECT_EQ(ClientB, File->allowableClients().at(1));

  InterfaceFileRef ReexportA("/u/l/l/libbar.dylib", AllTargets);
  InterfaceFileRef ReexportB("/u/l/l/libfoo.dylib", AllTargets);
  EXPECT_EQ(2U, File->reexportedLibraries().size());
  EXPECT_EQ(ReexportA, File->reexportedLibraries().at(0));
  EXPECT_EQ(ReexportB, File->reexportedLibraries().at(1));

  TargetToAttr RPaths = {
      {Target(AK_x86_64, PLATFORM_MACOS), "@executable_path/.../Frameworks"},
  };
  EXPECT_EQ(RPaths, File->rpaths());

  TargetToAttr Umbrellas = {{Target(AK_x86_64, PLATFORM_MACOS), "System"},
                            {Target(AK_arm64, PLATFORM_MACOS), "System"},
                            {Target(AK_arm64, PLATFORM_MACCATALYST), "System"}};
  EXPECT_EQ(Umbrellas, File->umbrellas());

  ExportedSymbolSeq Exports, Reexports, Undefineds;
  for (const auto *Sym : File->symbols()) {
    TargetList SymTargets{Sym->targets().begin(), Sym->targets().end()};
    ExportedSymbol Temp =
        ExportedSymbol{Sym->getKind(),
                       std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(),
                       Sym->isData(),
                       SymTargets};
    if (Sym->isUndefined())
      Undefineds.emplace_back(std::move(Temp));
    else
      Sym->isReexported() ? Reexports.emplace_back(std::move(Temp))
                          : Exports.emplace_back(std::move(Temp));
  }
  llvm::sort(Exports);
  llvm::sort(Reexports);
  llvm::sort(Undefineds);

  TargetList MacOSTargets = {Target(AK_x86_64, PLATFORM_MACOS),
                             Target(AK_arm64, PLATFORM_MACOS)};

  std::vector<ExportedSymbol> ExpectedExportedSymbols = {
      {EncodeKind::GlobalSymbol, "_func", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol,
       "_funcFoo",
       false,
       false,
       false,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol, "_global", false, false, true, MacOSTargets},
      {EncodeKind::GlobalSymbol,
       "_globalVar",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassA",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassB",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassData",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassA",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassB",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassA.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassA.ivar2",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassC.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
  };
  std::vector<ExportedSymbol> ExpectedReexportedSymbols = {
      {EncodeKind::GlobalSymbol, "_funcA", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_globalRe", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass, "ClassRexport", false, false, true,
       MacOSTargets},
  };

  std::vector<ExportedSymbol> ExpectedUndefinedSymbols = {
      {EncodeKind::GlobalSymbol,
       "_globalBind",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol,
       "referenced_sym",
       true,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
  };

  EXPECT_EQ(ExpectedExportedSymbols.size(), Exports.size());
  EXPECT_EQ(ExpectedReexportedSymbols.size(), Reexports.size());
  EXPECT_EQ(ExpectedUndefinedSymbols.size(), Undefineds.size());
  EXPECT_TRUE(std::equal(Exports.begin(), Exports.end(),
                         std::begin(ExpectedExportedSymbols)));
  EXPECT_TRUE(std::equal(Reexports.begin(), Reexports.end(),
                         std::begin(ExpectedReexportedSymbols)));
  EXPECT_TRUE(std::equal(Undefineds.begin(), Undefineds.end(),
                         std::begin(ExpectedUndefinedSymbols)));

  EXPECT_TRUE(
      File->getSymbol(EncodeKind::GlobalSymbol, "_globalBind").has_value());
}

TEST(TBDv5, ReadMultipleTargets) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library":  {
  "target_info": [
      {
          "target": "x86_64-macos",
          "min_deployment": "10.14" 
      },
      {
          "target": "arm64-macos",
          "min_deployment": "10.14"
      },
      {
          "target": "arm64-maccatalyst",
          "min_deployment": "12.1"
      }
  ],
  "install_names":[
      { "name":"/usr/lib/libFoo.dylib" }
  ],
  "swift_abi":[ { "abi":8 } ],
  "reexported_libraries": [
      {
          "targets": [ "x86_64-maccatalyst" ],
          "names": [
              "/u/l/l/libfoo.dylib",
              "/u/l/l/libbar.dylib"
          ]
      },
      {
          "targets": [ "arm64-maccatalyst" ],
          "names": [ "/u/l/l/libArmOnly.dylib" ]
      }
  ]
}
})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  EXPECT_EQ(std::string("/usr/lib/libFoo.dylib"), File->getInstallName());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), File->getCompatibilityVersion());
  EXPECT_EQ(8U, File->getSwiftABIVersion());

  TargetList AllTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(12, 1)),
  };
  EXPECT_EQ(mapToPlatformSet(AllTargets), File->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets), File->getArchitectures());

  InterfaceFileRef ReexportA("/u/l/l/libArmOnly.dylib",
                             {Target(AK_arm64, PLATFORM_MACCATALYST)});
  InterfaceFileRef ReexportB("/u/l/l/libbar.dylib",
                             {Target(AK_x86_64, PLATFORM_MACCATALYST)});
  InterfaceFileRef ReexportC("/u/l/l/libfoo.dylib",
                             {Target(AK_x86_64, PLATFORM_MACCATALYST)});
  EXPECT_EQ(3U, File->reexportedLibraries().size());
  EXPECT_EQ(ReexportA, File->reexportedLibraries().at(0));
  EXPECT_EQ(ReexportB, File->reexportedLibraries().at(1));
  EXPECT_EQ(ReexportC, File->reexportedLibraries().at(2));
}

TEST(TBDv5, ReadMultipleDocuments) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "armv7-ios",
      "min_deployment": "11.0" 
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ],
  "reexported_libraries": [
    { "names": ["/u/l/l/libfoo.dylib"] }
  ]
},
"libraries": [
  {
    "target_info": [
      {
        "target": "armv7-ios",
        "min_deployment": "11.0" 
      }
    ],
    "install_names":[
      { "name":"/u/l/l/libfoo.dylib" }
    ],
    "flags":[ 
      { "attributes": ["not_app_extension_safe"] }
    ], 
    "exported_symbols": [
      {
        "data": {
          "thread_local": [ "_globalVar" ],
          "objc_class": [ "ClassData", "ClassA", "ClassB"], 
          "objc_eh_type": [ "ClassA", "ClassB" ]
        },
        "text": {
          "global": [ "_funcFoo" ]
        }
      }
    ]
  }
]})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"), File->getInstallName());
  EXPECT_TRUE(File->isTwoLevelNamespace());
  EXPECT_TRUE(File->isApplicationExtensionSafe());

  TargetList Targets(File->targets().begin(), File->targets().end());
  Target iOSTarget(AK_armv7, PLATFORM_IOS, VersionTuple(11, 0));
  EXPECT_EQ(TargetList{iOSTarget}, Targets);
  std::vector<const Symbol *> Symbols(File->symbols().begin(),
                                      File->symbols().end());
  EXPECT_EQ(0U, Symbols.size());

  InterfaceFileRef Reexport("/u/l/l/libfoo.dylib", {iOSTarget});
  EXPECT_EQ(1U, File->reexportedLibraries().size());
  EXPECT_EQ(Reexport, File->reexportedLibraries().at(0));

  // Check inlined library.
  EXPECT_EQ(1U, File->documents().size());
  TBDReexportFile Document = File->documents().front();
  Targets = {Document->targets().begin(), Document->targets().end()};
  EXPECT_EQ(TargetList{iOSTarget}, Targets);
  EXPECT_EQ(std::string("/u/l/l/libfoo.dylib"), Document->getInstallName());
  EXPECT_EQ(0U, Document->getSwiftABIVersion());
  EXPECT_TRUE(Document->isTwoLevelNamespace());
  EXPECT_FALSE(Document->isApplicationExtensionSafe());

  ExportedSymbolSeq Exports;
  for (const auto *Sym : Document->symbols())
    Exports.emplace_back(
        ExportedSymbol{Sym->getKind(),
                       std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(),
                       Sym->isData(),
                       {iOSTarget}});

  llvm::sort(Exports);
  ExportedSymbolSeq ExpectedExports = {
      {EncodeKind::GlobalSymbol, "_funcFoo", false, false, false, {iOSTarget}},
      {EncodeKind::GlobalSymbol, "_globalVar", false, true, true, {iOSTarget}},
      {EncodeKind::ObjectiveCClass, "ClassA", false, false, true, {iOSTarget}},
      {EncodeKind::ObjectiveCClass, "ClassB", false, false, true, {iOSTarget}},
      {EncodeKind::ObjectiveCClass,
       "ClassData",
       false,
       false,
       true,
       {iOSTarget}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassA",
       false,
       false,
       true,
       {iOSTarget}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassB",
       false,
       false,
       true,
       {iOSTarget}},
  };

  EXPECT_EQ(ExpectedExports.size(), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(ExpectedExports)));
}

TEST(TBDv5, WriteFile) {
  static const char TBDv5File[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }
  ],
  "install_names": [
    {
        "name": "@rpath/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "compatibility_versions": [
    { "version": "1.1" }
  ],
  "flags": [
    {
      "attributes": [
            "flat_namespace"
        ]
    }
  ],
  "rpaths": [
    {
      "targets": [
          "x86_64-macos"
      ],
      "paths": [
          "@executable_path/.../Frameworks"
      ]
    }
  ],
  "parent_umbrellas": [
    {
      "umbrella": "System"
    }
  ],
  "allowable_clients": [
    {
        "clients": [
            "ClientA",
            "ClientB"
        ]
    }
  ],
  "reexported_libraries": [
    {
        "names": [
            "/u/l/l/libfoo.dylib",
            "/u/l/l/libbar.dylib"
        ]
    }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_global"
            ],
            "objc_class": [
                "ClassA"
            ],
            "weak": [],
            "thread_local": []
        },
        "text": {
            "global": [
                "_func"
            ],
            "weak": [],
            "thread_local": []
        }
    },
    {
      "targets": [
          "x86_64-macos"
      ],
      "data": {
          "global": [
              "_globalVar"
          ],
          "objc_class": [
              "ClassA",
              "ClassB",
              "ClassData"
          ],
          "objc_eh_type": [
              "ClassA",
              "ClassB"
          ],
          "objc_ivar": [
              "ClassA.ivar1",
              "ClassA.ivar2",
              "ClassC.ivar1"
          ]
      },
      "text": {
          "global": [
              "_funcFoo"
          ]
      }
    }
  ],
  "reexported_symbols": [
    {
        "data": {
            "global": [
                "_globalRe"
            ],
            "objc_class": [
                "ClassRexport"
            ]
        },
        "text": {
            "global": [
                "_funcA"
            ]
        }
    }
  ],
  "undefined_symbols": [
    {
        "targets": [
            "x86_64-macos"
        ],
        "data": {
            "global": [
                "_globalBind"
            ],
            "weak": [
                "referenced_sym"
            ]
        }
    }
  ]
}})";

  InterfaceFile File;
  File.setFileType(FileType::TBD_V5);

  TargetList AllTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(12, 1)),
  };
  File.addTargets(AllTargets);
  File.setInstallName("@rpath/S/L/F/Foo.framework/Foo");
  File.setCurrentVersion(PackedVersion(1, 2, 0));
  File.setCompatibilityVersion(PackedVersion(1, 1, 0));
  File.addRPath(AllTargets[0], "@executable_path/.../Frameworks");

  for (const auto &Targ : AllTargets) {
    File.addParentUmbrella(Targ, "System");
    File.addAllowableClient("ClientA", Targ);
    File.addAllowableClient("ClientB", Targ);
    File.addReexportedLibrary("/u/l/l/libfoo.dylib", Targ);
    File.addReexportedLibrary("/u/l/l/libbar.dylib", Targ);
  }

  SymbolFlags Flags = SymbolFlags::None;
  // Exports.
  File.addSymbol(EncodeKind::GlobalSymbol, "_global",
                 {AllTargets[0], AllTargets[1]}, Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::GlobalSymbol, "_func",
                 {AllTargets[0], AllTargets[1]}, Flags | SymbolFlags::Text);
  File.addSymbol(EncodeKind::ObjectiveCClass, "ClassA",
                 {AllTargets[0], AllTargets[1]}, Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::GlobalSymbol, "_funcFoo", {AllTargets[0]},
                 Flags | SymbolFlags::Text);
  File.addSymbol(EncodeKind::GlobalSymbol, "_globalVar", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::ObjectiveCClass, "ClassData", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::ObjectiveCClassEHType, "ClassA", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::ObjectiveCClassEHType, "ClassB", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::ObjectiveCInstanceVariable, "ClassA.ivar1",
                 {AllTargets[0]}, Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::ObjectiveCInstanceVariable, "ClassA.ivar2",
                 {AllTargets[0]}, Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::ObjectiveCInstanceVariable, "ClassC.ivar1",
                 {AllTargets[0]}, Flags | SymbolFlags::Data);

  // Reexports.
  Flags = SymbolFlags::Rexported;
  File.addSymbol(EncodeKind::GlobalSymbol, "_globalRe", AllTargets,
                 Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::GlobalSymbol, "_funcA", AllTargets,
                 Flags | SymbolFlags::Text);
  File.addSymbol(EncodeKind::ObjectiveCClass, "ClassRexport", AllTargets,
                 Flags | SymbolFlags::Data);

  // Undefineds.
  Flags = SymbolFlags::Undefined;
  File.addSymbol(EncodeKind::GlobalSymbol, "_globalBind", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(EncodeKind::GlobalSymbol, "referenced_sym", {AllTargets[0]},
                 Flags | SymbolFlags::Data | SymbolFlags::WeakReferenced);

  File.setTwoLevelNamespace(false);
  File.setApplicationExtensionSafe(true);

  // Write out file then process it back into IF and compare equality
  // against TBDv5File.
  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error Result = TextAPIWriter::writeToStream(OS, File);
  EXPECT_FALSE(Result);

  Expected<TBDFile> Input =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Input.tbd"));
  EXPECT_TRUE(!!Input);
  TBDFile InputFile = std::move(Input.get());

  Expected<TBDFile> Output =
      TextAPIReader::get(MemoryBufferRef(Buffer, "Output.tbd"));
  EXPECT_TRUE(!!Output);
  TBDFile OutputFile = std::move(Output.get());
  EXPECT_EQ(*InputFile, *OutputFile);
}

TEST(TBDv5, WriteMultipleDocuments) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "armv7-ios",
      "min_deployment": "11.0" 
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ],
  "reexported_libraries": [
    { "names": ["/u/l/l/libfoo.dylib"] 
    }
  ]
},
"libraries": [
  {
    "target_info": [
      {
        "target": "armv7-ios",
        "min_deployment": "11.0" 
      },
      {
        "target": "armv7s-ios",
        "min_deployment": "11.0" 
      }
    ],
    "install_names":[
      { "name":"/u/l/l/libfoo.dylib" }
    ],
    "current_versions": [
      {
          "version": "2.1.1"
      }
    ],
    "rpaths": [
      {
        "targets": [
            "armv7-ios"
        ],
        "paths": [
            "@executable_path/.../Frameworks"
        ]
      }],
    "reexported_libraries": [ { "names": ["@rpath/libfoo.dylib"] } ],
    "flags":[ 
      { "attributes": ["not_app_extension_safe"] }
    ], 
    "exported_symbols": [
      {
        "text": {
          "global": [ "_funcFoo" ]
        }
      }
    ]
  },
  {
    "target_info": [
      {
        "target": "armv7-ios",
        "min_deployment": "11.0" 
      }
    ],
    "install_names":[
      { "name":"@rpath/libfoo.dylib" }
    ],
    "exported_symbols": [
      {
        "data": {
          "global": [ "_varFooBaz" ]
        }
      }
    ]
  }
]})";

  InterfaceFile File;
  File.setFileType(FileType::TBD_V5);

  TargetList AllTargets = {
      Target(AK_armv7, PLATFORM_IOS, VersionTuple(11, 0)),
      Target(AK_armv7s, PLATFORM_IOS, VersionTuple(11, 0)),
  };
  File.setInstallName("/S/L/F/Foo.framework/Foo");
  File.addTarget(AllTargets[0]);
  File.setCurrentVersion(PackedVersion(1, 0, 0));
  File.setCompatibilityVersion(PackedVersion(1, 0, 0));
  File.addReexportedLibrary("/u/l/l/libfoo.dylib", AllTargets[0]);
  File.setTwoLevelNamespace();
  File.setApplicationExtensionSafe(true);

  InterfaceFile NestedFile;
  NestedFile.setFileType(FileType::TBD_V5);
  NestedFile.setInstallName("/u/l/l/libfoo.dylib");
  NestedFile.addTargets(AllTargets);
  NestedFile.setCompatibilityVersion(PackedVersion(1, 0, 0));
  NestedFile.setTwoLevelNamespace();
  NestedFile.setApplicationExtensionSafe(false);
  NestedFile.setCurrentVersion(PackedVersion(2, 1, 1));
  NestedFile.addRPath(AllTargets[0], "@executable_path/.../Frameworks");
  for (const auto &Targ : AllTargets)
    NestedFile.addReexportedLibrary("@rpath/libfoo.dylib", Targ);
  NestedFile.addSymbol(EncodeKind::GlobalSymbol, "_funcFoo", AllTargets,
                       SymbolFlags::Text);
  File.addDocument(std::make_shared<InterfaceFile>(std::move(NestedFile)));

  InterfaceFile NestedFileB;
  NestedFileB.setFileType(FileType::TBD_V5);
  NestedFileB.setInstallName("@rpath/libfoo.dylib");
  NestedFileB.addTarget(AllTargets[0]);
  NestedFileB.setCompatibilityVersion(PackedVersion(1, 0, 0));
  NestedFileB.setCurrentVersion(PackedVersion(1, 0, 0));
  NestedFileB.setTwoLevelNamespace();
  NestedFileB.setApplicationExtensionSafe(true);
  NestedFileB.addSymbol(EncodeKind::GlobalSymbol, "_varFooBaz", {AllTargets[0]},
                        SymbolFlags::Data);
  File.addDocument(std::make_shared<InterfaceFile>(std::move(NestedFileB)));

  // Write out file then process it back into IF and compare equality
  // against TBDv5File.
  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error Result = TextAPIWriter::writeToStream(OS, File, FileType::Invalid,
                                              /*Compact=*/true);
  EXPECT_FALSE(Result);

  Expected<TBDFile> Input =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Input.tbd"));
  EXPECT_TRUE(!!Input);
  TBDFile InputFile = std::move(Input.get());

  Expected<TBDFile> Output =
      TextAPIReader::get(MemoryBufferRef(Buffer, "Output.tbd"));
  EXPECT_TRUE(!!Output);
  TBDFile OutputFile = std::move(Output.get());
  EXPECT_EQ(*InputFile, *OutputFile);
}

TEST(TBDv5, Target_Simulator) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator",
      "min_deployment": "11.0"
    },
    {
      "target": "x86_64-ios-simulator",
      "min_deployment": "11.3" 
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  TargetList ExpectedTargets = {
      Target(AK_x86_64, PLATFORM_IOSSIMULATOR, VersionTuple(11, 3)),
      Target(AK_arm64, PLATFORM_IOSSIMULATOR, VersionTuple(14, 0)),
  };
  TargetList Targets{File->targets().begin(), File->targets().end()};
  llvm::sort(Targets);
  EXPECT_EQ(Targets, ExpectedTargets);

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);

  Expected<TBDFile> Output =
      TextAPIReader::get(MemoryBufferRef(Buffer, "Output.tbd"));
  EXPECT_TRUE(!!Output);
  TBDFile WriteResultFile = std::move(Output.get());
  EXPECT_EQ(*File, *WriteResultFile);
}

TEST(TBDv5, Target_UnsupportedMinOS) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14" 
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  TargetList ExpectedTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0)),
  };
  TargetList Targets{File->targets().begin(), File->targets().end()};
  llvm::sort(Targets);
  EXPECT_EQ(Targets, ExpectedTargets);

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);

  Expected<TBDFile> Output =
      TextAPIReader::get(MemoryBufferRef(Buffer, "Output.tbd"));
  EXPECT_TRUE(!!Output);
  TBDFile WriteResultFile = std::move(Output.get());
  EXPECT_EQ(*File, *WriteResultFile);
}

TEST(TBDv5, MisspelledKey) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator",
      "min_deployment": "11.0"
    }
  ],
  "intall_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("invalid install_names section\n", ErrorMessage);
}

TEST(TBDv5, InvalidVersion) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 11,
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator",
      "min_deployment": "11.0"
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("invalid tapi_tbd_version section\n", ErrorMessage);
}

TEST(TBDv5, MissingRequiredKey) {
  static const char TBDv5File[] = R"({ 
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator",
      "min_deployment": "11.0"
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("invalid tapi_tbd_version section\n", ErrorMessage);
}

TEST(TBDv5, InvalidSymbols) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-driverkit",
      "min_deployment": "11.0"
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ],
  "exported_symbols": [
    {
      "daa": {
        "global": {
            "weak": []
          }
      }
    }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("invalid exported_symbols section\n", ErrorMessage);
}

TEST(TBDv5, DefaultMinOS) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator"
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"), File->getInstallName());
  EXPECT_TRUE(File->targets().begin() != File->targets().end());
  EXPECT_EQ(*File->targets().begin(),
            Target(AK_arm64, PLATFORM_IOSSIMULATOR, VersionTuple(0, 0)));
}

TEST(TBDv5, InvalidMinOS) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator",
      "min_deployment": "swift-abi"
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_FALSE(!!Result);
  std::string ErrorMessage = toString(Result.takeError());
  EXPECT_EQ("invalid min_deployment section\n", ErrorMessage);
}

TEST(TBDv5, SimSupport) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-macos",
      "min_deployment": "11.1" 
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ],
  "flags":[ 
    { "attributes": ["sim_support"] }
  ] 
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  Target ExpectedTarget = Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 1));
  TBDFile ReadFile = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, ReadFile->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"),
            ReadFile->getInstallName());
  EXPECT_TRUE(ReadFile->targets().begin() != ReadFile->targets().end());
  EXPECT_EQ(*ReadFile->targets().begin(), ExpectedTarget);
  EXPECT_TRUE(ReadFile->hasSimulatorSupport());
}

TEST(TBDv5, NotForSharedCache) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-macos",
      "min_deployment": "11.1" 
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ],
  "flags":[ 
    { "attributes": ["not_for_dyld_shared_cache"] }
  ] 
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  Target ExpectedTarget = Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 1));
  TBDFile ReadFile = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, ReadFile->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"),
            ReadFile->getInstallName());
  EXPECT_TRUE(ReadFile->targets().begin() != ReadFile->targets().end());
  EXPECT_EQ(*ReadFile->targets().begin(), ExpectedTarget);
  EXPECT_FALSE(ReadFile->hasSimulatorSupport());
  EXPECT_TRUE(ReadFile->isOSLibNotForSharedCache());
}

TEST(TBDv5, ObjCInterfaces) {
  static const char TBDv5File[] = R"({ 
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "arm64-ios-simulator",
      "min_deployment": "14.0"
    }
  ],
  "install_names":[
    { "name":"/S/L/F/Foo.framework/Foo" }
  ],
  "exported_symbols": [
    {
      "data": {
         "global": [
              "_global",
              "_OBJC_METACLASS_$_Standalone",
              "_OBJC_CLASS_$_Standalone2"
          ],
          "weak": ["_OBJC_EHTYPE_$_NSObject"],
          "objc_class": [
              "ClassA",
              "ClassB"
          ],
          "objc_eh_type": ["ClassA"]
      }
    }]
}})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  Target ExpectedTarget =
      Target(AK_arm64, PLATFORM_IOSSIMULATOR, VersionTuple(14, 0));
  EXPECT_EQ(*File->targets().begin(), ExpectedTarget);

  // Check Symbols.
  ExportedSymbolSeq Exports;
  for (const auto *Sym : File->symbols()) {
    ExportedSymbol Temp =
        ExportedSymbol{Sym->getKind(), std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(), Sym->isData()};
    Exports.emplace_back(std::move(Temp));
  }
  llvm::sort(Exports);

  std::vector<ExportedSymbol> ExpectedExports = {
      {EncodeKind::GlobalSymbol, "_OBJC_CLASS_$_Standalone2", false, false,
       true},
      {EncodeKind::GlobalSymbol, "_OBJC_EHTYPE_$_NSObject", true, false, true},
      {EncodeKind::GlobalSymbol, "_OBJC_METACLASS_$_Standalone", false, false,
       true},
      {EncodeKind::GlobalSymbol, "_global", false, false, true},
      {EncodeKind::ObjectiveCClass, "ClassA", false, false, true},
      {EncodeKind::ObjectiveCClass, "ClassB", false, false, true},
      {EncodeKind::ObjectiveCClassEHType, "ClassA", false, false, true}};

  EXPECT_EQ(ExpectedExports.size(), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(ExpectedExports)));

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error WriteResult = TextAPIWriter::writeToStream(OS, *File);
  EXPECT_TRUE(!WriteResult);

  Expected<TBDFile> Output =
      TextAPIReader::get(MemoryBufferRef(Buffer, "Output.tbd"));
  EXPECT_TRUE(!!Output);
  TBDFile WriteResultFile = std::move(Output.get());
  EXPECT_EQ(*File, *WriteResultFile);
}

TEST(TBDv5, MergeIF) {
  static const char TBDv5FileA[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }
  ],
  "flags": [
    {
      "targets": [
            "x86_64-macos"
        ],
      "attributes": [
            "flat_namespace"
        ]
    }
  ],
  "install_names": [
    {
        "name": "/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "compatibility_versions": [
    { "version": "1.1" }
  ],
  "rpaths": [
    {
      "targets": [
          "x86_64-macos"
      ],
      "paths": [
          "@executable_path/.../Frameworks"
      ]
    }
  ],
  "parent_umbrellas": [
    {
      "umbrella": "System"
    }
  ],
  "allowable_clients": [
    {
        "clients": [
            "ClientA",
            "ClientB"
        ]
    }
  ],
  "reexported_libraries": [
    {
        "names": [
            "/u/l/l/libfoo.dylib",
            "/u/l/l/libbar.dylib"
        ]
    }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_global"
            ],
            "objc_class": [
                "ClassA"
            ],
            "weak": [],
            "thread_local": []
        },
        "text": {
            "global": [
                "_func"
            ],
            "weak": [],
            "thread_local": []
        }
    },
    {
      "targets": [
          "x86_64-macos"
      ],
      "data": {
          "global": [
              "_globalVar"
          ],
          "objc_class": [
              "ClassA",
              "ClassB",
              "ClassData"
          ],
          "objc_eh_type": [
              "ClassA",
              "ClassB"
          ],
          "objc_ivar": [
              "ClassA.ivar1",
              "ClassA.ivar2",
              "ClassC.ivar1"
          ]
      },
      "text": {
          "global": [
              "_funcFoo"
          ]
      }
    }
  ],
  "reexported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_globalRe"
            ],
            "objc_class": [
                "ClassRexport"
            ]
        },
        "text": {
            "global": [
                "_funcA"
            ]
        }
    }
  ],
  "undefined_symbols": [
    {
        "targets": [
            "x86_64-macos"
        ],
        "data": {
            "global": [
                "_globalBind"
            ],
            "weak": [
                "referenced_sym"
            ]
        }
    }
  ]
},
"libraries": []
})";

  static const char TBDv5FileB[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }
  ],
  "flags": [
    {
      "targets": [
            "x86_64-macos"
        ],
      "attributes": [
            "flat_namespace"
        ]
    }
  ],
  "install_names": [
    {
        "name": "/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "compatibility_versions": [
    { "version": "1.1" }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_globalZ"
            ],
            "objc_class": [
                "ClassZ"
            ],
            "weak": [],
            "thread_local": []
        },
        "text": {
            "global": [
                "_funcZ"
            ],
            "weak": [],
            "thread_local": []
        }
    },
    {
      "targets": [
          "x86_64-macos"
      ],
      "data": {
          "global": [
              "_globalVarZ"
          ],
          "objc_class": [
              "ClassZ",
              "ClassF"
          ],
          "objc_eh_type": [
              "ClassZ",
              "ClassF"
          ],
          "objc_ivar": [
              "ClassZ.ivar1",
              "ClassZ.ivar2",
              "ClassF.ivar1"
          ]
      },
      "text": {
          "global": [
              "_funcFooZ"
          ]
      }
    }
  ]
},
"libraries": []
})";

  Expected<TBDFile> ResultA =
      TextAPIReader::get(MemoryBufferRef(TBDv5FileA, "Test.tbd"));
  EXPECT_TRUE(!!ResultA);
  TBDFile FileA = std::move(ResultA.get());

  Expected<TBDFile> ResultB =
      TextAPIReader::get(MemoryBufferRef(TBDv5FileB, "Test.tbd"));
  EXPECT_TRUE(!!ResultB);
  TBDFile FileB = std::move(ResultB.get());

  Expected<TBDFile> MergedResult = FileA->merge(FileB.get());
  EXPECT_TRUE(!!MergedResult);
  TBDFile MergedFile = std::move(MergedResult.get());

  EXPECT_EQ(FileType::TBD_V5, MergedFile->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"),
            MergedFile->getInstallName());
  TargetList AllTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0, 0)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(14, 0)),
  };
  EXPECT_EQ(mapToPlatformSet(AllTargets), MergedFile->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets), MergedFile->getArchitectures());
  EXPECT_EQ(PackedVersion(1, 2, 0), MergedFile->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 1, 0), MergedFile->getCompatibilityVersion());
  EXPECT_TRUE(MergedFile->isApplicationExtensionSafe());
  EXPECT_FALSE(MergedFile->isTwoLevelNamespace());
  EXPECT_EQ(0U, MergedFile->documents().size());
  InterfaceFileRef ClientA("ClientA", AllTargets);
  InterfaceFileRef ClientB("ClientB", AllTargets);
  EXPECT_EQ(2U, MergedFile->allowableClients().size());
  EXPECT_EQ(ClientA, MergedFile->allowableClients().at(0));
  EXPECT_EQ(ClientB, MergedFile->allowableClients().at(1));

  InterfaceFileRef ReexportA("/u/l/l/libbar.dylib", AllTargets);
  InterfaceFileRef ReexportB("/u/l/l/libfoo.dylib", AllTargets);
  EXPECT_EQ(2U, MergedFile->reexportedLibraries().size());
  EXPECT_EQ(ReexportA, MergedFile->reexportedLibraries().at(0));
  EXPECT_EQ(ReexportB, MergedFile->reexportedLibraries().at(1));

  TargetToAttr RPaths = {
      {Target(AK_x86_64, PLATFORM_MACOS), "@executable_path/.../Frameworks"},
  };
  EXPECT_EQ(RPaths, MergedFile->rpaths());

  TargetToAttr Umbrellas = {{Target(AK_x86_64, PLATFORM_MACOS), "System"},
                            {Target(AK_arm64, PLATFORM_MACOS), "System"},
                            {Target(AK_arm64, PLATFORM_MACCATALYST), "System"}};
  EXPECT_EQ(Umbrellas, MergedFile->umbrellas());

  ExportedSymbolSeq Exports, Reexports, Undefineds;
  for (const auto *Sym : MergedFile->symbols()) {
    TargetList SymTargets{Sym->targets().begin(), Sym->targets().end()};
    ExportedSymbol Temp =
        ExportedSymbol{Sym->getKind(),
                       std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(),
                       Sym->isData(),
                       SymTargets};
    if (Sym->isUndefined())
      Undefineds.emplace_back(std::move(Temp));
    else
      Sym->isReexported() ? Reexports.emplace_back(std::move(Temp))
                          : Exports.emplace_back(std::move(Temp));
  }
  llvm::sort(Exports);
  llvm::sort(Reexports);
  llvm::sort(Undefineds);

  TargetList MacOSTargets = {Target(AK_x86_64, PLATFORM_MACOS),
                             Target(AK_arm64, PLATFORM_MACOS)};

  std::vector<ExportedSymbol> ExpectedExportedSymbols = {
      {EncodeKind::GlobalSymbol, "_func", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol,
       "_funcFoo",
       false,
       false,
       false,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol,
       "_funcFooZ",
       false,
       false,
       false,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol, "_funcZ", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_global", false, false, true, MacOSTargets},
      {EncodeKind::GlobalSymbol,
       "_globalVar",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol,
       "_globalVarZ",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol, "_globalZ", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass,
       "ClassA",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassB",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassData",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassF",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClass,
       "ClassZ",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassA",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassB",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassF",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCClassEHType,
       "ClassZ",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassA.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassA.ivar2",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassC.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassF.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassZ.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::ObjectiveCInstanceVariable,
       "ClassZ.ivar2",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
  };

  std::vector<ExportedSymbol> ExpectedReexportedSymbols = {
      {EncodeKind::GlobalSymbol, "_funcA", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_globalRe", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass, "ClassRexport", false, false, true,
       MacOSTargets},
  };

  std::vector<ExportedSymbol> ExpectedUndefinedSymbols = {
      {EncodeKind::GlobalSymbol,
       "_globalBind",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {EncodeKind::GlobalSymbol,
       "referenced_sym",
       true,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
  };

  EXPECT_EQ(ExpectedExportedSymbols.size(), Exports.size());
  EXPECT_EQ(ExpectedReexportedSymbols.size(), Reexports.size());
  EXPECT_EQ(ExpectedUndefinedSymbols.size(), Undefineds.size());
  EXPECT_TRUE(std::equal(Exports.begin(), Exports.end(),
                         std::begin(ExpectedExportedSymbols)));
  EXPECT_TRUE(std::equal(Reexports.begin(), Reexports.end(),
                         std::begin(ExpectedReexportedSymbols)));
  EXPECT_TRUE(std::equal(Undefineds.begin(), Undefineds.end(),
                         std::begin(ExpectedUndefinedSymbols)));
}

TEST(TBDv5, ExtractIF) {
  static const char TBDv5File[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }
  ],
  "flags": [
    {
      "targets": [
            "x86_64-macos"
        ],
      "attributes": [
            "flat_namespace"
        ]
    }
  ],
  "install_names": [
    {
        "name": "/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "compatibility_versions": [
    { "version": "1.1" }
  ],
  "rpaths": [
    {
      "targets": [
          "x86_64-macos"
      ],
      "paths": [
          "@executable_path/.../Frameworks"
      ]
    }
  ],
  "parent_umbrellas": [
    {
      "umbrella": "System"
    }
  ],
  "allowable_clients": [
    {
        "clients": [
            "ClientA",
            "ClientB"
        ]
    }
  ],
  "reexported_libraries": [
    {
        "names": [
            "/u/l/l/libfoo.dylib",
            "/u/l/l/libbar.dylib"
        ]
    }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_global"
            ],
            "objc_class": [
                "ClassA"
            ],
            "weak": [],
            "thread_local": []
        },
        "text": {
            "global": [
                "_func"
            ],
            "weak": [],
            "thread_local": []
        }
    },
    {
      "targets": [
          "x86_64-macos"
      ],
      "data": {
          "global": [
              "_globalVar"
          ],
          "objc_class": [
              "ClassA",
              "ClassB",
              "ClassData"
          ],
          "objc_eh_type": [
              "ClassA",
              "ClassB"
          ],
          "objc_ivar": [
              "ClassA.ivar1",
              "ClassA.ivar2",
              "ClassC.ivar1"
          ]
      },
      "text": {
          "global": [
              "_funcFoo"
          ]
      }
    }
  ],
  "reexported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_globalRe"
            ],
            "objc_class": [
                "ClassRexport"
            ]
        },
        "text": {
            "global": [
                "_funcA"
            ]
        }
    }
  ],
  "undefined_symbols": [
    {
        "targets": [
            "x86_64-macos"
        ],
        "data": {
            "global": [
                "_globalBind"
            ],
            "weak": [
                "referenced_sym"
            ]
        }
    }
  ]
},
"libraries": []
})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());

  Expected<TBDFile> ExtractedResult = File->extract(AK_arm64);
  EXPECT_TRUE(!!ExtractedResult);
  TBDFile ExtractedFile = std::move(ExtractedResult.get());

  EXPECT_EQ(FileType::TBD_V5, ExtractedFile->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"),
            ExtractedFile->getInstallName());

  TargetList AllTargets = {
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0, 0)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(14, 0)),
  };
  EXPECT_EQ(mapToPlatformSet(AllTargets), ExtractedFile->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets),
            ExtractedFile->getArchitectures());

  EXPECT_EQ(PackedVersion(1, 2, 0), ExtractedFile->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 1, 0), ExtractedFile->getCompatibilityVersion());
  EXPECT_TRUE(ExtractedFile->isApplicationExtensionSafe());
  EXPECT_FALSE(ExtractedFile->isTwoLevelNamespace());
  EXPECT_EQ(0U, ExtractedFile->documents().size());

  InterfaceFileRef ClientA("ClientA", AllTargets);
  InterfaceFileRef ClientB("ClientB", AllTargets);
  EXPECT_EQ(2U, ExtractedFile->allowableClients().size());
  EXPECT_EQ(ClientA, ExtractedFile->allowableClients().at(0));
  EXPECT_EQ(ClientB, ExtractedFile->allowableClients().at(1));

  InterfaceFileRef ReexportA("/u/l/l/libbar.dylib", AllTargets);
  InterfaceFileRef ReexportB("/u/l/l/libfoo.dylib", AllTargets);
  EXPECT_EQ(2U, ExtractedFile->reexportedLibraries().size());
  EXPECT_EQ(ReexportA, ExtractedFile->reexportedLibraries().at(0));
  EXPECT_EQ(ReexportB, ExtractedFile->reexportedLibraries().at(1));

  EXPECT_EQ(0u, ExtractedFile->rpaths().size());

  TargetToAttr Umbrellas = {{Target(AK_arm64, PLATFORM_MACOS), "System"},
                            {Target(AK_arm64, PLATFORM_MACCATALYST), "System"}};
  EXPECT_EQ(Umbrellas, ExtractedFile->umbrellas());

  ExportedSymbolSeq Exports, Reexports, Undefineds;
  for (const auto *Sym : ExtractedFile->symbols()) {
    TargetList SymTargets{Sym->targets().begin(), Sym->targets().end()};
    ExportedSymbol Temp =
        ExportedSymbol{Sym->getKind(),
                       std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(),
                       Sym->isData(),
                       SymTargets};
    if (Sym->isUndefined())
      Undefineds.emplace_back(std::move(Temp));
    else
      Sym->isReexported() ? Reexports.emplace_back(std::move(Temp))
                          : Exports.emplace_back(std::move(Temp));
  }
  llvm::sort(Exports);
  llvm::sort(Reexports);
  llvm::sort(Undefineds);

  TargetList MacOSTargets = {Target(AK_arm64, PLATFORM_MACOS)};

  std::vector<ExportedSymbol> ExpectedExportedSymbols = {
      {EncodeKind::GlobalSymbol, "_func", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_global", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass, "ClassA", false, false, true, MacOSTargets},
  };
  std::vector<ExportedSymbol> ExpectedReexportedSymbols = {
      {EncodeKind::GlobalSymbol, "_funcA", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_globalRe", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass, "ClassRexport", false, false, true,
       MacOSTargets},
  };

  EXPECT_EQ(ExpectedExportedSymbols.size(), Exports.size());
  EXPECT_EQ(ExpectedReexportedSymbols.size(), Reexports.size());
  EXPECT_EQ(0U, Undefineds.size());
  EXPECT_TRUE(std::equal(Exports.begin(), Exports.end(),
                         std::begin(ExpectedExportedSymbols)));
  EXPECT_TRUE(std::equal(Reexports.begin(), Reexports.end(),
                         std::begin(ExpectedReexportedSymbols)));
}

TEST(TBDv5, RemoveIF) {
  static const char TBDv5File[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }
  ],
  "flags": [
    {
      "targets": [
            "x86_64-macos"
        ],
      "attributes": [
            "flat_namespace"
        ]
    }
  ],
  "install_names": [
    {
        "name": "/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "compatibility_versions": [
    { "version": "1.1" }
  ],
  "rpaths": [
    {
      "targets": [
          "x86_64-macos"
      ],
      "paths": [
          "@executable_path/.../Frameworks"
      ]
    }
  ],
  "parent_umbrellas": [
    {
      "umbrella": "System"
    }
  ],
  "allowable_clients": [
    {
        "clients": [
            "ClientA",
            "ClientB"
        ]
    }
  ],
  "reexported_libraries": [
    {
        "names": [
            "/u/l/l/libfoo.dylib",
            "/u/l/l/libbar.dylib"
        ]
    }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_global"
            ],
            "objc_class": [
                "ClassA"
            ],
            "weak": [],
            "thread_local": []
        },
        "text": {
            "global": [
                "_func"
            ],
            "weak": [],
            "thread_local": []
        }
    },
    {
      "targets": [
          "x86_64-macos"
      ],
      "data": {
          "global": [
              "_globalVar"
          ],
          "objc_class": [
              "ClassA",
              "ClassB",
              "ClassData"
          ],
          "objc_eh_type": [
              "ClassA",
              "ClassB"
          ],
          "objc_ivar": [
              "ClassA.ivar1",
              "ClassA.ivar2",
              "ClassC.ivar1"
          ]
      },
      "text": {
          "global": [
              "_funcFoo"
          ]
      }
    }
  ],
  "reexported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_globalRe"
            ],
            "objc_class": [
                "ClassRexport"
            ]
        },
        "text": {
            "global": [
                "_funcA"
            ]
        }
    }
  ],
  "undefined_symbols": [
    {
        "targets": [
            "x86_64-macos"
        ],
        "data": {
            "global": [
                "_globalBind"
            ],
            "weak": [
                "referenced_sym"
            ]
        }
    }
  ]
},
"libraries": []
})";

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());

  Expected<TBDFile> RemovedResult = File->remove(AK_x86_64);
  EXPECT_TRUE(!!RemovedResult);
  TBDFile RemovedFile = std::move(RemovedResult.get());

  EXPECT_EQ(FileType::TBD_V5, RemovedFile->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"),
            RemovedFile->getInstallName());

  TargetList AllTargets = {
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0, 0)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(14, 0)),
  };
  EXPECT_EQ(mapToPlatformSet(AllTargets), RemovedFile->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets), RemovedFile->getArchitectures());

  EXPECT_EQ(PackedVersion(1, 2, 0), RemovedFile->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 1, 0), RemovedFile->getCompatibilityVersion());
  EXPECT_TRUE(RemovedFile->isApplicationExtensionSafe());
  EXPECT_FALSE(RemovedFile->isTwoLevelNamespace());
  EXPECT_EQ(0U, RemovedFile->documents().size());

  InterfaceFileRef ClientA("ClientA", AllTargets);
  InterfaceFileRef ClientB("ClientB", AllTargets);
  EXPECT_EQ(2U, RemovedFile->allowableClients().size());
  EXPECT_EQ(ClientA, RemovedFile->allowableClients().at(0));
  EXPECT_EQ(ClientB, RemovedFile->allowableClients().at(1));

  InterfaceFileRef ReexportA("/u/l/l/libbar.dylib", AllTargets);
  InterfaceFileRef ReexportB("/u/l/l/libfoo.dylib", AllTargets);
  EXPECT_EQ(2U, RemovedFile->reexportedLibraries().size());
  EXPECT_EQ(ReexportA, RemovedFile->reexportedLibraries().at(0));
  EXPECT_EQ(ReexportB, RemovedFile->reexportedLibraries().at(1));

  EXPECT_EQ(0u, RemovedFile->rpaths().size());

  TargetToAttr Umbrellas = {{Target(AK_arm64, PLATFORM_MACOS), "System"},
                            {Target(AK_arm64, PLATFORM_MACCATALYST), "System"}};
  EXPECT_EQ(Umbrellas, RemovedFile->umbrellas());

  ExportedSymbolSeq Exports, Reexports, Undefineds;
  for (const auto *Sym : RemovedFile->symbols()) {
    TargetList SymTargets{Sym->targets().begin(), Sym->targets().end()};
    ExportedSymbol Temp =
        ExportedSymbol{Sym->getKind(),
                       std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(),
                       Sym->isData(),
                       SymTargets};
    if (Sym->isUndefined())
      Undefineds.emplace_back(std::move(Temp));
    else
      Sym->isReexported() ? Reexports.emplace_back(std::move(Temp))
                          : Exports.emplace_back(std::move(Temp));
  }
  llvm::sort(Exports);
  llvm::sort(Reexports);
  llvm::sort(Undefineds);

  TargetList MacOSTargets = {Target(AK_arm64, PLATFORM_MACOS)};

  std::vector<ExportedSymbol> ExpectedExportedSymbols = {
      {EncodeKind::GlobalSymbol, "_func", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_global", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass, "ClassA", false, false, true, MacOSTargets},
  };
  std::vector<ExportedSymbol> ExpectedReexportedSymbols = {
      {EncodeKind::GlobalSymbol, "_funcA", false, false, false, MacOSTargets},
      {EncodeKind::GlobalSymbol, "_globalRe", false, false, true, MacOSTargets},
      {EncodeKind::ObjectiveCClass, "ClassRexport", false, false, true,
       MacOSTargets},
  };

  EXPECT_EQ(ExpectedExportedSymbols.size(), Exports.size());
  EXPECT_EQ(ExpectedReexportedSymbols.size(), Reexports.size());
  EXPECT_EQ(0U, Undefineds.size());
  EXPECT_TRUE(std::equal(Exports.begin(), Exports.end(),
                         std::begin(ExpectedExportedSymbols)));
  EXPECT_TRUE(std::equal(Reexports.begin(), Reexports.end(),
                         std::begin(ExpectedReexportedSymbols)));
}

TEST(TBDv5, InlineIF) {
  static const char UmbrellaFile[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    }
  ],
  "install_names": [
    {
        "name": "/S/L/F/Foo.framework/Foo"
    }
  ],
  "current_versions": [
    {
        "version": "1.2"
    }
  ],
  "reexported_libraries": [
    {
        "names": [
            "/u/l/l/libfoo.dylib",
            "/u/l/l/libbar.dylib"
        ]
    }
  ]
}})";

  static const char ReexportFile[] = R"({
"tapi_tbd_version": 5,
"main_library": {
  "target_info": [
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    }
  ],
  "install_names": [
    {
        "name" : "/u/l/l/libfoo.dylib"
    }
  ],
  "current_versions": [
    {
        "version": "1"
    }
  ],
  "rpaths": [
    {
      "targets": [
          "x86_64-macos"
      ],
      "paths": [
          "@executable_path/.../Frameworks"
      ]
    }
  ],
  "exported_symbols": [
    {
        "targets": [
            "x86_64-macos",
            "arm64-macos"
        ],
        "data": {
            "global": [
                "_global"
            ],
            "objc_class": [
                "ClassA"
            ],
            "weak": [],
            "thread_local": []
        }
    }
  ]}})";

  Expected<TBDFile> UmbrellaResult =
      TextAPIReader::get(MemoryBufferRef(UmbrellaFile, "Test.tbd"));
  EXPECT_TRUE(!!UmbrellaResult);
  TBDFile Umbrella = std::move(UmbrellaResult.get());

  Expected<TBDFile> ReexportResult =
      TextAPIReader::get(MemoryBufferRef(ReexportFile, "Test.tbd"));
  EXPECT_TRUE(!!ReexportResult);
  TBDReexportFile Reexport = std::move(ReexportResult.get());
  Umbrella->inlineLibrary(Reexport);

  EXPECT_EQ(FileType::TBD_V5, Umbrella->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"),
            Umbrella->getInstallName());

  TargetList AllTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0, 0)),
  };
  EXPECT_EQ(mapToPlatformSet(AllTargets), Umbrella->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets), Umbrella->getArchitectures());

  EXPECT_EQ(PackedVersion(1, 2, 0), Umbrella->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), Umbrella->getCompatibilityVersion());
  InterfaceFileRef ReexportA("/u/l/l/libbar.dylib", AllTargets);
  InterfaceFileRef ReexportB("/u/l/l/libfoo.dylib", AllTargets);
  EXPECT_EQ(2U, Umbrella->reexportedLibraries().size());
  EXPECT_EQ(ReexportA, Umbrella->reexportedLibraries().at(0));
  EXPECT_EQ(ReexportB, Umbrella->reexportedLibraries().at(1));
  EXPECT_EQ(1U, Umbrella->documents().size());

  TBDReexportFile Document = Umbrella->documents().front();
  EXPECT_EQ(std::string("/u/l/l/libfoo.dylib"), Document->getInstallName());
  EXPECT_EQ(0U, Document->getSwiftABIVersion());
  EXPECT_TRUE(Document->isTwoLevelNamespace());
  EXPECT_TRUE(Document->isApplicationExtensionSafe());
  EXPECT_EQ(PackedVersion(1, 0, 0), Document->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 0, 0), Document->getCompatibilityVersion());

  ExportedSymbolSeq Exports;
  for (const auto *Sym : Document->symbols()) {
    TargetList SymTargets{Sym->targets().begin(), Sym->targets().end()};
    Exports.emplace_back(
        ExportedSymbol{Sym->getKind(), std::string(Sym->getName()),
                       Sym->isWeakDefined() || Sym->isWeakReferenced(),
                       Sym->isThreadLocalValue(), Sym->isData(), SymTargets});
  }
  llvm::sort(Exports);

  ExportedSymbolSeq ExpectedExports = {
      {EncodeKind::GlobalSymbol, "_global", false, false, true, AllTargets},
      {EncodeKind::ObjectiveCClass, "ClassA", false, false, true, AllTargets},
  };
  EXPECT_EQ(ExpectedExports.size(), Exports.size());
  EXPECT_TRUE(
      std::equal(Exports.begin(), Exports.end(), std::begin(ExpectedExports)));
}
} // end namespace TBDv5
