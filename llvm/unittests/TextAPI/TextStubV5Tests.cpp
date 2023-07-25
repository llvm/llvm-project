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

  Expected<TBDFile> Result =
      TextAPIReader::get(MemoryBufferRef(TBDv5File, "Test.tbd"));
  EXPECT_TRUE(!!Result);
  TBDFile File = std::move(Result.get());
  EXPECT_EQ(FileType::TBD_V5, File->getFileType());
  EXPECT_EQ(std::string("/S/L/F/Foo.framework/Foo"), File->getInstallName());

  TargetList AllTargets = {
      Target(AK_x86_64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(11, 0, 0)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(14, 0)),
  };
  EXPECT_EQ(mapToPlatformSet(AllTargets), File->getPlatforms());
  EXPECT_EQ(mapToArchitectureSet(AllTargets), File->getArchitectures());

  EXPECT_EQ(PackedVersion(1, 2, 0), File->getCurrentVersion());
  EXPECT_EQ(PackedVersion(1, 1, 0), File->getCompatibilityVersion());
  EXPECT_TRUE(File->isApplicationExtensionSafe());
  EXPECT_FALSE(File->isTwoLevelNamespace());
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
      {SymbolKind::GlobalSymbol, "_func", false, false, false, MacOSTargets},
      {SymbolKind::GlobalSymbol,
       "_funcFoo",
       false,
       false,
       false,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::GlobalSymbol, "_global", false, false, true, MacOSTargets},
      {SymbolKind::GlobalSymbol,
       "_globalVar",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCClass,
       "ClassA",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCClass,
       "ClassB",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCClass,
       "ClassData",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCClassEHType,
       "ClassA",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCClassEHType,
       "ClassB",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCInstanceVariable,
       "ClassA.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCInstanceVariable,
       "ClassA.ivar2",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::ObjectiveCInstanceVariable,
       "ClassC.ivar1",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
  };
  std::vector<ExportedSymbol> ExpectedReexportedSymbols = {
      {SymbolKind::GlobalSymbol, "_funcA", false, false, false, MacOSTargets},
      {SymbolKind::GlobalSymbol, "_globalRe", false, false, true, MacOSTargets},
      {SymbolKind::ObjectiveCClass, "ClassRexport", false, false, true,
       MacOSTargets},
  };

  std::vector<ExportedSymbol> ExpectedUndefinedSymbols = {
      {SymbolKind::GlobalSymbol,
       "_globalBind",
       false,
       false,
       true,
       {Target(AK_x86_64, PLATFORM_MACOS)}},
      {SymbolKind::GlobalSymbol,
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
      File->getSymbol(SymbolKind::GlobalSymbol, "_globalBind").has_value());
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
      {SymbolKind::GlobalSymbol, "_funcFoo", false, false, false, {iOSTarget}},
      {SymbolKind::GlobalSymbol, "_globalVar", false, true, true, {iOSTarget}},
      {SymbolKind::ObjectiveCClass, "ClassA", false, false, true, {iOSTarget}},
      {SymbolKind::ObjectiveCClass, "ClassB", false, false, true, {iOSTarget}},
      {SymbolKind::ObjectiveCClass,
       "ClassData",
       false,
       false,
       true,
       {iOSTarget}},
      {SymbolKind::ObjectiveCClassEHType,
       "ClassA",
       false,
       false,
       true,
       {iOSTarget}},
      {SymbolKind::ObjectiveCClassEHType,
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
  File.addSymbol(SymbolKind::GlobalSymbol, "_global",
                 {AllTargets[0], AllTargets[1]}, Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::GlobalSymbol, "_func",
                 {AllTargets[0], AllTargets[1]}, Flags | SymbolFlags::Text);
  File.addSymbol(SymbolKind::ObjectiveCClass, "ClassA",
                 {AllTargets[0], AllTargets[1]}, Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::GlobalSymbol, "_funcFoo", {AllTargets[0]},
                 Flags | SymbolFlags::Text);
  File.addSymbol(SymbolKind::GlobalSymbol, "_globalVar", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::ObjectiveCClass, "ClassData", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::ObjectiveCClassEHType, "ClassA", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::ObjectiveCClassEHType, "ClassB", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::ObjectiveCInstanceVariable, "ClassA.ivar1",
                 {AllTargets[0]}, Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::ObjectiveCInstanceVariable, "ClassA.ivar2",
                 {AllTargets[0]}, Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::ObjectiveCInstanceVariable, "ClassC.ivar1",
                 {AllTargets[0]}, Flags | SymbolFlags::Data);

  // Reexports.
  Flags = SymbolFlags::Rexported;
  File.addSymbol(SymbolKind::GlobalSymbol, "_globalRe", AllTargets,
                 Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::GlobalSymbol, "_funcA", AllTargets,
                 Flags | SymbolFlags::Text);
  File.addSymbol(SymbolKind::ObjectiveCClass, "ClassRexport", AllTargets,
                 Flags | SymbolFlags::Data);

  // Undefineds.
  Flags = SymbolFlags::Undefined;
  File.addSymbol(SymbolKind::GlobalSymbol, "_globalBind", {AllTargets[0]},
                 Flags | SymbolFlags::Data);
  File.addSymbol(SymbolKind::GlobalSymbol, "referenced_sym", {AllTargets[0]},
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
  NestedFile.addSymbol(SymbolKind::GlobalSymbol, "_funcFoo", AllTargets,
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
  NestedFileB.addSymbol(SymbolKind::GlobalSymbol, "_varFooBaz", {AllTargets[0]},
                        SymbolFlags::Data);
  File.addDocument(std::make_shared<InterfaceFile>(std::move(NestedFileB)));

  // Write out file then process it back into IF and compare equality
  // against TBDv5File.
  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  Error Result = TextAPIWriter::writeToStream(OS, File, /*Compact=*/true);
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

} // end namespace TBDv5
