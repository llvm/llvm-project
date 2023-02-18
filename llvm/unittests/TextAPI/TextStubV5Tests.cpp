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
      Target(AK_arm64, PLATFORM_MACOS, VersionTuple(10, 14)),
      Target(AK_arm64, PLATFORM_MACCATALYST, VersionTuple(12, 1)),
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
      {SymbolKind::ObjectiveCClass, "ClassA", false, false, true, MacOSTargets},
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
          "objc_class": [ "ClassData" ], 
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

} // end namespace TBDv5
