//===---- llvm-jitlink.h - Session and format-specific decls ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm-jitlink Session class and tool utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_JITLINK_LLVM_JITLINK_H
#define LLVM_TOOLS_LLVM_JITLINK_LLVM_JITLINK_H

#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/LazyObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/LazyReexports.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/RuntimeDyldChecker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

struct Session {

  struct LazyLinkingSupport {
    LazyLinkingSupport(
        std::unique_ptr<orc::RedirectableSymbolManager> RSMgr,
        std::shared_ptr<orc::SimpleLazyReexportsSpeculator> Speculator,
        std::unique_ptr<orc::LazyReexportsManager> LRMgr,
        orc::ObjectLinkingLayer &ObjLinkingLayer)
        : RSMgr(std::move(RSMgr)), Speculator(std::move(Speculator)),
          LRMgr(std::move(LRMgr)),
          LazyObjLinkingLayer(ObjLinkingLayer, *this->LRMgr) {}

    std::unique_ptr<orc::RedirectableSymbolManager> RSMgr;
    std::shared_ptr<orc::SimpleLazyReexportsSpeculator> Speculator;
    std::unique_ptr<orc::LazyReexportsManager> LRMgr;
    orc::LazyObjectLinkingLayer LazyObjLinkingLayer;
  };

  orc::ExecutionSession ES;
  orc::JITDylib *MainJD = nullptr;
  orc::JITDylib *ProcessSymsJD = nullptr;
  orc::JITDylib *PlatformJD = nullptr;
  orc::ObjectLinkingLayer ObjLayer;
  std::unique_ptr<LazyLinkingSupport> LazyLinking;
  orc::JITDylibSearchOrder JDSearchOrder;
  SubtargetFeatures Features;
  std::vector<std::pair<std::string, orc::SymbolStringPtr>> LazyFnExecOrder;

  ~Session();

  static Expected<std::unique_ptr<Session>> Create(Triple TT,
                                                   SubtargetFeatures Features);
  void dumpSessionInfo(raw_ostream &OS);
  void modifyPassConfig(jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &PassConfig);

  /// For -check: wait for all files that are referenced (transitively) from
  /// the entry point *file* to be linked. (ORC's usual dependence tracking is
  /// to fine-grained here: a lookup of the main symbol will return as soon as
  /// all reachable symbols have been linked, but testcases may want to
  /// inspect side-effects in unreachable symbols)..
  void waitForFilesLinkedFromEntryPointFile() {
    std::unique_lock<std::mutex> Lock(M);
    return ActiveLinksCV.wait(Lock, [this]() { return ActiveLinks == 0; });
  }

  using MemoryRegionInfo = RuntimeDyldChecker::MemoryRegionInfo;

  struct FileInfo {
    StringMap<MemoryRegionInfo> SectionInfos;
    StringMap<SmallVector<MemoryRegionInfo, 1>> StubInfos;
    StringMap<MemoryRegionInfo> GOTEntryInfos;

    using Symbol = jitlink::Symbol;
    using LinkGraph = jitlink::LinkGraph;
    using GetSymbolTargetFunction =
        unique_function<Expected<Symbol &>(LinkGraph &G, jitlink::Block &)>;
    Error registerGOTEntry(LinkGraph &G, Symbol &Sym,
                           GetSymbolTargetFunction GetSymbolTarget);
    Error registerStubEntry(LinkGraph &G, Symbol &Sym,
                            GetSymbolTargetFunction GetSymbolTarget);
    Error registerMultiStubEntry(LinkGraph &G, Symbol &Sym,
                                 GetSymbolTargetFunction GetSymbolTarget);
  };

  using DynLibJDMap = std::map<std::string, orc::JITDylib *, std::less<>>;
  using SymbolInfoMap = DenseMap<orc::SymbolStringPtr, MemoryRegionInfo>;
  using FileInfoMap = StringMap<FileInfo>;

  Expected<orc::JITDylib *> getOrLoadDynamicLibrary(StringRef LibPath);
  Error loadAndLinkDynamicLibrary(orc::JITDylib &JD, StringRef LibPath);

  orc::ObjectLayer &getLinkLayer(bool Lazy) {
    assert((!Lazy || LazyLinking) &&
           "Lazy linking requested but not available");
    return Lazy ? static_cast<orc::ObjectLayer &>(
                      LazyLinking->LazyObjLinkingLayer)
                : static_cast<orc::ObjectLayer &>(ObjLayer);
  }

  Expected<FileInfo &> findFileInfo(StringRef FileName);
  Expected<MemoryRegionInfo &> findSectionInfo(StringRef FileName,
                                               StringRef SectionName);
  Expected<MemoryRegionInfo &> findStubInfo(StringRef FileName,
                                            StringRef TargetName,
                                            StringRef KindNameFilter);
  Expected<MemoryRegionInfo &> findGOTEntryInfo(StringRef FileName,
                                                StringRef TargetName);

  bool isSymbolRegistered(const orc::SymbolStringPtr &Name);
  Expected<MemoryRegionInfo &> findSymbolInfo(const orc::SymbolStringPtr &Name,
                                              Twine ErrorMsgStem);

  DynLibJDMap DynLibJDs;

  std::mutex M;
  std::condition_variable ActiveLinksCV;
  size_t ActiveLinks = 0;
  SymbolInfoMap SymbolInfos;
  FileInfoMap FileInfos;

  StringSet<> HarnessFiles;
  StringSet<> HarnessExternals;
  StringSet<> HarnessDefinitions;
  DenseMap<StringRef, StringRef> CanonicalWeakDefs;

  std::optional<Regex> ShowGraphsRegex;

private:
  Session(std::unique_ptr<orc::ExecutorProcessControl> EPC, Error &Err);
};

/// Record symbols, GOT entries, stubs, and sections for ELF file.
Error registerELFGraphInfo(Session &S, jitlink::LinkGraph &G);

/// Record symbols, GOT entries, stubs, and sections for MachO file.
Error registerMachOGraphInfo(Session &S, jitlink::LinkGraph &G);

/// Record symbols, GOT entries, stubs, and sections for COFF file.
Error registerCOFFGraphInfo(Session &S, jitlink::LinkGraph &G);

/// Adds a statistics gathering plugin if any stats options are used.
void enableStatistics(Session &S, bool UsingOrcRuntime);

} // end namespace llvm

#endif // LLVM_TOOLS_LLVM_JITLINK_LLVM_JITLINK_H
