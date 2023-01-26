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
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/RuntimeDyldChecker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace llvm {

struct Session;

struct Session {

  orc::ExecutionSession ES;
  orc::JITDylib *MainJD = nullptr;
  orc::ObjectLinkingLayer ObjLayer;
  orc::JITDylibSearchOrder JDSearchOrder;

  ~Session();

  static Expected<std::unique_ptr<Session>> Create(Triple TT);
  void dumpSessionInfo(raw_ostream &OS);
  void modifyPassConfig(const Triple &FTT,
                        jitlink::PassConfiguration &PassConfig);

  using MemoryRegionInfo = RuntimeDyldChecker::MemoryRegionInfo;

  struct FileInfo {
    StringMap<MemoryRegionInfo> SectionInfos;
    StringMap<MemoryRegionInfo> StubInfos;
    StringMap<MemoryRegionInfo> GOTEntryInfos;
  };

  using DynLibJDMap = std::map<std::string, orc::JITDylib *>;
  using SymbolInfoMap = StringMap<MemoryRegionInfo>;
  using FileInfoMap = StringMap<FileInfo>;

  Expected<orc::JITDylib *> getOrLoadDynamicLibrary(StringRef LibPath);
  Error loadAndLinkDynamicLibrary(orc::JITDylib &JD, StringRef LibPath);

  Expected<FileInfo &> findFileInfo(StringRef FileName);
  Expected<MemoryRegionInfo &> findSectionInfo(StringRef FileName,
                                               StringRef SectionName);
  Expected<MemoryRegionInfo &> findStubInfo(StringRef FileName,
                                            StringRef TargetName);
  Expected<MemoryRegionInfo &> findGOTEntryInfo(StringRef FileName,
                                                StringRef TargetName);

  bool isSymbolRegistered(StringRef Name);
  Expected<MemoryRegionInfo &> findSymbolInfo(StringRef SymbolName,
                                              Twine ErrorMsgStem);

  DynLibJDMap DynLibJDs;

  SymbolInfoMap SymbolInfos;
  FileInfoMap FileInfos;
  uint64_t SizeBeforePruning = 0;
  uint64_t SizeAfterFixups = 0;

  StringSet<> HarnessFiles;
  StringSet<> HarnessExternals;
  StringSet<> HarnessDefinitions;
  DenseMap<StringRef, StringRef> CanonicalWeakDefs;

private:
  Session(std::unique_ptr<orc::ExecutorProcessControl> EPC, Error &Err);
};

/// Record symbols, GOT entries, stubs, and sections for ELF file.
Error registerELFGraphInfo(Session &S, jitlink::LinkGraph &G);

/// Record symbols, GOT entries, stubs, and sections for MachO file.
Error registerMachOGraphInfo(Session &S, jitlink::LinkGraph &G);

/// Record symbols, GOT entries, stubs, and sections for COFF file.
Error registerCOFFGraphInfo(Session &S, jitlink::LinkGraph &G);

} // end namespace llvm

#endif // LLVM_TOOLS_LLVM_JITLINK_LLVM_JITLINK_H
