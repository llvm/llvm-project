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

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/OrcRPCTargetProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/FDRawByteChannel.h"
#include "llvm/ExecutionEngine/Orc/Shared/RPCUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/ExecutionEngine/RuntimeDyldChecker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace llvm {

struct Session;

/// ObjectLinkingLayer with additional support for symbol promotion.
class LLVMJITLinkObjectLinkingLayer : public orc::ObjectLinkingLayer {
public:
  using orc::ObjectLinkingLayer::add;

  LLVMJITLinkObjectLinkingLayer(Session &S,
                                jitlink::JITLinkMemoryManager &MemMgr);

  Error add(orc::ResourceTrackerSP RT,
            std::unique_ptr<MemoryBuffer> O) override;

private:
  Session &S;
};

using LLVMJITLinkChannel = orc::shared::FDRawByteChannel;
using LLVMJITLinkRPCEndpoint =
    orc::shared::MultiThreadedRPCEndpoint<LLVMJITLinkChannel>;
using LLVMJITLinkRemoteMemoryManager =
    orc::OrcRPCTPCJITLinkMemoryManager<LLVMJITLinkRPCEndpoint>;
using LLVMJITLinkRemoteMemoryAccess =
    orc::OrcRPCTPCMemoryAccess<LLVMJITLinkRPCEndpoint>;

class LLVMJITLinkRemoteTargetProcessControl
    : public orc::OrcRPCTargetProcessControlBase<LLVMJITLinkRPCEndpoint> {
public:
  using BaseT = orc::OrcRPCTargetProcessControlBase<LLVMJITLinkRPCEndpoint>;
  static Expected<std::unique_ptr<TargetProcessControl>> LaunchExecutor();

  static Expected<std::unique_ptr<TargetProcessControl>> ConnectToExecutor();

  Error disconnect() override;

private:
  using LLVMJITLinkRemoteMemoryAccess =
      orc::OrcRPCTPCMemoryAccess<LLVMJITLinkRemoteTargetProcessControl>;

  using LLVMJITLinkRemoteMemoryManager =
      orc::OrcRPCTPCJITLinkMemoryManager<LLVMJITLinkRemoteTargetProcessControl>;

  LLVMJITLinkRemoteTargetProcessControl(
      std::shared_ptr<orc::SymbolStringPool> SSP,
      std::unique_ptr<LLVMJITLinkChannel> Channel,
      std::unique_ptr<LLVMJITLinkRPCEndpoint> Endpoint,
      ErrorReporter ReportError, Error &Err)
      : BaseT(std::move(SSP), *Endpoint, std::move(ReportError)),
        Channel(std::move(Channel)), Endpoint(std::move(Endpoint)) {
    ErrorAsOutParameter _(&Err);

    ListenerThread = std::thread([&]() {
      while (!Finished) {
        if (auto Err = this->Endpoint->handleOne()) {
          reportError(std::move(Err));
          return;
        }
      }
    });

    if (auto Err2 = initializeORCRPCTPCBase()) {
      Err = joinErrors(std::move(Err2), disconnect());
      return;
    }

    OwnedMemAccess = std::make_unique<LLVMJITLinkRemoteMemoryAccess>(*this);
    MemAccess = OwnedMemAccess.get();
    OwnedMemMgr = std::make_unique<LLVMJITLinkRemoteMemoryManager>(*this);
    MemMgr = OwnedMemMgr.get();
  }

  std::unique_ptr<LLVMJITLinkChannel> Channel;
  std::unique_ptr<LLVMJITLinkRPCEndpoint> Endpoint;
  std::unique_ptr<TargetProcessControl::MemoryAccess> OwnedMemAccess;
  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
  std::atomic<bool> Finished{false};
  std::thread ListenerThread;
};

struct Session {
  std::unique_ptr<orc::TargetProcessControl> TPC;
  orc::ExecutionSession ES;
  orc::JITDylib *MainJD;
  LLVMJITLinkObjectLinkingLayer ObjLayer;
  std::vector<orc::JITDylib *> JDSearchOrder;

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

  using SymbolInfoMap = StringMap<MemoryRegionInfo>;
  using FileInfoMap = StringMap<FileInfo>;

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

  SymbolInfoMap SymbolInfos;
  FileInfoMap FileInfos;
  uint64_t SizeBeforePruning = 0;
  uint64_t SizeAfterFixups = 0;

  StringSet<> HarnessFiles;
  StringSet<> HarnessExternals;
  StringSet<> HarnessDefinitions;
  DenseMap<StringRef, StringRef> CanonicalWeakDefs;

private:
  Session(std::unique_ptr<orc::TargetProcessControl> TPC, Error &Err);
};

/// Record symbols, GOT entries, stubs, and sections for ELF file.
Error registerELFGraphInfo(Session &S, jitlink::LinkGraph &G);

/// Record symbols, GOT entries, stubs, and sections for MachO file.
Error registerMachOGraphInfo(Session &S, jitlink::LinkGraph &G);

} // end namespace llvm

#endif // LLVM_TOOLS_LLVM_JITLINK_LLVM_JITLINK_H
