//===-- RemoteJITUtils.h - Utilities for remote-JITing with LLI -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for remote-JITing with LLI.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_REMOTEJITUTILS_H
#define LLVM_TOOLS_LLI_REMOTEJITUTILS_H

#include "llvm/ExecutionEngine/Orc/Shared/FDRawByteChannel.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include <mutex>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

// launch the remote process (see lli.cpp) and return a channel to it.
std::unique_ptr<llvm::orc::shared::FDRawByteChannel> launchRemote();

namespace llvm {

// ForwardingMM - Adapter to connect MCJIT to Orc's Remote
// memory manager.
class ForwardingMemoryManager : public llvm::RTDyldMemoryManager {
public:
  void setMemMgr(std::unique_ptr<RuntimeDyld::MemoryManager> MemMgr) {
    this->MemMgr = std::move(MemMgr);
  }

  void setResolver(std::shared_ptr<LegacyJITSymbolResolver> Resolver) {
    this->Resolver = std::move(Resolver);
  }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    return MemMgr->allocateCodeSection(Size, Alignment, SectionID, SectionName);
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {
    return MemMgr->allocateDataSection(Size, Alignment, SectionID, SectionName,
                                       IsReadOnly);
  }

  void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                              uintptr_t RODataSize, uint32_t RODataAlign,
                              uintptr_t RWDataSize,
                              uint32_t RWDataAlign) override {
    MemMgr->reserveAllocationSpace(CodeSize, CodeAlign, RODataSize, RODataAlign,
                                   RWDataSize, RWDataAlign);
  }

  bool needsToReserveAllocationSpace() override {
    return MemMgr->needsToReserveAllocationSpace();
  }

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                        size_t Size) override {
    MemMgr->registerEHFrames(Addr, LoadAddr, Size);
  }

  void deregisterEHFrames() override {
    MemMgr->deregisterEHFrames();
  }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override {
    return MemMgr->finalizeMemory(ErrMsg);
  }

  void notifyObjectLoaded(RuntimeDyld &RTDyld,
                          const object::ObjectFile &Obj) override {
    MemMgr->notifyObjectLoaded(RTDyld, Obj);
  }

  // Don't hide the sibling notifyObjectLoaded from RTDyldMemoryManager.
  using RTDyldMemoryManager::notifyObjectLoaded;

  JITSymbol findSymbol(const std::string &Name) override {
    return Resolver->findSymbol(Name);
  }

  JITSymbol
  findSymbolInLogicalDylib(const std::string &Name) override {
    return Resolver->findSymbolInLogicalDylib(Name);
  }

private:
  std::unique_ptr<RuntimeDyld::MemoryManager> MemMgr;
  std::shared_ptr<LegacyJITSymbolResolver> Resolver;
};

template <typename RemoteT>
class RemoteResolver : public LegacyJITSymbolResolver {
public:

  RemoteResolver(RemoteT &R) : R(R) {}

  JITSymbol findSymbol(const std::string &Name) override {
    if (auto Addr = R.getSymbolAddress(Name))
      return JITSymbol(*Addr, JITSymbolFlags::Exported);
    else
      return Addr.takeError();
  }

  JITSymbol findSymbolInLogicalDylib(const std::string &Name) override {
    return nullptr;
  }

public:
  RemoteT &R;
};
}

#endif
