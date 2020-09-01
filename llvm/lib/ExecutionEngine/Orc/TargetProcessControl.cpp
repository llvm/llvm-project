//===------ TargetProcessControl.cpp -- Target process control APIs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Process.h"

#include <mutex>

namespace llvm {
namespace orc {

TargetProcessControl::MemoryAccess::~MemoryAccess() {}

TargetProcessControl::~TargetProcessControl() {}

SelfTargetProcessControl::SelfTargetProcessControl(
    Triple TT, unsigned PageSize,
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr) {

  OwnedMemMgr = std::move(MemMgr);
  if (!OwnedMemMgr)
    OwnedMemMgr = std::make_unique<jitlink::InProcessMemoryManager>();

  this->TT = std::move(TT);
  this->PageSize = PageSize;
  this->MemMgr = OwnedMemMgr.get();
  this->MemAccess = this;
  if (this->TT.isOSBinFormatMachO())
    GlobalManglingPrefix = '_';
}

Expected<std::unique_ptr<SelfTargetProcessControl>>
SelfTargetProcessControl::Create(
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr) {
  auto PageSize = sys::Process::getPageSize();
  if (!PageSize)
    return PageSize.takeError();

  Triple TT(sys::getProcessTriple());

  return std::make_unique<SelfTargetProcessControl>(std::move(TT), *PageSize,
                                                    std::move(MemMgr));
}

Expected<TargetProcessControl::DylibHandle>
SelfTargetProcessControl::loadDylib(const char *DylibPath) {
  std::string ErrMsg;
  auto Dylib = std::make_unique<sys::DynamicLibrary>(
      sys::DynamicLibrary::getPermanentLibrary(DylibPath, &ErrMsg));
  if (!Dylib->isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());
  DynamicLibraries.push_back(std::move(Dylib));
  return pointerToJITTargetAddress(DynamicLibraries.back().get());
}

Expected<TargetProcessControl::LookupResult>
SelfTargetProcessControl::lookupSymbols(LookupRequest Request) {
  LookupResult R;

  for (auto &Elem : Request) {
    auto *Dylib = jitTargetAddressToPointer<sys::DynamicLibrary *>(Elem.Handle);
    assert(llvm::find_if(DynamicLibraries,
                         [=](const std::unique_ptr<sys::DynamicLibrary> &DL) {
                           return DL.get() == Dylib;
                         }) != DynamicLibraries.end() &&
           "Invalid handle");

    R.push_back(std::vector<JITTargetAddress>());
    for (auto &KV : Elem.Symbols) {
      auto &Sym = KV.first;
      std::string Tmp((*Sym).data() + !!GlobalManglingPrefix,
                      (*Sym).size() - !!GlobalManglingPrefix);
      if (void *Addr = Dylib->getAddressOfSymbol(Tmp.c_str()))
        R.back().push_back(pointerToJITTargetAddress(Addr));
      else if (KV.second == SymbolLookupFlags::RequiredSymbol) {
        // FIXME: Collect all failing symbols before erroring out.
        SymbolNameVector MissingSymbols;
        MissingSymbols.push_back(Sym);
        return make_error<SymbolsNotFound>(std::move(MissingSymbols));
      }
    }
  }

  return R;
}

void SelfTargetProcessControl::writeUInt8s(ArrayRef<UInt8Write> Ws,
                                           WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint8_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfTargetProcessControl::writeUInt16s(ArrayRef<UInt16Write> Ws,
                                            WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint16_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfTargetProcessControl::writeUInt32s(ArrayRef<UInt32Write> Ws,
                                            WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint32_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfTargetProcessControl::writeUInt64s(ArrayRef<UInt64Write> Ws,
                                            WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint64_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfTargetProcessControl::writeBuffers(ArrayRef<BufferWrite> Ws,
                                            WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    memcpy(jitTargetAddressToPointer<char *>(W.Address), W.Buffer.data(),
           W.Buffer.size());
  OnWriteComplete(Error::success());
}

} // end namespace orc
} // end namespace llvm
