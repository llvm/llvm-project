//===------ TargetProcessControl.cpp -- Target process control APIs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Process.h"

#include <mutex>

namespace llvm {
namespace orc {

TargetProcessControl::MemoryAccess::~MemoryAccess() {}

TargetProcessControl::TargetProcessControl(Triple TT, unsigned PageSize)
    : TT(std::move(TT)), PageSize(PageSize) {}

TargetProcessControl::~TargetProcessControl() {}

SelfTargetProcessControl::SelfTargetProcessControl(Triple TT, unsigned PageSize)
    : TargetProcessControl(std::move(TT), PageSize) {
  this->MemMgr = IPMM.get();
  this->MemAccess = this;
}

Expected<std::unique_ptr<SelfTargetProcessControl>>
SelfTargetProcessControl::Create() {
  auto PageSize = sys::Process::getPageSize();
  if (!PageSize)
    return PageSize.takeError();

  Triple TT(sys::getProcessTriple());

  return std::make_unique<SelfTargetProcessControl>(std::move(TT), *PageSize);
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
