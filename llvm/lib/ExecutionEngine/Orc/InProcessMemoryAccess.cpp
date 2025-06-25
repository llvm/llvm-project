//===------ InProcessMemoryAccess.cpp - Direct, in-process mem access -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/InProcessMemoryAccess.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

MemoryAccess::~MemoryAccess() = default;

void InProcessMemoryAccess::writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                                             WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *W.Addr.toPtr<uint8_t *>() = W.Value;
  OnWriteComplete(Error::success());
}

void InProcessMemoryAccess::writeUInt16sAsync(
    ArrayRef<tpctypes::UInt16Write> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *W.Addr.toPtr<uint16_t *>() = W.Value;
  OnWriteComplete(Error::success());
}

void InProcessMemoryAccess::writeUInt32sAsync(
    ArrayRef<tpctypes::UInt32Write> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *W.Addr.toPtr<uint32_t *>() = W.Value;
  OnWriteComplete(Error::success());
}

void InProcessMemoryAccess::writeUInt64sAsync(
    ArrayRef<tpctypes::UInt64Write> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *W.Addr.toPtr<uint64_t *>() = W.Value;
  OnWriteComplete(Error::success());
}

void InProcessMemoryAccess::writeBuffersAsync(
    ArrayRef<tpctypes::BufferWrite> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    memcpy(W.Addr.toPtr<char *>(), W.Buffer.data(), W.Buffer.size());
  OnWriteComplete(Error::success());
}

void InProcessMemoryAccess::writePointersAsync(
    ArrayRef<tpctypes::PointerWrite> Ws, WriteResultFn OnWriteComplete) {
  if (IsArch64Bit) {
    for (auto &W : Ws)
      *W.Addr.toPtr<uint64_t *>() = W.Value.getValue();
  } else {
    for (auto &W : Ws)
      *W.Addr.toPtr<uint32_t *>() = static_cast<uint32_t>(W.Value.getValue());
  }

  OnWriteComplete(Error::success());
}
} // end namespace llvm::orc
