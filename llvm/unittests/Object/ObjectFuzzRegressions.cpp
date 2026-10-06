//===-- ObjectFuzzRegressions.cpp - Fuzz regression checking -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(ObjectFuzzRegressions, OSSFUZZ30308) {
  // Regression test for
  // https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30308
  const uint8_t data[47] = {
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x00, 0x10, 0x07, 0x6c,
      0x69, 0x6e, 0x6b, 0x69, 0x6e, 0x67, 0x02, 0x08, 0xe2, 0x29, 0x01, 0x01,
      0x02, 0xea, 0x06, 0xf9, 0xee, 0x28, 0xe1, 0x2b, 0x2f, 0x09, 0x00, 0xef,
      0xbf, 0xbf, 0x00, 0x00, 0xdd, 0x73, 0x66, 0x83, 0x7b, 0x00, 0x55};

  std::string Payload(reinterpret_cast<const char *>(data), sizeof(data));
  std::unique_ptr<MemoryBuffer> Buff = MemoryBuffer::getMemBuffer(Payload);
  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createObjectFile(Buff->getMemBufferRef());
  if (auto E = ObjOrErr.takeError()) {
    consumeError(std::move(E));
  }
}
