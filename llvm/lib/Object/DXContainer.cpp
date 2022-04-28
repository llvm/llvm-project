//===- DXContainer.cpp - DXContainer object file implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/DXContainer.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/Error.h"

using namespace llvm;
using namespace llvm::object;

static Error parseFailed(const Twine &Msg) {
  return make_error<GenericBinaryError>(Msg.str(), object_error::parse_failed);
}

template <typename T>
static Error readStruct(StringRef Buffer, const char *P, T &Struct) {
  // Don't read before the beginning or past the end of the file
  if (P < Buffer.begin() || P + sizeof(T) > Buffer.end())
    return parseFailed("Reading structure out of file bounds");

  memcpy(&Struct, P, sizeof(T));
  // DXContainer is always BigEndian
  if (sys::IsBigEndianHost)
    Struct.byteSwap();
  return Error::success();
}

DXContainer::DXContainer(MemoryBufferRef O) : Data(O) {}

Error DXContainer::parseHeader() {
  return readStruct(Data.getBuffer(), Data.getBuffer().data(), Header);
}

Expected<DXContainer> DXContainer::create(MemoryBufferRef Object) {
  DXContainer Container(Object);
  if (Error Err = Container.parseHeader())
    return Err;
  return Container;
}
