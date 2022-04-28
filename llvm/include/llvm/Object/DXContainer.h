//===- DXContainer.h - DXContainer file implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the DXContainerFile class, which implements the ObjectFile
// interface for DXContainer files.
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_DXCONTAINER_H
#define LLVM_OBJECT_DXCONTAINER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {
namespace object {
class DXContainer {
private:
  DXContainer(MemoryBufferRef O);

  MemoryBufferRef Data;
  dxbc::Header Header;

  Error parseHeader();

public:
  StringRef getData() const { return Data.getBuffer(); }
  static Expected<DXContainer> create(MemoryBufferRef Object);

  const dxbc::Header &getHeader() const { return Header; }
};

} // namespace object
} // namespace llvm

#endif // LLVM_OBJECT_DXCONTAINERFILE_H
