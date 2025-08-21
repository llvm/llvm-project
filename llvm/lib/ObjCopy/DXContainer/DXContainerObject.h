//===- DXContainerObject.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINEROBJECT_H
#define LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINEROBJECT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/DXContainer.h"
#include <vector>

namespace llvm {
namespace objcopy {
namespace dxbc {

using namespace object;

struct Part {
  StringRef Name;
  ArrayRef<uint8_t> Data;

  size_t size() const {
    return sizeof(::llvm::dxbc::PartHeader) // base header
           + Data.size();                   // contents size
  }
};

struct Object {
  ::llvm::dxbc::Header Header;
  SmallVector<Part> Parts;

  size_t headerSize() const {
    return sizeof(::llvm::dxbc::Header)       // base header
           + sizeof(uint32_t) * Parts.size(); // part offset values
  }
};

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINEROBJECT_H
