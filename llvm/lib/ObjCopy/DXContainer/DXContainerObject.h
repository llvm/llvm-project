//===- DXContainerObject.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJCOPY_DXContainer_DXContainerOBJECT_H
#define LLVM_LIB_OBJCOPY_DXContainer_DXContainerOBJECT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/DXContainer.h"
#include <vector>

namespace llvm {
namespace objcopy {
namespace dxc {

using namespace object;

struct Part {
  StringRef Name;
  uint32_t Offset;
  ArrayRef<uint8_t> Data;
};

struct Object {
  dxbc::Header Header;
  SmallVector<Part> Parts;
};

} // end namespace dxc
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_LIB_OBJCOPY_DXContainer_DXContainerOBJECT_H
