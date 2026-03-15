//===- DXContainerWriter.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINERWRITER_H
#define LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINERWRITER_H

#include "DXContainerObject.h"

namespace llvm {
namespace objcopy {
namespace dxbc {

using namespace object;

class DXContainerWriter {
public:
  explicit DXContainerWriter(const Object &Obj, raw_ostream &Out)
      : Obj(Obj), Out(Out) {}
  Error write();

private:
  const Object &Obj;
  raw_ostream &Out;

  SmallVector<uint32_t> Offsets;

  size_t finalize();
};

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINERWRITER_H
