//===- DXContainerReader.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINERREADER_H
#define LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINERREADER_H

#include "DXContainerObject.h"

namespace llvm {
namespace objcopy {
namespace dxbc {

using namespace object;

class DXContainerReader {
public:
  explicit DXContainerReader(const DXContainerObjectFile &Obj)
      : DXContainerObj(Obj) {}
  Expected<std::unique_ptr<Object>> create() const;

private:
  const DXContainerObjectFile &DXContainerObj;
};

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_LIB_OBJCOPY_DXCONTAINER_DXCONTAINERREADER_H
