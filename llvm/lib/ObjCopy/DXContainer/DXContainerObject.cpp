//===- DXContainerObject.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXContainerObject.h"

namespace llvm {
namespace objcopy {
namespace dxbc {

Error Object::removeParts(PartPred ToRemove) {
  erase_if(Parts, ToRemove);
  return Error::success();
}

void Object::recomputeHeader() {
  Header.FileSize = headerSize();
  Header.PartCount = Parts.size();
  for (const Part &P : Parts)
    Header.FileSize += P.size();
}

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm
