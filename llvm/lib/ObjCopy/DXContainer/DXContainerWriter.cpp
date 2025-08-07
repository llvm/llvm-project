//===- DXContainerWriter.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXContainerWriter.h"

namespace llvm {
namespace objcopy {
namespace dxc {

using namespace object;

size_t DXContainerWriter::finalize() {
  size_t ObjectSize = sizeof(dxbc::Header);
  ObjectSize += Obj.Parts.size() * sizeof(uint32_t);
  Offsets.reserve(Obj.Parts.size());
  for (const Part &P : Obj.Parts) {
    Offsets.push_back(ObjectSize);
    assert(P.Name.size() == 4 &&
           "Valid DXIL Part name consists of 4 characters");
    ObjectSize += 4 + sizeof(uint32_t) + P.Data.size();
  }
  return ObjectSize;
}

Error DXContainerWriter::write() {
  size_t TotalSize = finalize();
  Out.reserveExtraSpace(TotalSize);

  Out.write(reinterpret_cast<const char *>(&Obj.Header), sizeof(dxbc::Header));
  Out.write(reinterpret_cast<const char *>(Offsets.data()),
            Offsets.size() * sizeof(uint32_t));

  for (const Part &P : Obj.Parts) {
    Out.write(reinterpret_cast<const char *>(P.Name.data()), P.Name.size());
    uint32_t Size = P.Data.size();
    Out.write(reinterpret_cast<const char *>(&Size), sizeof(uint32_t));
    Out.write(reinterpret_cast<const char *>(P.Data.data()), P.Data.size());
  }

  return Error::success();
}

} // end namespace dxc
} // end namespace objcopy
} // end namespace llvm
