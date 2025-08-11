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
namespace dxbc {

using namespace object;

size_t DXContainerWriter::finalize() {
  Offsets.reserve(Obj.Parts.size());
  size_t Offset = Obj.headerSize();
  for (const Part &P : Obj.Parts) {
    Offsets.push_back(Offset);
    Offset += P.size();
  }
  return Obj.Header.FileSize;
}

Error DXContainerWriter::write() {
  size_t TotalSize = finalize();
  Out.reserveExtraSpace(TotalSize);

  Out.write(reinterpret_cast<const char *>(&Obj.Header),
            sizeof(::llvm::dxbc::Header));
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

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm
