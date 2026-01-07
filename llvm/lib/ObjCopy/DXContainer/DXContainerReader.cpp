//===- DXContainerReader.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXContainerReader.h"

namespace llvm {
namespace objcopy {
namespace dxbc {

using namespace object;

Expected<std::unique_ptr<Object>> DXContainerReader::create() const {
  auto Obj = std::make_unique<Object>();
  Obj->Header = DXContainerObj.getHeader();
  for (const SectionRef &Part : DXContainerObj.sections()) {
    DataRefImpl PartDRI = Part.getRawDataRefImpl();
    Expected<StringRef> Name = DXContainerObj.getSectionName(PartDRI);
    if (auto E = Name.takeError())
      return E;
    assert(Name->size() == 4 &&
           "Valid DXIL Part name consists of 4 characters");
    Expected<ArrayRef<uint8_t>> Data =
        DXContainerObj.getSectionContents(PartDRI);
    if (auto E = Data.takeError())
      return E;
    Obj->Parts.push_back({*Name, *Data});
  }
  return std::move(Obj);
}

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm
