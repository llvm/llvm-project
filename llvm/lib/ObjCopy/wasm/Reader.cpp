//===- Reader.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Reader.h"

namespace llvm {
namespace objcopy {
namespace wasm {

using namespace object;
using namespace llvm::wasm;

Expected<std::unique_ptr<Object>> Reader::create() const {
  auto Obj = std::make_unique<Object>();
  Obj->Header = WasmObj.getHeader();
  std::vector<Section> Sections;
  Obj->Sections.reserve(WasmObj.getNumSections());
  for (const SectionRef &Sec : WasmObj.sections()) {
    const WasmSection &WS = WasmObj.getWasmSection(Sec);
    Obj->Sections.push_back(
        {static_cast<uint8_t>(WS.Type), WS.Name, WS.Content});
  }
  return std::move(Obj);
}

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm
