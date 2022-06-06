//===- WasmReader.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WasmReader.h"

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
    // Give known sections standard names to allow them to be selected.
    Section &ReaderSec = Obj->Sections.back();
    if (ReaderSec.SectionType > WASM_SEC_CUSTOM &&
        ReaderSec.SectionType <= WASM_SEC_TAG)
      ReaderSec.Name = sectionTypeToString(ReaderSec.SectionType);
    // If the section type is CUSTOM, it has a name already. If it's a new type
    // of section that we don't explicitly handle here, it will have an empty
    // name and objcopy won't be able to select it by name (e.g. for removal
    // or dumping) but it will still be valid and able to be copied.
  }
  return std::move(Obj);
}

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm
