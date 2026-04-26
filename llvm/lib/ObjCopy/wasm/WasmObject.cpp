//===- WasmObject.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WasmObject.h"

namespace llvm {
namespace objcopy {
namespace wasm {

using namespace object;
using namespace llvm::wasm;

void Object::addSectionWithOwnedContents(
    Section NewSection, std::unique_ptr<MemoryBuffer> &&Content) {
  Sections.push_back(NewSection);
  OwnedContents.emplace_back(std::move(Content));
}

void Object::removeSections(function_ref<bool(const Section &)> ToRemove) {
  if (isRelocatableObject) {
    // For relocatable objects, avoid actually removing any sections,
    // since that can invalidate the symbol table and relocation sections.
    // TODO: Allow removal of sections by re-generating symbol table and
    // relocation sections here instead.
    for (auto &Sec : Sections) {
      if (ToRemove(Sec)) {
        Sec.Name = ".objcopy.removed";
        Sec.SectionType = wasm::WASM_SEC_CUSTOM;
        Sec.Contents = {};
        Sec.HeaderSecSizeEncodingLen = std::nullopt;
      }
    }
  } else {
    llvm::erase_if(Sections, ToRemove);
  }
}

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm
