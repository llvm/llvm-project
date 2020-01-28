//===- Writer.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJCOPY_WASM_WRITER_H
#define LLVM_TOOLS_LLVM_OBJCOPY_WASM_WRITER_H

#include "Buffer.h"
#include "Object.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace objcopy {
namespace wasm {

class Writer {
public:
  Writer(Object &Obj, Buffer &Buf) : Obj(Obj), Buf(Buf) {}
  Error write();

private:
  using SectionHeader = SmallVector<char, 8>;
  Object &Obj;
  Buffer &Buf;
  std::vector<SectionHeader> SectionHeaders;

  /// Generate a wasm section section header for S.
  /// The header consists of
  /// * A one-byte section ID (aka the section type).
  /// * The size of the section contents, encoded as ULEB128.
  /// * If the section is a custom section (type 0) it also has a name, which is
  ///   encoded as a length-prefixed string. The encoded section size *includes*
  ///   this string.
  /// See https://webassembly.github.io/spec/core/binary/modules.html#sections
  /// Return the header and store the total size in SectionSize.
  static SectionHeader createSectionHeader(const Section &S,
                                           size_t &SectionSize);
  size_t finalize();
};

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_LLVM_OBJCOPY_WASM_WRITER_H
