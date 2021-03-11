//===- Writer.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace objcopy {
namespace wasm {

using namespace object;
using namespace llvm::wasm;

Writer::SectionHeader Writer::createSectionHeader(const Section &S,
                                                  size_t &SectionSize) {
  SectionHeader Header;
  raw_svector_ostream OS(Header);
  OS << S.SectionType;
  bool HasName = S.SectionType == WASM_SEC_CUSTOM;
  SectionSize = S.Contents.size();
  if (HasName)
    SectionSize += getULEB128Size(S.Name.size()) + S.Name.size();
  // Pad the LEB value out to 5 bytes to make it a predictable size, and
  // match the behavior of clang.
  encodeULEB128(SectionSize, OS, 5);
  if (HasName) {
    encodeULEB128(S.Name.size(), OS);
    OS << S.Name;
  }
  // Total section size is the content size plus 1 for the section type and
  // 5 for the LEB-encoded size.
  SectionSize = SectionSize + 1 + 5;
  return Header;
}

size_t Writer::finalize() {
  size_t ObjectSize = sizeof(WasmMagic) + sizeof(WasmVersion);
  SectionHeaders.reserve(Obj.Sections.size());
  // Finalize the headers of each section so we know the total size.
  for (const Section &S : Obj.Sections) {
    size_t SectionSize;
    SectionHeaders.push_back(createSectionHeader(S, SectionSize));
    ObjectSize += SectionSize;
  }
  return ObjectSize;
}

Error Writer::write() {
  size_t TotalSize = finalize();
  std::unique_ptr<WritableMemoryBuffer> Buf =
      WritableMemoryBuffer::getNewMemBuffer(TotalSize);
  if (!Buf)
    return createStringError(errc::not_enough_memory,
                             "failed to allocate memory buffer of " +
                                 Twine::utohexstr(TotalSize) + " bytes");

  // Write the header.
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Buf->getBufferStart());
  Ptr = std::copy(Obj.Header.Magic.begin(), Obj.Header.Magic.end(), Ptr);
  support::endian::write32le(Ptr, Obj.Header.Version);
  Ptr += sizeof(Obj.Header.Version);

  // Write each section.
  for (size_t I = 0, S = SectionHeaders.size(); I < S; ++I) {
    Ptr = std::copy(SectionHeaders[I].begin(), SectionHeaders[I].end(), Ptr);
    ArrayRef<uint8_t> Contents = Obj.Sections[I].Contents;
    Ptr = std::copy(Contents.begin(), Contents.end(), Ptr);
  }

  // TODO: Implement direct writing to the output stream (without intermediate
  // memory buffer Buf).
  Out.write(Buf->getBufferStart(), Buf->getBufferSize());
  return Error::success();
}

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm
