//===- OffloadEmitter.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/OffloadBinary.h"
#include "llvm/ObjectYAML/OffloadYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace OffloadYAML;

namespace llvm {
namespace yaml {

bool yaml2offload(Binary &Doc, raw_ostream &Out, ErrorHandler EH) {
  SmallVector<object::OffloadBinary::OffloadingImage> Images;
  for (const auto &Member : Doc.Members) {
    object::OffloadBinary::OffloadingImage Image{};
    if (Member.ImageKind)
      Image.TheImageKind = *Member.ImageKind;
    if (Member.OffloadKind)
      Image.TheOffloadKind = *Member.OffloadKind;
    if (Member.Flags)
      Image.Flags = *Member.Flags;

    if (Member.StringEntries)
      for (const auto &Entry : *Member.StringEntries)
        Image.StringData[Entry.Key] = Entry.Value;

    SmallVector<char, 1024> Data;
    raw_svector_ostream OS(Data);
    if (Member.Content)
      Member.Content->writeAsBinary(OS);
    Image.Image = MemoryBuffer::getMemBufferCopy(OS.str());
    Images.push_back(std::move(Image));
  }

  // Copy the data to a new buffer so we can modify the bytes directly.
  auto Buffer = object::OffloadBinary::write(Images);
  auto *TheHeader =
      reinterpret_cast<object::OffloadBinary::Header *>(&Buffer[0]);
  if (Doc.Version)
    TheHeader->Version = *Doc.Version;
  if (Doc.Size)
    TheHeader->Size = *Doc.Size;
  if (Doc.EntriesOffset)
    TheHeader->EntriesOffset = *Doc.EntriesOffset;
  if (Doc.EntriesCount)
    TheHeader->EntriesCount = *Doc.EntriesCount;

  Out.write(Buffer.begin(), Buffer.size());

  return true;
}

} // namespace yaml
} // namespace llvm
