//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/Support/BLAKE3.h"

using namespace llvm;
using namespace llvm::cas;

template <class HasherT>
Expected<typename BuiltinObjectHasher<HasherT>::HashT>
BuiltinObjectHasher<HasherT>::hashFile(StringRef FilePath) {
  BuiltinObjectHasher H;
  H.updateSize(0); // 0 refs

  sys::fs::file_t FD;
  if (Error E = sys::fs::openNativeFileForRead(FilePath).moveInto(FD))
    return E;

  sys::fs::file_status Status;
  std::error_code EC = sys::fs::status(FD, Status);
  if (EC)
    return createFileError(FilePath, EC);
  // FIXME: Do we need to add a hash of the data size? If we remove that we can
  // avoid needing to read the file size before reading the file contents.
  H.updateSize(Status.getSize());

  size_t ChunkSize = sys::fs::DefaultReadChunkSize;
  SmallVector<char, 0> Buffer;
  Buffer.resize_for_overwrite(ChunkSize);
  for (;;) {
    Expected<size_t> ReadBytes =
        sys::fs::readNativeFile(FD, MutableArrayRef(Buffer.begin(), ChunkSize));
    if (!ReadBytes)
      return ReadBytes.takeError();
    if (*ReadBytes == 0)
      break;
    H.Hasher.update(toStringRef(ArrayRef(Buffer).take_front(*ReadBytes)));
  }

  return H.finish();
}

// Provide the definition for when using the BLAKE3 hasher.
template Expected<BuiltinObjectHasher<BLAKE3>::HashT>
BuiltinObjectHasher<BLAKE3>::hashFile(StringRef FilePath);
