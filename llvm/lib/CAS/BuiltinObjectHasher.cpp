//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/ADT/ScopeExit.h"

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

  llvm::scope_exit CloseFile([&]() { sys::fs::closeFile(FD); });

  ErrorOr<uint64_t> FileSize = sys::fs::file_size(FD);
  if (!FileSize)
    return createFileError(FilePath, FileSize.getError());
  H.updateSize(*FileSize);

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

namespace llvm::cas {
template class LLVM_EXPORT_TEMPLATE BuiltinObjectHasher<BLAKE3>;
}
