//===- CASUtil/Utils.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CASUtil/Utils.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CASObjectFormats/Encoding.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#if __has_include(<sys/xattr.h>) && defined(__APPLE__)
#include <sys/xattr.h>
#endif

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::casobjectformats;
using namespace llvm::casobjectformats::reader;

Expected<CASID> cas::readCASIDBuffer(cas::ObjectStore &CAS,
                                     MemoryBufferRef Buffer) {
  if (identify_magic(Buffer.getBuffer()) != file_magic::cas_id)
    return createStringError(std::errc::invalid_argument,
                             "buffer does not contain a CASID");

  StringRef Remaining =
      Buffer.getBuffer().substr(StringRef(casidObjectMagicPrefix).size());
  uint32_t Size;
  if (auto E = encoding::consumeVBR8(Remaining, Size))
    return std::move(E);

  StringRef CASIDStr = Remaining.substr(0, Size);
  return CAS.parseID(CASIDStr);
}

void cas::writeCASIDBuffer(const CASID &ID, llvm::raw_ostream &OS) {
  OS << casidObjectMagicPrefix;
  SmallString<256> CASIDStr;
  raw_svector_ostream(CASIDStr) << ID;

  // Write out the size of the CASID so that we can read it back properly even
  // if the buffer has additional padding (e.g. after getting added in an
  // archive).
  SmallString<2> SizeBuf;
  encoding::writeVBR8(CASIDStr.size(), SizeBuf);
  OS << SizeBuf << CASIDStr;
}

Error cas::writeCASHashXAttr(const CASID &ID, const llvm::Twine &Path) {
#if __has_include(<sys/xattr.h>) && defined(__APPLE__)
  SmallString<64> Buffer;
  raw_svector_ostream OS(Buffer);
  // Null-terminated hash schema identifier.
  OS << ID.getContext().getHashSchemaIdentifier() << '\0';
  // Hash size, little-endian, 4 bytes.
  support::endian::write(OS, static_cast<uint32_t>(ID.getHash().size()),
                         endianness::little);
  // Hash bytes.
  OS.write((const char *)ID.getHash().data(), ID.getHash().size());

  SmallString<128> PathStorage;
  StringRef PathRef = Path.toNullTerminatedStringRef(PathStorage);

  int RC =
      setxattr(PathRef.begin(), "com.apple.clang.cas_output_hash",
               Buffer.data(), Buffer.size(), /*Position=*/0, /*Options=*/0);

  if (RC)
    return createFileError(Path, make_error<StringError>(errnoAsErrorCode(),
                                                         "failed to setxattr"));
  return Error::success();
#else
  return createStringError(std::errc::not_supported,
                           "Platform does not support setxattr");
#endif
}
