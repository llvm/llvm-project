//===- GsymReader.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymReader.h"

#include "llvm/DebugInfo/GSYM/GsymReaderV1.h"
#include "llvm/DebugInfo/GSYM/GsymReaderV2.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

/// Detect the GSYM version from raw bytes.
static Expected<uint16_t> detectVersion(StringRef Data) {
  // Need at least 6 bytes: 4 (magic) + 2 (version).
  if (Data.size() < 6)
    return createStringError(std::errc::invalid_argument,
                             "data too small to be a GSYM file");
  uint32_t Magic;
  memcpy(&Magic, Data.data(), 4);
  if (Magic != GSYM_MAGIC && Magic != llvm::byteswap(GSYM_MAGIC))
    return createStringError(std::errc::invalid_argument,
                             "not a GSYM file (bad magic)");
  uint16_t Version;
  memcpy(&Version, Data.data() + 4, 2);
  if (Magic != GSYM_MAGIC)
    Version = llvm::byteswap(Version);
  return Version;
}

llvm::Expected<std::unique_ptr<GsymReader>>
GsymReader::openFile(StringRef Path) {
  auto BufOrErr = MemoryBuffer::getFileOrSTDIN(Path);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "failed to open '%s'",
                             Path.str().c_str());
  auto VersionOrErr = detectVersion((*BufOrErr)->getBuffer());
  if (!VersionOrErr)
    return VersionOrErr.takeError();
  if (*VersionOrErr == GSYM_VERSION_2) {
    auto R = GsymReaderV2::openFile(Path);
    if (!R)
      return R.takeError();
    return std::make_unique<GsymReaderV2>(std::move(*R));
  }
  auto R = GsymReaderV1::openFile(Path);
  if (!R)
    return R.takeError();
  return std::make_unique<GsymReaderV1>(std::move(*R));
}

llvm::Expected<std::unique_ptr<GsymReader>>
GsymReader::copyBuffer(StringRef Bytes) {
  auto VersionOrErr = detectVersion(Bytes);
  if (!VersionOrErr)
    return VersionOrErr.takeError();
  if (*VersionOrErr == GSYM_VERSION_2) {
    auto R = GsymReaderV2::copyBuffer(Bytes);
    if (!R)
      return R.takeError();
    return std::make_unique<GsymReaderV2>(std::move(*R));
  }
  auto R = GsymReaderV1::copyBuffer(Bytes);
  if (!R)
    return R.takeError();
  return std::make_unique<GsymReaderV1>(std::move(*R));
}
