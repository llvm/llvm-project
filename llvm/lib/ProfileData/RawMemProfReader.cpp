//===- RawMemProfReader.cpp - Instrumented memory profiling reader --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading MemProf profiling data.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <type_traits>

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/RawMemProfReader.h"

namespace llvm {
namespace memprof {
namespace {

struct Summary {
  uint64_t Version;
  uint64_t TotalSizeBytes;
  uint64_t NumSegments;
  uint64_t NumMIBInfo;
  uint64_t NumStackOffsets;
};

template <class T = uint64_t> inline T alignedRead(const char *Ptr) {
  static_assert(std::is_pod<T>::value, "Not a pod type.");
  assert(reinterpret_cast<size_t>(Ptr) % sizeof(T) == 0 && "Unaligned Read");
  return *reinterpret_cast<const T *>(Ptr);
}

Summary computeSummary(const char *Start) {
  auto *H = reinterpret_cast<const Header *>(Start);

  // Check alignment while reading the number of items in each section.
  return Summary{
      H->Version,
      H->TotalSize,
      alignedRead(Start + H->SegmentOffset),
      alignedRead(Start + H->MIBOffset),
      alignedRead(Start + H->StackOffset),
  };
}

} // namespace

Expected<std::unique_ptr<RawMemProfReader>>
RawMemProfReader::create(const Twine &Path) {
  auto BufferOr = MemoryBuffer::getFileOrSTDIN(Path, /*IsText=*/true);
  if (std::error_code EC = BufferOr.getError())
    return errorCodeToError(EC);

  std::unique_ptr<MemoryBuffer> Buffer(BufferOr.get().release());

  if (Buffer->getBufferSize() == 0)
    return make_error<InstrProfError>(instrprof_error::empty_raw_profile);

  if (!RawMemProfReader::hasFormat(*Buffer))
    return make_error<InstrProfError>(instrprof_error::bad_magic);

  if (Buffer->getBufferSize() < sizeof(Header)) {
    return make_error<InstrProfError>(instrprof_error::truncated);
  }

  // The size of the buffer can be > header total size since we allow repeated
  // serialization of memprof profiles to the same file.
  uint64_t TotalSize = 0;
  const char *Next = Buffer->getBufferStart();
  while (Next < Buffer->getBufferEnd()) {
    auto *H = reinterpret_cast<const Header *>(Next);
    if (H->Version != MEMPROF_RAW_VERSION) {
      return make_error<InstrProfError>(instrprof_error::unsupported_version);
    }

    TotalSize += H->TotalSize;
    Next += H->TotalSize;
  }

  if (Buffer->getBufferSize() != TotalSize) {
    return make_error<InstrProfError>(instrprof_error::malformed);
  }

  return std::make_unique<RawMemProfReader>(std::move(Buffer));
}

bool RawMemProfReader::hasFormat(const MemoryBuffer &Buffer) {
  if (Buffer.getBufferSize() < sizeof(uint64_t))
    return false;
  // Aligned read to sanity check that the buffer was allocated with at least 8b
  // alignment.
  const uint64_t Magic = alignedRead(Buffer.getBufferStart());
  return Magic == MEMPROF_RAW_MAGIC_64;
}

void RawMemProfReader::printSummaries(raw_ostream &OS) const {
  int Count = 0;
  const char *Next = DataBuffer->getBufferStart();
  while (Next < DataBuffer->getBufferEnd()) {
    auto Summary = computeSummary(Next);
    OS << "MemProf Profile " << ++Count << "\n";
    OS << "  Version: " << Summary.Version << "\n";
    OS << "  TotalSizeBytes: " << Summary.TotalSizeBytes << "\n";
    OS << "  NumSegments: " << Summary.NumSegments << "\n";
    OS << "  NumMIBInfo: " << Summary.NumMIBInfo << "\n";
    OS << "  NumStackOffsets: " << Summary.NumStackOffsets << "\n";
    // TODO: Print the build ids once we can record them using the
    // sanitizer_procmaps library for linux.

    auto *H = reinterpret_cast<const Header *>(Next);
    Next += H->TotalSize;
  }
}

} // namespace memprof
} // namespace llvm
