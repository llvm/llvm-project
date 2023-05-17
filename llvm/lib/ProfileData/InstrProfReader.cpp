//===- InstrProfReader.cpp - Instrumented profiling reader ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/ProfileData/SymbolRemappingReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;

// Extracts the variant information from the top 8 bits in the version and
// returns an enum specifying the variants present.
static InstrProfKind getProfileKindFromVersion(uint64_t Version) {
  InstrProfKind ProfileKind = InstrProfKind::Unknown;
  if (Version & VARIANT_MASK_IR_PROF) {
    ProfileKind |= InstrProfKind::IRInstrumentation;
  }
  if (Version & VARIANT_MASK_CSIR_PROF) {
    ProfileKind |= InstrProfKind::ContextSensitive;
  }
  if (Version & VARIANT_MASK_INSTR_ENTRY) {
    ProfileKind |= InstrProfKind::FunctionEntryInstrumentation;
  }
  if (Version & VARIANT_MASK_BYTE_COVERAGE) {
    ProfileKind |= InstrProfKind::SingleByteCoverage;
  }
  if (Version & VARIANT_MASK_FUNCTION_ENTRY_ONLY) {
    ProfileKind |= InstrProfKind::FunctionEntryOnly;
  }
  if (Version & VARIANT_MASK_MEMPROF) {
    ProfileKind |= InstrProfKind::MemProf;
  }
  if (Version & VARIANT_MASK_TEMPORAL_PROF) {
    ProfileKind |= InstrProfKind::TemporalProfile;
  }
  return ProfileKind;
}

static Expected<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(const Twine &Filename, vfs::FileSystem &FS) {
  auto BufferOrErr = Filename.str() == "-" ? MemoryBuffer::getSTDIN()
                                           : FS.getBufferForFile(Filename);
  if (std::error_code EC = BufferOrErr.getError())
    return errorCodeToError(EC);
  return std::move(BufferOrErr.get());
}

static Error initializeReader(InstrProfReader &Reader) {
  return Reader.readHeader();
}

/// Read a list of binary ids from a profile that consist of
/// a. uint64_t binary id length
/// b. uint8_t  binary id data
/// c. uint8_t  padding (if necessary)
/// This function is shared between raw and indexed profiles.
/// Raw profiles are in host-endian format, and indexed profiles are in
/// little-endian format. So, this function takes an argument indicating the
/// associated endian format to read the binary ids correctly.
static Error
readBinaryIdsInternal(const MemoryBuffer &DataBuffer,
                      const uint64_t BinaryIdsSize,
                      const uint8_t *BinaryIdsStart,
                      std::vector<llvm::object::BuildID> &BinaryIds,
                      const llvm::support::endianness Endian) {
  using namespace support;

  if (BinaryIdsSize == 0)
    return Error::success();

  const uint8_t *BI = BinaryIdsStart;
  const uint8_t *BIEnd = BinaryIdsStart + BinaryIdsSize;
  const uint8_t *End =
      reinterpret_cast<const uint8_t *>(DataBuffer.getBufferEnd());

  while (BI < BIEnd) {
    size_t Remaining = BIEnd - BI;
    // There should be enough left to read the binary id length.
    if (Remaining < sizeof(uint64_t))
      return make_error<InstrProfError>(
          instrprof_error::malformed,
          "not enough data to read binary id length");

    uint64_t BILen = 0;
    if (Endian == little)
      BILen = endian::readNext<uint64_t, little, unaligned>(BI);
    else
      BILen = endian::readNext<uint64_t, big, unaligned>(BI);

    if (BILen == 0)
      return make_error<InstrProfError>(instrprof_error::malformed,
                                        "binary id length is 0");

    Remaining = BIEnd - BI;
    // There should be enough left to read the binary id data.
    if (Remaining < alignToPowerOf2(BILen, sizeof(uint64_t)))
      return make_error<InstrProfError>(
          instrprof_error::malformed, "not enough data to read binary id data");

    // Add binary id to the binary ids list.
    BinaryIds.push_back(object::BuildID(BI, BI + BILen));

    // Increment by binary id data length, which aligned to the size of uint64.
    BI += alignToPowerOf2(BILen, sizeof(uint64_t));
    if (BI > End)
      return make_error<InstrProfError>(
          instrprof_error::malformed,
          "binary id section is greater than buffer size");
  }

  return Error::success();
}

static Error printBinaryIdsInternal(raw_ostream &OS,
                                    const MemoryBuffer &DataBuffer,
                                    uint64_t BinaryIdsSize,
                                    const uint8_t *BinaryIdsStart,
                                    llvm::support::endianness Endian) {
  if (BinaryIdsSize == 0)
    return Error::success();

  std::vector<llvm::object::BuildID> BinaryIds;
  if (Error E = readBinaryIdsInternal(DataBuffer, BinaryIdsSize, BinaryIdsStart,
                                      BinaryIds, Endian))
    return E;

  OS << "Binary IDs: \n";
  for (auto BI : BinaryIds) {
    for (uint64_t I = 0; I < BI.size(); I++)
      OS << format("%02x", BI[I]);
    OS << "\n";
  }

  return Error::success();
}

Expected<std::unique_ptr<InstrProfReader>>
InstrProfReader::create(const Twine &Path, vfs::FileSystem &FS,
                        const InstrProfCorrelator *Correlator) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path, FS);
  if (Error E = BufferOrError.takeError())
    return std::move(E);
  return InstrProfReader::create(std::move(BufferOrError.get()), Correlator);
}

Expected<std::unique_ptr<InstrProfReader>>
InstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer,
                        const InstrProfCorrelator *Correlator) {
  if (Buffer->getBufferSize() == 0)
    return make_error<InstrProfError>(instrprof_error::empty_raw_profile);

  std::unique_ptr<InstrProfReader> Result;
  // Create the reader.
  if (IndexedInstrProfReader::hasFormat(*Buffer))
    Result.reset(new IndexedInstrProfReader(std::move(Buffer)));
  else if (RawInstrProfReader64::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader64(std::move(Buffer), Correlator));
  else if (RawInstrProfReader32::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader32(std::move(Buffer), Correlator));
  else if (TextInstrProfReader::hasFormat(*Buffer))
    Result.reset(new TextInstrProfReader(std::move(Buffer)));
  else
    return make_error<InstrProfError>(instrprof_error::unrecognized_format);

  // Initialize the reader and return the result.
  if (Error E = initializeReader(*Result))
    return std::move(E);

  return std::move(Result);
}

Expected<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(const Twine &Path, vfs::FileSystem &FS,
                               const Twine &RemappingPath) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path, FS);
  if (Error E = BufferOrError.takeError())
    return std::move(E);

  // Set up the remapping buffer if requested.
  std::unique_ptr<MemoryBuffer> RemappingBuffer;
  std::string RemappingPathStr = RemappingPath.str();
  if (!RemappingPathStr.empty()) {
    auto RemappingBufferOrError = setupMemoryBuffer(RemappingPathStr, FS);
    if (Error E = RemappingBufferOrError.takeError())
      return std::move(E);
    RemappingBuffer = std::move(RemappingBufferOrError.get());
  }

  return IndexedInstrProfReader::create(std::move(BufferOrError.get()),
                                        std::move(RemappingBuffer));
}

Expected<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer,
                               std::unique_ptr<MemoryBuffer> RemappingBuffer) {
  // Create the reader.
  if (!IndexedInstrProfReader::hasFormat(*Buffer))
    return make_error<InstrProfError>(instrprof_error::bad_magic);
  auto Result = std::make_unique<IndexedInstrProfReader>(
      std::move(Buffer), std::move(RemappingBuffer));

  // Initialize the reader and return the result.
  if (Error E = initializeReader(*Result))
    return std::move(E);

  return std::move(Result);
}

bool TextInstrProfReader::hasFormat(const MemoryBuffer &Buffer) {
  // Verify that this really looks like plain ASCII text by checking a
  // 'reasonable' number of characters (up to profile magic size).
  size_t count = std::min(Buffer.getBufferSize(), sizeof(uint64_t));
  StringRef buffer = Buffer.getBufferStart();
  return count == 0 ||
         std::all_of(buffer.begin(), buffer.begin() + count,
                     [](char c) { return isPrint(c) || isSpace(c); });
}

// Read the profile variant flag from the header: ":FE" means this is a FE
// generated profile. ":IR" means this is an IR level profile. Other strings
// with a leading ':' will be reported an error format.
Error TextInstrProfReader::readHeader() {
  Symtab.reset(new InstrProfSymtab());

  while (Line->startswith(":")) {
    StringRef Str = Line->substr(1);
    if (Str.equals_insensitive("ir"))
      ProfileKind |= InstrProfKind::IRInstrumentation;
    else if (Str.equals_insensitive("fe"))
      ProfileKind |= InstrProfKind::FrontendInstrumentation;
    else if (Str.equals_insensitive("csir")) {
      ProfileKind |= InstrProfKind::IRInstrumentation;
      ProfileKind |= InstrProfKind::ContextSensitive;
    } else if (Str.equals_insensitive("entry_first"))
      ProfileKind |= InstrProfKind::FunctionEntryInstrumentation;
    else if (Str.equals_insensitive("not_entry_first"))
      ProfileKind &= ~InstrProfKind::FunctionEntryInstrumentation;
    else if (Str.equals_insensitive("temporal_prof_traces")) {
      ProfileKind |= InstrProfKind::TemporalProfile;
      if (auto Err = readTemporalProfTraceData())
        return error(std::move(Err));
    } else
      return error(instrprof_error::bad_header);
    ++Line;
  }
  return success();
}

/// Temporal profile trace data is stored in the header immediately after
/// ":temporal_prof_traces". The first integer is the number of traces, the
/// second integer is the stream size, then the following lines are the actual
/// traces which consist of a weight and a comma separated list of function
/// names.
Error TextInstrProfReader::readTemporalProfTraceData() {
  if ((++Line).is_at_end())
    return error(instrprof_error::eof);

  uint32_t NumTraces;
  if (Line->getAsInteger(0, NumTraces))
    return error(instrprof_error::malformed);

  if ((++Line).is_at_end())
    return error(instrprof_error::eof);

  if (Line->getAsInteger(0, TemporalProfTraceStreamSize))
    return error(instrprof_error::malformed);

  for (uint32_t i = 0; i < NumTraces; i++) {
    if ((++Line).is_at_end())
      return error(instrprof_error::eof);

    TemporalProfTraceTy Trace;
    if (Line->getAsInteger(0, Trace.Weight))
      return error(instrprof_error::malformed);

    if ((++Line).is_at_end())
      return error(instrprof_error::eof);

    SmallVector<StringRef> FuncNames;
    Line->split(FuncNames, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (auto &FuncName : FuncNames)
      Trace.FunctionNameRefs.push_back(
          IndexedInstrProf::ComputeHash(FuncName.trim()));
    TemporalProfTraces.push_back(std::move(Trace));
  }
  return success();
}

Error
TextInstrProfReader::readValueProfileData(InstrProfRecord &Record) {

#define CHECK_LINE_END(Line)                                                   \
  if (Line.is_at_end())                                                        \
    return error(instrprof_error::truncated);
#define READ_NUM(Str, Dst)                                                     \
  if ((Str).getAsInteger(10, (Dst)))                                           \
    return error(instrprof_error::malformed);
#define VP_READ_ADVANCE(Val)                                                   \
  CHECK_LINE_END(Line);                                                        \
  uint32_t Val;                                                                \
  READ_NUM((*Line), (Val));                                                    \
  Line++;

  if (Line.is_at_end())
    return success();

  uint32_t NumValueKinds;
  if (Line->getAsInteger(10, NumValueKinds)) {
    // No value profile data
    return success();
  }
  if (NumValueKinds == 0 || NumValueKinds > IPVK_Last + 1)
    return error(instrprof_error::malformed,
                 "number of value kinds is invalid");
  Line++;

  for (uint32_t VK = 0; VK < NumValueKinds; VK++) {
    VP_READ_ADVANCE(ValueKind);
    if (ValueKind > IPVK_Last)
      return error(instrprof_error::malformed, "value kind is invalid");
    ;
    VP_READ_ADVANCE(NumValueSites);
    if (!NumValueSites)
      continue;

    Record.reserveSites(VK, NumValueSites);
    for (uint32_t S = 0; S < NumValueSites; S++) {
      VP_READ_ADVANCE(NumValueData);

      std::vector<InstrProfValueData> CurrentValues;
      for (uint32_t V = 0; V < NumValueData; V++) {
        CHECK_LINE_END(Line);
        std::pair<StringRef, StringRef> VD = Line->rsplit(':');
        uint64_t TakenCount, Value;
        if (ValueKind == IPVK_IndirectCallTarget) {
          if (InstrProfSymtab::isExternalSymbol(VD.first)) {
            Value = 0;
          } else {
            if (Error E = Symtab->addFuncName(VD.first))
              return E;
            Value = IndexedInstrProf::ComputeHash(VD.first);
          }
        } else {
          READ_NUM(VD.first, Value);
        }
        READ_NUM(VD.second, TakenCount);
        CurrentValues.push_back({Value, TakenCount});
        Line++;
      }
      Record.addValueData(ValueKind, S, CurrentValues.data(), NumValueData,
                          nullptr);
    }
  }
  return success();

#undef CHECK_LINE_END
#undef READ_NUM
#undef VP_READ_ADVANCE
}

Error TextInstrProfReader::readNextRecord(NamedInstrProfRecord &Record) {
  // Skip empty lines and comments.
  while (!Line.is_at_end() && (Line->empty() || Line->startswith("#")))
    ++Line;
  // If we hit EOF while looking for a name, we're done.
  if (Line.is_at_end()) {
    return error(instrprof_error::eof);
  }

  // Read the function name.
  Record.Name = *Line++;
  if (Error E = Symtab->addFuncName(Record.Name))
    return error(std::move(E));

  // Read the function hash.
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(0, Record.Hash))
    return error(instrprof_error::malformed,
                 "function hash is not a valid integer");

  // Read the number of counters.
  uint64_t NumCounters;
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(10, NumCounters))
    return error(instrprof_error::malformed,
                 "number of counters is not a valid integer");
  if (NumCounters == 0)
    return error(instrprof_error::malformed, "number of counters is zero");

  // Read each counter and fill our internal storage with the values.
  Record.Clear();
  Record.Counts.reserve(NumCounters);
  for (uint64_t I = 0; I < NumCounters; ++I) {
    if (Line.is_at_end())
      return error(instrprof_error::truncated);
    uint64_t Count;
    if ((Line++)->getAsInteger(10, Count))
      return error(instrprof_error::malformed, "count is invalid");
    Record.Counts.push_back(Count);
  }

  // Check if value profile data exists and read it if so.
  if (Error E = readValueProfileData(Record))
    return error(std::move(E));

  return success();
}

template <class IntPtrT>
InstrProfKind RawInstrProfReader<IntPtrT>::getProfileKind() const {
  return getProfileKindFromVersion(Version);
}

template <class IntPtrT>
SmallVector<TemporalProfTraceTy> &
RawInstrProfReader<IntPtrT>::getTemporalProfTraces(
    std::optional<uint64_t> Weight) {
  if (TemporalProfTimestamps.empty()) {
    assert(TemporalProfTraces.empty());
    return TemporalProfTraces;
  }
  // Sort functions by their timestamps to build the trace.
  std::sort(TemporalProfTimestamps.begin(), TemporalProfTimestamps.end());
  TemporalProfTraceTy Trace;
  if (Weight)
    Trace.Weight = *Weight;
  for (auto &[TimestampValue, NameRef] : TemporalProfTimestamps)
    Trace.FunctionNameRefs.push_back(NameRef);
  TemporalProfTraces = {std::move(Trace)};
  return TemporalProfTraces;
}

template <class IntPtrT>
bool RawInstrProfReader<IntPtrT>::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < sizeof(uint64_t))
    return false;
  uint64_t Magic =
    *reinterpret_cast<const uint64_t *>(DataBuffer.getBufferStart());
  return RawInstrProf::getMagic<IntPtrT>() == Magic ||
         sys::getSwappedBytes(RawInstrProf::getMagic<IntPtrT>()) == Magic;
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readHeader() {
  if (!hasFormat(*DataBuffer))
    return error(instrprof_error::bad_magic);
  if (DataBuffer->getBufferSize() < sizeof(RawInstrProf::Header))
    return error(instrprof_error::bad_header);
  auto *Header = reinterpret_cast<const RawInstrProf::Header *>(
      DataBuffer->getBufferStart());
  ShouldSwapBytes = Header->Magic != RawInstrProf::getMagic<IntPtrT>();
  return readHeader(*Header);
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readNextHeader(const char *CurrentPos) {
  const char *End = DataBuffer->getBufferEnd();
  // Skip zero padding between profiles.
  while (CurrentPos != End && *CurrentPos == 0)
    ++CurrentPos;
  // If there's nothing left, we're done.
  if (CurrentPos == End)
    return make_error<InstrProfError>(instrprof_error::eof);
  // If there isn't enough space for another header, this is probably just
  // garbage at the end of the file.
  if (CurrentPos + sizeof(RawInstrProf::Header) > End)
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "not enough space for another header");
  // The writer ensures each profile is padded to start at an aligned address.
  if (reinterpret_cast<size_t>(CurrentPos) % alignof(uint64_t))
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "insufficient padding");
  // The magic should have the same byte order as in the previous header.
  uint64_t Magic = *reinterpret_cast<const uint64_t *>(CurrentPos);
  if (Magic != swap(RawInstrProf::getMagic<IntPtrT>()))
    return make_error<InstrProfError>(instrprof_error::bad_magic);

  // There's another profile to read, so we need to process the header.
  auto *Header = reinterpret_cast<const RawInstrProf::Header *>(CurrentPos);
  return readHeader(*Header);
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::createSymtab(InstrProfSymtab &Symtab) {
  if (Error E = Symtab.create(StringRef(NamesStart, NamesEnd - NamesStart)))
    return error(std::move(E));
  for (const RawInstrProf::ProfileData<IntPtrT> *I = Data; I != DataEnd; ++I) {
    const IntPtrT FPtr = swap(I->FunctionPointer);
    if (!FPtr)
      continue;
    Symtab.mapAddress(FPtr, I->NameRef);
  }
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readHeader(
    const RawInstrProf::Header &Header) {
  Version = swap(Header.Version);
  if (GET_VERSION(Version) != RawInstrProf::Version)
    return error(instrprof_error::raw_profile_version_mismatch,
                 ("Profile uses raw profile format version = " +
                  Twine(GET_VERSION(Version)) +
                  "; expected version = " + Twine(RawInstrProf::Version) +
                  "\nPLEASE update this tool to version in the raw profile, or "
                  "regenerate raw profile with expected version.")
                     .str());
  if (useDebugInfoCorrelate() && !Correlator)
    return error(instrprof_error::missing_debug_info_for_correlation);
  if (!useDebugInfoCorrelate() && Correlator)
    return error(instrprof_error::unexpected_debug_info_for_correlation);

  BinaryIdsSize = swap(Header.BinaryIdsSize);
  if (BinaryIdsSize % sizeof(uint64_t))
    return error(instrprof_error::bad_header);

  CountersDelta = swap(Header.CountersDelta);
  NamesDelta = swap(Header.NamesDelta);
  auto NumData = swap(Header.DataSize);
  auto PaddingBytesBeforeCounters = swap(Header.PaddingBytesBeforeCounters);
  auto CountersSize = swap(Header.CountersSize) * getCounterTypeSize();
  auto PaddingBytesAfterCounters = swap(Header.PaddingBytesAfterCounters);
  auto NamesSize = swap(Header.NamesSize);
  ValueKindLast = swap(Header.ValueKindLast);

  auto DataSize = NumData * sizeof(RawInstrProf::ProfileData<IntPtrT>);
  auto PaddingSize = getNumPaddingBytes(NamesSize);

  // Profile data starts after profile header and binary ids if exist.
  ptrdiff_t DataOffset = sizeof(RawInstrProf::Header) + BinaryIdsSize;
  ptrdiff_t CountersOffset = DataOffset + DataSize + PaddingBytesBeforeCounters;
  ptrdiff_t NamesOffset =
      CountersOffset + CountersSize + PaddingBytesAfterCounters;
  ptrdiff_t ValueDataOffset = NamesOffset + NamesSize + PaddingSize;

  auto *Start = reinterpret_cast<const char *>(&Header);
  if (Start + ValueDataOffset > DataBuffer->getBufferEnd())
    return error(instrprof_error::bad_header);

  if (Correlator) {
    // These sizes in the raw file are zero because we constructed them in the
    // Correlator.
    assert(DataSize == 0 && NamesSize == 0);
    assert(CountersDelta == 0 && NamesDelta == 0);
    Data = Correlator->getDataPointer();
    DataEnd = Data + Correlator->getDataSize();
    NamesStart = Correlator->getNamesPointer();
    NamesEnd = NamesStart + Correlator->getNamesSize();
  } else {
    Data = reinterpret_cast<const RawInstrProf::ProfileData<IntPtrT> *>(
        Start + DataOffset);
    DataEnd = Data + NumData;
    NamesStart = Start + NamesOffset;
    NamesEnd = NamesStart + NamesSize;
  }

  // Binary ids start just after the header.
  BinaryIdsStart =
      reinterpret_cast<const uint8_t *>(&Header) + sizeof(RawInstrProf::Header);
  CountersStart = Start + CountersOffset;
  CountersEnd = CountersStart + CountersSize;
  ValueDataStart = reinterpret_cast<const uint8_t *>(Start + ValueDataOffset);

  const uint8_t *BufferEnd = (const uint8_t *)DataBuffer->getBufferEnd();
  if (BinaryIdsStart + BinaryIdsSize > BufferEnd)
    return error(instrprof_error::bad_header);

  std::unique_ptr<InstrProfSymtab> NewSymtab = std::make_unique<InstrProfSymtab>();
  if (Error E = createSymtab(*NewSymtab))
    return E;

  Symtab = std::move(NewSymtab);
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readName(NamedInstrProfRecord &Record) {
  Record.Name = getName(Data->NameRef);
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readFuncHash(NamedInstrProfRecord &Record) {
  Record.Hash = swap(Data->FuncHash);
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readRawCounts(
    InstrProfRecord &Record) {
  uint32_t NumCounters = swap(Data->NumCounters);
  if (NumCounters == 0)
    return error(instrprof_error::malformed, "number of counters is zero");

  ptrdiff_t CounterBaseOffset = swap(Data->CounterPtr) - CountersDelta;
  if (CounterBaseOffset < 0)
    return error(
        instrprof_error::malformed,
        ("counter offset " + Twine(CounterBaseOffset) + " is negative").str());

  if (CounterBaseOffset >= CountersEnd - CountersStart)
    return error(instrprof_error::malformed,
                 ("counter offset " + Twine(CounterBaseOffset) +
                  " is greater than the maximum counter offset " +
                  Twine(CountersEnd - CountersStart - 1))
                     .str());

  uint64_t MaxNumCounters =
      (CountersEnd - (CountersStart + CounterBaseOffset)) /
      getCounterTypeSize();
  if (NumCounters > MaxNumCounters)
    return error(instrprof_error::malformed,
                 ("number of counters " + Twine(NumCounters) +
                  " is greater than the maximum number of counters " +
                  Twine(MaxNumCounters))
                     .str());

  Record.Counts.clear();
  Record.Counts.reserve(NumCounters);
  for (uint32_t I = 0; I < NumCounters; I++) {
    const char *Ptr =
        CountersStart + CounterBaseOffset + I * getCounterTypeSize();
    if (I == 0 && hasTemporalProfile()) {
      uint64_t TimestampValue = swap(*reinterpret_cast<const uint64_t *>(Ptr));
      if (TimestampValue != 0 &&
          TimestampValue != std::numeric_limits<uint64_t>::max()) {
        TemporalProfTimestamps.emplace_back(TimestampValue,
                                            swap(Data->NameRef));
        TemporalProfTraceStreamSize = 1;
      }
      if (hasSingleByteCoverage()) {
        // In coverage mode, getCounterTypeSize() returns 1 byte but our
        // timestamp field has size uint64_t. Increment I so that the next
        // iteration of this for loop points to the byte after the timestamp
        // field, i.e., I += 8.
        I += 7;
      }
      continue;
    }
    if (hasSingleByteCoverage()) {
      // A value of zero signifies the block is covered.
      Record.Counts.push_back(*Ptr == 0 ? 1 : 0);
    } else {
      const auto *CounterValue = reinterpret_cast<const uint64_t *>(Ptr);
      Record.Counts.push_back(swap(*CounterValue));
    }
  }

  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readValueProfilingData(
    InstrProfRecord &Record) {
  Record.clearValueData();
  CurValueDataSize = 0;
  // Need to match the logic in value profile dumper code in compiler-rt:
  uint32_t NumValueKinds = 0;
  for (uint32_t I = 0; I < IPVK_Last + 1; I++)
    NumValueKinds += (Data->NumValueSites[I] != 0);

  if (!NumValueKinds)
    return success();

  Expected<std::unique_ptr<ValueProfData>> VDataPtrOrErr =
      ValueProfData::getValueProfData(
          ValueDataStart, (const unsigned char *)DataBuffer->getBufferEnd(),
          getDataEndianness());

  if (Error E = VDataPtrOrErr.takeError())
    return E;

  // Note that besides deserialization, this also performs the conversion for
  // indirect call targets.  The function pointers from the raw profile are
  // remapped into function name hashes.
  VDataPtrOrErr.get()->deserializeTo(Record, Symtab.get());
  CurValueDataSize = VDataPtrOrErr.get()->getSize();
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readNextRecord(NamedInstrProfRecord &Record) {
  // Keep reading profiles that consist of only headers and no profile data and
  // counters.
  while (atEnd())
    // At this point, ValueDataStart field points to the next header.
    if (Error E = readNextHeader(getNextHeaderPos()))
      return error(std::move(E));

  // Read name and set it in Record.
  if (Error E = readName(Record))
    return error(std::move(E));

  // Read FuncHash and set it in Record.
  if (Error E = readFuncHash(Record))
    return error(std::move(E));

  // Read raw counts and set Record.
  if (Error E = readRawCounts(Record))
    return error(std::move(E));

  // Read value data and set Record.
  if (Error E = readValueProfilingData(Record))
    return error(std::move(E));

  // Iterate.
  advanceData();
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readBinaryIds(
    std::vector<llvm::object::BuildID> &BinaryIds) {
  return readBinaryIdsInternal(*DataBuffer, BinaryIdsSize, BinaryIdsStart,
                               BinaryIds, getDataEndianness());
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::printBinaryIds(raw_ostream &OS) {
  return printBinaryIdsInternal(OS, *DataBuffer, BinaryIdsSize, BinaryIdsStart,
                                getDataEndianness());
}

namespace llvm {

template class RawInstrProfReader<uint32_t>;
template class RawInstrProfReader<uint64_t>;

} // end namespace llvm

InstrProfLookupTrait::hash_value_type
InstrProfLookupTrait::ComputeHash(StringRef K) {
  return IndexedInstrProf::ComputeHash(HashType, K);
}

using data_type = InstrProfLookupTrait::data_type;
using offset_type = InstrProfLookupTrait::offset_type;

bool InstrProfLookupTrait::readValueProfilingData(
    const unsigned char *&D, const unsigned char *const End) {
  Expected<std::unique_ptr<ValueProfData>> VDataPtrOrErr =
      ValueProfData::getValueProfData(D, End, ValueProfDataEndianness);

  if (VDataPtrOrErr.takeError())
    return false;

  VDataPtrOrErr.get()->deserializeTo(DataBuffer.back(), nullptr);
  D += VDataPtrOrErr.get()->TotalSize;

  return true;
}

data_type InstrProfLookupTrait::ReadData(StringRef K, const unsigned char *D,
                                         offset_type N) {
  using namespace support;

  // Check if the data is corrupt. If so, don't try to read it.
  if (N % sizeof(uint64_t))
    return data_type();

  DataBuffer.clear();
  std::vector<uint64_t> CounterBuffer;

  const unsigned char *End = D + N;
  while (D < End) {
    // Read hash.
    if (D + sizeof(uint64_t) >= End)
      return data_type();
    uint64_t Hash = endian::readNext<uint64_t, little, unaligned>(D);

    // Initialize number of counters for GET_VERSION(FormatVersion) == 1.
    uint64_t CountsSize = N / sizeof(uint64_t) - 1;
    // If format version is different then read the number of counters.
    if (GET_VERSION(FormatVersion) != IndexedInstrProf::ProfVersion::Version1) {
      if (D + sizeof(uint64_t) > End)
        return data_type();
      CountsSize = endian::readNext<uint64_t, little, unaligned>(D);
    }
    // Read counter values.
    if (D + CountsSize * sizeof(uint64_t) > End)
      return data_type();

    CounterBuffer.clear();
    CounterBuffer.reserve(CountsSize);
    for (uint64_t J = 0; J < CountsSize; ++J)
      CounterBuffer.push_back(endian::readNext<uint64_t, little, unaligned>(D));

    DataBuffer.emplace_back(K, Hash, std::move(CounterBuffer));

    // Read value profiling data.
    if (GET_VERSION(FormatVersion) > IndexedInstrProf::ProfVersion::Version2 &&
        !readValueProfilingData(D, End)) {
      DataBuffer.clear();
      return data_type();
    }
  }
  return DataBuffer;
}

template <typename HashTableImpl>
Error InstrProfReaderIndex<HashTableImpl>::getRecords(
    StringRef FuncName, ArrayRef<NamedInstrProfRecord> &Data) {
  auto Iter = HashTable->find(FuncName);
  if (Iter == HashTable->end())
    return make_error<InstrProfError>(instrprof_error::unknown_function);

  Data = (*Iter);
  if (Data.empty())
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "profile data is empty");

  return Error::success();
}

template <typename HashTableImpl>
Error InstrProfReaderIndex<HashTableImpl>::getRecords(
    ArrayRef<NamedInstrProfRecord> &Data) {
  if (atEnd())
    return make_error<InstrProfError>(instrprof_error::eof);

  Data = *RecordIterator;

  if (Data.empty())
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "profile data is empty");

  return Error::success();
}

template <typename HashTableImpl>
InstrProfReaderIndex<HashTableImpl>::InstrProfReaderIndex(
    const unsigned char *Buckets, const unsigned char *const Payload,
    const unsigned char *const Base, IndexedInstrProf::HashT HashType,
    uint64_t Version) {
  FormatVersion = Version;
  HashTable.reset(HashTableImpl::Create(
      Buckets, Payload, Base,
      typename HashTableImpl::InfoType(HashType, Version)));
  RecordIterator = HashTable->data_begin();
}

template <typename HashTableImpl>
InstrProfKind InstrProfReaderIndex<HashTableImpl>::getProfileKind() const {
  return getProfileKindFromVersion(FormatVersion);
}

namespace {
/// A remapper that does not apply any remappings.
class InstrProfReaderNullRemapper : public InstrProfReaderRemapper {
  InstrProfReaderIndexBase &Underlying;

public:
  InstrProfReaderNullRemapper(InstrProfReaderIndexBase &Underlying)
      : Underlying(Underlying) {}

  Error getRecords(StringRef FuncName,
                   ArrayRef<NamedInstrProfRecord> &Data) override {
    return Underlying.getRecords(FuncName, Data);
  }
};
} // namespace

/// A remapper that applies remappings based on a symbol remapping file.
template <typename HashTableImpl>
class llvm::InstrProfReaderItaniumRemapper
    : public InstrProfReaderRemapper {
public:
  InstrProfReaderItaniumRemapper(
      std::unique_ptr<MemoryBuffer> RemapBuffer,
      InstrProfReaderIndex<HashTableImpl> &Underlying)
      : RemapBuffer(std::move(RemapBuffer)), Underlying(Underlying) {
  }

  /// Extract the original function name from a PGO function name.
  static StringRef extractName(StringRef Name) {
    // We can have multiple :-separated pieces; there can be pieces both
    // before and after the mangled name. Find the first part that starts
    // with '_Z'; we'll assume that's the mangled name we want.
    std::pair<StringRef, StringRef> Parts = {StringRef(), Name};
    while (true) {
      Parts = Parts.second.split(':');
      if (Parts.first.startswith("_Z"))
        return Parts.first;
      if (Parts.second.empty())
        return Name;
    }
  }

  /// Given a mangled name extracted from a PGO function name, and a new
  /// form for that mangled name, reconstitute the name.
  static void reconstituteName(StringRef OrigName, StringRef ExtractedName,
                               StringRef Replacement,
                               SmallVectorImpl<char> &Out) {
    Out.reserve(OrigName.size() + Replacement.size() - ExtractedName.size());
    Out.insert(Out.end(), OrigName.begin(), ExtractedName.begin());
    Out.insert(Out.end(), Replacement.begin(), Replacement.end());
    Out.insert(Out.end(), ExtractedName.end(), OrigName.end());
  }

  Error populateRemappings() override {
    if (Error E = Remappings.read(*RemapBuffer))
      return E;
    for (StringRef Name : Underlying.HashTable->keys()) {
      StringRef RealName = extractName(Name);
      if (auto Key = Remappings.insert(RealName)) {
        // FIXME: We could theoretically map the same equivalence class to
        // multiple names in the profile data. If that happens, we should
        // return NamedInstrProfRecords from all of them.
        MappedNames.insert({Key, RealName});
      }
    }
    return Error::success();
  }

  Error getRecords(StringRef FuncName,
                   ArrayRef<NamedInstrProfRecord> &Data) override {
    StringRef RealName = extractName(FuncName);
    if (auto Key = Remappings.lookup(RealName)) {
      StringRef Remapped = MappedNames.lookup(Key);
      if (!Remapped.empty()) {
        if (RealName.begin() == FuncName.begin() &&
            RealName.end() == FuncName.end())
          FuncName = Remapped;
        else {
          // Try rebuilding the name from the given remapping.
          SmallString<256> Reconstituted;
          reconstituteName(FuncName, RealName, Remapped, Reconstituted);
          Error E = Underlying.getRecords(Reconstituted, Data);
          if (!E)
            return E;

          // If we failed because the name doesn't exist, fall back to asking
          // about the original name.
          if (Error Unhandled = handleErrors(
                  std::move(E), [](std::unique_ptr<InstrProfError> Err) {
                    return Err->get() == instrprof_error::unknown_function
                               ? Error::success()
                               : Error(std::move(Err));
                  }))
            return Unhandled;
        }
      }
    }
    return Underlying.getRecords(FuncName, Data);
  }

private:
  /// The memory buffer containing the remapping configuration. Remappings
  /// holds pointers into this buffer.
  std::unique_ptr<MemoryBuffer> RemapBuffer;

  /// The mangling remapper.
  SymbolRemappingReader Remappings;

  /// Mapping from mangled name keys to the name used for the key in the
  /// profile data.
  /// FIXME: Can we store a location within the on-disk hash table instead of
  /// redoing lookup?
  DenseMap<SymbolRemappingReader::Key, StringRef> MappedNames;

  /// The real profile data reader.
  InstrProfReaderIndex<HashTableImpl> &Underlying;
};

bool IndexedInstrProfReader::hasFormat(const MemoryBuffer &DataBuffer) {
  using namespace support;

  if (DataBuffer.getBufferSize() < 8)
    return false;
  uint64_t Magic =
      endian::read<uint64_t, little, aligned>(DataBuffer.getBufferStart());
  // Verify that it's magical.
  return Magic == IndexedInstrProf::Magic;
}

const unsigned char *
IndexedInstrProfReader::readSummary(IndexedInstrProf::ProfVersion Version,
                                    const unsigned char *Cur, bool UseCS) {
  using namespace IndexedInstrProf;
  using namespace support;

  if (Version >= IndexedInstrProf::Version4) {
    const IndexedInstrProf::Summary *SummaryInLE =
        reinterpret_cast<const IndexedInstrProf::Summary *>(Cur);
    uint64_t NFields =
        endian::byte_swap<uint64_t, little>(SummaryInLE->NumSummaryFields);
    uint64_t NEntries =
        endian::byte_swap<uint64_t, little>(SummaryInLE->NumCutoffEntries);
    uint32_t SummarySize =
        IndexedInstrProf::Summary::getSize(NFields, NEntries);
    std::unique_ptr<IndexedInstrProf::Summary> SummaryData =
        IndexedInstrProf::allocSummary(SummarySize);

    const uint64_t *Src = reinterpret_cast<const uint64_t *>(SummaryInLE);
    uint64_t *Dst = reinterpret_cast<uint64_t *>(SummaryData.get());
    for (unsigned I = 0; I < SummarySize / sizeof(uint64_t); I++)
      Dst[I] = endian::byte_swap<uint64_t, little>(Src[I]);

    SummaryEntryVector DetailedSummary;
    for (unsigned I = 0; I < SummaryData->NumCutoffEntries; I++) {
      const IndexedInstrProf::Summary::Entry &Ent = SummaryData->getEntry(I);
      DetailedSummary.emplace_back((uint32_t)Ent.Cutoff, Ent.MinBlockCount,
                                   Ent.NumBlocks);
    }
    std::unique_ptr<llvm::ProfileSummary> &Summary =
        UseCS ? this->CS_Summary : this->Summary;

    // initialize InstrProfSummary using the SummaryData from disk.
    Summary = std::make_unique<ProfileSummary>(
        UseCS ? ProfileSummary::PSK_CSInstr : ProfileSummary::PSK_Instr,
        DetailedSummary, SummaryData->get(Summary::TotalBlockCount),
        SummaryData->get(Summary::MaxBlockCount),
        SummaryData->get(Summary::MaxInternalBlockCount),
        SummaryData->get(Summary::MaxFunctionCount),
        SummaryData->get(Summary::TotalNumBlocks),
        SummaryData->get(Summary::TotalNumFunctions));
    return Cur + SummarySize;
  } else {
    // The older versions do not support a profile summary. This just computes
    // an empty summary, which will not result in accurate hot/cold detection.
    // We would need to call addRecord for all NamedInstrProfRecords to get the
    // correct summary. However, this version is old (prior to early 2016) and
    // has not been supporting an accurate summary for several years.
    InstrProfSummaryBuilder Builder(ProfileSummaryBuilder::DefaultCutoffs);
    Summary = Builder.getSummary();
    return Cur;
  }
}

Error IndexedInstrProfReader::readHeader() {
  using namespace support;

  const unsigned char *Start =
      (const unsigned char *)DataBuffer->getBufferStart();
  const unsigned char *Cur = Start;
  if ((const unsigned char *)DataBuffer->getBufferEnd() - Cur < 24)
    return error(instrprof_error::truncated);

  auto HeaderOr = IndexedInstrProf::Header::readFromBuffer(Start);
  if (!HeaderOr)
    return HeaderOr.takeError();

  const IndexedInstrProf::Header *Header = &HeaderOr.get();
  Cur += Header->size();

  Cur = readSummary((IndexedInstrProf::ProfVersion)Header->formatVersion(), Cur,
                    /* UseCS */ false);
  if (Header->formatVersion() & VARIANT_MASK_CSIR_PROF)
    Cur =
        readSummary((IndexedInstrProf::ProfVersion)Header->formatVersion(), Cur,
                    /* UseCS */ true);
  // Read the hash type and start offset.
  IndexedInstrProf::HashT HashType = static_cast<IndexedInstrProf::HashT>(
      endian::byte_swap<uint64_t, little>(Header->HashType));
  if (HashType > IndexedInstrProf::HashT::Last)
    return error(instrprof_error::unsupported_hash_type);

  uint64_t HashOffset = endian::byte_swap<uint64_t, little>(Header->HashOffset);

  // The hash table with profile counts comes next.
  auto IndexPtr = std::make_unique<InstrProfReaderIndex<OnDiskHashTableImplV3>>(
      Start + HashOffset, Cur, Start, HashType, Header->formatVersion());

  // The MemProfOffset field in the header is only valid when the format
  // version is higher than 8 (when it was introduced).
  if (GET_VERSION(Header->formatVersion()) >= 8 &&
      Header->formatVersion() & VARIANT_MASK_MEMPROF) {
    uint64_t MemProfOffset =
        endian::byte_swap<uint64_t, little>(Header->MemProfOffset);

    const unsigned char *Ptr = Start + MemProfOffset;
    // The value returned from RecordTableGenerator.Emit.
    const uint64_t RecordTableOffset =
        support::endian::readNext<uint64_t, little, unaligned>(Ptr);
    // The offset in the stream right before invoking
    // FrameTableGenerator.Emit.
    const uint64_t FramePayloadOffset =
        support::endian::readNext<uint64_t, little, unaligned>(Ptr);
    // The value returned from FrameTableGenerator.Emit.
    const uint64_t FrameTableOffset =
        support::endian::readNext<uint64_t, little, unaligned>(Ptr);

    // Read the schema.
    auto SchemaOr = memprof::readMemProfSchema(Ptr);
    if (!SchemaOr)
      return SchemaOr.takeError();
    Schema = SchemaOr.get();

    // Now initialize the table reader with a pointer into data buffer.
    MemProfRecordTable.reset(MemProfRecordHashTable::Create(
        /*Buckets=*/Start + RecordTableOffset,
        /*Payload=*/Ptr,
        /*Base=*/Start, memprof::RecordLookupTrait(Schema)));

    // Initialize the frame table reader with the payload and bucket offsets.
    MemProfFrameTable.reset(MemProfFrameHashTable::Create(
        /*Buckets=*/Start + FrameTableOffset,
        /*Payload=*/Start + FramePayloadOffset,
        /*Base=*/Start, memprof::FrameLookupTrait()));
  }

  // BinaryIdOffset field in the header is only valid when the format version
  // is higher than 9 (when it was introduced).
  if (GET_VERSION(Header->formatVersion()) >= 9) {
    uint64_t BinaryIdOffset =
        endian::byte_swap<uint64_t, little>(Header->BinaryIdOffset);
    const unsigned char *Ptr = Start + BinaryIdOffset;
    // Read binary ids size.
    BinaryIdsSize = support::endian::readNext<uint64_t, little, unaligned>(Ptr);
    if (BinaryIdsSize % sizeof(uint64_t))
      return error(instrprof_error::bad_header);
    // Set the binary ids start.
    BinaryIdsStart = Ptr;
    if (BinaryIdsStart > (const unsigned char *)DataBuffer->getBufferEnd())
      return make_error<InstrProfError>(instrprof_error::malformed,
                                        "corrupted binary ids");
  }

  if (GET_VERSION(Header->formatVersion()) >= 10 &&
      Header->formatVersion() & VARIANT_MASK_TEMPORAL_PROF) {
    uint64_t TemporalProfTracesOffset =
        endian::byte_swap<uint64_t, little>(Header->TemporalProfTracesOffset);
    const unsigned char *Ptr = Start + TemporalProfTracesOffset;
    const auto *PtrEnd = (const unsigned char *)DataBuffer->getBufferEnd();
    // Expect at least two 64 bit fields: NumTraces, and TraceStreamSize
    if (Ptr + 2 * sizeof(uint64_t) > PtrEnd)
      return error(instrprof_error::truncated);
    const uint64_t NumTraces =
        support::endian::readNext<uint64_t, little, unaligned>(Ptr);
    TemporalProfTraceStreamSize =
        support::endian::readNext<uint64_t, little, unaligned>(Ptr);
    for (unsigned i = 0; i < NumTraces; i++) {
      // Expect at least two 64 bit fields: Weight and NumFunctions
      if (Ptr + 2 * sizeof(uint64_t) > PtrEnd)
        return error(instrprof_error::truncated);
      TemporalProfTraceTy Trace;
      Trace.Weight =
          support::endian::readNext<uint64_t, little, unaligned>(Ptr);
      const uint64_t NumFunctions =
          support::endian::readNext<uint64_t, little, unaligned>(Ptr);
      // Expect at least NumFunctions 64 bit fields
      if (Ptr + NumFunctions * sizeof(uint64_t) > PtrEnd)
        return error(instrprof_error::truncated);
      for (unsigned j = 0; j < NumFunctions; j++) {
        const uint64_t NameRef =
            support::endian::readNext<uint64_t, little, unaligned>(Ptr);
        Trace.FunctionNameRefs.push_back(NameRef);
      }
      TemporalProfTraces.push_back(std::move(Trace));
    }
  }

  // Load the remapping table now if requested.
  if (RemappingBuffer) {
    Remapper =
        std::make_unique<InstrProfReaderItaniumRemapper<OnDiskHashTableImplV3>>(
            std::move(RemappingBuffer), *IndexPtr);
    if (Error E = Remapper->populateRemappings())
      return E;
  } else {
    Remapper = std::make_unique<InstrProfReaderNullRemapper>(*IndexPtr);
  }
  Index = std::move(IndexPtr);

  return success();
}

InstrProfSymtab &IndexedInstrProfReader::getSymtab() {
  if (Symtab)
    return *Symtab;

  std::unique_ptr<InstrProfSymtab> NewSymtab = std::make_unique<InstrProfSymtab>();
  if (Error E = Index->populateSymtab(*NewSymtab)) {
    auto [ErrCode, Msg] = InstrProfError::take(std::move(E));
    consumeError(error(ErrCode, Msg));
  }

  Symtab = std::move(NewSymtab);
  return *Symtab;
}

Expected<InstrProfRecord> IndexedInstrProfReader::getInstrProfRecord(
    StringRef FuncName, uint64_t FuncHash, uint64_t *MismatchedFuncSum) {
  ArrayRef<NamedInstrProfRecord> Data;
  uint64_t FuncSum = 0;
  Error Err = Remapper->getRecords(FuncName, Data);
  if (Err)
    return std::move(Err);
  // Found it. Look for counters with the right hash.

  // A flag to indicate if the records are from the same type
  // of profile (i.e cs vs nocs).
  bool CSBitMatch = false;
  auto getFuncSum = [](const std::vector<uint64_t> &Counts) {
    uint64_t ValueSum = 0;
    for (uint64_t CountValue : Counts) {
      if (CountValue == (uint64_t)-1)
        continue;
      // Handle overflow -- if that happens, return max.
      if (std::numeric_limits<uint64_t>::max() - CountValue <= ValueSum)
        return std::numeric_limits<uint64_t>::max();
      ValueSum += CountValue;
    }
    return ValueSum;
  };

  for (const NamedInstrProfRecord &I : Data) {
    // Check for a match and fill the vector if there is one.
    if (I.Hash == FuncHash)
      return std::move(I);
    if (NamedInstrProfRecord::hasCSFlagInHash(I.Hash) ==
        NamedInstrProfRecord::hasCSFlagInHash(FuncHash)) {
      CSBitMatch = true;
      if (MismatchedFuncSum == nullptr)
        continue;
      FuncSum = std::max(FuncSum, getFuncSum(I.Counts));
    }
  }
  if (CSBitMatch) {
    if (MismatchedFuncSum != nullptr)
      *MismatchedFuncSum = FuncSum;
    return error(instrprof_error::hash_mismatch);
  }
  return error(instrprof_error::unknown_function);
}

Expected<memprof::MemProfRecord>
IndexedInstrProfReader::getMemProfRecord(const uint64_t FuncNameHash) {
  // TODO: Add memprof specific errors.
  if (MemProfRecordTable == nullptr)
    return make_error<InstrProfError>(instrprof_error::invalid_prof,
                                      "no memprof data available in profile");
  auto Iter = MemProfRecordTable->find(FuncNameHash);
  if (Iter == MemProfRecordTable->end())
    return make_error<InstrProfError>(
        instrprof_error::unknown_function,
        "memprof record not found for function hash " + Twine(FuncNameHash));

  // Setup a callback to convert from frame ids to frame using the on-disk
  // FrameData hash table.
  memprof::FrameId LastUnmappedFrameId = 0;
  bool HasFrameMappingError = false;
  auto IdToFrameCallback = [&](const memprof::FrameId Id) {
    auto FrIter = MemProfFrameTable->find(Id);
    if (FrIter == MemProfFrameTable->end()) {
      LastUnmappedFrameId = Id;
      HasFrameMappingError = true;
      return memprof::Frame(0, 0, 0, false);
    }
    return *FrIter;
  };

  memprof::MemProfRecord Record(*Iter, IdToFrameCallback);

  // Check that all frame ids were successfully converted to frames.
  if (HasFrameMappingError) {
    return make_error<InstrProfError>(instrprof_error::hash_mismatch,
                                      "memprof frame not found for frame id " +
                                          Twine(LastUnmappedFrameId));
  }
  return Record;
}

Error IndexedInstrProfReader::getFunctionCounts(StringRef FuncName,
                                                uint64_t FuncHash,
                                                std::vector<uint64_t> &Counts) {
  Expected<InstrProfRecord> Record = getInstrProfRecord(FuncName, FuncHash);
  if (Error E = Record.takeError())
    return error(std::move(E));

  Counts = Record.get().Counts;
  return success();
}

Error IndexedInstrProfReader::readNextRecord(NamedInstrProfRecord &Record) {
  ArrayRef<NamedInstrProfRecord> Data;

  Error E = Index->getRecords(Data);
  if (E)
    return error(std::move(E));

  Record = Data[RecordIndex++];
  if (RecordIndex >= Data.size()) {
    Index->advanceToNextKey();
    RecordIndex = 0;
  }
  return success();
}

Error IndexedInstrProfReader::readBinaryIds(
    std::vector<llvm::object::BuildID> &BinaryIds) {
  return readBinaryIdsInternal(*DataBuffer, BinaryIdsSize, BinaryIdsStart,
                               BinaryIds, llvm::support::little);
}

Error IndexedInstrProfReader::printBinaryIds(raw_ostream &OS) {
  return printBinaryIdsInternal(OS, *DataBuffer, BinaryIdsSize, BinaryIdsStart,
                                llvm::support::little);
}

void InstrProfReader::accumulateCounts(CountSumOrPercent &Sum, bool IsCS) {
  uint64_t NumFuncs = 0;
  for (const auto &Func : *this) {
    if (isIRLevelProfile()) {
      bool FuncIsCS = NamedInstrProfRecord::hasCSFlagInHash(Func.Hash);
      if (FuncIsCS != IsCS)
        continue;
    }
    Func.accumulateCounts(Sum);
    ++NumFuncs;
  }
  Sum.NumEntries = NumFuncs;
}
