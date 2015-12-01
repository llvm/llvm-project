//=-- InstrProfReader.cpp - Instrumented profiling reader -------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

using namespace llvm;

static ErrorOr<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(std::string Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = BufferOrErr.getError())
    return EC;
  return std::move(BufferOrErr.get());
}

static std::error_code initializeReader(InstrProfReader &Reader) {
  return Reader.readHeader();
}

ErrorOr<std::unique_ptr<InstrProfReader>>
InstrProfReader::create(std::string Path) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path);
  if (std::error_code EC = BufferOrError.getError())
    return EC;
  return InstrProfReader::create(std::move(BufferOrError.get()));
}

ErrorOr<std::unique_ptr<InstrProfReader>>
InstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer) {
  // Sanity check the buffer.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return instrprof_error::too_large;

  std::unique_ptr<InstrProfReader> Result;
  // Create the reader.
  if (IndexedInstrProfReader::hasFormat(*Buffer))
    Result.reset(new IndexedInstrProfReader(std::move(Buffer)));
  else if (RawInstrProfReader64::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader64(std::move(Buffer)));
  else if (RawInstrProfReader32::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader32(std::move(Buffer)));
  else if (TextInstrProfReader::hasFormat(*Buffer))
    Result.reset(new TextInstrProfReader(std::move(Buffer)));
  else
    return instrprof_error::unrecognized_format;

  // Initialize the reader and return the result.
  if (std::error_code EC = initializeReader(*Result))
    return EC;

  return std::move(Result);
}

ErrorOr<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(std::string Path) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path);
  if (std::error_code EC = BufferOrError.getError())
    return EC;
  return IndexedInstrProfReader::create(std::move(BufferOrError.get()));
}


ErrorOr<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer) {
  // Sanity check the buffer.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return instrprof_error::too_large;

  // Create the reader.
  if (!IndexedInstrProfReader::hasFormat(*Buffer))
    return instrprof_error::bad_magic;
  auto Result = llvm::make_unique<IndexedInstrProfReader>(std::move(Buffer));

  // Initialize the reader and return the result.
  if (std::error_code EC = initializeReader(*Result))
    return EC;

  return std::move(Result);
}

void InstrProfIterator::Increment() {
  if (Reader->readNextRecord(Record))
    *this = InstrProfIterator();
}

bool TextInstrProfReader::hasFormat(const MemoryBuffer &Buffer) {
  // Verify that this really looks like plain ASCII text by checking a
  // 'reasonable' number of characters (up to profile magic size).
  size_t count = std::min(Buffer.getBufferSize(), sizeof(uint64_t));
  StringRef buffer = Buffer.getBufferStart();
  return count == 0 || std::all_of(buffer.begin(), buffer.begin() + count,
    [](char c) { return ::isprint(c) || ::isspace(c); });
}

std::error_code TextInstrProfReader::readNextRecord(InstrProfRecord &Record) {
  // Skip empty lines and comments.
  while (!Line.is_at_end() && (Line->empty() || Line->startswith("#")))
    ++Line;
  // If we hit EOF while looking for a name, we're done.
  if (Line.is_at_end())
    return error(instrprof_error::eof);

  // Read the function name.
  Record.Name = *Line++;

  // Read the function hash.
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(0, Record.Hash))
    return error(instrprof_error::malformed);

  // Read the number of counters.
  uint64_t NumCounters;
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(10, NumCounters))
    return error(instrprof_error::malformed);
  if (NumCounters == 0)
    return error(instrprof_error::malformed);

  // Read each counter and fill our internal storage with the values.
  Record.Counts.clear();
  Record.Counts.reserve(NumCounters);
  for (uint64_t I = 0; I < NumCounters; ++I) {
    if (Line.is_at_end())
      return error(instrprof_error::truncated);
    uint64_t Count;
    if ((Line++)->getAsInteger(10, Count))
      return error(instrprof_error::malformed);
    Record.Counts.push_back(Count);
  }

  return success();
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
std::error_code RawInstrProfReader<IntPtrT>::readHeader() {
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
std::error_code
RawInstrProfReader<IntPtrT>::readNextHeader(const char *CurrentPos) {
  const char *End = DataBuffer->getBufferEnd();
  // Skip zero padding between profiles.
  while (CurrentPos != End && *CurrentPos == 0)
    ++CurrentPos;
  // If there's nothing left, we're done.
  if (CurrentPos == End)
    return instrprof_error::eof;
  // If there isn't enough space for another header, this is probably just
  // garbage at the end of the file.
  if (CurrentPos + sizeof(RawInstrProf::Header) > End)
    return instrprof_error::malformed;
  // The writer ensures each profile is padded to start at an aligned address.
  if (reinterpret_cast<size_t>(CurrentPos) % alignOf<uint64_t>())
    return instrprof_error::malformed;
  // The magic should have the same byte order as in the previous header.
  uint64_t Magic = *reinterpret_cast<const uint64_t *>(CurrentPos);
  if (Magic != swap(RawInstrProf::getMagic<IntPtrT>()))
    return instrprof_error::bad_magic;

  // There's another profile to read, so we need to process the header.
  auto *Header = reinterpret_cast<const RawInstrProf::Header *>(CurrentPos);
  return readHeader(*Header);
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readHeader(
    const RawInstrProf::Header &Header) {
  if (swap(Header.Version) != RawInstrProf::Version)
    return error(instrprof_error::unsupported_version);

  CountersDelta = swap(Header.CountersDelta);
  NamesDelta = swap(Header.NamesDelta);
  ValueDataDelta = swap(Header.ValueDataDelta);
  auto DataSize = swap(Header.DataSize);
  auto CountersSize = swap(Header.CountersSize);
  auto NamesSize = swap(Header.NamesSize);
  auto ValueDataSize = swap(Header.ValueDataSize);
  ValueKindLast = swap(Header.ValueKindLast);

  auto DataSizeInBytes = DataSize * sizeof(RawInstrProf::ProfileData<IntPtrT>);
  auto PaddingSize = getNumPaddingBytes(NamesSize);

  ptrdiff_t DataOffset = sizeof(RawInstrProf::Header);
  ptrdiff_t CountersOffset = DataOffset + DataSizeInBytes;
  ptrdiff_t NamesOffset = CountersOffset + sizeof(uint64_t) * CountersSize;
  ptrdiff_t ValueDataOffset = NamesOffset + NamesSize + PaddingSize;
  size_t ProfileSize = ValueDataOffset + ValueDataSize;

  auto *Start = reinterpret_cast<const char *>(&Header);
  if (Start + ProfileSize > DataBuffer->getBufferEnd())
    return error(instrprof_error::bad_header);

  Data = reinterpret_cast<const RawInstrProf::ProfileData<IntPtrT> *>(
      Start + DataOffset);
  DataEnd = Data + DataSize;
  CountersStart = reinterpret_cast<const uint64_t *>(Start + CountersOffset);
  NamesStart = Start + NamesOffset;
  ValueDataStart = reinterpret_cast<const uint8_t*>(Start + ValueDataOffset);
  ProfileEnd = Start + ProfileSize;

  FunctionPtrToNameMap.clear();
  for (const RawInstrProf::ProfileData<IntPtrT> *I = Data; I != DataEnd; ++I) {
    const IntPtrT FPtr = swap(I->FunctionPointer);
    if (!FPtr)
      continue;
    StringRef FunctionName(getName(I->NamePtr), swap(I->NameSize));
    const char* NameEntryPtr = StringTable.insertString(FunctionName);
    FunctionPtrToNameMap.push_back(std::pair<const IntPtrT, const char*>
                                   (FPtr, NameEntryPtr));
  }
  std::sort(FunctionPtrToNameMap.begin(), FunctionPtrToNameMap.end(), less_first());
  FunctionPtrToNameMap.erase(std::unique(FunctionPtrToNameMap.begin(),
                                         FunctionPtrToNameMap.end()),
                                         FunctionPtrToNameMap.end());
  return success();
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readName(InstrProfRecord &Record) {
  Record.Name = StringRef(getName(Data->NamePtr), swap(Data->NameSize));
  if (Record.Name.data() < NamesStart ||
      Record.Name.data() + Record.Name.size() >
          reinterpret_cast<const char *>(ValueDataStart))
    return error(instrprof_error::malformed);
  return success();
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readFuncHash(
    InstrProfRecord &Record) {
  Record.Hash = swap(Data->FuncHash);
  return success();
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readRawCounts(
    InstrProfRecord &Record) {
  uint32_t NumCounters = swap(Data->NumCounters);
  IntPtrT CounterPtr = Data->CounterPtr;
  if (NumCounters == 0)
    return error(instrprof_error::malformed);

  auto RawCounts = makeArrayRef(getCounter(CounterPtr), NumCounters);
  auto *NamesStartAsCounter = reinterpret_cast<const uint64_t *>(NamesStart);

  // Check bounds.
  if (RawCounts.data() < CountersStart ||
      RawCounts.data() + RawCounts.size() > NamesStartAsCounter)
    return error(instrprof_error::malformed);

  if (ShouldSwapBytes) {
    Record.Counts.clear();
    Record.Counts.reserve(RawCounts.size());
    for (uint64_t Count : RawCounts)
      Record.Counts.push_back(swap(Count));
  } else
    Record.Counts = RawCounts;

  return success();
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readValueProfilingData(
    InstrProfRecord &Record) {

  Record.clearValueData();
  if (!Data->Values || (ValueDataDelta == 0))
    return success();

  // Read value data.
  uint64_t NumVSites = 0;
  for (uint32_t Kind = IPVK_First; Kind <= ValueKindLast; ++Kind)
    NumVSites += swap(Data->NumValueSites[Kind]);
  NumVSites += getNumPaddingBytes(NumVSites);

  auto VDataCounts = makeArrayRef(getValueDataCounts(Data->Values), NumVSites);
  // Check bounds.
  if (VDataCounts.data() < ValueDataStart ||
      VDataCounts.data() + VDataCounts.size() >
          reinterpret_cast<const uint8_t *>(ProfileEnd))
    return error(instrprof_error::malformed);

  const InstrProfValueData *VDataPtr =
      getValueData(swap(Data->Values) + NumVSites);
  for (uint32_t Kind = IPVK_First; Kind <= ValueKindLast; ++Kind) {
    NumVSites = swap(Data->NumValueSites[Kind]);
    Record.reserveSites(Kind, NumVSites);
    for (uint32_t VSite = 0; VSite < NumVSites; ++VSite) {

      uint32_t VDataCount = VDataCounts[VSite];
      if ((const char *)(VDataPtr + VDataCount) > ProfileEnd)
        return error(instrprof_error::malformed);

      std::vector<InstrProfValueData> CurrentValues;
      CurrentValues.reserve(VDataCount);
      for (uint32_t VIndex = 0; VIndex < VDataCount; ++VIndex) {
        uint64_t TargetValue = swap(VDataPtr->Value);
        uint64_t Count = swap(VDataPtr->Count);
        CurrentValues.push_back({TargetValue, Count});
        ++VDataPtr;
      }
      Record.addValueData(Kind, VSite, CurrentValues.data(),
                          VDataCount, &FunctionPtrToNameMap);
    }
  }
  return success();
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readNextRecord(
    InstrProfRecord &Record) {
  if (atEnd())
    if (std::error_code EC = readNextHeader(ProfileEnd))
      return EC;

  // Read name ad set it in Record.
  if (std::error_code EC = readName(Record))
    return EC;

  // Read FuncHash and set it in Record.
  if (std::error_code EC = readFuncHash(Record))
    return EC;

  // Read raw counts and set Record.
  if (std::error_code EC = readRawCounts(Record))
    return EC;

  // Read value data and set Record.
  if (std::error_code EC = readValueProfilingData(Record)) return EC;

  // Iterate.
  advanceData();
  return success();
}

namespace llvm {
template class RawInstrProfReader<uint32_t>;
template class RawInstrProfReader<uint64_t>;
}

InstrProfLookupTrait::hash_value_type
InstrProfLookupTrait::ComputeHash(StringRef K) {
  return IndexedInstrProf::ComputeHash(HashType, K);
}

typedef InstrProfLookupTrait::data_type data_type;
typedef InstrProfLookupTrait::offset_type offset_type;

bool InstrProfLookupTrait::readValueProfilingData(
    const unsigned char *&D, const unsigned char *const End) {
  ErrorOr<std::unique_ptr<ValueProfData>> VDataPtrOrErr =
      ValueProfData::getValueProfData(D, End, ValueProfDataEndianness);

  if (VDataPtrOrErr.getError())
    return false;

  VDataPtrOrErr.get()->deserializeTo(DataBuffer.back(), &HashKeys);
  D += VDataPtrOrErr.get()->TotalSize;

  return true;
}

data_type InstrProfLookupTrait::ReadData(StringRef K, const unsigned char *D,
                                         offset_type N) {
  // Check if the data is corrupt. If so, don't try to read it.
  if (N % sizeof(uint64_t))
    return data_type();

  DataBuffer.clear();
  std::vector<uint64_t> CounterBuffer;

  using namespace support;
  const unsigned char *End = D + N;
  while (D < End) {
    // Read hash.
    if (D + sizeof(uint64_t) >= End)
      return data_type();
    uint64_t Hash = endian::readNext<uint64_t, little, unaligned>(D);

    // Initialize number of counters for FormatVersion == 1.
    uint64_t CountsSize = N / sizeof(uint64_t) - 1;
    // If format version is different then read the number of counters.
    if (FormatVersion != 1) {
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
    if (FormatVersion > 2 && !readValueProfilingData(D, End)) {
      DataBuffer.clear();
      return data_type();
    }
  }
  return DataBuffer;
}

template <typename HashTableImpl>
std::error_code InstrProfReaderIndex<HashTableImpl>::getRecords(
    StringRef FuncName, ArrayRef<InstrProfRecord> &Data) {
  auto Iter = HashTable->find(FuncName);
  if (Iter == HashTable->end())
    return instrprof_error::unknown_function;

  Data = (*Iter);
  if (Data.empty())
    return instrprof_error::malformed;

  return instrprof_error::success;
}

template <typename HashTableImpl>
std::error_code InstrProfReaderIndex<HashTableImpl>::getRecords(
    ArrayRef<InstrProfRecord> &Data) {
  if (atEnd())
    return instrprof_error::eof;

  Data = *RecordIterator;

  if (Data.empty()) return instrprof_error::malformed;

  return instrprof_error::success;
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
  // Form the map of hash values to const char* keys in profiling data.
  std::vector<std::pair<uint64_t, const char *>> HashKeys;
  for (auto Key : HashTable->keys()) {
    const char *KeyTableRef = StringTable.insertString(Key);
    HashKeys.push_back(std::make_pair(ComputeHash(HashType, Key), KeyTableRef));
  }
  std::sort(HashKeys.begin(), HashKeys.end(), less_first());
  HashKeys.erase(std::unique(HashKeys.begin(), HashKeys.end()), HashKeys.end());
  // Set the hash key map for the InstrLookupTrait
  HashTable->getInfoObj().setHashKeys(std::move(HashKeys));
  RecordIterator = HashTable->data_begin();
}

bool IndexedInstrProfReader::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < 8)
    return false;
  using namespace support;
  uint64_t Magic =
      endian::read<uint64_t, little, aligned>(DataBuffer.getBufferStart());
  // Verify that it's magical.
  return Magic == IndexedInstrProf::Magic;
}

std::error_code IndexedInstrProfReader::readHeader() {
  const unsigned char *Start =
      (const unsigned char *)DataBuffer->getBufferStart();
  const unsigned char *Cur = Start;
  if ((const unsigned char *)DataBuffer->getBufferEnd() - Cur < 24)
    return error(instrprof_error::truncated);

  using namespace support;

  auto *Header = reinterpret_cast<const IndexedInstrProf::Header *>(Cur);
  Cur += sizeof(IndexedInstrProf::Header);

  // Check the magic number.
  uint64_t Magic = endian::byte_swap<uint64_t, little>(Header->Magic);
  if (Magic != IndexedInstrProf::Magic)
    return error(instrprof_error::bad_magic);

  // Read the version.
  uint64_t FormatVersion = endian::byte_swap<uint64_t, little>(Header->Version);
  if (FormatVersion > IndexedInstrProf::Version)
    return error(instrprof_error::unsupported_version);

  // Read the maximal function count.
  MaxFunctionCount =
      endian::byte_swap<uint64_t, little>(Header->MaxFunctionCount);

  // Read the hash type and start offset.
  IndexedInstrProf::HashT HashType = static_cast<IndexedInstrProf::HashT>(
      endian::byte_swap<uint64_t, little>(Header->HashType));
  if (HashType > IndexedInstrProf::HashT::Last)
    return error(instrprof_error::unsupported_hash_type);

  uint64_t HashOffset = endian::byte_swap<uint64_t, little>(Header->HashOffset);

  // The rest of the file is an on disk hash table.
  InstrProfReaderIndexBase *IndexPtr = nullptr;
  IndexPtr = new InstrProfReaderIndex<OnDiskHashTableImplV3>(
      Start + HashOffset, Cur, Start, HashType, FormatVersion);
  Index.reset(IndexPtr);
  return success();
}

ErrorOr<InstrProfRecord>
IndexedInstrProfReader::getInstrProfRecord(StringRef FuncName,
                                           uint64_t FuncHash) {
  ArrayRef<InstrProfRecord> Data;
  std::error_code EC = Index->getRecords(FuncName, Data);
  if (EC != instrprof_error::success)
    return EC;
  // Found it. Look for counters with the right hash.
  for (unsigned I = 0, E = Data.size(); I < E; ++I) {
    // Check for a match and fill the vector if there is one.
    if (Data[I].Hash == FuncHash) {
      return std::move(Data[I]);
    }
  }
  return error(instrprof_error::hash_mismatch);
}

std::error_code
IndexedInstrProfReader::getFunctionCounts(StringRef FuncName, uint64_t FuncHash,
                                          std::vector<uint64_t> &Counts) {
  ErrorOr<InstrProfRecord> Record = getInstrProfRecord(FuncName, FuncHash);
  if (std::error_code EC = Record.getError())
    return EC;

  Counts = Record.get().Counts;
  return success();
}

std::error_code IndexedInstrProfReader::readNextRecord(
    InstrProfRecord &Record) {
  static unsigned RecordIndex = 0;

  ArrayRef<InstrProfRecord> Data;

  std::error_code EC = Index->getRecords(Data);
  if (EC != instrprof_error::success)
    return error(EC);

  Record = Data[RecordIndex++];
  if (RecordIndex >= Data.size()) {
    Index->advanceToNextKey();
    RecordIndex = 0;
  }
  return success();
}
