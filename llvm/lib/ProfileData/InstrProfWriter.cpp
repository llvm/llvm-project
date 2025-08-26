//===- InstrProfWriter.cpp - Instrumented profiling writer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ProfileData/IndexedMemProfData.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/OnDiskHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <ctime>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

namespace llvm {

class InstrProfRecordWriterTrait {
public:
  using key_type = StringRef;
  using key_type_ref = StringRef;

  using data_type = const InstrProfWriter::ProfilingData *const;
  using data_type_ref = const InstrProfWriter::ProfilingData *const;

  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

  llvm::endianness ValueProfDataEndianness = llvm::endianness::little;
  InstrProfSummaryBuilder *SummaryBuilder;
  InstrProfSummaryBuilder *CSSummaryBuilder;

  InstrProfRecordWriterTrait() = default;

  static hash_value_type ComputeHash(key_type_ref K) {
    return IndexedInstrProf::ComputeHash(K);
  }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace support;

    endian::Writer LE(Out, llvm::endianness::little);

    offset_type N = K.size();
    LE.write<offset_type>(N);

    offset_type M = 0;
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;
      M += sizeof(uint64_t); // The function hash
      M += sizeof(uint64_t); // The size of the Counts vector
      M += ProfRecord.Counts.size() * sizeof(uint64_t);
      M += sizeof(uint64_t); // The size of the Bitmap vector
      M += ProfRecord.BitmapBytes.size() * sizeof(uint64_t);

      // Value data
      M += ValueProfData::getSize(ProfileData.second);
    }
    LE.write<offset_type>(M);

    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type N) {
    Out.write(K.data(), N);
  }

  void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V, offset_type) {
    using namespace support;

    endian::Writer LE(Out, llvm::endianness::little);
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;
      if (NamedInstrProfRecord::hasCSFlagInHash(ProfileData.first))
        CSSummaryBuilder->addRecord(ProfRecord);
      else
        SummaryBuilder->addRecord(ProfRecord);

      LE.write<uint64_t>(ProfileData.first); // Function hash
      LE.write<uint64_t>(ProfRecord.Counts.size());
      for (uint64_t I : ProfRecord.Counts)
        LE.write<uint64_t>(I);

      LE.write<uint64_t>(ProfRecord.BitmapBytes.size());
      for (uint64_t I : ProfRecord.BitmapBytes)
        LE.write<uint64_t>(I);

      // Write value data
      std::unique_ptr<ValueProfData> VDataPtr =
          ValueProfData::serializeFrom(ProfileData.second);
      uint32_t S = VDataPtr->getSize();
      VDataPtr->swapBytesFromHost(ValueProfDataEndianness);
      Out.write((const char *)VDataPtr.get(), S);
    }
  }
};

} // end namespace llvm

InstrProfWriter::InstrProfWriter(
    bool Sparse, uint64_t TemporalProfTraceReservoirSize,
    uint64_t MaxTemporalProfTraceLength, bool WritePrevVersion,
    memprof::IndexedVersion MemProfVersionRequested, bool MemProfFullSchema,
    bool MemprofGenerateRandomHotness,
    unsigned MemprofGenerateRandomHotnessSeed)
    : Sparse(Sparse), MaxTemporalProfTraceLength(MaxTemporalProfTraceLength),
      TemporalProfTraceReservoirSize(TemporalProfTraceReservoirSize),
      InfoObj(new InstrProfRecordWriterTrait()),
      WritePrevVersion(WritePrevVersion),
      MemProfVersionRequested(MemProfVersionRequested),
      MemProfFullSchema(MemProfFullSchema),
      MemprofGenerateRandomHotness(MemprofGenerateRandomHotness) {
  // Set up the random number seed if requested.
  if (MemprofGenerateRandomHotness) {
    unsigned seed = MemprofGenerateRandomHotnessSeed
                        ? MemprofGenerateRandomHotnessSeed
                        : std::time(nullptr);
    errs() << "random hotness seed = " << seed << "\n";
    std::srand(seed);
  }
}

InstrProfWriter::~InstrProfWriter() { delete InfoObj; }

// Internal interface for testing purpose only.
void InstrProfWriter::setValueProfDataEndianness(llvm::endianness Endianness) {
  InfoObj->ValueProfDataEndianness = Endianness;
}

void InstrProfWriter::setOutputSparse(bool Sparse) { this->Sparse = Sparse; }

void InstrProfWriter::addRecord(NamedInstrProfRecord &&I, uint64_t Weight,
                                function_ref<void(Error)> Warn) {
  auto Name = I.Name;
  auto Hash = I.Hash;
  addRecord(Name, Hash, std::move(I), Weight, Warn);
}

void InstrProfWriter::overlapRecord(NamedInstrProfRecord &&Other,
                                    OverlapStats &Overlap,
                                    OverlapStats &FuncLevelOverlap,
                                    const OverlapFuncFilters &FuncFilter) {
  auto Name = Other.Name;
  auto Hash = Other.Hash;
  Other.accumulateCounts(FuncLevelOverlap.Test);
  auto It = FunctionData.find(Name);
  if (It == FunctionData.end()) {
    Overlap.addOneUnique(FuncLevelOverlap.Test);
    return;
  }
  if (FuncLevelOverlap.Test.CountSum < 1.0f) {
    Overlap.Overlap.NumEntries += 1;
    return;
  }
  auto &ProfileDataMap = It->second;
  auto [Where, NewFunc] = ProfileDataMap.try_emplace(Hash);
  if (NewFunc) {
    Overlap.addOneMismatch(FuncLevelOverlap.Test);
    return;
  }
  InstrProfRecord &Dest = Where->second;

  uint64_t ValueCutoff = FuncFilter.ValueCutoff;
  if (!FuncFilter.NameFilter.empty() && Name.contains(FuncFilter.NameFilter))
    ValueCutoff = 0;

  Dest.overlap(Other, Overlap, FuncLevelOverlap, ValueCutoff);
}

void InstrProfWriter::addRecord(StringRef Name, uint64_t Hash,
                                InstrProfRecord &&I, uint64_t Weight,
                                function_ref<void(Error)> Warn) {
  auto &ProfileDataMap = FunctionData[Name];

  auto [Where, NewFunc] = ProfileDataMap.try_emplace(Hash);
  InstrProfRecord &Dest = Where->second;

  auto MapWarn = [&](instrprof_error E) {
    Warn(make_error<InstrProfError>(E));
  };

  if (NewFunc) {
    // We've never seen a function with this name and hash, add it.
    Dest = std::move(I);
    if (Weight > 1)
      Dest.scale(Weight, 1, MapWarn);
  } else {
    // We're updating a function we've seen before.
    Dest.merge(I, Weight, MapWarn);
  }

  Dest.sortValueData();
}

void InstrProfWriter::addMemProfRecord(
    const Function::GUID Id, const memprof::IndexedMemProfRecord &Record) {
  auto NewRecord = Record;
  // Provoke random hotness values if requested. We specify the lifetime access
  // density and lifetime length that will result in a cold or not cold hotness.
  // See the logic in getAllocType() in Analysis/MemoryProfileInfo.cpp.
  if (MemprofGenerateRandomHotness) {
    for (auto &Alloc : NewRecord.AllocSites) {
      // To get a not cold context, set the lifetime access density to the
      // maximum value and the lifetime to 0.
      uint64_t NewTLAD = std::numeric_limits<uint64_t>::max();
      uint64_t NewTL = 0;
      bool IsCold = std::rand() % 2;
      if (IsCold) {
        // To get a cold context, set the lifetime access density to 0 and the
        // lifetime to the maximum value.
        NewTLAD = 0;
        NewTL = std::numeric_limits<uint64_t>::max();
      }
      Alloc.Info.setTotalLifetimeAccessDensity(NewTLAD);
      Alloc.Info.setTotalLifetime(NewTL);
    }
  }
  MemProfSumBuilder.addRecord(NewRecord);
  auto [Iter, Inserted] = MemProfData.Records.insert({Id, NewRecord});
  // If we inserted a new record then we are done.
  if (Inserted) {
    return;
  }
  memprof::IndexedMemProfRecord &Existing = Iter->second;
  Existing.merge(NewRecord);
}

bool InstrProfWriter::addMemProfFrame(const memprof::FrameId Id,
                                      const memprof::Frame &Frame,
                                      function_ref<void(Error)> Warn) {
  auto [Iter, Inserted] = MemProfData.Frames.insert({Id, Frame});
  // If a mapping already exists for the current frame id and it does not
  // match the new mapping provided then reset the existing contents and bail
  // out. We don't support the merging of memprof data whose Frame -> Id
  // mapping across profiles is inconsistent.
  if (!Inserted && Iter->second != Frame) {
    Warn(make_error<InstrProfError>(instrprof_error::malformed,
                                    "frame to id mapping mismatch"));
    return false;
  }
  return true;
}

bool InstrProfWriter::addMemProfCallStack(
    const memprof::CallStackId CSId,
    const llvm::SmallVector<memprof::FrameId> &CallStack,
    function_ref<void(Error)> Warn) {
  auto [Iter, Inserted] = MemProfData.CallStacks.insert({CSId, CallStack});
  // If a mapping already exists for the current call stack id and it does not
  // match the new mapping provided then reset the existing contents and bail
  // out. We don't support the merging of memprof data whose CallStack -> Id
  // mapping across profiles is inconsistent.
  if (!Inserted && Iter->second != CallStack) {
    Warn(make_error<InstrProfError>(instrprof_error::malformed,
                                    "call stack to id mapping mismatch"));
    return false;
  }
  return true;
}

bool InstrProfWriter::addMemProfData(memprof::IndexedMemProfData Incoming,
                                     function_ref<void(Error)> Warn) {
  // Return immediately if everything is empty.
  if (Incoming.Frames.empty() && Incoming.CallStacks.empty() &&
      Incoming.Records.empty())
    return true;

  // Otherwise, every component must be non-empty.
  assert(!Incoming.Frames.empty() && !Incoming.CallStacks.empty() &&
         !Incoming.Records.empty());

  if (MemProfData.Frames.empty())
    MemProfData.Frames = std::move(Incoming.Frames);
  else
    for (const auto &[Id, F] : Incoming.Frames)
      if (addMemProfFrame(Id, F, Warn))
        return false;

  if (MemProfData.CallStacks.empty())
    MemProfData.CallStacks = std::move(Incoming.CallStacks);
  else
    for (const auto &[CSId, CS] : Incoming.CallStacks)
      if (addMemProfCallStack(CSId, CS, Warn))
        return false;

  // Add one record at a time if randomization is requested.
  if (MemProfData.Records.empty() && !MemprofGenerateRandomHotness) {
    // Need to manually add each record to the builder, which is otherwise done
    // in addMemProfRecord.
    for (const auto &[GUID, Record] : Incoming.Records)
      MemProfSumBuilder.addRecord(Record);
    MemProfData.Records = std::move(Incoming.Records);
  } else {
    for (const auto &[GUID, Record] : Incoming.Records)
      addMemProfRecord(GUID, Record);
  }

  return true;
}

void InstrProfWriter::addBinaryIds(ArrayRef<llvm::object::BuildID> BIs) {
  llvm::append_range(BinaryIds, BIs);
}

void InstrProfWriter::addDataAccessProfData(
    std::unique_ptr<memprof::DataAccessProfData> DataAccessProfDataIn) {
  DataAccessProfileData = std::move(DataAccessProfDataIn);
}

void InstrProfWriter::addTemporalProfileTraces(
    SmallVectorImpl<TemporalProfTraceTy> &SrcTraces, uint64_t SrcStreamSize) {
  if (TemporalProfTraces.size() > TemporalProfTraceReservoirSize)
    TemporalProfTraces.truncate(TemporalProfTraceReservoirSize);
  for (auto &Trace : SrcTraces)
    if (Trace.FunctionNameRefs.size() > MaxTemporalProfTraceLength)
      Trace.FunctionNameRefs.resize(MaxTemporalProfTraceLength);
  llvm::erase_if(SrcTraces, [](auto &T) { return T.FunctionNameRefs.empty(); });
  // If there are no source traces, it is probably because
  // --temporal-profile-max-trace-length=0 was set to deliberately remove all
  // traces. In that case, we do not want to increase the stream size
  if (SrcTraces.empty())
    return;
  // Add traces until our reservoir is full or we run out of source traces
  auto SrcTraceIt = SrcTraces.begin();
  while (TemporalProfTraces.size() < TemporalProfTraceReservoirSize &&
         SrcTraceIt < SrcTraces.end())
    TemporalProfTraces.push_back(*SrcTraceIt++);
  // Our reservoir is full, we need to sample the source stream
  llvm::shuffle(SrcTraceIt, SrcTraces.end(), RNG);
  for (uint64_t I = TemporalProfTraces.size();
       I < SrcStreamSize && SrcTraceIt < SrcTraces.end(); I++) {
    std::uniform_int_distribution<uint64_t> Distribution(0, I);
    uint64_t RandomIndex = Distribution(RNG);
    if (RandomIndex < TemporalProfTraces.size())
      TemporalProfTraces[RandomIndex] = *SrcTraceIt++;
  }
  TemporalProfTraceStreamSize += SrcStreamSize;
}

void InstrProfWriter::mergeRecordsFromWriter(InstrProfWriter &&IPW,
                                             function_ref<void(Error)> Warn) {
  for (auto &I : IPW.FunctionData)
    for (auto &Func : I.getValue())
      addRecord(I.getKey(), Func.first, std::move(Func.second), 1, Warn);

  BinaryIds.reserve(BinaryIds.size() + IPW.BinaryIds.size());
  for (auto &I : IPW.BinaryIds)
    addBinaryIds(I);

  addTemporalProfileTraces(IPW.TemporalProfTraces,
                           IPW.TemporalProfTraceStreamSize);

  MemProfData.Frames.reserve(IPW.MemProfData.Frames.size());
  for (auto &[FrameId, Frame] : IPW.MemProfData.Frames) {
    // If we weren't able to add the frame mappings then it doesn't make sense
    // to try to merge the records from this profile.
    if (!addMemProfFrame(FrameId, Frame, Warn))
      return;
  }

  MemProfData.CallStacks.reserve(IPW.MemProfData.CallStacks.size());
  for (auto &[CSId, CallStack] : IPW.MemProfData.CallStacks) {
    if (!addMemProfCallStack(CSId, CallStack, Warn))
      return;
  }

  MemProfData.Records.reserve(IPW.MemProfData.Records.size());
  for (auto &[GUID, Record] : IPW.MemProfData.Records) {
    addMemProfRecord(GUID, Record);
  }
}

bool InstrProfWriter::shouldEncodeData(const ProfilingData &PD) {
  if (!Sparse)
    return true;
  for (const auto &Func : PD) {
    const InstrProfRecord &IPR = Func.second;
    if (llvm::any_of(IPR.Counts, [](uint64_t Count) { return Count > 0; }))
      return true;
    if (llvm::any_of(IPR.BitmapBytes, [](uint8_t Byte) { return Byte > 0; }))
      return true;
  }
  return false;
}

static void setSummary(IndexedInstrProf::Summary *TheSummary,
                       ProfileSummary &PS) {
  using namespace IndexedInstrProf;

  const std::vector<ProfileSummaryEntry> &Res = PS.getDetailedSummary();
  TheSummary->NumSummaryFields = Summary::NumKinds;
  TheSummary->NumCutoffEntries = Res.size();
  TheSummary->set(Summary::MaxFunctionCount, PS.getMaxFunctionCount());
  TheSummary->set(Summary::MaxBlockCount, PS.getMaxCount());
  TheSummary->set(Summary::MaxInternalBlockCount, PS.getMaxInternalCount());
  TheSummary->set(Summary::TotalBlockCount, PS.getTotalCount());
  TheSummary->set(Summary::TotalNumBlocks, PS.getNumCounts());
  TheSummary->set(Summary::TotalNumFunctions, PS.getNumFunctions());
  for (unsigned I = 0; I < Res.size(); I++)
    TheSummary->setEntry(I, Res[I]);
}

uint64_t InstrProfWriter::writeHeader(const IndexedInstrProf::Header &Header,
                                      const bool WritePrevVersion,
                                      ProfOStream &OS) {
  // Only write out the first four fields.
  for (int I = 0; I < 4; I++)
    OS.write(reinterpret_cast<const uint64_t *>(&Header)[I]);

  // Remember the offset of the remaining fields to allow back patching later.
  auto BackPatchStartOffset = OS.tell();

  // Reserve the space for back patching later.
  OS.write(0); // HashOffset
  OS.write(0); // MemProfOffset
  OS.write(0); // BinaryIdOffset
  OS.write(0); // TemporalProfTracesOffset
  if (!WritePrevVersion)
    OS.write(0); // VTableNamesOffset

  return BackPatchStartOffset;
}

Error InstrProfWriter::writeBinaryIds(ProfOStream &OS) {
  // BinaryIdSection has two parts:
  // 1. uint64_t BinaryIdsSectionSize
  // 2. list of binary ids that consist of:
  //    a. uint64_t BinaryIdLength
  //    b. uint8_t  BinaryIdData
  //    c. uint8_t  Padding (if necessary)
  // Calculate size of binary section.
  uint64_t BinaryIdsSectionSize = 0;

  // Remove duplicate binary ids.
  llvm::sort(BinaryIds);
  BinaryIds.erase(llvm::unique(BinaryIds), BinaryIds.end());

  for (const auto &BI : BinaryIds) {
    // Increment by binary id length data type size.
    BinaryIdsSectionSize += sizeof(uint64_t);
    // Increment by binary id data length, aligned to 8 bytes.
    BinaryIdsSectionSize += alignToPowerOf2(BI.size(), sizeof(uint64_t));
  }
  // Write binary ids section size.
  OS.write(BinaryIdsSectionSize);

  for (const auto &BI : BinaryIds) {
    uint64_t BILen = BI.size();
    // Write binary id length.
    OS.write(BILen);
    // Write binary id data.
    for (unsigned K = 0; K < BILen; K++)
      OS.writeByte(BI[K]);
    // Write padding if necessary.
    uint64_t PaddingSize = alignToPowerOf2(BILen, sizeof(uint64_t)) - BILen;
    for (unsigned K = 0; K < PaddingSize; K++)
      OS.writeByte(0);
  }

  return Error::success();
}

Error InstrProfWriter::writeVTableNames(ProfOStream &OS) {
  std::vector<std::string> VTableNameStrs;
  for (StringRef VTableName : VTableNames.keys())
    VTableNameStrs.push_back(VTableName.str());

  std::string CompressedVTableNames;
  if (!VTableNameStrs.empty())
    if (Error E = collectGlobalObjectNameStrings(
            VTableNameStrs, compression::zlib::isAvailable(),
            CompressedVTableNames))
      return E;

  const uint64_t CompressedStringLen = CompressedVTableNames.length();

  // Record the length of compressed string.
  OS.write(CompressedStringLen);

  // Write the chars in compressed strings.
  for (auto &c : CompressedVTableNames)
    OS.writeByte(static_cast<uint8_t>(c));

  // Pad up to a multiple of 8.
  // InstrProfReader could read bytes according to 'CompressedStringLen'.
  const uint64_t PaddedLength = alignTo(CompressedStringLen, 8);

  for (uint64_t K = CompressedStringLen; K < PaddedLength; K++)
    OS.writeByte(0);

  return Error::success();
}

Error InstrProfWriter::writeImpl(ProfOStream &OS) {
  using namespace IndexedInstrProf;
  using namespace support;

  OnDiskChainedHashTableGenerator<InstrProfRecordWriterTrait> Generator;

  InstrProfSummaryBuilder ISB(ProfileSummaryBuilder::DefaultCutoffs);
  InfoObj->SummaryBuilder = &ISB;
  InstrProfSummaryBuilder CSISB(ProfileSummaryBuilder::DefaultCutoffs);
  InfoObj->CSSummaryBuilder = &CSISB;

  // Populate the hash table generator.
  SmallVector<std::pair<StringRef, const ProfilingData *>> OrderedData;
  for (const auto &I : FunctionData)
    if (shouldEncodeData(I.getValue()))
      OrderedData.emplace_back((I.getKey()), &I.getValue());
  llvm::sort(OrderedData, less_first());
  for (const auto &I : OrderedData)
    Generator.insert(I.first, I.second);

  // Write the header.
  IndexedInstrProf::Header Header;
  Header.Version = WritePrevVersion
                       ? IndexedInstrProf::ProfVersion::Version11
                       : IndexedInstrProf::ProfVersion::CurrentVersion;
  // The WritePrevVersion handling will either need to be removed or updated
  // if the version is advanced beyond 12.
  static_assert(IndexedInstrProf::ProfVersion::CurrentVersion ==
                IndexedInstrProf::ProfVersion::Version12);
  if (static_cast<bool>(ProfileKind & InstrProfKind::IRInstrumentation))
    Header.Version |= VARIANT_MASK_IR_PROF;
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive))
    Header.Version |= VARIANT_MASK_CSIR_PROF;
  if (static_cast<bool>(ProfileKind &
                        InstrProfKind::FunctionEntryInstrumentation))
    Header.Version |= VARIANT_MASK_INSTR_ENTRY;
  if (static_cast<bool>(ProfileKind &
                        InstrProfKind::LoopEntriesInstrumentation))
    Header.Version |= VARIANT_MASK_INSTR_LOOP_ENTRIES;
  if (static_cast<bool>(ProfileKind & InstrProfKind::SingleByteCoverage))
    Header.Version |= VARIANT_MASK_BYTE_COVERAGE;
  if (static_cast<bool>(ProfileKind & InstrProfKind::FunctionEntryOnly))
    Header.Version |= VARIANT_MASK_FUNCTION_ENTRY_ONLY;
  if (static_cast<bool>(ProfileKind & InstrProfKind::MemProf))
    Header.Version |= VARIANT_MASK_MEMPROF;
  if (static_cast<bool>(ProfileKind & InstrProfKind::TemporalProfile))
    Header.Version |= VARIANT_MASK_TEMPORAL_PROF;

  const uint64_t BackPatchStartOffset =
      writeHeader(Header, WritePrevVersion, OS);

  // Reserve space to write profile summary data.
  uint32_t NumEntries = ProfileSummaryBuilder::DefaultCutoffs.size();
  uint32_t SummarySize = Summary::getSize(Summary::NumKinds, NumEntries);
  // Remember the summary offset.
  uint64_t SummaryOffset = OS.tell();
  for (unsigned I = 0; I < SummarySize / sizeof(uint64_t); I++)
    OS.write(0);
  uint64_t CSSummaryOffset = 0;
  uint64_t CSSummarySize = 0;
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive)) {
    CSSummaryOffset = OS.tell();
    CSSummarySize = SummarySize / sizeof(uint64_t);
    for (unsigned I = 0; I < CSSummarySize; I++)
      OS.write(0);
  }

  // Write the hash table.
  uint64_t HashTableStart = Generator.Emit(OS.OS, *InfoObj);

  // Write the MemProf profile data if we have it.
  uint64_t MemProfSectionStart = 0;
  if (static_cast<bool>(ProfileKind & InstrProfKind::MemProf)) {
    MemProfSectionStart = OS.tell();

    if (auto E = writeMemProf(
            OS, MemProfData, MemProfVersionRequested, MemProfFullSchema,
            std::move(DataAccessProfileData), MemProfSumBuilder.getSummary()))
      return E;
  }

  uint64_t BinaryIdSectionStart = OS.tell();
  if (auto E = writeBinaryIds(OS))
    return E;

  uint64_t VTableNamesSectionStart = OS.tell();

  if (!WritePrevVersion)
    if (Error E = writeVTableNames(OS))
      return E;

  uint64_t TemporalProfTracesSectionStart = 0;
  if (static_cast<bool>(ProfileKind & InstrProfKind::TemporalProfile)) {
    TemporalProfTracesSectionStart = OS.tell();
    OS.write(TemporalProfTraces.size());
    OS.write(TemporalProfTraceStreamSize);
    for (auto &Trace : TemporalProfTraces) {
      OS.write(Trace.Weight);
      OS.write(Trace.FunctionNameRefs.size());
      for (auto &NameRef : Trace.FunctionNameRefs)
        OS.write(NameRef);
    }
  }

  // Allocate space for data to be serialized out.
  std::unique_ptr<IndexedInstrProf::Summary> TheSummary =
      IndexedInstrProf::allocSummary(SummarySize);
  // Compute the Summary and copy the data to the data
  // structure to be serialized out (to disk or buffer).
  std::unique_ptr<ProfileSummary> PS = ISB.getSummary();
  setSummary(TheSummary.get(), *PS);
  InfoObj->SummaryBuilder = nullptr;

  // For Context Sensitive summary.
  std::unique_ptr<IndexedInstrProf::Summary> TheCSSummary = nullptr;
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive)) {
    TheCSSummary = IndexedInstrProf::allocSummary(SummarySize);
    std::unique_ptr<ProfileSummary> CSPS = CSISB.getSummary();
    setSummary(TheCSSummary.get(), *CSPS);
  }
  InfoObj->CSSummaryBuilder = nullptr;

  SmallVector<uint64_t, 8> HeaderOffsets = {HashTableStart, MemProfSectionStart,
                                            BinaryIdSectionStart,
                                            TemporalProfTracesSectionStart};
  if (!WritePrevVersion)
    HeaderOffsets.push_back(VTableNamesSectionStart);

  PatchItem PatchItems[] = {
      // Patch the Header fields
      {BackPatchStartOffset, HeaderOffsets},
      // Patch the summary data.
      {SummaryOffset,
       ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(TheSummary.get()),
                          SummarySize / sizeof(uint64_t))},
      {CSSummaryOffset,
       ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(TheCSSummary.get()),
                          CSSummarySize)}};

  OS.patch(PatchItems);

  for (const auto &I : FunctionData)
    for (const auto &F : I.getValue())
      if (Error E = validateRecord(F.second))
        return E;

  return Error::success();
}

Error InstrProfWriter::write(raw_fd_ostream &OS) {
  // Write the hash table.
  ProfOStream POS(OS);
  return writeImpl(POS);
}

Error InstrProfWriter::write(raw_string_ostream &OS) {
  ProfOStream POS(OS);
  return writeImpl(POS);
}

std::unique_ptr<MemoryBuffer> InstrProfWriter::writeBuffer() {
  std::string Data;
  raw_string_ostream OS(Data);
  // Write the hash table.
  if (Error E = write(OS))
    return nullptr;
  // Return this in an aligned memory buffer.
  return MemoryBuffer::getMemBufferCopy(Data);
}

static const char *ValueProfKindStr[] = {
#define VALUE_PROF_KIND(Enumerator, Value, Descr) #Enumerator,
#include "llvm/ProfileData/InstrProfData.inc"
};

Error InstrProfWriter::validateRecord(const InstrProfRecord &Func) {
  for (uint32_t VK = 0; VK <= IPVK_Last; VK++) {
    if (VK == IPVK_IndirectCallTarget || VK == IPVK_VTableTarget)
      continue;
    uint32_t NS = Func.getNumValueSites(VK);
    for (uint32_t S = 0; S < NS; S++) {
      DenseSet<uint64_t> SeenValues;
      for (const auto &V : Func.getValueArrayForSite(VK, S))
        if (!SeenValues.insert(V.Value).second)
          return make_error<InstrProfError>(instrprof_error::invalid_prof);
    }
  }

  return Error::success();
}

void InstrProfWriter::writeRecordInText(StringRef Name, uint64_t Hash,
                                        const InstrProfRecord &Func,
                                        InstrProfSymtab &Symtab,
                                        raw_fd_ostream &OS) {
  OS << Name << "\n";
  OS << "# Func Hash:\n" << Hash << "\n";
  OS << "# Num Counters:\n" << Func.Counts.size() << "\n";
  OS << "# Counter Values:\n";
  for (uint64_t Count : Func.Counts)
    OS << Count << "\n";

  if (Func.BitmapBytes.size() > 0) {
    OS << "# Num Bitmap Bytes:\n$" << Func.BitmapBytes.size() << "\n";
    OS << "# Bitmap Byte Values:\n";
    for (uint8_t Byte : Func.BitmapBytes) {
      OS << "0x";
      OS.write_hex(Byte);
      OS << "\n";
    }
    OS << "\n";
  }

  uint32_t NumValueKinds = Func.getNumValueKinds();
  if (!NumValueKinds) {
    OS << "\n";
    return;
  }

  OS << "# Num Value Kinds:\n" << Func.getNumValueKinds() << "\n";
  for (uint32_t VK = 0; VK < IPVK_Last + 1; VK++) {
    uint32_t NS = Func.getNumValueSites(VK);
    if (!NS)
      continue;
    OS << "# ValueKind = " << ValueProfKindStr[VK] << ":\n" << VK << "\n";
    OS << "# NumValueSites:\n" << NS << "\n";
    for (uint32_t S = 0; S < NS; S++) {
      auto VD = Func.getValueArrayForSite(VK, S);
      OS << VD.size() << "\n";
      for (const auto &V : VD) {
        if (VK == IPVK_IndirectCallTarget || VK == IPVK_VTableTarget)
          OS << Symtab.getFuncOrVarNameIfDefined(V.Value) << ":" << V.Count
             << "\n";
        else
          OS << V.Value << ":" << V.Count << "\n";
      }
    }
  }

  OS << "\n";
}

Error InstrProfWriter::writeText(raw_fd_ostream &OS) {
  // Check CS first since it implies an IR level profile.
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive))
    OS << "# CSIR level Instrumentation Flag\n:csir\n";
  else if (static_cast<bool>(ProfileKind & InstrProfKind::IRInstrumentation))
    OS << "# IR level Instrumentation Flag\n:ir\n";

  if (static_cast<bool>(ProfileKind &
                        InstrProfKind::FunctionEntryInstrumentation))
    OS << "# Always instrument the function entry block\n:entry_first\n";
  if (static_cast<bool>(ProfileKind &
                        InstrProfKind::LoopEntriesInstrumentation))
    OS << "# Always instrument the loop entry "
          "blocks\n:instrument_loop_entries\n";
  if (static_cast<bool>(ProfileKind & InstrProfKind::SingleByteCoverage))
    OS << "# Instrument block coverage\n:single_byte_coverage\n";
  InstrProfSymtab Symtab;

  using FuncPair = detail::DenseMapPair<uint64_t, InstrProfRecord>;
  using RecordType = std::pair<StringRef, FuncPair>;
  SmallVector<RecordType, 4> OrderedFuncData;

  for (const auto &I : FunctionData) {
    if (shouldEncodeData(I.getValue())) {
      if (Error E = Symtab.addFuncName(I.getKey()))
        return E;
      for (const auto &Func : I.getValue())
        OrderedFuncData.push_back(std::make_pair(I.getKey(), Func));
    }
  }

  for (const auto &VTableName : VTableNames)
    if (Error E = Symtab.addVTableName(VTableName.getKey()))
      return E;

  if (static_cast<bool>(ProfileKind & InstrProfKind::TemporalProfile))
    writeTextTemporalProfTraceData(OS, Symtab);

  llvm::sort(OrderedFuncData, [](const RecordType &A, const RecordType &B) {
    return std::tie(A.first, A.second.first) <
           std::tie(B.first, B.second.first);
  });

  for (const auto &record : OrderedFuncData) {
    const StringRef &Name = record.first;
    const FuncPair &Func = record.second;
    writeRecordInText(Name, Func.first, Func.second, Symtab, OS);
  }

  for (const auto &record : OrderedFuncData) {
    const FuncPair &Func = record.second;
    if (Error E = validateRecord(Func.second))
      return E;
  }

  return Error::success();
}

void InstrProfWriter::writeTextTemporalProfTraceData(raw_fd_ostream &OS,
                                                     InstrProfSymtab &Symtab) {
  OS << ":temporal_prof_traces\n";
  OS << "# Num Temporal Profile Traces:\n" << TemporalProfTraces.size() << "\n";
  OS << "# Temporal Profile Trace Stream Size:\n"
     << TemporalProfTraceStreamSize << "\n";
  for (auto &Trace : TemporalProfTraces) {
    OS << "# Weight:\n" << Trace.Weight << "\n";
    for (auto &NameRef : Trace.FunctionNameRefs)
      OS << Symtab.getFuncOrVarName(NameRef) << ",";
    OS << "\n";
  }
  OS << "\n";
}
