//=-- InstrProfWriter.cpp - Instrumented profiling writer -------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/OnDiskHashTable.h"

using namespace llvm;

namespace {
class InstrProfRecordTrait {
public:
  typedef StringRef key_type;
  typedef StringRef key_type_ref;

  typedef const InstrProfWriter::ProfilingData *const data_type;
  typedef const InstrProfWriter::ProfilingData *const data_type_ref;

  typedef uint64_t hash_value_type;
  typedef uint64_t offset_type;

  static hash_value_type ComputeHash(key_type_ref K) {
    return IndexedInstrProf::ComputeHash(IndexedInstrProf::HashType, K);
  }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);

    offset_type N = K.size();
    LE.write<offset_type>(N);

    offset_type M = 0;
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;
      M += sizeof(uint64_t); // The function hash
      M += sizeof(uint64_t); // The size of the Counts vector
      M += ProfRecord.Counts.size() * sizeof(uint64_t);

      // Value data
      M += sizeof(uint64_t); // Number of value kinds with value sites.
      for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind) {
        uint32_t NumValueSites = ProfRecord.getNumValueSites(Kind);
        if (NumValueSites == 0)
          continue;
        M += sizeof(uint64_t); // Value kind
        M += sizeof(uint64_t); // The number of value sites for given value kind
        for (uint32_t I = 0; I < NumValueSites; I++) {
          M += sizeof(uint64_t); // Number of value data pairs at a value site
          uint64_t NumValueDataForSite =
              ProfRecord.getNumValueDataForSite(Kind, I);
          M += 2 * sizeof(uint64_t) * NumValueDataForSite; // Value data pairs
        }
      }
    }
    LE.write<offset_type>(M);

    return std::make_pair(N, M);
  }

  static void EmitKey(raw_ostream &Out, key_type_ref K, offset_type N){
    Out.write(K.data(), N);
  }

  static void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V,
                       offset_type) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;

      LE.write<uint64_t>(ProfileData.first); // Function hash
      LE.write<uint64_t>(ProfRecord.Counts.size());
      for (uint64_t I : ProfRecord.Counts)
        LE.write<uint64_t>(I);

      // Compute the number of value kinds with value sites.
      uint64_t NumValueKinds = ProfRecord.getNumValueKinds();
      LE.write<uint64_t>(NumValueKinds);

      // Write value data
      for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind) {
        uint32_t NumValueSites = ProfRecord.getNumValueSites(Kind);
        if (NumValueSites == 0)
          continue;
        LE.write<uint64_t>(Kind); // Write value kind
        // Write number of value sites for current value kind
        LE.write<uint64_t>(NumValueSites);

        for (uint32_t I = 0; I < NumValueSites; I++) {
          // Write number of value data pairs at this value site
          uint64_t NumValueDataForSite =
              ProfRecord.getNumValueDataForSite(Kind, I);
          LE.write<uint64_t>(NumValueDataForSite);
          std::unique_ptr<InstrProfValueData[]> VD =
              ProfRecord.getValueForSite(Kind, I);

          for (uint32_t V = 0; V < NumValueDataForSite; V++) {
            if (Kind == IPVK_IndirectCallTarget)
              LE.write<uint64_t>(ComputeHash((const char *)VD[V].Value));
            else
              LE.write<uint64_t>(VD[V].Value);
            LE.write<uint64_t>(VD[V].Count);
          }
        }
      }
    }
  }
};
}

static std::error_code combineInstrProfRecords(InstrProfRecord &Dest,
                                               InstrProfRecord &Source,
                                               uint64_t &MaxFunctionCount) {
  // If the number of counters doesn't match we either have bad data
  // or a hash collision.
  if (Dest.Counts.size() != Source.Counts.size())
    return instrprof_error::count_mismatch;

  for (size_t I = 0, E = Source.Counts.size(); I < E; ++I) {
    if (Dest.Counts[I] + Source.Counts[I] < Dest.Counts[I])
      return instrprof_error::counter_overflow;
    Dest.Counts[I] += Source.Counts[I];
  }

  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind) {
    if (std::error_code EC = Dest.mergeValueProfData(Kind, Source))
      return EC;
  }

  // We keep track of the max function count as we go for simplicity.
  if (Dest.Counts[0] > MaxFunctionCount)
    MaxFunctionCount = Dest.Counts[0];

  return instrprof_error::success;
}

void InstrProfWriter::updateStringTableReferences(InstrProfRecord &I) {
  I.updateStrings(&StringTable);
}

std::error_code InstrProfWriter::addRecord(InstrProfRecord &&I) {
  updateStringTableReferences(I);
  auto &ProfileDataMap = FunctionData[I.Name];

  auto Where = ProfileDataMap.find(I.Hash);
  if (Where == ProfileDataMap.end()) {
    // We've never seen a function with this name and hash, add it.
    ProfileDataMap[I.Hash] = I;

    // We keep track of the max function count as we go for simplicity.
    if (I.Counts[0] > MaxFunctionCount)
      MaxFunctionCount = I.Counts[0];
    return instrprof_error::success;
  }

  // We're updating a function we've seen before.
  return combineInstrProfRecords(Where->second, I, MaxFunctionCount);
}

std::pair<uint64_t, uint64_t> InstrProfWriter::writeImpl(raw_ostream &OS) {
  OnDiskChainedHashTableGenerator<InstrProfRecordTrait> Generator;

  // Populate the hash table generator.
  for (const auto &I : FunctionData)
    Generator.insert(I.getKey(), &I.getValue());

  using namespace llvm::support;
  endian::Writer<little> LE(OS);

  // Write the header.
  IndexedInstrProf::Header Header;
  Header.Magic = IndexedInstrProf::Magic;
  Header.Version = IndexedInstrProf::Version;
  Header.MaxFunctionCount = MaxFunctionCount;
  Header.HashType = static_cast<uint64_t>(IndexedInstrProf::HashType);
  Header.HashOffset = 0;
  int N = sizeof(IndexedInstrProf::Header) / sizeof(uint64_t);

  // Only write out all the fields execpt 'HashOffset'. We need
  // to remember the offset of that field to allow back patching
  // later.
  for (int I = 0; I < N - 1; I++)
    LE.write<uint64_t>(reinterpret_cast<uint64_t *>(&Header)[I]);

  // Save a space to write the hash table start location.
  uint64_t HashTableStartLoc = OS.tell();
  // Reserve the space for HashOffset field.
  LE.write<uint64_t>(0);
  // Write the hash table.
  uint64_t HashTableStart = Generator.Emit(OS);

  return std::make_pair(HashTableStartLoc, HashTableStart);
}

void InstrProfWriter::write(raw_fd_ostream &OS) {
  // Write the hash table.
  auto TableStart = writeImpl(OS);

  // Go back and fill in the hash table start.
  using namespace support;
  OS.seek(TableStart.first);
  // Now patch the HashOffset field previously reserved.
  endian::Writer<little>(OS).write<uint64_t>(TableStart.second);
}

std::unique_ptr<MemoryBuffer> InstrProfWriter::writeBuffer() {
  std::string Data;
  llvm::raw_string_ostream OS(Data);
  // Write the hash table.
  auto TableStart = writeImpl(OS);
  OS.flush();

  // Go back and fill in the hash table start.
  using namespace support;
  uint64_t Bytes = endian::byte_swap<uint64_t, little>(TableStart.second);
  Data.replace(TableStart.first, sizeof(uint64_t), (const char *)&Bytes,
               sizeof(uint64_t));

  // Return this in an aligned memory buffer.
  return MemoryBuffer::getMemBufferCopy(Data);
}
