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
#include <tuple>

using namespace llvm;

// A struct to define how the data stream should be patched. For Indexed
// profiling, only uint64_t data type is needed.
struct PatchItem {
  uint64_t Pos; // Where to patch.
  uint64_t *D;  // Pointer to an array of source data.
  int N;        // Number of elements in \c D array.
};

namespace llvm {
// A wrapper class to abstract writer stream with support of bytes
// back patching.
class ProfOStream {

public:
  ProfOStream(llvm::raw_fd_ostream &FD) : IsFDOStream(true), OS(FD), LE(FD) {}
  ProfOStream(llvm::raw_string_ostream &STR)
      : IsFDOStream(false), OS(STR), LE(STR) {}

  uint64_t tell() { return OS.tell(); }
  void write(uint64_t V) { LE.write<uint64_t>(V); }
  // \c patch can only be called when all data is written and flushed.
  // For raw_string_ostream, the patch is done on the target string
  // directly and it won't be reflected in the stream's internal buffer.
  void patch(PatchItem *P, int NItems) {
    using namespace support;
    if (IsFDOStream) {
      llvm::raw_fd_ostream &FDOStream = static_cast<llvm::raw_fd_ostream &>(OS);
      for (int K = 0; K < NItems; K++) {
        FDOStream.seek(P[K].Pos);
        for (int I = 0; I < P[K].N; I++)
          write(P[K].D[I]);
      }
    } else {
      llvm::raw_string_ostream &SOStream =
          static_cast<llvm::raw_string_ostream &>(OS);
      std::string &Data = SOStream.str(); // with flush
      for (int K = 0; K < NItems; K++) {
        for (int I = 0; I < P[K].N; I++) {
          uint64_t Bytes = endian::byte_swap<uint64_t, little>(P[K].D[I]);
          Data.replace(P[K].Pos + I * sizeof(uint64_t), sizeof(uint64_t),
                       (const char *)&Bytes, sizeof(uint64_t));
        }
      }
    }
  }
  // If \c OS is an instance of \c raw_fd_ostream, this field will be
  // true. Otherwise, \c OS will be an raw_string_ostream.
  bool IsFDOStream;
  raw_ostream &OS;
  support::endian::Writer<support::little> LE;
};

class InstrProfRecordWriterTrait {
public:
  typedef StringRef key_type;
  typedef StringRef key_type_ref;

  typedef const InstrProfWriter::ProfilingData *const data_type;
  typedef const InstrProfWriter::ProfilingData *const data_type_ref;

  typedef uint64_t hash_value_type;
  typedef uint64_t offset_type;

  support::endianness ValueProfDataEndianness;
  ProfileSummary *TheProfileSummary;

  InstrProfRecordWriterTrait() : ValueProfDataEndianness(support::little) {}
  static hash_value_type ComputeHash(key_type_ref K) {
    return IndexedInstrProf::ComputeHash(K);
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
      M += ValueProfData::getSize(ProfileData.second);
    }
    LE.write<offset_type>(M);

    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type N) {
    Out.write(K.data(), N);
  }

  void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V, offset_type) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;
      TheProfileSummary->addRecord(ProfRecord);

      LE.write<uint64_t>(ProfileData.first); // Function hash
      LE.write<uint64_t>(ProfRecord.Counts.size());
      for (uint64_t I : ProfRecord.Counts)
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
}

InstrProfWriter::InstrProfWriter(bool Sparse)
    : Sparse(Sparse), FunctionData(),
      InfoObj(new InstrProfRecordWriterTrait()) {}

InstrProfWriter::~InstrProfWriter() { delete InfoObj; }

// Internal interface for testing purpose only.
void InstrProfWriter::setValueProfDataEndianness(
    support::endianness Endianness) {
  InfoObj->ValueProfDataEndianness = Endianness;
}
void InstrProfWriter::setOutputSparse(bool Sparse) {
  this->Sparse = Sparse;
}

std::error_code InstrProfWriter::addRecord(InstrProfRecord &&I,
                                           uint64_t Weight) {
  auto &ProfileDataMap = FunctionData[I.Name];

  bool NewFunc;
  ProfilingData::iterator Where;
  std::tie(Where, NewFunc) =
      ProfileDataMap.insert(std::make_pair(I.Hash, InstrProfRecord()));
  InstrProfRecord &Dest = Where->second;

  instrprof_error Result = instrprof_error::success;
  if (NewFunc) {
    // We've never seen a function with this name and hash, add it.
    Dest = std::move(I);
    // Fix up the name to avoid dangling reference.
    Dest.Name = FunctionData.find(Dest.Name)->getKey();
    if (Weight > 1)
      Result = Dest.scale(Weight);
  } else {
    // We're updating a function we've seen before.
    Result = Dest.merge(I, Weight);
  }

  Dest.sortValueData();

  return Result;
}

bool InstrProfWriter::shouldEncodeData(const ProfilingData &PD) {
  if (!Sparse)
    return true;
  for (const auto &Func : PD) {
    const InstrProfRecord &IPR = Func.second;
    if (std::any_of(IPR.Counts.begin(), IPR.Counts.end(),
                    [](uint64_t Count) { return Count > 0; }))
      return true;
  }
  return false;
}

static void setSummary(IndexedInstrProf::Summary *TheSummary,
                       ProfileSummary &PS) {
  using namespace IndexedInstrProf;
  std::vector<ProfileSummaryEntry> &Res = PS.getDetailedSummary();
  TheSummary->NumSummaryFields = Summary::NumKinds;
  TheSummary->NumCutoffEntries = Res.size();
  TheSummary->set(Summary::MaxFunctionCount, PS.getMaxFunctionCount());
  TheSummary->set(Summary::MaxBlockCount, PS.getMaxBlockCount());
  TheSummary->set(Summary::MaxInternalBlockCount,
                  PS.getMaxInternalBlockCount());
  TheSummary->set(Summary::TotalBlockCount, PS.getTotalCount());
  TheSummary->set(Summary::TotalNumBlocks, PS.getNumBlocks());
  TheSummary->set(Summary::TotalNumFunctions, PS.getNumFunctions());
  for (unsigned I = 0; I < Res.size(); I++)
    TheSummary->setEntry(I, Res[I]);
}

void InstrProfWriter::writeImpl(ProfOStream &OS) {
  OnDiskChainedHashTableGenerator<InstrProfRecordWriterTrait> Generator;

  using namespace IndexedInstrProf;
  std::vector<uint32_t> Cutoffs(&SummaryCutoffs[0],
                                &SummaryCutoffs[NumSummaryCutoffs]);
  ProfileSummary PS(Cutoffs);
  InfoObj->TheProfileSummary = &PS;

  // Populate the hash table generator.
  for (const auto &I : FunctionData)
    if (shouldEncodeData(I.getValue()))
      Generator.insert(I.getKey(), &I.getValue());
  // Write the header.
  IndexedInstrProf::Header Header;
  Header.Magic = IndexedInstrProf::Magic;
  Header.Version = IndexedInstrProf::ProfVersion::CurrentVersion;
  Header.Unused = 0;
  Header.HashType = static_cast<uint64_t>(IndexedInstrProf::HashType);
  Header.HashOffset = 0;
  int N = sizeof(IndexedInstrProf::Header) / sizeof(uint64_t);

  // Only write out all the fields except 'HashOffset'. We need
  // to remember the offset of that field to allow back patching
  // later.
  for (int I = 0; I < N - 1; I++)
    OS.write(reinterpret_cast<uint64_t *>(&Header)[I]);

  // Save the location of Header.HashOffset field in \c OS.
  uint64_t HashTableStartFieldOffset = OS.tell();
  // Reserve the space for HashOffset field.
  OS.write(0);

  // Reserve space to write profile summary data.
  uint32_t NumEntries = Cutoffs.size();
  uint32_t SummarySize = Summary::getSize(Summary::NumKinds, NumEntries);
  // Remember the summary offset.
  uint64_t SummaryOffset = OS.tell();
  for (unsigned I = 0; I < SummarySize / sizeof(uint64_t); I++)
    OS.write(0);

  // Write the hash table.
  uint64_t HashTableStart = Generator.Emit(OS.OS, *InfoObj);

  // Allocate space for data to be serialized out.
  std::unique_ptr<IndexedInstrProf::Summary> TheSummary =
      IndexedInstrProf::allocSummary(SummarySize);
  // Compute the Summary and copy the data to the data
  // structure to be serialized out (to disk or buffer).
  setSummary(TheSummary.get(), PS);
  InfoObj->TheProfileSummary = 0;

  // Now do the final patch:
  PatchItem PatchItems[] = {
      // Patch the Header.HashOffset field.
      {HashTableStartFieldOffset, &HashTableStart, 1},
      // Patch the summary data.
      {SummaryOffset, reinterpret_cast<uint64_t *>(TheSummary.get()),
       (int)(SummarySize / sizeof(uint64_t))}};
  OS.patch(PatchItems, sizeof(PatchItems) / sizeof(*PatchItems));
}

void InstrProfWriter::write(raw_fd_ostream &OS) {
  // Write the hash table.
  ProfOStream POS(OS);
  writeImpl(POS);
}

std::unique_ptr<MemoryBuffer> InstrProfWriter::writeBuffer() {
  std::string Data;
  llvm::raw_string_ostream OS(Data);
  ProfOStream POS(OS);
  // Write the hash table.
  writeImpl(POS);
  // Return this in an aligned memory buffer.
  return MemoryBuffer::getMemBufferCopy(Data);
}

static const char *ValueProfKindStr[] = {
#define VALUE_PROF_KIND(Enumerator, Value) #Enumerator,
#include "llvm/ProfileData/InstrProfData.inc"
};

void InstrProfWriter::writeRecordInText(const InstrProfRecord &Func,
                                        InstrProfSymtab &Symtab,
                                        raw_fd_ostream &OS) {
  OS << Func.Name << "\n";
  OS << "# Func Hash:\n" << Func.Hash << "\n";
  OS << "# Num Counters:\n" << Func.Counts.size() << "\n";
  OS << "# Counter Values:\n";
  for (uint64_t Count : Func.Counts)
    OS << Count << "\n";

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
      uint32_t ND = Func.getNumValueDataForSite(VK, S);
      OS << ND << "\n";
      std::unique_ptr<InstrProfValueData[]> VD = Func.getValueForSite(VK, S);
      for (uint32_t I = 0; I < ND; I++) {
        if (VK == IPVK_IndirectCallTarget)
          OS << Symtab.getFuncName(VD[I].Value) << ":" << VD[I].Count << "\n";
        else
          OS << VD[I].Value << ":" << VD[I].Count << "\n";
      }
    }
  }

  OS << "\n";
}

void InstrProfWriter::writeText(raw_fd_ostream &OS) {
  InstrProfSymtab Symtab;
  for (const auto &I : FunctionData)
    if (shouldEncodeData(I.getValue()))
      Symtab.addFuncName(I.getKey());
  Symtab.finalizeSymtab();

  for (const auto &I : FunctionData)
    if (shouldEncodeData(I.getValue()))
      for (const auto &Func : I.getValue())
        writeRecordInText(Func.second, Symtab, OS);
}
