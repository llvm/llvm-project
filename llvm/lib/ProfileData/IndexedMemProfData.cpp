//===- IndexedMemProfData.h - MemProf format support ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MemProf data is serialized in writeMemProf provided in this file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/OnDiskHashTable.h"

namespace llvm {

// Serialize Schema.
static void writeMemProfSchema(ProfOStream &OS,
                               const memprof::MemProfSchema &Schema) {
  OS.write(static_cast<uint64_t>(Schema.size()));
  for (const auto Id : Schema)
    OS.write(static_cast<uint64_t>(Id));
}

// Serialize MemProfRecordData.  Return RecordTableOffset.
static uint64_t writeMemProfRecords(
    ProfOStream &OS,
    llvm::MapVector<GlobalValue::GUID, memprof::IndexedMemProfRecord>
        &MemProfRecordData,
    memprof::MemProfSchema *Schema, memprof::IndexedVersion Version,
    llvm::DenseMap<memprof::CallStackId, memprof::LinearCallStackId>
        *MemProfCallStackIndexes = nullptr) {
  memprof::RecordWriterTrait RecordWriter(Schema, Version,
                                          MemProfCallStackIndexes);
  OnDiskChainedHashTableGenerator<memprof::RecordWriterTrait>
      RecordTableGenerator;
  for (auto &[GUID, Record] : MemProfRecordData) {
    // Insert the key (func hash) and value (memprof record).
    RecordTableGenerator.insert(GUID, Record, RecordWriter);
  }
  // Release the memory of this MapVector as it is no longer needed.
  MemProfRecordData.clear();

  // The call to Emit invokes RecordWriterTrait::EmitData which destructs
  // the memprof record copies owned by the RecordTableGenerator. This works
  // because the RecordTableGenerator is not used after this point.
  return RecordTableGenerator.Emit(OS.OS, RecordWriter);
}

// Serialize MemProfFrameData.  Return FrameTableOffset.
static uint64_t writeMemProfFrames(
    ProfOStream &OS,
    llvm::MapVector<memprof::FrameId, memprof::Frame> &MemProfFrameData) {
  OnDiskChainedHashTableGenerator<memprof::FrameWriterTrait>
      FrameTableGenerator;
  for (auto &[FrameId, Frame] : MemProfFrameData) {
    // Insert the key (frame id) and value (frame contents).
    FrameTableGenerator.insert(FrameId, Frame);
  }
  // Release the memory of this MapVector as it is no longer needed.
  MemProfFrameData.clear();

  return FrameTableGenerator.Emit(OS.OS);
}

// Serialize MemProfFrameData.  Return the mapping from FrameIds to their
// indexes within the frame array.
static llvm::DenseMap<memprof::FrameId, memprof::LinearFrameId>
writeMemProfFrameArray(
    ProfOStream &OS,
    llvm::MapVector<memprof::FrameId, memprof::Frame> &MemProfFrameData,
    llvm::DenseMap<memprof::FrameId, memprof::FrameStat> &FrameHistogram) {
  // Mappings from FrameIds to array indexes.
  llvm::DenseMap<memprof::FrameId, memprof::LinearFrameId> MemProfFrameIndexes;

  // Compute the order in which we serialize Frames.  The order does not matter
  // in terms of correctness, but we still compute it for deserialization
  // performance.  Specifically, if we serialize frequently used Frames one
  // after another, we have better cache utilization.  For two Frames that
  // appear equally frequently, we break a tie by serializing the one that tends
  // to appear earlier in call stacks.  We implement the tie-breaking mechanism
  // by computing the sum of indexes within call stacks for each Frame.  If we
  // still have a tie, then we just resort to compare two FrameIds, which is
  // just for stability of output.
  std::vector<std::pair<memprof::FrameId, const memprof::Frame *>> FrameIdOrder;
  FrameIdOrder.reserve(MemProfFrameData.size());
  for (const auto &[Id, Frame] : MemProfFrameData)
    FrameIdOrder.emplace_back(Id, &Frame);
  assert(MemProfFrameData.size() == FrameIdOrder.size());
  llvm::sort(FrameIdOrder,
             [&](const std::pair<memprof::FrameId, const memprof::Frame *> &L,
                 const std::pair<memprof::FrameId, const memprof::Frame *> &R) {
               const auto &SL = FrameHistogram[L.first];
               const auto &SR = FrameHistogram[R.first];
               // Popular FrameIds should come first.
               if (SL.Count != SR.Count)
                 return SL.Count > SR.Count;
               // If they are equally popular, then the one that tends to appear
               // earlier in call stacks should come first.
               if (SL.PositionSum != SR.PositionSum)
                 return SL.PositionSum < SR.PositionSum;
               // Compare their FrameIds for sort stability.
               return L.first < R.first;
             });

  // Serialize all frames while creating mappings from linear IDs to FrameIds.
  uint64_t Index = 0;
  MemProfFrameIndexes.reserve(FrameIdOrder.size());
  for (const auto &[Id, F] : FrameIdOrder) {
    F->serialize(OS.OS);
    MemProfFrameIndexes.insert({Id, Index});
    ++Index;
  }
  assert(MemProfFrameData.size() == Index);
  assert(MemProfFrameData.size() == MemProfFrameIndexes.size());

  // Release the memory of this MapVector as it is no longer needed.
  MemProfFrameData.clear();

  return MemProfFrameIndexes;
}

static uint64_t writeMemProfCallStacks(
    ProfOStream &OS,
    llvm::MapVector<memprof::CallStackId, llvm::SmallVector<memprof::FrameId>>
        &MemProfCallStackData) {
  OnDiskChainedHashTableGenerator<memprof::CallStackWriterTrait>
      CallStackTableGenerator;
  for (auto &[CSId, CallStack] : MemProfCallStackData)
    CallStackTableGenerator.insert(CSId, CallStack);
  // Release the memory of this vector as it is no longer needed.
  MemProfCallStackData.clear();

  return CallStackTableGenerator.Emit(OS.OS);
}

static llvm::DenseMap<memprof::CallStackId, memprof::LinearCallStackId>
writeMemProfCallStackArray(
    ProfOStream &OS,
    llvm::MapVector<memprof::CallStackId, llvm::SmallVector<memprof::FrameId>>
        &MemProfCallStackData,
    llvm::DenseMap<memprof::FrameId, memprof::LinearFrameId>
        &MemProfFrameIndexes,
    llvm::DenseMap<memprof::FrameId, memprof::FrameStat> &FrameHistogram,
    unsigned &NumElements) {
  llvm::DenseMap<memprof::CallStackId, memprof::LinearCallStackId>
      MemProfCallStackIndexes;

  memprof::CallStackRadixTreeBuilder<memprof::FrameId> Builder;
  Builder.build(std::move(MemProfCallStackData), &MemProfFrameIndexes,
                FrameHistogram);
  for (auto I : Builder.getRadixArray())
    OS.write32(I);
  NumElements = Builder.getRadixArray().size();
  MemProfCallStackIndexes = Builder.takeCallStackPos();

  // Release the memory of this vector as it is no longer needed.
  MemProfCallStackData.clear();

  return MemProfCallStackIndexes;
}

// Write out MemProf Version2 as follows:
// uint64_t Version
// uint64_t RecordTableOffset = RecordTableGenerator.Emit
// uint64_t FramePayloadOffset = Offset for the frame payload
// uint64_t FrameTableOffset = FrameTableGenerator.Emit
// uint64_t CallStackPayloadOffset = Offset for the call stack payload (NEW V2)
// uint64_t CallStackTableOffset = CallStackTableGenerator.Emit (NEW in V2)
// uint64_t Num schema entries
// uint64_t Schema entry 0
// uint64_t Schema entry 1
// ....
// uint64_t Schema entry N - 1
// OnDiskChainedHashTable MemProfRecordData
// OnDiskChainedHashTable MemProfFrameData
// OnDiskChainedHashTable MemProfCallStackData (NEW in V2)
static Error writeMemProfV2(ProfOStream &OS,
                            memprof::IndexedMemProfData &MemProfData,
                            bool MemProfFullSchema) {
  OS.write(memprof::Version2);
  uint64_t HeaderUpdatePos = OS.tell();
  OS.write(0ULL); // Reserve space for the memprof record table offset.
  OS.write(0ULL); // Reserve space for the memprof frame payload offset.
  OS.write(0ULL); // Reserve space for the memprof frame table offset.
  OS.write(0ULL); // Reserve space for the memprof call stack payload offset.
  OS.write(0ULL); // Reserve space for the memprof call stack table offset.

  auto Schema = memprof::getHotColdSchema();
  if (MemProfFullSchema)
    Schema = memprof::getFullSchema();
  writeMemProfSchema(OS, Schema);

  uint64_t RecordTableOffset =
      writeMemProfRecords(OS, MemProfData.Records, &Schema, memprof::Version2);

  uint64_t FramePayloadOffset = OS.tell();
  uint64_t FrameTableOffset = writeMemProfFrames(OS, MemProfData.Frames);

  uint64_t CallStackPayloadOffset = OS.tell();
  uint64_t CallStackTableOffset =
      writeMemProfCallStacks(OS, MemProfData.CallStacks);

  uint64_t Header[] = {
      RecordTableOffset,      FramePayloadOffset,   FrameTableOffset,
      CallStackPayloadOffset, CallStackTableOffset,
  };
  OS.patch({{HeaderUpdatePos, Header}});

  return Error::success();
}

// Write out MemProf Version3 as follows:
// uint64_t Version
// uint64_t CallStackPayloadOffset = Offset for the call stack payload
// uint64_t RecordPayloadOffset = Offset for the record payload
// uint64_t RecordTableOffset = RecordTableGenerator.Emit
// uint64_t Num schema entries
// uint64_t Schema entry 0
// uint64_t Schema entry 1
// ....
// uint64_t Schema entry N - 1
// Frames serialized one after another
// Call stacks encoded as a radix tree
// OnDiskChainedHashTable MemProfRecordData
static Error writeMemProfV3(ProfOStream &OS,
                            memprof::IndexedMemProfData &MemProfData,
                            bool MemProfFullSchema) {
  OS.write(memprof::Version3);
  uint64_t HeaderUpdatePos = OS.tell();
  OS.write(0ULL); // Reserve space for the memprof call stack payload offset.
  OS.write(0ULL); // Reserve space for the memprof record payload offset.
  OS.write(0ULL); // Reserve space for the memprof record table offset.

  auto Schema = memprof::getHotColdSchema();
  if (MemProfFullSchema)
    Schema = memprof::getFullSchema();
  writeMemProfSchema(OS, Schema);

  llvm::DenseMap<memprof::FrameId, memprof::FrameStat> FrameHistogram =
      memprof::computeFrameHistogram(MemProfData.CallStacks);
  assert(MemProfData.Frames.size() == FrameHistogram.size());

  llvm::DenseMap<memprof::FrameId, memprof::LinearFrameId> MemProfFrameIndexes =
      writeMemProfFrameArray(OS, MemProfData.Frames, FrameHistogram);

  uint64_t CallStackPayloadOffset = OS.tell();
  // The number of elements in the call stack array.
  unsigned NumElements = 0;
  llvm::DenseMap<memprof::CallStackId, memprof::LinearCallStackId>
      MemProfCallStackIndexes =
          writeMemProfCallStackArray(OS, MemProfData.CallStacks,
                                     MemProfFrameIndexes, FrameHistogram,
                                     NumElements);

  uint64_t RecordPayloadOffset = OS.tell();
  uint64_t RecordTableOffset =
      writeMemProfRecords(OS, MemProfData.Records, &Schema, memprof::Version3,
                          &MemProfCallStackIndexes);

  // IndexedMemProfReader::deserializeV3 computes the number of elements in the
  // call stack array from the difference between CallStackPayloadOffset and
  // RecordPayloadOffset.  Verify that the computation works.
  assert(CallStackPayloadOffset +
             NumElements * sizeof(memprof::LinearFrameId) ==
         RecordPayloadOffset);

  uint64_t Header[] = {
      CallStackPayloadOffset,
      RecordPayloadOffset,
      RecordTableOffset,
  };
  OS.patch({{HeaderUpdatePos, Header}});

  return Error::success();
}

// Write out the MemProf data in a requested version.
Error writeMemProf(ProfOStream &OS, memprof::IndexedMemProfData &MemProfData,
                   memprof::IndexedVersion MemProfVersionRequested,
                   bool MemProfFullSchema) {
  switch (MemProfVersionRequested) {
  case memprof::Version2:
    return writeMemProfV2(OS, MemProfData, MemProfFullSchema);
  case memprof::Version3:
    return writeMemProfV3(OS, MemProfData, MemProfFullSchema);
  }

  return make_error<InstrProfError>(
      instrprof_error::unsupported_version,
      formatv("MemProf version {} not supported; "
              "requires version between {} and {}, inclusive",
              MemProfVersionRequested, memprof::MinimumSupportedVersion,
              memprof::MaximumSupportedVersion));
}

} // namespace llvm
