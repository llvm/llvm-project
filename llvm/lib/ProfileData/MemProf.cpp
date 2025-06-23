#include "llvm/ProfileData/MemProf.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

namespace llvm {
namespace memprof {
MemProfSchema getFullSchema() {
  MemProfSchema List;
#define MIBEntryDef(NameTag, Name, Type) List.push_back(Meta::Name);
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  return List;
}

MemProfSchema getHotColdSchema() {
  return {Meta::AllocCount, Meta::TotalSize, Meta::TotalLifetime,
          Meta::TotalLifetimeAccessDensity};
}

static size_t serializedSizeV2(const IndexedAllocationInfo &IAI,
                               const MemProfSchema &Schema) {
  size_t Size = 0;
  // The CallStackId
  Size += sizeof(CallStackId);
  // The size of the payload.
  Size += PortableMemInfoBlock::serializedSize(Schema);
  return Size;
}

static size_t serializedSizeV3(const IndexedAllocationInfo &IAI,
                               const MemProfSchema &Schema) {
  size_t Size = 0;
  // The linear call stack ID.
  Size += sizeof(LinearCallStackId);
  // The size of the payload.
  Size += PortableMemInfoBlock::serializedSize(Schema);
  return Size;
}

size_t IndexedAllocationInfo::serializedSize(const MemProfSchema &Schema,
                                             IndexedVersion Version) const {
  switch (Version) {
  case Version2:
    return serializedSizeV2(*this, Schema);
  // Combine V3 and V4 as the size calculation is the same
  case Version3:
  case Version4:
    return serializedSizeV3(*this, Schema);
  }
  llvm_unreachable("unsupported MemProf version");
}

static size_t serializedSizeV2(const IndexedMemProfRecord &Record,
                               const MemProfSchema &Schema) {
  // The number of alloc sites to serialize.
  size_t Result = sizeof(uint64_t);
  for (const IndexedAllocationInfo &N : Record.AllocSites)
    Result += N.serializedSize(Schema, Version2);

  // The number of callsites we have information for.
  Result += sizeof(uint64_t);
  // The CallStackId
  Result += Record.CallSites.size() * sizeof(CallStackId);
  return Result;
}

static size_t serializedSizeV3(const IndexedMemProfRecord &Record,
                               const MemProfSchema &Schema) {
  // The number of alloc sites to serialize.
  size_t Result = sizeof(uint64_t);
  for (const IndexedAllocationInfo &N : Record.AllocSites)
    Result += N.serializedSize(Schema, Version3);

  // The number of callsites we have information for.
  Result += sizeof(uint64_t);
  // The linear call stack ID.
  // Note: V3 only stored the LinearCallStackId per call site.
  Result += Record.CallSites.size() * sizeof(LinearCallStackId);
  return Result;
}

static size_t serializedSizeV4(const IndexedMemProfRecord &Record,
                               const MemProfSchema &Schema) {
  // The number of alloc sites to serialize.
  size_t Result = sizeof(uint64_t);
  for (const IndexedAllocationInfo &N : Record.AllocSites)
    Result += N.serializedSize(Schema, Version4);

  // The number of callsites we have information for.
  Result += sizeof(uint64_t);
  for (const auto &CS : Record.CallSites)
    Result += sizeof(LinearCallStackId) + sizeof(uint64_t) +
              CS.CalleeGuids.size() * sizeof(GlobalValue::GUID);
  return Result;
}

size_t IndexedMemProfRecord::serializedSize(const MemProfSchema &Schema,
                                            IndexedVersion Version) const {
  switch (Version) {
  case Version2:
    return serializedSizeV2(*this, Schema);
  case Version3:
    return serializedSizeV3(*this, Schema);
  case Version4:
    return serializedSizeV4(*this, Schema);
  }
  llvm_unreachable("unsupported MemProf version");
}

static void serializeV2(const IndexedMemProfRecord &Record,
                        const MemProfSchema &Schema, raw_ostream &OS) {
  using namespace support;

  endian::Writer LE(OS, llvm::endianness::little);

  LE.write<uint64_t>(Record.AllocSites.size());
  for (const IndexedAllocationInfo &N : Record.AllocSites) {
    LE.write<CallStackId>(N.CSId);
    N.Info.serialize(Schema, OS);
  }

  // Related contexts.
  LE.write<uint64_t>(Record.CallSites.size());
  for (const auto &CS : Record.CallSites)
    LE.write<CallStackId>(CS.CSId);
}

static void serializeV3(
    const IndexedMemProfRecord &Record, const MemProfSchema &Schema,
    raw_ostream &OS,
    llvm::DenseMap<CallStackId, LinearCallStackId> &MemProfCallStackIndexes) {
  using namespace support;

  endian::Writer LE(OS, llvm::endianness::little);

  LE.write<uint64_t>(Record.AllocSites.size());
  for (const IndexedAllocationInfo &N : Record.AllocSites) {
    assert(MemProfCallStackIndexes.contains(N.CSId));
    LE.write<LinearCallStackId>(MemProfCallStackIndexes[N.CSId]);
    N.Info.serialize(Schema, OS);
  }

  // Related contexts.
  LE.write<uint64_t>(Record.CallSites.size());
  for (const auto &CS : Record.CallSites) {
    assert(MemProfCallStackIndexes.contains(CS.CSId));
    LE.write<LinearCallStackId>(MemProfCallStackIndexes[CS.CSId]);
  }
}

static void serializeV4(
    const IndexedMemProfRecord &Record, const MemProfSchema &Schema,
    raw_ostream &OS,
    llvm::DenseMap<CallStackId, LinearCallStackId> &MemProfCallStackIndexes) {
  using namespace support;

  endian::Writer LE(OS, llvm::endianness::little);

  LE.write<uint64_t>(Record.AllocSites.size());
  for (const IndexedAllocationInfo &N : Record.AllocSites) {
    assert(MemProfCallStackIndexes.contains(N.CSId));
    LE.write<LinearCallStackId>(MemProfCallStackIndexes[N.CSId]);
    N.Info.serialize(Schema, OS);
  }

  // Related contexts.
  LE.write<uint64_t>(Record.CallSites.size());
  for (const auto &CS : Record.CallSites) {
    assert(MemProfCallStackIndexes.contains(CS.CSId));
    LE.write<LinearCallStackId>(MemProfCallStackIndexes[CS.CSId]);
    LE.write<uint64_t>(CS.CalleeGuids.size());
    for (const auto &Guid : CS.CalleeGuids)
      LE.write<GlobalValue::GUID>(Guid);
  }
}

void IndexedMemProfRecord::serialize(
    const MemProfSchema &Schema, raw_ostream &OS, IndexedVersion Version,
    llvm::DenseMap<CallStackId, LinearCallStackId> *MemProfCallStackIndexes)
    const {
  switch (Version) {
  case Version2:
    serializeV2(*this, Schema, OS);
    return;
  case Version3:
    serializeV3(*this, Schema, OS, *MemProfCallStackIndexes);
    return;
  case Version4:
    serializeV4(*this, Schema, OS, *MemProfCallStackIndexes);
    return;
  }
  llvm_unreachable("unsupported MemProf version");
}

static IndexedMemProfRecord deserializeV2(const MemProfSchema &Schema,
                                          const unsigned char *Ptr) {
  using namespace support;

  IndexedMemProfRecord Record;

  // Read the meminfo nodes.
  const uint64_t NumNodes =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Record.AllocSites.reserve(NumNodes);
  for (uint64_t I = 0; I < NumNodes; I++) {
    IndexedAllocationInfo Node;
    Node.CSId = endian::readNext<CallStackId, llvm::endianness::little>(Ptr);
    Node.Info.deserialize(Schema, Ptr);
    Ptr += PortableMemInfoBlock::serializedSize(Schema);
    Record.AllocSites.push_back(Node);
  }

  // Read the callsite information.
  const uint64_t NumCtxs =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Record.CallSites.reserve(NumCtxs);
  for (uint64_t J = 0; J < NumCtxs; J++) {
    CallStackId CSId =
        endian::readNext<CallStackId, llvm::endianness::little>(Ptr);
    Record.CallSites.emplace_back(CSId);
  }

  return Record;
}

static IndexedMemProfRecord deserializeV3(const MemProfSchema &Schema,
                                          const unsigned char *Ptr) {
  using namespace support;

  IndexedMemProfRecord Record;

  // Read the meminfo nodes.
  const uint64_t NumNodes =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Record.AllocSites.reserve(NumNodes);
  const size_t SerializedSize = PortableMemInfoBlock::serializedSize(Schema);
  for (uint64_t I = 0; I < NumNodes; I++) {
    IndexedAllocationInfo Node;
    Node.CSId =
        endian::readNext<LinearCallStackId, llvm::endianness::little>(Ptr);
    Node.Info.deserialize(Schema, Ptr);
    Ptr += SerializedSize;
    Record.AllocSites.push_back(Node);
  }

  // Read the callsite information.
  const uint64_t NumCtxs =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Record.CallSites.reserve(NumCtxs);
  for (uint64_t J = 0; J < NumCtxs; J++) {
    // We are storing LinearCallStackId in CallSiteIds, which is a vector of
    // CallStackId.  Assert that CallStackId is no smaller than
    // LinearCallStackId.
    static_assert(sizeof(LinearCallStackId) <= sizeof(CallStackId));
    LinearCallStackId CSId =
        endian::readNext<LinearCallStackId, llvm::endianness::little>(Ptr);
    Record.CallSites.emplace_back(CSId);
  }

  return Record;
}

static IndexedMemProfRecord deserializeV4(const MemProfSchema &Schema,
                                          const unsigned char *Ptr) {
  using namespace support;

  IndexedMemProfRecord Record;

  // Read the meminfo nodes.
  const uint64_t NumNodes =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Record.AllocSites.reserve(NumNodes);
  const size_t SerializedSize = PortableMemInfoBlock::serializedSize(Schema);
  for (uint64_t I = 0; I < NumNodes; I++) {
    IndexedAllocationInfo Node;
    Node.CSId =
        endian::readNext<LinearCallStackId, llvm::endianness::little>(Ptr);
    Node.Info.deserialize(Schema, Ptr);
    Ptr += SerializedSize;
    Record.AllocSites.push_back(Node);
  }

  // Read the callsite information.
  const uint64_t NumCtxs =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Record.CallSites.reserve(NumCtxs);
  for (uint64_t J = 0; J < NumCtxs; J++) {
    static_assert(sizeof(LinearCallStackId) <= sizeof(CallStackId));
    LinearCallStackId CSId =
        endian::readNext<LinearCallStackId, llvm::endianness::little>(Ptr);
    const uint64_t NumGuids =
        endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    SmallVector<GlobalValue::GUID, 1> Guids;
    Guids.reserve(NumGuids);
    for (uint64_t K = 0; K < NumGuids; ++K)
      Guids.push_back(
          endian::readNext<GlobalValue::GUID, llvm::endianness::little>(Ptr));
    Record.CallSites.emplace_back(CSId, std::move(Guids));
  }

  return Record;
}

IndexedMemProfRecord
IndexedMemProfRecord::deserialize(const MemProfSchema &Schema,
                                  const unsigned char *Ptr,
                                  IndexedVersion Version) {
  switch (Version) {
  case Version2:
    return deserializeV2(Schema, Ptr);
  case Version3:
    return deserializeV3(Schema, Ptr);
  case Version4:
    return deserializeV4(Schema, Ptr);
  }
  llvm_unreachable("unsupported MemProf version");
}

MemProfRecord IndexedMemProfRecord::toMemProfRecord(
    llvm::function_ref<std::vector<Frame>(const CallStackId)> Callback) const {
  MemProfRecord Record;

  Record.AllocSites.reserve(AllocSites.size());
  for (const IndexedAllocationInfo &IndexedAI : AllocSites) {
    AllocationInfo AI;
    AI.Info = IndexedAI.Info;
    AI.CallStack = Callback(IndexedAI.CSId);
    Record.AllocSites.push_back(std::move(AI));
  }

  Record.CallSites.reserve(CallSites.size());
  for (const IndexedCallSiteInfo &CS : CallSites) {
    std::vector<Frame> Frames = Callback(CS.CSId);
    Record.CallSites.emplace_back(std::move(Frames), CS.CalleeGuids);
  }

  return Record;
}

GlobalValue::GUID getGUID(const StringRef FunctionName) {
  // Canonicalize the function name to drop suffixes such as ".llvm.". Note
  // we do not drop any ".__uniq." suffixes, as getCanonicalFnName does not drop
  // those by default. This is by design to differentiate internal linkage
  // functions during matching. By dropping the other suffixes we can then match
  // functions in the profile use phase prior to their addition. Note that this
  // applies to both instrumented and sampled function names.
  StringRef CanonicalName =
      sampleprof::FunctionSamples::getCanonicalFnName(FunctionName);

  // We use the function guid which we expect to be a uint64_t. At
  // this time, it is the lower 64 bits of the md5 of the canonical
  // function name.
  return Function::getGUIDAssumingExternalLinkage(CanonicalName);
}

Expected<MemProfSchema> readMemProfSchema(const unsigned char *&Buffer) {
  using namespace support;

  const unsigned char *Ptr = Buffer;
  const uint64_t NumSchemaIds =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  if (NumSchemaIds > static_cast<uint64_t>(Meta::Size)) {
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "memprof schema invalid");
  }

  MemProfSchema Result;
  for (size_t I = 0; I < NumSchemaIds; I++) {
    const uint64_t Tag =
        endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    if (Tag >= static_cast<uint64_t>(Meta::Size)) {
      return make_error<InstrProfError>(instrprof_error::malformed,
                                        "memprof schema invalid");
    }
    Result.push_back(static_cast<Meta>(Tag));
  }
  // Advance the buffer to one past the schema if we succeeded.
  Buffer = Ptr;
  return Result;
}
} // namespace memprof
} // namespace llvm
