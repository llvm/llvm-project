//===- MemProf.h - MemProf support ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in the reading and writing of
// memory profile data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_MEMPROF_H
#define LLVM_PROFILEDATA_MEMPROF_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

#include <bitset>
#include <cstdint>

namespace llvm {
namespace yaml {
template <typename T> struct CustomMappingTraits;
} // namespace yaml

namespace memprof {

struct MemProfRecord;

// The versions of the indexed MemProf format
enum IndexedVersion : uint64_t {
  // Version 2: Added a call stack table.
  Version2 = 2,
  // Version 3: Added a radix tree for call stacks.  Switched to linear IDs for
  // frames and call stacks.
  Version3 = 3,
  // Version 4: Added CalleeGuids to call site info.
  Version4 = 4,
};

constexpr uint64_t MinimumSupportedVersion = Version2;
constexpr uint64_t MaximumSupportedVersion = Version4;

// Verify that the minimum and maximum satisfy the obvious constraint.
static_assert(MinimumSupportedVersion <= MaximumSupportedVersion);

inline llvm::StringRef getMemprofOptionsSymbolDarwinLinkageName() {
  return "___memprof_default_options_str";
}

inline llvm::StringRef getMemprofOptionsSymbolName() {
  // Darwin linkage names are prefixed with an extra "_". See
  // DataLayout::getGlobalPrefix().
  return getMemprofOptionsSymbolDarwinLinkageName().drop_front();
}

enum class Meta : uint64_t {
  Start = 0,
#define MIBEntryDef(NameTag, Name, Type) NameTag,
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  Size
};

using MemProfSchema = llvm::SmallVector<Meta, static_cast<int>(Meta::Size)>;

// Returns the full schema currently in use.
LLVM_ABI MemProfSchema getFullSchema();

// Returns the schema consisting of the fields used for hot cold memory hinting.
LLVM_ABI MemProfSchema getHotColdSchema();

// Holds the actual MemInfoBlock data with all fields. Contents may be read or
// written partially by providing an appropriate schema to the serialize and
// deserialize methods.
struct PortableMemInfoBlock {
  PortableMemInfoBlock() = default;
  explicit PortableMemInfoBlock(const MemInfoBlock &Block,
                                const MemProfSchema &IncomingSchema) {
    for (const Meta Id : IncomingSchema)
      Schema.set(llvm::to_underlying(Id));
#define MIBEntryDef(NameTag, Name, Type) Name = Block.Name;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  }

  PortableMemInfoBlock(const MemProfSchema &Schema, const unsigned char *Ptr) {
    deserialize(Schema, Ptr);
  }

  // Read the contents of \p Ptr based on the \p Schema to populate the
  // MemInfoBlock member.
  void deserialize(const MemProfSchema &IncomingSchema,
                   const unsigned char *Ptr) {
    using namespace support;

    Schema.reset();
    for (const Meta Id : IncomingSchema) {
      switch (Id) {
#define MIBEntryDef(NameTag, Name, Type)                                       \
  case Meta::Name: {                                                           \
    Name = endian::readNext<Type, llvm::endianness::little>(Ptr);              \
  } break;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
      default:
        llvm_unreachable("Unknown meta type id, is the profile collected from "
                         "a newer version of the runtime?");
      }

      Schema.set(llvm::to_underlying(Id));
    }
  }

  // Write the contents of the MemInfoBlock based on the \p Schema provided to
  // the raw_ostream \p OS.
  void serialize(const MemProfSchema &Schema, raw_ostream &OS) const {
    using namespace support;

    endian::Writer LE(OS, llvm::endianness::little);
    for (const Meta Id : Schema) {
      switch (Id) {
#define MIBEntryDef(NameTag, Name, Type)                                       \
  case Meta::Name: {                                                           \
    LE.write<Type>(Name);                                                      \
  } break;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
      default:
        llvm_unreachable("Unknown meta type id, invalid input?");
      }
    }
  }

  // Print out the contents of the MemInfoBlock in YAML format.
  void printYAML(raw_ostream &OS) const {
    OS << "      MemInfoBlock:\n";
#define MIBEntryDef(NameTag, Name, Type)                                       \
  OS << "        " << #Name << ": " << Name << "\n";
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
    if (AccessHistogramSize > 0) {
      OS << "        " << "AccessHistogramValues" << ":";
      for (uint32_t I = 0; I < AccessHistogramSize; ++I) {
        OS << " " << ((uint64_t *)AccessHistogram)[I];
      }
      OS << "\n";
    }
  }

  // Return the schema, only for unit tests.
  std::bitset<llvm::to_underlying(Meta::Size)> getSchema() const {
    return Schema;
  }

  // Define getters for each type which can be called by analyses.
#define MIBEntryDef(NameTag, Name, Type)                                       \
  Type get##Name() const {                                                     \
    assert(Schema[llvm::to_underlying(Meta::Name)]);                           \
    return Name;                                                               \
  }
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef

  // Define setters for each type which can be called by the writer.
#define MIBEntryDef(NameTag, Name, Type)                                       \
  void set##Name(Type NewVal) {                                                \
    assert(Schema[llvm::to_underlying(Meta::Name)]);                           \
    Name = NewVal;                                                             \
  }
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef

  void clear() { *this = PortableMemInfoBlock(); }

  bool operator==(const PortableMemInfoBlock &Other) const {
    if (Other.Schema != Schema)
      return false;

#define MIBEntryDef(NameTag, Name, Type)                                       \
  if (Schema[llvm::to_underlying(Meta::Name)] &&                               \
      Other.get##Name() != get##Name())                                        \
    return false;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
    return true;
  }

  bool operator!=(const PortableMemInfoBlock &Other) const {
    return !operator==(Other);
  }

  static size_t serializedSize(const MemProfSchema &Schema) {
    size_t Result = 0;

    for (const Meta Id : Schema) {
      switch (Id) {
#define MIBEntryDef(NameTag, Name, Type)                                       \
  case Meta::Name: {                                                           \
    Result += sizeof(Type);                                                    \
  } break;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
      default:
        llvm_unreachable("Unknown meta type id, invalid input?");
      }
    }

    return Result;
  }

  // Give YAML access to the individual MIB fields.
  friend struct yaml::CustomMappingTraits<memprof::PortableMemInfoBlock>;

private:
  // The set of available fields, indexed by Meta::Name.
  std::bitset<llvm::to_underlying(Meta::Size)> Schema;

#define MIBEntryDef(NameTag, Name, Type) Type Name = Type();
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
};

// A type representing the id generated by hashing the contents of the Frame.
using FrameId = uint64_t;
// A type representing the id to index into the frame array.
using LinearFrameId = uint32_t;
// Describes a call frame for a dynamic allocation context. The contents of
// the frame are populated by symbolizing the stack depot call frame from the
// compiler runtime.
struct Frame {
  // A uuid (uint64_t) identifying the function. It is obtained by
  // llvm::md5(FunctionName) which returns the lower 64 bits.
  GlobalValue::GUID Function = 0;
  // The symbol name for the function. Only populated in the Frame by the reader
  // if requested during initialization. This field should not be serialized.
  std::unique_ptr<std::string> SymbolName;
  // The source line offset of the call from the beginning of parent function.
  uint32_t LineOffset = 0;
  // The source column number of the call to help distinguish multiple calls
  // on the same line.
  uint32_t Column = 0;
  // Whether the current frame is inlined.
  bool IsInlineFrame = false;

  Frame() = default;
  Frame(const Frame &Other) {
    Function = Other.Function;
    SymbolName = Other.SymbolName
                     ? std::make_unique<std::string>(*Other.SymbolName)
                     : nullptr;
    LineOffset = Other.LineOffset;
    Column = Other.Column;
    IsInlineFrame = Other.IsInlineFrame;
  }

  Frame(GlobalValue::GUID Hash, uint32_t Off, uint32_t Col, bool Inline)
      : Function(Hash), LineOffset(Off), Column(Col), IsInlineFrame(Inline) {}

  bool operator==(const Frame &Other) const {
    // Ignore the SymbolName field to avoid a string compare. Comparing the
    // function hash serves the same purpose.
    return Other.Function == Function && Other.LineOffset == LineOffset &&
           Other.Column == Column && Other.IsInlineFrame == IsInlineFrame;
  }

  Frame &operator=(const Frame &Other) {
    Function = Other.Function;
    SymbolName = Other.SymbolName
                     ? std::make_unique<std::string>(*Other.SymbolName)
                     : nullptr;
    LineOffset = Other.LineOffset;
    Column = Other.Column;
    IsInlineFrame = Other.IsInlineFrame;
    return *this;
  }

  bool operator!=(const Frame &Other) const { return !operator==(Other); }

  bool hasSymbolName() const { return !!SymbolName; }

  StringRef getSymbolName() const {
    assert(hasSymbolName());
    return *SymbolName;
  }

  std::string getSymbolNameOr(StringRef Alt) const {
    return std::string(hasSymbolName() ? getSymbolName() : Alt);
  }

  // Write the contents of the frame to the ostream \p OS.
  void serialize(raw_ostream &OS) const {
    using namespace support;

    endian::Writer LE(OS, llvm::endianness::little);

    // If the type of the GlobalValue::GUID changes, then we need to update
    // the reader and the writer.
    static_assert(std::is_same<GlobalValue::GUID, uint64_t>::value,
                  "Expect GUID to be uint64_t.");
    LE.write<uint64_t>(Function);

    LE.write<uint32_t>(LineOffset);
    LE.write<uint32_t>(Column);
    LE.write<bool>(IsInlineFrame);
  }

  // Read a frame from char data which has been serialized as little endian.
  static Frame deserialize(const unsigned char *Ptr) {
    using namespace support;

    const uint64_t F =
        endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    const uint32_t L =
        endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
    const uint32_t C =
        endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
    const bool I = endian::readNext<bool, llvm::endianness::little>(Ptr);
    return Frame(/*Function=*/F, /*LineOffset=*/L, /*Column=*/C,
                 /*IsInlineFrame=*/I);
  }

  // Returns the size of the frame information.
  static constexpr size_t serializedSize() {
    return sizeof(Frame::Function) + sizeof(Frame::LineOffset) +
           sizeof(Frame::Column) + sizeof(Frame::IsInlineFrame);
  }

  // Print the frame information in YAML format.
  void printYAML(raw_ostream &OS) const {
    OS << "      -\n"
       << "        Function: " << Function << "\n"
       << "        SymbolName: " << getSymbolNameOr("<None>") << "\n"
       << "        LineOffset: " << LineOffset << "\n"
       << "        Column: " << Column << "\n"
       << "        Inline: " << IsInlineFrame << "\n";
  }
};

// A type representing the index into the table of call stacks.
using CallStackId = uint64_t;

// A type representing the index into the call stack array.
using LinearCallStackId = uint32_t;

// Holds call site information with indexed frame contents.
struct IndexedCallSiteInfo {
  // The call stack ID for this call site
  CallStackId CSId = 0;
  // The GUIDs of the callees at this call site
  SmallVector<GlobalValue::GUID, 1> CalleeGuids;

  IndexedCallSiteInfo() = default;
  IndexedCallSiteInfo(CallStackId CSId) : CSId(CSId) {}
  IndexedCallSiteInfo(CallStackId CSId,
                      SmallVector<GlobalValue::GUID, 1> CalleeGuids)
      : CSId(CSId), CalleeGuids(std::move(CalleeGuids)) {}

  bool operator==(const IndexedCallSiteInfo &Other) const {
    return CSId == Other.CSId && CalleeGuids == Other.CalleeGuids;
  }

  bool operator!=(const IndexedCallSiteInfo &Other) const {
    return !operator==(Other);
  }
};

// Holds allocation information in a space efficient format where frames are
// represented using unique identifiers.
struct IndexedAllocationInfo {
  // The dynamic calling context for the allocation in bottom-up (leaf-to-root)
  // order. Frame contents are stored out-of-line.
  CallStackId CSId = 0;
  // The statistics obtained from the runtime for the allocation.
  PortableMemInfoBlock Info;

  IndexedAllocationInfo() = default;
  IndexedAllocationInfo(CallStackId CSId, const MemInfoBlock &MB,
                        const MemProfSchema &Schema = getFullSchema())
      : CSId(CSId), Info(MB, Schema) {}
  IndexedAllocationInfo(CallStackId CSId, const PortableMemInfoBlock &MB)
      : CSId(CSId), Info(MB) {}

  // Returns the size in bytes when this allocation info struct is serialized.
  LLVM_ABI size_t serializedSize(const MemProfSchema &Schema,
                                 IndexedVersion Version) const;

  bool operator==(const IndexedAllocationInfo &Other) const {
    if (Other.Info != Info)
      return false;

    if (Other.CSId != CSId)
      return false;
    return true;
  }

  bool operator!=(const IndexedAllocationInfo &Other) const {
    return !operator==(Other);
  }
};

// Holds allocation information with frame contents inline. The type should
// be used for temporary in-memory instances.
struct AllocationInfo {
  // Same as IndexedAllocationInfo::CallStack with the frame contents inline.
  std::vector<Frame> CallStack;
  // Same as IndexedAllocationInfo::Info;
  PortableMemInfoBlock Info;

  AllocationInfo() = default;

  void printYAML(raw_ostream &OS) const {
    OS << "    -\n";
    OS << "      Callstack:\n";
    // TODO: Print out the frame on one line with to make it easier for deep
    // callstacks once we have a test to check valid YAML is generated.
    for (const Frame &F : CallStack) {
      F.printYAML(OS);
    }
    Info.printYAML(OS);
  }
};

// Holds the memprof profile information for a function. The internal
// representation stores frame ids for efficiency. This representation should
// be used in the profile conversion and manipulation tools.
struct IndexedMemProfRecord {
  // Memory allocation sites in this function for which we have memory
  // profiling data.
  llvm::SmallVector<IndexedAllocationInfo> AllocSites;
  // Holds call sites in this function which are part of some memory
  // allocation context. We store this as a list of locations, each with its
  // list of inline locations in bottom-up order i.e. from leaf to root. The
  // inline location list may include additional entries, users should pick
  // the last entry in the list with the same function GUID.
  llvm::SmallVector<IndexedCallSiteInfo> CallSites;

  void clear() { *this = IndexedMemProfRecord(); }

  void merge(const IndexedMemProfRecord &Other) {
    // TODO: Filter out duplicates which may occur if multiple memprof
    // profiles are merged together using llvm-profdata.
    AllocSites.append(Other.AllocSites);
  }

  LLVM_ABI size_t serializedSize(const MemProfSchema &Schema,
                                 IndexedVersion Version) const;

  bool operator==(const IndexedMemProfRecord &Other) const {
    if (Other.AllocSites != AllocSites)
      return false;

    if (Other.CallSites != CallSites)
      return false;
    return true;
  }

  // Serializes the memprof records in \p Records to the ostream \p OS based
  // on the schema provided in \p Schema.
  LLVM_ABI void serialize(const MemProfSchema &Schema, raw_ostream &OS,
                          IndexedVersion Version,
                          llvm::DenseMap<CallStackId, LinearCallStackId>
                              *MemProfCallStackIndexes = nullptr) const;

  // Deserializes memprof records from the Buffer.
  LLVM_ABI static IndexedMemProfRecord deserialize(const MemProfSchema &Schema,
                                                   const unsigned char *Buffer,
                                                   IndexedVersion Version);

  // Convert IndexedMemProfRecord to MemProfRecord.  Callback is used to
  // translate CallStackId to call stacks with frames inline.
  LLVM_ABI MemProfRecord toMemProfRecord(
      llvm::function_ref<std::vector<Frame>(const CallStackId)> Callback) const;
};

// Returns the GUID for the function name after canonicalization. For
// memprof, we remove any .llvm suffix added by LTO. MemProfRecords are
// mapped to functions using this GUID.
LLVM_ABI GlobalValue::GUID getGUID(const StringRef FunctionName);

// Holds call site information with frame contents inline.
struct CallSiteInfo {
  // The frames in the call stack
  std::vector<Frame> Frames;

  // The GUIDs of the callees at this call site
  SmallVector<GlobalValue::GUID, 1> CalleeGuids;

  CallSiteInfo() = default;
  CallSiteInfo(std::vector<Frame> Frames) : Frames(std::move(Frames)) {}
  CallSiteInfo(std::vector<Frame> Frames,
               SmallVector<GlobalValue::GUID, 1> CalleeGuids)
      : Frames(std::move(Frames)), CalleeGuids(std::move(CalleeGuids)) {}

  bool operator==(const CallSiteInfo &Other) const {
    return Frames == Other.Frames && CalleeGuids == Other.CalleeGuids;
  }

  bool operator!=(const CallSiteInfo &Other) const {
    return !operator==(Other);
  }
};

// Holds the memprof profile information for a function. The internal
// representation stores frame contents inline. This representation should
// be used for small amount of temporary, in memory instances.
struct MemProfRecord {
  // Same as IndexedMemProfRecord::AllocSites with frame contents inline.
  llvm::SmallVector<AllocationInfo> AllocSites;
  // Same as IndexedMemProfRecord::CallSites with frame contents inline.
  llvm::SmallVector<CallSiteInfo> CallSites;

  MemProfRecord() = default;

  // Prints out the contents of the memprof record in YAML.
  void print(llvm::raw_ostream &OS) const {
    if (!AllocSites.empty()) {
      OS << "    AllocSites:\n";
      for (const AllocationInfo &N : AllocSites)
        N.printYAML(OS);
    }

    if (!CallSites.empty()) {
      OS << "    CallSites:\n";
      for (const CallSiteInfo &CS : CallSites) {
        for (const Frame &F : CS.Frames) {
          OS << "    -\n";
          F.printYAML(OS);
        }
      }
    }
  }
};

// Reads a memprof schema from a buffer. All entries in the buffer are
// interpreted as uint64_t. The first entry in the buffer denotes the number of
// ids in the schema. Subsequent entries are integers which map to memprof::Meta
// enum class entries. After successfully reading the schema, the pointer is one
// byte past the schema contents.
LLVM_ABI Expected<MemProfSchema>
readMemProfSchema(const unsigned char *&Buffer);

// Trait for reading IndexedMemProfRecord data from the on-disk hash table.
class RecordLookupTrait {
public:
  using data_type = const IndexedMemProfRecord &;
  using internal_key_type = uint64_t;
  using external_key_type = uint64_t;
  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

  RecordLookupTrait() = delete;
  RecordLookupTrait(IndexedVersion V, const MemProfSchema &S)
      : Version(V), Schema(S) {}

  static bool EqualKey(uint64_t A, uint64_t B) { return A == B; }
  static uint64_t GetInternalKey(uint64_t K) { return K; }
  static uint64_t GetExternalKey(uint64_t K) { return K; }

  hash_value_type ComputeHash(uint64_t K) { return K; }

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&D) {
    using namespace support;

    offset_type KeyLen =
        endian::readNext<offset_type, llvm::endianness::little>(D);
    offset_type DataLen =
        endian::readNext<offset_type, llvm::endianness::little>(D);
    return std::make_pair(KeyLen, DataLen);
  }

  uint64_t ReadKey(const unsigned char *D, offset_type /*Unused*/) {
    using namespace support;
    return endian::readNext<external_key_type, llvm::endianness::little>(D);
  }

  data_type ReadData(uint64_t K, const unsigned char *D,
                     offset_type /*Unused*/) {
    Record = IndexedMemProfRecord::deserialize(Schema, D, Version);
    return Record;
  }

private:
  // Holds the MemProf version.
  IndexedVersion Version;
  // Holds the memprof schema used to deserialize records.
  MemProfSchema Schema;
  // Holds the records from one function deserialized from the indexed format.
  IndexedMemProfRecord Record;
};

// Trait for writing IndexedMemProfRecord data to the on-disk hash table.
class RecordWriterTrait {
public:
  using key_type = uint64_t;
  using key_type_ref = uint64_t;

  using data_type = IndexedMemProfRecord;
  using data_type_ref = IndexedMemProfRecord &;

  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

private:
  // Pointer to the memprof schema to use for the generator.
  const MemProfSchema *Schema;
  // The MemProf version to use for the serialization.
  IndexedVersion Version;

  // Mappings from CallStackId to the indexes into the call stack array.
  llvm::DenseMap<CallStackId, LinearCallStackId> *MemProfCallStackIndexes;

public:
  // We do not support the default constructor, which does not set Version.
  RecordWriterTrait() = delete;
  RecordWriterTrait(
      const MemProfSchema *Schema, IndexedVersion V,
      llvm::DenseMap<CallStackId, LinearCallStackId> *MemProfCallStackIndexes)
      : Schema(Schema), Version(V),
        MemProfCallStackIndexes(MemProfCallStackIndexes) {}

  static hash_value_type ComputeHash(key_type_ref K) { return K; }

  std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace support;

    endian::Writer LE(Out, llvm::endianness::little);
    offset_type N = sizeof(K);
    LE.write<offset_type>(N);
    offset_type M = V.serializedSize(*Schema, Version);
    LE.write<offset_type>(M);
    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type /*Unused*/) {
    using namespace support;
    endian::Writer LE(Out, llvm::endianness::little);
    LE.write<uint64_t>(K);
  }

  void EmitData(raw_ostream &Out, key_type_ref /*Unused*/, data_type_ref V,
                offset_type /*Unused*/) {
    assert(Schema != nullptr && "MemProf schema is not initialized!");
    V.serialize(*Schema, Out, Version, MemProfCallStackIndexes);
    // Clear the IndexedMemProfRecord which results in clearing/freeing its
    // vectors of allocs and callsites. This is owned by the associated on-disk
    // hash table, but unused after this point. See also the comment added to
    // the client which constructs the on-disk hash table for this trait.
    V.clear();
  }
};

// Trait for writing frame mappings to the on-disk hash table.
class FrameWriterTrait {
public:
  using key_type = FrameId;
  using key_type_ref = FrameId;

  using data_type = Frame;
  using data_type_ref = Frame &;

  using hash_value_type = FrameId;
  using offset_type = uint64_t;

  static hash_value_type ComputeHash(key_type_ref K) { return K; }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace support;
    endian::Writer LE(Out, llvm::endianness::little);
    offset_type N = sizeof(K);
    LE.write<offset_type>(N);
    offset_type M = V.serializedSize();
    LE.write<offset_type>(M);
    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type /*Unused*/) {
    using namespace support;
    endian::Writer LE(Out, llvm::endianness::little);
    LE.write<key_type>(K);
  }

  void EmitData(raw_ostream &Out, key_type_ref /*Unused*/, data_type_ref V,
                offset_type /*Unused*/) {
    V.serialize(Out);
  }
};

// Trait for reading frame mappings from the on-disk hash table.
class FrameLookupTrait {
public:
  using data_type = const Frame;
  using internal_key_type = FrameId;
  using external_key_type = FrameId;
  using hash_value_type = FrameId;
  using offset_type = uint64_t;

  static bool EqualKey(internal_key_type A, internal_key_type B) {
    return A == B;
  }
  static uint64_t GetInternalKey(internal_key_type K) { return K; }
  static uint64_t GetExternalKey(external_key_type K) { return K; }

  hash_value_type ComputeHash(internal_key_type K) { return K; }

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&D) {
    using namespace support;

    offset_type KeyLen =
        endian::readNext<offset_type, llvm::endianness::little>(D);
    offset_type DataLen =
        endian::readNext<offset_type, llvm::endianness::little>(D);
    return std::make_pair(KeyLen, DataLen);
  }

  uint64_t ReadKey(const unsigned char *D, offset_type /*Unused*/) {
    using namespace support;
    return endian::readNext<external_key_type, llvm::endianness::little>(D);
  }

  data_type ReadData(uint64_t K, const unsigned char *D,
                     offset_type /*Unused*/) {
    return Frame::deserialize(D);
  }
};

// Trait for writing call stacks to the on-disk hash table.
class CallStackWriterTrait {
public:
  using key_type = CallStackId;
  using key_type_ref = CallStackId;

  using data_type = llvm::SmallVector<FrameId>;
  using data_type_ref = llvm::SmallVector<FrameId> &;

  using hash_value_type = CallStackId;
  using offset_type = uint64_t;

  static hash_value_type ComputeHash(key_type_ref K) { return K; }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace support;
    endian::Writer LE(Out, llvm::endianness::little);
    // We do not explicitly emit the key length because it is a constant.
    offset_type N = sizeof(K);
    offset_type M = sizeof(FrameId) * V.size();
    LE.write<offset_type>(M);
    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type /*Unused*/) {
    using namespace support;
    endian::Writer LE(Out, llvm::endianness::little);
    LE.write<key_type>(K);
  }

  void EmitData(raw_ostream &Out, key_type_ref /*Unused*/, data_type_ref V,
                offset_type /*Unused*/) {
    using namespace support;
    endian::Writer LE(Out, llvm::endianness::little);
    // Emit the frames.  We do not explicitly emit the length of the vector
    // because it can be inferred from the data length.
    for (FrameId F : V)
      LE.write<FrameId>(F);
  }
};

// Trait for reading call stack mappings from the on-disk hash table.
class CallStackLookupTrait {
public:
  using data_type = const llvm::SmallVector<FrameId>;
  using internal_key_type = CallStackId;
  using external_key_type = CallStackId;
  using hash_value_type = CallStackId;
  using offset_type = uint64_t;

  static bool EqualKey(internal_key_type A, internal_key_type B) {
    return A == B;
  }
  static uint64_t GetInternalKey(internal_key_type K) { return K; }
  static uint64_t GetExternalKey(external_key_type K) { return K; }

  hash_value_type ComputeHash(internal_key_type K) { return K; }

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&D) {
    using namespace support;

    // We do not explicitly read the key length because it is a constant.
    offset_type KeyLen = sizeof(external_key_type);
    offset_type DataLen =
        endian::readNext<offset_type, llvm::endianness::little>(D);
    return std::make_pair(KeyLen, DataLen);
  }

  uint64_t ReadKey(const unsigned char *D, offset_type /*Unused*/) {
    using namespace support;
    return endian::readNext<external_key_type, llvm::endianness::little>(D);
  }

  data_type ReadData(uint64_t K, const unsigned char *D, offset_type Length) {
    using namespace support;
    llvm::SmallVector<FrameId> CS;
    // Derive the number of frames from the data length.
    uint64_t NumFrames = Length / sizeof(FrameId);
    assert(Length % sizeof(FrameId) == 0);
    CS.reserve(NumFrames);
    for (size_t I = 0; I != NumFrames; ++I) {
      FrameId F = endian::readNext<FrameId, llvm::endianness::little>(D);
      CS.push_back(F);
    }
    return CS;
  }
};

struct LineLocation {
  LineLocation(uint32_t L, uint32_t D) : LineOffset(L), Column(D) {}

  bool operator<(const LineLocation &O) const {
    return std::tie(LineOffset, Column) < std::tie(O.LineOffset, O.Column);
  }

  bool operator==(const LineLocation &O) const {
    return LineOffset == O.LineOffset && Column == O.Column;
  }

  bool operator!=(const LineLocation &O) const {
    return LineOffset != O.LineOffset || Column != O.Column;
  }

  uint64_t getHashCode() const { return ((uint64_t)Column << 32) | LineOffset; }

  uint32_t LineOffset;
  uint32_t Column;
};

// A pair of a call site location and its corresponding callee GUID.
using CallEdgeTy = std::pair<LineLocation, uint64_t>;
} // namespace memprof
} // namespace llvm
#endif // LLVM_PROFILEDATA_MEMPROF_H
