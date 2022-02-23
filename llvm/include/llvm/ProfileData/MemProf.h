#ifndef LLVM_PROFILEDATA_MEMPROF_H_
#define LLVM_PROFILEDATA_MEMPROF_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

namespace llvm {
namespace memprof {

enum class Meta : uint64_t {
  Start = 0,
#define MIBEntryDef(NameTag, Name, Type) NameTag,
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  Size
};

using MemProfSchema = llvm::SmallVector<Meta, static_cast<int>(Meta::Size)>;

// Holds the actual MemInfoBlock data with all fields. Contents may be read or
// written partially by providing an appropriate schema to the serialize and
// deserialize methods.
struct PortableMemInfoBlock {
  PortableMemInfoBlock() = default;
  explicit PortableMemInfoBlock(const MemInfoBlock &Block) {
#define MIBEntryDef(NameTag, Name, Type) Name = Block.Name;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  }

  PortableMemInfoBlock(const MemProfSchema &Schema, const unsigned char *Ptr) {
    deserialize(Schema, Ptr);
  }

  // Read the contents of \p Ptr based on the \p Schema to populate the
  // MemInfoBlock member.
  void deserialize(const MemProfSchema &Schema, const unsigned char *Ptr) {
    using namespace support;

    for (const Meta Id : Schema) {
      switch (Id) {
#define MIBEntryDef(NameTag, Name, Type)                                       \
  case Meta::Name: {                                                           \
    Name = endian::readNext<Type, little, unaligned>(Ptr);                     \
  } break;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
      default:
        llvm_unreachable("Unknown meta type id, is the profile collected from "
                         "a newer version of the runtime?");
      }
    }
  }

  // Write the contents of the MemInfoBlock based on the \p Schema provided to
  // the raw_ostream \p OS.
  void serialize(const MemProfSchema &Schema, raw_ostream &OS) const {
    using namespace support;

    endian::Writer LE(OS, little);
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
    OS << "    MemInfoBlock:\n";
#define MIBEntryDef(NameTag, Name, Type)                                       \
  OS << "      " << #Name << ": " << Name << "\n";
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  }

  // Define getters for each type which can be called by analyses.
#define MIBEntryDef(NameTag, Name, Type)                                       \
  Type get##Name() const { return Name; }
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef

  void clear() { *this = PortableMemInfoBlock(); }

  // Returns the full schema currently in use.
  static MemProfSchema getSchema() {
    MemProfSchema List;
#define MIBEntryDef(NameTag, Name, Type) List.push_back(Meta::Name);
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
    return List;
  }

  bool operator==(const PortableMemInfoBlock &Other) const {
#define MIBEntryDef(NameTag, Name, Type)                                       \
  if (Other.get##Name() != get##Name())                                        \
    return false;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
    return true;
  }

  bool operator!=(const PortableMemInfoBlock &Other) const {
    return !operator==(Other);
  }

  static constexpr size_t serializedSize() {
    size_t Result = 0;
#define MIBEntryDef(NameTag, Name, Type) Result += sizeof(Type);
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
    return Result;
  }

private:
#define MIBEntryDef(NameTag, Name, Type) Type Name = Type();
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
};

struct MemProfRecord {
  // Describes a call frame for a dynamic allocation context. The contents of
  // the frame are populated by symbolizing the stack depot call frame from the
  // compiler runtime.
  struct Frame {
    // A uuid (uint64_t) identifying the function. It is obtained by
    // llvm::md5(FunctionName) which returns the lower 64 bits.
    GlobalValue::GUID Function;
    // The source line offset of the call from the beginning of parent function.
    uint32_t LineOffset;
    // The source column number of the call to help distinguish multiple calls
    // on the same line.
    uint32_t Column;
    // Whether the current frame is inlined.
    bool IsInlineFrame;

    Frame(uint64_t Hash, uint32_t Off, uint32_t Col, bool Inline)
        : Function(Hash), LineOffset(Off), Column(Col), IsInlineFrame(Inline) {}

    bool operator==(const Frame &Other) const {
      return Other.Function == Function && Other.LineOffset == LineOffset &&
             Other.Column == Column && Other.IsInlineFrame == IsInlineFrame;
    }

    bool operator!=(const Frame &Other) const { return !operator==(Other); }

    // Write the contents of the frame to the ostream \p OS.
    void serialize(raw_ostream & OS) const {
      using namespace support;

      endian::Writer LE(OS, little);

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

      const uint64_t F = endian::readNext<uint64_t, little, unaligned>(Ptr);
      const uint32_t L = endian::readNext<uint32_t, little, unaligned>(Ptr);
      const uint32_t C = endian::readNext<uint32_t, little, unaligned>(Ptr);
      const bool I = endian::readNext<bool, little, unaligned>(Ptr);
      return Frame(/*Function=*/F, /*LineOffset=*/L, /*Column=*/C,
                   /*IsInlineFrame=*/I);
    }

    // Returns the size of the frame information.
    static constexpr size_t serializedSize() {
      return sizeof(Frame::Function) + sizeof(Frame::LineOffset) +
             sizeof(Frame::Column) + sizeof(Frame::IsInlineFrame);
    }
  };

  // The dynamic calling context for the allocation.
  std::vector<Frame> CallStack;
  // The statistics obtained from the runtime for the allocation.
  PortableMemInfoBlock Info;

  void clear() {
    CallStack.clear();
    Info.clear();
  }

  size_t serializedSize() const {
    return sizeof(uint64_t) + // The number of frames to serialize.
           Frame::serializedSize() *
               CallStack.size() + // The contents of the frames.
           PortableMemInfoBlock::serializedSize(); // The size of the payload.
  }

  // Prints out the contents of the memprof record in YAML.
  void print(llvm::raw_ostream &OS) const {
    OS << "    Callstack:\n";
    // TODO: Print out the frame on one line with to make it easier for deep
    // callstacks once we have a test to check valid YAML is generated.
    for (const auto &Frame : CallStack) {
      OS << "    -\n"
         << "      Function: " << Frame.Function << "\n"
         << "      LineOffset: " << Frame.LineOffset << "\n"
         << "      Column: " << Frame.Column << "\n"
         << "      Inline: " << Frame.IsInlineFrame << "\n";
    }

    Info.printYAML(OS);
  }

  bool operator==(const MemProfRecord &Other) const {
    if (Other.Info != Info)
      return false;

    if (Other.CallStack.size() != CallStack.size())
      return false;

    for (size_t I = 0; I < Other.CallStack.size(); I++) {
      if (Other.CallStack[I] != CallStack[I])
        return false;
    }
    return true;
  }
};

// Serializes the memprof records in \p Records to the ostream \p OS based on
// the schema provided in \p Schema.
void serializeRecords(const ArrayRef<MemProfRecord> Records,
                      const MemProfSchema &Schema, raw_ostream &OS);

// Deserializes memprof records from the Buffer
SmallVector<MemProfRecord, 4> deserializeRecords(const MemProfSchema &Schema,
                                                 const unsigned char *Buffer);

// Reads a memprof schema from a buffer. All entries in the buffer are
// interpreted as uint64_t. The first entry in the buffer denotes the number of
// ids in the schema. Subsequent entries are integers which map to memprof::Meta
// enum class entries. After successfully reading the schema, the pointer is one
// byte past the schema contents.
Expected<MemProfSchema> readMemProfSchema(const unsigned char *&Buffer);

using FunctionMemProfMap =
    DenseMap<uint64_t, SmallVector<memprof::MemProfRecord, 4>>;

/// Trait for lookups into the on-disk hash table for memprof format in the
/// indexed profile.
class MemProfRecordLookupTrait {
public:
  using data_type = ArrayRef<MemProfRecord>;
  using internal_key_type = uint64_t;
  using external_key_type = uint64_t;
  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

  MemProfRecordLookupTrait() = delete;
  MemProfRecordLookupTrait(const MemProfSchema &S) : Schema(S) {}

  static bool EqualKey(uint64_t A, uint64_t B) { return A == B; }
  static uint64_t GetInternalKey(uint64_t K) { return K; }
  static uint64_t GetExternalKey(uint64_t K) { return K; }

  hash_value_type ComputeHash(uint64_t K) { return K; }

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&D) {
    using namespace support;

    offset_type KeyLen = endian::readNext<offset_type, little, unaligned>(D);
    offset_type DataLen = endian::readNext<offset_type, little, unaligned>(D);
    return std::make_pair(KeyLen, DataLen);
  }

  uint64_t ReadKey(const unsigned char *D, offset_type /*Unused*/) {
    using namespace support;
    return endian::readNext<external_key_type, little, unaligned>(D);
  }

  data_type ReadData(uint64_t K, const unsigned char *D,
                     offset_type /*Unused*/) {
    Records = deserializeRecords(Schema, D);
    return Records;
  }

private:
  // Holds the memprof schema used to deserialize records.
  MemProfSchema Schema;
  // Holds the records from one function deserialized from the indexed format.
  llvm::SmallVector<MemProfRecord, 4> Records;
};

class MemProfRecordWriterTrait {
public:
  using key_type = uint64_t;
  using key_type_ref = uint64_t;

  using data_type = ArrayRef<MemProfRecord>;
  using data_type_ref = ArrayRef<MemProfRecord>;

  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

  // Pointer to the memprof schema to use for the generator. Unlike the reader
  // we must use a default constructor with no params for the writer trait so we
  // have a public member which must be initialized by the user.
  MemProfSchema *Schema = nullptr;

  MemProfRecordWriterTrait() = default;

  static hash_value_type ComputeHash(key_type_ref K) { return K; }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace support;

    endian::Writer LE(Out, little);

    offset_type N = sizeof(K);
    LE.write<offset_type>(N);

    offset_type M = 0;

    M += sizeof(uint64_t);
    for (const auto &Record : V) {
      M += Record.serializedSize();
    }

    LE.write<offset_type>(M);
    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type /*Unused*/) {
    using namespace support;
    endian::Writer LE(Out, little);
    LE.write<uint64_t>(K);
  }

  void EmitData(raw_ostream &Out, key_type_ref /*Unused*/, data_type_ref V,
                offset_type /*Unused*/) {
    assert(Schema != nullptr && "MemProf schema is not initialized!");
    serializeRecords(V, *Schema, Out);
  }
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROF_H_
