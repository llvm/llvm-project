#ifndef LLVM_PROFILEDATA_MEMPROF_H_
#define LLVM_PROFILEDATA_MEMPROF_H_

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

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
  struct Frame {
    std::string Function;
    uint32_t LineOffset;
    uint32_t Column;
    bool IsInlineFrame;

    Frame(std::string Str, uint32_t Off, uint32_t Col, bool Inline)
        : Function(std::move(Str)), LineOffset(Off), Column(Col),
          IsInlineFrame(Inline) {}
  };

  std::vector<Frame> CallStack;
  PortableMemInfoBlock Info;

  void clear() {
    CallStack.clear();
    Info.clear();
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
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROF_H_
