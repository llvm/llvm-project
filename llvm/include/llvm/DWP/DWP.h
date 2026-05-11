#ifndef LLVM_DWP_DWP_H
#define LLVM_DWP_DWP_H

#include "DWPStringPool.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <deque>
#include <vector>

namespace llvm::object {
class ObjectFile;
}

namespace llvm {
class raw_pwrite_stream;

enum OnCuIndexOverflow {
  HardStop,
  SoftStop,
  Continue,
};

enum Dwarf64StrOffsetsPromotion {
  Disabled, ///< Don't do any conversion of .debug_str_offsets tables.
  Enabled,  ///< Convert any .debug_str_offsets tables to DWARF64 if needed.
  Always,   ///< Always emit .debug_str_offsets talbes as DWARF64 for testing.
};

/// Section identifiers for DWP output.
enum DWPSectionId : unsigned {
  DS_Info,
  DS_Types,
  DS_Abbrev,
  DS_Line,
  DS_Loc,
  DS_Loclists,
  DS_Rnglists,
  DS_Macro,
  DS_Str,
  DS_StrOffsets,
  DS_CUIndex,
  DS_TUIndex,
  DS_NumSections
};

/// Direct ELF writer for DWP output.
///
/// Section data is stored as zero-copy StringRef chunks pointing to the
/// mmap'd input files, plus an inline buffer for constructed data
/// (emitIntValue). This avoids copying gigabytes of debug section data
/// through the MC infrastructure (MCContext, MCAssembler, MCDataFragment
/// allocation, layout, etc.).
class LLVM_ABI DWPWriter {
  /// Per-section storage: ordered sequence of zero-copy chunks and inline
  /// data. emitBytes() adds zero-copy StringRef references, emitIntValue()
  /// appends to an inline buffer. When emitBytes() is called with pending
  /// inline data, the buffer is flushed to an owned block first to preserve
  /// the correct interleaving order in the output.
  struct SectionData {
    SmallVector<StringRef, 4> Chunks; // ordered segments (refs + flushed bufs)
    SmallVector<char, 0> Buffer;      // pending inline data (emitIntValue)
    // Heap storage for flushed buffers. Uses std::deque so that push_back
    // does not invalidate existing elements (StringRefs point into these).
    std::deque<SmallVector<char, 0>> OwnedBuffers;

    /// Flush pending Buffer data into Chunks as an owned block.
    void flushBuffer() {
      if (!Buffer.empty()) {
        OwnedBuffers.push_back(std::move(Buffer));
        auto &B = OwnedBuffers.back();
        Chunks.push_back(StringRef(B.data(), B.size()));
        Buffer = SmallVector<char, 0>();
      }
    }

    uint64_t totalSize() const {
      uint64_t Size = 0;
      for (auto &C : Chunks)
        Size += C.size();
      Size += Buffer.size();
      return Size;
    }

    bool empty() const { return Chunks.empty() && Buffer.empty(); }

    void writeTo(raw_ostream &OS) const {
      for (auto &C : Chunks)
        OS.write(C.data(), C.size());
      if (!Buffer.empty())
        OS.write(Buffer.data(), Buffer.size());
    }
  };

  SectionData Sections[DS_NumSections];
  DWPSectionId CurrentSection = DS_Info;
  uint16_t ELFMachine = 0;
  uint8_t ELFOSABI = 0;
  bool IsWASM = false;

public:
  DWPWriter() = default;

  void setMachine(uint16_t Machine) { ELFMachine = Machine; }
  void setOSABI(uint8_t OSABI) { ELFOSABI = OSABI; }
  void setIsWASM(bool V) { IsWASM = V; }

  SmallVectorImpl<char> &getSectionBuffer(DWPSectionId Id) {
    return Sections[Id].Buffer;
  }

  void switchSection(DWPSectionId Id) { CurrentSection = Id; }

  /// Zero-copy: stores a reference to the input data without copying.
  /// Flushes any pending inline data first to preserve output order.
  void emitBytes(StringRef Data) {
    if (!Data.empty()) {
      auto &SD = Sections[CurrentSection];
      SD.flushBuffer();
      SD.Chunks.push_back(Data);
    }
  }

  void emitIntValue(uint64_t Value, unsigned Size) {
    auto &Buf = Sections[CurrentSection].Buffer;
    for (unsigned I = 0; I < Size; ++I) {
      Buf.push_back(static_cast<char>(Value & 0xff));
      Value >>= 8;
    }
  }

  Error writeELF(raw_pwrite_stream &OS);
  Error writeWASM(raw_pwrite_stream &OS);
  Error write(raw_pwrite_stream &OS) {
    return IsWASM ? writeWASM(OS) : writeELF(OS);
  }
};

struct UnitIndexEntry {
  DWARFUnitIndex::Entry::SectionContribution Contributions[8];
  std::string Name;
  std::string DWOName;
  StringRef DWPName;
};

// Holds data for Skeleton, Split Compilation, and Type Unit Headers (only in
// v5) as defined in Dwarf 5 specification, 7.5.1.2, 7.5.1.3 and Dwarf 4
// specification 7.5.1.1.
struct InfoSectionUnitHeader {
  // unit_length field. Note that the type is uint64_t even in 32-bit dwarf.
  uint64_t Length = 0;

  // version field.
  uint16_t Version = 0;

  // unit_type field. Initialized only if Version >= 5.
  uint8_t UnitType = 0;

  // address_size field.
  uint8_t AddrSize = 0;

  // debug_abbrev_offset field. Note that the type is uint64_t even in 32-bit
  // dwarf. It is assumed to be 0.
  uint64_t DebugAbbrevOffset = 0;

  // dwo_id field. This resides in the header only if Version >= 5.
  // In earlier versions, it is read from DW_AT_GNU_dwo_id.
  std::optional<uint64_t> Signature;

  // Derived from the length of Length field.
  dwarf::DwarfFormat Format = dwarf::DwarfFormat::DWARF32;

  // The size of the Header in bytes. This is derived while parsing the header,
  // and is stored as a convenience.
  uint8_t HeaderSize = 0;
};

struct CompileUnitIdentifiers {
  uint64_t Signature = 0;
  const char *Name = "";
  const char *DWOName = "";
};

LLVM_ABI Error write(DWPWriter &Out, ArrayRef<std::string> Inputs,
                     OnCuIndexOverflow OverflowOptValue,
                     Dwarf64StrOffsetsPromotion StrOffsetsOptValue,
                     raw_pwrite_stream *OS = nullptr);

typedef std::vector<std::pair<DWARFSectionKind, uint32_t>> SectionLengths;

LLVM_ABI Expected<InfoSectionUnitHeader>
parseInfoSectionUnitHeader(StringRef Info);

} // namespace llvm
#endif // LLVM_DWP_DWP_H
