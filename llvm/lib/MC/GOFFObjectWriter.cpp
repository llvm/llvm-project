//===- lib/MC/GOFFObjectWriter.cpp - GOFF File Writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements GOFF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "goff-writer"

namespace {
// Common flag values on records.

// Flag: This record is continued.
constexpr uint8_t RecContinued = GOFF::Flags(7, 1, 1);

// Flag: This record is a continuation.
constexpr uint8_t RecContinuation = GOFF::Flags(6, 1, 1);

// The GOFFOstream is responsible to write the data into the fixed physical
// records of the format. A user of this class announces the begin of a new
// logical record. While writing the payload, the physical records are created
// for the data. Possible fill bytes at the end of a physical record are written
// automatically. In principle, the GOFFOstream is agnostic of the endianness of
// the payload. However, it also supports writing data in big endian byte order.
//
// The physical records use the flag field to indicate if the there is a
// successor and predecessor record. To be able to set these flags while
// writing, the basic implementation idea is to always buffer the last seen
// physical record.
class GOFFOstream {
  /// The underlying raw_pwrite_stream.
  raw_pwrite_stream &OS;

  /// The number of logical records emitted so far.
  uint32_t LogicalRecords = 0;

  /// The number of physical records emitted so far.
  uint32_t PhysicalRecords = 0;

  /// The size of the buffer. Same as the payload size of a physical record.
  static constexpr uint8_t BufferSize = GOFF::PayloadLength;

  /// Current position in buffer.
  char *BufferPtr = Buffer;

  /// Static allocated buffer for the stream.
  char Buffer[BufferSize];

  /// The type of the current logical record, and the flags (aka continued and
  /// continuation indicators) for the previous (physical) record.
  uint8_t TypeAndFlags = 0;

public:
  GOFFOstream(raw_pwrite_stream &OS);
  ~GOFFOstream();

  raw_pwrite_stream &getOS() { return OS; }
  size_t getWrittenSize() const { return PhysicalRecords * GOFF::RecordLength; }
  uint32_t getNumLogicalRecords() { return LogicalRecords; }

  /// Write the specified bytes.
  void write(const char *Ptr, size_t Size);

  /// Write zeroes, up to a maximum of 16 bytes.
  void write_zeros(unsigned NumZeros);

  /// Support for endian-specific data.
  template <typename value_type> void writebe(value_type Value) {
    Value =
        support::endian::byte_swap<value_type>(Value, llvm::endianness::big);
    write((const char *)&Value, sizeof(value_type));
  }

  /// Begin a new logical record. Implies finalizing the previous record.
  void newRecord(GOFF::RecordType Type);

  /// Ends a logical record.
  void finalizeRecord();

private:
  /// Updates the continued/continuation flags, and writes the record prefix of
  /// a physical record.
  void updateFlagsAndWritePrefix(bool IsContinued);

  /// Returns the remaining size in the buffer.
  size_t getRemainingSize();
};
} // namespace

GOFFOstream::GOFFOstream(raw_pwrite_stream &OS) : OS(OS) {}

GOFFOstream::~GOFFOstream() { finalizeRecord(); }

void GOFFOstream::updateFlagsAndWritePrefix(bool IsContinued) {
  // Update the flags based on the previous state and the flag IsContinued.
  if (TypeAndFlags & RecContinued)
    TypeAndFlags |= RecContinuation;
  if (IsContinued)
    TypeAndFlags |= RecContinued;
  else
    TypeAndFlags &= ~RecContinued;

  OS << static_cast<unsigned char>(GOFF::PTVPrefix) // Record Type
     << static_cast<unsigned char>(TypeAndFlags)    // Continuation
     << static_cast<unsigned char>(0);              // Version

  ++PhysicalRecords;
}

size_t GOFFOstream::getRemainingSize() {
  return size_t(&Buffer[BufferSize] - BufferPtr);
}

void GOFFOstream::write(const char *Ptr, size_t Size) {
  size_t RemainingSize = getRemainingSize();

  // Data fits into the buffer.
  if (LLVM_LIKELY(Size <= RemainingSize)) {
    memcpy(BufferPtr, Ptr, Size);
    BufferPtr += Size;
    return;
  }

  // Otherwise the buffer is partially filled or full, and data does not fit
  // into it.
  updateFlagsAndWritePrefix(/*IsContinued=*/true);
  OS.write(Buffer, size_t(BufferPtr - Buffer));
  if (RemainingSize > 0) {
    OS.write(Ptr, RemainingSize);
    Ptr += RemainingSize;
    Size -= RemainingSize;
  }

  while (Size > BufferSize) {
    updateFlagsAndWritePrefix(/*IsContinued=*/true);
    OS.write(Ptr, BufferSize);
    Ptr += BufferSize;
    Size -= BufferSize;
  }

  // The remaining bytes fit into the buffer.
  memcpy(Buffer, Ptr, Size);
  BufferPtr = &Buffer[Size];
}

void GOFFOstream::write_zeros(unsigned NumZeros) {
  assert(NumZeros <= 16 && "Range for zeros too large");

  // Handle the common case first: all fits in the buffer.
  size_t RemainingSize = getRemainingSize();
  if (LLVM_LIKELY(RemainingSize >= NumZeros)) {
    memset(BufferPtr, 0, NumZeros);
    BufferPtr += NumZeros;
    return;
  }

  // Otherwise some field value is cleared.
  static char Zeros[16] = {
      0,
  };
  write(Zeros, NumZeros);
}

void GOFFOstream::newRecord(GOFF::RecordType Type) {
  finalizeRecord();
  TypeAndFlags = Type << 4;
  ++LogicalRecords;
}

void GOFFOstream::finalizeRecord() {
  if (Buffer == BufferPtr)
    return;
  updateFlagsAndWritePrefix(/*IsContinued=*/false);
  OS.write(Buffer, size_t(BufferPtr - Buffer));
  OS.write_zeros(getRemainingSize());
  BufferPtr = Buffer;
}

namespace {
// A GOFFSymbol holds all the data required for writing an ESD record.
class GOFFSymbol {
public:
  std::string Name;
  uint32_t EsdId;
  uint32_t ParentEsdId;
  uint64_t Offset = 0; // Offset of the symbol into the section. LD only.
                       // Offset is only 32 bit, the larger type is used to
                       // enable error checking.
  GOFF::ESDSymbolType SymbolType;
  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_ProgramManagementBinder;

  GOFF::BehavioralAttributes BehavAttrs;
  GOFF::SymbolFlags SymbolFlags;
  uint32_t SortKey = 0;
  uint32_t SectionLength = 0;
  uint32_t ADAEsdId = 0;
  uint32_t EASectionEDEsdId = 0;
  uint32_t EASectionOffset = 0;
  uint8_t FillByteValue = 0;

  GOFFSymbol() : EsdId(0), ParentEsdId(0) {}

  GOFFSymbol(StringRef Name, uint32_t EsdID, const GOFF::SDAttr &Attr)
      : Name(Name.data(), Name.size()), EsdId(EsdID), ParentEsdId(0),
        SymbolType(GOFF::ESD_ST_SectionDefinition) {
    BehavAttrs.setTaskingBehavior(Attr.TaskingBehavior);
    BehavAttrs.setBindingScope(Attr.BindingScope);
  }

  GOFFSymbol(StringRef Name, uint32_t EsdID, uint32_t ParentEsdID,
             const GOFF::EDAttr &Attr)
      : Name(Name.data(), Name.size()), EsdId(EsdID), ParentEsdId(ParentEsdID),
        SymbolType(GOFF::ESD_ST_ElementDefinition) {
    this->NameSpace = Attr.NameSpace;
    // We always set a fill byte value.
    this->FillByteValue = Attr.FillByteValue;
    SymbolFlags.setFillBytePresence(1);
    SymbolFlags.setReservedQwords(Attr.ReservedQwords);
    // TODO Do we need/should set the "mangled" flag?
    BehavAttrs.setReadOnly(Attr.IsReadOnly);
    BehavAttrs.setRmode(Attr.Rmode);
    BehavAttrs.setTextStyle(Attr.TextStyle);
    BehavAttrs.setBindingAlgorithm(Attr.BindAlgorithm);
    BehavAttrs.setLoadingBehavior(Attr.LoadBehavior);
    BehavAttrs.setAlignment(Attr.Alignment);
  }

  GOFFSymbol(StringRef Name, uint32_t EsdID, uint32_t ParentEsdID,
             GOFF::ESDNameSpaceId NameSpace, const GOFF::LDAttr &Attr)
      : Name(Name.data(), Name.size()), EsdId(EsdID), ParentEsdId(ParentEsdID),
        SymbolType(GOFF::ESD_ST_LabelDefinition), NameSpace(NameSpace) {
    SymbolFlags.setRenameable(Attr.IsRenamable);
    BehavAttrs.setExecutable(Attr.Executable);
    BehavAttrs.setBindingStrength(Attr.BindingStrength);
    BehavAttrs.setLinkageType(Attr.Linkage);
    BehavAttrs.setAmode(Attr.Amode);
    BehavAttrs.setBindingScope(Attr.BindingScope);
  }

  GOFFSymbol(StringRef Name, uint32_t EsdID, uint32_t ParentEsdID,
             const GOFF::EDAttr &EDAttr, const GOFF::PRAttr &Attr)
      : Name(Name.data(), Name.size()), EsdId(EsdID), ParentEsdId(ParentEsdID),
        SymbolType(GOFF::ESD_ST_PartReference), NameSpace(EDAttr.NameSpace) {
    SymbolFlags.setRenameable(Attr.IsRenamable);
    BehavAttrs.setExecutable(Attr.Executable);
    BehavAttrs.setLinkageType(Attr.Linkage);
    BehavAttrs.setBindingScope(Attr.BindingScope);
    BehavAttrs.setAlignment(EDAttr.Alignment);
  }
};

class GOFFWriter {
  GOFFOstream OS;
  MCAssembler &Asm;

  void writeHeader();
  void writeSymbol(const GOFFSymbol &Symbol);
  void writeText(const MCSectionGOFF *MC);
  void writeEnd();

  void defineSectionSymbols(const MCSectionGOFF &Section);
  void defineLabel(const MCSymbolGOFF &Symbol);
  void defineSymbols();

public:
  GOFFWriter(raw_pwrite_stream &OS, MCAssembler &Asm);
  uint64_t writeObject();
};
} // namespace

GOFFWriter::GOFFWriter(raw_pwrite_stream &OS, MCAssembler &Asm)
    : OS(OS), Asm(Asm) {}

void GOFFWriter::defineSectionSymbols(const MCSectionGOFF &Section) {
  if (Section.isSD()) {
    GOFFSymbol SD(Section.getName(), Section.getOrdinal(),
                  Section.getSDAttributes());
    writeSymbol(SD);
  }

  if (Section.isED()) {
    GOFFSymbol ED(Section.getName(), Section.getOrdinal(),
                  Section.getParent()->getOrdinal(), Section.getEDAttributes());
    ED.SectionLength = Asm.getSectionAddressSize(Section);
    writeSymbol(ED);
  }

  if (Section.isPR()) {
    MCSectionGOFF *Parent = Section.getParent();
    GOFFSymbol PR(Section.getName(), Section.getOrdinal(), Parent->getOrdinal(),
                  Parent->getEDAttributes(), Section.getPRAttributes());
    PR.SectionLength = Asm.getSectionAddressSize(Section);
    if (Section.requiresNonZeroLength()) {
      // We cannot have a zero-length section for data.  If we do,
      // artificially inflate it. Use 2 bytes to avoid odd alignments. Note:
      // if this is ever changed, you will need to update the code in
      // SystemZAsmPrinter::emitCEEMAIN and SystemZAsmPrinter::emitCELQMAIN to
      // generate -1 if there is no ADA
      if (!PR.SectionLength)
        PR.SectionLength = 2;
    }
    writeSymbol(PR);
  }
}

void GOFFWriter::defineLabel(const MCSymbolGOFF &Symbol) {
  MCSectionGOFF &Section = static_cast<MCSectionGOFF &>(Symbol.getSection());
  GOFFSymbol LD(Symbol.getName(), Symbol.getIndex(), Section.getOrdinal(),
                Section.getEDAttributes().NameSpace, Symbol.getLDAttributes());
  if (Symbol.getADA())
    LD.ADAEsdId = Symbol.getADA()->getOrdinal();
  writeSymbol(LD);
}

void GOFFWriter::defineSymbols() {
  unsigned Ordinal = 0;
  // Process all sections.
  for (MCSection &S : Asm) {
    auto &Section = static_cast<MCSectionGOFF &>(S);
    Section.setOrdinal(++Ordinal);
    defineSectionSymbols(Section);
  }

  // Process all symbols
  for (const MCSymbol &Sym : Asm.symbols()) {
    if (Sym.isTemporary())
      continue;
    auto &Symbol = static_cast<const MCSymbolGOFF &>(Sym);
    if (Symbol.hasLDAttributes()) {
      Symbol.setIndex(++Ordinal);
      defineLabel(Symbol);
    }
  }
}

void GOFFWriter::writeHeader() {
  OS.newRecord(GOFF::RT_HDR);
  OS.write_zeros(1);       // Reserved
  OS.writebe<uint32_t>(0); // Target Hardware Environment
  OS.writebe<uint32_t>(0); // Target Operating System Environment
  OS.write_zeros(2);       // Reserved
  OS.writebe<uint16_t>(0); // CCSID
  OS.write_zeros(16);      // Character Set name
  OS.write_zeros(16);      // Language Product Identifier
  OS.writebe<uint32_t>(1); // Architecture Level
  OS.writebe<uint16_t>(0); // Module Properties Length
  OS.write_zeros(6);       // Reserved
}

void GOFFWriter::writeSymbol(const GOFFSymbol &Symbol) {
  if (Symbol.Offset >= (((uint64_t)1) << 31))
    report_fatal_error("ESD offset out of range");

  // All symbol names are in EBCDIC.
  SmallString<256> Name;
  ConverterEBCDIC::convertToEBCDIC(Symbol.Name, Name);

  // Check length here since this number is technically signed but we need uint
  // for writing to records.
  if (Name.size() >= GOFF::MaxDataLength)
    report_fatal_error("Symbol max name length exceeded");
  uint16_t NameLength = Name.size();

  OS.newRecord(GOFF::RT_ESD);
  OS.writebe<uint8_t>(Symbol.SymbolType);   // Symbol Type
  OS.writebe<uint32_t>(Symbol.EsdId);       // ESDID
  OS.writebe<uint32_t>(Symbol.ParentEsdId); // Parent or Owning ESDID
  OS.writebe<uint32_t>(0);                  // Reserved
  OS.writebe<uint32_t>(
      static_cast<uint32_t>(Symbol.Offset));     // Offset or Address
  OS.writebe<uint32_t>(0);                       // Reserved
  OS.writebe<uint32_t>(Symbol.SectionLength);    // Length
  OS.writebe<uint32_t>(Symbol.EASectionEDEsdId); // Extended Attribute ESDID
  OS.writebe<uint32_t>(Symbol.EASectionOffset);  // Extended Attribute Offset
  OS.writebe<uint32_t>(0);                       // Reserved
  OS.writebe<uint8_t>(Symbol.NameSpace);         // Name Space ID
  OS.writebe<uint8_t>(Symbol.SymbolFlags);       // Flags
  OS.writebe<uint8_t>(Symbol.FillByteValue);     // Fill-Byte Value
  OS.writebe<uint8_t>(0);                        // Reserved
  OS.writebe<uint32_t>(Symbol.ADAEsdId);         // ADA ESDID
  OS.writebe<uint32_t>(Symbol.SortKey);          // Sort Priority
  OS.writebe<uint64_t>(0);                       // Reserved
  for (auto F : Symbol.BehavAttrs.Attr)
    OS.writebe<uint8_t>(F);          // Behavioral Attributes
  OS.writebe<uint16_t>(NameLength);  // Name Length
  OS.write(Name.data(), NameLength); // Name
}

namespace {
/// Adapter stream to write a text section.
class TextStream : public raw_ostream {
  /// The underlying GOFFOstream.
  GOFFOstream &OS;

  /// The buffer size is the maximum number of bytes in a TXT section.
  static constexpr size_t BufferSize = GOFF::MaxDataLength;

  /// Static allocated buffer for the stream, used by the raw_ostream class. The
  /// buffer is sized to hold the payload of a logical TXT record.
  char Buffer[BufferSize];

  /// The offset for the next TXT record. This is equal to the number of bytes
  /// written.
  size_t Offset;

  /// The Esdid of the GOFF section.
  const uint32_t EsdId;

  /// The record style.
  const GOFF::ESDTextStyle RecordStyle;

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  uint64_t current_pos() const override { return Offset; }

public:
  explicit TextStream(GOFFOstream &OS, uint32_t EsdId,
                      GOFF::ESDTextStyle RecordStyle)
      : OS(OS), Offset(0), EsdId(EsdId), RecordStyle(RecordStyle) {
    SetBuffer(Buffer, sizeof(Buffer));
  }

  ~TextStream() { flush(); }
};
} // namespace

void TextStream::write_impl(const char *Ptr, size_t Size) {
  size_t WrittenLength = 0;

  // We only have signed 32bits of offset.
  if (Offset + Size > std::numeric_limits<int32_t>::max())
    report_fatal_error("TXT section too large");

  while (WrittenLength < Size) {
    size_t ToWriteLength =
        std::min(Size - WrittenLength, size_t(GOFF::MaxDataLength));

    OS.newRecord(GOFF::RT_TXT);
    OS.writebe<uint8_t>(GOFF::Flags(4, 4, RecordStyle)); // Text Record Style
    OS.writebe<uint32_t>(EsdId);                         // Element ESDID
    OS.writebe<uint32_t>(0);                             // Reserved
    OS.writebe<uint32_t>(static_cast<uint32_t>(Offset)); // Offset
    OS.writebe<uint32_t>(0);                      // Text Field True Length
    OS.writebe<uint16_t>(0);                      // Text Encoding
    OS.writebe<uint16_t>(ToWriteLength);          // Data Length
    OS.write(Ptr + WrittenLength, ToWriteLength); // Data

    WrittenLength += ToWriteLength;
    Offset += ToWriteLength;
  }
}

void GOFFWriter::writeText(const MCSectionGOFF *Section) {
  // A BSS section contains only zeros, no need to write this.
  if (Section->isBSS())
    return;

  TextStream S(OS, Section->getOrdinal(), Section->getTextStyle());
  Asm.writeSectionData(S, Section);
}

void GOFFWriter::writeEnd() {
  uint8_t F = GOFF::END_EPR_None;
  uint8_t AMODE = 0;
  uint32_t ESDID = 0;

  // TODO Set Flags/AMODE/ESDID for entry point.

  OS.newRecord(GOFF::RT_END);
  OS.writebe<uint8_t>(GOFF::Flags(6, 2, F)); // Indicator flags
  OS.writebe<uint8_t>(AMODE);                // AMODE
  OS.write_zeros(3);                         // Reserved
  // The record count is the number of logical records. In principle, this value
  // is available as OS.logicalRecords(). However, some tools rely on this field
  // being zero.
  OS.writebe<uint32_t>(0);     // Record Count
  OS.writebe<uint32_t>(ESDID); // ESDID (of entry point)
}

uint64_t GOFFWriter::writeObject() {
  writeHeader();

  defineSymbols();

  for (const MCSection &Section : Asm)
    writeText(static_cast<const MCSectionGOFF *>(&Section));

  writeEnd();

  // Make sure all records are written.
  OS.finalizeRecord();

  LLVM_DEBUG(dbgs() << "Wrote " << OS.getNumLogicalRecords()
                    << " logical records.");

  return OS.getWrittenSize();
}

GOFFObjectWriter::GOFFObjectWriter(
    std::unique_ptr<MCGOFFObjectTargetWriter> MOTW, raw_pwrite_stream &OS)
    : TargetObjectWriter(std::move(MOTW)), OS(OS) {}

GOFFObjectWriter::~GOFFObjectWriter() {}

uint64_t GOFFObjectWriter::writeObject() {
  uint64_t Size = GOFFWriter(OS, *Asm).writeObject();
  return Size;
}

std::unique_ptr<MCObjectWriter>
llvm::createGOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                             raw_pwrite_stream &OS) {
  return std::make_unique<GOFFObjectWriter>(std::move(MOTW), OS);
}
