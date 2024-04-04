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
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "goff-writer"

namespace {

// The standard System/390 convention is to name the high-order (leftmost) bit
// in a byte as bit zero. The Flags type helps to set bits in a byte according
// to this numeration order.
class Flags {
  uint8_t Val;

  constexpr static uint8_t bits(uint8_t BitIndex, uint8_t Length, uint8_t Value,
                                uint8_t OldValue) {
    assert(BitIndex < 8 && "Bit index out of bounds!");
    assert(Length + BitIndex <= 8 && "Bit length too long!");

    uint8_t Mask = ((1 << Length) - 1) << (8 - BitIndex - Length);
    Value = Value << (8 - BitIndex - Length);
    assert((Value & Mask) == Value && "Bits set outside of range!");

    return (OldValue & ~Mask) | Value;
  }

public:
  constexpr Flags() : Val(0) {}
  constexpr Flags(uint8_t BitIndex, uint8_t Length, uint8_t Value)
      : Val(bits(BitIndex, Length, Value, 0)) {}

  void set(uint8_t BitIndex, uint8_t Length, uint8_t Value) {
    Val = bits(BitIndex, Length, Value, Val);
  }

  constexpr operator uint8_t() const { return Val; }
};

// Common flag values on records.

// Flag: This record is continued.
constexpr uint8_t RecContinued = Flags(7, 1, 1);

// Flag: This record is a continuation.
constexpr uint8_t RecContinuation = Flags(6, 1, 1);

// The GOFFOstream is responsible to write the data into the fixed physical
// records of the format. A user of this class announces the start of a new
// logical record and the size of its content. While writing the content, the
// physical records are created for the data. Possible fill bytes at the end of
// a physical record are written automatically. In principle, the GOFFOstream
// is agnostic of the endianness of the content. However, it also supports
// writing data in big endian byte order.
class GOFFOstream : public raw_ostream {
  /// The underlying raw_pwrite_stream.
  raw_pwrite_stream &OS;

  /// The remaining size of this logical record, including fill bytes.
  size_t RemainingSize;

#ifndef NDEBUG
  /// The number of bytes needed to fill up the last physical record.
  size_t Gap = 0;
#endif

  /// The number of logical records emitted to far.
  uint32_t LogicalRecords;

  /// The type of the current (logical) record.
  GOFF::RecordType CurrentType;

  /// Signals start of new record.
  bool NewLogicalRecord;

  /// Static allocated buffer for the stream, used by the raw_ostream class. The
  /// buffer is sized to hold the content of a physical record.
  char Buffer[GOFF::RecordContentLength];

  // Return the number of bytes left to write until next physical record.
  // Please note that we maintain the total numbers of byte left, not the
  // written size.
  size_t bytesToNextPhysicalRecord() {
    size_t Bytes = RemainingSize % GOFF::RecordContentLength;
    return Bytes ? Bytes : GOFF::RecordContentLength;
  }

  /// Write the record prefix of a physical record, using the given record type.
  static void writeRecordPrefix(raw_ostream &OS, GOFF::RecordType Type,
                                size_t RemainingSize,
                                uint8_t Flags = RecContinuation);

  /// Fill the last physical record of a logical record with zero bytes.
  void fillRecord();

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  /// Return the current position within the stream, not counting the bytes
  /// currently in the buffer.
  uint64_t current_pos() const override { return OS.tell(); }

public:
  explicit GOFFOstream(raw_pwrite_stream &OS)
      : OS(OS), RemainingSize(0), LogicalRecords(0), NewLogicalRecord(false) {
    SetBuffer(Buffer, sizeof(Buffer));
  }

  ~GOFFOstream() { finalize(); }

  raw_pwrite_stream &getOS() { return OS; }

  void newRecord(GOFF::RecordType Type, size_t Size);

  void finalize() { fillRecord(); }

  uint32_t logicalRecords() { return LogicalRecords; }

  // Support for endian-specific data.
  template <typename value_type> void writebe(value_type Value) {
    Value =
        support::endian::byte_swap<value_type>(Value, llvm::endianness::big);
    write(reinterpret_cast<const char *>(&Value), sizeof(value_type));
  }
};

void GOFFOstream::writeRecordPrefix(raw_ostream &OS, GOFF::RecordType Type,
                                    size_t RemainingSize, uint8_t Flags) {
  uint8_t TypeAndFlags = Flags | (Type << 4);
  if (RemainingSize > GOFF::RecordLength)
    TypeAndFlags |= RecContinued;
  OS << static_cast<unsigned char>(GOFF::PTVPrefix) // Record Type
     << static_cast<unsigned char>(TypeAndFlags)    // Continuation
     << static_cast<unsigned char>(0);              // Version
}

void GOFFOstream::newRecord(GOFF::RecordType Type, size_t Size) {
  fillRecord();
  CurrentType = Type;
  RemainingSize = Size;
#ifdef NDEBUG
  size_t Gap;
#endif
  Gap = (RemainingSize % GOFF::RecordContentLength);
  if (Gap) {
    Gap = GOFF::RecordContentLength - Gap;
    RemainingSize += Gap;
  }
  NewLogicalRecord = true;
  ++LogicalRecords;
}

void GOFFOstream::fillRecord() {
  assert((GetNumBytesInBuffer() <= RemainingSize) &&
         "More bytes in buffer than expected");
  size_t Remains = RemainingSize - GetNumBytesInBuffer();
  if (Remains) {
    assert(Remains == Gap && "Wrong size of fill gap");
    assert((Remains < GOFF::RecordLength) &&
           "Attempt to fill more than one physical record");
    raw_ostream::write_zeros(Remains);
  }
  flush();
  assert(RemainingSize == 0 && "Not fully flushed");
  assert(GetNumBytesInBuffer() == 0 && "Buffer not fully empty");
}

// This function is called from the raw_ostream implementation if:
// - The internal buffer is full. Size is excactly the size of the buffer.
// - Data larger than the internal buffer is written. Size is a multiple of the
//   buffer size.
// - flush() has been called. Size is at most the buffer size.
// The GOFFOstream implementation ensures that flush() is called before a new
// logical record begins. Therefore it is sufficient to check for a new block
// only once.
void GOFFOstream::write_impl(const char *Ptr, size_t Size) {
  assert((RemainingSize >= Size) && "Attempt to write too much data");
  assert(RemainingSize && "Logical record overflow");
  if (!(RemainingSize % GOFF::RecordContentLength)) {
    writeRecordPrefix(OS, CurrentType, RemainingSize,
                      NewLogicalRecord ? 0 : RecContinuation);
    NewLogicalRecord = false;
  }
  assert(!NewLogicalRecord &&
         "New logical record not on physical record boundary");

  size_t Idx = 0;
  while (Size > 0) {
    size_t BytesToWrite = bytesToNextPhysicalRecord();
    if (BytesToWrite > Size)
      BytesToWrite = Size;
    OS.write(Ptr + Idx, BytesToWrite);
    Idx += BytesToWrite;
    Size -= BytesToWrite;
    RemainingSize -= BytesToWrite;
    if (Size)
      writeRecordPrefix(OS, CurrentType, RemainingSize);
  }
}

/// \brief Wrapper class for symbols used exclusively for the symbol table in a
/// GOFF file.
class GOFFSymbol {
public:
  std::string Name;
  uint32_t EsdId;
  uint32_t ParentEsdId;
  const MCSymbolGOFF *MCSym;
  GOFF::ESDSymbolType SymbolType;

  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_NormalName;
  GOFF::ESDAmode Amode = GOFF::ESD_AMODE_64;
  GOFF::ESDRmode Rmode = GOFF::ESD_RMODE_64;
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Byte;
  GOFF::ESDTextStyle TextStyle = GOFF::ESD_TS_ByteOriented;
  GOFF::ESDBindingAlgorithm BindAlgorithm = GOFF::ESD_BA_Concatenate;
  GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
  GOFF::ESDBindingStrength BindingStrength = GOFF::ESD_BST_Strong;
  GOFF::ESDReserveQwords ReservedQwords = GOFF::ESD_RQ_0;

  uint32_t SortKey = 0;
  uint32_t SectionLength = 0;
  uint32_t ADAEsdId = 0;
  bool Indirect = false;
  bool ForceRent = false;
  bool Renamable = false;
  bool ReadOnly = false;
  uint32_t EASectionEsdId = 0;
  uint32_t EASectionOffset = 0;

  GOFFSymbol(StringRef Name, GOFF::ESDSymbolType Type, uint32_t EsdID,
             uint32_t ParentEsdID)
      : Name(Name.data(), Name.size()), EsdId(EsdID), ParentEsdId(ParentEsdID),
        MCSym(nullptr), SymbolType(Type) {}
  GOFFSymbol() {}

  bool isForceRent() const { return ForceRent; }
  bool isReadOnly() const { return ReadOnly; }
  bool isRemovable() const { return false; }
  bool isExecutable() const { return Executable == GOFF::ESD_EXE_CODE; }
  bool isExecUnspecified() const {
    return Executable == GOFF::ESD_EXE_Unspecified;
  }
  bool isWeakRef() const { return BindingStrength == GOFF::ESD_BST_Weak; }
  bool isExternal() const {
    return (BindingScope == GOFF::ESD_BSC_Library) ||
           (BindingScope == GOFF::ESD_BSC_ImportExport);
  }

  static GOFF::ESDAlignment setGOFFAlignment(Align A) {
    // The GOFF alignment is encoded as log_2 value.
    GOFF::ESDAlignment Alignment;
    uint8_t Log = Log2(A);
    if (Log <= GOFF::ESD_ALIGN_4Kpage)
      Alignment = static_cast<GOFF::ESDAlignment>(Log);
    else
      llvm_unreachable("Unsupported alignment");
    return Alignment;
  }
};

/// \brief Wrapper class for sections used exclusively for representing sections
/// of the GOFF output that have actual bytes.  This could be a ED or a PR.
/// Relocations will always have a P-pointer to the ESDID of one of these.
struct GOFFSection {
  uint32_t PEsdId;
  uint32_t REsdId;
  uint32_t SDEsdId;
  const MCSectionGOFF *MC;

  GOFFSection(uint32_t PEsdId, uint32_t REsdId, uint32_t SDEsdId)
      : PEsdId(PEsdId), REsdId(REsdId), SDEsdId(SDEsdId), MC(nullptr) {}
  GOFFSection(uint32_t PEsdId, uint32_t REsdId, uint32_t SDEsdId,
              const MCSectionGOFF *MC)
      : PEsdId(PEsdId), REsdId(REsdId), SDEsdId(SDEsdId), MC(MC) {}
};

// An MCSymbol may map to up to two GOFF Symbols. The first is the
// "true" underlying symbol, the second one represents any indirect
// references to that symbol.
class GOFFObjectWriter : public MCObjectWriter {
  typedef DenseMap<MCSection const *, GOFFSection> SectionMapType;

  // The target specific GOFF writer instance.
  std::unique_ptr<MCGOFFObjectTargetWriter> TargetObjectWriter;

  /// Lookup table for MCSections to GOFFSections.  Needed to determine
  /// SymbolType on GOFFSymbols that reside in GOFFSections.
  SectionMapType SectionMap;

  // The stream used to write the GOFF records.
  GOFFOstream OS;

  uint32_t EsdCounter = 1;

  // The "root" SD node. This should have ESDID = 1
  GOFFSymbol RootSD;
  uint32_t ADAPREsdId;
  uint32_t EntryEDEsdId;
  uint32_t CodeLDEsdId;

public:
  GOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                   raw_pwrite_stream &OS)
      : TargetObjectWriter(std::move(MOTW)), OS(OS) {}

  ~GOFFObjectWriter() override {}

private:
  // Write GOFF records.
  void writeHeader();

  void writeSymbol(const GOFFSymbol &Symbol, const MCAsmLayout &Layout);

  void writeADAandCodeSectionSymbols(MCAssembler &Asm,
                                     const MCAsmLayout &Layout);
  // Write out all ESD Symbols that own a section. GOFF doesn't have
  // "sections" in the way other object file formats like ELF do; instead
  // a GOFF section is defined as a triple of ESD Symbols (SD, ED, PR/LD).
  // The PR/LD symbol should own a TXT record that contains the actual
  // data of the section.
  void writeSectionSymbols(MCAssembler &Asm, const MCAsmLayout &Layout);

  void writeText(const MCSectionGOFF *MCSec, uint32_t EsdId,
                 GOFF::TXTRecordStyle RecordStyle, MCAssembler &Asm,
                 const MCAsmLayout &Layout);

  void writeEnd();

public:
  // Implementation of the MCObjectWriter interface.
  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override {}
  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;
  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

private:
  GOFFSymbol createGOFFSymbol(StringRef Name, GOFF::ESDSymbolType Type,
                              uint32_t ParentEsdId);
  GOFFSymbol createSDSymbol(StringRef Name);
  GOFFSymbol
  createEDSymbol(StringRef Name, uint32_t ParentEsdId,
                 GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Doubleword);
  GOFFSymbol
  createEDSymbol(StringRef Name, uint32_t ParentEsdId, uint32_t SectionLength,
                 GOFF::ESDExecutable Executable, bool ForceRent,
                 GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Doubleword);
  GOFFSymbol createEDSymbol(
      StringRef Name, uint32_t ParentEsdId, uint32_t SectionLength,
      GOFF::ESDExecutable Executable, GOFF::ESDBindingScope BindingScope,
      GOFF::ESDNameSpaceId NameSpaceId,
      GOFF::ESDBindingAlgorithm BindingAlgorithm, bool ReadOnly,
      GOFF::ESDTextStyle TextStyle, GOFF::ESDLoadingBehavior LoadBehavior,
      GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Doubleword);
  GOFFSymbol createLDSymbol(StringRef Name, uint32_t ParentEsdId);
  GOFFSymbol createERSymbol(StringRef Name, uint32_t ParentEsdId,
                            const MCSymbolGOFF *Source = nullptr);
  GOFFSymbol createPRSymbol(StringRef Name, uint32_t ParentEsdId);
  GOFFSymbol
  createPRSymbol(StringRef Name, uint32_t ParentEsdId,
                 GOFF::ESDNameSpaceId NameSpaceType,
                 GOFF::ESDExecutable Executable, GOFF::ESDAlignment Alignment,
                 GOFF::ESDBindingScope BindingScope, uint32_t SectionLength,
                 GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial,
                 GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink);
  GOFFSymbol createWSASymbol(uint32_t ParentEsdId, bool ReserveQWords = false);

  // Define the root SD node as well as the symbols that make up the
  // ADA section.
  void defineRootSD(MCAssembler &Asm, const MCAsmLayout &Layout);
  void writeSymbolDefinedInModule(const MCSymbolGOFF &Symbol, MCAssembler &Asm,
                                  const MCAsmLayout &Layout);
  void writeSymbolDeclaredInModule(const MCSymbolGOFF &Symbol, MCAssembler &Asm,
                                   const MCAsmLayout &Layout);
};
} // end anonymous namespace

GOFFSymbol GOFFObjectWriter::createGOFFSymbol(StringRef Name,
                                              GOFF::ESDSymbolType Type,
                                              uint32_t ParentEsdId) {
  return GOFFSymbol(Name, Type, EsdCounter++, ParentEsdId);
}

GOFFSymbol GOFFObjectWriter::createSDSymbol(StringRef Name) {
  return createGOFFSymbol(Name, GOFF::ESD_ST_SectionDefinition, 0);
}

GOFFSymbol GOFFObjectWriter::createEDSymbol(StringRef Name,
                                            uint32_t ParentEsdId,
                                            GOFF::ESDAlignment Alignment) {
  GOFFSymbol ED =
      createGOFFSymbol(Name, GOFF::ESD_ST_ElementDefinition, ParentEsdId);

  ED.Alignment = Alignment;
  return ED;
}

GOFFSymbol GOFFObjectWriter::createEDSymbol(StringRef Name,
                                            uint32_t ParentEsdId,
                                            uint32_t SectionLength,
                                            GOFF::ESDExecutable Executable,
                                            bool ForceRent,
                                            GOFF::ESDAlignment Alignment) {
  GOFFSymbol ED =
      createGOFFSymbol(Name, GOFF::ESD_ST_ElementDefinition, ParentEsdId);
  ED.SectionLength = 0;
  ED.Executable = Executable;
  ED.ForceRent = ForceRent;
  ED.Alignment = Alignment;
  return ED;
}

GOFFSymbol GOFFObjectWriter::createEDSymbol(
    StringRef Name, uint32_t ParentEsdId, uint32_t SectionLength,
    GOFF::ESDExecutable Executable, GOFF::ESDBindingScope BindingScope,
    GOFF::ESDNameSpaceId NameSpaceId,
    GOFF::ESDBindingAlgorithm BindingAlgorithm, bool ReadOnly,
    GOFF::ESDTextStyle TextStyle, GOFF::ESDLoadingBehavior LoadBehavior,
    GOFF::ESDAlignment Alignment) {
  GOFFSymbol ED =
      createGOFFSymbol(Name, GOFF::ESD_ST_ElementDefinition, ParentEsdId);
  ED.SectionLength = 0;
  ED.Executable = Executable;
  ED.BindingScope = BindingScope;
  ED.NameSpace = NameSpaceId;
  ED.BindAlgorithm = BindingAlgorithm;
  ED.ReadOnly = ReadOnly;
  ED.TextStyle = TextStyle;
  ED.LoadBehavior = LoadBehavior;
  ED.Alignment = Alignment;
  return ED;
}

GOFFSymbol GOFFObjectWriter::createLDSymbol(StringRef Name,
                                            uint32_t ParentEsdId) {
  return createGOFFSymbol(Name, GOFF::ESD_ST_LabelDefinition, ParentEsdId);
}

GOFFSymbol GOFFObjectWriter::createERSymbol(StringRef Name,
                                            uint32_t ParentEsdId,
                                            const MCSymbolGOFF *Source) {
  GOFFSymbol ER =
      createGOFFSymbol(Name, GOFF::ESD_ST_ExternalReference, ParentEsdId);

  if (Source) {
    ER.Linkage = Source->isOSLinkage() ? GOFF::ESDLinkageType::ESD_LT_OS
                                       : GOFF::ESDLinkageType::ESD_LT_XPLink;
    ER.Executable = Source->getExecutable();
    ER.BindingScope = Source->isExternal()
                          ? GOFF::ESDBindingScope::ESD_BSC_Library
                          : GOFF::ESDBindingScope::ESD_BSC_Section;
    ER.BindingStrength = Source->isWeak()
                             ? GOFF::ESDBindingStrength::ESD_BST_Weak
                             : GOFF::ESDBindingStrength::ESD_BST_Strong;
  }

  return ER;
}

GOFFSymbol GOFFObjectWriter::createPRSymbol(StringRef Name,
                                            uint32_t ParentEsdId) {
  return createGOFFSymbol(Name, GOFF::ESD_ST_PartReference, ParentEsdId);
}

GOFFSymbol GOFFObjectWriter::createPRSymbol(
    StringRef Name, uint32_t ParentEsdId, GOFF::ESDNameSpaceId NameSpaceType,
    GOFF::ESDExecutable Executable, GOFF::ESDAlignment Alignment,
    GOFF::ESDBindingScope BindingScope, uint32_t SectionLength,
    GOFF::ESDLoadingBehavior LoadBehavior, GOFF::ESDLinkageType Linkage) {
  GOFFSymbol PR =
      createGOFFSymbol(Name, GOFF::ESD_ST_PartReference, ParentEsdId);
  PR.NameSpace = NameSpaceType;
  PR.Executable = Executable;
  PR.Alignment = Alignment;
  PR.BindingScope = BindingScope;
  PR.SectionLength = SectionLength;
  PR.LoadBehavior = LoadBehavior;
  PR.Linkage = Linkage;
  return PR;
}

GOFFSymbol GOFFObjectWriter::createWSASymbol(uint32_t ParentEsdId,
                                             bool ReserveQwords) {
  const char *WSAClassName = "C_WSA64";
  GOFFSymbol WSA = createEDSymbol(WSAClassName, ParentEsdId);

  WSA.Executable = GOFF::ESD_EXE_DATA;
  WSA.TextStyle = GOFF::ESD_TS_ByteOriented;
  WSA.BindAlgorithm = GOFF::ESD_BA_Merge;
  WSA.Alignment = GOFF::ESD_ALIGN_Quadword;
  WSA.LoadBehavior = GOFF::ESD_LB_Deferred;
  WSA.NameSpace = GOFF::ESD_NS_Parts;
  WSA.SectionLength = 0;
  WSA.ReservedQwords = ReserveQwords ? GOFF::ESD_RQ_1 : GOFF::ESD_RQ_0;

  return WSA;
}

static uint32_t getADASectionLength(MCAssembler &Asm,
                                    const MCAsmLayout &Layout) {
  uint32_t SecLen = 0;
  for (const MCSection &MCSec : Asm) {
    auto &GSec = cast<MCSectionGOFF>(MCSec);
    if (GSec.isStatic()) {
      SecLen = Layout.getSectionAddressSize(&MCSec);
    }
  }

  // The ADA section is not allowed to be zero-length. We also want to
  // avoid odd alignments, so we use 2 bytes.
  return std::max(SecLen, 2u);
}

void GOFFObjectWriter::defineRootSD(MCAssembler &Asm,
                                    const MCAsmLayout &Layout) {
  StringRef FileName = "";
  if (!Asm.getFileNames().empty())
    FileName = sys::path::stem((*(Asm.getFileNames().begin())).first);
  RootSD = createSDSymbol(FileName.str().append("#C"));
  RootSD.BindingScope = GOFF::ESD_BSC_Section;
  RootSD.Executable = GOFF::ESD_EXE_CODE;
}

void GOFFObjectWriter::writeSymbolDefinedInModule(const MCSymbolGOFF &Symbol,
                                                  MCAssembler &Asm,
                                                  const MCAsmLayout &Layout) {
  MCSection &Section = Symbol.getSection();
  SectionKind Kind = Section.getKind();
  auto &Sec = cast<MCSectionGOFF>(Section);

  StringRef SymbolName = Symbol.getName();
  // If it's a text section, then create a label for it.
  if (Kind.isText()) {
    auto &GSection = SectionMap.at(&Sec);
    GOFFSymbol LD = createLDSymbol(SymbolName, GSection.PEsdId);
    LD.BindingStrength = Symbol.isWeak()
                             ? GOFF::ESDBindingStrength::ESD_BST_Weak
                             : GOFF::ESDBindingStrength::ESD_BST_Strong;

    // If we don't know if it is code or data, assume it is code.
    LD.Executable = Symbol.getExecutable();
    if (LD.isExecUnspecified())
      LD.Executable = GOFF::ESD_EXE_CODE;

    // Determine the binding scope. Please note that the combination
    // !isExternal && isExported makes no sense.
    LD.BindingScope =
        Symbol.isExternal()
            ? (Symbol.isExported() ? GOFF::ESD_BSC_ImportExport
                                   : (LD.isExecutable() ? GOFF::ESD_BSC_Library
                                                        : GOFF::ESD_BSC_Module))
            : GOFF::ESD_BSC_Section;

    LD.ADAEsdId = ADAPREsdId;

    LD.MCSym = &Symbol;
    writeSymbol(LD, Layout);
  } else if ((Kind.isBSS() || Kind.isData() || Kind.isThreadData()) &&
             Symbol.isAlias()) {
    // Alias to variable at the object file level.
    // Not supported in RENT mode.
    llvm_unreachable("Alias to rent variable is unsupported");
  } else if (Kind.isBSS() || Kind.isData() || Kind.isThreadData()) {
    std::string SectionName = Section.getName().str();
    GOFFSymbol SD = createSDSymbol(SectionName);
    auto BindingScope = Symbol.isExternal()
                            ? (Symbol.isExported() ? GOFF::ESD_BSC_ImportExport
                                                   : GOFF::ESD_BSC_Library)
                            : GOFF::ESD_BSC_Section;
    if (!Symbol.isExternal())
      SD.BindingScope = GOFF::ESD_BSC_Section;

    GOFFSymbol ED = createWSASymbol(SD.EsdId);
    uint32_t PRSectionLen = Layout.getSectionAddressSize(&Section)
                                ? Layout.getSectionAddressSize(&Section)
                                : 0;
    GOFFSymbol PR = createPRSymbol(
        SectionName, ED.EsdId, GOFF::ESD_NS_Parts, GOFF::ESD_EXE_DATA,
        GOFFSymbol::setGOFFAlignment(Section.getAlign()), BindingScope,
        PRSectionLen);
    ED.Alignment =
        std::max(static_cast<GOFF::ESDAlignment>(Log2(Section.getAlign())),
                 GOFF::ESD_ALIGN_Quadword);

    writeSymbol(SD, Layout);
    writeSymbol(ED, Layout);
    writeSymbol(PR, Layout);

    GOFFSection GoffSec = GOFFSection(PR.EsdId, PR.EsdId, SD.EsdId);
    SectionMap.insert(std::make_pair(&Section, GoffSec));
  } else
    llvm_unreachable("Unhandled section kind for Symbol");
}

void GOFFObjectWriter::writeSymbolDeclaredInModule(const MCSymbolGOFF &Symbol,
                                                   MCAssembler &Asm,
                                                   const MCAsmLayout &Layout) {
  GOFFSymbol SD = RootSD;

  GOFF::ESDExecutable Exec = Symbol.getExecutable();

  GOFFSymbol GSym;
  switch (Exec) {
  case GOFF::ESD_EXE_CODE:
  case GOFF::ESD_EXE_Unspecified: {
    GOFFSymbol ER = createERSymbol(Symbol.getName(), SD.EsdId, &Symbol);
    ER.BindingScope = GOFF::ESD_BSC_ImportExport;
    GSym = ER;
    break;
  }
  case GOFF::ESD_EXE_DATA: {
    auto SymbAlign = Symbol.getCommonAlignment().valueOrOne();
    GOFFSymbol ED = createWSASymbol(SD.EsdId);
    writeSymbol(ED, Layout);
    GOFFSymbol PR = createPRSymbol(
        Symbol.getName(), ED.EsdId, GOFF::ESD_NS_Parts,
        GOFF::ESD_EXE_Unspecified, GOFFSymbol::setGOFFAlignment(SymbAlign),
        GOFF::ESD_BSC_ImportExport, 0);
    GSym = PR;
    break;
  }
  }

  GSym.BindingScope =
      Symbol.isExported() ? GOFF::ESD_BSC_ImportExport : GOFF::ESD_BSC_Library;

  writeSymbol(GSym, Layout);
}

void GOFFObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
  LLVM_DEBUG(dbgs() << "Entering " << __FUNCTION__ << "\n");

  defineRootSD(Asm, Layout);
}

void GOFFObjectWriter::writeHeader() {
  OS.newRecord(GOFF::RT_HDR, /*Size=*/57);
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

void GOFFObjectWriter::writeSymbol(const GOFFSymbol &Symbol,
                                   const MCAsmLayout &Layout) {
  uint32_t Offset = 0;
  uint32_t Length = 0;
  GOFF::ESDNameSpaceId NameSpaceId = GOFF::ESD_NS_ProgramManagementBinder;
  Flags SymbolFlags;
  uint8_t FillByteValue = 0;

  Flags BehavAttrs[10] = {};
  auto setAmode = [&BehavAttrs](GOFF::ESDAmode Amode) {
    BehavAttrs[0].set(0, 8, Amode);
  };
  auto setRmode = [&BehavAttrs](GOFF::ESDRmode Rmode) {
    BehavAttrs[1].set(0, 8, Rmode);
  };
  auto setTextStyle = [&BehavAttrs](GOFF::ESDTextStyle Style) {
    BehavAttrs[2].set(0, 4, Style);
  };
  auto setBindingAlgorithm =
      [&BehavAttrs](GOFF::ESDBindingAlgorithm Algorithm) {
        BehavAttrs[2].set(4, 4, Algorithm);
      };
  auto setTaskingBehavior =
      [&BehavAttrs](GOFF::ESDTaskingBehavior TaskingBehavior) {
        BehavAttrs[3].set(0, 3, TaskingBehavior);
      };
  auto setReadOnly = [&BehavAttrs](bool ReadOnly) {
    BehavAttrs[3].set(4, 1, ReadOnly);
  };
  auto setExecutable = [&BehavAttrs](GOFF::ESDExecutable Executable) {
    BehavAttrs[3].set(5, 3, Executable);
  };
  auto setDuplicateSeverity =
      [&BehavAttrs](GOFF::ESDDuplicateSymbolSeverity DSS) {
        BehavAttrs[4].set(2, 2, DSS);
      };
  auto setBindingStrength = [&BehavAttrs](GOFF::ESDBindingStrength Strength) {
    BehavAttrs[4].set(4, 4, Strength);
  };
  auto setLoadingBehavior = [&BehavAttrs](GOFF::ESDLoadingBehavior Behavior) {
    BehavAttrs[5].set(0, 2, Behavior);
  };
  auto setIndirectReference = [&BehavAttrs](bool Indirect) {
    uint8_t Value = Indirect ? 1 : 0;
    BehavAttrs[5].set(3, 1, Value);
  };
  auto setBindingScope = [&BehavAttrs](GOFF::ESDBindingScope Scope) {
    BehavAttrs[5].set(4, 4, Scope);
  };
  auto setLinkageType = [&BehavAttrs](GOFF::ESDLinkageType Type) {
    BehavAttrs[6].set(2, 1, Type);
  };
  auto setAlignment = [&BehavAttrs](GOFF::ESDAlignment Alignment) {
    BehavAttrs[6].set(3, 5, Alignment);
  };

  uint32_t AdaEsdId = 0;
  uint32_t SortPriority = 0;

  switch (Symbol.SymbolType) {
  case GOFF::ESD_ST_SectionDefinition: {
    if (Symbol.isExecutable()) // Unspecified otherwise
      setTaskingBehavior(GOFF::ESD_TA_Rent);
    if (Symbol.BindingScope == GOFF::ESD_BSC_Section)
      setBindingScope(Symbol.BindingScope);
  } break;
  case GOFF::ESD_ST_ElementDefinition: {
    SymbolFlags.set(3, 1, Symbol.isRemovable()); // Removable
    if (Symbol.isExecutable()) {
      setExecutable(GOFF::ESD_EXE_CODE);
      setReadOnly(true);
    } else {
      if (Symbol.isExecUnspecified())
        setExecutable(GOFF::ESD_EXE_Unspecified);
      else
        setExecutable(GOFF::ESD_EXE_DATA);

      if (Symbol.isForceRent() || Symbol.isReadOnly())
        setReadOnly(true);
    }
    Offset = 0; // TODO ED and SD are 1-1 for now
    setAlignment(Symbol.Alignment);
    SymbolFlags.set(0, 1, 1); // Fill-Byte Value Presence Flag
    FillByteValue = 0;
    SymbolFlags.set(1, 1, 0); // Mangled Flag TODO ?
    setAmode(Symbol.Amode);
    setRmode(Symbol.Rmode);
    setTextStyle(Symbol.TextStyle);
    setBindingAlgorithm(Symbol.BindAlgorithm);
    setLoadingBehavior(Symbol.LoadBehavior);
    SymbolFlags.set(5, 3, GOFF::ESD_RQ_0); // Reserved Qwords
    NameSpaceId = Symbol.NameSpace;
    Length = Symbol.SectionLength;
    break;
  }
  case GOFF::ESD_ST_LabelDefinition: {
    if (Symbol.isExecutable())
      setExecutable(GOFF::ESD_EXE_CODE);
    else
      setExecutable(GOFF::ESD_EXE_DATA);
    setBindingStrength(Symbol.BindingStrength);
    setLinkageType(Symbol.Linkage);
    SymbolFlags.set(2, 1, Symbol.Renamable); // Renamable;
    setAmode(Symbol.Amode);
    NameSpaceId = Symbol.NameSpace;
    setBindingScope(Symbol.BindingScope);
    AdaEsdId = Symbol.ADAEsdId;

    // Only symbol that doesn't have an MC is the SectionLabelSymbol which
    // implicitly has 0 offset into the parent SD!
    if (auto *MCSym = Symbol.MCSym) {
      uint64_t Ofs = Layout.getSymbolOffset(*MCSym);
      // We only have signed 32bits of offset!
      assert(Ofs < (((uint64_t)1) << 31) && "ESD offset out of range.");
      Offset = static_cast<uint32_t>(Ofs);
    }
    break;
  }
  case GOFF::ESD_ST_ExternalReference: {
    setExecutable(Symbol.isExecutable() ? GOFF::ESD_EXE_CODE
                                        : GOFF::ESD_EXE_DATA);
    setBindingStrength(Symbol.BindingStrength);
    setLinkageType(Symbol.Linkage);
    SymbolFlags.set(2, 1, Symbol.Renamable); // Renamable;
    setIndirectReference(Symbol.Indirect);
    Offset = 0; // ERs don't do offsets
    NameSpaceId = Symbol.NameSpace;
    setBindingScope(Symbol.BindingScope);
    setAmode(Symbol.Amode);
    break;
  }
  case GOFF::ESD_ST_PartReference: {
    setExecutable(Symbol.isExecutable() ? GOFF::ESD_EXE_CODE
                                        : GOFF::ESD_EXE_DATA);
    NameSpaceId = Symbol.NameSpace;
    setAlignment(Symbol.Alignment);
    setAmode(Symbol.Amode);
    setLinkageType(Symbol.Linkage);
    setBindingScope(Symbol.BindingScope);
    SymbolFlags.set(2, 1, Symbol.Renamable); // Renamable;
    setDuplicateSeverity(Symbol.isWeakRef() ? GOFF::ESD_DSS_NoWarning
                                            : GOFF::ESD_DSS_Warning);
    setIndirectReference(Symbol.Indirect);
    setReadOnly(Symbol.ReadOnly);
    SortPriority = Symbol.SortKey;

    Length = Symbol.SectionLength;
    break;
  }
  } // End switch

  SmallString<256> Res;
  ConverterEBCDIC::convertToEBCDIC(Symbol.Name, Res);
  StringRef Name = Res.str();

  // Assert here since this number is technically signed but we need uint for
  // writing to records.
  assert(Name.size() < GOFF::MaxDataLength &&
         "Symbol max name length exceeded");
  uint16_t NameLength = Name.size();

  OS.newRecord(GOFF::RT_ESD, GOFF::ESDMetadataLength + NameLength);
  OS.writebe<uint8_t>(Symbol.SymbolType);       // Symbol Type
  OS.writebe<uint32_t>(Symbol.EsdId);           // ESDID
  OS.writebe<uint32_t>(Symbol.ParentEsdId);     // Parent or Owning ESDID
  OS.writebe<uint32_t>(0);                      // Reserved
  OS.writebe<uint32_t>(Offset);                 // Offset or Address
  OS.writebe<uint32_t>(0);                      // Reserved
  OS.writebe<uint32_t>(Length);                 // Length
  OS.writebe<uint32_t>(Symbol.EASectionEsdId);  // Extended Attribute ESDID
  OS.writebe<uint32_t>(Symbol.EASectionOffset); // Extended Attribute Offset
  OS.writebe<uint32_t>(0);                      // Reserved
  OS.writebe<uint8_t>(NameSpaceId);             // Name Space ID
  OS.writebe<uint8_t>(SymbolFlags);             // Flags
  OS.writebe<uint8_t>(FillByteValue);           // Fill-Byte Value
  OS.writebe<uint8_t>(0);                       // Reserved
  OS.writebe<uint32_t>(AdaEsdId);               // ADA ESDID
  OS.writebe<uint32_t>(SortPriority);           // Sort Priority
  OS.writebe<uint64_t>(0);                      // Reserved
  for (auto F : BehavAttrs)
    OS.writebe<uint8_t>(F);          // Behavioral Attributes
  OS.writebe<uint16_t>(NameLength);  // Name Length
  OS.write(Name.data(), NameLength); // Name
}

void GOFFObjectWriter::writeADAandCodeSectionSymbols(
    MCAssembler &Asm, const MCAsmLayout &Layout) {
  // Write ESD Records for ADA Section
  GOFFSymbol ADAED = createWSASymbol(RootSD.EsdId, true);
  StringRef FileName = "";
  if (!Asm.getFileNames().empty())
    FileName = sys::path::stem((*(Asm.getFileNames().begin())).first);

  GOFFSymbol ADA;
  ADA = createPRSymbol(FileName.str().append("#S"), ADAED.EsdId,
                       GOFF::ESD_NS_Parts, GOFF::ESD_EXE_DATA,
                       GOFF::ESD_ALIGN_Quadword, GOFF::ESD_BSC_Section,
                       getADASectionLength(Asm, Layout));
  writeSymbol(ADAED, Layout);
  writeSymbol(ADA, Layout);
  ADAPREsdId = ADA.EsdId;

  // Write ESD Records for Code Section
  GOFFSymbol ED =
      createEDSymbol("C_CODE64", RootSD.EsdId, 0, GOFF::ESD_EXE_CODE, true);

  for (const MCSection &MCSec : Asm) {
    auto &GSec = cast<MCSectionGOFF>(MCSec);
    if (GSec.isCode()) {
      if (!ED.SectionLength)
        ED.SectionLength = Layout.getSectionAddressSize(&MCSec);
    }
  }

  GOFFSymbol LD = createLDSymbol(RootSD.Name, ED.EsdId);
  LD.Executable = GOFF::ESD_EXE_CODE;
  if (RootSD.BindingScope == GOFF::ESD_BSC_Section)
    LD.BindingScope = GOFF::ESD_BSC_Section;
  else
    LD.BindingScope = GOFF::ESD_BSC_Library;

  LD.ADAEsdId = ADAPREsdId;

  EntryEDEsdId = ED.EsdId;
  CodeLDEsdId = LD.EsdId;
  writeSymbol(ED, Layout);
  writeSymbol(LD, Layout);
}

void GOFFObjectWriter::writeSectionSymbols(MCAssembler &Asm,
                                           const MCAsmLayout &Layout) {
  for (MCSection &S : Asm) {
    auto &Section = cast<MCSectionGOFF>(S);
    SectionKind Kind = Section.getKind();

    if (Section.isCode()) {
      // The only text section is the code section and the relevant ESD Records
      // are already created. All we need to do is populate the SectionMap.
      uint32_t RootSDEsdId = RootSD.EsdId;

      GOFFSection GoffSec = GOFFSection(EntryEDEsdId, CodeLDEsdId, RootSDEsdId);
      SectionMap.insert(std::make_pair(&Section, GoffSec));
    } else if (Section.getName().starts_with(".lsda")) {
      GOFFSymbol SD = RootSD;

      const char *WSAClassName = "C_WSA64";
      GOFFSymbol ED = createEDSymbol(
          WSAClassName, SD.EsdId, 0, GOFF::ESD_EXE_DATA,
          GOFF::ESD_BSC_Unspecified, GOFF::ESD_NS_Parts, GOFF::ESD_BA_Merge,
          /*ReadOnly*/ false, GOFF::ESD_TS_Unstructured, GOFF::ESD_LB_Initial);
      GOFFSymbol PR = createPRSymbol(
          Section.getName(), ED.EsdId, GOFF::ESD_NS_Parts,
          GOFF::ESD_EXE_Unspecified, GOFF::ESD_ALIGN_Fullword,
          GOFF::ESD_BSC_Section, Layout.getSectionAddressSize(&Section),
          GOFF::ESD_LB_Deferred);

      writeSymbol(ED, Layout);
      writeSymbol(PR, Layout);

      GOFFSection GoffSec = GOFFSection(PR.EsdId, PR.EsdId, SD.EsdId);
      SectionMap.insert(std::make_pair(&Section, GoffSec));
    } else if (Section.isPPA2Offset()) {
      StringRef EDSectionName = "C_@@QPPA2";
      StringRef PRSectionName = ".&ppa2";
      GOFFSymbol SD = RootSD;
      GOFFSymbol ED = createEDSymbol(
          EDSectionName, SD.EsdId, 0, GOFF::ESD_EXE_DATA,
          GOFF::ESD_BSC_Unspecified, GOFF::ESD_NS_Parts, GOFF::ESD_BA_Merge,
          /*ReadOnly*/ true, GOFF::ESD_TS_ByteOriented, GOFF::ESD_LB_Initial);
      GOFFSymbol PR = createPRSymbol(
          PRSectionName, ED.EsdId, GOFF::ESD_NS_Parts,
          GOFF::ESD_EXE_Unspecified, GOFF::ESD_ALIGN_Doubleword,
          GOFF::ESD_BSC_Section, Layout.getSectionAddressSize(&Section),
          GOFF::ESD_LB_Initial, GOFF::ESD_LT_OS);
      writeSymbol(ED, Layout);
      writeSymbol(PR, Layout);

      GOFFSection GoffSec = GOFFSection(PR.EsdId, PR.EsdId, SD.EsdId);
      SectionMap.insert(std::make_pair(&Section, GoffSec));
    } else if (Section.isB_IDRL()) {
      GOFFSymbol SD = RootSD;
      GOFFSymbol ED = createEDSymbol(
          "B_IDRL", SD.EsdId, Layout.getSectionAddressSize(&Section),
          GOFF::ESD_EXE_Unspecified, GOFF::ESD_BSC_Module,
          GOFF::ESD_NS_NormalName, GOFF::ESD_BA_Concatenate,
          /*ReadOnly*/ true, GOFF::ESD_TS_Structured, GOFF::ESD_LB_NoLoad);
      writeSymbol(ED, Layout);

      GOFFSection GoffSec = GOFFSection(ED.EsdId, 0, 0);
      SectionMap.insert(std::make_pair(&Section, GoffSec));
    } else if (Section.isStatic()) {
      GOFFSection GoffSec = GOFFSection(ADAPREsdId, ADAPREsdId, RootSD.EsdId);
      SectionMap.insert(std::make_pair(&Section, GoffSec));
    } else if (Kind.isBSS() || Kind.isData() || Kind.isThreadData()) {
      // We handle this with the symbol definition, so there is no need to do
      // anything here.
    } else {
      llvm_unreachable("Unhandled section kind");
    }
  }
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
  const GOFF::TXTRecordStyle RecordStyle;

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  uint64_t current_pos() const override { return Offset; }

public:
  explicit TextStream(GOFFOstream &OS, uint32_t EsdId,
                      GOFF::TXTRecordStyle RecordStyle)
      : OS(OS), Offset(0), EsdId(EsdId), RecordStyle(RecordStyle) {
    SetBuffer(Buffer, sizeof(Buffer));
  }

  ~TextStream() { flush(); }
};

void TextStream::write_impl(const char *Ptr, size_t Size) {
  size_t WrittenLength = 0;

  // We only have signed 32bits of offset.
  if (Offset + Size > std::numeric_limits<int32_t>::max())
    report_fatal_error("TXT section too large");

  while (WrittenLength < Size) {
    size_t ToWriteLength =
        std::min(Size - WrittenLength, size_t(GOFF::MaxDataLength));

    OS.newRecord(GOFF::RT_TXT, GOFF::TXTMetadataLength + ToWriteLength);
    OS.writebe<uint8_t>(Flags(4, 4, RecordStyle));       // Text Record Style
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
} // namespace

void GOFFObjectWriter::writeText(const MCSectionGOFF *MCSec, uint32_t EsdId,
                                 GOFF::TXTRecordStyle RecordStyle,
                                 MCAssembler &Asm, const MCAsmLayout &Layout) {
  TextStream S(OS, EsdId, RecordStyle);
  Asm.writeSectionData(S, MCSec, Layout);
}

void GOFFObjectWriter::writeEnd() {
  uint8_t F = GOFF::END_EPR_None;
  uint8_t AMODE = 0;
  uint32_t ESDID = 0;

  // TODO Set Flags/AMODE/ESDID for entry point.

  OS.newRecord(GOFF::RT_END, /*Size=*/13);
  OS.writebe<uint8_t>(Flags(6, 2, F)); // Indicator flags
  OS.writebe<uint8_t>(AMODE);          // AMODE
  OS.write_zeros(3);                   // Reserved
  // The record count is the number of logical records. In principle, this value
  // is available as OS.logicalRecords(). However, some tools rely on this field
  // being zero.
  OS.writebe<uint32_t>(0);     // Record Count
  OS.writebe<uint32_t>(ESDID); // ESDID (of entry point)
  OS.finalize();
}

uint64_t GOFFObjectWriter::writeObject(MCAssembler &Asm,
                                       const MCAsmLayout &Layout) {
  uint64_t StartOffset = OS.tell();

  writeHeader();
  writeSymbol(RootSD, Layout);

  writeADAandCodeSectionSymbols(Asm, Layout);
  writeSectionSymbols(Asm, Layout);

  // Process all MCSymbols and generate the ESD Record(s) for them.
  // Symbols that are aliases of other symbols need to be processed
  // at the end, after the symbols they alias are processed.
  for (const MCSymbol &MCSym : Asm.symbols()) {
    if (!MCSym.isTemporary()) {
      auto &Symbol = cast<MCSymbolGOFF>(MCSym);
      if (Symbol.isDefined())
        writeSymbolDefinedInModule(Symbol, Asm, Layout);
      else
        writeSymbolDeclaredInModule(Symbol, Asm, Layout);
    }
  }

  for (auto GSecIter = SectionMap.begin(); GSecIter != SectionMap.end();
       GSecIter++) {
    auto &MCGOFFSec = cast<MCSectionGOFF>(*(GSecIter->first));
    GOFFSection &CurrGSec = GSecIter->second;
    auto TextStyle =
        static_cast<GOFF::TXTRecordStyle>(MCGOFFSec.getTextStyle());
    writeText(&MCGOFFSec, CurrGSec.PEsdId, TextStyle, Asm, Layout);
  }

  writeEnd();

  LLVM_DEBUG(dbgs() << "Wrote " << OS.logicalRecords() << " logical records.");

  return OS.tell() - StartOffset;
}

std::unique_ptr<MCObjectWriter>
llvm::createGOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                             raw_pwrite_stream &OS) {
  return std::make_unique<GOFFObjectWriter>(std::move(MOTW), OS);
}
