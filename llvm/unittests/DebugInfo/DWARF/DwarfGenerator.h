//===--- unittests/DebugInfo/DWARF/DwarfGenerator.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A file that can generate DWARF debug info for unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFGENERATOR_H
#define LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFGENERATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace llvm {

class AsmPrinter;
class DIE;
class DIEAbbrev;
class DwarfStringPool;
class MCAsmBackend;
class MCAsmInfo;
class MCCodeEmitter;
class MCContext;
struct MCDwarfLineTableParams;
class MCInstrInfo;
class MCRegisterInfo;
class MCStreamer;
class MCSubtargetInfo;
class raw_fd_ostream;
class TargetLoweringObjectFile;
class TargetMachine;
class Triple;

namespace dwarfgen {

class Generator;
class CompileUnit;

/// A DWARF debug information entry class used to generate DWARF DIEs.
///
/// This class is used to quickly generate DWARF debug information by creating
/// child DIEs or adding attributes to the current DIE. Instances of this class
/// are created from the compile unit (dwarfgen::CompileUnit::getUnitDIE()) or
/// by calling dwarfgen::DIE::addChild(...) and using the returned DIE object.
class DIE {
  dwarfgen::CompileUnit *CU;
  llvm::DIE *Die;

protected:
  friend class Generator;
  friend class CompileUnit;

  DIE(CompileUnit *U = nullptr, llvm::DIE *D = nullptr) : CU(U), Die(D) {}

  /// Called with a compile/type unit relative offset prior to generating the
  /// DWARF debug info.
  ///
  /// \param CUOffset the compile/type unit relative offset where the
  /// abbreviation code for this DIE will be encoded.
  unsigned computeSizeAndOffsets(unsigned CUOffset);

public:
  /// Add an attribute value that has no value.
  ///
  /// \param Attr a dwarf::Attribute enumeration value or any uint16_t that
  /// represents a user defined DWARF attribute.
  /// \param Form the dwarf::Form to use when encoding the attribute. This is
  /// only used with the DW_FORM_flag_present form encoding.
  void addAttribute(uint16_t Attr, dwarf::Form Form);

  /// Add an attribute value to be encoded as a DIEInteger
  ///
  /// \param Attr a dwarf::Attribute enumeration value or any uint16_t that
  /// represents a user defined DWARF attribute.
  /// \param Form the dwarf::Form to use when encoding the attribute.
  /// \param U the unsigned integer to encode.
  void addAttribute(uint16_t Attr, dwarf::Form Form, uint64_t U);

  /// Add an attribute value to be encoded as a DIEExpr
  ///
  /// \param Attr a dwarf::Attribute enumeration value or any uint16_t that
  /// represents a user defined DWARF attribute.
  /// \param Form the dwarf::Form to use when encoding the attribute.
  /// \param Expr the MC expression used to compute the value.
  void addAttribute(uint16_t Attr, dwarf::Form Form, const MCExpr &Expr);

  /// Add an attribute value to be encoded as a DIEString or DIEInlinedString.
  ///
  /// \param Attr a dwarf::Attribute enumeration value or any uint16_t that
  /// represents a user defined DWARF attribute.
  /// \param Form the dwarf::Form to use when encoding the attribute. The form
  /// must be one of DW_FORM_strp or DW_FORM_string.
  /// \param String the string to encode.
  void addAttribute(uint16_t Attr, dwarf::Form Form, StringRef String);

  /// Add an attribute value to be encoded as a DIEEntry.
  ///
  /// DIEEntry attributes refer to other llvm::DIE objects that have been
  /// created.
  ///
  /// \param Attr a dwarf::Attribute enumeration value or any uint16_t that
  /// represents a user defined DWARF attribute.
  /// \param Form the dwarf::Form to use when encoding the attribute. The form
  /// must be one of DW_FORM_strp or DW_FORM_string.
  /// \param RefDie the DIE that this attriute refers to.
  void addAttribute(uint16_t Attr, dwarf::Form Form, dwarfgen::DIE &RefDie);

  /// Add an attribute value to be encoded as a DIEBlock.
  ///
  /// DIEBlock attributes refers to binary data that is stored as the
  /// attribute's value.
  ///
  /// \param Attr a dwarf::Attribute enumeration value or any uint16_t that
  /// represents a user defined DWARF attribute.
  /// \param Form the dwarf::Form to use when encoding the attribute. The form
  /// must be one of DW_FORM_strp or DW_FORM_string.
  /// \param P a pointer to the data to store as the attribute value.
  /// \param S the size in bytes of the data pointed to by P .
  void addAttribute(uint16_t Attr, dwarf::Form Form, const void *P, size_t S);

  /// Add a DW_AT_str_offsets_base attribute to this DIE.
  void addStrOffsetsBaseAttribute();

  /// Add a new child to this DIE object.
  ///
  /// \param Tag the dwarf::Tag to assing to the llvm::DIE object.
  /// \returns the newly created DIE object that is now a child owned by this
  /// object.
  dwarfgen::DIE addChild(dwarf::Tag Tag);
};

/// A DWARF compile unit used to generate DWARF compile/type units.
///
/// Instances of these classes are created by instances of the Generator
/// class. All information required to generate a DWARF compile unit is
/// contained inside this class.
class CompileUnit {
  Generator &DG;
  BasicDIEUnit DU;
  uint64_t Length; /// The length in bytes of all of the DIEs in this unit.
  const uint16_t Version; /// The Dwarf version number for this unit.
  const uint8_t AddrSize; /// The size in bytes of an address for this unit.

public:
  CompileUnit(Generator &D, uint16_t V, uint8_t A)
      : DG(D), DU(dwarf::DW_TAG_compile_unit), Version(V), AddrSize(A) {}
  DIE getUnitDIE();
  Generator &getGenerator() { return DG; }
  uint64_t getOffset() const { return DU.getDebugSectionOffset(); }
  uint64_t getLength() const { return Length; }
  uint16_t getVersion() const { return Version; }
  uint16_t getAddressSize() const { return AddrSize; }
  void setOffset(uint64_t Offset) { DU.setDebugSectionOffset(Offset); }
  void setLength(uint64_t L) { Length = L; }
};

/// A DWARF line unit-like class used to generate DWARF line units.
///
/// Instances of this class are created by instances of the Generator class.
class LineTable {
public:
  enum ValueLength { Byte = 1, Half = 2, Long = 4, Quad = 8, ULEB, SLEB };

  struct ValueAndLength {
    uint64_t Value = 0;
    ValueLength Length = Byte;
  };

  LineTable(uint16_t Version, dwarf::DwarfFormat Format, uint8_t AddrSize,
            uint8_t SegSize = 0)
      : Version(Version), Format(Format), AddrSize(AddrSize), SegSize(SegSize) {
    assert(Version >= 2 && Version <= 5 && "unsupported version");
  }

  // Create a Prologue suitable to pass to setPrologue, with a single file and
  // include directory entry.
  DWARFDebugLine::Prologue createBasicPrologue() const;

  // Set or replace the current prologue with the specified prologue. If no
  // prologue is set, a default one will be used when generating.
  void setPrologue(DWARFDebugLine::Prologue NewPrologue);
  // Used to write an arbitrary payload instead of the standard prologue. This
  // is useful if you wish to test handling of corrupt .debug_line sections.
  void setCustomPrologue(ArrayRef<ValueAndLength> NewPrologue);

  // Add a byte to the program, with the given value. This can be used to
  // specify a special opcode, or to add arbitrary contents to the section.
  void addByte(uint8_t Value);
  // Add a standard opcode to the program. The opcode and operands do not have
  // to be valid.
  void addStandardOpcode(uint8_t Opcode, ArrayRef<ValueAndLength> Operands);
  // Add an extended opcode to the program with the specified length, opcode,
  // and operands. These values do not have to be valid.
  void addExtendedOpcode(uint64_t Length, uint8_t Opcode,
                         ArrayRef<ValueAndLength> Operands);

  // Write the contents of the LineUnit to the current section in the generator.
  void generate(MCContext &MC, AsmPrinter &Asm) const;

private:
  void writeData(ArrayRef<ValueAndLength> Data, AsmPrinter &Asm) const;
  MCSymbol *writeDefaultPrologue(AsmPrinter &Asm) const;
  void writePrologue(AsmPrinter &Asm) const;

  void writeProloguePayload(const DWARFDebugLine::Prologue &Prologue,
                            AsmPrinter &Asm) const;

  // Calculate the number of bytes the Contents will take up.
  size_t getContentsSize() const;

  llvm::Optional<DWARFDebugLine::Prologue> Prologue;
  std::vector<ValueAndLength> CustomPrologue;
  std::vector<ValueAndLength> Contents;

  // The Version field is used for determining how to write the Prologue, if a
  // non-custom prologue is used. The version value actually written, will be
  // that specified in the Prologue, if a custom prologue has been passed in.
  // Otherwise, it will be this value.
  uint16_t Version;

  dwarf::DwarfFormat Format;
  uint8_t AddrSize;
  uint8_t SegSize;
};

/// A DWARF generator.
///
/// Generate DWARF for unit tests by creating any instance of this class and
/// calling Generator::addCompileUnit(), and then getting the dwarfgen::DIE from
/// the returned compile unit and adding attributes and children to each DIE.
class Generator {
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCContext> MC;
  MCAsmBackend *MAB; // Owned by MCStreamer
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCSubtargetInfo> MSTI;
  MCCodeEmitter *MCE; // Owned by MCStreamer
  MCStreamer *MS;     // Owned by AsmPrinter
  std::unique_ptr<TargetMachine> TM;
  TargetLoweringObjectFile *TLOF; // Owned by TargetMachine;
  std::unique_ptr<AsmPrinter> Asm;
  BumpPtrAllocator Allocator;
  std::unique_ptr<DwarfStringPool> StringPool; // Entries owned by Allocator.
  std::vector<std::unique_ptr<CompileUnit>> CompileUnits;
  std::vector<std::unique_ptr<LineTable>> LineTables;
  DIEAbbrevSet Abbreviations;

  MCSymbol *StringOffsetsStartSym;

  SmallString<4096> FileBytes;
  /// The stream we use to generate the DWARF into as an ELF file.
  std::unique_ptr<raw_svector_ostream> Stream;
  /// The DWARF version to generate.
  uint16_t Version;

  /// Private constructor, call Generator::Create(...) to get a DWARF generator
  /// expected.
  Generator();

  /// Create the streamer and setup the output buffer.
  llvm::Error init(Triple TheTriple, uint16_t DwarfVersion);

public:
  /// Create a DWARF generator or get an appropriate error.
  ///
  /// \param TheTriple the triple to use when creating any required support
  /// classes needed to emit the DWARF.
  /// \param DwarfVersion the version of DWARF to emit.
  ///
  /// \returns a llvm::Expected that either contains a unique_ptr to a Generator
  /// or a llvm::Error.
  static llvm::Expected<std::unique_ptr<Generator>>
  create(Triple TheTriple, uint16_t DwarfVersion);

  ~Generator();

  /// Generate all DWARF sections and return a memory buffer that
  /// contains an ELF file that contains the DWARF.
  StringRef generate();

  /// Add a compile unit to be generated.
  ///
  /// \returns a dwarfgen::CompileUnit that can be used to retrieve the compile
  /// unit dwarfgen::DIE that can be used to add attributes and add child DIE
  /// objects to.
  dwarfgen::CompileUnit &addCompileUnit();

  /// Add a line table unit to be generated.
  /// \param DwarfFormat the DWARF format to use (DWARF32 or DWARF64).
  ///
  /// \returns a dwarfgen::LineTable that can be used to customise the contents
  /// of the line table.
  LineTable &
  addLineTable(dwarf::DwarfFormat DwarfFormat = dwarf::DwarfFormat::DWARF32);

  BumpPtrAllocator &getAllocator() { return Allocator; }
  AsmPrinter *getAsmPrinter() const { return Asm.get(); }
  MCContext *getMCContext() const { return MC.get(); }
  DIEAbbrevSet &getAbbrevSet() { return Abbreviations; }
  DwarfStringPool &getStringPool() { return *StringPool; }
  MCSymbol *getStringOffsetsStartSym() const { return StringOffsetsStartSym; }

  /// Save the generated DWARF file to disk.
  ///
  /// \param Path the path to save the ELF file to.
  bool saveFile(StringRef Path);
};

} // end namespace dwarfgen

} // end namespace llvm

#endif // LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFGENERATOR_H
