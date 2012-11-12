//===- lib/MC/MCDwarf.cpp - MCDwarf implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
using namespace llvm;

// Given a special op, return the address skip amount (in units of
// DWARF2_LINE_MIN_INSN_LENGTH.
#define SPECIAL_ADDR(op) (((op) - DWARF2_LINE_OPCODE_BASE)/DWARF2_LINE_RANGE)

// The maximum address skip amount that can be encoded with a special op.
#define MAX_SPECIAL_ADDR_DELTA         SPECIAL_ADDR(255)

// First special line opcode - leave room for the standard opcodes.
// Note: If you want to change this, you'll have to update the
// "standard_opcode_lengths" table that is emitted in DwarfFileTable::Emit().
#define DWARF2_LINE_OPCODE_BASE         13

// Minimum line offset in a special line info. opcode.  This value
// was chosen to give a reasonable range of values.
#define DWARF2_LINE_BASE                -5

// Range of line offsets in a special line info. opcode.
#define DWARF2_LINE_RANGE               14

// Define the architecture-dependent minimum instruction length (in bytes).
// This value should be rather too small than too big.
#define DWARF2_LINE_MIN_INSN_LENGTH     1

// Note: when DWARF2_LINE_MIN_INSN_LENGTH == 1 which is the current setting,
// this routine is a nop and will be optimized away.
static inline uint64_t ScaleAddrDelta(uint64_t AddrDelta) {
  if (DWARF2_LINE_MIN_INSN_LENGTH == 1)
    return AddrDelta;
  if (AddrDelta % DWARF2_LINE_MIN_INSN_LENGTH != 0) {
    // TODO: report this error, but really only once.
    ;
  }
  return AddrDelta / DWARF2_LINE_MIN_INSN_LENGTH;
}

//
// This is called when an instruction is assembled into the specified section
// and if there is information from the last .loc directive that has yet to have
// a line entry made for it is made.
//
void MCLineEntry::Make(MCStreamer *MCOS, const MCSection *Section) {
  if (!MCOS->getContext().getDwarfLocSeen())
    return;

  // Create a symbol at in the current section for use in the line entry.
  MCSymbol *LineSym = MCOS->getContext().CreateTempSymbol();
  // Set the value of the symbol to use for the MCLineEntry.
  MCOS->EmitLabel(LineSym);

  // Get the current .loc info saved in the context.
  const MCDwarfLoc &DwarfLoc = MCOS->getContext().getCurrentDwarfLoc();

  // Create a (local) line entry with the symbol and the current .loc info.
  MCLineEntry LineEntry(LineSym, DwarfLoc);

  // clear DwarfLocSeen saying the current .loc info is now used.
  MCOS->getContext().ClearDwarfLocSeen();

  // Get the MCLineSection for this section, if one does not exist for this
  // section create it.
  const DenseMap<const MCSection *, MCLineSection *> &MCLineSections =
    MCOS->getContext().getMCLineSections();
  MCLineSection *LineSection = MCLineSections.lookup(Section);
  if (!LineSection) {
    // Create a new MCLineSection.  This will be deleted after the dwarf line
    // table is created using it by iterating through the MCLineSections
    // DenseMap.
    LineSection = new MCLineSection;
    // Save a pointer to the new LineSection into the MCLineSections DenseMap.
    MCOS->getContext().addMCLineSection(Section, LineSection);
  }

  // Add the line entry to this section's entries.
  LineSection->addLineEntry(LineEntry);
}

//
// This helper routine returns an expression of End - Start + IntVal .
//
static inline const MCExpr *MakeStartMinusEndExpr(const MCStreamer &MCOS,
                                                  const MCSymbol &Start,
                                                  const MCSymbol &End,
                                                  int IntVal) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *Res =
    MCSymbolRefExpr::Create(&End, Variant, MCOS.getContext());
  const MCExpr *RHS =
    MCSymbolRefExpr::Create(&Start, Variant, MCOS.getContext());
  const MCExpr *Res1 =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, Res, RHS, MCOS.getContext());
  const MCExpr *Res2 =
    MCConstantExpr::Create(IntVal, MCOS.getContext());
  const MCExpr *Res3 =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, Res1, Res2, MCOS.getContext());
  return Res3;
}

//
// This emits the Dwarf line table for the specified section from the entries
// in the LineSection.
//
static inline void EmitDwarfLineTable(MCStreamer *MCOS,
                                      const MCSection *Section,
                                      const MCLineSection *LineSection) {
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  MCSymbol *LastLabel = NULL;

  // Loop through each MCLineEntry and encode the dwarf line number table.
  for (MCLineSection::const_iterator
         it = LineSection->getMCLineEntries()->begin(),
         ie = LineSection->getMCLineEntries()->end(); it != ie; ++it) {

    if (FileNum != it->getFileNum()) {
      FileNum = it->getFileNum();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_file, 1);
      MCOS->EmitULEB128IntValue(FileNum);
    }
    if (Column != it->getColumn()) {
      Column = it->getColumn();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_column, 1);
      MCOS->EmitULEB128IntValue(Column);
    }
    if (Isa != it->getIsa()) {
      Isa = it->getIsa();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_isa, 1);
      MCOS->EmitULEB128IntValue(Isa);
    }
    if ((it->getFlags() ^ Flags) & DWARF2_FLAG_IS_STMT) {
      Flags = it->getFlags();
      MCOS->EmitIntValue(dwarf::DW_LNS_negate_stmt, 1);
    }
    if (it->getFlags() & DWARF2_FLAG_BASIC_BLOCK)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_basic_block, 1);
    if (it->getFlags() & DWARF2_FLAG_PROLOGUE_END)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_prologue_end, 1);
    if (it->getFlags() & DWARF2_FLAG_EPILOGUE_BEGIN)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_epilogue_begin, 1);

    int64_t LineDelta = static_cast<int64_t>(it->getLine()) - LastLine;
    MCSymbol *Label = it->getLabel();

    // At this point we want to emit/create the sequence to encode the delta in
    // line numbers and the increment of the address from the previous Label
    // and the current Label.
    const MCAsmInfo &asmInfo = MCOS->getContext().getAsmInfo();
    MCOS->EmitDwarfAdvanceLineAddr(LineDelta, LastLabel, Label,
                                   asmInfo.getPointerSize());

    LastLine = it->getLine();
    LastLabel = Label;
  }

  // Emit a DW_LNE_end_sequence for the end of the section.
  // Using the pointer Section create a temporary label at the end of the
  // section and use that and the LastLabel to compute the address delta
  // and use INT64_MAX as the line delta which is the signal that this is
  // actually a DW_LNE_end_sequence.

  // Switch to the section to be able to create a symbol at its end.
  MCOS->SwitchSection(Section);

  MCContext &context = MCOS->getContext();
  // Create a symbol at the end of the section.
  MCSymbol *SectionEnd = context.CreateTempSymbol();
  // Set the value of the symbol, as we are at the end of the section.
  MCOS->EmitLabel(SectionEnd);

  // Switch back the dwarf line section.
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfLineSection());

  const MCAsmInfo &asmInfo = MCOS->getContext().getAsmInfo();
  MCOS->EmitDwarfAdvanceLineAddr(INT64_MAX, LastLabel, SectionEnd,
                                 asmInfo.getPointerSize());
}

//
// This emits the Dwarf file and the line tables.
//
const MCSymbol *MCDwarfFileTable::Emit(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  // Switch to the section where the table will be emitted into.
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfLineSection());

  // Create a symbol at the beginning of this section.
  MCSymbol *LineStartSym = context.CreateTempSymbol();
  // Set the value of the symbol, as we are at the start of the section.
  MCOS->EmitLabel(LineStartSym);

  // Create a symbol for the end of the section (to be set when we get there).
  MCSymbol *LineEndSym = context.CreateTempSymbol();

  // The first 4 bytes is the total length of the information for this
  // compilation unit (not including these 4 bytes for the length).
  MCOS->EmitAbsValue(MakeStartMinusEndExpr(*MCOS, *LineStartSym, *LineEndSym,4),
                     4);

  // Next 2 bytes is the Version, which is Dwarf 2.
  MCOS->EmitIntValue(2, 2);

  // Create a symbol for the end of the prologue (to be set when we get there).
  MCSymbol *ProEndSym = context.CreateTempSymbol(); // Lprologue_end

  // Length of the prologue, is the next 4 bytes.  Which is the start of the
  // section to the end of the prologue.  Not including the 4 bytes for the
  // total length, the 2 bytes for the version, and these 4 bytes for the
  // length of the prologue.
  MCOS->EmitAbsValue(MakeStartMinusEndExpr(*MCOS, *LineStartSym, *ProEndSym,
                                        (4 + 2 + 4)),
                  4, 0);

  // Parameters of the state machine, are next.
  MCOS->EmitIntValue(DWARF2_LINE_MIN_INSN_LENGTH, 1);
  MCOS->EmitIntValue(DWARF2_LINE_DEFAULT_IS_STMT, 1);
  MCOS->EmitIntValue(DWARF2_LINE_BASE, 1);
  MCOS->EmitIntValue(DWARF2_LINE_RANGE, 1);
  MCOS->EmitIntValue(DWARF2_LINE_OPCODE_BASE, 1);

  // Standard opcode lengths
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_copy
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_advance_pc
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_advance_line
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_set_file
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_set_column
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_negate_stmt
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_set_basic_block
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_const_add_pc
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_fixed_advance_pc
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_set_prologue_end
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_set_epilogue_begin
  MCOS->EmitIntValue(1, 1); // DW_LNS_set_isa

  // Put out the directory and file tables.

  // First the directory table.
  const std::vector<StringRef> &MCDwarfDirs =
    context.getMCDwarfDirs();
  for (unsigned i = 0; i < MCDwarfDirs.size(); i++) {
    MCOS->EmitBytes(MCDwarfDirs[i], 0); // the DirectoryName
    MCOS->EmitBytes(StringRef("\0", 1), 0); // the null term. of the string
  }
  MCOS->EmitIntValue(0, 1); // Terminate the directory list

  // Second the file table.
  const std::vector<MCDwarfFile *> &MCDwarfFiles =
    MCOS->getContext().getMCDwarfFiles();
  for (unsigned i = 1; i < MCDwarfFiles.size(); i++) {
    MCOS->EmitBytes(MCDwarfFiles[i]->getName(), 0); // FileName
    MCOS->EmitBytes(StringRef("\0", 1), 0); // the null term. of the string
    // the Directory num
    MCOS->EmitULEB128IntValue(MCDwarfFiles[i]->getDirIndex());
    MCOS->EmitIntValue(0, 1); // last modification timestamp (always 0)
    MCOS->EmitIntValue(0, 1); // filesize (always 0)
  }
  MCOS->EmitIntValue(0, 1); // Terminate the file list

  // This is the end of the prologue, so set the value of the symbol at the
  // end of the prologue (that was used in a previous expression).
  MCOS->EmitLabel(ProEndSym);

  // Put out the line tables.
  const DenseMap<const MCSection *, MCLineSection *> &MCLineSections =
    MCOS->getContext().getMCLineSections();
  const std::vector<const MCSection *> &MCLineSectionOrder =
    MCOS->getContext().getMCLineSectionOrder();
  for (std::vector<const MCSection*>::const_iterator it =
         MCLineSectionOrder.begin(), ie = MCLineSectionOrder.end(); it != ie;
       ++it) {
    const MCSection *Sec = *it;
    const MCLineSection *Line = MCLineSections.lookup(Sec);
    EmitDwarfLineTable(MCOS, Sec, Line);

    // Now delete the MCLineSections that were created in MCLineEntry::Make()
    // and used to emit the line table.
    delete Line;
  }

  if (MCOS->getContext().getAsmInfo().getLinkerRequiresNonEmptyDwarfLines()
      && MCLineSectionOrder.begin() == MCLineSectionOrder.end()) {
    // The darwin9 linker has a bug (see PR8715). For for 32-bit architectures
    // it requires:
    // total_length >= prologue_length + 10
    // We are 4 bytes short, since we have total_length = 51 and
    // prologue_length = 45

    // The regular end_sequence should be sufficient.
    MCDwarfLineAddr::Emit(MCOS, INT64_MAX, 0);
  }

  // This is the end of the section, so set the value of the symbol at the end
  // of this section (that was used in a previous expression).
  MCOS->EmitLabel(LineEndSym);

  return LineStartSym;
}

/// Utility function to write the encoding to an object writer.
void MCDwarfLineAddr::Write(MCObjectWriter *OW, int64_t LineDelta,
                            uint64_t AddrDelta) {
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfLineAddr::Encode(LineDelta, AddrDelta, OS);
  OW->WriteBytes(OS.str());
}

/// Utility function to emit the encoding to a streamer.
void MCDwarfLineAddr::Emit(MCStreamer *MCOS, int64_t LineDelta,
                           uint64_t AddrDelta) {
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfLineAddr::Encode(LineDelta, AddrDelta, OS);
  MCOS->EmitBytes(OS.str(), /*AddrSpace=*/0);
}

/// Utility function to encode a Dwarf pair of LineDelta and AddrDeltas.
void MCDwarfLineAddr::Encode(int64_t LineDelta, uint64_t AddrDelta,
                             raw_ostream &OS) {
  uint64_t Temp, Opcode;
  bool NeedCopy = false;

  // Scale the address delta by the minimum instruction length.
  AddrDelta = ScaleAddrDelta(AddrDelta);

  // A LineDelta of INT64_MAX is a signal that this is actually a
  // DW_LNE_end_sequence. We cannot use special opcodes here, since we want the
  // end_sequence to emit the matrix entry.
  if (LineDelta == INT64_MAX) {
    if (AddrDelta == MAX_SPECIAL_ADDR_DELTA)
      OS << char(dwarf::DW_LNS_const_add_pc);
    else {
      OS << char(dwarf::DW_LNS_advance_pc);
      encodeULEB128(AddrDelta, OS);
    }
    OS << char(dwarf::DW_LNS_extended_op);
    OS << char(1);
    OS << char(dwarf::DW_LNE_end_sequence);
    return;
  }

  // Bias the line delta by the base.
  Temp = LineDelta - DWARF2_LINE_BASE;

  // If the line increment is out of range of a special opcode, we must encode
  // it with DW_LNS_advance_line.
  if (Temp >= DWARF2_LINE_RANGE) {
    OS << char(dwarf::DW_LNS_advance_line);
    encodeSLEB128(LineDelta, OS);

    LineDelta = 0;
    Temp = 0 - DWARF2_LINE_BASE;
    NeedCopy = true;
  }

  // Use DW_LNS_copy instead of a "line +0, addr +0" special opcode.
  if (LineDelta == 0 && AddrDelta == 0) {
    OS << char(dwarf::DW_LNS_copy);
    return;
  }

  // Bias the opcode by the special opcode base.
  Temp += DWARF2_LINE_OPCODE_BASE;

  // Avoid overflow when addr_delta is large.
  if (AddrDelta < 256 + MAX_SPECIAL_ADDR_DELTA) {
    // Try using a special opcode.
    Opcode = Temp + AddrDelta * DWARF2_LINE_RANGE;
    if (Opcode <= 255) {
      OS << char(Opcode);
      return;
    }

    // Try using DW_LNS_const_add_pc followed by special op.
    Opcode = Temp + (AddrDelta - MAX_SPECIAL_ADDR_DELTA) * DWARF2_LINE_RANGE;
    if (Opcode <= 255) {
      OS << char(dwarf::DW_LNS_const_add_pc);
      OS << char(Opcode);
      return;
    }
  }

  // Otherwise use DW_LNS_advance_pc.
  OS << char(dwarf::DW_LNS_advance_pc);
  encodeULEB128(AddrDelta, OS);

  if (NeedCopy)
    OS << char(dwarf::DW_LNS_copy);
  else
    OS << char(Temp);
}

void MCDwarfFile::print(raw_ostream &OS) const {
  OS << '"' << getName() << '"';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MCDwarfFile::dump() const {
  print(dbgs());
}
#endif

// Utility function to write a tuple for .debug_abbrev.
static void EmitAbbrev(MCStreamer *MCOS, uint64_t Name, uint64_t Form) {
  MCOS->EmitULEB128IntValue(Name);
  MCOS->EmitULEB128IntValue(Form);
}

// When generating dwarf for assembly source files this emits
// the data for .debug_abbrev section which contains three DIEs.
static void EmitGenDwarfAbbrev(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfAbbrevSection());

  // DW_TAG_compile_unit DIE abbrev (1).
  MCOS->EmitULEB128IntValue(1);
  MCOS->EmitULEB128IntValue(dwarf::DW_TAG_compile_unit);
  MCOS->EmitIntValue(dwarf::DW_CHILDREN_yes, 1);
  EmitAbbrev(MCOS, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4);
  EmitAbbrev(MCOS, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr);
  EmitAbbrev(MCOS, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr);
  EmitAbbrev(MCOS, dwarf::DW_AT_name, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_comp_dir, dwarf::DW_FORM_string);
  StringRef DwarfDebugFlags = context.getDwarfDebugFlags();
  if (!DwarfDebugFlags.empty())
    EmitAbbrev(MCOS, dwarf::DW_AT_APPLE_flags, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_producer, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_language, dwarf::DW_FORM_data2);
  EmitAbbrev(MCOS, 0, 0);

  // DW_TAG_label DIE abbrev (2).
  MCOS->EmitULEB128IntValue(2);
  MCOS->EmitULEB128IntValue(dwarf::DW_TAG_label);
  MCOS->EmitIntValue(dwarf::DW_CHILDREN_yes, 1);
  EmitAbbrev(MCOS, dwarf::DW_AT_name, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data4);
  EmitAbbrev(MCOS, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data4);
  EmitAbbrev(MCOS, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr);
  EmitAbbrev(MCOS, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag);
  EmitAbbrev(MCOS, 0, 0);

  // DW_TAG_unspecified_parameters DIE abbrev (3).
  MCOS->EmitULEB128IntValue(3);
  MCOS->EmitULEB128IntValue(dwarf::DW_TAG_unspecified_parameters);
  MCOS->EmitIntValue(dwarf::DW_CHILDREN_no, 1);
  EmitAbbrev(MCOS, 0, 0);

  // Terminate the abbreviations for this compilation unit.
  MCOS->EmitIntValue(0, 1);
}

// When generating dwarf for assembly source files this emits the data for
// .debug_aranges section.  Which contains a header and a table of pairs of
// PointerSize'ed values for the address and size of section(s) with line table
// entries (just the default .text in our case) and a terminating pair of zeros.
static void EmitGenDwarfAranges(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();

  // Create a symbol at the end of the section that we are creating the dwarf
  // debugging info to use later in here as part of the expression to calculate
  // the size of the section for the table.
  MCOS->SwitchSection(context.getGenDwarfSection());
  MCSymbol *SectionEndSym = context.CreateTempSymbol();
  MCOS->EmitLabel(SectionEndSym);
  context.setGenDwarfSectionEndSym(SectionEndSym);

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfARangesSection());

  // This will be the length of the .debug_aranges section, first account for
  // the size of each item in the header (see below where we emit these items).
  int Length = 4 + 2 + 4 + 1 + 1;

  // Figure the padding after the header before the table of address and size
  // pairs who's values are PointerSize'ed.
  const MCAsmInfo &asmInfo = context.getAsmInfo();
  int AddrSize = asmInfo.getPointerSize();
  int Pad = 2 * AddrSize - (Length & (2 * AddrSize - 1));
  if (Pad == 2 * AddrSize)
    Pad = 0;
  Length += Pad;

  // Add the size of the pair of PointerSize'ed values for the address and size
  // of the one default .text section we have in the table.
  Length += 2 * AddrSize;
  // And the pair of terminating zeros.
  Length += 2 * AddrSize;


  // Emit the header for this section.
  // The 4 byte length not including the 4 byte value for the length.
  MCOS->EmitIntValue(Length - 4, 4);
  // The 2 byte version, which is 2.
  MCOS->EmitIntValue(2, 2);
  // The 4 byte offset to the compile unit in the .debug_info from the start
  // of the .debug_info, it is at the start of that section so this is zero.
  MCOS->EmitIntValue(0, 4);
  // The 1 byte size of an address.
  MCOS->EmitIntValue(AddrSize, 1);
  // The 1 byte size of a segment descriptor, we use a value of zero.
  MCOS->EmitIntValue(0, 1);
  // Align the header with the padding if needed, before we put out the table.
  for(int i = 0; i < Pad; i++)
    MCOS->EmitIntValue(0, 1);

  // Now emit the table of pairs of PointerSize'ed values for the section(s)
  // address and size, in our case just the one default .text section.
  const MCExpr *Addr = MCSymbolRefExpr::Create(
    context.getGenDwarfSectionStartSym(), MCSymbolRefExpr::VK_None, context);
  const MCExpr *Size = MakeStartMinusEndExpr(*MCOS,
    *context.getGenDwarfSectionStartSym(), *SectionEndSym, 0);
  MCOS->EmitAbsValue(Addr, AddrSize);
  MCOS->EmitAbsValue(Size, AddrSize);

  // And finally the pair of terminating zeros.
  MCOS->EmitIntValue(0, AddrSize);
  MCOS->EmitIntValue(0, AddrSize);
}

// When generating dwarf for assembly source files this emits the data for
// .debug_info section which contains three parts.  The header, the compile_unit
// DIE and a list of label DIEs.
static void EmitGenDwarfInfo(MCStreamer *MCOS,
                             const MCSymbol *AbbrevSectionSymbol,
                             const MCSymbol *LineSectionSymbol) {
  MCContext &context = MCOS->getContext();

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfInfoSection());

  // Create a symbol at the start and end of this section used in here for the
  // expression to calculate the length in the header.
  MCSymbol *InfoStart = context.CreateTempSymbol();
  MCOS->EmitLabel(InfoStart);
  MCSymbol *InfoEnd = context.CreateTempSymbol();

  // First part: the header.

  // The 4 byte total length of the information for this compilation unit, not
  // including these 4 bytes.
  const MCExpr *Length = MakeStartMinusEndExpr(*MCOS, *InfoStart, *InfoEnd, 4);
  MCOS->EmitAbsValue(Length, 4);

  // The 2 byte DWARF version, which is 2.
  MCOS->EmitIntValue(2, 2);

  // The 4 byte offset to the debug abbrevs from the start of the .debug_abbrev,
  // it is at the start of that section so this is zero.
  if (AbbrevSectionSymbol) {
    MCOS->EmitSymbolValue(AbbrevSectionSymbol, 4);
  } else {
    MCOS->EmitIntValue(0, 4);
  }

  const MCAsmInfo &asmInfo = context.getAsmInfo();
  int AddrSize = asmInfo.getPointerSize();
  // The 1 byte size of an address.
  MCOS->EmitIntValue(AddrSize, 1);

  // Second part: the compile_unit DIE.

  // The DW_TAG_compile_unit DIE abbrev (1).
  MCOS->EmitULEB128IntValue(1);

  // DW_AT_stmt_list, a 4 byte offset from the start of the .debug_line section,
  // which is at the start of that section so this is zero.
  if (LineSectionSymbol) {
    MCOS->EmitSymbolValue(LineSectionSymbol, 4);
  } else {
    MCOS->EmitIntValue(0, 4);
  }

  // AT_low_pc, the first address of the default .text section.
  const MCExpr *Start = MCSymbolRefExpr::Create(
    context.getGenDwarfSectionStartSym(), MCSymbolRefExpr::VK_None, context);
  MCOS->EmitAbsValue(Start, AddrSize);

  // AT_high_pc, the last address of the default .text section.
  const MCExpr *End = MCSymbolRefExpr::Create(
    context.getGenDwarfSectionEndSym(), MCSymbolRefExpr::VK_None, context);
  MCOS->EmitAbsValue(End, AddrSize);

  // AT_name, the name of the source file.  Reconstruct from the first directory
  // and file table entries.
  const std::vector<StringRef> &MCDwarfDirs =
    context.getMCDwarfDirs();
  if (MCDwarfDirs.size() > 0) {
    MCOS->EmitBytes(MCDwarfDirs[0], 0);
    MCOS->EmitBytes("/", 0);
  }
  const std::vector<MCDwarfFile *> &MCDwarfFiles =
    MCOS->getContext().getMCDwarfFiles();
  MCOS->EmitBytes(MCDwarfFiles[1]->getName(), 0);
  MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

  // AT_comp_dir, the working directory the assembly was done in.
  llvm::sys::Path CWD = llvm::sys::Path::GetCurrentDirectory();
  MCOS->EmitBytes(StringRef(CWD.c_str()), 0);
  MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

  // AT_APPLE_flags, the command line arguments of the assembler tool.
  StringRef DwarfDebugFlags = context.getDwarfDebugFlags();
  if (!DwarfDebugFlags.empty()){
    MCOS->EmitBytes(DwarfDebugFlags, 0);
    MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.
  }

  // AT_producer, the version of the assembler tool.
  MCOS->EmitBytes(StringRef("llvm-mc (based on LLVM "), 0);
  MCOS->EmitBytes(StringRef(PACKAGE_VERSION), 0);
  MCOS->EmitBytes(StringRef(")"), 0);
  MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

  // AT_language, a 4 byte value.  We use DW_LANG_Mips_Assembler as the dwarf2
  // draft has no standard code for assembler.
  MCOS->EmitIntValue(dwarf::DW_LANG_Mips_Assembler, 2);

  // Third part: the list of label DIEs.

  // Loop on saved info for dwarf labels and create the DIEs for them.
  const std::vector<const MCGenDwarfLabelEntry *> &Entries =
    MCOS->getContext().getMCGenDwarfLabelEntries();
  for (std::vector<const MCGenDwarfLabelEntry *>::const_iterator it =
       Entries.begin(), ie = Entries.end(); it != ie;
       ++it) {
    const MCGenDwarfLabelEntry *Entry = *it;

    // The DW_TAG_label DIE abbrev (2).
    MCOS->EmitULEB128IntValue(2);

    // AT_name, of the label without any leading underbar.
    MCOS->EmitBytes(Entry->getName(), 0);
    MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

    // AT_decl_file, index into the file table.
    MCOS->EmitIntValue(Entry->getFileNumber(), 4);

    // AT_decl_line, source line number.
    MCOS->EmitIntValue(Entry->getLineNumber(), 4);

    // AT_low_pc, start address of the label.
    const MCExpr *AT_low_pc = MCSymbolRefExpr::Create(Entry->getLabel(),
                                             MCSymbolRefExpr::VK_None, context);
    MCOS->EmitAbsValue(AT_low_pc, AddrSize);

    // DW_AT_prototyped, a one byte flag value of 0 saying we have no prototype.
    MCOS->EmitIntValue(0, 1);

    // The DW_TAG_unspecified_parameters DIE abbrev (3).
    MCOS->EmitULEB128IntValue(3);

    // Add the NULL DIE terminating the DW_TAG_unspecified_parameters DIE's.
    MCOS->EmitIntValue(0, 1);
  }
  // Deallocate the MCGenDwarfLabelEntry classes that saved away the info
  // for the dwarf labels.
  for (std::vector<const MCGenDwarfLabelEntry *>::const_iterator it =
       Entries.begin(), ie = Entries.end(); it != ie;
       ++it) {
    const MCGenDwarfLabelEntry *Entry = *it;
    delete Entry;
  }

  // Add the NULL DIE terminating the Compile Unit DIE's.
  MCOS->EmitIntValue(0, 1);

  // Now set the value of the symbol at the end of the info section.
  MCOS->EmitLabel(InfoEnd);
}

//
// When generating dwarf for assembly source files this emits the Dwarf
// sections.
//
void MCGenDwarfInfo::Emit(MCStreamer *MCOS, const MCSymbol *LineSectionSymbol) {
  // Create the dwarf sections in this order (.debug_line already created).
  MCContext &context = MCOS->getContext();
  const MCAsmInfo &AsmInfo = context.getAsmInfo();
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfInfoSection());
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfAbbrevSection());
  MCSymbol *AbbrevSectionSymbol;
  if (AsmInfo.doesDwarfUseRelocationsAcrossSections()) {
    AbbrevSectionSymbol = context.CreateTempSymbol();
    MCOS->EmitLabel(AbbrevSectionSymbol);
  } else {
    AbbrevSectionSymbol = NULL;
    LineSectionSymbol = NULL;
  }
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfARangesSection());

  // If there are no line table entries then do not emit any section contents.
  if (context.getMCLineSections().empty())
    return;

  // Output the data for .debug_aranges section.
  EmitGenDwarfAranges(MCOS);

  // Output the data for .debug_abbrev section.
  EmitGenDwarfAbbrev(MCOS);

  // Output the data for .debug_info section.
  EmitGenDwarfInfo(MCOS, AbbrevSectionSymbol, LineSectionSymbol);
}

//
// When generating dwarf for assembly source files this is called when symbol
// for a label is created.  If this symbol is not a temporary and is in the
// section that dwarf is being generated for, save the needed info to create
// a dwarf label.
//
void MCGenDwarfLabelEntry::Make(MCSymbol *Symbol, MCStreamer *MCOS,
                                     SourceMgr &SrcMgr, SMLoc &Loc) {
  // We won't create dwarf labels for temporary symbols or symbols not in
  // the default text.
  if (Symbol->isTemporary())
    return;
  MCContext &context = MCOS->getContext();
  if (context.getGenDwarfSection() != MCOS->getCurrentSection())
    return;

  // The dwarf label's name does not have the symbol name's leading
  // underbar if any.
  StringRef Name = Symbol->getName();
  if (Name.startswith("_"))
    Name = Name.substr(1, Name.size()-1);

  // Get the dwarf file number to be used for the dwarf label.
  unsigned FileNumber = context.getGenDwarfFileNumber();

  // Finding the line number is the expensive part which is why we just don't
  // pass it in as for some symbols we won't create a dwarf label.
  int CurBuffer = SrcMgr.FindBufferContainingLoc(Loc);
  unsigned LineNumber = SrcMgr.FindLineNumber(Loc, CurBuffer);

  // We create a temporary symbol for use for the AT_high_pc and AT_low_pc
  // values so that they don't have things like an ARM thumb bit from the
  // original symbol. So when used they won't get a low bit set after
  // relocation.
  MCSymbol *Label = context.CreateTempSymbol();
  MCOS->EmitLabel(Label);

  // Create and entry for the info and add it to the other entries.
  MCGenDwarfLabelEntry *Entry =
    new MCGenDwarfLabelEntry(Name, FileNumber, LineNumber, Label);
  MCOS->getContext().addMCGenDwarfLabelEntry(Entry);
}

static int getDataAlignmentFactor(MCStreamer &streamer) {
  MCContext &context = streamer.getContext();
  const MCAsmInfo &asmInfo = context.getAsmInfo();
  int size = asmInfo.getPointerSize();
  if (asmInfo.isStackGrowthDirectionUp())
    return size;
  else
    return -size;
}

static unsigned getSizeForEncoding(MCStreamer &streamer,
                                   unsigned symbolEncoding) {
  MCContext &context = streamer.getContext();
  unsigned format = symbolEncoding & 0x0f;
  switch (format) {
  default: llvm_unreachable("Unknown Encoding");
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    return context.getAsmInfo().getPointerSize();
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return 8;
  }
}

static void EmitSymbol(MCStreamer &streamer, const MCSymbol &symbol,
                       unsigned symbolEncoding, const char *comment = 0) {
  MCContext &context = streamer.getContext();
  const MCAsmInfo &asmInfo = context.getAsmInfo();
  const MCExpr *v = asmInfo.getExprForFDESymbol(&symbol,
                                                symbolEncoding,
                                                streamer);
  unsigned size = getSizeForEncoding(streamer, symbolEncoding);
  if (streamer.isVerboseAsm() && comment) streamer.AddComment(comment);
  streamer.EmitAbsValue(v, size);
}

static void EmitPersonality(MCStreamer &streamer, const MCSymbol &symbol,
                            unsigned symbolEncoding) {
  MCContext &context = streamer.getContext();
  const MCAsmInfo &asmInfo = context.getAsmInfo();
  const MCExpr *v = asmInfo.getExprForPersonalitySymbol(&symbol,
                                                        symbolEncoding,
                                                        streamer);
  unsigned size = getSizeForEncoding(streamer, symbolEncoding);
  streamer.EmitValue(v, size);
}

static const MachineLocation TranslateMachineLocation(
                                                  const MCRegisterInfo &MRI,
                                                  const MachineLocation &Loc) {
  unsigned Reg = Loc.getReg() == MachineLocation::VirtualFP ?
    MachineLocation::VirtualFP :
    unsigned(MRI.getDwarfRegNum(Loc.getReg(), true));
  const MachineLocation &NewLoc = Loc.isReg() ?
    MachineLocation(Reg) : MachineLocation(Reg, Loc.getOffset());
  return NewLoc;
}

namespace {
  class FrameEmitterImpl {
    int CFAOffset;
    int CIENum;
    bool UsingCFI;
    bool IsEH;
    const MCSymbol *SectionStart;
  public:
    FrameEmitterImpl(bool usingCFI, bool isEH)
      : CFAOffset(0), CIENum(0), UsingCFI(usingCFI), IsEH(isEH),
        SectionStart(0) {}

    void setSectionStart(const MCSymbol *Label) { SectionStart = Label; }

    /// EmitCompactUnwind - Emit the unwind information in a compact way. If
    /// we're successful, return 'true'. Otherwise, return 'false' and it will
    /// emit the normal CIE and FDE.
    bool EmitCompactUnwind(MCStreamer &streamer,
                           const MCDwarfFrameInfo &frame);

    const MCSymbol &EmitCIE(MCStreamer &streamer,
                            const MCSymbol *personality,
                            unsigned personalityEncoding,
                            const MCSymbol *lsda,
                            bool IsSignalFrame,
                            unsigned lsdaEncoding);
    MCSymbol *EmitFDE(MCStreamer &streamer,
                      const MCSymbol &cieStart,
                      const MCDwarfFrameInfo &frame);
    void EmitCFIInstructions(MCStreamer &streamer,
                             const std::vector<MCCFIInstruction> &Instrs,
                             MCSymbol *BaseLabel);
    void EmitCFIInstruction(MCStreamer &Streamer,
                            const MCCFIInstruction &Instr);
  };

} // end anonymous namespace

static void EmitEncodingByte(MCStreamer &Streamer, unsigned Encoding,
                             StringRef Prefix) {
  if (Streamer.isVerboseAsm()) {
    const char *EncStr;
    switch (Encoding) {
    default: EncStr = "<unknown encoding>"; break;
    case dwarf::DW_EH_PE_absptr: EncStr = "absptr"; break;
    case dwarf::DW_EH_PE_omit:   EncStr = "omit"; break;
    case dwarf::DW_EH_PE_pcrel:  EncStr = "pcrel"; break;
    case dwarf::DW_EH_PE_udata4: EncStr = "udata4"; break;
    case dwarf::DW_EH_PE_udata8: EncStr = "udata8"; break;
    case dwarf::DW_EH_PE_sdata4: EncStr = "sdata4"; break;
    case dwarf::DW_EH_PE_sdata8: EncStr = "sdata8"; break;
    case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata4:
      EncStr = "pcrel udata4";
      break;
    case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4:
      EncStr = "pcrel sdata4";
      break;
    case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8:
      EncStr = "pcrel udata8";
      break;
    case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8:
      EncStr = "screl sdata8";
      break;
    case dwarf::DW_EH_PE_indirect |dwarf::DW_EH_PE_pcrel|dwarf::DW_EH_PE_udata4:
      EncStr = "indirect pcrel udata4";
      break;
    case dwarf::DW_EH_PE_indirect |dwarf::DW_EH_PE_pcrel|dwarf::DW_EH_PE_sdata4:
      EncStr = "indirect pcrel sdata4";
      break;
    case dwarf::DW_EH_PE_indirect |dwarf::DW_EH_PE_pcrel|dwarf::DW_EH_PE_udata8:
      EncStr = "indirect pcrel udata8";
      break;
    case dwarf::DW_EH_PE_indirect |dwarf::DW_EH_PE_pcrel|dwarf::DW_EH_PE_sdata8:
      EncStr = "indirect pcrel sdata8";
      break;
    }

    Streamer.AddComment(Twine(Prefix) + " = " + EncStr);
  }

  Streamer.EmitIntValue(Encoding, 1);
}

void FrameEmitterImpl::EmitCFIInstruction(MCStreamer &Streamer,
                                          const MCCFIInstruction &Instr) {
  int dataAlignmentFactor = getDataAlignmentFactor(Streamer);
  bool VerboseAsm = Streamer.isVerboseAsm();

  switch (Instr.getOperation()) {
  case MCCFIInstruction::Move:
  case MCCFIInstruction::RelMove: {
    const MachineLocation &Dst = Instr.getDestination();
    const MachineLocation &Src = Instr.getSource();
    const bool IsRelative = Instr.getOperation() == MCCFIInstruction::RelMove;

    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      if (Src.getReg() == MachineLocation::VirtualFP) {
        if (VerboseAsm) Streamer.AddComment("DW_CFA_def_cfa_offset");
        Streamer.EmitIntValue(dwarf::DW_CFA_def_cfa_offset, 1);
      } else {
        if (VerboseAsm) Streamer.AddComment("DW_CFA_def_cfa");
        Streamer.EmitIntValue(dwarf::DW_CFA_def_cfa, 1);
        if (VerboseAsm) Streamer.AddComment(Twine("Reg ") +
                                            Twine(Src.getReg()));
        Streamer.EmitULEB128IntValue(Src.getReg());
      }

      if (IsRelative)
        CFAOffset += Src.getOffset();
      else
        CFAOffset = -Src.getOffset();

      if (VerboseAsm) Streamer.AddComment(Twine("Offset " + Twine(CFAOffset)));
      Streamer.EmitULEB128IntValue(CFAOffset);
      return;
    }

    if (Src.isReg() && Src.getReg() == MachineLocation::VirtualFP) {
      assert(Dst.isReg() && "Machine move not supported yet.");
      if (VerboseAsm) Streamer.AddComment("DW_CFA_def_cfa_register");
      Streamer.EmitIntValue(dwarf::DW_CFA_def_cfa_register, 1);
      if (VerboseAsm) Streamer.AddComment(Twine("Reg ") + Twine(Dst.getReg()));
      Streamer.EmitULEB128IntValue(Dst.getReg());
      return;
    }

    unsigned Reg = Src.getReg();
    int Offset = Dst.getOffset();
    if (IsRelative)
      Offset -= CFAOffset;
    Offset = Offset / dataAlignmentFactor;

    if (Offset < 0) {
      if (VerboseAsm) Streamer.AddComment("DW_CFA_offset_extended_sf");
      Streamer.EmitIntValue(dwarf::DW_CFA_offset_extended_sf, 1);
      if (VerboseAsm) Streamer.AddComment(Twine("Reg ") + Twine(Reg));
      Streamer.EmitULEB128IntValue(Reg);
      if (VerboseAsm) Streamer.AddComment(Twine("Offset ") + Twine(Offset));
      Streamer.EmitSLEB128IntValue(Offset);
    } else if (Reg < 64) {
      if (VerboseAsm) Streamer.AddComment(Twine("DW_CFA_offset + Reg(") +
                                          Twine(Reg) + ")");
      Streamer.EmitIntValue(dwarf::DW_CFA_offset + Reg, 1);
      if (VerboseAsm) Streamer.AddComment(Twine("Offset ") + Twine(Offset));
      Streamer.EmitULEB128IntValue(Offset);
    } else {
      if (VerboseAsm) Streamer.AddComment("DW_CFA_offset_extended");
      Streamer.EmitIntValue(dwarf::DW_CFA_offset_extended, 1);
      if (VerboseAsm) Streamer.AddComment(Twine("Reg ") + Twine(Reg));
      Streamer.EmitULEB128IntValue(Reg);
      if (VerboseAsm) Streamer.AddComment(Twine("Offset ") + Twine(Offset));
      Streamer.EmitULEB128IntValue(Offset);
    }
    return;
  }
  case MCCFIInstruction::RememberState:
    if (VerboseAsm) Streamer.AddComment("DW_CFA_remember_state");
    Streamer.EmitIntValue(dwarf::DW_CFA_remember_state, 1);
    return;
  case MCCFIInstruction::RestoreState:
    if (VerboseAsm) Streamer.AddComment("DW_CFA_restore_state");
    Streamer.EmitIntValue(dwarf::DW_CFA_restore_state, 1);
    return;
  case MCCFIInstruction::SameValue: {
    unsigned Reg = Instr.getDestination().getReg();
    if (VerboseAsm) Streamer.AddComment("DW_CFA_same_value");
    Streamer.EmitIntValue(dwarf::DW_CFA_same_value, 1);
    if (VerboseAsm) Streamer.AddComment(Twine("Reg ") + Twine(Reg));
    Streamer.EmitULEB128IntValue(Reg);
    return;
  }
  case MCCFIInstruction::Restore: {
    unsigned Reg = Instr.getDestination().getReg();
    if (VerboseAsm) {
      Streamer.AddComment("DW_CFA_restore");
      Streamer.AddComment(Twine("Reg ") + Twine(Reg));
    }
    Streamer.EmitIntValue(dwarf::DW_CFA_restore | Reg, 1);
    return;
  }
  case MCCFIInstruction::Escape:
    if (VerboseAsm) Streamer.AddComment("Escape bytes");
    Streamer.EmitBytes(Instr.getValues(), 0);
    return;
  }
  llvm_unreachable("Unhandled case in switch");
}

/// EmitFrameMoves - Emit frame instructions to describe the layout of the
/// frame.
void FrameEmitterImpl::EmitCFIInstructions(MCStreamer &streamer,
                                    const std::vector<MCCFIInstruction> &Instrs,
                                           MCSymbol *BaseLabel) {
  for (unsigned i = 0, N = Instrs.size(); i < N; ++i) {
    const MCCFIInstruction &Instr = Instrs[i];
    MCSymbol *Label = Instr.getLabel();
    // Throw out move if the label is invalid.
    if (Label && !Label->isDefined()) continue; // Not emitted, in dead code.

    // Advance row if new location.
    if (BaseLabel && Label) {
      MCSymbol *ThisSym = Label;
      if (ThisSym != BaseLabel) {
        if (streamer.isVerboseAsm()) streamer.AddComment("DW_CFA_advance_loc4");
        streamer.EmitDwarfAdvanceFrameAddr(BaseLabel, ThisSym);
        BaseLabel = ThisSym;
      }
    }

    EmitCFIInstruction(streamer, Instr);
  }
}

/// EmitCompactUnwind - Emit the unwind information in a compact way. If we're
/// successful, return 'true'. Otherwise, return 'false' and it will emit the
/// normal CIE and FDE.
bool FrameEmitterImpl::EmitCompactUnwind(MCStreamer &Streamer,
                                         const MCDwarfFrameInfo &Frame) {
  MCContext &Context = Streamer.getContext();
  const MCObjectFileInfo *MOFI = Context.getObjectFileInfo();
  bool VerboseAsm = Streamer.isVerboseAsm();

  // range-start range-length  compact-unwind-enc personality-func   lsda
  //  _foo       LfooEnd-_foo  0x00000023          0                 0
  //  _bar       LbarEnd-_bar  0x00000025         __gxx_personality  except_tab1
  //
  //   .section __LD,__compact_unwind,regular,debug
  //
  //   # compact unwind for _foo
  //   .quad _foo
  //   .set L1,LfooEnd-_foo
  //   .long L1
  //   .long 0x01010001
  //   .quad 0
  //   .quad 0
  //
  //   # compact unwind for _bar
  //   .quad _bar
  //   .set L2,LbarEnd-_bar
  //   .long L2
  //   .long 0x01020011
  //   .quad __gxx_personality
  //   .quad except_tab1

  uint32_t Encoding = Frame.CompactUnwindEncoding;
  if (!Encoding) return false;

  // The encoding needs to know we have an LSDA.
  if (Frame.Lsda)
    Encoding |= 0x40000000;

  Streamer.SwitchSection(MOFI->getCompactUnwindSection());

  // Range Start
  unsigned FDEEncoding = MOFI->getFDEEncoding(UsingCFI);
  unsigned Size = getSizeForEncoding(Streamer, FDEEncoding);
  if (VerboseAsm) Streamer.AddComment("Range Start");
  Streamer.EmitSymbolValue(Frame.Function, Size);

  // Range Length
  const MCExpr *Range = MakeStartMinusEndExpr(Streamer, *Frame.Begin,
                                              *Frame.End, 0);
  if (VerboseAsm) Streamer.AddComment("Range Length");
  Streamer.EmitAbsValue(Range, 4);

  // Compact Encoding
  Size = getSizeForEncoding(Streamer, dwarf::DW_EH_PE_udata4);
  if (VerboseAsm) Streamer.AddComment("Compact Unwind Encoding: 0x" +
                                      Twine::utohexstr(Encoding));
  Streamer.EmitIntValue(Encoding, Size);


  // Personality Function
  Size = getSizeForEncoding(Streamer, dwarf::DW_EH_PE_absptr);
  if (VerboseAsm) Streamer.AddComment("Personality Function");
  if (Frame.Personality)
    Streamer.EmitSymbolValue(Frame.Personality, Size);
  else
    Streamer.EmitIntValue(0, Size); // No personality fn

  // LSDA
  Size = getSizeForEncoding(Streamer, Frame.LsdaEncoding);
  if (VerboseAsm) Streamer.AddComment("LSDA");
  if (Frame.Lsda)
    Streamer.EmitSymbolValue(Frame.Lsda, Size);
  else
    Streamer.EmitIntValue(0, Size); // No LSDA

  return true;
}

const MCSymbol &FrameEmitterImpl::EmitCIE(MCStreamer &streamer,
                                          const MCSymbol *personality,
                                          unsigned personalityEncoding,
                                          const MCSymbol *lsda,
                                          bool IsSignalFrame,
                                          unsigned lsdaEncoding) {
  MCContext &context = streamer.getContext();
  const MCRegisterInfo &MRI = context.getRegisterInfo();
  const MCObjectFileInfo *MOFI = context.getObjectFileInfo();
  bool verboseAsm = streamer.isVerboseAsm();

  MCSymbol *sectionStart;
  if (MOFI->isFunctionEHFrameSymbolPrivate() || !IsEH)
    sectionStart = context.CreateTempSymbol();
  else
    sectionStart = context.GetOrCreateSymbol(Twine("EH_frame") + Twine(CIENum));

  streamer.EmitLabel(sectionStart);
  CIENum++;

  MCSymbol *sectionEnd = context.CreateTempSymbol();

  // Length
  const MCExpr *Length = MakeStartMinusEndExpr(streamer, *sectionStart,
                                               *sectionEnd, 4);
  if (verboseAsm) streamer.AddComment("CIE Length");
  streamer.EmitAbsValue(Length, 4);

  // CIE ID
  unsigned CIE_ID = IsEH ? 0 : -1;
  if (verboseAsm) streamer.AddComment("CIE ID Tag");
  streamer.EmitIntValue(CIE_ID, 4);

  // Version
  if (verboseAsm) streamer.AddComment("DW_CIE_VERSION");
  streamer.EmitIntValue(dwarf::DW_CIE_VERSION, 1);

  // Augmentation String
  SmallString<8> Augmentation;
  if (IsEH) {
    if (verboseAsm) streamer.AddComment("CIE Augmentation");
    Augmentation += "z";
    if (personality)
      Augmentation += "P";
    if (lsda)
      Augmentation += "L";
    Augmentation += "R";
    if (IsSignalFrame)
      Augmentation += "S";
    streamer.EmitBytes(Augmentation.str(), 0);
  }
  streamer.EmitIntValue(0, 1);

  // Code Alignment Factor
  if (verboseAsm) streamer.AddComment("CIE Code Alignment Factor");
  streamer.EmitULEB128IntValue(1);

  // Data Alignment Factor
  if (verboseAsm) streamer.AddComment("CIE Data Alignment Factor");
  streamer.EmitSLEB128IntValue(getDataAlignmentFactor(streamer));

  // Return Address Register
  if (verboseAsm) streamer.AddComment("CIE Return Address Column");
  streamer.EmitULEB128IntValue(MRI.getDwarfRegNum(MRI.getRARegister(), true));

  // Augmentation Data Length (optional)

  unsigned augmentationLength = 0;
  if (IsEH) {
    if (personality) {
      // Personality Encoding
      augmentationLength += 1;
      // Personality
      augmentationLength += getSizeForEncoding(streamer, personalityEncoding);
    }
    if (lsda)
      augmentationLength += 1;
    // Encoding of the FDE pointers
    augmentationLength += 1;

    if (verboseAsm) streamer.AddComment("Augmentation Size");
    streamer.EmitULEB128IntValue(augmentationLength);

    // Augmentation Data (optional)
    if (personality) {
      // Personality Encoding
      EmitEncodingByte(streamer, personalityEncoding,
                       "Personality Encoding");
      // Personality
      if (verboseAsm) streamer.AddComment("Personality");
      EmitPersonality(streamer, *personality, personalityEncoding);
    }

    if (lsda)
      EmitEncodingByte(streamer, lsdaEncoding, "LSDA Encoding");

    // Encoding of the FDE pointers
    EmitEncodingByte(streamer, MOFI->getFDEEncoding(UsingCFI),
                     "FDE Encoding");
  }

  // Initial Instructions

  const MCAsmInfo &MAI = context.getAsmInfo();
  const std::vector<MachineMove> &Moves = MAI.getInitialFrameState();
  std::vector<MCCFIInstruction> Instructions;

  for (int i = 0, n = Moves.size(); i != n; ++i) {
    MCSymbol *Label = Moves[i].getLabel();
    const MachineLocation &Dst =
      TranslateMachineLocation(MRI, Moves[i].getDestination());
    const MachineLocation &Src =
      TranslateMachineLocation(MRI, Moves[i].getSource());
    MCCFIInstruction Inst(Label, Dst, Src);
    Instructions.push_back(Inst);
  }

  EmitCFIInstructions(streamer, Instructions, NULL);

  // Padding
  streamer.EmitValueToAlignment(IsEH
                                ? 4 : context.getAsmInfo().getPointerSize());

  streamer.EmitLabel(sectionEnd);
  return *sectionStart;
}

MCSymbol *FrameEmitterImpl::EmitFDE(MCStreamer &streamer,
                                    const MCSymbol &cieStart,
                                    const MCDwarfFrameInfo &frame) {
  MCContext &context = streamer.getContext();
  MCSymbol *fdeStart = context.CreateTempSymbol();
  MCSymbol *fdeEnd = context.CreateTempSymbol();
  const MCObjectFileInfo *MOFI = context.getObjectFileInfo();
  bool verboseAsm = streamer.isVerboseAsm();

  if (IsEH && frame.Function && !MOFI->isFunctionEHFrameSymbolPrivate()) {
    MCSymbol *EHSym =
      context.GetOrCreateSymbol(frame.Function->getName() + Twine(".eh"));
    streamer.EmitEHSymAttributes(frame.Function, EHSym);
    streamer.EmitLabel(EHSym);
  }

  // Length
  const MCExpr *Length = MakeStartMinusEndExpr(streamer, *fdeStart, *fdeEnd, 0);
  if (verboseAsm) streamer.AddComment("FDE Length");
  streamer.EmitAbsValue(Length, 4);

  streamer.EmitLabel(fdeStart);

  // CIE Pointer
  const MCAsmInfo &asmInfo = context.getAsmInfo();
  if (IsEH) {
    const MCExpr *offset = MakeStartMinusEndExpr(streamer, cieStart, *fdeStart,
                                                 0);
    if (verboseAsm) streamer.AddComment("FDE CIE Offset");
    streamer.EmitAbsValue(offset, 4);
  } else if (!asmInfo.doesDwarfUseRelocationsAcrossSections()) {
    const MCExpr *offset = MakeStartMinusEndExpr(streamer, *SectionStart,
                                                 cieStart, 0);
    streamer.EmitAbsValue(offset, 4);
  } else {
    streamer.EmitSymbolValue(&cieStart, 4);
  }

  // PC Begin
  unsigned PCEncoding = IsEH ? MOFI->getFDEEncoding(UsingCFI)
                             : (unsigned)dwarf::DW_EH_PE_absptr;
  unsigned PCSize = getSizeForEncoding(streamer, PCEncoding);
  EmitSymbol(streamer, *frame.Begin, PCEncoding, "FDE initial location");

  // PC Range
  const MCExpr *Range = MakeStartMinusEndExpr(streamer, *frame.Begin,
                                              *frame.End, 0);
  if (verboseAsm) streamer.AddComment("FDE address range");
  streamer.EmitAbsValue(Range, PCSize);

  if (IsEH) {
    // Augmentation Data Length
    unsigned augmentationLength = 0;

    if (frame.Lsda)
      augmentationLength += getSizeForEncoding(streamer, frame.LsdaEncoding);

    if (verboseAsm) streamer.AddComment("Augmentation size");
    streamer.EmitULEB128IntValue(augmentationLength);

    // Augmentation Data
    if (frame.Lsda)
      EmitSymbol(streamer, *frame.Lsda, frame.LsdaEncoding,
                 "Language Specific Data Area");
  }

  // Call Frame Instructions

  EmitCFIInstructions(streamer, frame.Instructions, frame.Begin);

  // Padding
  streamer.EmitValueToAlignment(PCSize);

  return fdeEnd;
}

namespace {
  struct CIEKey {
    static const CIEKey getEmptyKey() { return CIEKey(0, 0, -1, false); }
    static const CIEKey getTombstoneKey() { return CIEKey(0, -1, 0, false); }

    CIEKey(const MCSymbol* Personality_, unsigned PersonalityEncoding_,
           unsigned LsdaEncoding_, bool IsSignalFrame_) :
      Personality(Personality_), PersonalityEncoding(PersonalityEncoding_),
      LsdaEncoding(LsdaEncoding_), IsSignalFrame(IsSignalFrame_) {
    }
    const MCSymbol* Personality;
    unsigned PersonalityEncoding;
    unsigned LsdaEncoding;
    bool IsSignalFrame;
  };
}

namespace llvm {
  template <>
  struct DenseMapInfo<CIEKey> {
    static CIEKey getEmptyKey() {
      return CIEKey::getEmptyKey();
    }
    static CIEKey getTombstoneKey() {
      return CIEKey::getTombstoneKey();
    }
    static unsigned getHashValue(const CIEKey &Key) {
      return static_cast<unsigned>(hash_combine(Key.Personality,
                                                Key.PersonalityEncoding,
                                                Key.LsdaEncoding,
                                                Key.IsSignalFrame));
    }
    static bool isEqual(const CIEKey &LHS,
                        const CIEKey &RHS) {
      return LHS.Personality == RHS.Personality &&
        LHS.PersonalityEncoding == RHS.PersonalityEncoding &&
        LHS.LsdaEncoding == RHS.LsdaEncoding &&
        LHS.IsSignalFrame == RHS.IsSignalFrame;
    }
  };
}

void MCDwarfFrameEmitter::Emit(MCStreamer &Streamer,
                               bool UsingCFI,
                               bool IsEH) {
  MCContext &Context = Streamer.getContext();
  MCObjectFileInfo *MOFI =
    const_cast<MCObjectFileInfo*>(Context.getObjectFileInfo());
  FrameEmitterImpl Emitter(UsingCFI, IsEH);
  ArrayRef<MCDwarfFrameInfo> FrameArray = Streamer.getFrameInfos();

  // Emit the compact unwind info if available.
  if (IsEH && MOFI->getCompactUnwindSection())
    for (unsigned i = 0, n = Streamer.getNumFrameInfos(); i < n; ++i) {
      const MCDwarfFrameInfo &Frame = Streamer.getFrameInfo(i);
      if (Frame.CompactUnwindEncoding)
        Emitter.EmitCompactUnwind(Streamer, Frame);
    }

  const MCSection &Section = IsEH ? *MOFI->getEHFrameSection() :
                                    *MOFI->getDwarfFrameSection();
  Streamer.SwitchSection(&Section);
  MCSymbol *SectionStart = Context.CreateTempSymbol();
  Streamer.EmitLabel(SectionStart);
  Emitter.setSectionStart(SectionStart);

  MCSymbol *FDEEnd = NULL;
  DenseMap<CIEKey, const MCSymbol*> CIEStarts;

  const MCSymbol *DummyDebugKey = NULL;
  for (unsigned i = 0, n = FrameArray.size(); i < n; ++i) {
    const MCDwarfFrameInfo &Frame = FrameArray[i];
    CIEKey Key(Frame.Personality, Frame.PersonalityEncoding,
               Frame.LsdaEncoding, Frame.IsSignalFrame);
    const MCSymbol *&CIEStart = IsEH ? CIEStarts[Key] : DummyDebugKey;
    if (!CIEStart)
      CIEStart = &Emitter.EmitCIE(Streamer, Frame.Personality,
                                  Frame.PersonalityEncoding, Frame.Lsda,
                                  Frame.IsSignalFrame,
                                  Frame.LsdaEncoding);

    FDEEnd = Emitter.EmitFDE(Streamer, *CIEStart, Frame);

    if (i != n - 1)
      Streamer.EmitLabel(FDEEnd);
  }

  Streamer.EmitValueToAlignment(Context.getAsmInfo().getPointerSize());
  if (FDEEnd)
    Streamer.EmitLabel(FDEEnd);
}

void MCDwarfFrameEmitter::EmitAdvanceLoc(MCStreamer &Streamer,
                                         uint64_t AddrDelta) {
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfFrameEmitter::EncodeAdvanceLoc(AddrDelta, OS);
  Streamer.EmitBytes(OS.str(), /*AddrSpace=*/0);
}

void MCDwarfFrameEmitter::EncodeAdvanceLoc(uint64_t AddrDelta,
                                           raw_ostream &OS) {
  // FIXME: Assumes the code alignment factor is 1.
  if (AddrDelta == 0) {
  } else if (isUIntN(6, AddrDelta)) {
    uint8_t Opcode = dwarf::DW_CFA_advance_loc | AddrDelta;
    OS << Opcode;
  } else if (isUInt<8>(AddrDelta)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc1);
    OS << uint8_t(AddrDelta);
  } else if (isUInt<16>(AddrDelta)) {
    // FIXME: check what is the correct behavior on a big endian machine.
    OS << uint8_t(dwarf::DW_CFA_advance_loc2);
    OS << uint8_t( AddrDelta       & 0xff);
    OS << uint8_t((AddrDelta >> 8) & 0xff);
  } else {
    // FIXME: check what is the correct behavior on a big endian machine.
    assert(isUInt<32>(AddrDelta));
    OS << uint8_t(dwarf::DW_CFA_advance_loc4);
    OS << uint8_t( AddrDelta        & 0xff);
    OS << uint8_t((AddrDelta >> 8)  & 0xff);
    OS << uint8_t((AddrDelta >> 16) & 0xff);
    OS << uint8_t((AddrDelta >> 24) & 0xff);

  }
}
