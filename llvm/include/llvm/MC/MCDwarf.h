//===- MCDwarf.h - Machine Code Dwarf support -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCDwarfFile to support the dwarf
// .file directive and the .loc directive.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCDWARF_H
#define LLVM_MC_MCDWARF_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Compiler.h"
#include <vector>

namespace llvm {
  class MCContext;
  class MCObjectWriter;
  class MCSection;
  class MCStreamer;
  class MCSymbol;
  class SourceMgr;
  class SMLoc;

  /// MCDwarfFile - Instances of this class represent the name of the dwarf
  /// .file directive and its associated dwarf file number in the MC file,
  /// and MCDwarfFile's are created and unique'd by the MCContext class where
  /// the file number for each is its index into the vector of DwarfFiles (note
  /// index 0 is not used and not a valid dwarf file number).
  class MCDwarfFile {
    // Name - the base name of the file without its directory path.
    // The StringRef references memory allocated in the MCContext.
    StringRef Name;

    // DirIndex - the index into the list of directory names for this file name.
    unsigned DirIndex;

  private:  // MCContext creates and uniques these.
    friend class MCContext;
    MCDwarfFile(StringRef name, unsigned dirIndex)
      : Name(name), DirIndex(dirIndex) {}

    MCDwarfFile(const MCDwarfFile&) LLVM_DELETED_FUNCTION;
    void operator=(const MCDwarfFile&) LLVM_DELETED_FUNCTION;
  public:
    /// getName - Get the base name of this MCDwarfFile.
    StringRef getName() const { return Name; }

    /// getDirIndex - Get the dirIndex of this MCDwarfFile.
    unsigned getDirIndex() const { return DirIndex; }


    /// print - Print the value to the stream \p OS.
    void print(raw_ostream &OS) const;

    /// dump - Print the value to stderr.
    void dump() const;
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const MCDwarfFile &DwarfFile){
    DwarfFile.print(OS);
    return OS;
  }

  /// MCDwarfLoc - Instances of this class represent the information from a
  /// dwarf .loc directive.
  class MCDwarfLoc {
    // FileNum - the file number.
    unsigned FileNum;
    // Line - the line number.
    unsigned Line;
    // Column - the column position.
    unsigned Column;
    // Flags (see #define's below)
    unsigned Flags;
    // Isa
    unsigned Isa;
    // Discriminator
    unsigned Discriminator;

// Flag that indicates the initial value of the is_stmt_start flag.
#define DWARF2_LINE_DEFAULT_IS_STMT     1

#define DWARF2_FLAG_IS_STMT        (1 << 0)
#define DWARF2_FLAG_BASIC_BLOCK    (1 << 1)
#define DWARF2_FLAG_PROLOGUE_END   (1 << 2)
#define DWARF2_FLAG_EPILOGUE_BEGIN (1 << 3)

  private:  // MCContext manages these
    friend class MCContext;
    friend class MCLineEntry;
    MCDwarfLoc(unsigned fileNum, unsigned line, unsigned column, unsigned flags,
               unsigned isa, unsigned discriminator)
      : FileNum(fileNum), Line(line), Column(column), Flags(flags), Isa(isa),
        Discriminator(discriminator) {}

    // Allow the default copy constructor and assignment operator to be used
    // for an MCDwarfLoc object.

  public:
    /// getFileNum - Get the FileNum of this MCDwarfLoc.
    unsigned getFileNum() const { return FileNum; }

    /// getLine - Get the Line of this MCDwarfLoc.
    unsigned getLine() const { return Line; }

    /// getColumn - Get the Column of this MCDwarfLoc.
    unsigned getColumn() const { return Column; }

    /// getFlags - Get the Flags of this MCDwarfLoc.
    unsigned getFlags() const { return Flags; }

    /// getIsa - Get the Isa of this MCDwarfLoc.
    unsigned getIsa() const { return Isa; }

    /// getDiscriminator - Get the Discriminator of this MCDwarfLoc.
    unsigned getDiscriminator() const { return Discriminator; }

    /// setFileNum - Set the FileNum of this MCDwarfLoc.
    void setFileNum(unsigned fileNum) { FileNum = fileNum; }

    /// setLine - Set the Line of this MCDwarfLoc.
    void setLine(unsigned line) { Line = line; }

    /// setColumn - Set the Column of this MCDwarfLoc.
    void setColumn(unsigned column) { Column = column; }

    /// setFlags - Set the Flags of this MCDwarfLoc.
    void setFlags(unsigned flags) { Flags = flags; }

    /// setIsa - Set the Isa of this MCDwarfLoc.
    void setIsa(unsigned isa) { Isa = isa; }

    /// setDiscriminator - Set the Discriminator of this MCDwarfLoc.
    void setDiscriminator(unsigned discriminator) {
      Discriminator = discriminator;
    }
  };

  /// MCLineEntry - Instances of this class represent the line information for
  /// the dwarf line table entries.  Which is created after a machine
  /// instruction is assembled and uses an address from a temporary label
  /// created at the current address in the current section and the info from
  /// the last .loc directive seen as stored in the context.
  class MCLineEntry : public MCDwarfLoc {
    MCSymbol *Label;

  private:
    // Allow the default copy constructor and assignment operator to be used
    // for an MCLineEntry object.

  public:
    // Constructor to create an MCLineEntry given a symbol and the dwarf loc.
    MCLineEntry(MCSymbol *label, const MCDwarfLoc loc) : MCDwarfLoc(loc),
                Label(label) {}

    MCSymbol *getLabel() const { return Label; }

    // This is called when an instruction is assembled into the specified
    // section and if there is information from the last .loc directive that
    // has yet to have a line entry made for it is made.
    static void Make(MCStreamer *MCOS, const MCSection *Section);
  };

  /// MCLineSection - Instances of this class represent the line information
  /// for a section where machine instructions have been assembled after seeing
  /// .loc directives.  This is the information used to build the dwarf line
  /// table for a section.
  class MCLineSection {

  private:
    MCLineSection(const MCLineSection&) LLVM_DELETED_FUNCTION;
    void operator=(const MCLineSection&) LLVM_DELETED_FUNCTION;

  public:
    // Constructor to create an MCLineSection with an empty MCLineEntries
    // vector.
    MCLineSection() {}

    // addLineEntry - adds an entry to this MCLineSection's line entries
    void addLineEntry(const MCLineEntry &LineEntry) {
      MCLineEntries.push_back(LineEntry);
    }

    typedef std::vector<MCLineEntry> MCLineEntryCollection;
    typedef MCLineEntryCollection::iterator iterator;
    typedef MCLineEntryCollection::const_iterator const_iterator;

  private:
    MCLineEntryCollection MCLineEntries;

  public:
    const MCLineEntryCollection *getMCLineEntries() const {
      return &MCLineEntries;
    }
  };

  class MCDwarfFileTable {
  public:
    //
    // This emits the Dwarf file and the line tables.
    //
    static const MCSymbol *Emit(MCStreamer *MCOS);
  };

  class MCDwarfLineAddr {
  public:
    /// Utility function to encode a Dwarf pair of LineDelta and AddrDeltas.
    static void Encode(int64_t LineDelta, uint64_t AddrDelta, raw_ostream &OS);

    /// Utility function to emit the encoding to a streamer.
    static void Emit(MCStreamer *MCOS,
                     int64_t LineDelta,uint64_t AddrDelta);

    /// Utility function to write the encoding to an object writer.
    static void Write(MCObjectWriter *OW,
                      int64_t LineDelta, uint64_t AddrDelta);
  };

  class MCGenDwarfInfo {
  public:
    //
    // When generating dwarf for assembly source files this emits the Dwarf
    // sections.
    //
    static void Emit(MCStreamer *MCOS, const MCSymbol *LineSectionSymbol);
  };

  // When generating dwarf for assembly source files this is the info that is
  // needed to be gathered for each symbol that will have a dwarf label.
  class MCGenDwarfLabelEntry {
  private:
    // Name of the symbol without a leading underbar, if any.
    StringRef Name;
    // The dwarf file number this symbol is in.
    unsigned FileNumber;
    // The line number this symbol is at.
    unsigned LineNumber;
    // The low_pc for the dwarf label is taken from this symbol.
    MCSymbol *Label;

  public:
    MCGenDwarfLabelEntry(StringRef name, unsigned fileNumber,
                         unsigned lineNumber, MCSymbol *label) :
      Name(name), FileNumber(fileNumber), LineNumber(lineNumber), Label(label){}

    StringRef getName() const { return Name; }
    unsigned getFileNumber() const { return FileNumber; }
    unsigned getLineNumber() const { return LineNumber; }
    MCSymbol *getLabel() const { return Label; }

    // This is called when label is created when we are generating dwarf for
    // assembly source files.
    static void Make(MCSymbol *Symbol, MCStreamer *MCOS, SourceMgr &SrcMgr,
                     SMLoc &Loc);
  };

  class MCCFIInstruction {
  public:
    enum OpType { SameValue, RememberState, RestoreState, Move, RelMove, Escape,
                  Restore};
  private:
    OpType Operation;
    MCSymbol *Label;
    // Move to & from location.
    MachineLocation Destination;
    MachineLocation Source;
    std::vector<char> Values;
  public:
    MCCFIInstruction(OpType Op, MCSymbol *L)
      : Operation(Op), Label(L) {
      assert(Op == RememberState || Op == RestoreState);
    }
    MCCFIInstruction(OpType Op, MCSymbol *L, unsigned Register)
      : Operation(Op), Label(L), Destination(Register) {
      assert(Op == SameValue || Op == Restore);
    }
    MCCFIInstruction(MCSymbol *L, const MachineLocation &D,
                     const MachineLocation &S)
      : Operation(Move), Label(L), Destination(D), Source(S) {
    }
    MCCFIInstruction(OpType Op, MCSymbol *L, const MachineLocation &D,
                     const MachineLocation &S)
      : Operation(Op), Label(L), Destination(D), Source(S) {
      assert(Op == RelMove);
    }
    MCCFIInstruction(OpType Op, MCSymbol *L, StringRef Vals)
      : Operation(Op), Label(L), Values(Vals.begin(), Vals.end()) {
      assert(Op == Escape);
    }
    OpType getOperation() const { return Operation; }
    MCSymbol *getLabel() const { return Label; }
    const MachineLocation &getDestination() const { return Destination; }
    const MachineLocation &getSource() const { return Source; }
    const StringRef getValues() const {
      return StringRef(&Values[0], Values.size());
    }
  };

  struct MCDwarfFrameInfo {
    MCDwarfFrameInfo() : Begin(0), End(0), Personality(0), Lsda(0),
                         Function(0), Instructions(), PersonalityEncoding(),
                         LsdaEncoding(0), CompactUnwindEncoding(0),
                         IsSignalFrame(false) {}
    MCSymbol *Begin;
    MCSymbol *End;
    const MCSymbol *Personality;
    const MCSymbol *Lsda;
    const MCSymbol *Function;
    std::vector<MCCFIInstruction> Instructions;
    unsigned PersonalityEncoding;
    unsigned LsdaEncoding;
    uint32_t CompactUnwindEncoding;
    bool IsSignalFrame;
  };

  class MCDwarfFrameEmitter {
  public:
    //
    // This emits the frame info section.
    //
    static void Emit(MCStreamer &streamer, bool usingCFI,
                     bool isEH);
    static void EmitAdvanceLoc(MCStreamer &Streamer, uint64_t AddrDelta);
    static void EncodeAdvanceLoc(uint64_t AddrDelta, raw_ostream &OS);
  };
} // end namespace llvm

#endif
