//===- MCContext.h - Machine Code Context -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCONTEXT_H
#define LLVM_MC_MCCONTEXT_H

#include "llvm/MC/SectionKind.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <vector> // FIXME: Shouldn't be needed.

namespace llvm {
  class MCAsmInfo;
  class MCExpr;
  class MCSection;
  class MCSymbol;
  class MCLabel;
  class MCDwarfFile;
  class MCDwarfLoc;
  class MCObjectFileInfo;
  class MCRegisterInfo;
  class MCLineSection;
  class SMLoc;
  class StringRef;
  class Twine;
  class MCSectionMachO;
  class MCSectionELF;

  /// MCContext - Context object for machine code objects.  This class owns all
  /// of the sections that it creates.
  ///
  class MCContext {
    MCContext(const MCContext&) LLVM_DELETED_FUNCTION;
    MCContext &operator=(const MCContext&) LLVM_DELETED_FUNCTION;
  public:
    typedef StringMap<MCSymbol*, BumpPtrAllocator&> SymbolTable;
  private:
    /// The SourceMgr for this object, if any.
    const SourceMgr *SrcMgr;

    /// The MCAsmInfo for this target.
    const MCAsmInfo &MAI;

    /// The MCRegisterInfo for this target.
    const MCRegisterInfo &MRI;

    /// The MCObjectFileInfo for this target.
    const MCObjectFileInfo *MOFI;

    /// Allocator - Allocator object used for creating machine code objects.
    ///
    /// We use a bump pointer allocator to avoid the need to track all allocated
    /// objects.
    BumpPtrAllocator Allocator;

    /// Symbols - Bindings of names to symbols.
    SymbolTable Symbols;

    /// UsedNames - Keeps tracks of names that were used both for used declared
    /// and artificial symbols.
    StringMap<bool, BumpPtrAllocator&> UsedNames;

    /// NextUniqueID - The next ID to dole out to an unnamed assembler temporary
    /// symbol.
    unsigned NextUniqueID;

    /// Instances of directional local labels.
    DenseMap<unsigned, MCLabel *> Instances;
    /// NextInstance() creates the next instance of the directional local label
    /// for the LocalLabelVal and adds it to the map if needed.
    unsigned NextInstance(int64_t LocalLabelVal);
    /// GetInstance() gets the current instance of the directional local label
    /// for the LocalLabelVal and adds it to the map if needed.
    unsigned GetInstance(int64_t LocalLabelVal);

    /// The file name of the log file from the environment variable
    /// AS_SECURE_LOG_FILE.  Which must be set before the .secure_log_unique
    /// directive is used or it is an error.
    char *SecureLogFile;
    /// The stream that gets written to for the .secure_log_unique directive.
    raw_ostream *SecureLog;
    /// Boolean toggled when .secure_log_unique / .secure_log_reset is seen to
    /// catch errors if .secure_log_unique appears twice without
    /// .secure_log_reset appearing between them.
    bool SecureLogUsed;

    /// The dwarf file and directory tables from the dwarf .file directive.
    std::vector<MCDwarfFile *> MCDwarfFiles;
    std::vector<StringRef> MCDwarfDirs;

    /// The current dwarf line information from the last dwarf .loc directive.
    MCDwarfLoc CurrentDwarfLoc;
    bool DwarfLocSeen;

    /// Generate dwarf debugging info for assembly source files.
    bool GenDwarfForAssembly;

    /// The current dwarf file number when generate dwarf debugging info for
    /// assembly source files.
    unsigned GenDwarfFileNumber;

    /// The default initial text section that we generate dwarf debugging line
    /// info for when generating dwarf assembly source files.
    const MCSection *GenDwarfSection;
    /// Symbols created for the start and end of this section.
    MCSymbol *GenDwarfSectionStartSym, *GenDwarfSectionEndSym;

    /// The information gathered from labels that will have dwarf label
    /// entries when generating dwarf assembly source files.
    std::vector<const MCGenDwarfLabelEntry *> MCGenDwarfLabelEntries;

    /// The string to embed in the debug information for the compile unit, if
    /// non-empty.
    StringRef DwarfDebugFlags;

    /// Honor temporary labels, this is useful for debugging semantic
    /// differences between temporary and non-temporary labels (primarily on
    /// Darwin).
    bool AllowTemporaryLabels;

    /// The dwarf line information from the .loc directives for the sections
    /// with assembled machine instructions have after seeing .loc directives.
    DenseMap<const MCSection *, MCLineSection *> MCLineSections;
    /// We need a deterministic iteration order, so we remember the order
    /// the elements were added.
    std::vector<const MCSection *> MCLineSectionOrder;

    void *MachOUniquingMap, *ELFUniquingMap, *COFFUniquingMap;

    MCSymbol *CreateSymbol(StringRef Name);

  public:
    explicit MCContext(const MCAsmInfo &MAI, const MCRegisterInfo &MRI,
                       const MCObjectFileInfo *MOFI, const SourceMgr *Mgr = 0);
    ~MCContext();

    const SourceMgr *getSourceManager() const { return SrcMgr; }

    const MCAsmInfo &getAsmInfo() const { return MAI; }

    const MCRegisterInfo &getRegisterInfo() const { return MRI; }

    const MCObjectFileInfo *getObjectFileInfo() const { return MOFI; }

    void setAllowTemporaryLabels(bool Value) { AllowTemporaryLabels = Value; }

    /// @name Symbol Management
    /// @{

    /// CreateTempSymbol - Create and return a new assembler temporary symbol
    /// with a unique but unspecified name.
    MCSymbol *CreateTempSymbol();

    /// getUniqueSymbolID() - Return a unique identifier for use in constructing
    /// symbol names.
    unsigned getUniqueSymbolID() { return NextUniqueID++; }

    /// CreateDirectionalLocalSymbol - Create the definition of a directional
    /// local symbol for numbered label (used for "1:" definitions).
    MCSymbol *CreateDirectionalLocalSymbol(int64_t LocalLabelVal);

    /// GetDirectionalLocalSymbol - Create and return a directional local
    /// symbol for numbered label (used for "1b" or 1f" references).
    MCSymbol *GetDirectionalLocalSymbol(int64_t LocalLabelVal, int bORf);

    /// GetOrCreateSymbol - Lookup the symbol inside with the specified
    /// @p Name.  If it exists, return it.  If not, create a forward
    /// reference and return it.
    ///
    /// @param Name - The symbol name, which must be unique across all symbols.
    MCSymbol *GetOrCreateSymbol(StringRef Name);
    MCSymbol *GetOrCreateSymbol(const Twine &Name);

    /// LookupSymbol - Get the symbol for \p Name, or null.
    MCSymbol *LookupSymbol(StringRef Name) const;
    MCSymbol *LookupSymbol(const Twine &Name) const;

    /// getSymbols - Get a reference for the symbol table for clients that
    /// want to, for example, iterate over all symbols. 'const' because we
    /// still want any modifications to the table itself to use the MCContext
    /// APIs.
    const SymbolTable &getSymbols() const {
      return Symbols;
    }

    /// @}

    /// @name Section Management
    /// @{

    /// getMachOSection - Return the MCSection for the specified mach-o section.
    /// This requires the operands to be valid.
    const MCSectionMachO *getMachOSection(StringRef Segment,
                                          StringRef Section,
                                          unsigned TypeAndAttributes,
                                          unsigned Reserved2,
                                          SectionKind K);
    const MCSectionMachO *getMachOSection(StringRef Segment,
                                          StringRef Section,
                                          unsigned TypeAndAttributes,
                                          SectionKind K) {
      return getMachOSection(Segment, Section, TypeAndAttributes, 0, K);
    }

    const MCSectionELF *getELFSection(StringRef Section, unsigned Type,
                                      unsigned Flags, SectionKind Kind);

    const MCSectionELF *getELFSection(StringRef Section, unsigned Type,
                                      unsigned Flags, SectionKind Kind,
                                      unsigned EntrySize, StringRef Group);

    const MCSectionELF *CreateELFGroupSection();

    const MCSection *getCOFFSection(StringRef Section, unsigned Characteristics,
                                    int Selection, SectionKind Kind);

    const MCSection *getCOFFSection(StringRef Section, unsigned Characteristics,
                                    SectionKind Kind) {
      return getCOFFSection (Section, Characteristics, 0, Kind);
    }


    /// @}

    /// @name Dwarf Management
    /// @{

    /// GetDwarfFile - creates an entry in the dwarf file and directory tables.
    unsigned GetDwarfFile(StringRef Directory, StringRef FileName,
                          unsigned FileNumber);

    bool isValidDwarfFileNumber(unsigned FileNumber);

    bool hasDwarfFiles() const {
      return !MCDwarfFiles.empty();
    }

    const std::vector<MCDwarfFile *> &getMCDwarfFiles() {
      return MCDwarfFiles;
    }
    const std::vector<StringRef> &getMCDwarfDirs() {
      return MCDwarfDirs;
    }

    const DenseMap<const MCSection *, MCLineSection *>
    &getMCLineSections() const {
      return MCLineSections;
    }
    const std::vector<const MCSection *> &getMCLineSectionOrder() const {
      return MCLineSectionOrder;
    }
    void addMCLineSection(const MCSection *Sec, MCLineSection *Line) {
      MCLineSections[Sec] = Line;
      MCLineSectionOrder.push_back(Sec);
    }

    /// setCurrentDwarfLoc - saves the information from the currently parsed
    /// dwarf .loc directive and sets DwarfLocSeen.  When the next instruction
    /// is assembled an entry in the line number table with this information and
    /// the address of the instruction will be created.
    void setCurrentDwarfLoc(unsigned FileNum, unsigned Line, unsigned Column,
                            unsigned Flags, unsigned Isa,
                            unsigned Discriminator) {
      CurrentDwarfLoc.setFileNum(FileNum);
      CurrentDwarfLoc.setLine(Line);
      CurrentDwarfLoc.setColumn(Column);
      CurrentDwarfLoc.setFlags(Flags);
      CurrentDwarfLoc.setIsa(Isa);
      CurrentDwarfLoc.setDiscriminator(Discriminator);
      DwarfLocSeen = true;
    }
    void ClearDwarfLocSeen() { DwarfLocSeen = false; }

    bool getDwarfLocSeen() { return DwarfLocSeen; }
    const MCDwarfLoc &getCurrentDwarfLoc() { return CurrentDwarfLoc; }

    bool getGenDwarfForAssembly() { return GenDwarfForAssembly; }
    void setGenDwarfForAssembly(bool Value) { GenDwarfForAssembly = Value; }
    unsigned getGenDwarfFileNumber() { return GenDwarfFileNumber; }
    unsigned nextGenDwarfFileNumber() { return ++GenDwarfFileNumber; }
    const MCSection *getGenDwarfSection() { return GenDwarfSection; }
    void setGenDwarfSection(const MCSection *Sec) { GenDwarfSection = Sec; }
    MCSymbol *getGenDwarfSectionStartSym() { return GenDwarfSectionStartSym; }
    void setGenDwarfSectionStartSym(MCSymbol *Sym) {
      GenDwarfSectionStartSym = Sym;
    }
    MCSymbol *getGenDwarfSectionEndSym() { return GenDwarfSectionEndSym; }
    void setGenDwarfSectionEndSym(MCSymbol *Sym) {
      GenDwarfSectionEndSym = Sym;
    }
    const std::vector<const MCGenDwarfLabelEntry *>
      &getMCGenDwarfLabelEntries() const {
      return MCGenDwarfLabelEntries;
    }
    void addMCGenDwarfLabelEntry(const MCGenDwarfLabelEntry *E) {
      MCGenDwarfLabelEntries.push_back(E);
    }

    void setDwarfDebugFlags(StringRef S) { DwarfDebugFlags = S; }
    StringRef getDwarfDebugFlags() { return DwarfDebugFlags; }

    /// @}

    char *getSecureLogFile() { return SecureLogFile; }
    raw_ostream *getSecureLog() { return SecureLog; }
    bool getSecureLogUsed() { return SecureLogUsed; }
    void setSecureLog(raw_ostream *Value) {
      SecureLog = Value;
    }
    void setSecureLogUsed(bool Value) {
      SecureLogUsed = Value;
    }

    void *Allocate(unsigned Size, unsigned Align = 8) {
      return Allocator.Allocate(Size, Align);
    }
    void Deallocate(void *Ptr) {
    }

    // Unrecoverable error has occured. Display the best diagnostic we can
    // and bail via exit(1). For now, most MC backend errors are unrecoverable.
    // FIXME: We should really do something about that.
    LLVM_ATTRIBUTE_NORETURN void FatalError(SMLoc L, const Twine &Msg);
  };

} // end namespace llvm

// operator new and delete aren't allowed inside namespaces.
// The throw specifications are mandated by the standard.
/// @brief Placement new for using the MCContext's allocator.
///
/// This placement form of operator new uses the MCContext's allocator for
/// obtaining memory. It is a non-throwing new, which means that it returns
/// null on error. (If that is what the allocator does. The current does, so if
/// this ever changes, this operator will have to be changed, too.)
/// Usage looks like this (assuming there's an MCContext 'Context' in scope):
/// @code
/// // Default alignment (16)
/// IntegerLiteral *Ex = new (Context) IntegerLiteral(arguments);
/// // Specific alignment
/// IntegerLiteral *Ex2 = new (Context, 8) IntegerLiteral(arguments);
/// @endcode
/// Please note that you cannot use delete on the pointer; it must be
/// deallocated using an explicit destructor call followed by
/// @c Context.Deallocate(Ptr).
///
/// @param Bytes The number of bytes to allocate. Calculated by the compiler.
/// @param C The MCContext that provides the allocator.
/// @param Alignment The alignment of the allocated memory (if the underlying
///                  allocator supports it).
/// @return The allocated memory. Could be NULL.
inline void *operator new(size_t Bytes, llvm::MCContext &C,
                          size_t Alignment = 16) throw () {
  return C.Allocate(Bytes, Alignment);
}
/// @brief Placement delete companion to the new above.
///
/// This operator is just a companion to the new above. There is no way of
/// invoking it directly; see the new operator for more details. This operator
/// is called implicitly by the compiler if a placement new expression using
/// the MCContext throws in the object constructor.
inline void operator delete(void *Ptr, llvm::MCContext &C, size_t)
              throw () {
  C.Deallocate(Ptr);
}

/// This placement form of operator new[] uses the MCContext's allocator for
/// obtaining memory. It is a non-throwing new[], which means that it returns
/// null on error.
/// Usage looks like this (assuming there's an MCContext 'Context' in scope):
/// @code
/// // Default alignment (16)
/// char *data = new (Context) char[10];
/// // Specific alignment
/// char *data = new (Context, 8) char[10];
/// @endcode
/// Please note that you cannot use delete on the pointer; it must be
/// deallocated using an explicit destructor call followed by
/// @c Context.Deallocate(Ptr).
///
/// @param Bytes The number of bytes to allocate. Calculated by the compiler.
/// @param C The MCContext that provides the allocator.
/// @param Alignment The alignment of the allocated memory (if the underlying
///                  allocator supports it).
/// @return The allocated memory. Could be NULL.
inline void *operator new[](size_t Bytes, llvm::MCContext& C,
                            size_t Alignment = 16) throw () {
  return C.Allocate(Bytes, Alignment);
}

/// @brief Placement delete[] companion to the new[] above.
///
/// This operator is just a companion to the new[] above. There is no way of
/// invoking it directly; see the new[] operator for more details. This operator
/// is called implicitly by the compiler if a placement new[] expression using
/// the MCContext throws in the object constructor.
inline void operator delete[](void *Ptr, llvm::MCContext &C) throw () {
  C.Deallocate(Ptr);
}

#endif
