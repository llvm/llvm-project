//===-- llvm/Target/TargetAsmInfo.h - Asm info ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the basis for target specific
// asm writers.  This class primarily takes care of global printing constants,
// which are used in very similar ways across all targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ASM_INFO_H
#define LLVM_TARGET_ASM_INFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
  // DWARF encoding query type
  namespace DwarfEncoding {
    enum Target {
      Data       = 0,
      CodeLabels = 1,
      Functions  = 2
    };
  }

  namespace SectionKind {
    enum Kind {
      Unknown = 0,      ///< Custom section
      Text,             ///< Text section
      Data,             ///< Data section
      BSS,              ///< BSS section
      ROData,           ///< Readonly data section
      RODataMergeStr,   ///< Readonly data section (mergeable strings)
      RODataMergeConst, ///< Readonly data section (mergeable constants)
      SmallData,        ///< Small data section
      SmallBSS,         ///< Small bss section
      SmallROData,      ///< Small readonly section
      ThreadData,       ///< Initialized TLS data objects
      ThreadBSS         ///< Uninitialized TLS data objects
    };

    static inline bool isReadOnly(Kind K) {
      return (K == SectionKind::ROData ||
              K == SectionKind::RODataMergeConst ||
              K == SectionKind::RODataMergeStr ||
              K == SectionKind::SmallROData);
    }

    static inline bool isBSS(Kind K) {
      return (K == SectionKind::BSS ||
              K == SectionKind::SmallBSS);
    }
  }

  namespace SectionFlags {
    const unsigned Invalid    = -1U;
    const unsigned None       = 0;
    const unsigned Code       = 1 << 0;  ///< Section contains code
    const unsigned Writeable  = 1 << 1;  ///< Section is writeable
    const unsigned BSS        = 1 << 2;  ///< Section contains only zeroes
    const unsigned Mergeable  = 1 << 3;  ///< Section contains mergeable data
    const unsigned Strings    = 1 << 4;  ///< Section contains C-type strings
    const unsigned TLS        = 1 << 5;  ///< Section contains thread-local data
    const unsigned Debug      = 1 << 6;  ///< Section contains debug data
    const unsigned Linkonce   = 1 << 7;  ///< Section is linkonce
    const unsigned Small      = 1 << 8;  ///< Section is small
    const unsigned TypeFlags  = 0xFF;
    // Some gap for future flags
    const unsigned Named      = 1 << 23; ///< Section is named
    const unsigned EntitySize = 0xFF << 24; ///< Entity size for mergeable stuff

    static inline unsigned getEntitySize(unsigned Flags) {
      return (Flags >> 24) & 0xFF;
    }

    static inline unsigned setEntitySize(unsigned Flags, unsigned Size) {
      return ((Flags & ~EntitySize) | ((Size & 0xFF) << 24));
    }

    struct KeyInfo {
      static inline unsigned getEmptyKey() { return Invalid; }
      static inline unsigned getTombstoneKey() { return Invalid - 1; }
      static unsigned getHashValue(const unsigned &Key) { return Key; }
      static bool isEqual(unsigned LHS, unsigned RHS) { return LHS == RHS; }
      static bool isPod() { return true; }
    };

    typedef DenseMap<unsigned, std::string, KeyInfo> FlagsStringsMapType;
  }

  class TargetMachine;
  class CallInst;
  class GlobalValue;
  class Type;
  class Mangler;

  class Section {
    friend class TargetAsmInfo;
    friend class StringMapEntry<Section>;

    std::string Name;
    unsigned Flags;

    explicit Section(unsigned F = SectionFlags::Invalid):Flags(F) { }
  public:
    bool isNamed() const { return Flags & SectionFlags::Named; }
    unsigned getEntitySize() const { return (Flags >> 24) & 0xFF; }

    const std::string& getName() const { return Name; }
    unsigned getFlags() const { return Flags; }
  };

  /// TargetAsmInfo - This class is intended to be used as a base class for asm
  /// properties and features specific to the target.
  class TargetAsmInfo {
  private:
    mutable StringMap<Section> Sections;
    mutable SectionFlags::FlagsStringsMapType FlagsStrings;
    void fillDefaultValues();
  protected:
    /// TM - The current TargetMachine.
    const TargetMachine &TM;

    //===------------------------------------------------------------------===//
    // Properties to be set by the target writer, used to configure asm printer.
    //

    /// TextSection - Section directive for standard text.
    ///
    const Section *TextSection;           // Defaults to ".text".

    /// DataSection - Section directive for standard data.
    ///
    const Section *DataSection;           // Defaults to ".data".

    /// BSSSection - Section directive for uninitialized data.  Null if this
    /// target doesn't support a BSS section.
    ///
    const char *BSSSection;               // Default to ".bss".
    const Section *BSSSection_;

    /// ReadOnlySection - This is the directive that is emitted to switch to a
    /// read-only section for constant data (e.g. data declared const,
    /// jump tables).
    const Section *ReadOnlySection;       // Defaults to NULL

    /// SmallDataSection - This is the directive that is emitted to switch to a
    /// small data section.
    ///
    const Section *SmallDataSection;      // Defaults to NULL

    /// SmallBSSSection - This is the directive that is emitted to switch to a
    /// small bss section.
    ///
    const Section *SmallBSSSection;       // Defaults to NULL

    /// SmallRODataSection - This is the directive that is emitted to switch to 
    /// a small read-only data section.
    ///
    const Section *SmallRODataSection;    // Defaults to NULL

    /// TLSDataSection - Section directive for Thread Local data.
    ///
    const Section *TLSDataSection;        // Defaults to ".tdata".

    /// TLSBSSSection - Section directive for Thread Local uninitialized data.
    /// Null if this target doesn't support a BSS section.
    ///
    const Section *TLSBSSSection;         // Defaults to ".tbss".

    /// ZeroFillDirective - Directive for emitting a global to the ZeroFill
    /// section on this target.  Null if this target doesn't support zerofill.
    const char *ZeroFillDirective;        // Default is null.

    /// NonexecutableStackDirective - Directive for declaring to the
    /// linker and beyond that the emitted code does not require stack
    /// memory to be executable.
    const char *NonexecutableStackDirective; // Default is null.

    /// NeedsSet - True if target asm treats expressions in data directives
    /// as linktime-relocatable.  For assembly-time computation, we need to
    /// use a .set.  Thus:
    /// .set w, x-y
    /// .long w
    /// is computed at assembly time, while
    /// .long x-y
    /// is relocated if the relative locations of x and y change at linktime.
    /// We want both these things in different places.
    bool NeedsSet;                        // Defaults to false.
    
    /// MaxInstLength - This is the maximum possible length of an instruction,
    /// which is needed to compute the size of an inline asm.
    unsigned MaxInstLength;               // Defaults to 4.
    
    /// PCSymbol - The symbol used to represent the current PC.  Used in PC
    /// relative expressions.
    const char *PCSymbol;                 // Defaults to "$".

    /// SeparatorChar - This character, if specified, is used to separate
    /// instructions from each other when on the same line.  This is used to
    /// measure inline asm instructions.
    char SeparatorChar;                   // Defaults to ';'

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;            // Defaults to "#"

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;             // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .s file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;      // Defaults to "."
    
    /// LessPrivateGlobalPrefix - This prefix is used for symbols that should
    /// be passed through the assembler but be removed by the linker.  This
    /// is "l" on Darwin, currently used for some ObjC metadata.
    const char *LessPrivateGlobalPrefix;      // Defaults to ""
    
    /// JumpTableSpecialLabelPrefix - If not null, a extra (dead) label is
    /// emitted before jump tables with the specified prefix.
    const char *JumpTableSpecialLabelPrefix;  // Default to null.
    
    /// GlobalVarAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable (that isn't a function)
    ///
    const char *GlobalVarAddrPrefix;      // Defaults to ""
    const char *GlobalVarAddrSuffix;      // Defaults to ""

    /// FunctionAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable that points to a function.
    /// For example, this is used by the IA64 backend to materialize
    /// function descriptors, by decorating the ".data8" object with the
    /// @verbatim @fptr( ) @endverbatim
    /// link-relocation operator.
    ///
    const char *FunctionAddrPrefix;       // Defaults to ""
    const char *FunctionAddrSuffix;       // Defaults to ""

    /// PersonalityPrefix/Suffix - If these are nonempty, these strings will
    /// enclose any personality function in the common frame section.
    /// 
    const char *PersonalityPrefix;        // Defaults to ""
    const char *PersonalitySuffix;        // Defaults to ""

    /// NeedsIndirectEncoding - If set, we need to set the indirect encoding bit
    /// for EH in Dwarf.
    /// 
    bool NeedsIndirectEncoding;           // Defaults to false

    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assembly statement.
    const char *InlineAsmStart;           // Defaults to "#APP\n"
    const char *InlineAsmEnd;             // Defaults to "#NO_APP\n"

    /// AssemblerDialect - Which dialect of an assembler variant to use.
    unsigned AssemblerDialect;            // Defaults to 0

    /// StringConstantPrefix - Prefix for FEs to use when generating unnamed
    /// constant strings.  These names get run through the Mangler later; if
    /// you want the Mangler not to add the GlobalPrefix as well, 
    /// use '\1' as the first character.
    const char *StringConstantPrefix;     // Defaults to ".str"

    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;            // Defaults to "\t.zero\t"
    const char *ZeroDirectiveSuffix;      // Defaults to ""

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;           // Defaults to "\t.ascii\t"
    
    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;           // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;       // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;      // Defaults to "\t.short\t"
    const char *Data32bitsDirective;      // Defaults to "\t.long\t"
    const char *Data64bitsDirective;      // Defaults to "\t.quad\t"

    /// getASDirective - Targets can override it to provide different data
    /// directives for various sizes and non-default address spaces.
    virtual const char *getASDirective(unsigned size, 
                                       unsigned AS) const {
      assert (AS > 0 
              && "Dont know the directives for default addr space");
      return NULL;
    }

    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;           // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;              // Defaults to true

    /// TextAlignFillValue - If non-zero, this is used to fill the executable
    /// space created as the result of a alignment directive.
    unsigned TextAlignFillValue;

    //===--- Section Switching Directives ---------------------------------===//
    
    /// SwitchToSectionDirective - This is the directive used when we want to
    /// emit a global to an arbitrary section.  The section name is emited after
    /// this.
    const char *SwitchToSectionDirective; // Defaults to "\t.section\t"
    
    /// TextSectionStartSuffix - This is printed after each start of section
    /// directive for text sections.
    const char *TextSectionStartSuffix;   // Defaults to "".

    /// DataSectionStartSuffix - This is printed after each start of section
    /// directive for data sections.
    const char *DataSectionStartSuffix;   // Defaults to "".
    
    /// SectionEndDirectiveSuffix - If non-null, the asm printer will close each
    /// section with the section name and this suffix printed.
    const char *SectionEndDirectiveSuffix;// Defaults to null.
    
    /// ConstantPoolSection - This is the section that we SwitchToSection right
    /// before emitting the constant pool for a function.
    const char *ConstantPoolSection;      // Defaults to "\t.section .rodata"

    /// JumpTableDataSection - This is the section that we SwitchToSection right
    /// before emitting the jump tables for a function when the relocation model
    /// is not PIC.
    const char *JumpTableDataSection;     // Defaults to "\t.section .rodata"
    
    /// JumpTableDirective - if non-null, the directive to emit before a jump
    /// table.
    const char *JumpTableDirective;

    /// CStringSection - If not null, this allows for special handling of
    /// cstring constants (null terminated string that does not contain any
    /// other null bytes) on this target. This is commonly supported as
    /// ".cstring".
    const char *CStringSection;           // Defaults to NULL
    const Section *CStringSection_;

    /// StaticCtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static constructor list.
    /// Defaults to "\t.section .ctors,\"aw\",@progbits".
    const char *StaticCtorsSection;

    /// StaticDtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static destructor list.
    /// Defaults to "\t.section .dtors,\"aw\",@progbits".
    const char *StaticDtorsSection;

    //===--- Global Variable Emission Directives --------------------------===//
    
    /// GlobalDirective - This is the directive used to declare a global entity.
    ///
    const char *GlobalDirective;          // Defaults to NULL.
    
    /// SetDirective - This is the name of a directive that can be used to tell
    /// the assembler to set the value of a variable to some expression.
    const char *SetDirective;             // Defaults to null.
    
    /// LCOMMDirective - This is the name of a directive (if supported) that can
    /// be used to efficiently declare a local (internal) block of zero
    /// initialized data in the .bss/.data section.  The syntax expected is:
    /// @verbatim <LCOMMDirective> SYMBOLNAME LENGTHINBYTES, ALIGNMENT
    /// @endverbatim
    const char *LCOMMDirective;           // Defaults to null.
    
    const char *COMMDirective;            // Defaults to "\t.comm\t".

    /// COMMDirectiveTakesAlignment - True if COMMDirective take a third
    /// argument that specifies the alignment of the declaration.
    bool COMMDirectiveTakesAlignment;     // Defaults to true.
    
    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;      // Defaults to true.

    /// HasSingleParameterDotFile - True if the target has a single parameter
    /// .file directive, this is true for ELF targets.
    bool HasSingleParameterDotFile;      // Defaults to true.

    /// UsedDirective - This directive, if non-null, is used to declare a global
    /// as being used somehow that the assembler can't see.  This prevents dead
    /// code elimination on some targets.
    const char *UsedDirective;            // Defaults to null.

    /// WeakRefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak undefined symbol.
    const char *WeakRefDirective;         // Defaults to null.
    
    /// WeakDefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak defined symbol.
    const char *WeakDefDirective;         // Defaults to null.
    
    /// HiddenDirective - This directive, if non-null, is used to declare a
    /// global or function as having hidden visibility.
    const char *HiddenDirective;          // Defaults to "\t.hidden\t".

    /// ProtectedDirective - This directive, if non-null, is used to declare a
    /// global or function as having protected visibility.
    const char *ProtectedDirective;       // Defaults to "\t.protected\t".

    //===--- Dwarf Emission Directives -----------------------------------===//

    /// AbsoluteDebugSectionOffsets - True if we should emit abolute section
    /// offsets for debug information. Defaults to false.
    bool AbsoluteDebugSectionOffsets;

    /// AbsoluteEHSectionOffsets - True if we should emit abolute section
    /// offsets for EH information. Defaults to false.
    bool AbsoluteEHSectionOffsets;

    /// HasLEB128 - True if target asm supports leb128 directives.
    ///
    bool HasLEB128; // Defaults to false.

    /// hasDotLocAndDotFile - True if target asm supports .loc and .file
    /// directives for emitting debugging information.
    ///
    bool HasDotLocAndDotFile; // Defaults to false.

    /// SupportsDebugInformation - True if target supports emission of debugging
    /// information.
    bool SupportsDebugInformation;

    /// SupportsExceptionHandling - True if target supports
    /// exception handling.
    ///
    bool SupportsExceptionHandling; // Defaults to false.

    /// RequiresFrameSection - true if the Dwarf2 output needs a frame section
    ///
    bool DwarfRequiresFrameSection; // Defaults to true.

    /// SupportsMacInfo - true if the Dwarf output supports macro information
    ///
    bool SupportsMacInfoSection;            // Defaults to true

    /// NonLocalEHFrameLabel - If set, the EH_frame label needs to be non-local.
    ///
    bool NonLocalEHFrameLabel;              // Defaults to false.

    /// GlobalEHDirective - This is the directive used to make exception frame
    /// tables globally visible.
    ///
    const char *GlobalEHDirective;          // Defaults to NULL.

    /// SupportsWeakEmptyEHFrame - True if target assembler and linker will
    /// handle a weak_definition of constant 0 for an omitted EH frame.
    bool SupportsWeakOmittedEHFrame;  // Defaults to true.

    /// DwarfSectionOffsetDirective - Special section offset directive.
    const char* DwarfSectionOffsetDirective; // Defaults to NULL
    
    /// DwarfAbbrevSection - Section directive for Dwarf abbrev.
    ///
    const char *DwarfAbbrevSection; // Defaults to ".debug_abbrev".

    /// DwarfInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfInfoSection; // Defaults to ".debug_info".

    /// DwarfLineSection - Section directive for Dwarf info.
    ///
    const char *DwarfLineSection; // Defaults to ".debug_line".
    
    /// DwarfFrameSection - Section directive for Dwarf info.
    ///
    const char *DwarfFrameSection; // Defaults to ".debug_frame".
    
    /// DwarfPubNamesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubNamesSection; // Defaults to ".debug_pubnames".
    
    /// DwarfPubTypesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubTypesSection; // Defaults to ".debug_pubtypes".
    
    /// DwarfStrSection - Section directive for Dwarf info.
    ///
    const char *DwarfStrSection; // Defaults to ".debug_str".

    /// DwarfLocSection - Section directive for Dwarf info.
    ///
    const char *DwarfLocSection; // Defaults to ".debug_loc".

    /// DwarfARangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfARangesSection; // Defaults to ".debug_aranges".

    /// DwarfRangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfRangesSection; // Defaults to ".debug_ranges".

    /// DwarfMacInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfMacInfoSection; // Defaults to ".debug_macinfo".
    
    /// DwarfEHFrameSection - Section directive for Exception frames.
    ///
    const char *DwarfEHFrameSection; // Defaults to ".eh_frame".
    
    /// DwarfExceptionSection - Section directive for Exception table.
    ///
    const char *DwarfExceptionSection; // Defaults to ".gcc_except_table".

    //===--- CBE Asm Translation Table -----------------------------------===//

    const char *const *AsmTransCBE; // Defaults to empty

  public:
    explicit TargetAsmInfo(const TargetMachine &TM);
    virtual ~TargetAsmInfo();

    const Section* getNamedSection(const char *Name,
                                   unsigned Flags = SectionFlags::None,
                                   bool Override = false) const;
    const Section* getUnnamedSection(const char *Directive,
                                     unsigned Flags = SectionFlags::None,
                                     bool Override = false) const;

    /// Measure the specified inline asm to determine an approximation of its
    /// length.
    virtual unsigned getInlineAsmLength(const char *Str) const;

    /// ExpandInlineAsm - This hook allows the target to expand an inline asm
    /// call to be explicit llvm code if it wants to.  This is useful for
    /// turning simple inline asms into LLVM intrinsics, which gives the
    /// compiler more information about the behavior of the code.
    virtual bool ExpandInlineAsm(CallInst *CI) const {
      return false;
    }

    /// emitUsedDirectiveFor - This hook allows targets to selectively decide
    /// not to emit the UsedDirective for some symbols in llvm.used.
    virtual bool emitUsedDirectiveFor(const GlobalValue *GV,
                                      Mangler *Mang) const {
      return (GV!=0);
    }

    /// PreferredEHDataFormat - This hook allows the target to select data
    /// format used for encoding pointers in exception handling data. Reason is
    /// 0 for data, 1 for code labels, 2 for function pointers. Global is true
    /// if the symbol can be relocated.
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;

    /// SectionKindForGlobal - This hook allows the target to select proper
    /// section kind used for global emission.
    virtual SectionKind::Kind
    SectionKindForGlobal(const GlobalValue *GV) const;


    /// SectionFlagsForGlobal - This hook allows the target to select proper
    /// section flags either for given global or for section.
    virtual unsigned
    SectionFlagsForGlobal(const GlobalValue *GV = NULL,
                          const char* name = NULL) const;

    /// SectionForGlobal - This hooks returns proper section name for given
    /// global with all necessary flags and marks.
    virtual const Section* SectionForGlobal(const GlobalValue *GV) const;

    // Helper methods for SectionForGlobal
    virtual std::string UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const;

    const std::string& getSectionFlags(unsigned Flags) const;
    virtual std::string printSectionFlags(unsigned flags) const { return ""; }

    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV) const;

    virtual const Section* SelectSectionForMachineConst(const Type *Ty) const;

    /// getSLEB128Size - Compute the number of bytes required for a signed
    /// leb128 value.

    static unsigned getSLEB128Size(int Value);

    /// getULEB128Size - Compute the number of bytes required for an unsigned
    /// leb128 value.

    static unsigned getULEB128Size(unsigned Value);

    // Data directive accessors
    //
    const char *getData8bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data8bitsDirective : getASDirective(8, AS);
    }
    const char *getData16bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data16bitsDirective : getASDirective(16, AS);
    }
    const char *getData32bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data32bitsDirective : getASDirective(32, AS);
    }
    const char *getData64bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data64bitsDirective : getASDirective(64, AS);
    }


    // Accessors.
    //
    const Section *getTextSection() const {
      return TextSection;
    }
    const Section *getDataSection() const {
      return DataSection;
    }
    const char *getBSSSection() const {
      return BSSSection;
    }
    const Section *getBSSSection_() const {
      return BSSSection_;
    }
    const Section *getReadOnlySection() const {
      return ReadOnlySection;
    }
    const Section *getSmallDataSection() const {
      return SmallDataSection;
    }
    const Section *getSmallBSSSection() const {
      return SmallBSSSection;
    }
    const Section *getSmallRODataSection() const {
      return SmallRODataSection;
    }
    const Section *getTLSDataSection() const {
      return TLSDataSection;
    }
    const Section *getTLSBSSSection() const {
      return TLSBSSSection;
    }
    const char *getZeroFillDirective() const {
      return ZeroFillDirective;
    }
    const char *getNonexecutableStackDirective() const {
      return NonexecutableStackDirective;
    }
    bool needsSet() const {
      return NeedsSet;
    }
    const char *getPCSymbol() const {
      return PCSymbol;
    }
    char getSeparatorChar() const {
      return SeparatorChar;
    }
    const char *getCommentString() const {
      return CommentString;
    }
    const char *getGlobalPrefix() const {
      return GlobalPrefix;
    }
    const char *getPrivateGlobalPrefix() const {
      return PrivateGlobalPrefix;
    }
    /// EHGlobalPrefix - Prefix for EH_frame and the .eh symbols.
    /// This is normally PrivateGlobalPrefix, but some targets want
    /// these symbols to be visible.
    virtual const char *getEHGlobalPrefix() const {
      return PrivateGlobalPrefix;
    }
    const char *getLessPrivateGlobalPrefix() const {
      return LessPrivateGlobalPrefix;
    }
    const char *getJumpTableSpecialLabelPrefix() const {
      return JumpTableSpecialLabelPrefix;
    }
    const char *getGlobalVarAddrPrefix() const {
      return GlobalVarAddrPrefix;
    }
    const char *getGlobalVarAddrSuffix() const {
      return GlobalVarAddrSuffix;
    }
    const char *getFunctionAddrPrefix() const {
      return FunctionAddrPrefix;
    }
    const char *getFunctionAddrSuffix() const {
      return FunctionAddrSuffix;
    }
    const char *getPersonalityPrefix() const {
      return PersonalityPrefix;
    }
    const char *getPersonalitySuffix() const {
      return PersonalitySuffix;
    }
    bool getNeedsIndirectEncoding() const {
      return NeedsIndirectEncoding;
    }
    const char *getInlineAsmStart() const {
      return InlineAsmStart;
    }
    const char *getInlineAsmEnd() const {
      return InlineAsmEnd;
    }
    unsigned getAssemblerDialect() const {
      return AssemblerDialect;
    }
    const char *getStringConstantPrefix() const {
      return StringConstantPrefix;
    }
    const char *getZeroDirective() const {
      return ZeroDirective;
    }
    const char *getZeroDirectiveSuffix() const {
      return ZeroDirectiveSuffix;
    }
    const char *getAsciiDirective() const {
      return AsciiDirective;
    }
    const char *getAscizDirective() const {
      return AscizDirective;
    }
    const char *getJumpTableDirective() const {
      return JumpTableDirective;
    }
    const char *getAlignDirective() const {
      return AlignDirective;
    }
    bool getAlignmentIsInBytes() const {
      return AlignmentIsInBytes;
    }
    unsigned getTextAlignFillValue() const {
      return TextAlignFillValue;
    }
    const char *getSwitchToSectionDirective() const {
      return SwitchToSectionDirective;
    }
    const char *getTextSectionStartSuffix() const {
      return TextSectionStartSuffix;
    }
    const char *getDataSectionStartSuffix() const {
      return DataSectionStartSuffix;
    }
    const char *getSectionEndDirectiveSuffix() const {
      return SectionEndDirectiveSuffix;
    }
    const char *getConstantPoolSection() const {
      return ConstantPoolSection;
    }
    const char *getJumpTableDataSection() const {
      return JumpTableDataSection;
    }
    const char *getCStringSection() const {
      return CStringSection;
    }
    const Section *getCStringSection_() const {
      return CStringSection_;
    }
    const char *getStaticCtorsSection() const {
      return StaticCtorsSection;
    }
    const char *getStaticDtorsSection() const {
      return StaticDtorsSection;
    }
    const char *getGlobalDirective() const {
      return GlobalDirective;
    }
    const char *getSetDirective() const {
      return SetDirective;
    }
    const char *getLCOMMDirective() const {
      return LCOMMDirective;
    }
    const char *getCOMMDirective() const {
      return COMMDirective;
    }
    bool getCOMMDirectiveTakesAlignment() const {
      return COMMDirectiveTakesAlignment;
    }
    bool hasDotTypeDotSizeDirective() const {
      return HasDotTypeDotSizeDirective;
    }
    bool hasSingleParameterDotFile() const {
      return HasSingleParameterDotFile;
    }
    const char *getUsedDirective() const {
      return UsedDirective;
    }
    const char *getWeakRefDirective() const {
      return WeakRefDirective;
    }
    const char *getWeakDefDirective() const {
      return WeakDefDirective;
    }
    const char *getHiddenDirective() const {
      return HiddenDirective;
    }
    const char *getProtectedDirective() const {
      return ProtectedDirective;
    }
    bool isAbsoluteDebugSectionOffsets() const {
      return AbsoluteDebugSectionOffsets;
    }
    bool isAbsoluteEHSectionOffsets() const {
      return AbsoluteEHSectionOffsets;
    }
    bool hasLEB128() const {
      return HasLEB128;
    }
    bool hasDotLocAndDotFile() const {
      return HasDotLocAndDotFile;
    }
    bool doesSupportDebugInformation() const {
      return SupportsDebugInformation;
    }
    bool doesSupportExceptionHandling() const {
      return SupportsExceptionHandling;
    }
    bool doesDwarfRequireFrameSection() const {
      return DwarfRequiresFrameSection;
    }
    bool doesSupportMacInfoSection() const {
      return SupportsMacInfoSection;
    }
    bool doesRequireNonLocalEHFrameLabel() const {
      return NonLocalEHFrameLabel;
    }
    const char *getGlobalEHDirective() const {
      return GlobalEHDirective;
    }
    bool getSupportsWeakOmittedEHFrame() const {
      return SupportsWeakOmittedEHFrame;
    }
    const char *getDwarfSectionOffsetDirective() const {
      return DwarfSectionOffsetDirective;
    }
    const char *getDwarfAbbrevSection() const {
      return DwarfAbbrevSection;
    }
    const char *getDwarfInfoSection() const {
      return DwarfInfoSection;
    }
    const char *getDwarfLineSection() const {
      return DwarfLineSection;
    }
    const char *getDwarfFrameSection() const {
      return DwarfFrameSection;
    }
    const char *getDwarfPubNamesSection() const {
      return DwarfPubNamesSection;
    }
    const char *getDwarfPubTypesSection() const {
      return DwarfPubTypesSection;
    }
    const char *getDwarfStrSection() const {
      return DwarfStrSection;
    }
    const char *getDwarfLocSection() const {
      return DwarfLocSection;
    }
    const char *getDwarfARangesSection() const {
      return DwarfARangesSection;
    }
    const char *getDwarfRangesSection() const {
      return DwarfRangesSection;
    }
    const char *getDwarfMacInfoSection() const {
      return DwarfMacInfoSection;
    }
    const char *getDwarfEHFrameSection() const {
      return DwarfEHFrameSection;
    }
    const char *getDwarfExceptionSection() const {
      return DwarfExceptionSection;
    }
    const char *const *getAsmCBE() const {
      return AsmTransCBE;
    }
  };
}

#endif
