//===-- llvm/MC/MCAsmInfo.h - Asm info --------------------------*- C++ -*-===//
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

#ifndef LLVM_MC_MCASMINFO_H
#define LLVM_MC_MCASMINFO_H

#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MachineLocation.h"
#include <cassert>
#include <vector>

namespace llvm {
  class MCExpr;
  class MCSection;
  class MCStreamer;
  class MCSymbol;
  class MCContext;

  namespace ExceptionHandling {
    enum ExceptionsType { None, DwarfCFI, SjLj, ARM, Win64 };
  }

  namespace LCOMM {
    enum LCOMMType { NoAlignment, ByteAlignment, Log2Alignment };
  }

  /// MCAsmInfo - This class is intended to be used as a base class for asm
  /// properties and features specific to the target.
  class MCAsmInfo {
  protected:
    //===------------------------------------------------------------------===//
    // Properties to be set by the target writer, used to configure asm printer.
    //

    /// PointerSize - Pointer size in bytes.
    ///               Default is 4.
    unsigned PointerSize;

    /// CalleeSaveStackSlotSize - Size of the stack slot reserved for
    ///                           callee-saved registers, in bytes.
    ///                           Default is same as pointer size.
    unsigned CalleeSaveStackSlotSize;

    /// IsLittleEndian - True if target is little endian.
    ///                  Default is true.
    bool IsLittleEndian;

    /// StackGrowsUp - True if target stack grow up.
    ///                Default is false.
    bool StackGrowsUp;

    /// HasSubsectionsViaSymbols - True if this target has the MachO
    /// .subsections_via_symbols directive.
    bool HasSubsectionsViaSymbols;           // Default is false.

    /// HasMachoZeroFillDirective - True if this is a MachO target that supports
    /// the macho-specific .zerofill directive for emitting BSS Symbols.
    bool HasMachoZeroFillDirective;               // Default is false.

    /// HasMachoTBSSDirective - True if this is a MachO target that supports
    /// the macho-specific .tbss directive for emitting thread local BSS Symbols
    bool HasMachoTBSSDirective;                 // Default is false.

    /// HasStaticCtorDtorReferenceInStaticMode - True if the compiler should
    /// emit a ".reference .constructors_used" or ".reference .destructors_used"
    /// directive after the a static ctor/dtor list.  This directive is only
    /// emitted in Static relocation model.
    bool HasStaticCtorDtorReferenceInStaticMode;  // Default is false.

    /// LinkerRequiresNonEmptyDwarfLines - True if the linker has a bug and
    /// requires that the debug_line section be of a minimum size. In practice
    /// such a linker requires a non empty line sequence if a file is present.
    bool LinkerRequiresNonEmptyDwarfLines; // Default to false.

    /// MaxInstLength - This is the maximum possible length of an instruction,
    /// which is needed to compute the size of an inline asm.
    unsigned MaxInstLength;                  // Defaults to 4.

    /// MinInstAlignment - Every possible instruction length is a multiple of
    /// this value.  Factored out in .debug_frame and .debug_line.
    unsigned MinInstAlignment;                  // Defaults to 1.

    /// DollarIsPC - The '$' token, when not referencing an identifier or
    /// constant, refers to the current PC.
    bool DollarIsPC;                         // Defaults to false.

    /// SeparatorString - This string, if specified, is used to separate
    /// instructions from each other when on the same line.
    const char *SeparatorString;             // Defaults to ';'

    /// CommentColumn - This indicates the comment num (zero-based) at
    /// which asm comments should be printed.
    unsigned CommentColumn;                  // Defaults to 40

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;               // Defaults to "#"

    /// LabelSuffix - This is appended to emitted labels.
    const char *LabelSuffix;                 // Defaults to ":"

    /// LabelSuffix - This is appended to emitted labels.
    const char *DebugLabelSuffix;                 // Defaults to ":"

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;                // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .s file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;         // Defaults to "."

    /// LinkerPrivateGlobalPrefix - This prefix is used for symbols that should
    /// be passed through the assembler but be removed by the linker.  This
    /// is "l" on Darwin, currently used for some ObjC metadata.
    const char *LinkerPrivateGlobalPrefix;   // Defaults to ""

    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assembly statement.
    const char *InlineAsmStart;              // Defaults to "#APP\n"
    const char *InlineAsmEnd;                // Defaults to "#NO_APP\n"

    /// Code16Directive, Code32Directive, Code64Directive - These are assembly
    /// directives that tells the assembler to interpret the following
    /// instructions differently.
    const char *Code16Directive;             // Defaults to ".code16"
    const char *Code32Directive;             // Defaults to ".code32"
    const char *Code64Directive;             // Defaults to ".code64"

    /// AssemblerDialect - Which dialect of an assembler variant to use.
    unsigned AssemblerDialect;               // Defaults to 0

    /// \brief This is true if the assembler allows @ characters in symbol
    /// names. Defaults to false.
    bool AllowAtInName;

    /// UseDataRegionDirectives - This is true if data region markers should
    /// be printed as ".data_region/.end_data_region" directives. If false,
    /// use "$d/$a" labels instead.
    bool UseDataRegionDirectives;

    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;               // Defaults to "\t.zero\t"

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;              // Defaults to "\t.ascii\t"

    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;              // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;          // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;         // Defaults to "\t.short\t"
    const char *Data32bitsDirective;         // Defaults to "\t.long\t"
    const char *Data64bitsDirective;         // Defaults to "\t.quad\t"

    /// GPRel64Directive - if non-null, a directive that is used to emit a word
    /// which should be relocated as a 64-bit GP-relative offset, e.g. .gpdword
    /// on Mips.
    const char *GPRel64Directive;            // Defaults to NULL.

    /// GPRel32Directive - if non-null, a directive that is used to emit a word
    /// which should be relocated as a 32-bit GP-relative offset, e.g. .gpword
    /// on Mips or .gprel32 on Alpha.
    const char *GPRel32Directive;            // Defaults to NULL.

    /// SunStyleELFSectionSwitchSyntax - This is true if this target uses "Sun
    /// Style" syntax for section switching ("#alloc,#write" etc) instead of the
    /// normal ELF syntax (,"a,w") in .section directives.
    bool SunStyleELFSectionSwitchSyntax;     // Defaults to false.

    /// UsesELFSectionDirectiveForBSS - This is true if this target uses ELF
    /// '.section' directive before the '.bss' one. It's used for PPC/Linux
    /// which doesn't support the '.bss' directive only.
    bool UsesELFSectionDirectiveForBSS;      // Defaults to false.

    /// HasMicrosoftFastStdCallMangling - True if this target uses microsoft
    /// style mangling for functions with X86_StdCall/X86_FastCall calling
    /// convention.
    bool HasMicrosoftFastStdCallMangling;    // Defaults to false.

    bool NeedsDwarfSectionOffsetDirective;

    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;              // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;                 // Defaults to true

    /// TextAlignFillValue - If non-zero, this is used to fill the executable
    /// space created as the result of a alignment directive.
    unsigned TextAlignFillValue;             // Defaults to 0

    //===--- Global Variable Emission Directives --------------------------===//

    /// GlobalDirective - This is the directive used to declare a global entity.
    ///
    const char *GlobalDirective;             // Defaults to NULL.

    /// HasSetDirective - True if the assembler supports the .set directive.
    bool HasSetDirective;                    // Defaults to true.

    /// HasAggressiveSymbolFolding - False if the assembler requires that we use
    /// Lc = a - b
    /// .long Lc
    /// instead of
    /// .long a - b
    bool HasAggressiveSymbolFolding;           // Defaults to true.

    /// COMMDirectiveAlignmentIsInBytes - True is .comm's and .lcomms optional
    /// alignment is to be specified in bytes instead of log2(n).
    bool COMMDirectiveAlignmentIsInBytes;    // Defaults to true;

    /// LCOMMDirectiveAlignment - Describes if the .lcomm directive for the
    /// target supports an alignment argument and how it is interpreted.
    LCOMM::LCOMMType LCOMMDirectiveAlignmentType; // Defaults to NoAlignment.

    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;         // Defaults to true.

    /// HasSingleParameterDotFile - True if the target has a single parameter
    /// .file directive, this is true for ELF targets.
    bool HasSingleParameterDotFile;          // Defaults to true.

    /// hasIdentDirective - True if the target has a .ident directive, this is
    /// true for ELF targets.
    bool HasIdentDirective;                  // Defaults to false.

    /// HasNoDeadStrip - True if this target supports the MachO .no_dead_strip
    /// directive.
    bool HasNoDeadStrip;                     // Defaults to false.

    /// WeakRefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak undefined symbol.
    const char *WeakRefDirective;            // Defaults to NULL.

    /// True if we have a directive to declare a global as being a weak
    /// defined symbol.
    bool HasWeakDefDirective;                // Defaults to false.

    /// True if we have a directive to declare a global as being a weak
    /// defined symbol that can be hidden (unexported).
    bool HasWeakDefCanBeHiddenDirective;     // Defaults to false.

    /// True if we have a .linkonce directive.  This is used on cygwin/mingw.
    bool HasLinkOnceDirective;               // Defaults to false.

    /// HiddenVisibilityAttr - This attribute, if not MCSA_Invalid, is used to
    /// declare a symbol as having hidden visibility.
    MCSymbolAttr HiddenVisibilityAttr;       // Defaults to MCSA_Hidden.

    /// HiddenDeclarationVisibilityAttr - This attribute, if not MCSA_Invalid,
    /// is used to declare an undefined symbol as having hidden visibility.
    MCSymbolAttr HiddenDeclarationVisibilityAttr;   // Defaults to MCSA_Hidden.


    /// ProtectedVisibilityAttr - This attribute, if not MCSA_Invalid, is used
    /// to declare a symbol as having protected visibility.
    MCSymbolAttr ProtectedVisibilityAttr;    // Defaults to MCSA_Protected

    //===--- Dwarf Emission Directives -----------------------------------===//

    /// HasLEB128 - True if target asm supports leb128 directives.
    bool HasLEB128;                          // Defaults to false.

    /// SupportsDebugInformation - True if target supports emission of debugging
    /// information.
    bool SupportsDebugInformation;           // Defaults to false.

    /// SupportsExceptionHandling - True if target supports exception handling.
    ExceptionHandling::ExceptionsType ExceptionsType; // Defaults to None

    /// DwarfUsesRelocationsAcrossSections - True if Dwarf2 output generally
    /// uses relocations for references to other .debug_* sections.
    bool DwarfUsesRelocationsAcrossSections;

    /// DwarfFDESymbolsUseAbsDiff - true if DWARF FDE symbol reference
    /// relocations should be replaced by an absolute difference.
    bool DwarfFDESymbolsUseAbsDiff;

    /// DwarfRegNumForCFI - True if dwarf register numbers are printed
    /// instead of symbolic register names in .cfi_* directives.
    bool DwarfRegNumForCFI;  // Defaults to false;

    //===--- Prologue State ----------------------------------------------===//

    std::vector<MCCFIInstruction> InitialFrameState;

  public:
    explicit MCAsmInfo();
    virtual ~MCAsmInfo();

    // FIXME: move these methods to DwarfPrinter when the JIT stops using them.
    static unsigned getSLEB128Size(int64_t Value);
    static unsigned getULEB128Size(uint64_t Value);

    /// getPointerSize - Get the pointer size in bytes.
    unsigned getPointerSize() const {
      return PointerSize;
    }

    /// getCalleeSaveStackSlotSize - Get the callee-saved register stack slot
    /// size in bytes.
    unsigned getCalleeSaveStackSlotSize() const {
      return CalleeSaveStackSlotSize;
    }

    /// isLittleEndian - True if the target is little endian.
    bool isLittleEndian() const {
      return IsLittleEndian;
    }

    /// isStackGrowthDirectionUp - True if target stack grow up.
    bool isStackGrowthDirectionUp() const {
      return StackGrowsUp;
    }

    bool hasSubsectionsViaSymbols() const { return HasSubsectionsViaSymbols; }

    // Data directive accessors.
    //
    const char *getData8bitsDirective() const {
      return Data8bitsDirective;
    }
    const char *getData16bitsDirective() const {
      return Data16bitsDirective;
    }
    const char *getData32bitsDirective() const {
      return Data32bitsDirective;
    }
    const char *getData64bitsDirective() const {
      return Data64bitsDirective;
    }
    const char *getGPRel64Directive() const { return GPRel64Directive; }
    const char *getGPRel32Directive() const { return GPRel32Directive; }

    /// getNonexecutableStackSection - Targets can implement this method to
    /// specify a section to switch to if the translation unit doesn't have any
    /// trampolines that require an executable stack.
    virtual const MCSection *getNonexecutableStackSection(MCContext &Ctx) const{
      return 0;
    }

    virtual const MCExpr *
    getExprForPersonalitySymbol(const MCSymbol *Sym,
                                unsigned Encoding,
                                MCStreamer &Streamer) const;

    const MCExpr *
    getExprForFDESymbol(const MCSymbol *Sym,
                        unsigned Encoding,
                        MCStreamer &Streamer) const;

    bool usesSunStyleELFSectionSwitchSyntax() const {
      return SunStyleELFSectionSwitchSyntax;
    }

    bool usesELFSectionDirectiveForBSS() const {
      return UsesELFSectionDirectiveForBSS;
    }

    bool hasMicrosoftFastStdCallMangling() const {
      return HasMicrosoftFastStdCallMangling;
    }

    bool needsDwarfSectionOffsetDirective() const {
      return NeedsDwarfSectionOffsetDirective;
    }

    // Accessors.
    //
    bool hasMachoZeroFillDirective() const { return HasMachoZeroFillDirective; }
    bool hasMachoTBSSDirective() const { return HasMachoTBSSDirective; }
    bool hasStaticCtorDtorReferenceInStaticMode() const {
      return HasStaticCtorDtorReferenceInStaticMode;
    }
    bool getLinkerRequiresNonEmptyDwarfLines() const {
      return LinkerRequiresNonEmptyDwarfLines;
    }
    unsigned getMaxInstLength() const {
      return MaxInstLength;
    }
    unsigned getMinInstAlignment() const {
      return MinInstAlignment;
    }
    bool getDollarIsPC() const {
      return DollarIsPC;
    }
    const char *getSeparatorString() const {
      return SeparatorString;
    }
    unsigned getCommentColumn() const {
      return CommentColumn;
    }
    const char *getCommentString() const {
      return CommentString;
    }
    const char *getLabelSuffix() const {
      return LabelSuffix;
    }

    const char *getDebugLabelSuffix() const {
      return DebugLabelSuffix;
    }

    const char *getGlobalPrefix() const {
      return GlobalPrefix;
    }
    const char *getPrivateGlobalPrefix() const {
      return PrivateGlobalPrefix;
    }
    const char *getLinkerPrivateGlobalPrefix() const {
      return LinkerPrivateGlobalPrefix;
    }
    const char *getInlineAsmStart() const {
      return InlineAsmStart;
    }
    const char *getInlineAsmEnd() const {
      return InlineAsmEnd;
    }
    const char *getCode16Directive() const {
      return Code16Directive;
    }
    const char *getCode32Directive() const {
      return Code32Directive;
    }
    const char *getCode64Directive() const {
      return Code64Directive;
    }
    unsigned getAssemblerDialect() const {
      return AssemblerDialect;
    }
    bool doesAllowAtInName() const {
      return AllowAtInName;
    }
    bool doesSupportDataRegionDirectives() const {
      return UseDataRegionDirectives;
    }
    const char *getZeroDirective() const {
      return ZeroDirective;
    }
    const char *getAsciiDirective() const {
      return AsciiDirective;
    }
    const char *getAscizDirective() const {
      return AscizDirective;
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
    const char *getGlobalDirective() const {
      return GlobalDirective;
    }
    bool hasSetDirective() const { return HasSetDirective; }
    bool hasAggressiveSymbolFolding() const {
      return HasAggressiveSymbolFolding;
    }
    bool getCOMMDirectiveAlignmentIsInBytes() const {
      return COMMDirectiveAlignmentIsInBytes;
    }
    LCOMM::LCOMMType getLCOMMDirectiveAlignmentType() const {
      return LCOMMDirectiveAlignmentType;
    }
    bool hasDotTypeDotSizeDirective() const {return HasDotTypeDotSizeDirective;}
    bool hasSingleParameterDotFile() const { return HasSingleParameterDotFile; }
    bool hasIdentDirective() const { return HasIdentDirective; }
    bool hasNoDeadStrip() const { return HasNoDeadStrip; }
    const char *getWeakRefDirective() const { return WeakRefDirective; }
    bool hasWeakDefDirective() const { return HasWeakDefDirective; }
    bool hasWeakDefCanBeHiddenDirective() const {
      return HasWeakDefCanBeHiddenDirective;
    }
    bool hasLinkOnceDirective() const { return HasLinkOnceDirective; }

    MCSymbolAttr getHiddenVisibilityAttr() const { return HiddenVisibilityAttr;}
    MCSymbolAttr getHiddenDeclarationVisibilityAttr() const {
      return HiddenDeclarationVisibilityAttr;
    }
    MCSymbolAttr getProtectedVisibilityAttr() const {
      return ProtectedVisibilityAttr;
    }
    bool hasLEB128() const {
      return HasLEB128;
    }
    bool doesSupportDebugInformation() const {
      return SupportsDebugInformation;
    }
    bool doesSupportExceptionHandling() const {
      return ExceptionsType != ExceptionHandling::None;
    }
    ExceptionHandling::ExceptionsType getExceptionHandlingType() const {
      return ExceptionsType;
    }
    bool isExceptionHandlingDwarf() const {
      return
        (ExceptionsType == ExceptionHandling::DwarfCFI ||
         ExceptionsType == ExceptionHandling::ARM ||
         ExceptionsType == ExceptionHandling::Win64);
    }
    bool doesDwarfUseRelocationsAcrossSections() const {
      return DwarfUsesRelocationsAcrossSections;
    }
    bool doDwarfFDESymbolsUseAbsDiff() const {
      return DwarfFDESymbolsUseAbsDiff;
    }
    bool useDwarfRegNumForCFI() const {
      return DwarfRegNumForCFI;
    }

    void addInitialFrameState(const MCCFIInstruction &Inst) {
      InitialFrameState.push_back(Inst);
    }

    const std::vector<MCCFIInstruction> &getInitialFrameState() const {
      return InitialFrameState;
    }
  };
}

#endif
