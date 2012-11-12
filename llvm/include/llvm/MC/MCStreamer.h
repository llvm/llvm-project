//===- MCStreamer.h - High-level Streaming Machine Code Output --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCStreamer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSTREAMER_H
#define LLVM_MC_MCSTREAMER_H

#include "llvm/Support/DataTypes.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  class MCAsmBackend;
  class MCCodeEmitter;
  class MCContext;
  class MCExpr;
  class MCInst;
  class MCInstPrinter;
  class MCSection;
  class MCSymbol;
  class StringRef;
  class Twine;
  class raw_ostream;
  class formatted_raw_ostream;

  /// MCStreamer - Streaming machine code generation interface.  This interface
  /// is intended to provide a programatic interface that is very similar to the
  /// level that an assembler .s file provides.  It has callbacks to emit bytes,
  /// handle directives, etc.  The implementation of this interface retains
  /// state to know what the current section is etc.
  ///
  /// There are multiple implementations of this interface: one for writing out
  /// a .s file, and implementations that write out .o files of various formats.
  ///
  class MCStreamer {
    MCContext &Context;

    MCStreamer(const MCStreamer&) LLVM_DELETED_FUNCTION;
    MCStreamer &operator=(const MCStreamer&) LLVM_DELETED_FUNCTION;

    bool EmitEHFrame;
    bool EmitDebugFrame;

    std::vector<MCDwarfFrameInfo> FrameInfos;
    MCDwarfFrameInfo *getCurrentFrameInfo();
    void EnsureValidFrame();

    std::vector<MCWin64EHUnwindInfo *> W64UnwindInfos;
    MCWin64EHUnwindInfo *CurrentW64UnwindInfo;
    void setCurrentW64UnwindInfo(MCWin64EHUnwindInfo *Frame);
    void EnsureValidW64UnwindInfo();

    MCSymbol* LastSymbol;

    /// SectionStack - This is stack of current and previous section
    /// values saved by PushSection.
    SmallVector<std::pair<const MCSection *,
                const MCSection *>, 4> SectionStack;

  protected:
    MCStreamer(MCContext &Ctx);

    const MCExpr *BuildSymbolDiff(MCContext &Context, const MCSymbol *A,
                                  const MCSymbol *B);

    const MCExpr *ForceExpAbs(const MCExpr* Expr);

    void RecordProcStart(MCDwarfFrameInfo &Frame);
    virtual void EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame);
    void RecordProcEnd(MCDwarfFrameInfo &Frame);
    virtual void EmitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame);
    void EmitFrames(bool usingCFI);

    MCWin64EHUnwindInfo *getCurrentW64UnwindInfo(){return CurrentW64UnwindInfo;}
    void EmitW64Tables();

  public:
    virtual ~MCStreamer();

    MCContext &getContext() const { return Context; }

    unsigned getNumFrameInfos() {
      return FrameInfos.size();
    }

    const MCDwarfFrameInfo &getFrameInfo(unsigned i) {
      return FrameInfos[i];
    }

    ArrayRef<MCDwarfFrameInfo> getFrameInfos() {
      return FrameInfos;
    }

    unsigned getNumW64UnwindInfos() {
      return W64UnwindInfos.size();
    }

    MCWin64EHUnwindInfo &getW64UnwindInfo(unsigned i) {
      return *W64UnwindInfos[i];
    }

    /// @name Assembly File Formatting.
    /// @{

    /// isVerboseAsm - Return true if this streamer supports verbose assembly
    /// and if it is enabled.
    virtual bool isVerboseAsm() const { return false; }

    /// hasRawTextSupport - Return true if this asm streamer supports emitting
    /// unformatted text to the .s file with EmitRawText.
    virtual bool hasRawTextSupport() const { return false; }

    /// AddComment - Add a comment that can be emitted to the generated .s
    /// file if applicable as a QoI issue to make the output of the compiler
    /// more readable.  This only affects the MCAsmStreamer, and only when
    /// verbose assembly output is enabled.
    ///
    /// If the comment includes embedded \n's, they will each get the comment
    /// prefix as appropriate.  The added comment should not end with a \n.
    virtual void AddComment(const Twine &T) {}

    /// GetCommentOS - Return a raw_ostream that comments can be written to.
    /// Unlike AddComment, you are required to terminate comments with \n if you
    /// use this method.
    virtual raw_ostream &GetCommentOS();

    /// AddBlankLine - Emit a blank line to a .s file to pretty it up.
    virtual void AddBlankLine() {}

    /// @}

    /// @name Symbol & Section Management
    /// @{

    /// getCurrentSection - Return the current section that the streamer is
    /// emitting code to.
    const MCSection *getCurrentSection() const {
      if (!SectionStack.empty())
        return SectionStack.back().first;
      return NULL;
    }

    /// getPreviousSection - Return the previous section that the streamer is
    /// emitting code to.
    const MCSection *getPreviousSection() const {
      if (!SectionStack.empty())
        return SectionStack.back().second;
      return NULL;
    }

    /// ChangeSection - Update streamer for a new active section.
    ///
    /// This is called by PopSection and SwitchSection, if the current
    /// section changes.
    virtual void ChangeSection(const MCSection *) = 0;

    /// pushSection - Save the current and previous section on the
    /// section stack.
    void PushSection() {
      SectionStack.push_back(std::make_pair(getCurrentSection(),
                                            getPreviousSection()));
    }

    /// popSection - Restore the current and previous section from
    /// the section stack.  Calls ChangeSection as needed.
    ///
    /// Returns false if the stack was empty.
    bool PopSection() {
      if (SectionStack.size() <= 1)
        return false;
      const MCSection *oldSection = SectionStack.pop_back_val().first;
      const MCSection *curSection = SectionStack.back().first;

      if (oldSection != curSection)
        ChangeSection(curSection);
      return true;
    }

    /// SwitchSection - Set the current section where code is being emitted to
    /// @p Section.  This is required to update CurSection.
    ///
    /// This corresponds to assembler directives like .section, .text, etc.
    void SwitchSection(const MCSection *Section) {
      assert(Section && "Cannot switch to a null section!");
      const MCSection *curSection = SectionStack.back().first;
      SectionStack.back().second = curSection;
      if (Section != curSection) {
        SectionStack.back().first = Section;
        ChangeSection(Section);
      }
    }

    /// SwitchSectionNoChange - Set the current section where code is being
    /// emitted to @p Section.  This is required to update CurSection. This
    /// version does not call ChangeSection.
    void SwitchSectionNoChange(const MCSection *Section) {
      assert(Section && "Cannot switch to a null section!");
      const MCSection *curSection = SectionStack.back().first;
      SectionStack.back().second = curSection;
      if (Section != curSection)
        SectionStack.back().first = Section;
    }

    /// InitSections - Create the default sections and set the initial one.
    virtual void InitSections() = 0;

    /// EmitLabel - Emit a label for @p Symbol into the current section.
    ///
    /// This corresponds to an assembler statement such as:
    ///   foo:
    ///
    /// @param Symbol - The symbol to emit. A given symbol should only be
    /// emitted as a label once, and symbols emitted as a label should never be
    /// used in an assignment.
    virtual void EmitLabel(MCSymbol *Symbol);

    virtual void EmitEHSymAttributes(const MCSymbol *Symbol,
                                     MCSymbol *EHSymbol);

    /// EmitAssemblerFlag - Note in the output the specified @p Flag.
    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) = 0;

    /// EmitDataRegion - Note in the output the specified region @p Kind.
    virtual void EmitDataRegion(MCDataRegionType Kind) {}

    /// EmitThumbFunc - Note in the output that the specified @p Func is
    /// a Thumb mode function (ARM target only).
    virtual void EmitThumbFunc(MCSymbol *Func) = 0;

    /// EmitAssignment - Emit an assignment of @p Value to @p Symbol.
    ///
    /// This corresponds to an assembler statement such as:
    ///  symbol = value
    ///
    /// The assignment generates no code, but has the side effect of binding the
    /// value in the current context. For the assembly streamer, this prints the
    /// binding into the .s file.
    ///
    /// @param Symbol - The symbol being assigned to.
    /// @param Value - The value for the symbol.
    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) = 0;

    /// EmitWeakReference - Emit an weak reference from @p Alias to @p Symbol.
    ///
    /// This corresponds to an assembler statement such as:
    ///  .weakref alias, symbol
    ///
    /// @param Alias - The alias that is being created.
    /// @param Symbol - The symbol being aliased.
    virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) = 0;

    /// EmitSymbolAttribute - Add the given @p Attribute to @p Symbol.
    virtual void EmitSymbolAttribute(MCSymbol *Symbol,
                                     MCSymbolAttr Attribute) = 0;

    /// EmitSymbolDesc - Set the @p DescValue for the @p Symbol.
    ///
    /// @param Symbol - The symbol to have its n_desc field set.
    /// @param DescValue - The value to set into the n_desc field.
    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) = 0;

    /// BeginCOFFSymbolDef - Start emitting COFF symbol definition
    ///
    /// @param Symbol - The symbol to have its External & Type fields set.
    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) = 0;

    /// EmitCOFFSymbolStorageClass - Emit the storage class of the symbol.
    ///
    /// @param StorageClass - The storage class the symbol should have.
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) = 0;

    /// EmitCOFFSymbolType - Emit the type of the symbol.
    ///
    /// @param Type - A COFF type identifier (see COFF::SymbolType in X86COFF.h)
    virtual void EmitCOFFSymbolType(int Type) = 0;

    /// EndCOFFSymbolDef - Marks the end of the symbol definition.
    virtual void EndCOFFSymbolDef() = 0;

    /// EmitCOFFSecRel32 - Emits a COFF section relative relocation.
    ///
    /// @param Symbol - Symbol the section relative realocation should point to.
    virtual void EmitCOFFSecRel32(MCSymbol const *Symbol);

    /// EmitELFSize - Emit an ELF .size directive.
    ///
    /// This corresponds to an assembler statement such as:
    ///  .size symbol, expression
    ///
    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) = 0;

    /// EmitCommonSymbol - Emit a common symbol.
    ///
    /// @param Symbol - The common symbol to emit.
    /// @param Size - The size of the common symbol.
    /// @param ByteAlignment - The alignment of the symbol if
    /// non-zero. This must be a power of 2.
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) = 0;

    /// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
    ///
    /// @param Symbol - The common symbol to emit.
    /// @param Size - The size of the common symbol.
    /// @param ByteAlignment - The alignment of the common symbol in bytes.
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) = 0;

    /// EmitZerofill - Emit the zerofill section and an optional symbol.
    ///
    /// @param Section - The zerofill section to create and or to put the symbol
    /// @param Symbol - The zerofill symbol to emit, if non-NULL.
    /// @param Size - The size of the zerofill symbol.
    /// @param ByteAlignment - The alignment of the zerofill symbol if
    /// non-zero. This must be a power of 2 on some targets.
    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                              uint64_t Size = 0,unsigned ByteAlignment = 0) = 0;

    /// EmitTBSSSymbol - Emit a thread local bss (.tbss) symbol.
    ///
    /// @param Section - The thread local common section.
    /// @param Symbol - The thread local common symbol to emit.
    /// @param Size - The size of the symbol.
    /// @param ByteAlignment - The alignment of the thread local common symbol
    /// if non-zero.  This must be a power of 2 on some targets.
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment = 0) = 0;

    /// @}
    /// @name Generating Data
    /// @{

    /// EmitBytes - Emit the bytes in \p Data into the output.
    ///
    /// This is used to implement assembler directives such as .byte, .ascii,
    /// etc.
    virtual void EmitBytes(StringRef Data, unsigned AddrSpace) = 0;

    /// EmitValue - Emit the expression @p Value into the output as a native
    /// integer of the given @p Size bytes.
    ///
    /// This is used to implement assembler directives such as .word, .quad,
    /// etc.
    ///
    /// @param Value - The value to emit.
    /// @param Size - The size of the integer (in bytes) to emit. This must
    /// match a native machine width.
    virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                               unsigned AddrSpace) = 0;

    void EmitValue(const MCExpr *Value, unsigned Size, unsigned AddrSpace = 0);

    /// EmitIntValue - Special case of EmitValue that avoids the client having
    /// to pass in a MCExpr for constant integers.
    virtual void EmitIntValue(uint64_t Value, unsigned Size,
                              unsigned AddrSpace = 0);

    /// EmitAbsValue - Emit the Value, but try to avoid relocations. On MachO
    /// this is done by producing
    /// foo = value
    /// .long foo
    void EmitAbsValue(const MCExpr *Value, unsigned Size,
                      unsigned AddrSpace = 0);

    virtual void EmitULEB128Value(const MCExpr *Value) = 0;

    virtual void EmitSLEB128Value(const MCExpr *Value) = 0;

    /// EmitULEB128Value - Special case of EmitULEB128Value that avoids the
    /// client having to pass in a MCExpr for constant integers.
    void EmitULEB128IntValue(uint64_t Value, unsigned AddrSpace = 0,
                             unsigned Padding = 0);

    /// EmitSLEB128Value - Special case of EmitSLEB128Value that avoids the
    /// client having to pass in a MCExpr for constant integers.
    void EmitSLEB128IntValue(int64_t Value, unsigned AddrSpace = 0);

    /// EmitSymbolValue - Special case of EmitValue that avoids the client
    /// having to pass in a MCExpr for MCSymbols.
    void EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                         unsigned AddrSpace = 0);

    /// EmitGPRel64Value - Emit the expression @p Value into the output as a
    /// gprel64 (64-bit GP relative) value.
    ///
    /// This is used to implement assembler directives such as .gpdword on
    /// targets that support them.
    virtual void EmitGPRel64Value(const MCExpr *Value);

    /// EmitGPRel32Value - Emit the expression @p Value into the output as a
    /// gprel32 (32-bit GP relative) value.
    ///
    /// This is used to implement assembler directives such as .gprel32 on
    /// targets that support them.
    virtual void EmitGPRel32Value(const MCExpr *Value);

    /// EmitFill - Emit NumBytes bytes worth of the value specified by
    /// FillValue.  This implements directives such as '.space'.
    virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                          unsigned AddrSpace);

    /// EmitZeros - Emit NumBytes worth of zeros.  This is a convenience
    /// function that just wraps EmitFill.
    void EmitZeros(uint64_t NumBytes, unsigned AddrSpace) {
      EmitFill(NumBytes, 0, AddrSpace);
    }


    /// EmitValueToAlignment - Emit some number of copies of @p Value until
    /// the byte alignment @p ByteAlignment is reached.
    ///
    /// If the number of bytes need to emit for the alignment is not a multiple
    /// of @p ValueSize, then the contents of the emitted fill bytes is
    /// undefined.
    ///
    /// This used to implement the .align assembler directive.
    ///
    /// @param ByteAlignment - The alignment to reach. This must be a power of
    /// two on some targets.
    /// @param Value - The value to use when filling bytes.
    /// @param ValueSize - The size of the integer (in bytes) to emit for
    /// @p Value. This must match a native machine width.
    /// @param MaxBytesToEmit - The maximum numbers of bytes to emit, or 0. If
    /// the alignment cannot be reached in this many bytes, no bytes are
    /// emitted.
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0) = 0;

    /// EmitCodeAlignment - Emit nops until the byte alignment @p ByteAlignment
    /// is reached.
    ///
    /// This used to align code where the alignment bytes may be executed.  This
    /// can emit different bytes for different sizes to optimize execution.
    ///
    /// @param ByteAlignment - The alignment to reach. This must be a power of
    /// two on some targets.
    /// @param MaxBytesToEmit - The maximum numbers of bytes to emit, or 0. If
    /// the alignment cannot be reached in this many bytes, no bytes are
    /// emitted.
    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit = 0) = 0;

    /// EmitValueToOffset - Emit some number of copies of @p Value until the
    /// byte offset @p Offset is reached.
    ///
    /// This is used to implement assembler directives such as .org.
    ///
    /// @param Offset - The offset to reach. This may be an expression, but the
    /// expression must be associated with the current section.
    /// @param Value - The value to use when filling bytes.
    /// @return false on success, true if the offset was invalid.
    virtual bool EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value = 0) = 0;

    /// @}

    /// EmitFileDirective - Switch to a new logical file.  This is used to
    /// implement the '.file "foo.c"' assembler directive.
    virtual void EmitFileDirective(StringRef Filename) = 0;

    /// EmitDwarfFileDirective - Associate a filename with a specified logical
    /// file number.  This implements the DWARF2 '.file 4 "foo.c"' assembler
    /// directive.
    virtual bool EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                        StringRef Filename);

    /// EmitDwarfLocDirective - This implements the DWARF2
    // '.loc fileno lineno ...' assembler directive.
    virtual void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa,
                                       unsigned Discriminator,
                                       StringRef FileName);

    virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                          const MCSymbol *LastLabel,
                                          const MCSymbol *Label,
                                          unsigned PointerSize) = 0;

    virtual void EmitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                           const MCSymbol *Label) {
    }

    void EmitDwarfSetLineAddr(int64_t LineDelta, const MCSymbol *Label,
                              int PointerSize);

    virtual void EmitCompactUnwindEncoding(uint32_t CompactUnwindEncoding);
    virtual void EmitCFISections(bool EH, bool Debug);
    void EmitCFIStartProc();
    void EmitCFIEndProc();
    virtual void EmitCFIDefCfa(int64_t Register, int64_t Offset);
    virtual void EmitCFIDefCfaOffset(int64_t Offset);
    virtual void EmitCFIDefCfaRegister(int64_t Register);
    virtual void EmitCFIOffset(int64_t Register, int64_t Offset);
    virtual void EmitCFIPersonality(const MCSymbol *Sym, unsigned Encoding);
    virtual void EmitCFILsda(const MCSymbol *Sym, unsigned Encoding);
    virtual void EmitCFIRememberState();
    virtual void EmitCFIRestoreState();
    virtual void EmitCFISameValue(int64_t Register);
    virtual void EmitCFIRestore(int64_t Register);
    virtual void EmitCFIRelOffset(int64_t Register, int64_t Offset);
    virtual void EmitCFIAdjustCfaOffset(int64_t Adjustment);
    virtual void EmitCFIEscape(StringRef Values);
    virtual void EmitCFISignalFrame();

    virtual void EmitWin64EHStartProc(const MCSymbol *Symbol);
    virtual void EmitWin64EHEndProc();
    virtual void EmitWin64EHStartChained();
    virtual void EmitWin64EHEndChained();
    virtual void EmitWin64EHHandler(const MCSymbol *Sym, bool Unwind,
                                    bool Except);
    virtual void EmitWin64EHHandlerData();
    virtual void EmitWin64EHPushReg(unsigned Register);
    virtual void EmitWin64EHSetFrame(unsigned Register, unsigned Offset);
    virtual void EmitWin64EHAllocStack(unsigned Size);
    virtual void EmitWin64EHSaveReg(unsigned Register, unsigned Offset);
    virtual void EmitWin64EHSaveXMM(unsigned Register, unsigned Offset);
    virtual void EmitWin64EHPushFrame(bool Code);
    virtual void EmitWin64EHEndProlog();

    /// EmitInstruction - Emit the given @p Instruction into the current
    /// section.
    virtual void EmitInstruction(const MCInst &Inst) = 0;

    /// EmitRawText - If this file is backed by a assembly streamer, this dumps
    /// the specified string in the output .s file.  This capability is
    /// indicated by the hasRawTextSupport() predicate.  By default this aborts.
    virtual void EmitRawText(StringRef String);
    void EmitRawText(const Twine &String);

    /// ARM-related methods.
    /// FIXME: Eventually we should have some "target MC streamer" and move
    /// these methods there.
    virtual void EmitFnStart();
    virtual void EmitFnEnd();
    virtual void EmitCantUnwind();
    virtual void EmitPersonality(const MCSymbol *Personality);
    virtual void EmitHandlerData();
    virtual void EmitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset = 0);
    virtual void EmitPad(int64_t Offset);
    virtual void EmitRegSave(const SmallVectorImpl<unsigned> &RegList,
                             bool isVector);

    /// PPC-related methods.
    /// FIXME: Eventually replace it with some "target MC streamer" and move
    /// these methods there.
    virtual void EmitTCEntry(const MCSymbol &S);

    /// FinishImpl - Streamer specific finalization.
    virtual void FinishImpl() = 0;
    /// Finish - Finish emission of machine code.
    void Finish();
  };

  /// createNullStreamer - Create a dummy machine code streamer, which does
  /// nothing. This is useful for timing the assembler front end.
  MCStreamer *createNullStreamer(MCContext &Ctx);

  /// createAsmStreamer - Create a machine code streamer which will print out
  /// assembly for the native target, suitable for compiling with a native
  /// assembler.
  ///
  /// \param InstPrint - If given, the instruction printer to use. If not given
  /// the MCInst representation will be printed.  This method takes ownership of
  /// InstPrint.
  ///
  /// \param CE - If given, a code emitter to use to show the instruction
  /// encoding inline with the assembly. This method takes ownership of \p CE.
  ///
  /// \param TAB - If given, a target asm backend to use to show the fixup
  /// information in conjunction with encoding information. This method takes
  /// ownership of \p TAB.
  ///
  /// \param ShowInst - Whether to show the MCInst representation inline with
  /// the assembly.
  MCStreamer *createAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                bool isVerboseAsm,
                                bool useLoc,
                                bool useCFI,
                                bool useDwarfDirectory,
                                MCInstPrinter *InstPrint = 0,
                                MCCodeEmitter *CE = 0,
                                MCAsmBackend *TAB = 0,
                                bool ShowInst = false);

  /// createMachOStreamer - Create a machine code streamer which will generate
  /// Mach-O format object files.
  ///
  /// Takes ownership of \p TAB and \p CE.
  MCStreamer *createMachOStreamer(MCContext &Ctx, MCAsmBackend &TAB,
                                  raw_ostream &OS, MCCodeEmitter *CE,
                                  bool RelaxAll = false);

  /// createWinCOFFStreamer - Create a machine code streamer which will
  /// generate Microsoft COFF format object files.
  ///
  /// Takes ownership of \p TAB and \p CE.
  MCStreamer *createWinCOFFStreamer(MCContext &Ctx,
                                    MCAsmBackend &TAB,
                                    MCCodeEmitter &CE, raw_ostream &OS,
                                    bool RelaxAll = false);

  /// createELFStreamer - Create a machine code streamer which will generate
  /// ELF format object files.
  MCStreamer *createELFStreamer(MCContext &Ctx, MCAsmBackend &TAB,
                                raw_ostream &OS, MCCodeEmitter *CE,
                                bool RelaxAll, bool NoExecStack);

  /// createPureStreamer - Create a machine code streamer which will generate
  /// "pure" MC object files, for use with MC-JIT and testing tools.
  ///
  /// Takes ownership of \p TAB and \p CE.
  MCStreamer *createPureStreamer(MCContext &Ctx, MCAsmBackend &TAB,
                                 raw_ostream &OS, MCCodeEmitter *CE);

} // end namespace llvm

#endif
