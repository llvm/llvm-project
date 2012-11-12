//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/PathV2.h"
#include <cctype>
using namespace llvm;

namespace {

class MCAsmStreamer : public MCStreamer {
protected:
  formatted_raw_ostream &OS;
  const MCAsmInfo &MAI;
private:
  OwningPtr<MCInstPrinter> InstPrinter;
  OwningPtr<MCCodeEmitter> Emitter;
  OwningPtr<MCAsmBackend> AsmBackend;

  SmallString<128> CommentToEmit;
  raw_svector_ostream CommentStream;

  unsigned IsVerboseAsm : 1;
  unsigned ShowInst : 1;
  unsigned UseLoc : 1;
  unsigned UseCFI : 1;
  unsigned UseDwarfDirectory : 1;

  enum EHSymbolFlags { EHGlobal         = 1,
                       EHWeakDefinition = 1 << 1,
                       EHPrivateExtern  = 1 << 2 };
  DenseMap<const MCSymbol*, unsigned> FlagMap;

  bool needsSet(const MCExpr *Value);

  void EmitRegisterName(int64_t Register);
  virtual void EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame);
  virtual void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame);

public:
  MCAsmStreamer(MCContext &Context, formatted_raw_ostream &os,
                bool isVerboseAsm, bool useLoc, bool useCFI,
                bool useDwarfDirectory,
                MCInstPrinter *printer, MCCodeEmitter *emitter,
                MCAsmBackend *asmbackend,
                bool showInst)
    : MCStreamer(Context), OS(os), MAI(Context.getAsmInfo()),
      InstPrinter(printer), Emitter(emitter), AsmBackend(asmbackend),
      CommentStream(CommentToEmit), IsVerboseAsm(isVerboseAsm),
      ShowInst(showInst), UseLoc(useLoc), UseCFI(useCFI),
      UseDwarfDirectory(useDwarfDirectory) {
    if (InstPrinter && IsVerboseAsm)
      InstPrinter->setCommentStream(CommentStream);
  }
  ~MCAsmStreamer() {}

  inline void EmitEOL() {
    // If we don't have any comments, just emit a \n.
    if (!IsVerboseAsm) {
      OS << '\n';
      return;
    }
    EmitCommentsAndEOL();
  }
  void EmitCommentsAndEOL();

  /// isVerboseAsm - Return true if this streamer supports verbose assembly at
  /// all.
  virtual bool isVerboseAsm() const { return IsVerboseAsm; }

  /// hasRawTextSupport - We support EmitRawText.
  virtual bool hasRawTextSupport() const { return true; }

  /// AddComment - Add a comment that can be emitted to the generated .s
  /// file if applicable as a QoI issue to make the output of the compiler
  /// more readable.  This only affects the MCAsmStreamer, and only when
  /// verbose assembly output is enabled.
  virtual void AddComment(const Twine &T);

  /// AddEncodingComment - Add a comment showing the encoding of an instruction.
  virtual void AddEncodingComment(const MCInst &Inst);

  /// GetCommentOS - Return a raw_ostream that comments can be written to.
  /// Unlike AddComment, you are required to terminate comments with \n if you
  /// use this method.
  virtual raw_ostream &GetCommentOS() {
    if (!IsVerboseAsm)
      return nulls();  // Discard comments unless in verbose asm mode.
    return CommentStream;
  }

  /// AddBlankLine - Emit a blank line to a .s file to pretty it up.
  virtual void AddBlankLine() {
    EmitEOL();
  }

  /// @name MCStreamer Interface
  /// @{

  virtual void ChangeSection(const MCSection *Section);

  virtual void InitSections() {
    // FIXME, this is MachO specific, but the testsuite
    // expects this.
    SwitchSection(getContext().getMachOSection("__TEXT", "__text",
                         MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                         0, SectionKind::getText()));
  }

  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitEHSymAttributes(const MCSymbol *Symbol,
                                   MCSymbol *EHSymbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitDataRegion(MCDataRegionType Kind);
  virtual void EmitThumbFunc(MCSymbol *Func);

  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);
  virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                        const MCSymbol *LastLabel,
                                        const MCSymbol *Label,
                                        unsigned PointerSize);
  virtual void EmitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                         const MCSymbol *Label);

  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);

  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol);
  virtual void EmitCOFFSymbolStorageClass(int StorageClass);
  virtual void EmitCOFFSymbolType(int Type);
  virtual void EndCOFFSymbolDef();
  virtual void EmitCOFFSecRel32(MCSymbol const *Symbol);
  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);

  /// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
  ///
  /// @param Symbol - The common symbol to emit.
  /// @param Size - The size of the common symbol.
  /// @param ByteAlignment - The alignment of the common symbol in bytes.
  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment);

  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            uint64_t Size = 0, unsigned ByteAlignment = 0);

  virtual void EmitTBSSSymbol (const MCSection *Section, MCSymbol *Symbol,
                               uint64_t Size, unsigned ByteAlignment = 0);

  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);

  virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                             unsigned AddrSpace);
  virtual void EmitIntValue(uint64_t Value, unsigned Size,
                            unsigned AddrSpace = 0);

  virtual void EmitULEB128Value(const MCExpr *Value);

  virtual void EmitSLEB128Value(const MCExpr *Value);

  virtual void EmitGPRel64Value(const MCExpr *Value);

  virtual void EmitGPRel32Value(const MCExpr *Value);


  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                        unsigned AddrSpace);

  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);

  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);

  virtual bool EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);

  virtual void EmitFileDirective(StringRef Filename);
  virtual bool EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                      StringRef Filename);
  virtual void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                     unsigned Column, unsigned Flags,
                                     unsigned Isa, unsigned Discriminator,
                                     StringRef FileName);

  virtual void EmitCFISections(bool EH, bool Debug);
  virtual void EmitCFIDefCfa(int64_t Register, int64_t Offset);
  virtual void EmitCFIDefCfaOffset(int64_t Offset);
  virtual void EmitCFIDefCfaRegister(int64_t Register);
  virtual void EmitCFIOffset(int64_t Register, int64_t Offset);
  virtual void EmitCFIPersonality(const MCSymbol *Sym, unsigned Encoding);
  virtual void EmitCFILsda(const MCSymbol *Sym, unsigned Encoding);
  virtual void EmitCFIRememberState();
  virtual void EmitCFIRestoreState();
  virtual void EmitCFISameValue(int64_t Register);
  virtual void EmitCFIRelOffset(int64_t Register, int64_t Offset);
  virtual void EmitCFIAdjustCfaOffset(int64_t Adjustment);
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

  virtual void EmitFnStart();
  virtual void EmitFnEnd();
  virtual void EmitCantUnwind();
  virtual void EmitPersonality(const MCSymbol *Personality);
  virtual void EmitHandlerData();
  virtual void EmitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset = 0);
  virtual void EmitPad(int64_t Offset);
  virtual void EmitRegSave(const SmallVectorImpl<unsigned> &RegList, bool);

  virtual void EmitTCEntry(const MCSymbol &S);

  virtual void EmitInstruction(const MCInst &Inst);

  /// EmitRawText - If this file is backed by an assembly streamer, this dumps
  /// the specified string in the output .s file.  This capability is
  /// indicated by the hasRawTextSupport() predicate.
  virtual void EmitRawText(StringRef String);

  virtual void FinishImpl();

  /// @}
};

} // end anonymous namespace.

/// AddComment - Add a comment that can be emitted to the generated .s
/// file if applicable as a QoI issue to make the output of the compiler
/// more readable.  This only affects the MCAsmStreamer, and only when
/// verbose assembly output is enabled.
void MCAsmStreamer::AddComment(const Twine &T) {
  if (!IsVerboseAsm) return;

  // Make sure that CommentStream is flushed.
  CommentStream.flush();

  T.toVector(CommentToEmit);
  // Each comment goes on its own line.
  CommentToEmit.push_back('\n');

  // Tell the comment stream that the vector changed underneath it.
  CommentStream.resync();
}

void MCAsmStreamer::EmitCommentsAndEOL() {
  if (CommentToEmit.empty() && CommentStream.GetNumBytesInBuffer() == 0) {
    OS << '\n';
    return;
  }

  CommentStream.flush();
  StringRef Comments = CommentToEmit.str();

  assert(Comments.back() == '\n' &&
         "Comment array not newline terminated");
  do {
    // Emit a line of comments.
    OS.PadToColumn(MAI.getCommentColumn());
    size_t Position = Comments.find('\n');
    OS << MAI.getCommentString() << ' ' << Comments.substr(0, Position) << '\n';

    Comments = Comments.substr(Position+1);
  } while (!Comments.empty());

  CommentToEmit.clear();
  // Tell the comment stream that the vector changed underneath it.
  CommentStream.resync();
}

static inline int64_t truncateToSize(int64_t Value, unsigned Bytes) {
  assert(Bytes && "Invalid size!");
  return Value & ((uint64_t) (int64_t) -1 >> (64 - Bytes * 8));
}

void MCAsmStreamer::ChangeSection(const MCSection *Section) {
  assert(Section && "Cannot switch to a null section!");
  Section->PrintSwitchToSection(MAI, OS);
}

void MCAsmStreamer::EmitEHSymAttributes(const MCSymbol *Symbol,
                                        MCSymbol *EHSymbol) {
  if (UseCFI)
    return;

  unsigned Flags = FlagMap.lookup(Symbol);

  if (Flags & EHGlobal)
    EmitSymbolAttribute(EHSymbol, MCSA_Global);
  if (Flags & EHWeakDefinition)
    EmitSymbolAttribute(EHSymbol, MCSA_WeakDefinition);
  if (Flags & EHPrivateExtern)
    EmitSymbolAttribute(EHSymbol, MCSA_PrivateExtern);
}

void MCAsmStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  MCStreamer::EmitLabel(Symbol);

  OS << *Symbol << MAI.getLabelSuffix();
  EmitEOL();
}

void MCAsmStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  case MCAF_SyntaxUnified:         OS << "\t.syntax unified"; break;
  case MCAF_SubsectionsViaSymbols: OS << ".subsections_via_symbols"; break;
  case MCAF_Code16:                OS << '\t'<< MAI.getCode16Directive(); break;
  case MCAF_Code32:                OS << '\t'<< MAI.getCode32Directive(); break;
  case MCAF_Code64:                OS << '\t'<< MAI.getCode64Directive(); break;
  }
  EmitEOL();
}

void MCAsmStreamer::EmitDataRegion(MCDataRegionType Kind) {
  MCContext &Ctx = getContext();
  const MCAsmInfo &MAI = Ctx.getAsmInfo();
  if (!MAI.doesSupportDataRegionDirectives())
    return;
  switch (Kind) {
  case MCDR_DataRegion:            OS << "\t.data_region"; break;
  case MCDR_DataRegionJT8:         OS << "\t.data_region jt8"; break;
  case MCDR_DataRegionJT16:        OS << "\t.data_region jt16"; break;
  case MCDR_DataRegionJT32:        OS << "\t.data_region jt32"; break;
  case MCDR_DataRegionEnd:         OS << "\t.end_data_region"; break;
  }
  EmitEOL();
}

void MCAsmStreamer::EmitThumbFunc(MCSymbol *Func) {
  // This needs to emit to a temporary string to get properly quoted
  // MCSymbols when they have spaces in them.
  OS << "\t.thumb_func";
  // Only Mach-O hasSubsectionsViaSymbols()
  if (MAI.hasSubsectionsViaSymbols())
    OS << '\t' << *Func;
  EmitEOL();
}

void MCAsmStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  OS << *Symbol << " = " << *Value;
  EmitEOL();

  // FIXME: Lift context changes into super class.
  Symbol->setVariableValue(Value);
}

void MCAsmStreamer::EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) {
  OS << ".weakref " << *Alias << ", " << *Symbol;
  EmitEOL();
}

void MCAsmStreamer::EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                             const MCSymbol *LastLabel,
                                             const MCSymbol *Label,
                                             unsigned PointerSize) {
  EmitDwarfSetLineAddr(LineDelta, Label, PointerSize);
}

void MCAsmStreamer::EmitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                              const MCSymbol *Label) {
  EmitIntValue(dwarf::DW_CFA_advance_loc4, 1);
  const MCExpr *AddrDelta = BuildSymbolDiff(getContext(), Label, LastLabel);
  AddrDelta = ForceExpAbs(AddrDelta);
  EmitValue(AddrDelta, 4);
}


void MCAsmStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                        MCSymbolAttr Attribute) {
  switch (Attribute) {
  case MCSA_Invalid: llvm_unreachable("Invalid symbol attribute");
  case MCSA_ELF_TypeFunction:    /// .type _foo, STT_FUNC  # aka @function
  case MCSA_ELF_TypeIndFunction: /// .type _foo, STT_GNU_IFUNC
  case MCSA_ELF_TypeObject:      /// .type _foo, STT_OBJECT  # aka @object
  case MCSA_ELF_TypeTLS:         /// .type _foo, STT_TLS     # aka @tls_object
  case MCSA_ELF_TypeCommon:      /// .type _foo, STT_COMMON  # aka @common
  case MCSA_ELF_TypeNoType:      /// .type _foo, STT_NOTYPE  # aka @notype
  case MCSA_ELF_TypeGnuUniqueObject:  /// .type _foo, @gnu_unique_object
    assert(MAI.hasDotTypeDotSizeDirective() && "Symbol Attr not supported");
    OS << "\t.type\t" << *Symbol << ','
       << ((MAI.getCommentString()[0] != '@') ? '@' : '%');
    switch (Attribute) {
    default: llvm_unreachable("Unknown ELF .type");
    case MCSA_ELF_TypeFunction:    OS << "function"; break;
    case MCSA_ELF_TypeIndFunction: OS << "gnu_indirect_function"; break;
    case MCSA_ELF_TypeObject:      OS << "object"; break;
    case MCSA_ELF_TypeTLS:         OS << "tls_object"; break;
    case MCSA_ELF_TypeCommon:      OS << "common"; break;
    case MCSA_ELF_TypeNoType:      OS << "no_type"; break;
    case MCSA_ELF_TypeGnuUniqueObject: OS << "gnu_unique_object"; break;
    }
    EmitEOL();
    return;
  case MCSA_Global: // .globl/.global
    OS << MAI.getGlobalDirective();
    FlagMap[Symbol] |= EHGlobal;
    break;
  case MCSA_Hidden:         OS << "\t.hidden\t";          break;
  case MCSA_IndirectSymbol: OS << "\t.indirect_symbol\t"; break;
  case MCSA_Internal:       OS << "\t.internal\t";        break;
  case MCSA_LazyReference:  OS << "\t.lazy_reference\t";  break;
  case MCSA_Local:          OS << "\t.local\t";           break;
  case MCSA_NoDeadStrip:    OS << "\t.no_dead_strip\t";   break;
  case MCSA_SymbolResolver: OS << "\t.symbol_resolver\t"; break;
  case MCSA_PrivateExtern:
    OS << "\t.private_extern\t";
    FlagMap[Symbol] |= EHPrivateExtern;
    break;
  case MCSA_Protected:      OS << "\t.protected\t";       break;
  case MCSA_Reference:      OS << "\t.reference\t";       break;
  case MCSA_Weak:           OS << "\t.weak\t";            break;
  case MCSA_WeakDefinition:
    OS << "\t.weak_definition\t";
    FlagMap[Symbol] |= EHWeakDefinition;
    break;
      // .weak_reference
  case MCSA_WeakReference:  OS << MAI.getWeakRefDirective(); break;
  case MCSA_WeakDefAutoPrivate: OS << "\t.weak_def_can_be_hidden\t"; break;
  }

  OS << *Symbol;
  EmitEOL();
}

void MCAsmStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  OS << ".desc" << ' ' << *Symbol << ',' << DescValue;
  EmitEOL();
}

void MCAsmStreamer::BeginCOFFSymbolDef(const MCSymbol *Symbol) {
  OS << "\t.def\t " << *Symbol << ';';
  EmitEOL();
}

void MCAsmStreamer::EmitCOFFSymbolStorageClass (int StorageClass) {
  OS << "\t.scl\t" << StorageClass << ';';
  EmitEOL();
}

void MCAsmStreamer::EmitCOFFSymbolType (int Type) {
  OS << "\t.type\t" << Type << ';';
  EmitEOL();
}

void MCAsmStreamer::EndCOFFSymbolDef() {
  OS << "\t.endef";
  EmitEOL();
}

void MCAsmStreamer::EmitCOFFSecRel32(MCSymbol const *Symbol) {
  OS << "\t.secrel32\t" << *Symbol << '\n';
  EmitEOL();
}

void MCAsmStreamer::EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  assert(MAI.hasDotTypeDotSizeDirective());
  OS << "\t.size\t" << *Symbol << ", " << *Value << '\n';
}

void MCAsmStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment) {
  OS << "\t.comm\t" << *Symbol << ',' << Size;
  if (ByteAlignment != 0) {
    if (MAI.getCOMMDirectiveAlignmentIsInBytes())
      OS << ',' << ByteAlignment;
    else
      OS << ',' << Log2_32(ByteAlignment);
  }
  EmitEOL();
}

/// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
///
/// @param Symbol - The common symbol to emit.
/// @param Size - The size of the common symbol.
void MCAsmStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                          unsigned ByteAlign) {
  OS << "\t.lcomm\t" << *Symbol << ',' << Size;
  if (ByteAlign > 1) {
    switch (MAI.getLCOMMDirectiveAlignmentType()) {
    case LCOMM::NoAlignment:
      llvm_unreachable("alignment not supported on .lcomm!");
    case LCOMM::ByteAlignment:
      OS << ',' << ByteAlign;
      break;
    case LCOMM::Log2Alignment:
      assert(isPowerOf2_32(ByteAlign) && "alignment must be a power of 2");
      OS << ',' << Log2_32(ByteAlign);
      break;
    }
  }
  EmitEOL();
}

void MCAsmStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                 uint64_t Size, unsigned ByteAlignment) {
  // Note: a .zerofill directive does not switch sections.
  OS << ".zerofill ";

  // This is a mach-o specific directive.
  const MCSectionMachO *MOSection = ((const MCSectionMachO*)Section);
  OS << MOSection->getSegmentName() << "," << MOSection->getSectionName();

  if (Symbol != NULL) {
    OS << ',' << *Symbol << ',' << Size;
    if (ByteAlignment != 0)
      OS << ',' << Log2_32(ByteAlignment);
  }
  EmitEOL();
}

// .tbss sym, size, align
// This depends that the symbol has already been mangled from the original,
// e.g. _a.
void MCAsmStreamer::EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                   uint64_t Size, unsigned ByteAlignment) {
  assert(Symbol != NULL && "Symbol shouldn't be NULL!");
  // Instead of using the Section we'll just use the shortcut.
  // This is a mach-o specific directive and section.
  OS << ".tbss " << *Symbol << ", " << Size;

  // Output align if we have it.  We default to 1 so don't bother printing
  // that.
  if (ByteAlignment > 1) OS << ", " << Log2_32(ByteAlignment);

  EmitEOL();
}

static inline char toOctal(int X) { return (X&7)+'0'; }

static void PrintQuotedString(StringRef Data, raw_ostream &OS) {
  OS << '"';

  for (unsigned i = 0, e = Data.size(); i != e; ++i) {
    unsigned char C = Data[i];
    if (C == '"' || C == '\\') {
      OS << '\\' << (char)C;
      continue;
    }

    if (isprint((unsigned char)C)) {
      OS << (char)C;
      continue;
    }

    switch (C) {
      case '\b': OS << "\\b"; break;
      case '\f': OS << "\\f"; break;
      case '\n': OS << "\\n"; break;
      case '\r': OS << "\\r"; break;
      case '\t': OS << "\\t"; break;
      default:
        OS << '\\';
        OS << toOctal(C >> 6);
        OS << toOctal(C >> 3);
        OS << toOctal(C >> 0);
        break;
    }
  }

  OS << '"';
}


void MCAsmStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  assert(getCurrentSection() && "Cannot emit contents before setting section!");
  if (Data.empty()) return;

  if (Data.size() == 1) {
    OS << MAI.getData8bitsDirective(AddrSpace);
    OS << (unsigned)(unsigned char)Data[0];
    EmitEOL();
    return;
  }

  // If the data ends with 0 and the target supports .asciz, use it, otherwise
  // use .ascii
  if (MAI.getAscizDirective() && Data.back() == 0) {
    OS << MAI.getAscizDirective();
    Data = Data.substr(0, Data.size()-1);
  } else {
    OS << MAI.getAsciiDirective();
  }

  OS << ' ';
  PrintQuotedString(Data, OS);
  EmitEOL();
}

void MCAsmStreamer::EmitIntValue(uint64_t Value, unsigned Size,
                                 unsigned AddrSpace) {
  EmitValue(MCConstantExpr::Create(Value, getContext()), Size, AddrSpace);
}

void MCAsmStreamer::EmitValueImpl(const MCExpr *Value, unsigned Size,
                                  unsigned AddrSpace) {
  assert(getCurrentSection() && "Cannot emit contents before setting section!");
  const char *Directive = 0;
  switch (Size) {
  default: break;
  case 1: Directive = MAI.getData8bitsDirective(AddrSpace); break;
  case 2: Directive = MAI.getData16bitsDirective(AddrSpace); break;
  case 4: Directive = MAI.getData32bitsDirective(AddrSpace); break;
  case 8:
    Directive = MAI.getData64bitsDirective(AddrSpace);
    // If the target doesn't support 64-bit data, emit as two 32-bit halves.
    if (Directive) break;
    int64_t IntValue;
    if (!Value->EvaluateAsAbsolute(IntValue))
      report_fatal_error("Don't know how to emit this value.");
    if (getContext().getAsmInfo().isLittleEndian()) {
      EmitIntValue((uint32_t)(IntValue >> 0 ), 4, AddrSpace);
      EmitIntValue((uint32_t)(IntValue >> 32), 4, AddrSpace);
    } else {
      EmitIntValue((uint32_t)(IntValue >> 32), 4, AddrSpace);
      EmitIntValue((uint32_t)(IntValue >> 0 ), 4, AddrSpace);
    }
    return;
  }

  assert(Directive && "Invalid size for machine code value!");
  OS << Directive << *Value;
  EmitEOL();
}

void MCAsmStreamer::EmitULEB128Value(const MCExpr *Value) {
  int64_t IntValue;
  if (Value->EvaluateAsAbsolute(IntValue)) {
    EmitULEB128IntValue(IntValue);
    return;
  }
  assert(MAI.hasLEB128() && "Cannot print a .uleb");
  OS << ".uleb128 " << *Value;
  EmitEOL();
}

void MCAsmStreamer::EmitSLEB128Value(const MCExpr *Value) {
  int64_t IntValue;
  if (Value->EvaluateAsAbsolute(IntValue)) {
    EmitSLEB128IntValue(IntValue);
    return;
  }
  assert(MAI.hasLEB128() && "Cannot print a .sleb");
  OS << ".sleb128 " << *Value;
  EmitEOL();
}

void MCAsmStreamer::EmitGPRel64Value(const MCExpr *Value) {
  assert(MAI.getGPRel64Directive() != 0);
  OS << MAI.getGPRel64Directive() << *Value;
  EmitEOL();
}

void MCAsmStreamer::EmitGPRel32Value(const MCExpr *Value) {
  assert(MAI.getGPRel32Directive() != 0);
  OS << MAI.getGPRel32Directive() << *Value;
  EmitEOL();
}


/// EmitFill - Emit NumBytes bytes worth of the value specified by
/// FillValue.  This implements directives such as '.space'.
void MCAsmStreamer::EmitFill(uint64_t NumBytes, uint8_t FillValue,
                             unsigned AddrSpace) {
  if (NumBytes == 0) return;

  if (AddrSpace == 0)
    if (const char *ZeroDirective = MAI.getZeroDirective()) {
      OS << ZeroDirective << NumBytes;
      if (FillValue != 0)
        OS << ',' << (int)FillValue;
      EmitEOL();
      return;
    }

  // Emit a byte at a time.
  MCStreamer::EmitFill(NumBytes, FillValue, AddrSpace);
}

void MCAsmStreamer::EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                         unsigned ValueSize,
                                         unsigned MaxBytesToEmit) {
  // Some assemblers don't support non-power of two alignments, so we always
  // emit alignments as a power of two if possible.
  if (isPowerOf2_32(ByteAlignment)) {
    switch (ValueSize) {
    default: llvm_unreachable("Invalid size for machine code value!");
    case 1: OS << MAI.getAlignDirective(); break;
    // FIXME: use MAI for this!
    case 2: OS << ".p2alignw "; break;
    case 4: OS << ".p2alignl "; break;
    case 8: llvm_unreachable("Unsupported alignment size!");
    }

    if (MAI.getAlignmentIsInBytes())
      OS << ByteAlignment;
    else
      OS << Log2_32(ByteAlignment);

    if (Value || MaxBytesToEmit) {
      OS << ", 0x";
      OS.write_hex(truncateToSize(Value, ValueSize));

      if (MaxBytesToEmit)
        OS << ", " << MaxBytesToEmit;
    }
    EmitEOL();
    return;
  }

  // Non-power of two alignment.  This is not widely supported by assemblers.
  // FIXME: Parameterize this based on MAI.
  switch (ValueSize) {
  default: llvm_unreachable("Invalid size for machine code value!");
  case 1: OS << ".balign";  break;
  case 2: OS << ".balignw"; break;
  case 4: OS << ".balignl"; break;
  case 8: llvm_unreachable("Unsupported alignment size!");
  }

  OS << ' ' << ByteAlignment;
  OS << ", " << truncateToSize(Value, ValueSize);
  if (MaxBytesToEmit)
    OS << ", " << MaxBytesToEmit;
  EmitEOL();
}

void MCAsmStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                      unsigned MaxBytesToEmit) {
  // Emit with a text fill value.
  EmitValueToAlignment(ByteAlignment, MAI.getTextAlignFillValue(),
                       1, MaxBytesToEmit);
}

bool MCAsmStreamer::EmitValueToOffset(const MCExpr *Offset,
                                      unsigned char Value) {
  // FIXME: Verify that Offset is associated with the current section.
  OS << ".org " << *Offset << ", " << (unsigned) Value;
  EmitEOL();
  return false;
}


void MCAsmStreamer::EmitFileDirective(StringRef Filename) {
  assert(MAI.hasSingleParameterDotFile());
  OS << "\t.file\t";
  PrintQuotedString(Filename, OS);
  EmitEOL();
}

bool MCAsmStreamer::EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                           StringRef Filename) {
  if (!UseDwarfDirectory && !Directory.empty()) {
    if (sys::path::is_absolute(Filename))
      return EmitDwarfFileDirective(FileNo, "", Filename);

    SmallString<128> FullPathName = Directory;
    sys::path::append(FullPathName, Filename);
    return EmitDwarfFileDirective(FileNo, "", FullPathName);
  }

  if (UseLoc) {
    OS << "\t.file\t" << FileNo << ' ';
    if (!Directory.empty()) {
      PrintQuotedString(Directory, OS);
      OS << ' ';
    }
    PrintQuotedString(Filename, OS);
    EmitEOL();
  }
  return this->MCStreamer::EmitDwarfFileDirective(FileNo, Directory, Filename);
}

void MCAsmStreamer::EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                          unsigned Column, unsigned Flags,
                                          unsigned Isa,
                                          unsigned Discriminator,
                                          StringRef FileName) {
  this->MCStreamer::EmitDwarfLocDirective(FileNo, Line, Column, Flags,
                                          Isa, Discriminator, FileName);
  if (!UseLoc)
    return;

  OS << "\t.loc\t" << FileNo << " " << Line << " " << Column;
  if (Flags & DWARF2_FLAG_BASIC_BLOCK)
    OS << " basic_block";
  if (Flags & DWARF2_FLAG_PROLOGUE_END)
    OS << " prologue_end";
  if (Flags & DWARF2_FLAG_EPILOGUE_BEGIN)
    OS << " epilogue_begin";

  unsigned OldFlags = getContext().getCurrentDwarfLoc().getFlags();
  if ((Flags & DWARF2_FLAG_IS_STMT) != (OldFlags & DWARF2_FLAG_IS_STMT)) {
    OS << " is_stmt ";

    if (Flags & DWARF2_FLAG_IS_STMT)
      OS << "1";
    else
      OS << "0";
  }

  if (Isa)
    OS << "isa " << Isa;
  if (Discriminator)
    OS << "discriminator " << Discriminator;

  if (IsVerboseAsm) {
    OS.PadToColumn(MAI.getCommentColumn());
    OS << MAI.getCommentString() << ' ' << FileName << ':'
       << Line << ':' << Column;
  }
  EmitEOL();
}

void MCAsmStreamer::EmitCFISections(bool EH, bool Debug) {
  MCStreamer::EmitCFISections(EH, Debug);

  if (!UseCFI)
    return;

  OS << "\t.cfi_sections ";
  if (EH) {
    OS << ".eh_frame";
    if (Debug)
      OS << ", .debug_frame";
  } else if (Debug) {
    OS << ".debug_frame";
  }

  EmitEOL();
}

void MCAsmStreamer::EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
  if (!UseCFI) {
    RecordProcStart(Frame);
    return;
  }

  OS << "\t.cfi_startproc";
  EmitEOL();
}

void MCAsmStreamer::EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) {
  if (!UseCFI) {
    RecordProcEnd(Frame);
    return;
  }

  // Put a dummy non-null value in Frame.End to mark that this frame has been
  // closed.
  Frame.End = (MCSymbol *) 1;

  OS << "\t.cfi_endproc";
  EmitEOL();
}

void MCAsmStreamer::EmitRegisterName(int64_t Register) {
  if (InstPrinter && !MAI.useDwarfRegNumForCFI()) {
    const MCRegisterInfo &MRI = getContext().getRegisterInfo();
    unsigned LLVMRegister = MRI.getLLVMRegNum(Register, true);
    InstPrinter->printRegName(OS, LLVMRegister);
  } else {
    OS << Register;
  }
}

void MCAsmStreamer::EmitCFIDefCfa(int64_t Register, int64_t Offset) {
  MCStreamer::EmitCFIDefCfa(Register, Offset);

  if (!UseCFI)
    return;

  OS << "\t.cfi_def_cfa ";
  EmitRegisterName(Register);
  OS << ", " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitCFIDefCfaOffset(int64_t Offset) {
  MCStreamer::EmitCFIDefCfaOffset(Offset);

  if (!UseCFI)
    return;

  OS << "\t.cfi_def_cfa_offset " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitCFIDefCfaRegister(int64_t Register) {
  MCStreamer::EmitCFIDefCfaRegister(Register);

  if (!UseCFI)
    return;

  OS << "\t.cfi_def_cfa_register ";
  EmitRegisterName(Register);
  EmitEOL();
}

void MCAsmStreamer::EmitCFIOffset(int64_t Register, int64_t Offset) {
  this->MCStreamer::EmitCFIOffset(Register, Offset);

  if (!UseCFI)
    return;

  OS << "\t.cfi_offset ";
  EmitRegisterName(Register);
  OS << ", " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitCFIPersonality(const MCSymbol *Sym,
                                       unsigned Encoding) {
  MCStreamer::EmitCFIPersonality(Sym, Encoding);

  if (!UseCFI)
    return;

  OS << "\t.cfi_personality " << Encoding << ", " << *Sym;
  EmitEOL();
}

void MCAsmStreamer::EmitCFILsda(const MCSymbol *Sym, unsigned Encoding) {
  MCStreamer::EmitCFILsda(Sym, Encoding);

  if (!UseCFI)
    return;

  OS << "\t.cfi_lsda " << Encoding << ", " << *Sym;
  EmitEOL();
}

void MCAsmStreamer::EmitCFIRememberState() {
  MCStreamer::EmitCFIRememberState();

  if (!UseCFI)
    return;

  OS << "\t.cfi_remember_state";
  EmitEOL();
}

void MCAsmStreamer::EmitCFIRestoreState() {
  MCStreamer::EmitCFIRestoreState();

  if (!UseCFI)
    return;

  OS << "\t.cfi_restore_state";
  EmitEOL();
}

void MCAsmStreamer::EmitCFISameValue(int64_t Register) {
  MCStreamer::EmitCFISameValue(Register);

  if (!UseCFI)
    return;

  OS << "\t.cfi_same_value ";
  EmitRegisterName(Register);
  EmitEOL();
}

void MCAsmStreamer::EmitCFIRelOffset(int64_t Register, int64_t Offset) {
  MCStreamer::EmitCFIRelOffset(Register, Offset);

  if (!UseCFI)
    return;

  OS << "\t.cfi_rel_offset ";
  EmitRegisterName(Register);
  OS << ", " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitCFIAdjustCfaOffset(int64_t Adjustment) {
  MCStreamer::EmitCFIAdjustCfaOffset(Adjustment);

  if (!UseCFI)
    return;

  OS << "\t.cfi_adjust_cfa_offset " << Adjustment;
  EmitEOL();
}

void MCAsmStreamer::EmitCFISignalFrame() {
  MCStreamer::EmitCFISignalFrame();

  if (!UseCFI)
    return;

  OS << "\t.cfi_signal_frame";
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHStartProc(const MCSymbol *Symbol) {
  MCStreamer::EmitWin64EHStartProc(Symbol);

  OS << ".seh_proc " << *Symbol;
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHEndProc() {
  MCStreamer::EmitWin64EHEndProc();

  OS << "\t.seh_endproc";
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHStartChained() {
  MCStreamer::EmitWin64EHStartChained();

  OS << "\t.seh_startchained";
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHEndChained() {
  MCStreamer::EmitWin64EHEndChained();

  OS << "\t.seh_endchained";
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHHandler(const MCSymbol *Sym, bool Unwind,
                                       bool Except) {
  MCStreamer::EmitWin64EHHandler(Sym, Unwind, Except);

  OS << "\t.seh_handler " << *Sym;
  if (Unwind)
    OS << ", @unwind";
  if (Except)
    OS << ", @except";
  EmitEOL();
}

static const MCSection *getWin64EHTableSection(StringRef suffix,
                                               MCContext &context) {
  // FIXME: This doesn't belong in MCObjectFileInfo. However,
  /// this duplicate code in MCWin64EH.cpp.
  if (suffix == "")
    return context.getObjectFileInfo()->getXDataSection();
  return context.getCOFFSection((".xdata"+suffix).str(),
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ |
                                COFF::IMAGE_SCN_MEM_WRITE,
                                SectionKind::getDataRel());
}

void MCAsmStreamer::EmitWin64EHHandlerData() {
  MCStreamer::EmitWin64EHHandlerData();

  // Switch sections. Don't call SwitchSection directly, because that will
  // cause the section switch to be visible in the emitted assembly.
  // We only do this so the section switch that terminates the handler
  // data block is visible.
  MCWin64EHUnwindInfo *CurFrame = getCurrentW64UnwindInfo();
  StringRef suffix=MCWin64EHUnwindEmitter::GetSectionSuffix(CurFrame->Function);
  const MCSection *xdataSect = getWin64EHTableSection(suffix, getContext());
  if (xdataSect)
    SwitchSectionNoChange(xdataSect);

  OS << "\t.seh_handlerdata";
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHPushReg(unsigned Register) {
  MCStreamer::EmitWin64EHPushReg(Register);

  OS << "\t.seh_pushreg " << Register;
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHSetFrame(unsigned Register, unsigned Offset) {
  MCStreamer::EmitWin64EHSetFrame(Register, Offset);

  OS << "\t.seh_setframe " << Register << ", " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHAllocStack(unsigned Size) {
  MCStreamer::EmitWin64EHAllocStack(Size);

  OS << "\t.seh_stackalloc " << Size;
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHSaveReg(unsigned Register, unsigned Offset) {
  MCStreamer::EmitWin64EHSaveReg(Register, Offset);

  OS << "\t.seh_savereg " << Register << ", " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHSaveXMM(unsigned Register, unsigned Offset) {
  MCStreamer::EmitWin64EHSaveXMM(Register, Offset);

  OS << "\t.seh_savexmm " << Register << ", " << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHPushFrame(bool Code) {
  MCStreamer::EmitWin64EHPushFrame(Code);

  OS << "\t.seh_pushframe";
  if (Code)
    OS << " @code";
  EmitEOL();
}

void MCAsmStreamer::EmitWin64EHEndProlog(void) {
  MCStreamer::EmitWin64EHEndProlog();

  OS << "\t.seh_endprologue";
  EmitEOL();
}

void MCAsmStreamer::AddEncodingComment(const MCInst &Inst) {
  raw_ostream &OS = GetCommentOS();
  SmallString<256> Code;
  SmallVector<MCFixup, 4> Fixups;
  raw_svector_ostream VecOS(Code);
  Emitter->EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  // If we are showing fixups, create symbolic markers in the encoded
  // representation. We do this by making a per-bit map to the fixup item index,
  // then trying to display it as nicely as possible.
  SmallVector<uint8_t, 64> FixupMap;
  FixupMap.resize(Code.size() * 8);
  for (unsigned i = 0, e = Code.size() * 8; i != e; ++i)
    FixupMap[i] = 0;

  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    MCFixup &F = Fixups[i];
    const MCFixupKindInfo &Info = AsmBackend->getFixupKindInfo(F.getKind());
    for (unsigned j = 0; j != Info.TargetSize; ++j) {
      unsigned Index = F.getOffset() * 8 + Info.TargetOffset + j;
      assert(Index < Code.size() * 8 && "Invalid offset in fixup!");
      FixupMap[Index] = 1 + i;
    }
  }

  // FIXME: Note the fixup comments for Thumb2 are completely bogus since the
  // high order halfword of a 32-bit Thumb2 instruction is emitted first.
  OS << "encoding: [";
  for (unsigned i = 0, e = Code.size(); i != e; ++i) {
    if (i)
      OS << ',';

    // See if all bits are the same map entry.
    uint8_t MapEntry = FixupMap[i * 8 + 0];
    for (unsigned j = 1; j != 8; ++j) {
      if (FixupMap[i * 8 + j] == MapEntry)
        continue;

      MapEntry = uint8_t(~0U);
      break;
    }

    if (MapEntry != uint8_t(~0U)) {
      if (MapEntry == 0) {
        OS << format("0x%02x", uint8_t(Code[i]));
      } else {
        if (Code[i]) {
          // FIXME: Some of the 8 bits require fix up.
          OS << format("0x%02x", uint8_t(Code[i])) << '\''
             << char('A' + MapEntry - 1) << '\'';
        } else
          OS << char('A' + MapEntry - 1);
      }
    } else {
      // Otherwise, write out in binary.
      OS << "0b";
      for (unsigned j = 8; j--;) {
        unsigned Bit = (Code[i] >> j) & 1;

        unsigned FixupBit;
        if (getContext().getAsmInfo().isLittleEndian())
          FixupBit = i * 8 + j;
        else
          FixupBit = i * 8 + (7-j);

        if (uint8_t MapEntry = FixupMap[FixupBit]) {
          assert(Bit == 0 && "Encoder wrote into fixed up bit!");
          OS << char('A' + MapEntry - 1);
        } else
          OS << Bit;
      }
    }
  }
  OS << "]\n";

  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    MCFixup &F = Fixups[i];
    const MCFixupKindInfo &Info = AsmBackend->getFixupKindInfo(F.getKind());
    OS << "  fixup " << char('A' + i) << " - " << "offset: " << F.getOffset()
       << ", value: " << *F.getValue() << ", kind: " << Info.Name << "\n";
  }
}

void MCAsmStreamer::EmitFnStart() {
  OS << "\t.fnstart";
  EmitEOL();
}

void MCAsmStreamer::EmitFnEnd() {
  OS << "\t.fnend";
  EmitEOL();
}

void MCAsmStreamer::EmitCantUnwind() {
  OS << "\t.cantunwind";
  EmitEOL();
}

void MCAsmStreamer::EmitHandlerData() {
  OS << "\t.handlerdata";
  EmitEOL();
}

void MCAsmStreamer::EmitPersonality(const MCSymbol *Personality) {
  OS << "\t.personality " << Personality->getName();
  EmitEOL();
}

void MCAsmStreamer::EmitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset) {
  OS << "\t.setfp\t";
  InstPrinter->printRegName(OS, FpReg);
  OS << ", ";
  InstPrinter->printRegName(OS, SpReg);
  if (Offset)
    OS << ", #" << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitPad(int64_t Offset) {
  OS << "\t.pad\t#" << Offset;
  EmitEOL();
}

void MCAsmStreamer::EmitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                bool isVector) {
  assert(RegList.size() && "RegList should not be empty");
  if (isVector)
    OS << "\t.vsave\t{";
  else
    OS << "\t.save\t{";

  InstPrinter->printRegName(OS, RegList[0]);

  for (unsigned i = 1, e = RegList.size(); i != e; ++i) {
    OS << ", ";
    InstPrinter->printRegName(OS, RegList[i]);
  }

  OS << "}";
  EmitEOL();
}

void MCAsmStreamer::EmitTCEntry(const MCSymbol &S) {
  OS << "\t.tc ";
  OS << S.getName();
  OS << "[TC],";
  OS << S.getName();
  EmitEOL();
}

void MCAsmStreamer::EmitInstruction(const MCInst &Inst) {
  assert(getCurrentSection() && "Cannot emit contents before setting section!");

  // Show the encoding in a comment if we have a code emitter.
  if (Emitter)
    AddEncodingComment(Inst);

  // Show the MCInst if enabled.
  if (ShowInst) {
    Inst.dump_pretty(GetCommentOS(), &MAI, InstPrinter.get(), "\n ");
    GetCommentOS() << "\n";
  }

  // If we have an AsmPrinter, use that to print, otherwise print the MCInst.
  if (InstPrinter)
    InstPrinter->printInst(&Inst, OS, "");
  else
    Inst.print(OS, &MAI);
  EmitEOL();
}

/// EmitRawText - If this file is backed by an assembly streamer, this dumps
/// the specified string in the output .s file.  This capability is
/// indicated by the hasRawTextSupport() predicate.
void MCAsmStreamer::EmitRawText(StringRef String) {
  if (!String.empty() && String.back() == '\n')
    String = String.substr(0, String.size()-1);
  OS << String;
  EmitEOL();
}

void MCAsmStreamer::FinishImpl() {
  // FIXME: This header is duplicated with MCObjectStreamer
  // Dump out the dwarf file & directory tables and line tables.
  const MCSymbol *LineSectionSymbol = NULL;
  if (getContext().hasDwarfFiles() && !UseLoc)
    LineSectionSymbol = MCDwarfFileTable::Emit(this);

  // If we are generating dwarf for assembly source files dump out the sections.
  if (getContext().getGenDwarfForAssembly())
    MCGenDwarfInfo::Emit(this, LineSectionSymbol);

  if (!UseCFI)
    EmitFrames(false);
}
MCStreamer *llvm::createAsmStreamer(MCContext &Context,
                                    formatted_raw_ostream &OS,
                                    bool isVerboseAsm, bool useLoc,
                                    bool useCFI, bool useDwarfDirectory,
                                    MCInstPrinter *IP, MCCodeEmitter *CE,
                                    MCAsmBackend *MAB, bool ShowInst) {
  return new MCAsmStreamer(Context, OS, isVerboseAsm, useLoc, useCFI,
                           useDwarfDirectory, IP, CE, MAB, ShowInst);
}
