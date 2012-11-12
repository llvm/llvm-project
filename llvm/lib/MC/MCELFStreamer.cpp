//===- lib/MC/MCELFStreamer.cpp - ELF Object Output ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits ELF .o object files.
//
//===----------------------------------------------------------------------===//

#include "MCELF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class MCELFStreamer : public MCObjectStreamer {
public:
  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter)
    : MCObjectStreamer(Context, TAB, OS, Emitter) {}

  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                raw_ostream &OS, MCCodeEmitter *Emitter,
                MCAssembler *Assembler)
    : MCObjectStreamer(Context, TAB, OS, Emitter, Assembler) {}


  ~MCELFStreamer() {}

  /// @name MCStreamer Interface
  /// @{

  virtual void InitSections();
  virtual void ChangeSection(const MCSection *Section);
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitThumbFunc(MCSymbol *Func);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
    llvm_unreachable("ELF doesn't support this directive");
  }
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {
    llvm_unreachable("ELF doesn't support this directive");
  }

  virtual void EmitCOFFSymbolStorageClass(int StorageClass) {
    llvm_unreachable("ELF doesn't support this directive");
  }

  virtual void EmitCOFFSymbolType(int Type) {
    llvm_unreachable("ELF doesn't support this directive");
  }

  virtual void EndCOFFSymbolDef() {
    llvm_unreachable("ELF doesn't support this directive");
  }

  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
     MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
     SD.setSize(Value);
  }

  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment);

  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            uint64_t Size = 0, unsigned ByteAlignment = 0) {
    llvm_unreachable("ELF doesn't support this directive");
  }
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0) {
    llvm_unreachable("ELF doesn't support this directive");
  }
  virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                             unsigned AddrSpace);

  virtual void EmitFileDirective(StringRef Filename);

  virtual void EmitTCEntry(const MCSymbol &S);

  virtual void FinishImpl();

private:
  virtual void EmitInstToFragment(const MCInst &Inst);
  virtual void EmitInstToData(const MCInst &Inst);

  void fixSymbolsInTLSFixups(const MCExpr *expr);

  struct LocalCommon {
    MCSymbolData *SD;
    uint64_t Size;
    unsigned ByteAlignment;
  };
  std::vector<LocalCommon> LocalCommons;

  SmallPtrSet<MCSymbol *, 16> BindingExplicitlySet;
  /// @}
  void SetSection(StringRef Section, unsigned Type, unsigned Flags,
                  SectionKind Kind) {
    SwitchSection(getContext().getELFSection(Section, Type, Flags, Kind));
  }

  void SetSectionData() {
    SetSection(".data", ELF::SHT_PROGBITS,
               ELF::SHF_WRITE |ELF::SHF_ALLOC,
               SectionKind::getDataRel());
    EmitCodeAlignment(4, 0);
  }
  void SetSectionText() {
    SetSection(".text", ELF::SHT_PROGBITS,
               ELF::SHF_EXECINSTR |
               ELF::SHF_ALLOC, SectionKind::getText());
    EmitCodeAlignment(4, 0);
  }
  void SetSectionBss() {
    SetSection(".bss", ELF::SHT_NOBITS,
               ELF::SHF_WRITE |
               ELF::SHF_ALLOC, SectionKind::getBSS());
    EmitCodeAlignment(4, 0);
  }
};
}

void MCELFStreamer::InitSections() {
  // This emulates the same behavior of GNU as. This makes it easier
  // to compare the output as the major sections are in the same order.
  SetSectionText();
  SetSectionData();
  SetSectionBss();
  SetSectionText();
}

void MCELFStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCObjectStreamer::EmitLabel(Symbol);

  const MCSectionELF &Section =
    static_cast<const MCSectionELF&>(Symbol->getSection());
  MCSymbolData &SD = getAssembler().getSymbolData(*Symbol);
  if (Section.getFlags() & ELF::SHF_TLS)
    MCELF::SetType(SD, ELF::STT_TLS);
}

void MCELFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  case MCAF_SyntaxUnified: return; // no-op here.
  case MCAF_Code16: return; // Change parsing mode; no-op here.
  case MCAF_Code32: return; // Change parsing mode; no-op here.
  case MCAF_Code64: return; // Change parsing mode; no-op here.
  case MCAF_SubsectionsViaSymbols:
    getAssembler().setSubsectionsViaSymbols(true);
    return;
  }

  llvm_unreachable("invalid assembler flag!");
}

void MCELFStreamer::EmitThumbFunc(MCSymbol *Func) {
  // FIXME: Anything needed here to flag the function as thumb?

  getAssembler().setIsThumbFunc(Func);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Func);
  SD.setFlags(SD.getFlags() | ELF_Other_ThumbFunc);
}

void MCELFStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  // FIXME: Lift context changes into super class.
  getAssembler().getOrCreateSymbolData(*Symbol);
  Symbol->setVariableValue(AddValueSymbols(Value));
}

void MCELFStreamer::ChangeSection(const MCSection *Section) {
  const MCSymbol *Grp = static_cast<const MCSectionELF *>(Section)->getGroup();
  if (Grp)
    getAssembler().getOrCreateSymbolData(*Grp);
  this->MCObjectStreamer::ChangeSection(Section);
}

void MCELFStreamer::EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) {
  getAssembler().getOrCreateSymbolData(*Symbol);
  MCSymbolData &AliasSD = getAssembler().getOrCreateSymbolData(*Alias);
  AliasSD.setFlags(AliasSD.getFlags() | ELF_Other_Weakref);
  const MCExpr *Value = MCSymbolRefExpr::Create(Symbol, getContext());
  Alias->setVariableValue(Value);
}

void MCELFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  // Indirect symbols are handled differently, to match how 'as' handles
  // them. This makes writing matching .o files easier.
  if (Attribute == MCSA_IndirectSymbol) {
    // Note that we intentionally cannot use the symbol data here; this is
    // important for matching the string table that 'as' generates.
    IndirectSymbolData ISD;
    ISD.Symbol = Symbol;
    ISD.SectionData = getCurrentSectionData();
    getAssembler().getIndirectSymbols().push_back(ISD);
    return;
  }

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling getOrCreateSymbolData here is to register
  // the symbol with the assembler.
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCSA_LazyReference:
  case MCSA_Reference:
  case MCSA_SymbolResolver:
  case MCSA_PrivateExtern:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_Invalid:
  case MCSA_IndirectSymbol:
    llvm_unreachable("Invalid symbol attribute for ELF!");

  case MCSA_NoDeadStrip:
  case MCSA_ELF_TypeGnuUniqueObject:
    // Ignore for now.
    break;

  case MCSA_Global:
    MCELF::SetBinding(SD, ELF::STB_GLOBAL);
    SD.setExternal(true);
    BindingExplicitlySet.insert(Symbol);
    break;

  case MCSA_WeakReference:
  case MCSA_Weak:
    MCELF::SetBinding(SD, ELF::STB_WEAK);
    SD.setExternal(true);
    BindingExplicitlySet.insert(Symbol);
    break;

  case MCSA_Local:
    MCELF::SetBinding(SD, ELF::STB_LOCAL);
    SD.setExternal(false);
    BindingExplicitlySet.insert(Symbol);
    break;

  case MCSA_ELF_TypeFunction:
    MCELF::SetType(SD, ELF::STT_FUNC);
    break;

  case MCSA_ELF_TypeIndFunction:
    MCELF::SetType(SD, ELF::STT_GNU_IFUNC);
    break;

  case MCSA_ELF_TypeObject:
    MCELF::SetType(SD, ELF::STT_OBJECT);
    break;

  case MCSA_ELF_TypeTLS:
    MCELF::SetType(SD, ELF::STT_TLS);
    break;

  case MCSA_ELF_TypeCommon:
    MCELF::SetType(SD, ELF::STT_COMMON);
    break;

  case MCSA_ELF_TypeNoType:
    MCELF::SetType(SD, ELF::STT_NOTYPE);
    break;

  case MCSA_Protected:
    MCELF::SetVisibility(SD, ELF::STV_PROTECTED);
    break;

  case MCSA_Hidden:
    MCELF::SetVisibility(SD, ELF::STV_HIDDEN);
    break;

  case MCSA_Internal:
    MCELF::SetVisibility(SD, ELF::STV_INTERNAL);
    break;
  }
}

void MCELFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  if (!BindingExplicitlySet.count(Symbol)) {
    MCELF::SetBinding(SD, ELF::STB_GLOBAL);
    SD.setExternal(true);
  }

  MCELF::SetType(SD, ELF::STT_OBJECT);

  if (MCELF::GetBinding(SD) == ELF_STB_Local) {
    const MCSection *Section = getAssembler().getContext().getELFSection(".bss",
                                                         ELF::SHT_NOBITS,
                                                         ELF::SHF_WRITE |
                                                         ELF::SHF_ALLOC,
                                                         SectionKind::getBSS());
    Symbol->setSection(*Section);

    struct LocalCommon L = {&SD, Size, ByteAlignment};
    LocalCommons.push_back(L);
  } else {
    SD.setCommon(Size, ByteAlignment);
  }

  SD.setSize(MCConstantExpr::Create(Size, getContext()));
}

void MCELFStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                          unsigned ByteAlignment) {
  // FIXME: Should this be caught and done earlier?
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  MCELF::SetBinding(SD, ELF::STB_LOCAL);
  SD.setExternal(false);
  BindingExplicitlySet.insert(Symbol);
  EmitCommonSymbol(Symbol, Size, ByteAlignment);
}

void MCELFStreamer::EmitValueImpl(const MCExpr *Value, unsigned Size,
                                  unsigned AddrSpace) {
  fixSymbolsInTLSFixups(Value);
  MCObjectStreamer::EmitValueImpl(Value, Size, AddrSpace);
}


// Add a symbol for the file name of this module. This is the second
// entry in the module's symbol table (the first being the null symbol).
void MCELFStreamer::EmitFileDirective(StringRef Filename) {
  MCSymbol *Symbol = getAssembler().getContext().GetOrCreateSymbol(Filename);
  Symbol->setSection(*getCurrentSection());
  Symbol->setAbsolute();

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  SD.setFlags(ELF_STT_File | ELF_STB_Local | ELF_STV_Default);
}

void  MCELFStreamer::fixSymbolsInTLSFixups(const MCExpr *expr) {
  switch (expr->getKind()) {
  case MCExpr::Target: llvm_unreachable("Can't handle target exprs yet!");
  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *be = cast<MCBinaryExpr>(expr);
    fixSymbolsInTLSFixups(be->getLHS());
    fixSymbolsInTLSFixups(be->getRHS());
    break;
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &symRef = *cast<MCSymbolRefExpr>(expr);
    switch (symRef.getKind()) {
    default:
      return;
    case MCSymbolRefExpr::VK_GOTTPOFF:
    case MCSymbolRefExpr::VK_INDNTPOFF:
    case MCSymbolRefExpr::VK_NTPOFF:
    case MCSymbolRefExpr::VK_GOTNTPOFF:
    case MCSymbolRefExpr::VK_TLSGD:
    case MCSymbolRefExpr::VK_TLSLD:
    case MCSymbolRefExpr::VK_TLSLDM:
    case MCSymbolRefExpr::VK_TPOFF:
    case MCSymbolRefExpr::VK_DTPOFF:
    case MCSymbolRefExpr::VK_ARM_TLSGD:
    case MCSymbolRefExpr::VK_ARM_TPOFF:
    case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
    case MCSymbolRefExpr::VK_Mips_TLSGD:
    case MCSymbolRefExpr::VK_Mips_GOTTPREL:
    case MCSymbolRefExpr::VK_Mips_TPREL_HI:
    case MCSymbolRefExpr::VK_Mips_TPREL_LO:
      break;
    }
    MCSymbolData &SD = getAssembler().getOrCreateSymbolData(symRef.getSymbol());
    MCELF::SetType(SD, ELF::STT_TLS);
    break;
  }

  case MCExpr::Unary:
    fixSymbolsInTLSFixups(cast<MCUnaryExpr>(expr)->getSubExpr());
    break;
  }
}

void MCELFStreamer::EmitInstToFragment(const MCInst &Inst) {
  this->MCObjectStreamer::EmitInstToFragment(Inst);
  MCInstFragment &F = *cast<MCInstFragment>(getCurrentFragment());

  for (unsigned i = 0, e = F.getFixups().size(); i != e; ++i)
    fixSymbolsInTLSFixups(F.getFixups()[i].getValue());
}

void MCELFStreamer::EmitInstToData(const MCInst &Inst) {
  MCDataFragment *DF = getOrCreateDataFragment();

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  for (unsigned i = 0, e = Fixups.size(); i != e; ++i)
    fixSymbolsInTLSFixups(Fixups[i].getValue());

  // Add the fixups and data.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    Fixups[i].setOffset(Fixups[i].getOffset() + DF->getContents().size());
    DF->addFixup(Fixups[i]);
  }
  DF->getContents().append(Code.begin(), Code.end());
}

void MCELFStreamer::FinishImpl() {
  EmitFrames(true);

  for (std::vector<LocalCommon>::const_iterator i = LocalCommons.begin(),
                                                e = LocalCommons.end();
       i != e; ++i) {
    MCSymbolData *SD = i->SD;
    uint64_t Size = i->Size;
    unsigned ByteAlignment = i->ByteAlignment;
    const MCSymbol &Symbol = SD->getSymbol();
    const MCSection &Section = Symbol.getSection();

    MCSectionData &SectData = getAssembler().getOrCreateSectionData(Section);
    new MCAlignFragment(ByteAlignment, 0, 1, ByteAlignment, &SectData);

    MCFragment *F = new MCFillFragment(0, 0, Size, &SectData);
    SD->setFragment(F);

    // Update the maximum alignment of the section if necessary.
    if (ByteAlignment > SectData.getAlignment())
      SectData.setAlignment(ByteAlignment);
  }

  this->MCObjectStreamer::FinishImpl();
}

void MCELFStreamer::EmitTCEntry(const MCSymbol &S)
{
  // Creates a R_PPC64_TOC relocation
  MCObjectStreamer::EmitSymbolValue(&S, 8, 0);
}

MCStreamer *llvm::createELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                    raw_ostream &OS, MCCodeEmitter *CE,
                                    bool RelaxAll, bool NoExecStack) {
  MCELFStreamer *S = new MCELFStreamer(Context, MAB, OS, CE);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  if (NoExecStack)
    S->getAssembler().setNoExecStack(true);
  return S;
}
