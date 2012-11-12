//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCMachOSymbolFlags.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class MCMachOStreamer : public MCObjectStreamer {
private:
  virtual void EmitInstToData(const MCInst &Inst);

  void EmitDataRegion(DataRegionData::KindTy Kind);
  void EmitDataRegionEnd();
public:
  MCMachOStreamer(MCContext &Context, MCAsmBackend &MAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter)
    : MCObjectStreamer(Context, MAB, OS, Emitter) {}

  /// @name MCStreamer Interface
  /// @{

  virtual void InitSections();
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitEHSymAttributes(const MCSymbol *Symbol,
                                   MCSymbol *EHSymbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitDataRegion(MCDataRegionType Kind);
  virtual void EmitThumbFunc(MCSymbol *Func);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {
    llvm_unreachable("macho doesn't support this directive");
  }
  virtual void EmitCOFFSymbolStorageClass(int StorageClass) {
    llvm_unreachable("macho doesn't support this directive");
  }
  virtual void EmitCOFFSymbolType(int Type) {
    llvm_unreachable("macho doesn't support this directive");
  }
  virtual void EndCOFFSymbolDef() {
    llvm_unreachable("macho doesn't support this directive");
  }
  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
    llvm_unreachable("macho doesn't support this directive");
  }
  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment);
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            uint64_t Size = 0, unsigned ByteAlignment = 0);
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0);

  virtual void EmitFileDirective(StringRef Filename) {
    // FIXME: Just ignore the .file; it isn't important enough to fail the
    // entire assembly.

    //report_fatal_error("unsupported directive: '.file'");
  }

  virtual void FinishImpl();

  /// @}
};

} // end anonymous namespace.

void MCMachOStreamer::InitSections() {
  SwitchSection(getContext().getMachOSection("__TEXT", "__text",
                                    MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                    0, SectionKind::getText()));

}

void MCMachOStreamer::EmitEHSymAttributes(const MCSymbol *Symbol,
                                          MCSymbol *EHSymbol) {
  MCSymbolData &SD =
    getAssembler().getOrCreateSymbolData(*Symbol);
  if (SD.isExternal())
    EmitSymbolAttribute(EHSymbol, MCSA_Global);
  if (SD.getFlags() & SF_WeakDefinition)
    EmitSymbolAttribute(EHSymbol, MCSA_WeakDefinition);
  if (SD.isPrivateExtern())
    EmitSymbolAttribute(EHSymbol, MCSA_PrivateExtern);
}

void MCMachOStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  // isSymbolLinkerVisible uses the section.
  Symbol->setSection(*getCurrentSection());
  // We have to create a new fragment if this is an atom defining symbol,
  // fragments cannot span atoms.
  if (getAssembler().isSymbolLinkerVisible(*Symbol))
    new MCDataFragment(getCurrentSectionData());

  MCObjectStreamer::EmitLabel(Symbol);

  MCSymbolData &SD = getAssembler().getSymbolData(*Symbol);
  // This causes the reference type flag to be cleared. Darwin 'as' was "trying"
  // to clear the weak reference and weak definition bits too, but the
  // implementation was buggy. For now we just try to match 'as', for
  // diffability.
  //
  // FIXME: Cleanup this code, these bits should be emitted based on semantic
  // properties, not on the order of definition, etc.
  SD.setFlags(SD.getFlags() & ~SF_ReferenceTypeMask);
}

void MCMachOStreamer::EmitDataRegion(DataRegionData::KindTy Kind) {
  if (!getAssembler().getBackend().hasDataInCodeSupport())
    return;
  // Create a temporary label to mark the start of the data region.
  MCSymbol *Start = getContext().CreateTempSymbol();
  EmitLabel(Start);
  // Record the region for the object writer to use.
  DataRegionData Data = { Kind, Start, NULL };
  std::vector<DataRegionData> &Regions = getAssembler().getDataRegions();
  Regions.push_back(Data);
}

void MCMachOStreamer::EmitDataRegionEnd() {
  if (!getAssembler().getBackend().hasDataInCodeSupport())
    return;
  std::vector<DataRegionData> &Regions = getAssembler().getDataRegions();
  assert(Regions.size() && "Mismatched .end_data_region!");
  DataRegionData &Data = Regions.back();
  assert(Data.End == NULL && "Mismatched .end_data_region!");
  // Create a temporary label to mark the end of the data region.
  Data.End = getContext().CreateTempSymbol();
  EmitLabel(Data.End);
}

void MCMachOStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  // Let the target do whatever target specific stuff it needs to do.
  getAssembler().getBackend().handleAssemblerFlag(Flag);
  // Do any generic stuff we need to do.
  switch (Flag) {
  case MCAF_SyntaxUnified: return; // no-op here.
  case MCAF_Code16: return; // Change parsing mode; no-op here.
  case MCAF_Code32: return; // Change parsing mode; no-op here.
  case MCAF_Code64: return; // Change parsing mode; no-op here.
  case MCAF_SubsectionsViaSymbols:
    getAssembler().setSubsectionsViaSymbols(true);
    return;
  }
}

void MCMachOStreamer::EmitDataRegion(MCDataRegionType Kind) {
  switch (Kind) {
  case MCDR_DataRegion:
    EmitDataRegion(DataRegionData::Data);
    return;
  case MCDR_DataRegionJT8:
    EmitDataRegion(DataRegionData::JumpTable8);
    return;
  case MCDR_DataRegionJT16:
    EmitDataRegion(DataRegionData::JumpTable16);
    return;
  case MCDR_DataRegionJT32:
    EmitDataRegion(DataRegionData::JumpTable32);
    return;
  case MCDR_DataRegionEnd:
    EmitDataRegionEnd();
    return;
  }
}

void MCMachOStreamer::EmitThumbFunc(MCSymbol *Symbol) {
  // Remember that the function is a thumb function. Fixup and relocation
  // values will need adjusted.
  getAssembler().setIsThumbFunc(Symbol);

  // Mark the thumb bit on the symbol.
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  SD.setFlags(SD.getFlags() | SF_ThumbFunc);
}

void MCMachOStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  // FIXME: Lift context changes into super class.
  getAssembler().getOrCreateSymbolData(*Symbol);
  Symbol->setVariableValue(AddValueSymbols(Value));
}

void MCMachOStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
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
  case MCSA_Invalid:
  case MCSA_ELF_TypeFunction:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_ELF_TypeObject:
  case MCSA_ELF_TypeTLS:
  case MCSA_ELF_TypeCommon:
  case MCSA_ELF_TypeNoType:
  case MCSA_ELF_TypeGnuUniqueObject:
  case MCSA_Hidden:
  case MCSA_IndirectSymbol:
  case MCSA_Internal:
  case MCSA_Protected:
  case MCSA_Weak:
  case MCSA_Local:
    llvm_unreachable("Invalid symbol attribute for Mach-O!");

  case MCSA_Global:
    SD.setExternal(true);
    // This effectively clears the undefined lazy bit, in Darwin 'as', although
    // it isn't very consistent because it implements this as part of symbol
    // lookup.
    //
    // FIXME: Cleanup this code, these bits should be emitted based on semantic
    // properties, not on the order of definition, etc.
    SD.setFlags(SD.getFlags() & ~SF_ReferenceTypeUndefinedLazy);
    break;

  case MCSA_LazyReference:
    // FIXME: This requires -dynamic.
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_ReferenceTypeUndefinedLazy);
    break;

    // Since .reference sets the no dead strip bit, it is equivalent to
    // .no_dead_strip in practice.
  case MCSA_Reference:
  case MCSA_NoDeadStrip:
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    break;

  case MCSA_SymbolResolver:
    SD.setFlags(SD.getFlags() | SF_SymbolResolver);
    break;

  case MCSA_PrivateExtern:
    SD.setExternal(true);
    SD.setPrivateExtern(true);
    break;

  case MCSA_WeakReference:
    // FIXME: This requires -dynamic.
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_WeakReference);
    break;

  case MCSA_WeakDefinition:
    // FIXME: 'as' enforces that this is defined and global. The manual claims
    // it has to be in a coalesced section, but this isn't enforced.
    SD.setFlags(SD.getFlags() | SF_WeakDefinition);
    break;

  case MCSA_WeakDefAutoPrivate:
    SD.setFlags(SD.getFlags() | SF_WeakDefinition | SF_WeakReference);
    break;
  }
}

void MCMachOStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  // Encode the 'desc' value into the lowest implementation defined bits.
  assert(DescValue == (DescValue & SF_DescFlagsMask) &&
         "Invalid .desc value!");
  getAssembler().getOrCreateSymbolData(*Symbol).setFlags(
    DescValue & SF_DescFlagsMask);
}

void MCMachOStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  // FIXME: Darwin 'as' does appear to allow redef of a .comm by itself.
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  SD.setExternal(true);
  SD.setCommon(Size, ByteAlignment);
}

void MCMachOStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                            unsigned ByteAlignment) {
  // '.lcomm' is equivalent to '.zerofill'.
  return EmitZerofill(getContext().getMachOSection("__DATA", "__bss",
                                                   MCSectionMachO::S_ZEROFILL,
                                                   0, SectionKind::getBSS()),
                      Symbol, Size, ByteAlignment);
}

void MCMachOStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                   uint64_t Size, unsigned ByteAlignment) {
  MCSectionData &SectData = getAssembler().getOrCreateSectionData(*Section);

  // The symbol may not be present, which only creates the section.
  if (!Symbol)
    return;

  // FIXME: Assert that this section has the zerofill type.

  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // Emit an align fragment if necessary.
  if (ByteAlignment != 1)
    new MCAlignFragment(ByteAlignment, 0, 0, ByteAlignment, &SectData);

  MCFragment *F = new MCFillFragment(0, 0, Size, &SectData);
  SD.setFragment(F);

  Symbol->setSection(*Section);

  // Update the maximum alignment on the zero fill section if necessary.
  if (ByteAlignment > SectData.getAlignment())
    SectData.setAlignment(ByteAlignment);
}

// This should always be called with the thread local bss section.  Like the
// .zerofill directive this doesn't actually switch sections on us.
void MCMachOStreamer::EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                     uint64_t Size, unsigned ByteAlignment) {
  EmitZerofill(Section, Symbol, Size, ByteAlignment);
  return;
}

void MCMachOStreamer::EmitInstToData(const MCInst &Inst) {
  MCDataFragment *DF = getOrCreateDataFragment();

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  // Add the fixups and data.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    Fixups[i].setOffset(Fixups[i].getOffset() + DF->getContents().size());
    DF->addFixup(Fixups[i]);
  }
  DF->getContents().append(Code.begin(), Code.end());
}

void MCMachOStreamer::FinishImpl() {
  EmitFrames(true);

  // We have to set the fragment atom associations so we can relax properly for
  // Mach-O.

  // First, scan the symbol table to build a lookup table from fragments to
  // defining symbols.
  DenseMap<const MCFragment*, MCSymbolData*> DefiningSymbolMap;
  for (MCAssembler::symbol_iterator it = getAssembler().symbol_begin(),
         ie = getAssembler().symbol_end(); it != ie; ++it) {
    if (getAssembler().isSymbolLinkerVisible(it->getSymbol()) &&
        it->getFragment()) {
      // An atom defining symbol should never be internal to a fragment.
      assert(it->getOffset() == 0 && "Invalid offset in atom defining symbol!");
      DefiningSymbolMap[it->getFragment()] = it;
    }
  }

  // Set the fragment atom associations by tracking the last seen atom defining
  // symbol.
  for (MCAssembler::iterator it = getAssembler().begin(),
         ie = getAssembler().end(); it != ie; ++it) {
    MCSymbolData *CurrentAtom = 0;
    for (MCSectionData::iterator it2 = it->begin(),
           ie2 = it->end(); it2 != ie2; ++it2) {
      if (MCSymbolData *SD = DefiningSymbolMap.lookup(it2))
        CurrentAtom = SD;
      it2->setAtom(CurrentAtom);
    }
  }

  this->MCObjectStreamer::FinishImpl();
}

MCStreamer *llvm::createMachOStreamer(MCContext &Context, MCAsmBackend &MAB,
                                      raw_ostream &OS, MCCodeEmitter *CE,
                                      bool RelaxAll) {
  MCMachOStreamer *S = new MCMachOStreamer(Context, MAB, OS, CE);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
