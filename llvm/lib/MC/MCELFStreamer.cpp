//===- lib/MC/MCELFStreamer.cpp - ELF Object Output -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits ELF .o object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFragment.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

MCELFStreamer::MCELFStreamer(MCContext &Context,
                             std::unique_ptr<MCAsmBackend> TAB,
                             std::unique_ptr<MCObjectWriter> OW,
                             std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(TAB), std::move(OW),
                       std::move(Emitter)) {}

bool MCELFStreamer::isBundleLocked() const {
  return getCurrentSectionOnly()->isBundleLocked();
}

void MCELFStreamer::mergeFragment(MCDataFragment *DF,
                                  MCDataFragment *EF) {
  MCAssembler &Assembler = getAssembler();

  if (Assembler.isBundlingEnabled() && Assembler.getRelaxAll()) {
    uint64_t FSize = EF->getContents().size();

    if (FSize > Assembler.getBundleAlignSize())
      report_fatal_error("Fragment can't be larger than a bundle size");

    uint64_t RequiredBundlePadding = computeBundlePadding(
        Assembler, EF, DF->getContents().size(), FSize);

    if (RequiredBundlePadding > UINT8_MAX)
      report_fatal_error("Padding cannot exceed 255 bytes");

    if (RequiredBundlePadding > 0) {
      SmallString<256> Code;
      raw_svector_ostream VecOS(Code);
      EF->setBundlePadding(static_cast<uint8_t>(RequiredBundlePadding));
      Assembler.writeFragmentPadding(VecOS, *EF, FSize);

      DF->getContents().append(Code.begin(), Code.end());
    }
  }

  flushPendingLabels(DF, DF->getContents().size());

  for (unsigned i = 0, e = EF->getFixups().size(); i != e; ++i) {
    EF->getFixups()[i].setOffset(EF->getFixups()[i].getOffset() +
                                 DF->getContents().size());
    DF->getFixups().push_back(EF->getFixups()[i]);
  }
  if (DF->getSubtargetInfo() == nullptr && EF->getSubtargetInfo())
    DF->setHasInstructions(*EF->getSubtargetInfo());
  DF->getContents().append(EF->getContents().begin(), EF->getContents().end());
}

void MCELFStreamer::initSections(bool NoExecStack, const MCSubtargetInfo &STI) {
  MCContext &Ctx = getContext();
  switchSection(Ctx.getObjectFileInfo()->getTextSection());
  emitCodeAlignment(Align(Ctx.getObjectFileInfo()->getTextSectionAlignment()),
                    &STI);

  if (NoExecStack)
    switchSection(Ctx.getAsmInfo()->getNonexecutableStackSection(Ctx));
}

void MCELFStreamer::emitLabel(MCSymbol *S, SMLoc Loc) {
  auto *Symbol = cast<MCSymbolELF>(S);
  MCObjectStreamer::emitLabel(Symbol, Loc);

  const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(*getCurrentSectionOnly());
  if (Section.getFlags() & ELF::SHF_TLS)
    Symbol->setType(ELF::STT_TLS);
}

void MCELFStreamer::emitLabelAtPos(MCSymbol *S, SMLoc Loc, MCFragment *F,
                                   uint64_t Offset) {
  auto *Symbol = cast<MCSymbolELF>(S);
  MCObjectStreamer::emitLabelAtPos(Symbol, Loc, F, Offset);

  const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(*getCurrentSectionOnly());
  if (Section.getFlags() & ELF::SHF_TLS)
    Symbol->setType(ELF::STT_TLS);
}

void MCELFStreamer::emitAssemblerFlag(MCAssemblerFlag Flag) {
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

  llvm_unreachable("invalid assembler flag!");
}

// If bundle alignment is used and there are any instructions in the section, it
// needs to be aligned to at least the bundle size.
static void setSectionAlignmentForBundling(const MCAssembler &Assembler,
                                           MCSection *Section) {
  if (Section && Assembler.isBundlingEnabled() && Section->hasInstructions())
    Section->ensureMinAlignment(Align(Assembler.getBundleAlignSize()));
}

void MCELFStreamer::changeSection(MCSection *Section,
                                  const MCExpr *Subsection) {
  MCSection *CurSection = getCurrentSectionOnly();
  if (CurSection && isBundleLocked())
    report_fatal_error("Unterminated .bundle_lock when changing a section");

  MCAssembler &Asm = getAssembler();
  // Ensure the previous section gets aligned if necessary.
  setSectionAlignmentForBundling(Asm, CurSection);
  auto *SectionELF = static_cast<const MCSectionELF *>(Section);
  const MCSymbol *Grp = SectionELF->getGroup();
  if (Grp)
    Asm.registerSymbol(*Grp);
  if (SectionELF->getFlags() & ELF::SHF_GNU_RETAIN)
    Asm.getWriter().markGnuAbi();

  changeSectionImpl(Section, Subsection);
  Asm.registerSymbol(*Section->getBeginSymbol());
}

void MCELFStreamer::emitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) {
  getAssembler().registerSymbol(*Symbol);
  const MCExpr *Value = MCSymbolRefExpr::create(
      Symbol, MCSymbolRefExpr::VK_WEAKREF, getContext());
  Alias->setVariableValue(Value);
}

// When GNU as encounters more than one .type declaration for an object it seems
// to use a mechanism similar to the one below to decide which type is actually
// used in the object file.  The greater of T1 and T2 is selected based on the
// following ordering:
//  STT_NOTYPE < STT_OBJECT < STT_FUNC < STT_GNU_IFUNC < STT_TLS < anything else
// If neither T1 < T2 nor T2 < T1 according to this ordering, use T2 (the user
// provided type).
static unsigned CombineSymbolTypes(unsigned T1, unsigned T2) {
  for (unsigned Type : {ELF::STT_NOTYPE, ELF::STT_OBJECT, ELF::STT_FUNC,
                        ELF::STT_GNU_IFUNC, ELF::STT_TLS}) {
    if (T1 == Type)
      return T2;
    if (T2 == Type)
      return T1;
  }

  return T2;
}

bool MCELFStreamer::emitSymbolAttribute(MCSymbol *S, MCSymbolAttr Attribute) {
  auto *Symbol = cast<MCSymbolELF>(S);

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling registerSymbol here is to register
  // the symbol with the assembler.
  getAssembler().registerSymbol(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCSA_Cold:
  case MCSA_Extern:
  case MCSA_LazyReference:
  case MCSA_Reference:
  case MCSA_SymbolResolver:
  case MCSA_PrivateExtern:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_Invalid:
  case MCSA_IndirectSymbol:
  case MCSA_Exported:
    return false;

  case MCSA_NoDeadStrip:
    // Ignore for now.
    break;

  case MCSA_ELF_TypeGnuUniqueObject:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_OBJECT));
    Symbol->setBinding(ELF::STB_GNU_UNIQUE);
    getAssembler().getWriter().markGnuAbi();
    break;

  case MCSA_Global:
    // For `.weak x; .global x`, GNU as sets the binding to STB_WEAK while we
    // traditionally set the binding to STB_GLOBAL. This is error-prone, so we
    // error on such cases. Note, we also disallow changed binding from .local.
    if (Symbol->isBindingSet() && Symbol->getBinding() != ELF::STB_GLOBAL)
      getContext().reportError(getStartTokLoc(),
                               Symbol->getName() +
                                   " changed binding to STB_GLOBAL");
    Symbol->setBinding(ELF::STB_GLOBAL);
    break;

  case MCSA_WeakReference:
  case MCSA_Weak:
    // For `.global x; .weak x`, both MC and GNU as set the binding to STB_WEAK.
    // We emit a warning for now but may switch to an error in the future.
    if (Symbol->isBindingSet() && Symbol->getBinding() != ELF::STB_WEAK)
      getContext().reportWarning(
          getStartTokLoc(), Symbol->getName() + " changed binding to STB_WEAK");
    Symbol->setBinding(ELF::STB_WEAK);
    break;

  case MCSA_Local:
    if (Symbol->isBindingSet() && Symbol->getBinding() != ELF::STB_LOCAL)
      getContext().reportError(getStartTokLoc(),
                               Symbol->getName() +
                                   " changed binding to STB_LOCAL");
    Symbol->setBinding(ELF::STB_LOCAL);
    break;

  case MCSA_ELF_TypeFunction:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_FUNC));
    break;

  case MCSA_ELF_TypeIndFunction:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_GNU_IFUNC));
    getAssembler().getWriter().markGnuAbi();
    break;

  case MCSA_ELF_TypeObject:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_OBJECT));
    break;

  case MCSA_ELF_TypeTLS:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_TLS));
    break;

  case MCSA_ELF_TypeCommon:
    // TODO: Emit these as a common symbol.
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_OBJECT));
    break;

  case MCSA_ELF_TypeNoType:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_NOTYPE));
    break;

  case MCSA_Protected:
    Symbol->setVisibility(ELF::STV_PROTECTED);
    break;

  case MCSA_Memtag:
    Symbol->setMemtag(true);
    break;

  case MCSA_Hidden:
    Symbol->setVisibility(ELF::STV_HIDDEN);
    break;

  case MCSA_Internal:
    Symbol->setVisibility(ELF::STV_INTERNAL);
    break;

  case MCSA_AltEntry:
    llvm_unreachable("ELF doesn't support the .alt_entry attribute");

  case MCSA_LGlobal:
    llvm_unreachable("ELF doesn't support the .lglobl attribute");
  }

  return true;
}

void MCELFStreamer::emitCommonSymbol(MCSymbol *S, uint64_t Size,
                                     Align ByteAlignment) {
  auto *Symbol = cast<MCSymbolELF>(S);
  getAssembler().registerSymbol(*Symbol);

  if (!Symbol->isBindingSet())
    Symbol->setBinding(ELF::STB_GLOBAL);

  Symbol->setType(ELF::STT_OBJECT);

  if (Symbol->getBinding() == ELF::STB_LOCAL) {
    MCSection &Section = *getAssembler().getContext().getELFSection(
        ".bss", ELF::SHT_NOBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);
    MCSectionSubPair P = getCurrentSection();
    switchSection(&Section);

    emitValueToAlignment(ByteAlignment, 0, 1, 0);
    emitLabel(Symbol);
    emitZeros(Size);

    switchSection(P.first, P.second);
  } else {
    if (Symbol->declareCommon(Size, ByteAlignment))
      report_fatal_error(Twine("Symbol: ") + Symbol->getName() +
                         " redeclared as different type");
  }

  cast<MCSymbolELF>(Symbol)
      ->setSize(MCConstantExpr::create(Size, getContext()));
}

void MCELFStreamer::emitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  cast<MCSymbolELF>(Symbol)->setSize(Value);
}

void MCELFStreamer::emitELFSymverDirective(const MCSymbol *OriginalSym,
                                           StringRef Name,
                                           bool KeepOriginalSym) {
  getAssembler().Symvers.push_back(MCAssembler::Symver{
      getStartTokLoc(), OriginalSym, Name, KeepOriginalSym});
}

void MCELFStreamer::emitLocalCommonSymbol(MCSymbol *S, uint64_t Size,
                                          Align ByteAlignment) {
  auto *Symbol = cast<MCSymbolELF>(S);
  // FIXME: Should this be caught and done earlier?
  getAssembler().registerSymbol(*Symbol);
  Symbol->setBinding(ELF::STB_LOCAL);
  emitCommonSymbol(Symbol, Size, ByteAlignment);
}

void MCELFStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                  SMLoc Loc) {
  if (isBundleLocked())
    report_fatal_error("Emitting values inside a locked bundle is forbidden");
  fixSymbolsInTLSFixups(Value);
  MCObjectStreamer::emitValueImpl(Value, Size, Loc);
}

void MCELFStreamer::emitValueToAlignment(Align Alignment, int64_t Value,
                                         unsigned ValueSize,
                                         unsigned MaxBytesToEmit) {
  if (isBundleLocked())
    report_fatal_error("Emitting values inside a locked bundle is forbidden");
  MCObjectStreamer::emitValueToAlignment(Alignment, Value, ValueSize,
                                         MaxBytesToEmit);
}

void MCELFStreamer::emitCGProfileEntry(const MCSymbolRefExpr *From,
                                       const MCSymbolRefExpr *To,
                                       uint64_t Count) {
  getAssembler().CGProfile.push_back({From, To, Count});
}

void MCELFStreamer::emitIdent(StringRef IdentString) {
  MCSection *Comment = getAssembler().getContext().getELFSection(
      ".comment", ELF::SHT_PROGBITS, ELF::SHF_MERGE | ELF::SHF_STRINGS, 1);
  pushSection();
  switchSection(Comment);
  if (!SeenIdent) {
    emitInt8(0);
    SeenIdent = true;
  }
  emitBytes(IdentString);
  emitInt8(0);
  popSection();
}

void MCELFStreamer::fixSymbolsInTLSFixups(const MCExpr *expr) {
  switch (expr->getKind()) {
  case MCExpr::Target:
    cast<MCTargetExpr>(expr)->fixELFSymbolsInTLSFixups(getAssembler());
    break;
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
    case MCSymbolRefExpr::VK_TLSCALL:
    case MCSymbolRefExpr::VK_TLSDESC:
    case MCSymbolRefExpr::VK_TLSGD:
    case MCSymbolRefExpr::VK_TLSLD:
    case MCSymbolRefExpr::VK_TLSLDM:
    case MCSymbolRefExpr::VK_TPOFF:
    case MCSymbolRefExpr::VK_TPREL:
    case MCSymbolRefExpr::VK_DTPOFF:
    case MCSymbolRefExpr::VK_DTPREL:
    case MCSymbolRefExpr::VK_PPC_DTPMOD:
    case MCSymbolRefExpr::VK_PPC_TPREL_LO:
    case MCSymbolRefExpr::VK_PPC_TPREL_HI:
    case MCSymbolRefExpr::VK_PPC_TPREL_HA:
    case MCSymbolRefExpr::VK_PPC_TPREL_HIGH:
    case MCSymbolRefExpr::VK_PPC_TPREL_HIGHA:
    case MCSymbolRefExpr::VK_PPC_TPREL_HIGHER:
    case MCSymbolRefExpr::VK_PPC_TPREL_HIGHERA:
    case MCSymbolRefExpr::VK_PPC_TPREL_HIGHEST:
    case MCSymbolRefExpr::VK_PPC_TPREL_HIGHESTA:
    case MCSymbolRefExpr::VK_PPC_DTPREL_LO:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HI:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HA:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HIGH:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HIGHA:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HIGHER:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HIGHERA:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HIGHEST:
    case MCSymbolRefExpr::VK_PPC_DTPREL_HIGHESTA:
    case MCSymbolRefExpr::VK_PPC_GOT_TPREL:
    case MCSymbolRefExpr::VK_PPC_GOT_TPREL_LO:
    case MCSymbolRefExpr::VK_PPC_GOT_TPREL_HI:
    case MCSymbolRefExpr::VK_PPC_GOT_TPREL_HA:
    case MCSymbolRefExpr::VK_PPC_GOT_TPREL_PCREL:
    case MCSymbolRefExpr::VK_PPC_GOT_DTPREL:
    case MCSymbolRefExpr::VK_PPC_GOT_DTPREL_LO:
    case MCSymbolRefExpr::VK_PPC_GOT_DTPREL_HI:
    case MCSymbolRefExpr::VK_PPC_GOT_DTPREL_HA:
    case MCSymbolRefExpr::VK_PPC_TLS:
    case MCSymbolRefExpr::VK_PPC_TLS_PCREL:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSGD:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSGD_LO:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HI:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HA:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSGD_PCREL:
    case MCSymbolRefExpr::VK_PPC_TLSGD:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSLD:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSLD_LO:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HI:
    case MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HA:
    case MCSymbolRefExpr::VK_PPC_TLSLD:
      break;
    }
    getAssembler().registerSymbol(symRef.getSymbol());
    cast<MCSymbolELF>(symRef.getSymbol()).setType(ELF::STT_TLS);
    break;
  }

  case MCExpr::Unary:
    fixSymbolsInTLSFixups(cast<MCUnaryExpr>(expr)->getSubExpr());
    break;
  }
}

void MCELFStreamer::finalizeCGProfileEntry(const MCSymbolRefExpr *&SRE,
                                           uint64_t Offset) {
  const MCSymbol *S = &SRE->getSymbol();
  if (S->isTemporary()) {
    if (!S->isInSection()) {
      getContext().reportError(
          SRE->getLoc(), Twine("Reference to undefined temporary symbol ") +
                             "`" + S->getName() + "`");
      return;
    }
    S = S->getSection().getBeginSymbol();
    S->setUsedInReloc();
    SRE = MCSymbolRefExpr::create(S, MCSymbolRefExpr::VK_None, getContext(),
                                  SRE->getLoc());
  }
  const MCConstantExpr *MCOffset = MCConstantExpr::create(Offset, getContext());
  MCObjectStreamer::visitUsedExpr(*SRE);
  if (std::optional<std::pair<bool, std::string>> Err =
          MCObjectStreamer::emitRelocDirective(
              *MCOffset, "BFD_RELOC_NONE", SRE, SRE->getLoc(),
              *getContext().getSubtargetInfo()))
    report_fatal_error("Relocation for CG Profile could not be created: " +
                       Twine(Err->second));
}

void MCELFStreamer::finalizeCGProfile() {
  MCAssembler &Asm = getAssembler();
  if (Asm.CGProfile.empty())
    return;
  MCSection *CGProfile = getAssembler().getContext().getELFSection(
      ".llvm.call-graph-profile", ELF::SHT_LLVM_CALL_GRAPH_PROFILE,
      ELF::SHF_EXCLUDE, /*sizeof(Elf_CGProfile_Impl<>)=*/8);
  pushSection();
  switchSection(CGProfile);
  uint64_t Offset = 0;
  for (MCAssembler::CGProfileEntry &E : Asm.CGProfile) {
    finalizeCGProfileEntry(E.From, Offset);
    finalizeCGProfileEntry(E.To, Offset);
    emitIntValue(E.Count, sizeof(uint64_t));
    Offset += sizeof(uint64_t);
  }
  popSection();
}

void MCELFStreamer::emitInstToFragment(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  this->MCObjectStreamer::emitInstToFragment(Inst, STI);
  MCRelaxableFragment &F = *cast<MCRelaxableFragment>(getCurrentFragment());

  for (auto &Fixup : F.getFixups())
    fixSymbolsInTLSFixups(Fixup.getValue());
}

// A fragment can only have one Subtarget, and when bundling is enabled we
// sometimes need to use the same fragment. We give an error if there
// are conflicting Subtargets.
static void CheckBundleSubtargets(const MCSubtargetInfo *OldSTI,
                                  const MCSubtargetInfo *NewSTI) {
  if (OldSTI && NewSTI && OldSTI != NewSTI)
    report_fatal_error("A Bundle can only have one Subtarget.");
}

void MCELFStreamer::emitInstToData(const MCInst &Inst,
                                   const MCSubtargetInfo &STI) {
  MCAssembler &Assembler = getAssembler();
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  Assembler.getEmitter().encodeInstruction(Inst, VecOS, Fixups, STI);

  for (auto &Fixup : Fixups)
    fixSymbolsInTLSFixups(Fixup.getValue());

  // There are several possibilities here:
  //
  // If bundling is disabled, append the encoded instruction to the current data
  // fragment (or create a new such fragment if the current fragment is not a
  // data fragment, or the Subtarget has changed).
  //
  // If bundling is enabled:
  // - If we're not in a bundle-locked group, emit the instruction into a
  //   fragment of its own. If there are no fixups registered for the
  //   instruction, emit a MCCompactEncodedInstFragment. Otherwise, emit a
  //   MCDataFragment.
  // - If we're in a bundle-locked group, append the instruction to the current
  //   data fragment because we want all the instructions in a group to get into
  //   the same fragment. Be careful not to do that for the first instruction in
  //   the group, though.
  MCDataFragment *DF;

  if (Assembler.isBundlingEnabled()) {
    MCSection &Sec = *getCurrentSectionOnly();
    if (Assembler.getRelaxAll() && isBundleLocked()) {
      // If the -mc-relax-all flag is used and we are bundle-locked, we re-use
      // the current bundle group.
      DF = BundleGroups.back();
      CheckBundleSubtargets(DF->getSubtargetInfo(), &STI);
    }
    else if (Assembler.getRelaxAll() && !isBundleLocked())
      // When not in a bundle-locked group and the -mc-relax-all flag is used,
      // we create a new temporary fragment which will be later merged into
      // the current fragment.
      DF = new MCDataFragment();
    else if (isBundleLocked() && !Sec.isBundleGroupBeforeFirstInst()) {
      // If we are bundle-locked, we re-use the current fragment.
      // The bundle-locking directive ensures this is a new data fragment.
      DF = cast<MCDataFragment>(getCurrentFragment());
      CheckBundleSubtargets(DF->getSubtargetInfo(), &STI);
    }
    else if (!isBundleLocked() && Fixups.size() == 0) {
      // Optimize memory usage by emitting the instruction to a
      // MCCompactEncodedInstFragment when not in a bundle-locked group and
      // there are no fixups registered.
      MCCompactEncodedInstFragment *CEIF = new MCCompactEncodedInstFragment();
      insert(CEIF);
      CEIF->getContents().append(Code.begin(), Code.end());
      CEIF->setHasInstructions(STI);
      return;
    } else {
      DF = new MCDataFragment();
      insert(DF);
    }
    if (Sec.getBundleLockState() == MCSection::BundleLockedAlignToEnd) {
      // If this fragment is for a group marked "align_to_end", set a flag
      // in the fragment. This can happen after the fragment has already been
      // created if there are nested bundle_align groups and an inner one
      // is the one marked align_to_end.
      DF->setAlignToBundleEnd(true);
    }

    // We're now emitting an instruction in a bundle group, so this flag has
    // to be turned off.
    Sec.setBundleGroupBeforeFirstInst(false);
  } else {
    DF = getOrCreateDataFragment(&STI);
  }

  // Add the fixups and data.
  for (auto &Fixup : Fixups) {
    Fixup.setOffset(Fixup.getOffset() + DF->getContents().size());
    DF->getFixups().push_back(Fixup);
  }

  DF->setHasInstructions(STI);
  DF->getContents().append(Code.begin(), Code.end());

  if (Assembler.isBundlingEnabled() && Assembler.getRelaxAll()) {
    if (!isBundleLocked()) {
      mergeFragment(getOrCreateDataFragment(&STI), DF);
      delete DF;
    }
  }
}

void MCELFStreamer::emitBundleAlignMode(Align Alignment) {
  assert(Log2(Alignment) <= 30 && "Invalid bundle alignment");
  MCAssembler &Assembler = getAssembler();
  if (Alignment > 1 && (Assembler.getBundleAlignSize() == 0 ||
                        Assembler.getBundleAlignSize() == Alignment.value()))
    Assembler.setBundleAlignSize(Alignment.value());
  else
    report_fatal_error(".bundle_align_mode cannot be changed once set");
}

void MCELFStreamer::emitBundleLock(bool AlignToEnd) {
  MCSection &Sec = *getCurrentSectionOnly();

  if (!getAssembler().isBundlingEnabled())
    report_fatal_error(".bundle_lock forbidden when bundling is disabled");

  if (!isBundleLocked())
    Sec.setBundleGroupBeforeFirstInst(true);

  if (getAssembler().getRelaxAll() && !isBundleLocked()) {
    // TODO: drop the lock state and set directly in the fragment
    MCDataFragment *DF = new MCDataFragment();
    BundleGroups.push_back(DF);
  }

  Sec.setBundleLockState(AlignToEnd ? MCSection::BundleLockedAlignToEnd
                                    : MCSection::BundleLocked);
}

void MCELFStreamer::emitBundleUnlock() {
  MCSection &Sec = *getCurrentSectionOnly();

  if (!getAssembler().isBundlingEnabled())
    report_fatal_error(".bundle_unlock forbidden when bundling is disabled");
  else if (!isBundleLocked())
    report_fatal_error(".bundle_unlock without matching lock");
  else if (Sec.isBundleGroupBeforeFirstInst())
    report_fatal_error("Empty bundle-locked group is forbidden");

  // When the -mc-relax-all flag is used, we emit instructions to fragments
  // stored on a stack. When the bundle unlock is emitted, we pop a fragment
  // from the stack a merge it to the one below.
  if (getAssembler().getRelaxAll()) {
    assert(!BundleGroups.empty() && "There are no bundle groups");
    MCDataFragment *DF = BundleGroups.back();

    // FIXME: Use BundleGroups to track the lock state instead.
    Sec.setBundleLockState(MCSection::NotBundleLocked);

    // FIXME: Use more separate fragments for nested groups.
    if (!isBundleLocked()) {
      mergeFragment(getOrCreateDataFragment(DF->getSubtargetInfo()), DF);
      BundleGroups.pop_back();
      delete DF;
    }

    if (Sec.getBundleLockState() != MCSection::BundleLockedAlignToEnd)
      getOrCreateDataFragment()->setAlignToBundleEnd(false);
  } else
    Sec.setBundleLockState(MCSection::NotBundleLocked);
}

void MCELFStreamer::finishImpl() {
  // Emit the .gnu attributes section if any attributes have been added.
  if (!GNUAttributes.empty()) {
    MCSection *DummyAttributeSection = nullptr;
    createAttributesSection("gnu", ".gnu.attributes", ELF::SHT_GNU_ATTRIBUTES,
                            DummyAttributeSection, GNUAttributes);
  }

  // Ensure the last section gets aligned if necessary.
  MCSection *CurSection = getCurrentSectionOnly();
  setSectionAlignmentForBundling(getAssembler(), CurSection);

  finalizeCGProfile();
  emitFrames(nullptr);

  this->MCObjectStreamer::finishImpl();
}

void MCELFStreamer::emitThumbFunc(MCSymbol *Func) {
  llvm_unreachable("Generic ELF doesn't support this directive");
}

void MCELFStreamer::emitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  llvm_unreachable("ELF doesn't support this directive");
}

void MCELFStreamer::emitZerofill(MCSection *Section, MCSymbol *Symbol,
                                 uint64_t Size, Align ByteAlignment,
                                 SMLoc Loc) {
  llvm_unreachable("ELF doesn't support this directive");
}

void MCELFStreamer::emitTBSSSymbol(MCSection *Section, MCSymbol *Symbol,
                                   uint64_t Size, Align ByteAlignment) {
  llvm_unreachable("ELF doesn't support this directive");
}

void MCELFStreamer::setAttributeItem(unsigned Attribute, unsigned Value,
                                     bool OverwriteExisting) {
  // Look for existing attribute item
  if (AttributeItem *Item = getAttributeItem(Attribute)) {
    if (!OverwriteExisting)
      return;
    Item->Type = AttributeItem::NumericAttribute;
    Item->IntValue = Value;
    return;
  }

  // Create new attribute item
  AttributeItem Item = {AttributeItem::NumericAttribute, Attribute, Value,
                        std::string(StringRef(""))};
  Contents.push_back(Item);
}

void MCELFStreamer::setAttributeItem(unsigned Attribute, StringRef Value,
                                     bool OverwriteExisting) {
  // Look for existing attribute item
  if (AttributeItem *Item = getAttributeItem(Attribute)) {
    if (!OverwriteExisting)
      return;
    Item->Type = AttributeItem::TextAttribute;
    Item->StringValue = std::string(Value);
    return;
  }

  // Create new attribute item
  AttributeItem Item = {AttributeItem::TextAttribute, Attribute, 0,
                        std::string(Value)};
  Contents.push_back(Item);
}

void MCELFStreamer::setAttributeItems(unsigned Attribute, unsigned IntValue,
                                      StringRef StringValue,
                                      bool OverwriteExisting) {
  // Look for existing attribute item
  if (AttributeItem *Item = getAttributeItem(Attribute)) {
    if (!OverwriteExisting)
      return;
    Item->Type = AttributeItem::NumericAndTextAttributes;
    Item->IntValue = IntValue;
    Item->StringValue = std::string(StringValue);
    return;
  }

  // Create new attribute item
  AttributeItem Item = {AttributeItem::NumericAndTextAttributes, Attribute,
                        IntValue, std::string(StringValue)};
  Contents.push_back(Item);
}

MCELFStreamer::AttributeItem *
MCELFStreamer::getAttributeItem(unsigned Attribute) {
  for (size_t I = 0; I < Contents.size(); ++I)
    if (Contents[I].Tag == Attribute)
      return &Contents[I];
  return nullptr;
}

size_t
MCELFStreamer::calculateContentSize(SmallVector<AttributeItem, 64> &AttrsVec) {
  size_t Result = 0;
  for (size_t I = 0; I < AttrsVec.size(); ++I) {
    AttributeItem Item = AttrsVec[I];
    switch (Item.Type) {
    case AttributeItem::HiddenAttribute:
      break;
    case AttributeItem::NumericAttribute:
      Result += getULEB128Size(Item.Tag);
      Result += getULEB128Size(Item.IntValue);
      break;
    case AttributeItem::TextAttribute:
      Result += getULEB128Size(Item.Tag);
      Result += Item.StringValue.size() + 1; // string + '\0'
      break;
    case AttributeItem::NumericAndTextAttributes:
      Result += getULEB128Size(Item.Tag);
      Result += getULEB128Size(Item.IntValue);
      Result += Item.StringValue.size() + 1; // string + '\0';
      break;
    }
  }
  return Result;
}

void MCELFStreamer::createAttributesSection(
    StringRef Vendor, const Twine &Section, unsigned Type,
    MCSection *&AttributeSection, SmallVector<AttributeItem, 64> &AttrsVec) {
  // <format-version>
  // [ <section-length> "vendor-name"
  // [ <file-tag> <size> <attribute>*
  //   | <section-tag> <size> <section-number>* 0 <attribute>*
  //   | <symbol-tag> <size> <symbol-number>* 0 <attribute>*
  //   ]+
  // ]*

  // Switch section to AttributeSection or get/create the section.
  if (AttributeSection) {
    switchSection(AttributeSection);
  } else {
    AttributeSection = getContext().getELFSection(Section, Type, 0);
    switchSection(AttributeSection);

    // Format version
    emitInt8(0x41);
  }

  // Vendor size + Vendor name + '\0'
  const size_t VendorHeaderSize = 4 + Vendor.size() + 1;

  // Tag + Tag Size
  const size_t TagHeaderSize = 1 + 4;

  const size_t ContentsSize = calculateContentSize(AttrsVec);

  emitInt32(VendorHeaderSize + TagHeaderSize + ContentsSize);
  emitBytes(Vendor);
  emitInt8(0); // '\0'

  emitInt8(ARMBuildAttrs::File);
  emitInt32(TagHeaderSize + ContentsSize);

  // Size should have been accounted for already, now
  // emit each field as its type (ULEB or String)
  for (size_t I = 0; I < AttrsVec.size(); ++I) {
    AttributeItem Item = AttrsVec[I];
    emitULEB128IntValue(Item.Tag);
    switch (Item.Type) {
    default:
      llvm_unreachable("Invalid attribute type");
    case AttributeItem::NumericAttribute:
      emitULEB128IntValue(Item.IntValue);
      break;
    case AttributeItem::TextAttribute:
      emitBytes(Item.StringValue);
      emitInt8(0); // '\0'
      break;
    case AttributeItem::NumericAndTextAttributes:
      emitULEB128IntValue(Item.IntValue);
      emitBytes(Item.StringValue);
      emitInt8(0); // '\0'
      break;
    }
  }

  AttrsVec.clear();
}

MCStreamer *llvm::createELFStreamer(MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&CE,
                                    bool RelaxAll) {
  MCELFStreamer *S =
      new MCELFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
