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
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
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
#include <cassert>
#include <cstdint>

using namespace llvm;

MCELFStreamer::MCELFStreamer(MCContext &Context,
                             std::unique_ptr<MCAsmBackend> TAB,
                             std::unique_ptr<MCObjectWriter> OW,
                             std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(TAB), std::move(OW),
                       std::move(Emitter)) {}

ELFObjectWriter &MCELFStreamer::getWriter() {
  return static_cast<ELFObjectWriter &>(getAssembler().getWriter());
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

void MCELFStreamer::emitLabelAtPos(MCSymbol *S, SMLoc Loc, MCFragment &F,
                                   uint64_t Offset) {
  auto *Symbol = cast<MCSymbolELF>(S);
  MCObjectStreamer::emitLabelAtPos(Symbol, Loc, F, Offset);

  const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(*getCurrentSectionOnly());
  if (Section.getFlags() & ELF::SHF_TLS)
    Symbol->setType(ELF::STT_TLS);
}

void MCELFStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  MCAssembler &Asm = getAssembler();
  auto *SectionELF = static_cast<const MCSectionELF *>(Section);
  const MCSymbol *Grp = SectionELF->getGroup();
  if (Grp)
    Asm.registerSymbol(*Grp);
  if (SectionELF->getFlags() & ELF::SHF_GNU_RETAIN)
    getWriter().markGnuAbi();

  MCObjectStreamer::changeSection(Section, Subsection);
  Asm.registerSymbol(*Section->getBeginSymbol());
}

void MCELFStreamer::emitWeakReference(MCSymbol *Alias, const MCSymbol *Target) {
  auto *A = cast<MCSymbolELF>(Alias);
  if (A->isDefined()) {
    getContext().reportError(getStartTokLoc(), "symbol '" + A->getName() +
                                                   "' is already defined");
    return;
  }
  A->setVariableValue(MCSymbolRefExpr::create(Target, getContext()));
  A->setIsWeakref();
  getWriter().Weakrefs.push_back(A);
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
  case MCSA_WeakAntiDep:
    return false;

  case MCSA_NoDeadStrip:
    // Ignore for now.
    break;

  case MCSA_ELF_TypeGnuUniqueObject:
    Symbol->setType(CombineSymbolTypes(Symbol->getType(), ELF::STT_OBJECT));
    Symbol->setBinding(ELF::STB_GNU_UNIQUE);
    getWriter().markGnuAbi();
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
    getWriter().markGnuAbi();
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
  getWriter().Symvers.push_back(ELFObjectWriter::Symver{
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

void MCELFStreamer::emitCGProfileEntry(const MCSymbolRefExpr *From,
                                       const MCSymbolRefExpr *To,
                                       uint64_t Count) {
  getWriter().getCGProfile().push_back({From, To, Count});
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

void MCELFStreamer::finalizeCGProfileEntry(const MCSymbolRefExpr *Sym,
                                           uint64_t Offset,
                                           const MCSymbolRefExpr *&SRE) {
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
    SRE = MCSymbolRefExpr::create(S, getContext(), SRE->getLoc());
  }
  auto *O = MCBinaryExpr::createAdd(
      Sym, MCConstantExpr::create(Offset, getContext()), getContext());
  MCObjectStreamer::emitRelocDirective(*O, "BFD_RELOC_NONE", SRE);
}

void MCELFStreamer::finalizeCGProfile() {
  ELFObjectWriter &W = getWriter();
  if (W.getCGProfile().empty())
    return;
  MCSection *CGProfile = getAssembler().getContext().getELFSection(
      ".llvm.call-graph-profile", ELF::SHT_LLVM_CALL_GRAPH_PROFILE,
      ELF::SHF_EXCLUDE, /*sizeof(Elf_CGProfile_Impl<>)=*/8);
  pushSection();
  switchSection(CGProfile);
  uint64_t Offset = 0;
  auto *Sym =
      MCSymbolRefExpr::create(CGProfile->getBeginSymbol(), getContext());
  for (auto &E : W.getCGProfile()) {
    finalizeCGProfileEntry(Sym, Offset, E.From);
    finalizeCGProfileEntry(Sym, Offset, E.To);
    emitIntValue(E.Count, sizeof(uint64_t));
    Offset += sizeof(uint64_t);
  }
  popSection();
}

void MCELFStreamer::finishImpl() {
  // Emit the .gnu attributes section if any attributes have been added.
  if (!GNUAttributes.empty()) {
    MCSection *DummyAttributeSection = nullptr;
    createAttributesSection("gnu", ".gnu.attributes", ELF::SHT_GNU_ATTRIBUTES,
                            DummyAttributeSection, GNUAttributes);
  }

  finalizeCGProfile();
  emitFrames(nullptr);

  this->MCObjectStreamer::finishImpl();
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
  for (AttributeItem &Item : Contents)
    if (Item.Tag == Attribute)
      return &Item;
  return nullptr;
}

size_t MCELFStreamer::calculateContentSize(
    SmallVector<AttributeItem, 64> &AttrsVec) const {
  size_t Result = 0;
  for (const AttributeItem &Item : AttrsVec) {
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
  for (const AttributeItem &Item : AttrsVec) {
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

void MCELFStreamer::createAttributesWithSubsection(
    MCSection *&AttributeSection, const Twine &Section, unsigned Type,
    SmallVector<AttributeSubSection, 64> &SubSectionVec) {
  // <format-version: 'A'>
  // [ <uint32: subsection-length> NTBS: vendor-name
  //   <bytes: vendor-data>
  // ]*
  // vendor-data expends to:
  // <uint8: optional> <uint8: parameter type> <attribute>*
  if (0 == SubSectionVec.size()) {
    return;
  }

  // Switch section to AttributeSection or get/create the section.
  if (AttributeSection) {
    switchSection(AttributeSection);
  } else {
    AttributeSection = getContext().getELFSection(Section, Type, 0);
    switchSection(AttributeSection);

    // Format version
    emitInt8(0x41);
  }

  for (AttributeSubSection &SubSection : SubSectionVec) {
    // subsection-length + vendor-name + '\0'
    const size_t VendorHeaderSize = 4 + SubSection.VendorName.size() + 1;
    // optional + parameter-type
    const size_t VendorParameters = 1 + 1;
    const size_t ContentsSize = calculateContentSize(SubSection.Content);

    emitInt32(VendorHeaderSize + VendorParameters + ContentsSize);
    emitBytes(SubSection.VendorName);
    emitInt8(0); // '\0'
    emitInt8(SubSection.IsOptional);
    emitInt8(SubSection.ParameterType);

    for (AttributeItem &Item : SubSection.Content) {
      emitULEB128IntValue(Item.Tag);
      switch (Item.Type) {
      default:
        assert(0 && "Invalid attribute type");
        break;
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
  }
  SubSectionVec.clear();
}

MCStreamer *llvm::createELFStreamer(MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&CE) {
  MCELFStreamer *S =
      new MCELFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  return S;
}
