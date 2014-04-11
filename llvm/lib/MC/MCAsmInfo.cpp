//===-- MCAsmInfo.cpp - Asm Info -------------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Dwarf.h"
#include <cctype>
#include <cstring>
using namespace llvm;

MCAsmInfo::MCAsmInfo() {
  PointerSize = 4;
  CalleeSaveStackSlotSize = 4;

  IsLittleEndian = true;
  StackGrowsUp = false;
  HasSubsectionsViaSymbols = false;
  HasMachoZeroFillDirective = false;
  HasMachoTBSSDirective = false;
  HasStaticCtorDtorReferenceInStaticMode = false;
  LinkerRequiresNonEmptyDwarfLines = false;
  MaxInstLength = 4;
  MinInstAlignment = 1;
  DollarIsPC = false;
  SeparatorString = ";";
  CommentColumn = 40;
  CommentString = "#";
  LabelSuffix = ":";
  DebugLabelSuffix = ":";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".";
  LinkerPrivateGlobalPrefix = "";
  InlineAsmStart = "APP";
  InlineAsmEnd = "NO_APP";
  Code16Directive = ".code16";
  Code32Directive = ".code32";
  Code64Directive = ".code64";
  AssemblerDialect = 0;
  AllowAtInName = false;
  UseDataRegionDirectives = false;
  ZeroDirective = "\t.zero\t";
  AsciiDirective = "\t.ascii\t";
  AscizDirective = "\t.asciz\t";
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = "\t.quad\t";
  SunStyleELFSectionSwitchSyntax = false;
  UsesELFSectionDirectiveForBSS = false;
  AlignDirective = "\t.align\t";
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;
  GPRel64Directive = 0;
  GPRel32Directive = 0;
  GlobalDirective = "\t.globl\t";
  HasSetDirective = true;
  HasAggressiveSymbolFolding = true;
  COMMDirectiveAlignmentIsInBytes = true;
  LCOMMDirectiveAlignmentType = LCOMM::NoAlignment;
  HasDotTypeDotSizeDirective = true;
  HasSingleParameterDotFile = true;
  HasIdentDirective = false;
  HasNoDeadStrip = false;
  WeakRefDirective = 0;
  HasWeakDefDirective = false;
  HasWeakDefCanBeHiddenDirective = false;
  HasLinkOnceDirective = false;
  HiddenVisibilityAttr = MCSA_Hidden;
  HiddenDeclarationVisibilityAttr = MCSA_Hidden;
  ProtectedVisibilityAttr = MCSA_Protected;
  HasLEB128 = false;
  SupportsDebugInformation = false;
  ExceptionsType = ExceptionHandling::None;
  DwarfUsesRelocationsAcrossSections = true;
  DwarfFDESymbolsUseAbsDiff = false;
  DwarfRegNumForCFI = false;
  HasMicrosoftFastStdCallMangling = false;
  NeedsDwarfSectionOffsetDirective = false;
}

MCAsmInfo::~MCAsmInfo() {
}


unsigned MCAsmInfo::getULEB128Size(uint64_t Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

unsigned MCAsmInfo::getSLEB128Size(int64_t Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;

  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}

const MCExpr *
MCAsmInfo::getExprForPersonalitySymbol(const MCSymbol *Sym,
                                       unsigned Encoding,
                                       MCStreamer &Streamer) const {
  return getExprForFDESymbol(Sym, Encoding, Streamer);
}

const MCExpr *
MCAsmInfo::getExprForFDESymbol(const MCSymbol *Sym,
                               unsigned Encoding,
                               MCStreamer &Streamer) const {
  if (!(Encoding & dwarf::DW_EH_PE_pcrel))
    return MCSymbolRefExpr::Create(Sym, Streamer.getContext());

  MCContext &Context = Streamer.getContext();
  const MCExpr *Res = MCSymbolRefExpr::Create(Sym, Context);
  MCSymbol *PCSym = Context.CreateTempSymbol();
  Streamer.EmitLabel(PCSym);
  const MCExpr *PC = MCSymbolRefExpr::Create(PCSym, Context);
  return MCBinaryExpr::CreateSub(Res, PC, Context);
}
