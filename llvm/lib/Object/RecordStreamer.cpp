//===-- RecordStreamer.cpp - Record asm defined and used symbols ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RecordStreamer.h"
#include "llvm/MC/MCSymbol.h"
using namespace llvm;

void RecordStreamer::markDefined(const MCSymbol &Symbol) {
  State &S = Symbols[Symbol.getName()];
  switch (S) {
  case DefinedGlobal:
  case Global:
    S = DefinedGlobal;
    break;
  case NeverSeen:
  case Defined:
  case Used:
    S = Defined;
    break;
  case DefinedWeak:
    break;
  case UndefinedWeak:
    S = DefinedWeak;
  }
}

void RecordStreamer::markGlobal(const MCSymbol &Symbol,
                                MCSymbolAttr Attribute) {
  State &S = Symbols[Symbol.getName()];
  switch (S) {
  case DefinedGlobal:
  case Defined:
    S = (Attribute == MCSA_Weak) ? DefinedWeak : DefinedGlobal;
    break;

  case NeverSeen:
  case Global:
  case Used:
    S = (Attribute == MCSA_Weak) ? UndefinedWeak : Global;
    break;
  case UndefinedWeak:
  case DefinedWeak:
    break;
  }
}

void RecordStreamer::markUsed(const MCSymbol &Symbol) {
  State &S = Symbols[Symbol.getName()];
  switch (S) {
  case DefinedGlobal:
  case Defined:
  case Global:
  case DefinedWeak:
  case UndefinedWeak:
    break;

  case NeverSeen:
  case Used:
    S = Used;
    break;
  }
}

void RecordStreamer::visitUsedSymbol(const MCSymbol &Sym) { markUsed(Sym); }

RecordStreamer::const_iterator RecordStreamer::begin() {
  return Symbols.begin();
}

RecordStreamer::const_iterator RecordStreamer::end() { return Symbols.end(); }

RecordStreamer::RecordStreamer(MCContext &Context) : MCStreamer(Context) {}

void RecordStreamer::EmitInstruction(const MCInst &Inst,
                                     const MCSubtargetInfo &STI, bool) {
  MCStreamer::EmitInstruction(Inst, STI);
}

void RecordStreamer::EmitLabel(MCSymbol *Symbol, SMLoc Loc) {
  MCStreamer::EmitLabel(Symbol);
  markDefined(*Symbol);
}

void RecordStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  markDefined(*Symbol);
  MCStreamer::EmitAssignment(Symbol, Value);
}

bool RecordStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                         MCSymbolAttr Attribute) {
  if (Attribute == MCSA_Global || Attribute == MCSA_Weak)
    markGlobal(*Symbol, Attribute);
  if (Attribute == MCSA_LazyReference)
    markUsed(*Symbol);
  return true;
}

void RecordStreamer::EmitZerofill(MCSection *Section, MCSymbol *Symbol,
                                  uint64_t Size, unsigned ByteAlignment) {
  markDefined(*Symbol);
}

void RecordStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                      unsigned ByteAlignment) {
  markDefined(*Symbol);
}

void RecordStreamer::emitELFSymverDirective(MCSymbol *Alias,
                                            const MCSymbol *Aliasee) {
  SymverAliasMap[Aliasee].push_back(Alias);
}
