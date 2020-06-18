//===- RecordStreamer.h - Record asm defined and used symbols ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJECT_RECORDSTREAMER_H
#define LLVM_LIB_OBJECT_RECORDSTREAMER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/SMLoc.h"
#include <vector>

namespace llvm {

class MCSymbol;
class Module;

class RecordStreamer : public MCStreamer {
public:
  enum State { NeverSeen, Global, Defined, DefinedGlobal, DefinedWeak, Used,
               UndefinedWeak};

private:
  const Module &M;
  StringMap<State> Symbols;
  // Map of aliases created by .symver directives, saved so we can update
  // their symbol binding after parsing complete. This maps from each
  // aliasee to its list of aliases.
  DenseMap<const MCSymbol *, std::vector<StringRef>> SymverAliasMap;

  /// Get the state recorded for the given symbol.
  State getSymbolState(const MCSymbol *Sym);

  void markDefined(const MCSymbol &Symbol);
  void markGlobal(const MCSymbol &Symbol, MCSymbolAttr Attribute);
  void markUsed(const MCSymbol &Symbol);
  void visitUsedSymbol(const MCSymbol &Sym) override;

public:
  RecordStreamer(MCContext &Context, const Module &M);

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  void emitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void emitZerofill(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                    unsigned ByteAlignment, SMLoc Loc = SMLoc()) override;
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;

  // Ignore COFF-specific directives; we do not need any information from them,
  // but the default implementation of these methods crashes, so we override
  // them with versions that do nothing.
  void BeginCOFFSymbolDef(const MCSymbol *Symbol) override {}
  void EmitCOFFSymbolStorageClass(int StorageClass) override {}
  void EmitCOFFSymbolType(int Type) override {}
  void EndCOFFSymbolDef() override {}

  /// Record .symver aliases for later processing.
  void emitELFSymverDirective(StringRef AliasName,
                              const MCSymbol *Aliasee) override;

  // Emit ELF .symver aliases and ensure they have the same binding as the
  // defined symbol they alias with.
  void flushSymverDirectives();

  // Symbols iterators
  using const_iterator = StringMap<State>::const_iterator;
  const_iterator begin();
  const_iterator end();

  // SymverAliasMap iterators
  using const_symver_iterator = decltype(SymverAliasMap)::const_iterator;
  iterator_range<const_symver_iterator> symverAliases();
};

} // end namespace llvm

#endif // LLVM_LIB_OBJECT_RECORDSTREAMER_H
