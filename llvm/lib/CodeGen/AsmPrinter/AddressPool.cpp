//===- llvm/CodeGen/AddressPool.cpp - Dwarf Debug Framework ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AddressPool.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include <utility>

using namespace llvm;

unsigned AddressPool::getIndex(const MCSymbol *Sym, bool TLS) {
  HasBeenUsed = true;
  auto IterBool =
      Pool.insert(std::make_pair(Sym, AddressPoolEntry(Pool.size(), TLS)));
  return IterBool.first->second.Number;
}


void AddressPool::emitHeader(AsmPrinter &Asm, MCSection *Section) {
  static const uint8_t AddrSize = Asm.getDataLayout().getPointerSize();
  Asm.OutStreamer->SwitchSection(Section);

  uint64_t Length = sizeof(uint16_t) // version
                  + sizeof(uint8_t)  // address_size
                  + sizeof(uint8_t)  // segment_selector_size
                  + AddrSize * Pool.size(); // entries
  Asm.emitInt32(Length); // TODO: Support DWARF64 format.
  Asm.emitInt16(Asm.getDwarfVersion());
  Asm.emitInt8(AddrSize);
  Asm.emitInt8(0); // TODO: Support non-zero segment_selector_size.
}

// Emit addresses into the section given.
void AddressPool::emit(AsmPrinter &Asm, MCSection *AddrSection) {
  if (Asm.getDwarfVersion() >= 5)
    emitHeader(Asm, AddrSection);

  if (Pool.empty())
    return;

  // Start the dwarf addr section.
  Asm.OutStreamer->SwitchSection(AddrSection);

  // Order the address pool entries by ID
  SmallVector<const MCExpr *, 64> Entries(Pool.size());

  for (const auto &I : Pool)
    Entries[I.second.Number] =
        I.second.TLS
            ? Asm.getObjFileLowering().getDebugThreadLocalSymbol(I.first)
            : MCSymbolRefExpr::create(I.first, Asm.OutContext);

  for (const MCExpr *Entry : Entries)
    Asm.OutStreamer->EmitValue(Entry, Asm.getDataLayout().getPointerSize());
}
