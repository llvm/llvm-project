//===-- llvm/MC/MCWasmObjectWriter.h - Wasm Object Writer -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWASMOBJECTWRITER_H
#define LLVM_MC_MCWASMOBJECTWRITER_H

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace llvm {
class MCAssembler;
class MCContext;
class MCFixup;
class MCFragment;
class MCObjectWriter;
class MCSectionWasm;
class MCSymbol;
class MCSymbolWasm;
class MCValue;
class raw_pwrite_stream;

// Information about a single relocation.
struct WasmRelocationEntry {
  uint64_t Offset;            // Where is the relocation.
  const MCSymbolWasm *Symbol; // The symbol to relocate with.
  uint64_t Addend;            // A value to add to the symbol.
  unsigned Type;              // The type of the relocation.
  MCSectionWasm *FixupSection;// The section the relocation is targeting.

  WasmRelocationEntry(uint64_t Offset, const MCSymbolWasm *Symbol,
                      uint64_t Addend, unsigned Type,
                      MCSectionWasm *FixupSection)
      : Offset(Offset), Symbol(Symbol), Addend(Addend), Type(Type),
        FixupSection(FixupSection) {}

  void print(raw_ostream &Out) const {
    Out << "Off=" << Offset << ", Sym=" << Symbol << ", Addend=" << Addend
        << ", Type=" << Type << ", FixupSection=" << FixupSection;
  }
  void dump() const { print(errs()); }
};

class MCWasmObjectTargetWriter {
  const unsigned Is64Bit : 1;

protected:
  explicit MCWasmObjectTargetWriter(bool Is64Bit_);

public:
  virtual ~MCWasmObjectTargetWriter() {}

  virtual unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                                const MCFixup &Fixup, bool IsPCRel) const = 0;

  virtual bool needsRelocateWithSymbol(const MCSymbol &Sym,
                                       unsigned Type) const;

  virtual void sortRelocs(const MCAssembler &Asm,
                          std::vector<WasmRelocationEntry> &Relocs);

  /// \name Accessors
  /// @{
  bool is64Bit() const { return Is64Bit; }
  /// @}
};

/// \brief Construct a new Wasm writer instance.
///
/// \param MOTW - The target specific Wasm writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createWasmObjectWriter(MCWasmObjectTargetWriter *MOTW,
                                       raw_pwrite_stream &OS);
} // End llvm namespace

#endif
