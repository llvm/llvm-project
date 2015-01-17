//===-- llvm/MC/MCObjectWriter.h - Object File Writer Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTWRITER_H
#define LLVM_MC_MCOBJECTWRITER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace llvm {
class MCAsmLayout;
class MCAssembler;
class MCFixup;
class MCFragment;
class MCSymbolData;
class MCSymbolRefExpr;
class MCValue;

/// MCObjectWriter - Defines the object file and target independent interfaces
/// used by the assembler backend to write native file format object files.
///
/// The object writer contains a few callbacks used by the assembler to allow
/// the object writer to modify the assembler data structures at appropriate
/// points. Once assembly is complete, the object writer is given the
/// MCAssembler instance, which contains all the symbol and section data which
/// should be emitted as part of WriteObject().
///
/// The object writer also contains a number of helper methods for writing
/// binary data to the output stream.
class MCObjectWriter {
  MCObjectWriter(const MCObjectWriter &) LLVM_DELETED_FUNCTION;
  void operator=(const MCObjectWriter &) LLVM_DELETED_FUNCTION;

protected:
  raw_ostream &OS;

  unsigned IsLittleEndian : 1;

protected: // Can only create subclasses.
  MCObjectWriter(raw_ostream &_OS, bool _IsLittleEndian)
    : OS(_OS), IsLittleEndian(_IsLittleEndian) {}

public:
  virtual ~MCObjectWriter();

  /// lifetime management
  virtual void reset() { }

  bool isLittleEndian() const { return IsLittleEndian; }

  raw_ostream &getStream() { return OS; }

  /// @name High-Level API
  /// @{

  /// \brief Perform any late binding of symbols (for example, to assign symbol
  /// indices for use when generating relocations).
  ///
  /// This routine is called by the assembler after layout and relaxation is
  /// complete.
  virtual void ExecutePostLayoutBinding(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) = 0;

  /// \brief Record a relocation entry.
  ///
  /// This routine is called by the assembler after layout and relaxation, and
  /// post layout binding. The implementation is responsible for storing
  /// information about the relocation so that it can be emitted during
  /// WriteObject().
  virtual void RecordRelocation(const MCAssembler &Asm,
                                const MCAsmLayout &Layout,
                                const MCFragment *Fragment,
                                const MCFixup &Fixup, MCValue Target,
                                bool &IsPCRel,
                                uint64_t &FixedValue) = 0;

  /// \brief Check whether the difference (A - B) between two symbol
  /// references is fully resolved.
  ///
  /// Clients are not required to answer precisely and may conservatively return
  /// false, even when a difference is fully resolved.
  bool
  IsSymbolRefDifferenceFullyResolved(const MCAssembler &Asm,
                                     const MCSymbolRefExpr *A,
                                     const MCSymbolRefExpr *B,
                                     bool InSet) const;

  virtual bool
  IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                         const MCSymbolData &DataA,
                                         const MCFragment &FB,
                                         bool InSet,
                                         bool IsPCRel) const;

  /// \brief Write the object file.
  ///
  /// This routine is called by the assembler after layout and relaxation is
  /// complete, fixups have been evaluated and applied, and relocations
  /// generated.
  virtual void WriteObject(MCAssembler &Asm,
                           const MCAsmLayout &Layout) = 0;

  /// @}
  /// @name Binary Output
  /// @{

  void Write8(uint8_t Value) {
    OS << char(Value);
  }

  void WriteLE16(uint16_t Value) {
    Write8(uint8_t(Value >> 0));
    Write8(uint8_t(Value >> 8));
  }

  void WriteLE32(uint32_t Value) {
    WriteLE16(uint16_t(Value >> 0));
    WriteLE16(uint16_t(Value >> 16));
  }

  void WriteLE64(uint64_t Value) {
    WriteLE32(uint32_t(Value >> 0));
    WriteLE32(uint32_t(Value >> 32));
  }

  void WriteBE16(uint16_t Value) {
    Write8(uint8_t(Value >> 8));
    Write8(uint8_t(Value >> 0));
  }

  void WriteBE32(uint32_t Value) {
    WriteBE16(uint16_t(Value >> 16));
    WriteBE16(uint16_t(Value >> 0));
  }

  void WriteBE64(uint64_t Value) {
    WriteBE32(uint32_t(Value >> 32));
    WriteBE32(uint32_t(Value >> 0));
  }

  void Write16(uint16_t Value) {
    if (IsLittleEndian)
      WriteLE16(Value);
    else
      WriteBE16(Value);
  }

  void Write32(uint32_t Value) {
    if (IsLittleEndian)
      WriteLE32(Value);
    else
      WriteBE32(Value);
  }

  void Write64(uint64_t Value) {
    if (IsLittleEndian)
      WriteLE64(Value);
    else
      WriteBE64(Value);
  }

  void WriteZeros(unsigned N) {
    const char Zeros[16] = { 0 };

    for (unsigned i = 0, e = N / 16; i != e; ++i)
      OS << StringRef(Zeros, 16);

    OS << StringRef(Zeros, N % 16);
  }

  void WriteBytes(const SmallVectorImpl<char> &ByteVec, unsigned ZeroFillSize = 0) {
    WriteBytes(StringRef(ByteVec.data(), ByteVec.size()), ZeroFillSize);
  }

  void WriteBytes(StringRef Str, unsigned ZeroFillSize = 0) {
    // TODO: this version may need to go away once all fragment contents are
    // converted to SmallVector<char, N>
    assert((ZeroFillSize == 0 || Str.size () <= ZeroFillSize) &&
      "data size greater than fill size, unexpected large write will occur");
    OS << Str;
    if (ZeroFillSize)
      WriteZeros(ZeroFillSize - Str.size());
  }

  /// @}

};

} // End llvm namespace

#endif
