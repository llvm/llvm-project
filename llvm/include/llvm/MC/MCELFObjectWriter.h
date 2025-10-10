//===- llvm/MC/MCELFObjectWriter.h - ELF Object Writer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFOBJECTWRITER_H
#define LLVM_MC_MCELFOBJECTWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace llvm {

class MCAssembler;
class MCContext;
class MCFixup;
class MCSymbol;
class MCSymbolELF;
class MCTargetOptions;
class MCValue;

struct ELFRelocationEntry {
  uint64_t Offset; // Where is the relocation.
  const MCSymbolELF *Symbol; // The symbol to relocate with.
  unsigned Type;   // The type of the relocation.
  uint64_t Addend; // The addend to use.

  ELFRelocationEntry(uint64_t Offset, const MCSymbolELF *Symbol, unsigned Type,
                     uint64_t Addend)
      : Offset(Offset), Symbol(Symbol), Type(Type), Addend(Addend) {}

  void print(raw_ostream &Out) const {
    Out << "Off=" << Offset << ", Sym=" << Symbol << ", Type=" << Type
        << ", Addend=" << Addend;
  }

  LLVM_DUMP_METHOD void dump() const { print(errs()); }
};

class MCELFObjectTargetWriter : public MCObjectTargetWriter {
  const uint8_t OSABI;
  const uint8_t ABIVersion;
  const uint16_t EMachine;
  const unsigned HasRelocationAddend : 1;
  const unsigned Is64Bit : 1;

protected:
  MCELFObjectTargetWriter(bool Is64Bit_, uint8_t OSABI_, uint16_t EMachine_,
                          bool HasRelocationAddend_, uint8_t ABIVersion_ = 0);

public:
  virtual ~MCELFObjectTargetWriter() = default;

  Triple::ObjectFormatType getFormat() const override { return Triple::ELF; }
  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::ELF;
  }

  static uint8_t getOSABI(Triple::OSType OSType) {
    switch (OSType) {
      case Triple::HermitCore:
        return ELF::ELFOSABI_STANDALONE;
      case Triple::PS4:
      case Triple::FreeBSD:
        return ELF::ELFOSABI_FREEBSD;
      case Triple::Solaris:
        return ELF::ELFOSABI_SOLARIS;
      case Triple::OpenBSD:
        return ELF::ELFOSABI_OPENBSD;
      default:
        return ELF::ELFOSABI_NONE;
    }
  }

  virtual unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target,
                                bool IsPCRel) const = 0;

  virtual bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const {
    return false;
  }

  virtual void sortRelocs(std::vector<ELFRelocationEntry> &Relocs);

  /// \name Accessors
  /// @{
  uint8_t getOSABI() const { return OSABI; }
  uint8_t getABIVersion() const { return ABIVersion; }
  uint16_t getEMachine() const { return EMachine; }
  bool hasRelocationAddend() const { return HasRelocationAddend; }
  bool is64Bit() const { return Is64Bit; }
  /// @}

  // Instead of changing everyone's API we pack the N64 Type fields
  // into the existing 32 bit data unsigned.
#define R_TYPE_SHIFT 0
#define R_TYPE_MASK 0xffffff00
#define R_TYPE2_SHIFT 8
#define R_TYPE2_MASK 0xffff00ff
#define R_TYPE3_SHIFT 16
#define R_TYPE3_MASK 0xff00ffff
#define R_SSYM_SHIFT 24
#define R_SSYM_MASK 0x00ffffff

  // N64 relocation type accessors
  uint8_t getRType(uint32_t Type) const {
    return (unsigned)((Type >> R_TYPE_SHIFT) & 0xff);
  }
  uint8_t getRType2(uint32_t Type) const {
    return (unsigned)((Type >> R_TYPE2_SHIFT) & 0xff);
  }
  uint8_t getRType3(uint32_t Type) const {
    return (unsigned)((Type >> R_TYPE3_SHIFT) & 0xff);
  }
  uint8_t getRSsym(uint32_t Type) const {
    return (unsigned)((Type >> R_SSYM_SHIFT) & 0xff);
  }

  // N64 relocation type setting
  static unsigned setRTypes(unsigned Value1, unsigned Value2, unsigned Value3) {
    return ((Value1 & 0xff) << R_TYPE_SHIFT) |
           ((Value2 & 0xff) << R_TYPE2_SHIFT) |
           ((Value3 & 0xff) << R_TYPE3_SHIFT);
  }
  unsigned setRSsym(unsigned Value, unsigned Type) const {
    return (Type & R_SSYM_MASK) | ((Value & 0xff) << R_SSYM_SHIFT);
  }
};

class ELFObjectWriter final : public MCObjectWriter {
  unsigned ELFHeaderEFlags = 0;

public:
  std::unique_ptr<MCELFObjectTargetWriter> TargetObjectWriter;
  raw_pwrite_stream &OS;
  raw_pwrite_stream *DwoOS = nullptr;

  DenseMap<const MCSectionELF *, std::vector<ELFRelocationEntry>> Relocations;
  DenseMap<const MCSymbolELF *, const MCSymbolELF *> Renames;
  // .weakref aliases
  SmallVector<const MCSymbolELF *, 0> Weakrefs;
  bool IsLittleEndian = false;
  bool SeenGnuAbi = false;
  std::optional<uint8_t> OverrideABIVersion;

  struct Symver {
    SMLoc Loc;
    const MCSymbol *Sym;
    StringRef Name;
    // True if .symver *, *@@@* or .symver *, *, remove.
    bool KeepOriginalSym;
  };
  SmallVector<Symver, 0> Symvers;

  ELFObjectWriter(std::unique_ptr<MCELFObjectTargetWriter> MOTW,
                  raw_pwrite_stream &OS, bool IsLittleEndian);
  ELFObjectWriter(std::unique_ptr<MCELFObjectTargetWriter> MOTW,
                  raw_pwrite_stream &OS, raw_pwrite_stream &DwoOS,
                  bool IsLittleEndian);

  void reset() override;
  void setAssembler(MCAssembler *Asm) override;
  void executePostLayoutBinding() override;
  void recordRelocation(const MCFragment &F, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override;
  bool isSymbolRefDifferenceFullyResolvedImpl(const MCSymbol &SymA,
                                              const MCFragment &FB, bool InSet,
                                              bool IsPCRel) const override;
  uint64_t writeObject() override;

  bool hasRelocationAddend() const;
  bool usesRela(const MCTargetOptions *TO, const MCSectionELF &Sec) const;

  bool useSectionSymbol(const MCValue &Val, const MCSymbolELF *Sym, uint64_t C,
                        unsigned Type) const;

  bool checkRelocation(SMLoc Loc, const MCSectionELF *From,
                       const MCSectionELF *To);

  unsigned getELFHeaderEFlags() const { return ELFHeaderEFlags; }
  void setELFHeaderEFlags(unsigned Flags) { ELFHeaderEFlags = Flags; }

  // Mark that we have seen GNU ABI usage (e.g. SHF_GNU_RETAIN, STB_GNU_UNIQUE).
  void markGnuAbi() { SeenGnuAbi = true; }
  bool seenGnuAbi() const { return SeenGnuAbi; }

  // Override the default e_ident[EI_ABIVERSION] in the ELF header.
  void setOverrideABIVersion(uint8_t V) { OverrideABIVersion = V; }
};
} // end namespace llvm

#endif // LLVM_MC_MCELFOBJECTWRITER_H
