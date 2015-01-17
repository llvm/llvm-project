//===-- llvm/MC/MCMachObjectWriter.h - Mach Object Writer -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMACHOBJECTWRITER_H
#define LLVM_MC_MCMACHOBJECTWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MachO.h"
#include <vector>

namespace llvm {

class MCSectionData;
class MachObjectWriter;

class MCMachObjectTargetWriter {
  const unsigned Is64Bit : 1;
  const uint32_t CPUType;
  const uint32_t CPUSubtype;
  // FIXME: Remove this, we should just always use it once we no longer care
  // about Darwin 'as' compatibility.
  const unsigned UseAggressiveSymbolFolding : 1;
  unsigned LocalDifference_RIT;

protected:
  MCMachObjectTargetWriter(bool Is64Bit_, uint32_t CPUType_,
                           uint32_t CPUSubtype_,
                           bool UseAggressiveSymbolFolding_ = false);

  void setLocalDifferenceRelocationType(unsigned Type) {
    LocalDifference_RIT = Type;
  }

public:
  virtual ~MCMachObjectTargetWriter();

  /// @name Lifetime Management
  /// @{

  virtual void reset() {};

  /// @}

  /// @name Accessors
  /// @{

  bool is64Bit() const { return Is64Bit; }
  bool useAggressiveSymbolFolding() const { return UseAggressiveSymbolFolding; }
  uint32_t getCPUType() const { return CPUType; }
  uint32_t getCPUSubtype() const { return CPUSubtype; }
  unsigned getLocalDifferenceRelocationType() const {
    return LocalDifference_RIT;
  }

  /// @}

  /// @name API
  /// @{

  virtual void RecordRelocation(MachObjectWriter *Writer,
                                const MCAssembler &Asm,
                                const MCAsmLayout &Layout,
                                const MCFragment *Fragment,
                                const MCFixup &Fixup,
                                MCValue Target,
                                uint64_t &FixedValue) = 0;

  /// @}
};

class MachObjectWriter : public MCObjectWriter {
  /// MachSymbolData - Helper struct for containing some precomputed information
  /// on symbols.
  struct MachSymbolData {
    MCSymbolData *SymbolData;
    uint64_t StringIndex;
    uint8_t SectionIndex;

    // Support lexicographic sorting.
    bool operator<(const MachSymbolData &RHS) const;
  };

  /// The target specific Mach-O writer instance.
  std::unique_ptr<MCMachObjectTargetWriter> TargetObjectWriter;

  /// @name Relocation Data
  /// @{

  llvm::DenseMap<const MCSectionData*,
                 std::vector<MachO::any_relocation_info> > Relocations;
  llvm::DenseMap<const MCSectionData*, unsigned> IndirectSymBase;

  /// @}
  /// @name Symbol Table Data
  /// @{

  StringTableBuilder StringTable;
  std::vector<MachSymbolData> LocalSymbolData;
  std::vector<MachSymbolData> ExternalSymbolData;
  std::vector<MachSymbolData> UndefinedSymbolData;

  /// @}

  MachSymbolData *findSymbolData(const MCSymbol &Sym);

public:
  MachObjectWriter(MCMachObjectTargetWriter *MOTW, raw_ostream &_OS,
                   bool _IsLittleEndian)
    : MCObjectWriter(_OS, _IsLittleEndian), TargetObjectWriter(MOTW) {
  }

  /// @name Lifetime management Methods
  /// @{

  void reset() override;

  /// @}

  /// @name Utility Methods
  /// @{

  bool isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind);

  SectionAddrMap SectionAddress;

  SectionAddrMap &getSectionAddressMap() { return SectionAddress; }

  uint64_t getSectionAddress(const MCSectionData* SD) const {
    return SectionAddress.lookup(SD);
  }
  uint64_t getSymbolAddress(const MCSymbolData* SD,
                            const MCAsmLayout &Layout) const;

  uint64_t getFragmentAddress(const MCFragment *Fragment,
                              const MCAsmLayout &Layout) const;

  uint64_t getPaddingSize(const MCSectionData *SD,
                          const MCAsmLayout &Layout) const;

  bool doesSymbolRequireExternRelocation(const MCSymbolData *SD);

  /// @}

  /// @name Target Writer Proxy Accessors
  /// @{

  bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
  bool isX86_64() const {
    uint32_t CPUType = TargetObjectWriter->getCPUType();
    return CPUType == MachO::CPU_TYPE_X86_64;
  }

  /// @}

  void WriteHeader(unsigned NumLoadCommands, unsigned LoadCommandsSize,
                   bool SubsectionsViaSymbols);

  /// WriteSegmentLoadCommand - Write a segment load command.
  ///
  /// \param NumSections The number of sections in this segment.
  /// \param SectionDataSize The total size of the sections.
  void WriteSegmentLoadCommand(unsigned NumSections,
                               uint64_t VMSize,
                               uint64_t SectionDataStartOffset,
                               uint64_t SectionDataSize);

  void WriteSection(const MCAssembler &Asm, const MCAsmLayout &Layout,
                    const MCSectionData &SD, uint64_t FileOffset,
                    uint64_t RelocationsStart, unsigned NumRelocations);

  void WriteSymtabLoadCommand(uint32_t SymbolOffset, uint32_t NumSymbols,
                              uint32_t StringTableOffset,
                              uint32_t StringTableSize);

  void WriteDysymtabLoadCommand(uint32_t FirstLocalSymbol,
                                uint32_t NumLocalSymbols,
                                uint32_t FirstExternalSymbol,
                                uint32_t NumExternalSymbols,
                                uint32_t FirstUndefinedSymbol,
                                uint32_t NumUndefinedSymbols,
                                uint32_t IndirectSymbolOffset,
                                uint32_t NumIndirectSymbols);

  void WriteNlist(MachSymbolData &MSD, const MCAsmLayout &Layout);

  void WriteLinkeditLoadCommand(uint32_t Type, uint32_t DataOffset,
                                uint32_t DataSize);

  void WriteLinkerOptionsLoadCommand(const std::vector<std::string> &Options);

  // FIXME: We really need to improve the relocation validation. Basically, we
  // want to implement a separate computation which evaluates the relocation
  // entry as the linker would, and verifies that the resultant fixup value is
  // exactly what the encoder wanted. This will catch several classes of
  // problems:
  //
  //  - Relocation entry bugs, the two algorithms are unlikely to have the same
  //    exact bug.
  //
  //  - Relaxation issues, where we forget to relax something.
  //
  //  - Input errors, where something cannot be correctly encoded. 'as' allows
  //    these through in many cases.

  void addRelocation(const MCSectionData *SD,
                     MachO::any_relocation_info &MRE) {
    Relocations[SD].push_back(MRE);
  }

  void RecordScatteredRelocation(const MCAssembler &Asm,
                                 const MCAsmLayout &Layout,
                                 const MCFragment *Fragment,
                                 const MCFixup &Fixup, MCValue Target,
                                 unsigned Log2Size,
                                 uint64_t &FixedValue);

  void RecordTLVPRelocation(const MCAssembler &Asm,
                            const MCAsmLayout &Layout,
                            const MCFragment *Fragment,
                            const MCFixup &Fixup, MCValue Target,
                            uint64_t &FixedValue);

  void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, bool &IsPCRel,
                        uint64_t &FixedValue) override;

  void BindIndirectSymbols(MCAssembler &Asm);

  /// ComputeSymbolTable - Compute the symbol table data
  ///
  void ComputeSymbolTable(MCAssembler &Asm,
                          std::vector<MachSymbolData> &LocalSymbolData,
                          std::vector<MachSymbolData> &ExternalSymbolData,
                          std::vector<MachSymbolData> &UndefinedSymbolData);

  void computeSectionAddresses(const MCAssembler &Asm,
                               const MCAsmLayout &Layout);

  void markAbsoluteVariableSymbols(MCAssembler &Asm,
                                   const MCAsmLayout &Layout);
  void ExecutePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  bool IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                              const MCSymbolData &DataA,
                                              const MCFragment &FB,
                                              bool InSet,
                                              bool IsPCRel) const override;

  void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
};


/// \brief Construct a new Mach-O writer instance.
///
/// This routine takes ownership of the target writer subclass.
///
/// \param MOTW - The target specific Mach-O writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createMachObjectWriter(MCMachObjectTargetWriter *MOTW,
                                       raw_ostream &OS, bool IsLittleEndian);

} // End llvm namespace

#endif
