//===- DwarfEmitterImpl.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFEMITTERIMPL_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFEMITTERIMPL_H

#include "DWARFLinkerCompileUnit.h"
#include "llvm/BinaryFormat/Swift.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

///   User of DwarfEmitterImpl should call initialization code
///   for AsmPrinter:
///
///   InitializeAllTargetInfos();
///   InitializeAllTargetMCs();
///   InitializeAllTargets();
///   InitializeAllAsmPrinters();

template <typename DataT> class AccelTable;
class MCCodeEmitter;

namespace dwarflinker_parallel {

using DebugNamesUnitsOffsets = std::vector<std::variant<MCSymbol *, uint64_t>>;
using CompUnitIDToIdx = DenseMap<unsigned, size_t>;

/// This class emits DWARF data to the output stream. It emits already
/// generated section data and specific data, which could not be generated
/// by CompileUnit.
class DwarfEmitterImpl : public ExtraDwarfEmitter {
public:
  DwarfEmitterImpl(DWARFLinker::OutputFileType OutFileType,
                   raw_pwrite_stream &OutFile)
      : OutFile(OutFile), OutFileType(OutFileType) {}

  /// Initialize AsmPrinter data.
  Error init(Triple TheTriple, StringRef Swift5ReflectionSegmentName);

  /// Returns triple of output stream.
  const Triple &getTargetTriple() { return MC->getTargetTriple(); }

  /// Dump the file to the disk.
  void finish() override { MS->finish(); }

  /// Returns AsmPrinter.
  AsmPrinter &getAsmPrinter() const override { return *Asm; }

  /// Emit the swift_ast section stored in \p Buffer.
  void emitSwiftAST(StringRef Buffer) override;

  /// Emit the swift reflection section stored in \p Buffer.
  void emitSwiftReflectionSection(
      llvm::binaryformat::Swift5ReflectionSectionKind ReflSectionKind,
      StringRef Buffer, uint32_t Alignment, uint32_t) override;

  /// Emit specified section data.
  void emitSectionContents(StringRef SecData, StringRef SecName) override;

  /// Emit abbreviations.
  void emitAbbrevs(const SmallVector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
                   unsigned DwarfVersion);

  /// Emit compile unit header.
  void emitCompileUnitHeader(DwarfUnit &Unit);

  /// Emit DIE recursively.
  void emitDIE(DIE &Die);

  /// Returns size of generated .debug_info section.
  uint64_t getDebugInfoSectionSize() const { return DebugInfoSectionSize; }

  /// Emits .debug_names section according to the specified \p Table.
  void emitDebugNames(DWARF5AccelTable &Table,
                      DebugNamesUnitsOffsets &CUOffsets,
                      CompUnitIDToIdx &UnitIDToIdxMap);

  /// Emits .apple_names section according to the specified \p Table.
  void emitAppleNames(AccelTable<AppleAccelTableStaticOffsetData> &Table);

  /// Emits .apple_namespaces section according to the specified \p Table.
  void emitAppleNamespaces(AccelTable<AppleAccelTableStaticOffsetData> &Table);

  /// Emits .apple_objc section according to the specified \p Table.
  void emitAppleObjc(AccelTable<AppleAccelTableStaticOffsetData> &Table);

  /// Emits .apple_types section according to the specified \p Table.
  void emitAppleTypes(AccelTable<AppleAccelTableStaticTypeData> &Table);

private:
  // Enumerate all string patches and write them into the destination section.
  // Order of patches is the same as in original input file. To avoid emitting
  // the same string twice we accumulate NextOffset value. Thus if string
  // offset smaller than NextOffset value then the patch is skipped (as that
  // string was emitted earlier).
  template <typename PatchTy>
  void emitStringsImpl(ArrayList<PatchTy> &StringPatches,
                       const StringEntryToDwarfStringPoolEntryMap &Strings,
                       uint64_t &NextOffset, MCSection *OutSection);

  MCSection *switchSection(StringRef SecName);

  /// \defgroup MCObjects MC layer objects constructed by the streamer
  /// @{
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCObjectFileInfo> MOFI;
  std::unique_ptr<MCContext> MC;
  MCAsmBackend *MAB; // Owned by MCStreamer
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCSubtargetInfo> MSTI;
  MCInstPrinter *MIP; // Owned by AsmPrinter
  MCCodeEmitter *MCE; // Owned by MCStreamer
  MCStreamer *MS;     // Owned by AsmPrinter
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<AsmPrinter> Asm;
  /// @}

  /// The output file we stream the linked Dwarf to.
  raw_pwrite_stream &OutFile;
  DWARFLinker::OutputFileType OutFileType = DWARFLinker::OutputFileType::Object;

  uint64_t DebugInfoSectionSize = 0;
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFEMITTERIMPL_H
