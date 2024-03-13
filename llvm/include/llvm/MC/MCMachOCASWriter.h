//===- llvm/MC/MCMachOCASWriter.h - Mach CAS Object Writer ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMACHOCASWRITER_H
#define LLVM_MC_MCMACHOCASWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {

namespace cas {
class ObjectStore;
class CASID;
} // namespace cas

class MachOCASWriter : public MCObjectWriter {
public:
  /// ObjectStore
  const Triple Target;
  cas::ObjectStore &CAS;
  CASBackendMode Mode;
  std::optional<MCTargetOptions::ResultCallBackTy> ResultCallBack;

  MachOCASWriter(
      std::unique_ptr<MCMachObjectTargetWriter> MOTW, const Triple &TT,
      cas::ObjectStore &CAS, CASBackendMode Mode, raw_pwrite_stream &OS,
      bool IsLittleEndian,
      std::function<const cas::ObjectProxy(
          llvm::MachOCASWriter &, llvm::MCAssembler &,
          const llvm::MCAsmLayout &, cas::ObjectStore &, raw_ostream *)>
          CreateFromMcAssembler,
      std::function<Error(cas::ObjectProxy, cas::ObjectStore &, raw_ostream &)>
          SerializeObjectFile,
      std::optional<MCTargetOptions::ResultCallBackTy> CallBack,
      raw_pwrite_stream *CasIDOS = nullptr);

  uint8_t getAddressSize() { return Target.isArch32Bit() ? 4 : 8; }

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override {
    MOW.recordRelocation(Asm, Layout, Fragment, Fixup, Target, FixedValue);
  }

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override {
    MOW.executePostLayoutBinding(Asm, Layout);
  }

  bool isSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                              const MCSymbol &A,
                                              const MCSymbol &B,
                                              bool InSet) const override {
    return MOW.isSymbolRefDifferenceFullyResolvedImpl(Asm, A, B, InSet);
  }

  bool isSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                              const MCSymbol &SymA,
                                              const MCFragment &FB, bool InSet,
                                              bool IsPCRel) const override {
    return MOW.isSymbolRefDifferenceFullyResolvedImpl(Asm, SymA, FB, InSet,
                                                      IsPCRel);
  }

  uint64_t getPaddingSize(const MCSection *SD,
                          const MCAsmLayout &Layout) const {
    return MOW.getPaddingSize(SD, Layout);
  }

  void prepareObject(MCAssembler &Asm, const MCAsmLayout &Layout) {
    MOW.prepareObject(Asm, Layout);
  }

  void writeMachOHeader(MCAssembler &Asm, const MCAsmLayout &Layout) {
    MOW.writeMachOHeader(Asm, Layout);
  }

  void writeSectionData(MCAssembler &Asm, const MCAsmLayout &Layout) {
    MOW.writeSectionData(Asm, Layout);
  }

  void writeRelocations(MCAssembler &Asm, const MCAsmLayout &Layout) {
    MOW.writeRelocations(Asm, Layout);
  }

  void writeDataInCodeRegion(MCAssembler &Asm, const MCAsmLayout &Layout) {
    MOW.writeDataInCodeRegion(Asm, Layout);
  }

  void writeSymbolTable(MCAssembler &Asm, const MCAsmLayout &Layout) {
    MOW.writeSymbolTable(Asm, Layout);
  }

  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

  void resetBuffer() { OSOffset = InternalOS.tell(); }

  StringRef getContent() const { return InternalBuffer.substr(OSOffset); }

  DenseMap<const MCSection *, std::vector<MachObjectWriter::RelAndSymbol>> &
  getRelocations() {
    return MOW.getRelocations();
  }

private:
  raw_pwrite_stream &OS;
  raw_pwrite_stream *CasIDOS;

  SmallString<512> InternalBuffer;
  raw_svector_ostream InternalOS;

  MachObjectWriter MOW;

  uint64_t OSOffset = 0;

  std::function<const cas::ObjectProxy(
      llvm::MachOCASWriter &, llvm::MCAssembler &, const llvm::MCAsmLayout &,
      cas::ObjectStore &, raw_ostream *)>
      CreateFromMcAssembler;
  std::function<Error(cas::ObjectProxy, cas::ObjectStore &, raw_ostream &)>
      SerializeObjectFile;
};

/// Construct a new Mach-O CAS writer instance.
///
/// This routine takes ownership of the target writer subclass.
///
/// \param MOTW - The target specific Mach-O writer subclass.
/// \param TT - The target triple.
/// \param CAS - The ObjectStore instance.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter> createMachOCASWriter(
    std::unique_ptr<MCMachObjectTargetWriter> MOTW, const Triple &TT,
    cas::ObjectStore &CAS, CASBackendMode Mode, raw_pwrite_stream &OS,
    bool IsLittleEndian,
    std::function<const cas::ObjectProxy(
        llvm::MachOCASWriter &, llvm::MCAssembler &, const llvm::MCAsmLayout &,
        cas::ObjectStore &, raw_ostream *)>
        CreateFromMcAssembler,
    std::function<Error(cas::ObjectProxy, cas::ObjectStore &, raw_ostream &)>
        SerializeObjectFile,
    std::optional<MCTargetOptions::ResultCallBackTy> CallBack = std::nullopt,
    raw_pwrite_stream *CasIDOS = nullptr);

} // end namespace llvm

#endif // LLVM_MC_MCMACHOCASWRITER_H
