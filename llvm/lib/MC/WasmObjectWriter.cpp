//===- lib/MC/WasmObjectWriter.cpp - Wasm File Writer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Wasm object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWasmObjectWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"
#include <vector>

using namespace llvm;

#undef DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

namespace {
typedef DenseMap<const MCSectionWasm *, uint32_t> SectionIndexMapTy;

class WasmObjectWriter : public MCObjectWriter {
  /// Helper struct for containing some precomputed information on symbols.
  struct WasmSymbolData {
    const MCSymbolWasm *Symbol;
    StringRef Name;

    // Support lexicographic sorting.
    bool operator<(const WasmSymbolData &RHS) const { return Name < RHS.Name; }
  };

  /// The target specific Wasm writer instance.
  std::unique_ptr<MCWasmObjectTargetWriter> TargetObjectWriter;

  // TargetObjectWriter wrappers.
  bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const {
    return TargetObjectWriter->getRelocType(Ctx, Target, Fixup, IsPCRel);
  }

public:
  WasmObjectWriter(MCWasmObjectTargetWriter *MOTW, raw_pwrite_stream &OS)
      : MCObjectWriter(OS, /*IsLittleEndian=*/true), TargetObjectWriter(MOTW) {}

  void reset() override {
    MCObjectWriter::reset();
  }

  ~WasmObjectWriter() override;

  void writeHeader(const MCAssembler &Asm);

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, bool &IsPCRel,
                        uint64_t &FixedValue) override;

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
};
} // end anonymous namespace

WasmObjectWriter::~WasmObjectWriter() {}

// Emit the Wasm header.
void WasmObjectWriter::writeHeader(const MCAssembler &Asm) {
  // TODO: write the magic cookie and the version.
}

void WasmObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
}

void WasmObjectWriter::recordRelocation(MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCFixup &Fixup, MCValue Target,
                                        bool &IsPCRel, uint64_t &FixedValue) {
  // TODO: Implement
}

void WasmObjectWriter::writeObject(MCAssembler &Asm,
                                   const MCAsmLayout &Layout) {
  // Write out the Wasm header.
  writeHeader(Asm);

  // TODO: Write the contents.
}

MCObjectWriter *llvm::createWasmObjectWriter(MCWasmObjectTargetWriter *MOTW,
                                             raw_pwrite_stream &OS) {
  return new WasmObjectWriter(MOTW, OS);
}
