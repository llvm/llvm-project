//===- lib/MC/MCHexStreamer.cpp - Hex Output --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

namespace {

class MCHexStreamer final : public MCStreamer {
  std::unique_ptr<formatted_raw_ostream> OSOwner;
  formatted_raw_ostream &OS;
  std::unique_ptr<MCInstPrinter> InstPrinter;
  std::unique_ptr<MCAssembler> Assembler;

  raw_null_ostream NullStream;

public:
  MCHexStreamer(MCContext &Context, std::unique_ptr<formatted_raw_ostream> os,
                MCInstPrinter *printer, std::unique_ptr<MCCodeEmitter> emitter,
                std::unique_ptr<MCAsmBackend> asmbackend)
      : MCStreamer(Context), OSOwner(std::move(os)), OS(*OSOwner),
        InstPrinter(printer),
        Assembler(std::make_unique<MCAssembler>(
            Context, std::move(asmbackend), std::move(emitter),
            asmbackend ? asmbackend->createObjectWriter(NullStream)
                       : nullptr)) {
    assert(InstPrinter);
  }

  MCAssembler &getAssembler() { return *Assembler; }

  /// @name MCStreamer Interface
  /// @{

  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}

  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, Align ByteAlignment = Align(1),
                    SMLoc Loc = SMLoc()) override {}

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
};

} // end anonymous namespace.

void MCHexStreamer::emitInstruction(const MCInst &Inst,
                                    const MCSubtargetInfo &STI) {
  SmallString<256> Code;
  SmallVector<MCFixup, 4> Fixups;
  getAssembler().getEmitter().encodeInstruction(Inst, Code, Fixups, STI);

  OS << '[';
  ListSeparator LS;
  for (auto [I, B] : enumerate(Code))
    OS << LS << format("0x%02x", B & 0xff);
  OS << "]  #";

  if (getTargetStreamer())
    getTargetStreamer()->prettyPrintAsm(*InstPrinter, 0, Inst, STI, OS);
  else
    InstPrinter->printInst(&Inst, 0, "", STI, OS);

  OS << '\n';
}

MCStreamer *llvm::createHexStreamer(MCContext &Context,
                                    std::unique_ptr<formatted_raw_ostream> OS,
                                    MCInstPrinter *IP,
                                    std::unique_ptr<MCCodeEmitter> &&CE,
                                    std::unique_ptr<MCAsmBackend> &&MAB) {
  return new MCHexStreamer(Context, std::move(OS), IP, std::move(CE),
                           std::move(MAB));
}
