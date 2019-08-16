//===-- lib/MC/XCOFFObjectWriter.cpp - XCOFF file writer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XCOFF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCXCOFFObjectWriter.h"

using namespace llvm;

namespace {

class XCOFFObjectWriter : public MCObjectWriter {
  support::endian::Writer W;
  std::unique_ptr<MCXCOFFObjectTargetWriter> TargetObjectWriter;

  void executePostLayoutBinding(MCAssembler &, const MCAsmLayout &) override;

  void recordRelocation(MCAssembler &, const MCAsmLayout &, const MCFragment *,
                        const MCFixup &, MCValue, uint64_t &) override;

  uint64_t writeObject(MCAssembler &, const MCAsmLayout &) override;

public:
  XCOFFObjectWriter(std::unique_ptr<MCXCOFFObjectTargetWriter> MOTW,
                    raw_pwrite_stream &OS);
};

XCOFFObjectWriter::XCOFFObjectWriter(
    std::unique_ptr<MCXCOFFObjectTargetWriter> MOTW, raw_pwrite_stream &OS)
    : W(OS, support::big), TargetObjectWriter(std::move(MOTW)) {}

void XCOFFObjectWriter::executePostLayoutBinding(MCAssembler &,
                                                 const MCAsmLayout &) {
  // TODO Implement once we have sections and symbols to handle.
}

void XCOFFObjectWriter::recordRelocation(MCAssembler &, const MCAsmLayout &,
                                         const MCFragment *, const MCFixup &,
                                         MCValue, uint64_t &) {
  report_fatal_error("XCOFF relocations not supported.");
}

uint64_t XCOFFObjectWriter::writeObject(MCAssembler &Asm, const MCAsmLayout &) {
  // We always emit a timestamp of 0 for reproducibility, so ensure incremental
  // linking is not enabled, in case, like with Windows COFF, such a timestamp
  // is incompatible with incremental linking of XCOFF.
  if (Asm.isIncrementalLinkerCompatible())
    report_fatal_error("Incremental linking not supported for XCOFF.");

  if (TargetObjectWriter->is64Bit())
    report_fatal_error("64-bit XCOFF object files are not supported yet.");

  uint64_t StartOffset = W.OS.tell();

  // TODO FIXME Assign section numbers/finalize sections.

  // TODO FIXME Finalize symbols.

  // Magic.
  W.write<uint16_t>(0x01df);
  // Number of sections.
  W.write<uint16_t>(0);
  // Timestamp field. For reproducible output we write a 0, which represents no
  // timestamp.
  W.write<int32_t>(0);
  // Byte Offset to the start of the symbol table.
  W.write<uint32_t>(0);
  // Number of entries in the symbol table.
  W.write<int32_t>(0);
  // Size of the optional header.
  W.write<uint16_t>(0);
  // Flags.
  W.write<uint16_t>(0);

  return W.OS.tell() - StartOffset;
}

} // end anonymous namespace

std::unique_ptr<MCObjectWriter>
llvm::createXCOFFObjectWriter(std::unique_ptr<MCXCOFFObjectTargetWriter> MOTW,
                              raw_pwrite_stream &OS) {
  return llvm::make_unique<XCOFFObjectWriter>(std::move(MOTW), OS);
}
