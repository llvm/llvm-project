//===-- WebAssemblyELFObjectWriter.cpp - WebAssembly ELF Writer -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file handles ELF-specific object emission, converting LLVM's
/// internal fixups into the appropriate relocations.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

namespace {
class WebAssemblyELFObjectWriter final : public MCELFObjectTargetWriter {
public:
  WebAssemblyELFObjectWriter(bool Is64Bit, uint8_t OSABI);

protected:
  unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                        bool IsPCRel) const override;
};
} // end anonymous namespace

// FIXME: Use EM_NONE as a temporary hack. Should we decide to pursue ELF
// writing seriously, we should email generic-abi@googlegroups.com and ask
// for our own ELF code.
WebAssemblyELFObjectWriter::WebAssemblyELFObjectWriter(bool Is64Bit,
                                                       uint8_t OSABI)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_NONE,
                              /*HasRelocationAddend=*/true) {}

unsigned WebAssemblyELFObjectWriter::GetRelocType(const MCValue &Target,
                                                  const MCFixup &Fixup,
                                                  bool IsPCRel) const {
  // FIXME: Do we need our own relocs?
  return Fixup.getKind();
}

MCObjectWriter *llvm::createWebAssemblyELFObjectWriter(raw_pwrite_stream &OS,
                                                       bool Is64Bit,
                                                       uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW =
      new WebAssemblyELFObjectWriter(Is64Bit, OSABI);
  return createELFObjectWriter(MOTW, OS, /*IsLittleEndian=*/true);
}
