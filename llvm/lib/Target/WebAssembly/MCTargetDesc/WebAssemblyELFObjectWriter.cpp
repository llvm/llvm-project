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

WebAssemblyELFObjectWriter::WebAssemblyELFObjectWriter(bool Is64Bit,
                                                       uint8_t OSABI)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_WEBASSEMBLY,
                              /*HasRelocationAddend=*/false) {}

unsigned WebAssemblyELFObjectWriter::GetRelocType(const MCValue &Target,
                                                  const MCFixup &Fixup,
                                                  bool IsPCRel) const {
  // WebAssembly functions are not allocated in the address space. To resolve a
  // pointer to a function, we must use a special relocation type.
  if (const MCSymbolRefExpr *SyExp =
          dyn_cast<MCSymbolRefExpr>(Fixup.getValue()))
    if (SyExp->getKind() == MCSymbolRefExpr::VK_WebAssembly_FUNCTION)
      return ELF::R_WEBASSEMBLY_FUNCTION;

  switch (Fixup.getKind()) {
  case FK_Data_4:
    assert(!is64Bit() && "4-byte relocations only supported on wasm32");
    return ELF::R_WEBASSEMBLY_DATA;
  case FK_Data_8:
    assert(is64Bit() && "8-byte relocations only supported on wasm64");
    return ELF::R_WEBASSEMBLY_DATA;
  default:
    llvm_unreachable("unimplemented fixup kind");
  }
}

MCObjectWriter *llvm::createWebAssemblyELFObjectWriter(raw_pwrite_stream &OS,
                                                       bool Is64Bit,
                                                       uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW =
      new WebAssemblyELFObjectWriter(Is64Bit, OSABI);
  return createELFObjectWriter(MOTW, OS, /*IsLittleEndian=*/true);
}
