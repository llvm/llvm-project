//===-- WebAssemblyWasmObjectWriter.cpp - WebAssembly Wasm Writer ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file handles Wasm-specific object emission, converting LLVM's
/// internal fixups into the appropriate relocations.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCWasmObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

namespace {
class WebAssemblyWasmObjectWriter final : public MCWasmObjectTargetWriter {
public:
  explicit WebAssemblyWasmObjectWriter(bool Is64Bit);

private:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};
} // end anonymous namespace

WebAssemblyWasmObjectWriter::WebAssemblyWasmObjectWriter(bool Is64Bit)
    : MCWasmObjectTargetWriter(Is64Bit) {}

unsigned WebAssemblyWasmObjectWriter::getRelocType(MCContext &Ctx,
                                                   const MCValue &Target,
                                                   const MCFixup &Fixup,
                                                   bool IsPCRel) const {
  llvm_unreachable("Relocations not yet implemented");
  return 0;
}

MCObjectWriter *llvm::createWebAssemblyWasmObjectWriter(raw_pwrite_stream &OS,
                                                        bool Is64Bit) {
  MCWasmObjectTargetWriter *MOTW = new WebAssemblyWasmObjectWriter(Is64Bit);
  return createWasmObjectWriter(MOTW, OS);
}
