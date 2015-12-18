//=- WebAssemblyMCCodeEmitter.cpp - Convert WebAssembly code to machine code -//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the WebAssemblyMCCodeEmitter class.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {
class WebAssemblyMCCodeEmitter final : public MCCodeEmitter {
  const MCRegisterInfo &MRI;

public:
  WebAssemblyMCCodeEmitter(const MCInstrInfo &, const MCRegisterInfo &mri,
                           MCContext &)
      : MRI(mri) {}

  ~WebAssemblyMCCodeEmitter() override {}

  /// TableGen'erated function for getting the binary encoding for an
  /// instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// Return binary encoding of operand. If the machine operand requires
  /// relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  uint64_t getMemoryOpValue(const MCInst &MI, unsigned Op,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;

  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;
};
} // end anonymous namespace

MCCodeEmitter *llvm::createWebAssemblyMCCodeEmitter(const MCInstrInfo &MCII,
                                                    const MCRegisterInfo &MRI,
                                                    MCContext &Ctx) {
  return new WebAssemblyMCCodeEmitter(MCII, MRI, Ctx);
}

unsigned WebAssemblyMCCodeEmitter::getMachineOpValue(
    const MCInst &MI, const MCOperand &MO, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return MRI.getEncodingValue(MO.getReg());
  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  assert(MO.isExpr());

  assert(MO.getExpr()->getKind() == MCExpr::SymbolRef);

  assert(false && "FIXME: not implemented yet");

  return 0;
}

void WebAssemblyMCCodeEmitter::encodeInstruction(
    const MCInst &MI, raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  assert(false && "FIXME: not implemented yet");
}

// Encode WebAssembly Memory Operand
uint64_t
WebAssemblyMCCodeEmitter::getMemoryOpValue(const MCInst &MI, unsigned Op,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  assert(false && "FIXME: not implemented yet");
  return 0;
}

#include "WebAssemblyGenMCCodeEmitter.inc"
