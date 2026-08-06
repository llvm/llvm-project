//===-- SuperHAsmBackend.cpp - SuperH Assembler Backend ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuperHAsmBackend.h"
#include "SuperHFixupKinds.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

SuperHAsmBackend::SuperHAsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI) : MCAsmBackend(STI.getTargetTriple().isLittleEndian()
                         ? llvm::endianness::little
                         : llvm::endianness::big),
      STI(STI), OSABI(OSABI) {

}

bool SuperHAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const {

  // If the count is not 4-byte aligned, we must be writing data into the
  // text section (otherwise we have unaligned instructions, and thus have
  // far bigger problems), so just write zeros instead.
  OS.write_zeros(Count % 2);
  return true;
}
std::optional<MCFixupKind> SuperHAsmBackend::getFixupKind(StringRef Name) const {
  if (STI.getTargetTriple().isOSBinFormatELF()) {
    unsigned Type;
    Type = llvm::StringSwitch<unsigned>(Name)
#define ELF_RELOC(NAME, ID) .Case(#NAME, ID)
#include "llvm/BinaryFormat/ELFRelocs/SuperH.def"
#undef ELF_RELOC
               .Default(-1u);
    if (Type != -1u)
      return static_cast<MCFixupKind>(FirstLiteralRelocationKind + Type);
  }
  return std::nullopt;
}

MCFixupKindInfo SuperHAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[SuperH::NumTargetFixupKinds] = {
      // This table *must* be in same the order of fixup_* kinds in
      // AVRFixupKinds.h.
      //
      // name                    offset  bits  flags
      {"fixup_12_pcrel",         12,     16,   0},
      {"fixup_8_pcrel",          8,      16,   0},
      {"fixup_4_pcrel",          4,      16,   0},
  };

  if (mc::isRelocation(Kind))
    return {};

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < SuperH::NumTargetFixupKinds &&
         "Invalid kind!");

  return Infos[Kind - FirstTargetFixupKind];
}

void SuperHAsmBackend::applyFixup(const MCFragment &F, const MCFixup &Fixup,
                                 const MCValue &Target, uint8_t *Data,
                                 uint64_t Value, bool IsResolved) {
  
  // Handle Relocations
  IsResolved = addReloc(F, Fixup, Target, Value, IsResolved);
  MCFixupKind Kind = Fixup.getKind();
  if (mc::isRelocation(Kind))
    return;

  // Handle non-relocations
  MCContext &Ctx = getContext();
  MCFixupKindInfo Info = getFixupKindInfo(Kind);
  if (!Value)
    return; // No encoding change.

  unsigned NumBits = Info.TargetSize + Info.TargetOffset;
  unsigned NumBytes = (NumBits / 8) + ((NumBits % 8) == 0 ? 0 : 1);
  assert(Fixup.getOffset() + NumBytes <= F.getSize() &&
         "Invalid fixup offset!");

  // Flip the bits if neccesary, then spit them out
  Value <<= Info.TargetOffset;
  bool SwapValue = Endian == llvm::endianness::big;
  for (unsigned i = 0; i < NumBytes; ++i) {
    unsigned Idx = SwapValue ? (NumBytes - 1 - i) : i;
    uint8_t mask = (((Value >> (i * 8)) & 0xff));
    Data[Idx] |= mask;
  }
}

bool SuperHAsmBackend::addReloc(const MCFragment &F, const MCFixup &Fixup,
                               const MCValue &Target, uint64_t &FixedValue,
                               bool IsResolved) {
  
  MCValue PCITarget; // PC-Indirect Target

  // Get indirect target location.
  switch(Fixup.getKind()) {
  default: 
    return {};

  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8: {
    const auto *EValue = Fixup.getValue();
    if (!EValue->evaluateAsRelocatable(PCITarget, Asm))
      return true;
    break;
  }
  }

  // No target?
  if (!PCITarget.getAddSym())
    return false;

  // Evaluate as ELF.
  auto &SA = static_cast<const MCSymbolELF &>(*PCITarget.getAddSym());
  if (SA.isUndefined())
    return false;

  // Check if resolvable.
  IsResolved = &SA.getSection() == F.getParent() &&
                SA.getBinding() == ELF::STB_LOCAL &&
                SA.getType() != ELF::STT_GNU_IFUNC;
  if (!IsResolved)
    return false;

  // Calculate fixed offset value.
  // Note that the PC relative jumps are based on the start of the address.
  // So it must be subtracted from the fixed value.
  FixedValue = Asm->getSymbolOffset(SA) + PCITarget.getConstant();
  FixedValue -= (Asm->getFragmentOffset(F) + Fixup.getOffset());
  FixedValue /= 4; // Values are aligned to 4 bytes.
  return true;
}

std::unique_ptr<MCObjectTargetWriter> 
SuperHAsmBackend::createObjectTargetWriter() const {
  return createSuperHELFObjectWriter(OSABI);
}

MCAsmBackend *llvm::createSuperHAsmBackend(const Target &T,
                                          const MCSubtargetInfo &STI,
                                          const MCRegisterInfo &MRI,
                                          const MCTargetOptions &Options) {
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(STI.getTargetTriple().getOS());
  return new SuperHAsmBackend(STI, OSABI);
}