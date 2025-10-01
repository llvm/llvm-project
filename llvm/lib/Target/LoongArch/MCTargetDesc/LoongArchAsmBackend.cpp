//===-- LoongArchAsmBackend.cpp - LoongArch Assembler Backend -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LoongArchAsmBackend class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchAsmBackend.h"
#include "LoongArchFixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "loongarch-asmbackend"

using namespace llvm;

LoongArchAsmBackend::LoongArchAsmBackend(const MCSubtargetInfo &STI,
                                         uint8_t OSABI, bool Is64Bit,
                                         const MCTargetOptions &Options)
    : MCAsmBackend(llvm::endianness::little), STI(STI), OSABI(OSABI),
      Is64Bit(Is64Bit), TargetOptions(Options) {}

std::optional<MCFixupKind>
LoongArchAsmBackend::getFixupKind(StringRef Name) const {
  if (STI.getTargetTriple().isOSBinFormatELF()) {
    auto Type = llvm::StringSwitch<unsigned>(Name)
#define ELF_RELOC(X, Y) .Case(#X, Y)
#include "llvm/BinaryFormat/ELFRelocs/LoongArch.def"
#undef ELF_RELOC
                    .Case("BFD_RELOC_NONE", ELF::R_LARCH_NONE)
                    .Case("BFD_RELOC_32", ELF::R_LARCH_32)
                    .Case("BFD_RELOC_64", ELF::R_LARCH_64)
                    .Default(-1u);
    if (Type != -1u)
      return static_cast<MCFixupKind>(FirstLiteralRelocationKind + Type);
  }
  return std::nullopt;
}

MCFixupKindInfo LoongArchAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[] = {
      // This table *must* be in the order that the fixup_* kinds are defined in
      // LoongArchFixupKinds.h.
      //
      // {name, offset, bits, flags}
      {"fixup_loongarch_b16", 10, 16, 0},
      {"fixup_loongarch_b21", 0, 26, 0},
      {"fixup_loongarch_b26", 0, 26, 0},
      {"fixup_loongarch_abs_hi20", 5, 20, 0},
      {"fixup_loongarch_abs_lo12", 10, 12, 0},
      {"fixup_loongarch_abs64_lo20", 5, 20, 0},
      {"fixup_loongarch_abs64_hi12", 10, 12, 0},
  };

  static_assert((std::size(Infos)) == LoongArch::NumTargetFixupKinds,
                "Not all fixup kinds added to Infos array");

  // Fixup kinds from .reloc directive are like R_LARCH_NONE. They
  // do not require any extra processing.
  if (mc::isRelocation(Kind))
    return {};

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) <
             LoongArch::NumTargetFixupKinds &&
         "Invalid kind!");
  return Infos[Kind - FirstTargetFixupKind];
}

static void reportOutOfRangeError(MCContext &Ctx, SMLoc Loc, unsigned N) {
  Ctx.reportError(Loc, "fixup value out of range [" + Twine(llvm::minIntN(N)) +
                           ", " + Twine(llvm::maxIntN(N)) + "]");
}

static uint64_t adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext &Ctx) {
  switch (Fixup.getKind()) {
  default:
    llvm_unreachable("Unknown fixup kind");
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
  case FK_Data_leb128:
    return Value;
  case LoongArch::fixup_loongarch_b16: {
    if (!isInt<18>(Value))
      reportOutOfRangeError(Ctx, Fixup.getLoc(), 18);
    if (Value % 4)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned");
    return (Value >> 2) & 0xffff;
  }
  case LoongArch::fixup_loongarch_b21: {
    if (!isInt<23>(Value))
      reportOutOfRangeError(Ctx, Fixup.getLoc(), 23);
    if (Value % 4)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned");
    return ((Value & 0x3fffc) << 8) | ((Value >> 18) & 0x1f);
  }
  case LoongArch::fixup_loongarch_b26: {
    if (!isInt<28>(Value))
      reportOutOfRangeError(Ctx, Fixup.getLoc(), 28);
    if (Value % 4)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned");
    return ((Value & 0x3fffc) << 8) | ((Value >> 18) & 0x3ff);
  }
  case LoongArch::fixup_loongarch_abs_hi20:
    return (Value >> 12) & 0xfffff;
  case LoongArch::fixup_loongarch_abs_lo12:
    return Value & 0xfff;
  case LoongArch::fixup_loongarch_abs64_lo20:
    return (Value >> 32) & 0xfffff;
  case LoongArch::fixup_loongarch_abs64_hi12:
    return (Value >> 52) & 0xfff;
  }
}

static void fixupLeb128(MCContext &Ctx, const MCFixup &Fixup, uint8_t *Data,
                        uint64_t Value) {
  unsigned I;
  for (I = 0; Value; ++I, Value >>= 7)
    Data[I] |= uint8_t(Value & 0x7f);
  if (Value)
    Ctx.reportError(Fixup.getLoc(), "Invalid uleb128 value!");
}

void LoongArchAsmBackend::applyFixup(const MCFragment &F, const MCFixup &Fixup,
                                     const MCValue &Target, uint8_t *Data,
                                     uint64_t Value, bool IsResolved) {
  IsResolved = addReloc(F, Fixup, Target, Value, IsResolved);
  if (!Value)
    return; // Doesn't change encoding.

  auto Kind = Fixup.getKind();
  if (mc::isRelocation(Kind))
    return;
  MCFixupKindInfo Info = getFixupKindInfo(Kind);
  MCContext &Ctx = getContext();

  // Fixup leb128 separately.
  if (Fixup.getKind() == FK_Data_leb128)
    return fixupLeb128(Ctx, Fixup, Data, Value);

  // Apply any target-specific value adjustments.
  Value = adjustFixupValue(Fixup, Value, Ctx);

  // Shift the value into position.
  Value <<= Info.TargetOffset;

  unsigned NumBytes = alignTo(Info.TargetSize + Info.TargetOffset, 8) / 8;

  assert(Fixup.getOffset() + NumBytes <= F.getSize() &&
         "Invalid fixup offset!");
  // For each byte of the fragment that the fixup touches, mask in the
  // bits from the fixup value.
  for (unsigned I = 0; I != NumBytes; ++I) {
    Data[I] |= uint8_t((Value >> (I * 8)) & 0xff);
  }
}

static inline std::pair<MCFixupKind, MCFixupKind>
getRelocPairForSize(unsigned Size) {
  switch (Size) {
  default:
    llvm_unreachable("unsupported fixup size");
  case 6:
    return std::make_pair(ELF::R_LARCH_ADD6, ELF::R_LARCH_SUB6);
  case 8:
    return std::make_pair(ELF::R_LARCH_ADD8, ELF::R_LARCH_SUB8);
  case 16:
    return std::make_pair(ELF::R_LARCH_ADD16, ELF::R_LARCH_SUB16);
  case 32:
    return std::make_pair(ELF::R_LARCH_ADD32, ELF::R_LARCH_SUB32);
  case 64:
    return std::make_pair(ELF::R_LARCH_ADD64, ELF::R_LARCH_SUB64);
  case 128:
    return std::make_pair(ELF::R_LARCH_ADD_ULEB128, ELF::R_LARCH_SUB_ULEB128);
  }
}

// Check if an R_LARCH_ALIGN relocation is needed for an alignment directive.
// If conditions are met, compute the padding size and create a fixup encoding
// the padding size in the addend. If MaxBytesToEmit is smaller than the padding
// size, the fixup encodes MaxBytesToEmit in the higher bits and references a
// per-section marker symbol.
bool LoongArchAsmBackend::relaxAlign(MCFragment &F, unsigned &Size) {
  // Alignments before the first linker-relaxable instruction have fixed sizes
  // and do not require relocations. Alignments after a linker-relaxable
  // instruction require a relocation, even if the STI specifies norelax.
  //
  // firstLinkerRelaxable is the layout order within the subsection, which may
  // be smaller than the section's order. Therefore, alignments in a
  // lower-numbered subsection may be unnecessarily treated as linker-relaxable.
  auto *Sec = F.getParent();
  if (F.getLayoutOrder() <= Sec->firstLinkerRelaxable())
    return false;

  // Use default handling unless linker relaxation is enabled and the
  // MaxBytesToEmit >= the nop size.
  const unsigned MinNopLen = 4;
  unsigned MaxBytesToEmit = F.getAlignMaxBytesToEmit();
  if (MaxBytesToEmit < MinNopLen)
    return false;

  Size = F.getAlignment().value() - MinNopLen;
  if (F.getAlignment() <= MinNopLen)
    return false;

  MCContext &Ctx = getContext();
  const MCExpr *Expr = nullptr;
  if (MaxBytesToEmit >= Size) {
    Expr = MCConstantExpr::create(Size, getContext());
  } else {
    MCSection *Sec = F.getParent();
    const MCSymbolRefExpr *SymRef = getSecToAlignSym()[Sec];
    if (SymRef == nullptr) {
      // Define a marker symbol at the section with an offset of 0.
      MCSymbol *Sym = Ctx.createNamedTempSymbol("la-relax-align");
      Sym->setFragment(&*Sec->getBeginSymbol()->getFragment());
      Asm->registerSymbol(*Sym);
      SymRef = MCSymbolRefExpr::create(Sym, Ctx);
      getSecToAlignSym()[Sec] = SymRef;
    }
    Expr = MCBinaryExpr::createAdd(
        SymRef,
        MCConstantExpr::create((MaxBytesToEmit << 8) | Log2(F.getAlignment()),
                               Ctx),
        Ctx);
  }
  MCFixup Fixup =
      MCFixup::create(0, Expr, FirstLiteralRelocationKind + ELF::R_LARCH_ALIGN);
  F.setVarFixups({Fixup});
  F.setLinkerRelaxable();
  return true;
}

std::pair<bool, bool> LoongArchAsmBackend::relaxLEB128(MCFragment &F,
                                                       int64_t &Value) const {
  const MCExpr &Expr = F.getLEBValue();
  if (F.isLEBSigned() || !Expr.evaluateKnownAbsolute(Value, *Asm))
    return std::make_pair(false, false);
  F.setVarFixups({MCFixup::create(0, &Expr, FK_Data_leb128)});
  return std::make_pair(true, true);
}

bool LoongArchAsmBackend::relaxDwarfLineAddr(MCFragment &F) const {
  MCContext &C = getContext();
  int64_t LineDelta = F.getDwarfLineDelta();
  const MCExpr &AddrDelta = F.getDwarfAddrDelta();
  int64_t Value;
  if (AddrDelta.evaluateAsAbsolute(Value, *Asm))
    return false;
  [[maybe_unused]] bool IsAbsolute =
      AddrDelta.evaluateKnownAbsolute(Value, *Asm);
  assert(IsAbsolute);

  SmallVector<char> Data;
  raw_svector_ostream OS(Data);

  // INT64_MAX is a signal that this is actually a DW_LNE_end_sequence.
  if (LineDelta != INT64_MAX) {
    OS << uint8_t(dwarf::DW_LNS_advance_line);
    encodeSLEB128(LineDelta, OS);
  }

  // According to the DWARF specification, the `DW_LNS_fixed_advance_pc` opcode
  // takes a single unsigned half (unencoded) operand. The maximum encodable
  // value is therefore 65535.  Set a conservative upper bound for relaxation.
  unsigned PCBytes;
  if (Value > 60000) {
    unsigned PtrSize = C.getAsmInfo()->getCodePointerSize();
    assert((PtrSize == 4 || PtrSize == 8) && "Unexpected pointer size");
    PCBytes = PtrSize;
    OS << uint8_t(dwarf::DW_LNS_extended_op) << uint8_t(PtrSize + 1)
       << uint8_t(dwarf::DW_LNE_set_address);
    OS.write_zeros(PtrSize);
  } else {
    PCBytes = 2;
    OS << uint8_t(dwarf::DW_LNS_fixed_advance_pc);
    support::endian::write<uint16_t>(OS, 0, llvm::endianness::little);
  }
  auto Offset = OS.tell() - PCBytes;

  if (LineDelta == INT64_MAX) {
    OS << uint8_t(dwarf::DW_LNS_extended_op);
    OS << uint8_t(1);
    OS << uint8_t(dwarf::DW_LNE_end_sequence);
  } else {
    OS << uint8_t(dwarf::DW_LNS_copy);
  }

  F.setVarContents(Data);
  F.setVarFixups({MCFixup::create(Offset, &AddrDelta,
                                  MCFixup::getDataKindForSize(PCBytes))});
  return true;
}

bool LoongArchAsmBackend::relaxDwarfCFA(MCFragment &F) const {
  const MCExpr &AddrDelta = F.getDwarfAddrDelta();
  SmallVector<MCFixup, 2> Fixups;
  int64_t Value;
  if (AddrDelta.evaluateAsAbsolute(Value, *Asm))
    return false;
  bool IsAbsolute = AddrDelta.evaluateKnownAbsolute(Value, *Asm);
  assert(IsAbsolute && "CFA with invalid expression");
  (void)IsAbsolute;

  assert(getContext().getAsmInfo()->getMinInstAlignment() == 1 &&
         "expected 1-byte alignment");
  if (Value == 0) {
    F.clearVarContents();
    F.clearVarFixups();
    return true;
  }

  auto AddFixups = [&Fixups,
                    &AddrDelta](unsigned Offset,
                                std::pair<MCFixupKind, MCFixupKind> FK) {
    const MCBinaryExpr &MBE = cast<MCBinaryExpr>(AddrDelta);
    Fixups.push_back(MCFixup::create(Offset, MBE.getLHS(), std::get<0>(FK)));
    Fixups.push_back(MCFixup::create(Offset, MBE.getRHS(), std::get<1>(FK)));
  };

  SmallVector<char, 8> Data;
  raw_svector_ostream OS(Data);
  if (isUIntN(6, Value)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc);
    AddFixups(0, getRelocPairForSize(6));
  } else if (isUInt<8>(Value)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc1);
    support::endian::write<uint8_t>(OS, 0, llvm::endianness::little);
    AddFixups(1, getRelocPairForSize(8));
  } else if (isUInt<16>(Value)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc2);
    support::endian::write<uint16_t>(OS, 0, llvm::endianness::little);
    AddFixups(1, getRelocPairForSize(16));
  } else if (isUInt<32>(Value)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc4);
    support::endian::write<uint32_t>(OS, 0, llvm::endianness::little);
    AddFixups(1, getRelocPairForSize(32));
  } else {
    llvm_unreachable("unsupported CFA encoding");
  }
  F.setVarContents(Data);
  F.setVarFixups(Fixups);
  return true;
}

bool LoongArchAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                       const MCSubtargetInfo *STI) const {
  // We mostly follow binutils' convention here: align to 4-byte boundary with a
  // 0-fill padding.
  OS.write_zeros(Count % 4);

  // The remainder is now padded with 4-byte nops.
  // nop: andi r0, r0, 0
  for (; Count >= 4; Count -= 4)
    OS.write("\0\0\x40\x03", 4);

  return true;
}

bool LoongArchAsmBackend::isPCRelFixupResolved(const MCSymbol *SymA,
                                               const MCFragment &F) {
  // If the section does not contain linker-relaxable fragments, PC-relative
  // fixups can be resolved.
  if (!F.getParent()->isLinkerRelaxable())
    return true;

  // Otherwise, check if the offset between the symbol and fragment is fully
  // resolved, unaffected by linker-relaxable fragments (e.g. instructions or
  // offset-affected FT_Align fragments). Complements the generic
  // isSymbolRefDifferenceFullyResolvedImpl.
  if (!PCRelTemp)
    PCRelTemp = getContext().createTempSymbol();
  PCRelTemp->setFragment(const_cast<MCFragment *>(&F));
  MCValue Res;
  MCExpr::evaluateSymbolicAdd(Asm, false, MCValue::get(SymA),
                              MCValue::get(nullptr, PCRelTemp), Res);
  return !Res.getSubSym();
}

bool LoongArchAsmBackend::addReloc(const MCFragment &F, const MCFixup &Fixup,
                                   const MCValue &Target, uint64_t &FixedValue,
                                   bool IsResolved) {
  auto Fallback = [&]() {
    MCAsmBackend::maybeAddReloc(F, Fixup, Target, FixedValue, IsResolved);
    return true;
  };
  uint64_t FixedValueA, FixedValueB;
  if (Target.getSubSym()) {
    assert(Target.getSpecifier() == 0 &&
           "relocatable SymA-SymB cannot have relocation specifier");
    std::pair<MCFixupKind, MCFixupKind> FK;
    const MCSymbol &SA = *Target.getAddSym();
    const MCSymbol &SB = *Target.getSubSym();

    bool force = !SA.isInSection() || !SB.isInSection();
    if (!force) {
      const MCSection &SecA = SA.getSection();
      const MCSection &SecB = SB.getSection();
      const MCSection &SecCur = *F.getParent();

      // To handle the case of A - B which B is same section with the current,
      // generate PCRel relocations is better than ADD/SUB relocation pair.
      // We can resolve it as A - PC + PC - B. The A - PC will be resolved
      // as a PCRel relocation, while PC - B will serve as the addend.
      // If the linker relaxation is disabled, it can be done directly since
      // PC - B is constant. Otherwise, we should evaluate whether PC - B
      // is constant. If it can be resolved as PCRel, use Fallback which
      // generates R_LARCH_{32,64}_PCREL relocation later.
      if (&SecA != &SecB && &SecB == &SecCur &&
          isPCRelFixupResolved(Target.getSubSym(), F))
        return Fallback();

      // In SecA == SecB case. If the section is not linker-relaxable, the
      // FixedValue has already been calculated out in evaluateFixup,
      // return true and avoid record relocations.
      if (&SecA == &SecB && !SecA.isLinkerRelaxable())
        return true;
    }

    switch (Fixup.getKind()) {
    case llvm::FK_Data_1:
      FK = getRelocPairForSize(8);
      break;
    case llvm::FK_Data_2:
      FK = getRelocPairForSize(16);
      break;
    case llvm::FK_Data_4:
      FK = getRelocPairForSize(32);
      break;
    case llvm::FK_Data_8:
      FK = getRelocPairForSize(64);
      break;
    case llvm::FK_Data_leb128:
      FK = getRelocPairForSize(128);
      break;
    default:
      llvm_unreachable("unsupported fixup size");
    }
    MCValue A = MCValue::get(Target.getAddSym(), nullptr, Target.getConstant());
    MCValue B = MCValue::get(Target.getSubSym());
    auto FA = MCFixup::create(Fixup.getOffset(), nullptr, std::get<0>(FK));
    auto FB = MCFixup::create(Fixup.getOffset(), nullptr, std::get<1>(FK));
    Asm->getWriter().recordRelocation(F, FA, A, FixedValueA);
    Asm->getWriter().recordRelocation(F, FB, B, FixedValueB);
    FixedValue = FixedValueA - FixedValueB;
    return false;
  }

  // If linker relaxation is enabled and supported by the current relocation,
  // generate a relocation and then append a RELAX.
  if (Fixup.isLinkerRelaxable())
    IsResolved = false;
  if (IsResolved && Fixup.isPCRel())
    IsResolved = isPCRelFixupResolved(Target.getAddSym(), F);

  if (!IsResolved)
    Asm->getWriter().recordRelocation(F, Fixup, Target, FixedValue);

  if (Fixup.isLinkerRelaxable()) {
    auto FA = MCFixup::create(Fixup.getOffset(), nullptr, ELF::R_LARCH_RELAX);
    Asm->getWriter().recordRelocation(F, FA, MCValue::get(nullptr),
                                      FixedValueA);
  }

  return true;
}

std::unique_ptr<MCObjectTargetWriter>
LoongArchAsmBackend::createObjectTargetWriter() const {
  return createLoongArchELFObjectWriter(OSABI, Is64Bit);
}

MCAsmBackend *llvm::createLoongArchAsmBackend(const Target &T,
                                              const MCSubtargetInfo &STI,
                                              const MCRegisterInfo &MRI,
                                              const MCTargetOptions &Options) {
  const Triple &TT = STI.getTargetTriple();
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  return new LoongArchAsmBackend(STI, OSABI, TT.isArch64Bit(), Options);
}
