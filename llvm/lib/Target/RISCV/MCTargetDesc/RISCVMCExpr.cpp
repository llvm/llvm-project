//===-- RISCVMCExpr.cpp - RISC-V specific MC expression classes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the RISC-V architecture (e.g. ":lo12:", ":gottprel_g1:", ...).
//
//===----------------------------------------------------------------------===//

#include "RISCVMCExpr.h"
#include "MCTargetDesc/RISCVAsmBackend.h"
#include "RISCVFixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "riscvmcexpr"

const RISCVMCExpr *RISCVMCExpr::create(const MCExpr *Expr, Specifier S,
                                       MCContext &Ctx) {
  return new (Ctx) RISCVMCExpr(Expr, S);
}

void RISCVMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  Specifier S = getSpecifier();
  bool HasVariant = ((S != VK_None) && (S != VK_CALL) && (S != VK_CALL_PLT));

  if (HasVariant)
    OS << '%' << getSpecifierName(S) << '(';
  Expr->print(OS, MAI);
  if (HasVariant)
    OS << ')';
}

const MCFixup *RISCVMCExpr::getPCRelHiFixup(const MCFragment **DFOut) const {
  MCValue AUIPCLoc;
  if (!getSubExpr()->evaluateAsRelocatable(AUIPCLoc, nullptr))
    return nullptr;

  const MCSymbol *AUIPCSymbol = AUIPCLoc.getAddSym();
  if (!AUIPCSymbol)
    return nullptr;
  const auto *DF = dyn_cast_or_null<MCDataFragment>(AUIPCSymbol->getFragment());

  if (!DF)
    return nullptr;

  uint64_t Offset = AUIPCSymbol->getOffset();
  if (DF->getContents().size() == Offset) {
    DF = dyn_cast_or_null<MCDataFragment>(DF->getNext());
    if (!DF)
      return nullptr;
    Offset = 0;
  }

  for (const MCFixup &F : DF->getFixups()) {
    if (F.getOffset() != Offset)
      continue;
    auto Kind = F.getTargetKind();
    if (!mc::isRelocation(F.getKind())) {
      if (Kind == RISCV::fixup_riscv_pcrel_hi20) {
        *DFOut = DF;
        return &F;
      }
      break;
    }
    switch (Kind) {
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_TLS_GOT_HI20:
    case ELF::R_RISCV_TLS_GD_HI20:
    case ELF::R_RISCV_TLSDESC_HI20:
      *DFOut = DF;
      return &F;
    }
  }

  return nullptr;
}

bool RISCVMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                            const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(specifier);

  // Custom fixup types are not valid with symbol difference expressions.
  return !Res.getSubSym();
}

void RISCVMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

std::optional<RISCVMCExpr::Specifier>
RISCVMCExpr::getSpecifierForName(StringRef name) {
  return StringSwitch<std::optional<RISCVMCExpr::Specifier>>(name)
      .Case("lo", VK_LO)
      .Case("hi", VK_HI)
      .Case("pcrel_lo", VK_PCREL_LO)
      .Case("pcrel_hi", VK_PCREL_HI)
      .Case("got_pcrel_hi", VK_GOT_HI)
      .Case("tprel_lo", VK_TPREL_LO)
      .Case("tprel_hi", VK_TPREL_HI)
      .Case("tprel_add", VK_TPREL_ADD)
      .Case("tls_ie_pcrel_hi", VK_TLS_GOT_HI)
      .Case("tls_gd_pcrel_hi", VK_TLS_GD_HI)
      .Case("tlsdesc_hi", VK_TLSDESC_HI)
      .Case("tlsdesc_load_lo", VK_TLSDESC_LOAD_LO)
      .Case("tlsdesc_add_lo", VK_TLSDESC_ADD_LO)
      .Case("tlsdesc_call", VK_TLSDESC_CALL)
      .Case("qc.abs20", VK_QC_ABS20)
      // Used in data directives
      .Case("pltpcrel", VK_PLTPCREL)
      .Case("gotpcrel", VK_GOTPCREL)
      .Default(std::nullopt);
}

StringRef RISCVMCExpr::getSpecifierName(Specifier S) {
  switch (S) {
  case VK_None:
    llvm_unreachable("not used as %specifier()");
  case VK_LO:
    return "lo";
  case VK_HI:
    return "hi";
  case VK_PCREL_LO:
    return "pcrel_lo";
  case VK_PCREL_HI:
    return "pcrel_hi";
  case VK_GOT_HI:
    return "got_pcrel_hi";
  case VK_TPREL_LO:
    return "tprel_lo";
  case VK_TPREL_HI:
    return "tprel_hi";
  case VK_TPREL_ADD:
    return "tprel_add";
  case VK_TLS_GOT_HI:
    return "tls_ie_pcrel_hi";
  case VK_TLSDESC_HI:
    return "tlsdesc_hi";
  case VK_TLSDESC_LOAD_LO:
    return "tlsdesc_load_lo";
  case VK_TLSDESC_ADD_LO:
    return "tlsdesc_add_lo";
  case VK_TLSDESC_CALL:
    return "tlsdesc_call";
  case VK_TLS_GD_HI:
    return "tls_gd_pcrel_hi";
  case VK_CALL:
    return "call";
  case VK_CALL_PLT:
    return "call_plt";
  case VK_32_PCREL:
    return "32_pcrel";
  case VK_GOTPCREL:
    return "gotpcrel";
  case VK_PLTPCREL:
    return "pltpcrel";
  case VK_QC_ABS20:
    return "qc.abs20";
  }
  llvm_unreachable("Invalid ELF symbol kind");
}
