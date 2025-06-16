//===-- AVRMCExpr.cpp - AVR specific MC expression classes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AVRMCExpr.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

namespace {

const struct ModifierEntry {
  const char *const Spelling;
  AVRMCExpr::Specifier specifier;
} ModifierNames[] = {
    {"lo8", AVRMCExpr::VK_LO8},       {"hi8", AVRMCExpr::VK_HI8},
    {"hh8", AVRMCExpr::VK_HH8}, // synonym with hlo8
    {"hlo8", AVRMCExpr::VK_HH8},      {"hhi8", AVRMCExpr::VK_HHI8},

    {"pm", AVRMCExpr::VK_PM},         {"pm_lo8", AVRMCExpr::VK_PM_LO8},
    {"pm_hi8", AVRMCExpr::VK_PM_HI8}, {"pm_hh8", AVRMCExpr::VK_PM_HH8},

    {"lo8_gs", AVRMCExpr::VK_LO8_GS}, {"hi8_gs", AVRMCExpr::VK_HI8_GS},
    {"gs", AVRMCExpr::VK_GS},
};

} // end of anonymous namespace

const AVRMCExpr *AVRMCExpr::create(Specifier Kind, const MCExpr *Expr,
                                   bool Negated, MCContext &Ctx) {
  return new (Ctx) AVRMCExpr(Kind, Expr, Negated);
}

void AVRMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  assert(specifier != VK_AVR_NONE);
  OS << getName() << '(';
  if (isNegated())
    OS << '-' << '(';
  MAI->printExpr(OS, *getSubExpr());
  if (isNegated())
    OS << ')';
  OS << ')';
}

bool AVRMCExpr::evaluateAsConstant(int64_t &Result) const {
  MCValue Value;

  bool isRelocatable = getSubExpr()->evaluateAsRelocatable(Value, nullptr);

  if (!isRelocatable)
    return false;

  if (Value.isAbsolute()) {
    Result = evaluateAsInt64(Value.getConstant());
    return true;
  }

  return false;
}

bool AVRMCExpr::evaluateAsRelocatableImpl(MCValue &Result,
                                          const MCAssembler *Asm) const {
  MCValue Value;
  bool isRelocatable = getSubExpr()->evaluateAsRelocatable(Value, Asm);
  if (!isRelocatable)
    return false;

  if (Value.isAbsolute()) {
    Result = MCValue::get(evaluateAsInt64(Value.getConstant()));
  } else {
    if (!Asm || !Asm->hasLayout())
      return false;

    auto Spec = AVRMCExpr::VK_None;
    if (Value.getSpecifier() != MCSymbolRefExpr::VK_None)
      return false;
    assert(!Value.getSubSym());
    if (specifier == VK_PM)
      Spec = AVRMCExpr::VK_PM;

    // TODO: don't attach specifier to MCSymbolRefExpr.
    Result =
        MCValue::get(Value.getAddSym(), nullptr, Value.getConstant(), Spec);
  }

  return true;
}

int64_t AVRMCExpr::evaluateAsInt64(int64_t Value) const {
  if (Negated)
    Value *= -1;

  switch (specifier) {
  case AVRMCExpr::VK_LO8:
    Value &= 0xff;
    break;
  case AVRMCExpr::VK_HI8:
    Value &= 0xff00;
    Value >>= 8;
    break;
  case AVRMCExpr::VK_HH8:
    Value &= 0xff0000;
    Value >>= 16;
    break;
  case AVRMCExpr::VK_HHI8:
    Value &= 0xff000000;
    Value >>= 24;
    break;
  case AVRMCExpr::VK_PM_LO8:
  case AVRMCExpr::VK_LO8_GS:
    Value >>= 1; // Program memory addresses must always be shifted by one.
    Value &= 0xff;
    break;
  case AVRMCExpr::VK_PM_HI8:
  case AVRMCExpr::VK_HI8_GS:
    Value >>= 1; // Program memory addresses must always be shifted by one.
    Value &= 0xff00;
    Value >>= 8;
    break;
  case AVRMCExpr::VK_PM_HH8:
    Value >>= 1; // Program memory addresses must always be shifted by one.
    Value &= 0xff0000;
    Value >>= 16;
    break;
  case AVRMCExpr::VK_PM:
  case AVRMCExpr::VK_GS:
    Value >>= 1; // Program memory addresses must always be shifted by one.
    break;

  case AVRMCExpr::VK_AVR_NONE:
  default:
    llvm_unreachable("Uninitialized expression.");
  }
  return static_cast<uint64_t>(Value) & 0xff;
}

AVR::Fixups AVRMCExpr::getFixupKind() const {
  AVR::Fixups Kind = AVR::Fixups::LastTargetFixupKind;

  switch (specifier) {
  case VK_LO8:
    Kind = isNegated() ? AVR::fixup_lo8_ldi_neg : AVR::fixup_lo8_ldi;
    break;
  case VK_HI8:
    Kind = isNegated() ? AVR::fixup_hi8_ldi_neg : AVR::fixup_hi8_ldi;
    break;
  case VK_HH8:
    Kind = isNegated() ? AVR::fixup_hh8_ldi_neg : AVR::fixup_hh8_ldi;
    break;
  case VK_HHI8:
    Kind = isNegated() ? AVR::fixup_ms8_ldi_neg : AVR::fixup_ms8_ldi;
    break;

  case VK_PM_LO8:
    Kind = isNegated() ? AVR::fixup_lo8_ldi_pm_neg : AVR::fixup_lo8_ldi_pm;
    break;
  case VK_PM_HI8:
    Kind = isNegated() ? AVR::fixup_hi8_ldi_pm_neg : AVR::fixup_hi8_ldi_pm;
    break;
  case VK_PM_HH8:
    Kind = isNegated() ? AVR::fixup_hh8_ldi_pm_neg : AVR::fixup_hh8_ldi_pm;
    break;
  case VK_PM:
  case VK_GS:
    Kind = AVR::fixup_16_pm;
    break;
  case VK_LO8_GS:
    Kind = AVR::fixup_lo8_ldi_gs;
    break;
  case VK_HI8_GS:
    Kind = AVR::fixup_hi8_ldi_gs;
    break;

  default:
    llvm_unreachable("Uninitialized expression");
  }

  return Kind;
}

const char *AVRMCExpr::getName() const {
  const auto &Modifier =
      llvm::find_if(ModifierNames, [this](ModifierEntry const &Mod) {
        return Mod.specifier == specifier;
      });

  if (Modifier != std::end(ModifierNames)) {
    return Modifier->Spelling;
  }
  return nullptr;
}

AVRMCExpr::Specifier AVRMCExpr::parseSpecifier(StringRef Name) {
  const auto &Modifier =
      llvm::find_if(ModifierNames, [&Name](ModifierEntry const &Mod) {
        return Mod.Spelling == Name;
      });

  if (Modifier != std::end(ModifierNames)) {
    return Modifier->specifier;
  }
  return VK_AVR_NONE;
}

} // end of namespace llvm
