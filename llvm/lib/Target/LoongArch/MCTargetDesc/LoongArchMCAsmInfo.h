//===-- LoongArchMCAsmInfo.h - LoongArch Asm Info --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the LoongArchMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCASMINFO_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {
class Triple;
class StringRef;

class LoongArchMCExpr : public MCSpecifierExpr {
public:
  using Specifier = uint16_t;
  enum { VK_None };

private:
  const bool RelaxHint;

  explicit LoongArchMCExpr(const MCExpr *Expr, Specifier S, bool Hint)
      : MCSpecifierExpr(Expr, S), RelaxHint(Hint) {}

public:
  static const LoongArchMCExpr *create(const MCExpr *Expr, uint16_t S,
                                       MCContext &Ctx, bool Hint = false);

  bool getRelaxHint() const { return RelaxHint; }
};

class LoongArchMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit LoongArchMCAsmInfo(const Triple &TargetTriple);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
};

namespace LoongArch {
uint16_t parseSpecifier(StringRef name);
} // namespace LoongArch

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCASMINFO_H
