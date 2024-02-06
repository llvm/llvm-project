//===--- AMDHSAKernelDescriptor.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"

using namespace llvm;
using namespace llvm::amdhsa;

void kernel_descriptor_t::bits_set(const MCExpr *&Dst, const MCExpr *Value,
                                   uint32_t Shift, uint32_t Mask,
                                   MCContext &Ctx) {
  auto Sft = MCConstantExpr::create(Shift, Ctx);
  auto Msk = MCConstantExpr::create(Mask, Ctx);
  Dst = MCBinaryExpr::createAnd(Dst, MCUnaryExpr::createNot(Msk, Ctx), Ctx);
  Dst = MCBinaryExpr::createOr(Dst, MCBinaryExpr::createShl(Value, Sft, Ctx),
                               Ctx);
}

const MCExpr *kernel_descriptor_t::bits_get(const MCExpr *Src, uint32_t Shift,
                                            uint32_t Mask, MCContext &Ctx) {
  auto Sft = MCConstantExpr::create(Shift, Ctx);
  auto Msk = MCConstantExpr::create(Mask, Ctx);
  return MCBinaryExpr::createLShr(MCBinaryExpr::createAnd(Src, Msk, Ctx), Sft,
                                  Ctx);
}
