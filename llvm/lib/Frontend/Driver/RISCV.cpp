//===--- RISCV.cpp - Shared RISC-V frontend logic -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Driver/RISCV.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

namespace llvm::driver::riscv {

std::optional<std::pair<unsigned, unsigned>>
getVScaleRange(const RISCVISAInfo &ISAInfo, unsigned ExplicitMin,
               unsigned ExplicitMax) {
  // RISCV::RVVBitsPerBlock is 64.
  unsigned VScaleMin = ISAInfo.getMinVLen() / RISCV::RVVBitsPerBlock;

  if (ExplicitMin || ExplicitMax) {
    // Treat Zvl*b as a lower bound on vscale.
    VScaleMin = std::max(VScaleMin, ExplicitMin);
    unsigned VScaleMax = ExplicitMax;
    if (VScaleMax != 0 && VScaleMax < VScaleMin)
      VScaleMax = VScaleMin;
    return std::pair<unsigned, unsigned>(VScaleMin ? VScaleMin : 1, VScaleMax);
  }

  if (VScaleMin > 0) {
    unsigned VScaleMax = ISAInfo.getMaxVLen() / RISCV::RVVBitsPerBlock;
    return std::make_pair(VScaleMin, VScaleMax);
  }

  return std::nullopt;
}

} // namespace llvm::driver::riscv
