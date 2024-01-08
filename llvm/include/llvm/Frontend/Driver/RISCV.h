//===--- RISCV.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines RISC-V frontend logic common to clang and flang
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_DRIVER_RISCV_H
#define LLVM_FRONTEND_DRIVER_RISCV_H

#include "llvm/Support/RISCVISAInfo.h"
#include <optional>

namespace llvm::driver::riscv {

std::optional<std::pair<unsigned, unsigned>>
getVScaleRange(const RISCVISAInfo &ISAInfo, unsigned ExplicitMin,
               unsigned ExplicitMax);

} // namespace llvm::driver::riscv

#endif
