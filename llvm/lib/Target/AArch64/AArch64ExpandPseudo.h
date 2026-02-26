//===- AArch64ExpandPseudo.h - AArch64 Pseudo Expansion --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 pseudo-instruction expansion models.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64EXPANDPSEUDO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64EXPANDPSEUDO_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {

namespace AArch64_ExpandPseudo {

struct ImmInsnModel {
  unsigned Opcode;
  uint64_t Op1;
  uint64_t Op2;
};

struct AddrInsnModel {
  unsigned Opcode;
};

void expandMOVImm(uint64_t Imm, unsigned BitSize,
                  SmallVectorImpl<ImmInsnModel> &Insn);

void expandMOVAddr(unsigned Opcode, unsigned TargetFlags, bool IsTargetMachO,
                   SmallVectorImpl<AddrInsnModel> &Insn);

} // end namespace AArch64_ExpandPseudo

} // end namespace llvm

#endif
