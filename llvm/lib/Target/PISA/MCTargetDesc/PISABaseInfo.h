//===-- PISABaseInfo.h - Top level PISA definitions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISABASEINFO_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISABASEINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {

// Return a string representation of the operands from startIndex onwards.
// Templated to allow both MachineInstr and MCInst to use the same logic.
template <class InstType>
std::string getPISAStringOperand(const InstType &MI, unsigned StartIndex) {
  std::string S; // Iteratively append to this string.

  const unsigned NumOps = MI.getNumOperands();
  bool IsFinished = false;
  for (unsigned I = StartIndex; I < NumOps && !IsFinished; ++I) {
    const auto &Op = MI.getOperand(I);
    if (!Op.isImm()) // Stop if we hit a register operand.
      break;
    assert((Op.getImm() >> 32) == 0 && "Imm operand should be i32 word");
    const uint32_t Imm = Op.getImm(); // Each i32 word is up to 4 characters.
    for (unsigned ShiftAmount = 0; ShiftAmount < 32; ShiftAmount += 8) {
      char C = (Imm >> ShiftAmount) & 0xff;
      if (C == 0) { // Stop if we hit a null-terminator character.
        IsFinished = true;
        break;
      }
      S += C; // Otherwise, append the character to the result string.
    }
  }
  return S;
}
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISABASEINFO_H
