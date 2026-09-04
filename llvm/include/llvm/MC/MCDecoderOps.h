//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Disassembler decoder state machine ops.
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCDECODEROPS_H
#define LLVM_MC_MCDECODEROPS_H

namespace llvm::MCD {

// Disassembler state machine opcodes.
enum DecoderOps {
  OPC_Scope = 1,      // OPC_Scope(uleb128 Size)
  OPC_SwitchField,    // OPC_SwitchField(uleb128 Start, uint8_t Len,
                      //                 [uleb128 Val, uleb128 Size]...)
  OPC_CheckField,     // OPC_CheckField(uleb128 Start, uint8_t Len, uleb128 Val)
  OPC_CheckPredicate, // OPC_CheckPredicate(uleb128 PIdx)
  OPC_Decode,         // OPC_Decode(uleb128 Opcode, uleb128 DIdx)
  OPC_SoftFail,       // OPC_SoftFail(uleb128 PMask, uleb128 NMask)
};

} // namespace llvm::MCD

#endif
