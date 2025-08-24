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
// nts_t is either uint16_t or uint24_t based on whether large decoder table is
// enabled.
enum DecoderOps {
  OPC_ExtractField = 1,     // OPC_ExtractField(uleb128 Start, uint8_t Len)
  OPC_FilterValue,          // OPC_FilterValue(uleb128 Val, nts_t NumToSkip)
  OPC_FilterValueOrFail,    // OPC_FilterValueOrFail(uleb128 Val)
  OPC_CheckField,           // OPC_CheckField(uleb128 Start, uint8_t Len,
                            //                uleb128 Val, nts_t NumToSkip)
  OPC_CheckFieldOrFail,     // OPC_CheckFieldOrFail(uleb128 Start, uint8_t Len,
                            //                uleb128 Val)
  OPC_CheckPredicate,       // OPC_CheckPredicate(uleb128 PIdx, nts_t NumToSkip)
  OPC_CheckPredicateOrFail, // OPC_CheckPredicateOrFail(uleb128 PIdx)
  OPC_Decode,               // OPC_Decode(uleb128 Opcode, uleb128 DIdx)
  OPC_TryDecode,            // OPC_TryDecode(uleb128 Opcode, uleb128 DIdx,
                            //               nts_t NumToSkip)
  OPC_TryDecodeOrFail,      // OPC_TryDecodeOrFail(uleb128 Opcode, uleb128 DIdx)
  OPC_SoftFail,             // OPC_SoftFail(uleb128 PMask, uleb128 NMask)
  OPC_Fail                  // OPC_Fail()
};

} // namespace llvm::MCD

#endif
