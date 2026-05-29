//===- BitcodeEncoding.h - AIR bitstream encoding helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Header-only helpers that map LLVM IR enums to the small integer codes the
/// AIR bitstream expects in INST_BINOP, INST_CAST and MODULE_GLOBAL records.
/// Kept inline because every helper is hot in the FunctionWriter inner loop
/// and the mapping is fixed by the Apple bitcode reader rather than by LLVM
/// (so the values cannot share LLVM's own `Bitcode/LLVMBitCodes.h` constants).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEENCODING_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEENCODING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"

namespace llvm {
namespace metal {

/// Map an LLVM `Instruction::BinaryOps` opcode to the AIR INST_BINOP code.
/// Unknown opcodes return 0 (Add) because the bitstream record requires a
/// value; callers should pre-filter unsupported binops before reaching here.
inline unsigned encodeBinop(unsigned Op) {
  using namespace llvm;
  switch (Op) {
  case Instruction::Add:
  case Instruction::FAdd:
    return 0;
  case Instruction::Sub:
  case Instruction::FSub:
    return 1;
  case Instruction::Mul:
  case Instruction::FMul:
    return 2;
  case Instruction::UDiv:
    return 3;
  case Instruction::SDiv:
  case Instruction::FDiv:
    return 4;
  case Instruction::URem:
    return 5;
  case Instruction::SRem:
  case Instruction::FRem:
    return 6;
  case Instruction::Shl:
    return 7;
  case Instruction::LShr:
    return 8;
  case Instruction::AShr:
    return 9;
  case Instruction::And:
    return 10;
  case Instruction::Or:
    return 11;
  case Instruction::Xor:
    return 12;
  default:
    return 0;
  }
}

/// Map an LLVM `Instruction::CastOps` opcode to the AIR INST_CAST code.
/// See `encodeBinop` for the fallback rationale.
inline unsigned encodeCast(unsigned Op) {
  using namespace llvm;
  switch (Op) {
  case Instruction::Trunc:
    return 0;
  case Instruction::ZExt:
    return 1;
  case Instruction::SExt:
    return 2;
  case Instruction::FPToUI:
    return 3;
  case Instruction::FPToSI:
    return 4;
  case Instruction::UIToFP:
    return 5;
  case Instruction::SIToFP:
    return 6;
  case Instruction::FPTrunc:
    return 7;
  case Instruction::FPExt:
    return 8;
  case Instruction::PtrToInt:
    return 9;
  case Instruction::IntToPtr:
    return 10;
  case Instruction::BitCast:
    return 11;
  case Instruction::AddrSpaceCast:
    return 12;
  default:
    return 0;
  }
}

/// Map an LLVM `GlobalValue::LinkageTypes` enumerator to the AIR
/// MODULE_GLOBAL linkage field. Only the linkages the Triton frontend emits
/// are listed; anything else collapses to External so the metallib loads.
inline unsigned encodeLinkage(llvm::GlobalValue::LinkageTypes L) {
  using namespace llvm;
  switch (L) {
  case GlobalValue::ExternalLinkage:
    return 0;
  case GlobalValue::WeakAnyLinkage:
    return 1;
  case GlobalValue::InternalLinkage:
    return 3;
  case GlobalValue::PrivateLinkage:
    return 9;
  default:
    return 0;
  }
}

/// Emit a string (e.g. a symbol name) as a single bitstream record under
/// `Code`, one VBR-encoded character per operand.
inline void emitString(llvm::BitstreamWriter &W, unsigned Code,
                       llvm::StringRef S) {
  llvm::SmallVector<uint64_t, 64> V;
  for (char C : S)
    V.push_back((uint64_t)(unsigned char)C);
  W.EmitRecord(Code, V);
}

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEENCODING_H
