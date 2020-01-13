//===- MCDisassembler.cpp - Disassembler interface ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

MCDisassembler::~MCDisassembler() = default;

MCDisassembler::DecodeStatus
MCDisassembler::onSymbolStart(StringRef Name, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const {
  Size = 0;
  return MCDisassembler::Success;
}

bool MCDisassembler::tryAddingSymbolicOperand(MCInst &Inst, int64_t Value,
                                              uint64_t Address, bool IsBranch,
                                              uint64_t Offset,
                                              uint64_t InstSize) const {
  if (Symbolizer)
    return Symbolizer->tryAddingSymbolicOperand(
        Inst, *CommentStream, Value, Address, IsBranch, Offset, InstSize);
  return false;
}

void MCDisassembler::tryAddingPcLoadReferenceComment(int64_t Value,
                                                     uint64_t Address) const {
  if (Symbolizer)
    Symbolizer->tryAddingPcLoadReferenceComment(*CommentStream, Value, Address);
}

void MCDisassembler::setSymbolizer(std::unique_ptr<MCSymbolizer> Symzer) {
  Symbolizer = std::move(Symzer);
}
