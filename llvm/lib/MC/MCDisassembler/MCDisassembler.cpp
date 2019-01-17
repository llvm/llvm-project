//===- MCDisassembler.cpp - Disassembler interface ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

MCDisassembler::~MCDisassembler() = default;

MCDisassembler::DecodeStatus MCDisassembler::onSymbolStart(
    StringRef Name, uint64_t &Size, ArrayRef<uint8_t> Bytes, uint64_t Address,
    raw_ostream &VStream, raw_ostream &CStream) const {
  Size = 0;
  return MCDisassembler::Success;
}

bool MCDisassembler::tryAddingSymbolicOperand(MCInst &Inst, int64_t Value,
                                              uint64_t Address, bool IsBranch,
                                              uint64_t Offset,
                                              uint64_t InstSize) const {
  raw_ostream &cStream = CommentStream ? *CommentStream : nulls();
  if (Symbolizer)
    return Symbolizer->tryAddingSymbolicOperand(Inst, cStream, Value, Address,
                                                IsBranch, Offset, InstSize);
  return false;
}

void MCDisassembler::tryAddingPcLoadReferenceComment(int64_t Value,
                                                     uint64_t Address) const {
  raw_ostream &cStream = CommentStream ? *CommentStream : nulls();
  if (Symbolizer)
    Symbolizer->tryAddingPcLoadReferenceComment(cStream, Value, Address);
}

void MCDisassembler::setSymbolizer(std::unique_ptr<MCSymbolizer> Symzer) {
  Symbolizer = std::move(Symzer);
}
