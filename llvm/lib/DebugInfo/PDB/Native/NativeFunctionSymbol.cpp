//===- NativeFunctionSymbol.cpp - info about function symbols----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeFunctionSymbol.h"

#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypeEnum.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeFunctionSymbol::NativeFunctionSymbol(NativeSession &Session,
                                           SymIndexId Id,
                                           const codeview::ProcSym &Sym)
    : NativeRawSymbol(Session, PDB_SymType::Data, Id), Sym(Sym) {}

NativeFunctionSymbol::~NativeFunctionSymbol() {}

void NativeFunctionSymbol::dump(raw_ostream &OS, int Indent,
                                PdbSymbolIdField ShowIdFields,
                                PdbSymbolIdField RecurseIdFields) const {
  NativeRawSymbol::dump(OS, Indent, ShowIdFields, RecurseIdFields);
  dumpSymbolField(OS, "name", getName(), Indent);
  dumpSymbolField(OS, "length", getLength(), Indent);
  dumpSymbolField(OS, "offset", getAddressOffset(), Indent);
  dumpSymbolField(OS, "section", getAddressSection(), Indent);
}

uint32_t NativeFunctionSymbol::getAddressOffset() const {
  return Sym.CodeOffset;
}

uint32_t NativeFunctionSymbol::getAddressSection() const { return Sym.Segment; }
std::string NativeFunctionSymbol::getName() const {
  return std::string(Sym.Name);
}

PDB_SymType NativeFunctionSymbol::getSymTag() const {
  return PDB_SymType::Function;
}

uint64_t NativeFunctionSymbol::getLength() const { return Sym.CodeSize; }

uint32_t NativeFunctionSymbol::getRelativeVirtualAddress() const {
  return Session.getRVAFromSectOffset(Sym.Segment, Sym.CodeOffset);
}

uint64_t NativeFunctionSymbol::getVirtualAddress() const {
  return Session.getVAFromSectOffset(Sym.Segment, Sym.CodeOffset);
}
