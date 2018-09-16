//===- NativeTypePointer.cpp - info about pointer type ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeTypePointer.h"

#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"

#include <cassert>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeTypePointer::NativeTypePointer(NativeSession &Session, SymIndexId Id,
                                     codeview::TypeIndex TI,
                                     codeview::PointerRecord Record)
    : NativeRawSymbol(Session, PDB_SymType::PointerType, Id), TI(TI),
      Record(std::move(Record)) {}

NativeTypePointer::~NativeTypePointer() {}

void NativeTypePointer::dump(raw_ostream &OS, int Indent) const {
  NativeRawSymbol::dump(OS, Indent);

  dumpSymbolField(OS, "lexicalParentId", 0, Indent);
  dumpSymbolField(OS, "typeId", getTypeId(), Indent);
  dumpSymbolField(OS, "length", getLength(), Indent);
  dumpSymbolField(OS, "constType", isConstType(), Indent);
  dumpSymbolField(OS, "isPointerToDataMember", isPointerToDataMember(), Indent);
  dumpSymbolField(OS, "isPointerToMemberFunction", isPointerToMemberFunction(),
                  Indent);
  dumpSymbolField(OS, "RValueReference", isRValueReference(), Indent);
  dumpSymbolField(OS, "reference", isReference(), Indent);
  dumpSymbolField(OS, "restrictedType", isRestrictedType(), Indent);
  dumpSymbolField(OS, "unalignedType", isUnalignedType(), Indent);
  dumpSymbolField(OS, "volatileType", isVolatileType(), Indent);
}

bool NativeTypePointer::isConstType() const { return false; }

uint64_t NativeTypePointer::getLength() const { return Record.getSize(); }

SymIndexId NativeTypePointer::getTypeId() const {
  // This is the pointee SymIndexId.
  return Session.getSymbolCache().findSymbolByTypeIndex(Record.ReferentType);
}

bool NativeTypePointer::isReference() const {
  return Record.getMode() == PointerMode::LValueReference ||
         isRValueReference();
}

bool NativeTypePointer::isRValueReference() const {
  return Record.getMode() == PointerMode::RValueReference;
}

bool NativeTypePointer::isPointerToDataMember() const {
  return Record.getMode() == PointerMode::PointerToDataMember;
}

bool NativeTypePointer::isPointerToMemberFunction() const {
  return Record.getMode() == PointerMode::PointerToMemberFunction;
}

bool NativeTypePointer::isRestrictedType() const { return false; }

bool NativeTypePointer::isVolatileType() const { return false; }

bool NativeTypePointer::isUnalignedType() const { return false; }
