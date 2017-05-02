//===-- CompactTypeDumpVisitor.cpp - CodeView type info dumper --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CompactTypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

static const EnumEntry<TypeLeafKind> LeafTypeNames[] = {
#define CV_TYPE(enum, val) {#enum, enum},
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
};

static StringRef getLeafName(TypeLeafKind K) {
  for (const auto &E : LeafTypeNames) {
    if (E.Value == K)
      return E.Name;
  }
  return StringRef();
}

CompactTypeDumpVisitor::CompactTypeDumpVisitor(TypeDatabase &TypeDB,
                                               ScopedPrinter *W)
    : CompactTypeDumpVisitor(TypeDB, TypeIndex(TypeIndex::FirstNonSimpleIndex),
                             W) {}

CompactTypeDumpVisitor::CompactTypeDumpVisitor(TypeDatabase &TypeDB,
                                               TypeIndex FirstTI,
                                               ScopedPrinter *W)
    : W(W), TI(FirstTI), Offset(0), TypeDB(TypeDB) {}

Error CompactTypeDumpVisitor::visitTypeBegin(CVType &Record) {
  return Error::success();
}

Error CompactTypeDumpVisitor::visitTypeEnd(CVType &Record) {
  uint32_t I = TI.getIndex();
  StringRef Leaf = getLeafName(Record.Type);
  StringRef Name = TypeDB.getTypeName(TI);
  W->printString(
      llvm::formatv("Index: {0:x} ({1:N} bytes, offset {2:N}) {3} \"{4}\"", I,
                    Record.length(), Offset, Leaf, Name)
          .str());

  Offset += Record.length();
  TI.setIndex(TI.getIndex() + 1);

  return Error::success();
}
