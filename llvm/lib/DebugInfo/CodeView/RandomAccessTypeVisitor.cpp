//===- RandomAccessTypeVisitor.cpp ---------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/RandomAccessTypeVisitor.h"

#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

using namespace llvm;
using namespace llvm::codeview;

RandomAccessTypeVisitor::RandomAccessTypeVisitor(
    const CVTypeArray &Types, uint32_t NumRecords,
    PartialOffsetArray PartialOffsets)
    : Database(NumRecords), Types(Types), DatabaseVisitor(Database),
      InternalVisitor(Pipeline), PartialOffsets(PartialOffsets) {
  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(DatabaseVisitor);

  KnownOffsets.resize(Database.capacity());
}

Error RandomAccessTypeVisitor::visitTypeIndex(TypeIndex TI,
                                              TypeVisitorCallbacks &Callbacks) {
  assert(TI.toArrayIndex() < Database.capacity());

  if (!Database.contains(TI)) {
    if (auto EC = visitRangeForType(TI))
      return EC;
  }

  assert(Database.contains(TI));
  auto &Record = Database.getTypeRecord(TI);
  CVTypeVisitor V(Callbacks);
  return V.visitTypeRecord(Record, TI);
}

Error RandomAccessTypeVisitor::visitRangeForType(TypeIndex TI) {
  if (PartialOffsets.empty()) {
    TypeIndex TIB(TypeIndex::FirstNonSimpleIndex);
    TypeIndex TIE = TIB + Database.capacity();
    return visitRange(TIB, 0, TIE);
  }

  auto Next = std::upper_bound(PartialOffsets.begin(), PartialOffsets.end(), TI,
                               [](TypeIndex Value, const TypeIndexOffset &IO) {
                                 return Value < IO.Type;
                               });

  assert(Next != PartialOffsets.begin());
  auto Prev = std::prev(Next);

  TypeIndex TIB = Prev->Type;
  TypeIndex TIE;
  if (Next == PartialOffsets.end()) {
    TIE = TypeIndex::fromArrayIndex(Database.capacity());
  } else {
    TIE = Next->Type;
  }

  if (auto EC = visitRange(TIB, Prev->Offset, TIE))
    return EC;
  return Error::success();
}

Error RandomAccessTypeVisitor::visitRange(TypeIndex Begin, uint32_t BeginOffset,
                                          TypeIndex End) {

  auto RI = Types.at(BeginOffset);
  assert(RI != Types.end());

  while (Begin != End) {
    assert(!Database.contains(Begin));
    if (auto EC = InternalVisitor.visitTypeRecord(*RI, Begin))
      return EC;
    KnownOffsets[Begin.toArrayIndex()] = BeginOffset;

    BeginOffset += RI.getRecordLength();
    ++Begin;
    ++RI;
  }

  return Error::success();
}
