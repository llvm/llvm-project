//===-- TypeStreamMerger.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeStreamMerger.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeStream.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::codeview;

namespace {

/// Implementation of CodeView type stream merging.
///
/// A CodeView type stream is a series of records that reference each other
/// through type indices. A type index is either "simple", meaning it is less
/// than 0x1000 and refers to a builtin type, or it is complex, meaning it
/// refers to a prior type record in the current stream. The type index of a
/// record is equal to the number of records before it in the stream plus
/// 0x1000.
///
/// Type records are only allowed to use type indices smaller than their own, so
/// a type stream is effectively a topologically sorted DAG. Cycles occuring in
/// the type graph of the source program are resolved with forward declarations
/// of composite types. This class implements the following type stream merging
/// algorithm, which relies on this DAG structure:
///
/// - Begin with a new empty stream, and a new empty hash table that maps from
///   type record contents to new type index.
/// - For each new type stream, maintain a map from source type index to
///   destination type index.
/// - For each record, copy it and rewrite its type indices to be valid in the
///   destination type stream.
/// - If the new type record is not already present in the destination stream
///   hash table, append it to the destination type stream, assign it the next
///   type index, and update the two hash tables.
/// - If the type record already exists in the destination stream, discard it
///   and update the type index map to forward the source type index to the
///   existing destination type index.
class TypeStreamMerger : public CVTypeVisitor<TypeStreamMerger> {
public:
  TypeStreamMerger(TypeTableBuilder &DestStream) : DestStream(DestStream) {
    assert(!hadError());
  }

  /// CVTypeVisitor overrides.
#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  void visit##Name(TypeLeafKind LeafType, Name##Record &Record);
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  void visit##Name(TypeLeafKind LeafType, Name##Record &Record);
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

  void visitUnknownMember(TypeLeafKind Leaf);

  void visitTypeBegin(TypeLeafKind Leaf, ArrayRef<uint8_t> RecordData);
  void visitTypeEnd(TypeLeafKind Leaf, ArrayRef<uint8_t> RecordData);

  void visitFieldList(TypeLeafKind Leaf, ArrayRef<uint8_t> FieldData);

  bool mergeStream(ArrayRef<uint8_t> SrcStream);

private:
  bool hadError() { return FoundBadTypeIndex || CVTypeVisitor::hadError(); }

  bool FoundBadTypeIndex = false;

  FieldListRecordBuilder FieldBuilder;

  TypeTableBuilder &DestStream;

  size_t BeginIndexMapSize = 0;

  /// Map from source type index to destination type index. Indexed by source
  /// type index minus 0x1000.
  SmallVector<TypeIndex, 0> IndexMap;
};

} // end anonymous namespace

void TypeStreamMerger::visitTypeBegin(TypeLeafKind Leaf,
                                      ArrayRef<uint8_t> RecordData) {
  BeginIndexMapSize = IndexMap.size();
}

void TypeStreamMerger::visitTypeEnd(TypeLeafKind Leaf,
                                    ArrayRef<uint8_t> RecordData) {
  assert(IndexMap.size() == BeginIndexMapSize + 1);
}

void TypeStreamMerger::visitFieldList(TypeLeafKind Leaf,
                                      ArrayRef<uint8_t> FieldData) {
  CVTypeVisitor::visitFieldList(Leaf, FieldData);
  IndexMap.push_back(DestStream.writeFieldList(FieldBuilder));
  FieldBuilder.reset();
}

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  void TypeStreamMerger::visit##Name(TypeLeafKind LeafType,                    \
                                     Name##Record &Record) {                   \
    FoundBadTypeIndex |= !Record.remapTypeIndices(IndexMap);              \
    IndexMap.push_back(DestStream.write##Name(Record));                        \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  void TypeStreamMerger::visit##Name(TypeLeafKind LeafType,                    \
                                     Name##Record &Record) {                   \
    FoundBadTypeIndex |= !Record.remapTypeIndices(IndexMap);              \
    FieldBuilder.write##Name(Record);                                          \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

void TypeStreamMerger::visitUnknownMember(TypeLeafKind LF) {
  // We failed to translate a type. Translate this index as "not translated".
  IndexMap.push_back(
      TypeIndex(SimpleTypeKind::NotTranslated, SimpleTypeMode::Direct));
  parseError();
}

bool TypeStreamMerger::mergeStream(ArrayRef<uint8_t> SrcStream) {
  assert(IndexMap.empty());
  visitTypeStream(SrcStream);
  IndexMap.clear();
  return !hadError();
}

bool llvm::codeview::mergeTypeStreams(TypeTableBuilder &DestStream,
                                      ArrayRef<uint8_t> SrcStream) {
  return TypeStreamMerger(DestStream).mergeStream(SrcStream);
}
