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
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/Error.h"
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
///
/// As an additional complication, type stream merging actually produces two
/// streams: an item (or IPI) stream and a type stream, as this is what is
/// actually stored in the final PDB. We choose which records go where by
/// looking at the record kind.
class TypeStreamMerger : public TypeVisitorCallbacks {
public:
  TypeStreamMerger(TypeTableBuilder &DestIdStream,
                   TypeTableBuilder &DestTypeStream, TypeServerHandler *Handler)
      : DestIdStream(DestIdStream), DestTypeStream(DestTypeStream),
        FieldListBuilder(DestTypeStream), Handler(Handler) {}

  static const TypeIndex Untranslated;

/// TypeVisitorCallbacks overrides.
#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(CVType &CVR, Name##Record &Record) override;
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record) override;
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

  Error visitUnknownType(CVType &Record) override;

  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeEnd(CVType &Record) override;
  Error visitMemberEnd(CVMemberRecord &Record) override;

  Error mergeStream(const CVTypeArray &Types);

private:
  void addMapping(TypeIndex Idx);

  bool remapIndex(TypeIndex &Idx);

  size_t slotForIndex(TypeIndex Idx) const {
    assert(!Idx.isSimple() && "simple type indices have no slots");
    return Idx.getIndex() - TypeIndex::FirstNonSimpleIndex;
  }

  Error errorCorruptRecord() const {
    return llvm::make_error<CodeViewError>(cv_error_code::corrupt_record);
  }

  template <typename RecordType>
  Error writeRecord(RecordType &R, bool RemapSuccess) {
    TypeIndex DestIdx = Untranslated;
    if (RemapSuccess)
      DestIdx = DestTypeStream.writeKnownType(R);
    addMapping(DestIdx);
    return Error::success();
  }

  template <typename RecordType>
  Error writeIdRecord(RecordType &R, bool RemapSuccess) {
    TypeIndex DestIdx = Untranslated;
    if (RemapSuccess)
      DestIdx = DestIdStream.writeKnownType(R);
    addMapping(DestIdx);
    return Error::success();
  }

  template <typename RecordType>
  Error writeMember(RecordType &R, bool RemapSuccess) {
    if (RemapSuccess)
      FieldListBuilder.writeMemberType(R);
    else
      HadUntranslatedMember = true;
    return Error::success();
  }

  Optional<Error> LastError;

  bool IsSecondPass = false;

  bool HadUntranslatedMember = false;

  unsigned NumBadIndices = 0;

  BumpPtrAllocator Allocator;

  TypeTableBuilder &DestIdStream;
  TypeTableBuilder &DestTypeStream;
  FieldListRecordBuilder FieldListBuilder;
  TypeServerHandler *Handler;

  TypeIndex CurIndex{TypeIndex::FirstNonSimpleIndex};

  /// Map from source type index to destination type index. Indexed by source
  /// type index minus 0x1000.
  SmallVector<TypeIndex, 0> IndexMap;
};

} // end anonymous namespace

const TypeIndex TypeStreamMerger::Untranslated(SimpleTypeKind::NotTranslated);

Error TypeStreamMerger::visitTypeBegin(CVRecord<TypeLeafKind> &Rec) {
  return Error::success();
}

Error TypeStreamMerger::visitTypeEnd(CVRecord<TypeLeafKind> &Rec) {
  CurIndex = TypeIndex(CurIndex.getIndex() + 1);
  if (!IsSecondPass)
    assert(IndexMap.size() == slotForIndex(CurIndex) &&
           "visitKnownRecord should add one index map entry");
  return Error::success();
}

Error TypeStreamMerger::visitMemberEnd(CVMemberRecord &Rec) {
  return Error::success();
}

void TypeStreamMerger::addMapping(TypeIndex Idx) {
  if (!IsSecondPass) {
    assert(IndexMap.size() == slotForIndex(CurIndex) &&
           "visitKnownRecord should add one index map entry");
    IndexMap.push_back(Idx);
  } else {
    assert(slotForIndex(CurIndex) < IndexMap.size());
    IndexMap[slotForIndex(CurIndex)] = Idx;
  }
}

bool TypeStreamMerger::remapIndex(TypeIndex &Idx) {
  // Simple types are unchanged.
  if (Idx.isSimple())
    return true;

  // Check if this type index refers to a record we've already translated
  // successfully. If it refers to a type later in the stream or a record we
  // had to defer, defer it until later pass.
  unsigned MapPos = slotForIndex(Idx);
  if (MapPos < IndexMap.size() && IndexMap[MapPos] != Untranslated) {
    Idx = IndexMap[MapPos];
    return true;
  }

  // If this is the second pass and this index isn't in the map, then it points
  // outside the current type stream, and this is a corrupt record.
  if (IsSecondPass && MapPos >= IndexMap.size()) {
    // FIXME: Print a more useful error. We can give the current record and the
    // index that we think its pointing to.
    LastError = joinErrors(std::move(*LastError), errorCorruptRecord());
  }

  ++NumBadIndices;

  // This type index is invalid. Remap this to "not translated by cvpack",
  // and return failure.
  Idx = Untranslated;
  return false;
}

//----------------------------------------------------------------------------//
// Item records
//----------------------------------------------------------------------------//

Error TypeStreamMerger::visitKnownRecord(CVType &, FuncIdRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.ParentScope);
  Success &= remapIndex(R.FunctionType);
  return writeIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, MemberFuncIdRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.ClassType);
  Success &= remapIndex(R.FunctionType);
  return writeIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, StringIdRecord &R) {
  return writeIdRecord(R, remapIndex(R.Id));
}

Error TypeStreamMerger::visitKnownRecord(CVType &, StringListRecord &R) {
  bool Success = true;
  for (TypeIndex &Str : R.StringIndices)
    Success &= remapIndex(Str);
  return writeIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, BuildInfoRecord &R) {
  bool Success = true;
  for (TypeIndex &Arg : R.ArgIndices)
    Success &= remapIndex(Arg);
  return writeIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, UdtSourceLineRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.UDT);
  Success &= remapIndex(R.SourceFile);
  // FIXME: Translate UdtSourceLineRecord into UdtModSourceLineRecords in the
  // IPI stream.
  return writeIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, UdtModSourceLineRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.UDT);
  Success &= remapIndex(R.SourceFile);
  return writeIdRecord(R, Success);
}

//----------------------------------------------------------------------------//
// Type records
//----------------------------------------------------------------------------//

Error TypeStreamMerger::visitKnownRecord(CVType &, ModifierRecord &R) {
  return writeRecord(R, remapIndex(R.ModifiedType));
}

Error TypeStreamMerger::visitKnownRecord(CVType &, ProcedureRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.ReturnType);
  Success &= remapIndex(R.ArgumentList);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, MemberFunctionRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.ReturnType);
  Success &= remapIndex(R.ClassType);
  Success &= remapIndex(R.ThisType);
  Success &= remapIndex(R.ArgumentList);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &Type, ArgListRecord &R) {
  bool Success = true;
  for (TypeIndex &Arg : R.ArgIndices)
    Success &= remapIndex(Arg);
  if (auto EC = writeRecord(R, Success))
    return EC;
  return Error::success();
}

Error TypeStreamMerger::visitKnownRecord(CVType &, PointerRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.ReferentType);
  if (R.isPointerToMember())
    Success &= remapIndex(R.MemberInfo->ContainingType);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, ArrayRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.ElementType);
  Success &= remapIndex(R.IndexType);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, ClassRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.FieldList);
  Success &= remapIndex(R.DerivationList);
  Success &= remapIndex(R.VTableShape);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, UnionRecord &R) {
  return writeRecord(R, remapIndex(R.FieldList));
}

Error TypeStreamMerger::visitKnownRecord(CVType &, EnumRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.FieldList);
  Success &= remapIndex(R.UnderlyingType);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, BitFieldRecord &R) {
  return writeRecord(R, remapIndex(R.Type));
}

Error TypeStreamMerger::visitKnownRecord(CVType &, VFTableShapeRecord &R) {
  return writeRecord(R, true);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, TypeServer2Record &R) {
  return writeRecord(R, true);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, LabelRecord &R) {
  return writeRecord(R, true);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, VFTableRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.CompleteClass);
  Success &= remapIndex(R.OverriddenVFTable);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &,
                                         MethodOverloadListRecord &R) {
  bool Success = true;
  for (OneMethodRecord &Meth : R.Methods)
    Success &= remapIndex(Meth.Type);
  return writeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &, FieldListRecord &R) {
  // Visit the members inside the field list.
  HadUntranslatedMember = false;
  FieldListBuilder.begin();
  if (auto EC = codeview::visitMemberRecordStream(R.Data, *this))
    return EC;

  // Write the record if we translated all field list members.
  TypeIndex DestIdx = Untranslated;
  if (!HadUntranslatedMember)
    DestIdx = FieldListBuilder.end();
  else
    FieldListBuilder.reset();
  addMapping(DestIdx);

  return Error::success();
}

//----------------------------------------------------------------------------//
// Member records
//----------------------------------------------------------------------------//

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         NestedTypeRecord &R) {
  return writeMember(R, remapIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &, OneMethodRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.Type);
  return writeMember(R, Success);
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         OverloadedMethodRecord &R) {
  return writeMember(R, remapIndex(R.MethodList));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         DataMemberRecord &R) {
  return writeMember(R, remapIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         StaticDataMemberRecord &R) {
  return writeMember(R, remapIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         EnumeratorRecord &R) {
  return writeMember(R, true);
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &, VFPtrRecord &R) {
  return writeMember(R, remapIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &, BaseClassRecord &R) {
  return writeMember(R, remapIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         VirtualBaseClassRecord &R) {
  bool Success = true;
  Success &= remapIndex(R.BaseType);
  Success &= remapIndex(R.VBPtrType);
  return writeMember(R, Success);
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         ListContinuationRecord &R) {
  return writeMember(R, remapIndex(R.ContinuationIndex));
}

Error TypeStreamMerger::visitUnknownType(CVType &Rec) {
  // We failed to translate a type. Translate this index as "not translated".
  addMapping(TypeIndex(SimpleTypeKind::NotTranslated));
  return errorCorruptRecord();
}

Error TypeStreamMerger::mergeStream(const CVTypeArray &Types) {
  assert(IndexMap.empty());
  LastError = Error::success();

  if (auto EC = codeview::visitTypeStream(Types, *this, Handler))
    return EC;

  // If we found bad indices but no other errors, try doing another pass and see
  // if we can resolve the indices that weren't in the map on the first pass.
  // This may require multiple passes, but we should always make progress. MASM
  // is the only known CodeView producer that makes type streams that aren't
  // topologically sorted. The standard library contains MASM-produced objects,
  // so this is important to handle correctly, but we don't have to be too
  // efficient. MASM type streams are usually very small.
  while (!*LastError && NumBadIndices > 0) {
    unsigned BadIndicesRemaining = NumBadIndices;
    IsSecondPass = true;
    NumBadIndices = 0;
    CurIndex = TypeIndex(TypeIndex::FirstNonSimpleIndex);

    if (auto EC = codeview::visitTypeStream(Types, *this, Handler))
      return EC;

    assert(NumBadIndices <= BadIndicesRemaining &&
           "second pass found more bad indices");
    if (!*LastError && NumBadIndices == BadIndicesRemaining) {
      return llvm::make_error<CodeViewError>(
          cv_error_code::corrupt_record, "input type graph contains cycles");
    }
  }

  IndexMap.clear();

  Error Ret = std::move(*LastError);
  LastError.reset();
  return Ret;
}

Error llvm::codeview::mergeTypeStreams(TypeTableBuilder &DestIdStream,
                                       TypeTableBuilder &DestTypeStream,
                                       TypeServerHandler *Handler,
                                       const CVTypeArray &Types) {
  return TypeStreamMerger(DestIdStream, DestTypeStream, Handler)
      .mergeStream(Types);
}
