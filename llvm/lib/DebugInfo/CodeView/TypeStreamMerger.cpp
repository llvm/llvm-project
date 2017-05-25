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
  explicit TypeStreamMerger(SmallVectorImpl<TypeIndex> &SourceToDest,
                            TypeServerHandler *Handler)
      : Handler(Handler), IndexMap(SourceToDest) {
    SourceToDest.clear();
  }

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

  Error mergeTypesAndIds(TypeTableBuilder &DestIds, TypeTableBuilder &DestTypes,
    const CVTypeArray &IdsAndTypes);
  Error mergeIdRecords(TypeTableBuilder &Dest,
                       ArrayRef<TypeIndex> TypeSourceToDest,
    const CVTypeArray &Ids);
  Error mergeTypeRecords(TypeTableBuilder &Dest, const CVTypeArray &Types);

private:
  Error doit(const CVTypeArray &Types);

  void addMapping(TypeIndex Idx);

  bool remapTypeIndex(TypeIndex &Idx);
  bool remapItemIndex(TypeIndex &Idx);

  bool remapIndices(RemappedType &Record, ArrayRef<uint32_t> TidOffs,
                    ArrayRef<uint32_t> IidOffs) {
    auto OriginalData = Record.OriginalRecord.content();
    bool Success = true;
    for (auto Off : TidOffs) {
      ArrayRef<uint8_t> Bytes = OriginalData.slice(Off, sizeof(TypeIndex));
      TypeIndex OldTI(
          *reinterpret_cast<const support::ulittle32_t *>(Bytes.data()));
      TypeIndex NewTI = OldTI;
      bool ThisSuccess = remapTypeIndex(NewTI);
      if (ThisSuccess && NewTI != OldTI)
        Record.Mappings.emplace_back(Off, NewTI);
      Success &= ThisSuccess;
    }
    for (auto Off : IidOffs) {
      ArrayRef<uint8_t> Bytes = OriginalData.slice(Off, sizeof(TypeIndex));
      TypeIndex OldTI(
          *reinterpret_cast<const support::ulittle32_t *>(Bytes.data()));
      TypeIndex NewTI = OldTI;
      bool ThisSuccess = remapItemIndex(NewTI);
      if (ThisSuccess && NewTI != OldTI)
        Record.Mappings.emplace_back(Off, NewTI);
      Success &= ThisSuccess;
    }
    return Success;
  }

  bool remapIndex(TypeIndex &Idx, ArrayRef<TypeIndex> Map);

  size_t slotForIndex(TypeIndex Idx) const {
    assert(!Idx.isSimple() && "simple type indices have no slots");
    return Idx.getIndex() - TypeIndex::FirstNonSimpleIndex;
  }

  Error errorCorruptRecord() const {
    return llvm::make_error<CodeViewError>(cv_error_code::corrupt_record);
  }

  template <typename RecordType>
  Error writeKnownRecord(TypeTableBuilder &Dest, RecordType &R,
                         bool RemapSuccess) {
    TypeIndex DestIdx = Untranslated;
    if (RemapSuccess)
      DestIdx = Dest.writeKnownType(R);
    addMapping(DestIdx);
    return Error::success();
  }

  template <typename RecordType>
  Error writeKnownTypeRecord(RecordType &R, bool RemapSuccess) {
    return writeKnownRecord(*DestTypeStream, R, RemapSuccess);
  }

  template <typename RecordType>
  Error writeKnownIdRecord(RecordType &R, bool RemapSuccess) {
    return writeKnownRecord(*DestIdStream, R, RemapSuccess);
  }

  Error writeRecord(TypeTableBuilder &Dest, const RemappedType &Record,
                    bool RemapSuccess) {
    TypeIndex DestIdx = Untranslated;
    if (RemapSuccess)
      DestIdx = Dest.writeSerializedRecord(Record);
    addMapping(DestIdx);
    return Error::success();
  }

  Error writeTypeRecord(const CVType &Record) {
    TypeIndex DestIdx =
        DestTypeStream->writeSerializedRecord(Record.RecordData);
    addMapping(DestIdx);
    return Error::success();
  }

  Error writeTypeRecord(const RemappedType &Record, bool RemapSuccess) {
    return writeRecord(*DestTypeStream, Record, RemapSuccess);
  }

  Error writeIdRecord(const RemappedType &Record, bool RemapSuccess) {
    return writeRecord(*DestIdStream, Record, RemapSuccess);
  }

  template <typename RecordType>
  Error writeMember(RecordType &R, bool RemapSuccess) {
    if (RemapSuccess)
      FieldListBuilder->writeMemberType(R);
    else
      HadUntranslatedMember = true;
    return Error::success();
  }

  Optional<Error> LastError;

  bool IsSecondPass = false;

  bool HadUntranslatedMember = false;

  unsigned NumBadIndices = 0;

  BumpPtrAllocator Allocator;

  TypeIndex CurIndex{TypeIndex::FirstNonSimpleIndex};

  TypeTableBuilder *DestIdStream = nullptr;
  TypeTableBuilder *DestTypeStream = nullptr;
  std::unique_ptr<FieldListRecordBuilder> FieldListBuilder;
  TypeServerHandler *Handler = nullptr;

  // If we're only mapping id records, this array contains the mapping for
  // type records.
  ArrayRef<TypeIndex> TypeLookup;

  /// Map from source type index to destination type index. Indexed by source
  /// type index minus 0x1000.
  SmallVectorImpl<TypeIndex> &IndexMap;
};

} // end anonymous namespace

const TypeIndex TypeStreamMerger::Untranslated(SimpleTypeKind::NotTranslated);

Error TypeStreamMerger::visitTypeBegin(CVType &Rec) { return Error::success(); }

Error TypeStreamMerger::visitTypeEnd(CVType &Rec) {
  ++CurIndex;
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

bool TypeStreamMerger::remapIndex(TypeIndex &Idx, ArrayRef<TypeIndex> Map) {
  // Simple types are unchanged.
  if (Idx.isSimple())
    return true;

  // Check if this type index refers to a record we've already translated
  // successfully. If it refers to a type later in the stream or a record we
  // had to defer, defer it until later pass.
  unsigned MapPos = slotForIndex(Idx);
  if (MapPos < Map.size() && Map[MapPos] != Untranslated) {
    Idx = Map[MapPos];
    return true;
  }

  // If this is the second pass and this index isn't in the map, then it points
  // outside the current type stream, and this is a corrupt record.
  if (IsSecondPass && MapPos >= Map.size()) {
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

bool TypeStreamMerger::remapTypeIndex(TypeIndex &Idx) {
  // If we're mapping a pure index stream, then IndexMap only contains mappings
  // from OldIdStream -> NewIdStream, in which case we will need to use the
  // special mapping from OldTypeStream -> NewTypeStream which was computed
  // externally.  Regardless, we use this special map if and only if we are
  // doing an id-only mapping.
  if (DestTypeStream == nullptr)
    return remapIndex(Idx, TypeLookup);

  assert(TypeLookup.empty());
  return remapIndex(Idx, IndexMap);
}

bool TypeStreamMerger::remapItemIndex(TypeIndex &Idx) {
  assert(DestIdStream);
  return remapIndex(Idx, IndexMap);
}

//----------------------------------------------------------------------------//
// Item records
//----------------------------------------------------------------------------//

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, FuncIdRecord &R) {
  assert(DestIdStream);

  RemappedType RR(CVR);
  return writeIdRecord(RR, remapIndices(RR, {4}, {0}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, MemberFuncIdRecord &R) {
  assert(DestIdStream);

  RemappedType RR(CVR);
  return writeIdRecord(RR, remapIndices(RR, {0, 4}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, StringIdRecord &R) {
  assert(DestIdStream);

  RemappedType RR(CVR);
  return writeIdRecord(RR, remapIndices(RR, {}, {0}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, StringListRecord &R) {
  assert(DestIdStream);

  if (auto EC = TypeDeserializer::deserializeAs<StringListRecord>(CVR, R))
    return EC;
  bool Success = true;

  for (TypeIndex &Id : R.StringIndices)
    Success &= remapItemIndex(Id);
  return writeKnownIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, BuildInfoRecord &R) {
  assert(DestIdStream);

  if (auto EC = TypeDeserializer::deserializeAs(CVR, R))
    return EC;

  bool Success = true;
  for (TypeIndex &Str : R.ArgIndices)
    Success &= remapItemIndex(Str);
  return writeKnownIdRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, UdtSourceLineRecord &R) {
  assert(DestIdStream);

  RemappedType RR(CVR);

  // FIXME: Translate UdtSourceLineRecord into UdtModSourceLineRecords in the
  // IPI stream.
  return writeIdRecord(RR, remapIndices(RR, {0}, {4}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR,
                                         UdtModSourceLineRecord &R) {
  assert(DestIdStream);

  RemappedType RR(CVR);

  // UdtModSourceLine Source File Ids are offsets into the global string table,
  // not type indices.
  // FIXME: We need to merge string table records for this to be valid.
  return writeIdRecord(RR, remapIndices(RR, {0}, {}));
}

//----------------------------------------------------------------------------//
// Type records
//----------------------------------------------------------------------------//

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, ModifierRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {0}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, ProcedureRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {0, 8}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, MemberFunctionRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {0, 4, 8, 16}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, ArgListRecord &R) {
  assert(DestTypeStream);

  if (auto EC = TypeDeserializer::deserializeAs(CVR, R))
    return EC;

  bool Success = true;
  for (TypeIndex &Arg : R.ArgIndices)
    Success &= remapTypeIndex(Arg);

  return writeKnownTypeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, PointerRecord &R) {
  assert(DestTypeStream);

  // Pointer records have a different number of TypeIndex mappings depending
  // on whether or not it is a pointer to member.
  if (auto EC = TypeDeserializer::deserializeAs(CVR, R))
    return EC;

  bool Success = remapTypeIndex(R.ReferentType);
  if (R.isPointerToMember())
    Success &= remapTypeIndex(R.MemberInfo->ContainingType);
  return writeKnownTypeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, ArrayRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {0, 4}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, ClassRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {4, 8, 12}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, UnionRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {4}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, EnumRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {4, 8}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, BitFieldRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {0}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, VFTableShapeRecord &R) {
  assert(DestTypeStream);

  return writeTypeRecord(CVR);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, TypeServer2Record &R) {
  assert(DestTypeStream);

  return writeTypeRecord(CVR);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, LabelRecord &R) {
  assert(DestTypeStream);

  return writeTypeRecord(CVR);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, VFTableRecord &R) {
  assert(DestTypeStream);

  RemappedType RR(CVR);
  return writeTypeRecord(RR, remapIndices(RR, {0, 4}, {}));
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR,
                                         MethodOverloadListRecord &R) {
  assert(DestTypeStream);

  if (auto EC = TypeDeserializer::deserializeAs(CVR, R))
    return EC;

  bool Success = true;
  for (OneMethodRecord &Meth : R.Methods)
    Success &= remapTypeIndex(Meth.Type);
  return writeKnownTypeRecord(R, Success);
}

Error TypeStreamMerger::visitKnownRecord(CVType &CVR, FieldListRecord &R) {
  assert(DestTypeStream);
  // Visit the members inside the field list.
  HadUntranslatedMember = false;
  if (!FieldListBuilder)
    FieldListBuilder =
        llvm::make_unique<FieldListRecordBuilder>(*DestTypeStream);

  FieldListBuilder->begin();
  if (auto EC = codeview::visitMemberRecordStream(CVR.content(), *this))
    return EC;

  // Write the record if we translated all field list members.
  TypeIndex DestIdx = FieldListBuilder->end(!HadUntranslatedMember);
  addMapping(HadUntranslatedMember ? Untranslated : DestIdx);

  return Error::success();
}

//----------------------------------------------------------------------------//
// Member records
//----------------------------------------------------------------------------//

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         NestedTypeRecord &R) {
  return writeMember(R, remapTypeIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &, OneMethodRecord &R) {
  bool Success = true;
  Success &= remapTypeIndex(R.Type);
  return writeMember(R, Success);
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         OverloadedMethodRecord &R) {
  return writeMember(R, remapTypeIndex(R.MethodList));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         DataMemberRecord &R) {
  return writeMember(R, remapTypeIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         StaticDataMemberRecord &R) {
  return writeMember(R, remapTypeIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         EnumeratorRecord &R) {
  return writeMember(R, true);
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &, VFPtrRecord &R) {
  return writeMember(R, remapTypeIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &, BaseClassRecord &R) {
  return writeMember(R, remapTypeIndex(R.Type));
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         VirtualBaseClassRecord &R) {
  bool Success = true;
  Success &= remapTypeIndex(R.BaseType);
  Success &= remapTypeIndex(R.VBPtrType);
  return writeMember(R, Success);
}

Error TypeStreamMerger::visitKnownMember(CVMemberRecord &,
                                         ListContinuationRecord &R) {
  return writeMember(R, remapTypeIndex(R.ContinuationIndex));
}

Error TypeStreamMerger::visitUnknownType(CVType &Rec) {
  // We failed to translate a type. Translate this index as "not translated".
  addMapping(TypeIndex(SimpleTypeKind::NotTranslated));
  return errorCorruptRecord();
}

Error TypeStreamMerger::mergeTypeRecords(TypeTableBuilder &Dest,
  const CVTypeArray &Types) {
  DestTypeStream = &Dest;

  return doit(Types);
}

Error TypeStreamMerger::mergeIdRecords(TypeTableBuilder &Dest,
                                       ArrayRef<TypeIndex> TypeSourceToDest,
  const CVTypeArray &Ids) {
  DestIdStream = &Dest;
  TypeLookup = TypeSourceToDest;

  return doit(Ids);
}

Error TypeStreamMerger::mergeTypesAndIds(TypeTableBuilder &DestIds,
                                         TypeTableBuilder &DestTypes,
  const CVTypeArray &IdsAndTypes) {
  DestIdStream = &DestIds;
  DestTypeStream = &DestTypes;

  return doit(IdsAndTypes);
}

Error TypeStreamMerger::doit(const CVTypeArray &Types) {
  LastError = Error::success();

  // We don't want to deserialize records.  I guess this flag is poorly named,
  // but it really means "Don't deserialize records before switching on the
  // concrete type.
  if (auto EC =
          codeview::visitTypeStream(Types, *this, VDS_BytesExternal, Handler))
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

    if (auto EC =
            codeview::visitTypeStream(Types, *this, VDS_BytesExternal, Handler))
      return EC;

    assert(NumBadIndices <= BadIndicesRemaining &&
           "second pass found more bad indices");
    if (!*LastError && NumBadIndices == BadIndicesRemaining) {
      return llvm::make_error<CodeViewError>(
          cv_error_code::corrupt_record, "input type graph contains cycles");
    }
  }

  Error Ret = std::move(*LastError);
  LastError.reset();
  return Ret;
}

Error llvm::codeview::mergeTypeRecords(TypeTableBuilder &Dest,
                                       SmallVectorImpl<TypeIndex> &SourceToDest,
                                       TypeServerHandler *Handler,
  const CVTypeArray &Types) {
  TypeStreamMerger M(SourceToDest, Handler);
  return M.mergeTypeRecords(Dest, Types);
}

Error llvm::codeview::mergeIdRecords(TypeTableBuilder &Dest,
                                     ArrayRef<TypeIndex> TypeSourceToDest,
                                     SmallVectorImpl<TypeIndex> &SourceToDest,
  const CVTypeArray &Ids) {
  TypeStreamMerger M(SourceToDest, nullptr);
  return M.mergeIdRecords(Dest, TypeSourceToDest, Ids);
}

Error llvm::codeview::mergeTypeAndIdRecords(
    TypeTableBuilder &DestIds, TypeTableBuilder &DestTypes,
    SmallVectorImpl<TypeIndex> &SourceToDest, TypeServerHandler *Handler,
  const CVTypeArray &IdsAndTypes) {

  TypeStreamMerger M(SourceToDest, Handler);
  return M.mergeTypesAndIds(DestIds, DestTypes, IdsAndTypes);
}
