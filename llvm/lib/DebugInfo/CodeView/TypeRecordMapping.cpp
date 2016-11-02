//===- TypeRecordMapping.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"

using namespace llvm;
using namespace llvm::codeview;

#define error(X)                                                               \
  if (auto EC = X)                                                             \
    return EC;

namespace {
struct MapStringZ {
  Error operator()(CodeViewRecordIO &IO, StringRef &S) const {
    return IO.mapStringZ(S);
  }
};

struct MapInteger {
  template <typename T> Error operator()(CodeViewRecordIO &IO, T &N) const {
    return IO.mapInteger(N);
  }
};

struct MapEnum {
  template <typename T> Error operator()(CodeViewRecordIO &IO, T &N) const {
    return IO.mapEnum(N);
  }
};

struct MapOneMethodRecord {
  explicit MapOneMethodRecord(bool IsFromOverloadList)
      : IsFromOverloadList(IsFromOverloadList) {}

  Error operator()(CodeViewRecordIO &IO, OneMethodRecord &Method) const {
    error(IO.mapInteger(Method.Attrs.Attrs));
    if (IsFromOverloadList) {
      uint16_t Padding = 0;
      error(IO.mapInteger(Padding));
    }
    error(IO.mapInteger(Method.Type));
    if (Method.isIntroducingVirtual()) {
      error(IO.mapInteger(Method.VFTableOffset));
    } else if (!IO.isWriting())
      Method.VFTableOffset = -1;

    if (!IsFromOverloadList)
      error(IO.mapStringZ(Method.Name));

    return Error::success();
  }

private:
  bool IsFromOverloadList;
};
}

static Error mapNameAndUniqueName(CodeViewRecordIO &IO, StringRef &Name,
                                  StringRef &UniqueName, bool HasUniqueName) {
  error(IO.mapStringZ(Name));
  if (HasUniqueName)
    error(IO.mapStringZ(UniqueName));

  return Error::success();
}

Error TypeRecordMapping::visitTypeBegin(CVType &CVR) {
  error(IO.beginRecord(CVR.Type));
  TypeKind = CVR.Type;
  return Error::success();
}

Error TypeRecordMapping::visitTypeEnd(CVType &Record) {
  error(IO.endRecord());
  TypeKind.reset();
  return Error::success();
}

Error TypeRecordMapping::visitMemberBegin(CVMemberRecord &Record) {
  return Error::success();
}

Error TypeRecordMapping::visitMemberEnd(CVMemberRecord &Record) {
  if (!IO.isWriting()) {
    if (auto EC = IO.skipPadding())
      return EC;
  }

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, ModifierRecord &Record) {
  error(IO.mapInteger(Record.ModifiedType));
  error(IO.mapEnum(Record.Modifiers));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          ProcedureRecord &Record) {
  error(IO.mapInteger(Record.ReturnType));
  error(IO.mapEnum(Record.CallConv));
  error(IO.mapEnum(Record.Options));
  error(IO.mapInteger(Record.ParameterCount));
  error(IO.mapInteger(Record.ArgumentList));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          MemberFunctionRecord &Record) {
  error(IO.mapInteger(Record.ReturnType));
  error(IO.mapInteger(Record.ClassType));
  error(IO.mapInteger(Record.ThisType));
  error(IO.mapEnum(Record.CallConv));
  error(IO.mapEnum(Record.Options));
  error(IO.mapInteger(Record.ParameterCount));
  error(IO.mapInteger(Record.ArgumentList));
  error(IO.mapInteger(Record.ThisPointerAdjustment));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, ArgListRecord &Record) {
  error(IO.mapVectorN<uint32_t>(Record.StringIndices, MapInteger()));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, PointerRecord &Record) {
  error(IO.mapInteger(Record.ReferentType));
  error(IO.mapInteger(Record.Attrs));

  if (Record.isPointerToMember()) {
    if (!IO.isWriting())
      Record.MemberInfo.emplace();

    MemberPointerInfo &M = *Record.MemberInfo;
    error(IO.mapInteger(M.ContainingType));
    error(IO.mapEnum(M.Representation));
  }

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, ArrayRecord &Record) {
  error(IO.mapInteger(Record.ElementType));
  error(IO.mapInteger(Record.IndexType));
  error(IO.mapEncodedInteger(Record.Size));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, ClassRecord &Record) {
  assert((CVR.Type == TypeLeafKind::LF_STRUCTURE) ||
         (CVR.Type == TypeLeafKind::LF_CLASS) ||
         (CVR.Type == TypeLeafKind::LF_INTERFACE));

  error(IO.mapInteger(Record.MemberCount));
  error(IO.mapEnum(Record.Options));
  error(IO.mapInteger(Record.FieldList));
  error(IO.mapInteger(Record.DerivationList));
  error(IO.mapInteger(Record.VTableShape));
  error(IO.mapEncodedInteger(Record.Size));
  error(mapNameAndUniqueName(IO, Record.Name, Record.UniqueName,
                             Record.hasUniqueName()));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, UnionRecord &Record) {
  error(IO.mapInteger(Record.MemberCount));
  error(IO.mapEnum(Record.Options));
  error(IO.mapInteger(Record.FieldList));
  error(IO.mapEncodedInteger(Record.Size));
  error(mapNameAndUniqueName(IO, Record.Name, Record.UniqueName,
                             Record.hasUniqueName()));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, EnumRecord &Record) {
  error(IO.mapInteger(Record.MemberCount));
  error(IO.mapEnum(Record.Options));
  error(IO.mapInteger(Record.UnderlyingType));
  error(IO.mapInteger(Record.FieldList));
  error(mapNameAndUniqueName(IO, Record.Name, Record.UniqueName,
                             Record.hasUniqueName()));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, BitFieldRecord &Record) {
  error(IO.mapInteger(Record.Type));
  error(IO.mapInteger(Record.BitSize));
  error(IO.mapInteger(Record.BitOffset));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          VFTableShapeRecord &Record) {
  uint16_t Size;
  if (IO.isWriting()) {
    ArrayRef<VFTableSlotKind> Slots = Record.getSlots();
    Size = Slots.size();
    error(IO.mapInteger(Size));

    for (size_t SlotIndex = 0; SlotIndex < Slots.size(); SlotIndex += 2) {
      uint8_t Byte = static_cast<uint8_t>(Slots[SlotIndex]) << 4;
      if ((SlotIndex + 1) < Slots.size()) {
        Byte |= static_cast<uint8_t>(Slots[SlotIndex + 1]);
      }
      error(IO.mapInteger(Byte));
    }
  } else {
    error(IO.mapInteger(Size));
    for (uint16_t I = 0; I < Size; I += 2) {
      uint8_t Byte;
      error(IO.mapInteger(Byte));
      Record.Slots.push_back(static_cast<VFTableSlotKind>(Byte & 0xF));
      if ((I + 1) < Size)
        Record.Slots.push_back(static_cast<VFTableSlotKind>(Byte >> 4));
    }
  }

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, VFTableRecord &Record) {
  error(IO.mapInteger(Record.CompleteClass));
  error(IO.mapInteger(Record.OverriddenVFTable));
  error(IO.mapInteger(Record.VFPtrOffset));
  uint32_t NamesLen = 0;
  if (IO.isWriting()) {
    for (auto Name : Record.MethodNames)
      NamesLen += Name.size() + 1;
  }
  error(IO.mapInteger(NamesLen));
  error(IO.mapVectorTail(Record.MethodNames, MapStringZ()));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, StringIdRecord &Record) {
  error(IO.mapInteger(Record.Id));
  error(IO.mapStringZ(Record.String));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          UdtSourceLineRecord &Record) {
  error(IO.mapInteger(Record.UDT));
  error(IO.mapInteger(Record.SourceFile));
  error(IO.mapInteger(Record.LineNumber));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          UdtModSourceLineRecord &Record) {
  error(IO.mapInteger(Record.UDT));
  error(IO.mapInteger(Record.SourceFile));
  error(IO.mapInteger(Record.LineNumber));
  error(IO.mapInteger(Record.Module));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR, FuncIdRecord &Record) {
  error(IO.mapInteger(Record.ParentScope));
  error(IO.mapInteger(Record.FunctionType));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          MemberFuncIdRecord &Record) {
  error(IO.mapInteger(Record.ClassType));
  error(IO.mapInteger(Record.FunctionType));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          BuildInfoRecord &Record) {
  error(IO.mapVectorN<uint16_t>(Record.ArgIndices, MapInteger()));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          MethodOverloadListRecord &Record) {
  // TODO: Split the list into multiple records if it's longer than 64KB, using
  // a subrecord of TypeRecordKind::Index to chain the records together.
  error(IO.mapVectorTail(Record.Methods, MapOneMethodRecord(true)));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          FieldListRecord &Record) {
  error(IO.mapByteVectorTail(Record.Data));

  return Error::success();
}

Error TypeRecordMapping::visitKnownRecord(CVType &CVR,
                                          TypeServer2Record &Record) {
  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          BaseClassRecord &Record) {
  error(IO.mapInteger(Record.Attrs.Attrs));
  error(IO.mapInteger(Record.Type));
  error(IO.mapEncodedInteger(Record.Offset));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          EnumeratorRecord &Record) {
  error(IO.mapInteger(Record.Attrs.Attrs));

  // FIXME: Handle full APInt such as __int128.
  error(IO.mapEncodedInteger(Record.Value));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          DataMemberRecord &Record) {
  error(IO.mapInteger(Record.Attrs.Attrs));
  error(IO.mapInteger(Record.Type));
  error(IO.mapEncodedInteger(Record.FieldOffset));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          OverloadedMethodRecord &Record) {
  error(IO.mapInteger(Record.NumOverloads));
  error(IO.mapInteger(Record.MethodList));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          OneMethodRecord &Record) {
  MapOneMethodRecord Mapper(false);
  return Mapper(IO, Record);
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          NestedTypeRecord &Record) {
  uint16_t Padding = 0;
  error(IO.mapInteger(Padding));
  error(IO.mapInteger(Record.Type));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          StaticDataMemberRecord &Record) {

  error(IO.mapInteger(Record.Attrs.Attrs));
  error(IO.mapInteger(Record.Type));
  error(IO.mapStringZ(Record.Name));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          VirtualBaseClassRecord &Record) {

  error(IO.mapInteger(Record.Attrs.Attrs));
  error(IO.mapInteger(Record.BaseType));
  error(IO.mapInteger(Record.VBPtrType));
  error(IO.mapEncodedInteger(Record.VBPtrOffset));
  error(IO.mapEncodedInteger(Record.VTableIndex));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          VFPtrRecord &Record) {
  uint16_t Padding = 0;
  error(IO.mapInteger(Padding));
  error(IO.mapInteger(Record.Type));

  return Error::success();
}

Error TypeRecordMapping::visitKnownMember(CVMemberRecord &CVR,
                                          ListContinuationRecord &Record) {
  uint16_t Padding = 0;
  error(IO.mapInteger(Padding));
  error(IO.mapInteger(Record.ContinuationIndex));

  return Error::success();
}
