//===-- TypeRecord.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

//===----------------------------------------------------------------------===//
// Type index remapping
//===----------------------------------------------------------------------===//

static bool remapIndex(ArrayRef<TypeIndex> IndexMap, TypeIndex &Idx) {
  // Simple types are unchanged.
  if (Idx.isSimple())
    return true;
  unsigned MapPos = Idx.getIndex() - TypeIndex::FirstNonSimpleIndex;
  if (MapPos < IndexMap.size()) {
    Idx = IndexMap[MapPos];
    return true;
  }

  // This type index is invalid. Remap this to "not translated by cvpack",
  // and return failure.
  Idx = TypeIndex(SimpleTypeKind::NotTranslated, SimpleTypeMode::Direct);
  return false;
}

bool ModifierRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, ModifiedType);
}

bool ProcedureRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, ReturnType);
  Success &= remapIndex(IndexMap, ArgumentList);
  return Success;
}

bool MemberFunctionRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, ReturnType);
  Success &= remapIndex(IndexMap, ClassType);
  Success &= remapIndex(IndexMap, ThisType);
  Success &= remapIndex(IndexMap, ArgumentList);
  return Success;
}

bool MemberFuncIdRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, ClassType);
  Success &= remapIndex(IndexMap, FunctionType);
  return Success;
}

bool ArgListRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  for (TypeIndex &Str : StringIndices)
    Success &= remapIndex(IndexMap, Str);
  return Success;
}

bool MemberPointerInfo::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, ContainingType);
}

bool PointerRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, ReferentType);
  if (isPointerToMember())
    Success &= MemberInfo->remapTypeIndices(IndexMap);
  return Success;
}

bool NestedTypeRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Type);
}

bool ArrayRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, ElementType);
  Success &= remapIndex(IndexMap, IndexType);
  return Success;
}

bool TagRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, FieldList);
}

bool ClassRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= TagRecord::remapTypeIndices(IndexMap);
  Success &= remapIndex(IndexMap, DerivationList);
  Success &= remapIndex(IndexMap, VTableShape);
  return Success;
}

bool EnumRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= TagRecord::remapTypeIndices(IndexMap);
  Success &= remapIndex(IndexMap, UnderlyingType);
  return Success;
}

bool BitFieldRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Type);
}

bool VFTableShapeRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return true;
}

bool TypeServer2Record::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return true;
}

bool StringIdRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Id);
}

bool FuncIdRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, ParentScope);
  Success &= remapIndex(IndexMap, FunctionType);
  return Success;
}

bool UdtSourceLineRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, UDT);
  Success &= remapIndex(IndexMap, SourceFile);
  return Success;
}

bool UdtModSourceLineRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, UDT);
  Success &= remapIndex(IndexMap, SourceFile);
  return Success;
}

bool BuildInfoRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  for (TypeIndex &Arg : ArgIndices)
    Success &= remapIndex(IndexMap, Arg);
  return Success;
}

bool VFTableRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, CompleteClass);
  Success &= remapIndex(IndexMap, OverriddenVFTable);
  return Success;
}

bool OneMethodRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, Type);
  return Success;
}

bool MethodOverloadListRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  for (OneMethodRecord &Meth : Methods)
    if ((Success = Meth.remapTypeIndices(IndexMap)))
      return Success;
  return Success;
}

bool OverloadedMethodRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, MethodList);
}

bool DataMemberRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Type);
}

bool StaticDataMemberRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Type);
}

bool EnumeratorRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return true;
}

bool VFPtrRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Type);
}

bool BaseClassRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, Type);
}

bool VirtualBaseClassRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  bool Success = true;
  Success &= remapIndex(IndexMap, BaseType);
  Success &= remapIndex(IndexMap, VBPtrType);
  return Success;
}

bool ListContinuationRecord::remapTypeIndices(ArrayRef<TypeIndex> IndexMap) {
  return remapIndex(IndexMap, ContinuationIndex);
}
