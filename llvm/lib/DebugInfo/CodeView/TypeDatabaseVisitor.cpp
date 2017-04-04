//===- TypeDatabaseVisitor.cpp -------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeDatabaseVisitor.h"

#include "llvm/ADT/SmallString.h"

using namespace llvm;

using namespace llvm::codeview;

Error TypeDatabaseVisitor::visitTypeBegin(CVRecord<TypeLeafKind> &Record) {
  assert(!IsInFieldList);
  // Reset Name to the empty string. If the visitor sets it, we know it.
  Name = "";

  if (Record.Type == LF_FIELDLIST) {
    // Record that we're in a field list so that members do not get assigned
    // type indices.
    IsInFieldList = true;
  }
  return Error::success();
}

Error TypeDatabaseVisitor::visitTypeEnd(CVType &CVR) {
  if (CVR.Type == LF_FIELDLIST) {
    assert(IsInFieldList);
    IsInFieldList = false;
  }
  assert(!IsInFieldList);

  // Record every type that is not a field list member, even if Name is empty.
  // CVUDTNames is indexed by type index, and must have one entry for every
  // type.  Field list members are not recorded, and are only referenced by
  // their containing field list record.
  TypeDB.recordType(Name, CVR);
  return Error::success();
}

Error TypeDatabaseVisitor::visitMemberBegin(CVMemberRecord &Record) {
  assert(IsInFieldList);
  // Reset Name to the empty string. If the visitor sets it, we know it.
  Name = "";
  return Error::success();
}

Error TypeDatabaseVisitor::visitMemberEnd(CVMemberRecord &Record) {
  assert(IsInFieldList);
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            FieldListRecord &FieldList) {
  Name = "<field list>";
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVRecord<TypeLeafKind> &CVR,
                                            StringIdRecord &String) {
  // Put this in the database so it gets printed with LF_UDT_SRC_LINE.
  Name = String.getString();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, ArgListRecord &Args) {
  auto Indices = Args.getIndices();
  uint32_t Size = Indices.size();
  SmallString<256> TypeName("(");
  for (uint32_t I = 0; I < Size; ++I) {
    StringRef ArgTypeName = TypeDB.getTypeName(Indices[I]);
    TypeName.append(ArgTypeName);
    if (I + 1 != Size)
      TypeName.append(", ");
  }
  TypeName.push_back(')');
  Name = TypeDB.saveTypeName(TypeName);
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            StringListRecord &Strings) {
  auto Indices = Strings.getIndices();
  uint32_t Size = Indices.size();
  SmallString<256> TypeName("\"");
  for (uint32_t I = 0; I < Size; ++I) {
    StringRef ArgTypeName = TypeDB.getTypeName(Indices[I]);
    TypeName.append(ArgTypeName);
    if (I + 1 != Size)
      TypeName.append("\" \"");
  }
  TypeName.push_back('\"');
  Name = TypeDB.saveTypeName(TypeName);
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, ClassRecord &Class) {
  Name = Class.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, UnionRecord &Union) {
  Name = Union.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, EnumRecord &Enum) {
  Name = Enum.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, ArrayRecord &AT) {
  Name = AT.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, VFTableRecord &VFT) {
  Name = VFT.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            MemberFuncIdRecord &Id) {
  Name = Id.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            ProcedureRecord &Proc) {
  StringRef ReturnTypeName = TypeDB.getTypeName(Proc.getReturnType());
  StringRef ArgListTypeName = TypeDB.getTypeName(Proc.getArgumentList());
  SmallString<256> TypeName(ReturnTypeName);
  TypeName.push_back(' ');
  TypeName.append(ArgListTypeName);
  Name = TypeDB.saveTypeName(TypeName);
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            MemberFunctionRecord &MF) {
  StringRef ReturnTypeName = TypeDB.getTypeName(MF.getReturnType());
  StringRef ClassTypeName = TypeDB.getTypeName(MF.getClassType());
  StringRef ArgListTypeName = TypeDB.getTypeName(MF.getArgumentList());
  SmallString<256> TypeName(ReturnTypeName);
  TypeName.push_back(' ');
  TypeName.append(ClassTypeName);
  TypeName.append("::");
  TypeName.append(ArgListTypeName);
  Name = TypeDB.saveTypeName(TypeName);
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, FuncIdRecord &Func) {
  Name = Func.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            TypeServer2Record &TS) {
  Name = TS.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, PointerRecord &Ptr) {

  if (Ptr.isPointerToMember()) {
    const MemberPointerInfo &MI = Ptr.getMemberInfo();

    StringRef PointeeName = TypeDB.getTypeName(Ptr.getReferentType());
    StringRef ClassName = TypeDB.getTypeName(MI.getContainingType());
    SmallString<256> TypeName(PointeeName);
    TypeName.push_back(' ');
    TypeName.append(ClassName);
    TypeName.append("::*");
    Name = TypeDB.saveTypeName(TypeName);
  } else {
    SmallString<256> TypeName;
    if (Ptr.isConst())
      TypeName.append("const ");
    if (Ptr.isVolatile())
      TypeName.append("volatile ");
    if (Ptr.isUnaligned())
      TypeName.append("__unaligned ");

    TypeName.append(TypeDB.getTypeName(Ptr.getReferentType()));

    if (Ptr.getMode() == PointerMode::LValueReference)
      TypeName.append("&");
    else if (Ptr.getMode() == PointerMode::RValueReference)
      TypeName.append("&&");
    else if (Ptr.getMode() == PointerMode::Pointer)
      TypeName.append("*");

    if (!TypeName.empty())
      Name = TypeDB.saveTypeName(TypeName);
  }
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, ModifierRecord &Mod) {
  uint16_t Mods = static_cast<uint16_t>(Mod.getModifiers());

  StringRef ModifiedName = TypeDB.getTypeName(Mod.getModifiedType());
  SmallString<256> TypeName;
  if (Mods & uint16_t(ModifierOptions::Const))
    TypeName.append("const ");
  if (Mods & uint16_t(ModifierOptions::Volatile))
    TypeName.append("volatile ");
  if (Mods & uint16_t(ModifierOptions::Unaligned))
    TypeName.append("__unaligned ");
  TypeName.append(ModifiedName);
  Name = TypeDB.saveTypeName(TypeName);
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            VFTableShapeRecord &Shape) {
  Name = TypeDB.saveTypeName("<vftable " + utostr(Shape.getEntryCount()) +
                             " methods>");
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            NestedTypeRecord &Nested) {
  Name = Nested.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            OneMethodRecord &Method) {
  Name = Method.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            OverloadedMethodRecord &Method) {
  Name = Method.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            DataMemberRecord &Field) {
  Name = Field.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            StaticDataMemberRecord &Field) {
  Name = Field.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            EnumeratorRecord &Enum) {
  Name = Enum.getName();
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            BaseClassRecord &Base) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            VirtualBaseClassRecord &VBase) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            ListContinuationRecord &Cont) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(
    CVType &CVR, UdtModSourceLineRecord &ModSourceLine) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR,
                                            UdtSourceLineRecord &SourceLine) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, BitFieldRecord &BF) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(
    CVType &CVR, MethodOverloadListRecord &Overloads) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, BuildInfoRecord &BI) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownRecord(CVType &CVR, LabelRecord &R) {
  return Error::success();
}

Error TypeDatabaseVisitor::visitKnownMember(CVMemberRecord &CVR,
                                            VFPtrRecord &VFP) {
  return Error::success();
}
