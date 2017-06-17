//===- TypeName.cpp ------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeName.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::codeview;

namespace {
class TypeNameComputer : public TypeVisitorCallbacks {
  /// The type collection.  Used to calculate names of nested types.
  TypeCollection &Types;
  TypeIndex CurrentTypeIndex = TypeIndex::None();

  /// Name of the current type. Only valid before visitTypeEnd.
  SmallString<256> Name;

public:
  explicit TypeNameComputer(TypeCollection &Types) : Types(Types) {}

  StringRef name() const { return Name; }

  /// Paired begin/end actions for all types. Receives all record data,
  /// including the fixed-length record prefix.
  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeBegin(CVType &Record, TypeIndex Index) override;
  Error visitTypeEnd(CVType &Record) override;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(CVType &CVR, Name##Record &Record) override;
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"
};
} // namespace

Error TypeNameComputer::visitTypeBegin(CVType &Record) {
  llvm_unreachable("Must call visitTypeBegin with a TypeIndex!");
  return Error::success();
}

Error TypeNameComputer::visitTypeBegin(CVType &Record, TypeIndex Index) {
  // Reset Name to the empty string. If the visitor sets it, we know it.
  Name = "";
  CurrentTypeIndex = Index;
  return Error::success();
}

Error TypeNameComputer::visitTypeEnd(CVType &CVR) { return Error::success(); }

Error TypeNameComputer::visitKnownRecord(CVType &CVR,
                                         FieldListRecord &FieldList) {
  Name = "<field list>";
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVRecord<TypeLeafKind> &CVR,
                                         StringIdRecord &String) {
  Name = String.getString();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, ArgListRecord &Args) {
  auto Indices = Args.getIndices();
  uint32_t Size = Indices.size();
  Name = "(";
  for (uint32_t I = 0; I < Size; ++I) {
    assert(Indices[I] < CurrentTypeIndex);

    Name.append(Types.getTypeName(Indices[I]));
    if (I + 1 != Size)
      Name.append(", ");
  }
  Name.push_back(')');
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR,
                                         StringListRecord &Strings) {
  auto Indices = Strings.getIndices();
  uint32_t Size = Indices.size();
  Name = "\"";
  for (uint32_t I = 0; I < Size; ++I) {
    Name.append(Types.getTypeName(Indices[I]));
    if (I + 1 != Size)
      Name.append("\" \"");
  }
  Name.push_back('\"');
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, ClassRecord &Class) {
  Name = Class.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, UnionRecord &Union) {
  Name = Union.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, EnumRecord &Enum) {
  Name = Enum.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, ArrayRecord &AT) {
  Name = AT.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, VFTableRecord &VFT) {
  Name = VFT.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, MemberFuncIdRecord &Id) {
  Name = Id.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, ProcedureRecord &Proc) {
  StringRef Ret = Types.getTypeName(Proc.getReturnType());
  StringRef Params = Types.getTypeName(Proc.getArgumentList());
  Name = formatv("{0} {1}", Ret, Params).sstr<256>();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR,
                                         MemberFunctionRecord &MF) {
  StringRef Ret = Types.getTypeName(MF.getReturnType());
  StringRef Class = Types.getTypeName(MF.getClassType());
  StringRef Params = Types.getTypeName(MF.getArgumentList());
  Name = formatv("{0} {1}::{2}", Ret, Class, Params).sstr<256>();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, FuncIdRecord &Func) {
  Name = Func.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, TypeServer2Record &TS) {
  Name = TS.getName();
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, PointerRecord &Ptr) {

  if (Ptr.isPointerToMember()) {
    const MemberPointerInfo &MI = Ptr.getMemberInfo();

    StringRef Pointee = Types.getTypeName(Ptr.getReferentType());
    StringRef Class = Types.getTypeName(MI.getContainingType());
    Name = formatv("{0} {1}::*", Pointee, Class);
  } else {
    if (Ptr.isConst())
      Name.append("const ");
    if (Ptr.isVolatile())
      Name.append("volatile ");
    if (Ptr.isUnaligned())
      Name.append("__unaligned ");

    Name.append(Types.getTypeName(Ptr.getReferentType()));

    if (Ptr.getMode() == PointerMode::LValueReference)
      Name.append("&");
    else if (Ptr.getMode() == PointerMode::RValueReference)
      Name.append("&&");
    else if (Ptr.getMode() == PointerMode::Pointer)
      Name.append("*");
  }
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, ModifierRecord &Mod) {
  uint16_t Mods = static_cast<uint16_t>(Mod.getModifiers());

  SmallString<256> TypeName;
  if (Mods & uint16_t(ModifierOptions::Const))
    Name.append("const ");
  if (Mods & uint16_t(ModifierOptions::Volatile))
    Name.append("volatile ");
  if (Mods & uint16_t(ModifierOptions::Unaligned))
    Name.append("__unaligned ");
  Name.append(Types.getTypeName(Mod.getModifiedType()));
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR,
                                         VFTableShapeRecord &Shape) {
  Name = formatv("<vftable {0} methods>", Shape.getEntryCount());
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(
    CVType &CVR, UdtModSourceLineRecord &ModSourceLine) {
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR,
                                         UdtSourceLineRecord &SourceLine) {
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, BitFieldRecord &BF) {
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR,
                                         MethodOverloadListRecord &Overloads) {
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, BuildInfoRecord &BI) {
  return Error::success();
}

Error TypeNameComputer::visitKnownRecord(CVType &CVR, LabelRecord &R) {
  return Error::success();
}

std::string llvm::codeview::computeTypeName(TypeCollection &Types,
                                            TypeIndex Index) {
  TypeNameComputer Computer(Types);
  CVType Record = Types.getType(Index);
  if (auto EC = visitTypeRecord(Record, Index, Computer)) {
    consumeError(std::move(EC));
    return "<unknown UDT>";
  }
  return Computer.name();
}
