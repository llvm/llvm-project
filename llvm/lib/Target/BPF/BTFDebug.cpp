//===- BTFDebug.cpp - BTF Generator ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing BTF debug info.
//
//===----------------------------------------------------------------------===//

#include "BTFDebug.h"
#include "BPF.h"
#include "BPFCORE.h"
#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm;

static const char *BTFKindStr[] = {
#define HANDLE_BTF_KIND(ID, NAME) "BTF_KIND_" #NAME,
#include "BTF.def"
};

static const DIType * stripQualifiers(const DIType *Ty) {
  while (const auto *DTy = dyn_cast<DIDerivedType>(Ty)) {
    unsigned Tag = DTy->getTag();
    if (Tag != dwarf::DW_TAG_typedef && Tag != dwarf::DW_TAG_const_type &&
        Tag != dwarf::DW_TAG_volatile_type && Tag != dwarf::DW_TAG_restrict_type)
      break;
    Ty = DTy->getBaseType();
  }

  return Ty;
}

/// Emit a BTF common type.
void BTFTypeBase::emitType(MCStreamer &OS) {
  OS.AddComment(std::string(BTFKindStr[Kind]) + "(id = " + std::to_string(Id) +
                ")");
  OS.EmitIntValue(BTFType.NameOff, 4);
  OS.AddComment("0x" + Twine::utohexstr(BTFType.Info));
  OS.EmitIntValue(BTFType.Info, 4);
  OS.EmitIntValue(BTFType.Size, 4);
}

BTFTypeDerived::BTFTypeDerived(const DIDerivedType *DTy, unsigned Tag,
                               bool NeedsFixup)
    : DTy(DTy), NeedsFixup(NeedsFixup) {
  switch (Tag) {
  case dwarf::DW_TAG_pointer_type:
    Kind = BTF::BTF_KIND_PTR;
    break;
  case dwarf::DW_TAG_const_type:
    Kind = BTF::BTF_KIND_CONST;
    break;
  case dwarf::DW_TAG_volatile_type:
    Kind = BTF::BTF_KIND_VOLATILE;
    break;
  case dwarf::DW_TAG_typedef:
    Kind = BTF::BTF_KIND_TYPEDEF;
    break;
  case dwarf::DW_TAG_restrict_type:
    Kind = BTF::BTF_KIND_RESTRICT;
    break;
  default:
    llvm_unreachable("Unknown DIDerivedType Tag");
  }
  BTFType.Info = Kind << 24;
}

void BTFTypeDerived::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(DTy->getName());

  if (NeedsFixup)
    return;

  // The base type for PTR/CONST/VOLATILE could be void.
  const DIType *ResolvedType = DTy->getBaseType();
  if (!ResolvedType) {
    assert((Kind == BTF::BTF_KIND_PTR || Kind == BTF::BTF_KIND_CONST ||
            Kind == BTF::BTF_KIND_VOLATILE) &&
           "Invalid null basetype");
    BTFType.Type = 0;
  } else {
    BTFType.Type = BDebug.getTypeId(ResolvedType);
  }
}

void BTFTypeDerived::emitType(MCStreamer &OS) { BTFTypeBase::emitType(OS); }

void BTFTypeDerived::setPointeeType(uint32_t PointeeType) {
  BTFType.Type = PointeeType;
}

/// Represent a struct/union forward declaration.
BTFTypeFwd::BTFTypeFwd(StringRef Name, bool IsUnion) : Name(Name) {
  Kind = BTF::BTF_KIND_FWD;
  BTFType.Info = IsUnion << 31 | Kind << 24;
  BTFType.Type = 0;
}

void BTFTypeFwd::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

void BTFTypeFwd::emitType(MCStreamer &OS) { BTFTypeBase::emitType(OS); }

BTFTypeInt::BTFTypeInt(uint32_t Encoding, uint32_t SizeInBits,
                       uint32_t OffsetInBits, StringRef TypeName)
    : Name(TypeName) {
  // Translate IR int encoding to BTF int encoding.
  uint8_t BTFEncoding;
  switch (Encoding) {
  case dwarf::DW_ATE_boolean:
    BTFEncoding = BTF::INT_BOOL;
    break;
  case dwarf::DW_ATE_signed:
  case dwarf::DW_ATE_signed_char:
    BTFEncoding = BTF::INT_SIGNED;
    break;
  case dwarf::DW_ATE_unsigned:
  case dwarf::DW_ATE_unsigned_char:
    BTFEncoding = 0;
    break;
  default:
    llvm_unreachable("Unknown BTFTypeInt Encoding");
  }

  Kind = BTF::BTF_KIND_INT;
  BTFType.Info = Kind << 24;
  BTFType.Size = roundupToBytes(SizeInBits);
  IntVal = (BTFEncoding << 24) | OffsetInBits << 16 | SizeInBits;
}

void BTFTypeInt::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

void BTFTypeInt::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.AddComment("0x" + Twine::utohexstr(IntVal));
  OS.EmitIntValue(IntVal, 4);
}

BTFTypeEnum::BTFTypeEnum(const DICompositeType *ETy, uint32_t VLen) : ETy(ETy) {
  Kind = BTF::BTF_KIND_ENUM;
  BTFType.Info = Kind << 24 | VLen;
  BTFType.Size = roundupToBytes(ETy->getSizeInBits());
}

void BTFTypeEnum::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(ETy->getName());

  DINodeArray Elements = ETy->getElements();
  for (const auto Element : Elements) {
    const auto *Enum = cast<DIEnumerator>(Element);

    struct BTF::BTFEnum BTFEnum;
    BTFEnum.NameOff = BDebug.addString(Enum->getName());
    // BTF enum value is 32bit, enforce it.
    BTFEnum.Val = static_cast<uint32_t>(Enum->getValue());
    EnumValues.push_back(BTFEnum);
  }
}

void BTFTypeEnum::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Enum : EnumValues) {
    OS.EmitIntValue(Enum.NameOff, 4);
    OS.EmitIntValue(Enum.Val, 4);
  }
}

BTFTypeArray::BTFTypeArray(const DIType *Ty, uint32_t ElemTypeId,
                           uint32_t ElemSize, uint32_t NumElems)
    : ElemTyNoQual(Ty), ElemSize(ElemSize) {
  Kind = BTF::BTF_KIND_ARRAY;
  BTFType.NameOff = 0;
  BTFType.Info = Kind << 24;
  BTFType.Size = 0;

  ArrayInfo.ElemType = ElemTypeId;
  ArrayInfo.Nelems = NumElems;
}

/// Represent a BTF array.
void BTFTypeArray::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  // The IR does not really have a type for the index.
  // A special type for array index should have been
  // created during initial type traversal. Just
  // retrieve that type id.
  ArrayInfo.IndexType = BDebug.getArrayIndexTypeId();

  ElemTypeNoQual = ElemTyNoQual ? BDebug.getTypeId(ElemTyNoQual)
                                : ArrayInfo.ElemType;
}

void BTFTypeArray::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.EmitIntValue(ArrayInfo.ElemType, 4);
  OS.EmitIntValue(ArrayInfo.IndexType, 4);
  OS.EmitIntValue(ArrayInfo.Nelems, 4);
}

void BTFTypeArray::getLocInfo(uint32_t Loc, uint32_t &LocOffset,
                              uint32_t &ElementTypeId) {
  ElementTypeId = ElemTypeNoQual;
  LocOffset = Loc * ElemSize;
}

/// Represent either a struct or a union.
BTFTypeStruct::BTFTypeStruct(const DICompositeType *STy, bool IsStruct,
                             bool HasBitField, uint32_t Vlen)
    : STy(STy), HasBitField(HasBitField) {
  Kind = IsStruct ? BTF::BTF_KIND_STRUCT : BTF::BTF_KIND_UNION;
  BTFType.Size = roundupToBytes(STy->getSizeInBits());
  BTFType.Info = (HasBitField << 31) | (Kind << 24) | Vlen;
}

void BTFTypeStruct::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(STy->getName());

  // Add struct/union members.
  const DINodeArray Elements = STy->getElements();
  for (const auto *Element : Elements) {
    struct BTF::BTFMember BTFMember;
    const auto *DDTy = cast<DIDerivedType>(Element);

    BTFMember.NameOff = BDebug.addString(DDTy->getName());
    if (HasBitField) {
      uint8_t BitFieldSize = DDTy->isBitField() ? DDTy->getSizeInBits() : 0;
      BTFMember.Offset = BitFieldSize << 24 | DDTy->getOffsetInBits();
    } else {
      BTFMember.Offset = DDTy->getOffsetInBits();
    }
    const auto *BaseTy = DDTy->getBaseType();
    BTFMember.Type = BDebug.getTypeId(BaseTy);
    MemberTypeNoQual.push_back(BDebug.getTypeId(stripQualifiers(BaseTy)));
    Members.push_back(BTFMember);
  }
}

void BTFTypeStruct::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Member : Members) {
    OS.EmitIntValue(Member.NameOff, 4);
    OS.EmitIntValue(Member.Type, 4);
    OS.AddComment("0x" + Twine::utohexstr(Member.Offset));
    OS.EmitIntValue(Member.Offset, 4);
  }
}

std::string BTFTypeStruct::getName() { return STy->getName(); }

void BTFTypeStruct::getMemberInfo(uint32_t Loc, uint32_t &MemberOffset,
                                  uint32_t &MemberType) {
  MemberType = MemberTypeNoQual[Loc];
  MemberOffset =
      HasBitField ? Members[Loc].Offset & 0xffffff : Members[Loc].Offset;
}

uint32_t BTFTypeStruct::getStructSize() { return STy->getSizeInBits() >> 3; }

/// The Func kind represents both subprogram and pointee of function
/// pointers. If the FuncName is empty, it represents a pointee of function
/// pointer. Otherwise, it represents a subprogram. The func arg names
/// are empty for pointee of function pointer case, and are valid names
/// for subprogram.
BTFTypeFuncProto::BTFTypeFuncProto(
    const DISubroutineType *STy, uint32_t VLen,
    const std::unordered_map<uint32_t, StringRef> &FuncArgNames)
    : STy(STy), FuncArgNames(FuncArgNames) {
  Kind = BTF::BTF_KIND_FUNC_PROTO;
  BTFType.Info = (Kind << 24) | VLen;
}

void BTFTypeFuncProto::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  DITypeRefArray Elements = STy->getTypeArray();
  auto RetType = Elements[0];
  BTFType.Type = RetType ? BDebug.getTypeId(RetType) : 0;
  BTFType.NameOff = 0;

  // For null parameter which is typically the last one
  // to represent the vararg, encode the NameOff/Type to be 0.
  for (unsigned I = 1, N = Elements.size(); I < N; ++I) {
    struct BTF::BTFParam Param;
    auto Element = Elements[I];
    if (Element) {
      Param.NameOff = BDebug.addString(FuncArgNames[I]);
      Param.Type = BDebug.getTypeId(Element);
    } else {
      Param.NameOff = 0;
      Param.Type = 0;
    }
    Parameters.push_back(Param);
  }
}

void BTFTypeFuncProto::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Param : Parameters) {
    OS.EmitIntValue(Param.NameOff, 4);
    OS.EmitIntValue(Param.Type, 4);
  }
}

BTFTypeFunc::BTFTypeFunc(StringRef FuncName, uint32_t ProtoTypeId)
    : Name(FuncName) {
  Kind = BTF::BTF_KIND_FUNC;
  BTFType.Info = Kind << 24;
  BTFType.Type = ProtoTypeId;
}

void BTFTypeFunc::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

void BTFTypeFunc::emitType(MCStreamer &OS) { BTFTypeBase::emitType(OS); }

BTFKindVar::BTFKindVar(StringRef VarName, uint32_t TypeId, uint32_t VarInfo)
    : Name(VarName) {
  Kind = BTF::BTF_KIND_VAR;
  BTFType.Info = Kind << 24;
  BTFType.Type = TypeId;
  Info = VarInfo;
}

void BTFKindVar::completeType(BTFDebug &BDebug) {
  BTFType.NameOff = BDebug.addString(Name);
}

void BTFKindVar::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.EmitIntValue(Info, 4);
}

BTFKindDataSec::BTFKindDataSec(AsmPrinter *AsmPrt, std::string SecName)
    : Asm(AsmPrt), Name(SecName) {
  Kind = BTF::BTF_KIND_DATASEC;
  BTFType.Info = Kind << 24;
  BTFType.Size = 0;
}

void BTFKindDataSec::completeType(BTFDebug &BDebug) {
  BTFType.NameOff = BDebug.addString(Name);
  BTFType.Info |= Vars.size();
}

void BTFKindDataSec::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);

  for (const auto &V : Vars) {
    OS.EmitIntValue(std::get<0>(V), 4);
    Asm->EmitLabelReference(std::get<1>(V), 4);
    OS.EmitIntValue(std::get<2>(V), 4);
  }
}

uint32_t BTFStringTable::addString(StringRef S) {
  // Check whether the string already exists.
  for (auto &OffsetM : OffsetToIdMap) {
    if (Table[OffsetM.second] == S)
      return OffsetM.first;
  }
  // Not find, add to the string table.
  uint32_t Offset = Size;
  OffsetToIdMap[Offset] = Table.size();
  Table.push_back(S);
  Size += S.size() + 1;
  return Offset;
}

BTFDebug::BTFDebug(AsmPrinter *AP)
    : DebugHandlerBase(AP), OS(*Asm->OutStreamer), SkipInstruction(false),
      LineInfoGenerated(false), SecNameOff(0), ArrayIndexTypeId(0),
      MapDefNotCollected(true) {
  addString("\0");
}

uint32_t BTFDebug::addType(std::unique_ptr<BTFTypeBase> TypeEntry,
                           const DIType *Ty) {
  TypeEntry->setId(TypeEntries.size() + 1);
  uint32_t Id = TypeEntry->getId();
  DIToIdMap[Ty] = Id;
  TypeEntries.push_back(std::move(TypeEntry));
  return Id;
}

uint32_t BTFDebug::addType(std::unique_ptr<BTFTypeBase> TypeEntry) {
  TypeEntry->setId(TypeEntries.size() + 1);
  uint32_t Id = TypeEntry->getId();
  TypeEntries.push_back(std::move(TypeEntry));
  return Id;
}

void BTFDebug::visitBasicType(const DIBasicType *BTy, uint32_t &TypeId) {
  // Only int types are supported in BTF.
  uint32_t Encoding = BTy->getEncoding();
  if (Encoding != dwarf::DW_ATE_boolean && Encoding != dwarf::DW_ATE_signed &&
      Encoding != dwarf::DW_ATE_signed_char &&
      Encoding != dwarf::DW_ATE_unsigned &&
      Encoding != dwarf::DW_ATE_unsigned_char)
    return;

  // Create a BTF type instance for this DIBasicType and put it into
  // DIToIdMap for cross-type reference check.
  auto TypeEntry = llvm::make_unique<BTFTypeInt>(
      Encoding, BTy->getSizeInBits(), BTy->getOffsetInBits(), BTy->getName());
  TypeId = addType(std::move(TypeEntry), BTy);
}

/// Handle subprogram or subroutine types.
void BTFDebug::visitSubroutineType(
    const DISubroutineType *STy, bool ForSubprog,
    const std::unordered_map<uint32_t, StringRef> &FuncArgNames,
    uint32_t &TypeId) {
  DITypeRefArray Elements = STy->getTypeArray();
  uint32_t VLen = Elements.size() - 1;
  if (VLen > BTF::MAX_VLEN)
    return;

  // Subprogram has a valid non-zero-length name, and the pointee of
  // a function pointer has an empty name. The subprogram type will
  // not be added to DIToIdMap as it should not be referenced by
  // any other types.
  auto TypeEntry = llvm::make_unique<BTFTypeFuncProto>(STy, VLen, FuncArgNames);
  if (ForSubprog)
    TypeId = addType(std::move(TypeEntry)); // For subprogram
  else
    TypeId = addType(std::move(TypeEntry), STy); // For func ptr

  // Visit return type and func arg types.
  for (const auto Element : Elements) {
    visitTypeEntry(Element);
  }
}

/// Handle structure/union types.
void BTFDebug::visitStructType(const DICompositeType *CTy, bool IsStruct,
                               uint32_t &TypeId) {
  const DINodeArray Elements = CTy->getElements();
  uint32_t VLen = Elements.size();
  if (VLen > BTF::MAX_VLEN)
    return;

  // Check whether we have any bitfield members or not
  bool HasBitField = false;
  for (const auto *Element : Elements) {
    auto E = cast<DIDerivedType>(Element);
    if (E->isBitField()) {
      HasBitField = true;
      break;
    }
  }

  auto TypeEntry =
      llvm::make_unique<BTFTypeStruct>(CTy, IsStruct, HasBitField, VLen);
  StructTypes.push_back(TypeEntry.get());
  TypeId = addType(std::move(TypeEntry), CTy);

  // Visit all struct members.
  for (const auto *Element : Elements)
    visitTypeEntry(cast<DIDerivedType>(Element));
}

void BTFDebug::visitArrayType(const DICompositeType *CTy, uint32_t &TypeId) {
  // Visit array element type.
  uint32_t ElemTypeId, ElemSize;
  const DIType *ElemType = CTy->getBaseType();
  visitTypeEntry(ElemType, ElemTypeId, false, false);

  // Strip qualifiers from element type to get accurate element size.
  ElemType = stripQualifiers(ElemType);
  ElemSize = ElemType->getSizeInBits() >> 3;

  if (!CTy->getSizeInBits()) {
    auto TypeEntry = llvm::make_unique<BTFTypeArray>(ElemType, ElemTypeId, 0, 0);
    ArrayTypes.push_back(TypeEntry.get());
    ElemTypeId = addType(std::move(TypeEntry), CTy);
  } else {
    // Visit array dimensions.
    DINodeArray Elements = CTy->getElements();
    for (int I = Elements.size() - 1; I >= 0; --I) {
      if (auto *Element = dyn_cast_or_null<DINode>(Elements[I]))
        if (Element->getTag() == dwarf::DW_TAG_subrange_type) {
          const DISubrange *SR = cast<DISubrange>(Element);
          auto *CI = SR->getCount().dyn_cast<ConstantInt *>();
          int64_t Count = CI->getSExtValue();
          const DIType *ArrayElemTy = (I == 0) ? ElemType : nullptr;

          auto TypeEntry =
              llvm::make_unique<BTFTypeArray>(ArrayElemTy, ElemTypeId,
                                              ElemSize, Count);
          ArrayTypes.push_back(TypeEntry.get());
          if (I == 0)
            ElemTypeId = addType(std::move(TypeEntry), CTy);
          else
            ElemTypeId = addType(std::move(TypeEntry));
          ElemSize = ElemSize * Count;
        }
    }
  }

  // The array TypeId is the type id of the outermost dimension.
  TypeId = ElemTypeId;

  // The IR does not have a type for array index while BTF wants one.
  // So create an array index type if there is none.
  if (!ArrayIndexTypeId) {
    auto TypeEntry = llvm::make_unique<BTFTypeInt>(dwarf::DW_ATE_unsigned, 32,
                                                   0, "__ARRAY_SIZE_TYPE__");
    ArrayIndexTypeId = addType(std::move(TypeEntry));
  }
}

void BTFDebug::visitEnumType(const DICompositeType *CTy, uint32_t &TypeId) {
  DINodeArray Elements = CTy->getElements();
  uint32_t VLen = Elements.size();
  if (VLen > BTF::MAX_VLEN)
    return;

  auto TypeEntry = llvm::make_unique<BTFTypeEnum>(CTy, VLen);
  TypeId = addType(std::move(TypeEntry), CTy);
  // No need to visit base type as BTF does not encode it.
}

/// Handle structure/union forward declarations.
void BTFDebug::visitFwdDeclType(const DICompositeType *CTy, bool IsUnion,
                                uint32_t &TypeId) {
  auto TypeEntry = llvm::make_unique<BTFTypeFwd>(CTy->getName(), IsUnion);
  TypeId = addType(std::move(TypeEntry), CTy);
}

/// Handle structure, union, array and enumeration types.
void BTFDebug::visitCompositeType(const DICompositeType *CTy,
                                  uint32_t &TypeId) {
  auto Tag = CTy->getTag();
  if (Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type) {
    // Handle forward declaration differently as it does not have members.
    if (CTy->isForwardDecl())
      visitFwdDeclType(CTy, Tag == dwarf::DW_TAG_union_type, TypeId);
    else
      visitStructType(CTy, Tag == dwarf::DW_TAG_structure_type, TypeId);
  } else if (Tag == dwarf::DW_TAG_array_type)
    visitArrayType(CTy, TypeId);
  else if (Tag == dwarf::DW_TAG_enumeration_type)
    visitEnumType(CTy, TypeId);
}

/// Handle pointer, typedef, const, volatile, restrict and member types.
void BTFDebug::visitDerivedType(const DIDerivedType *DTy, uint32_t &TypeId,
                                bool CheckPointer, bool SeenPointer) {
  unsigned Tag = DTy->getTag();

  /// Try to avoid chasing pointees, esp. structure pointees which may
  /// unnecessary bring in a lot of types.
  if (CheckPointer && !SeenPointer) {
    SeenPointer = Tag == dwarf::DW_TAG_pointer_type;
  }

  if (CheckPointer && SeenPointer) {
    const DIType *Base = DTy->getBaseType();
    if (Base) {
      if (const auto *CTy = dyn_cast<DICompositeType>(Base)) {
        auto CTag = CTy->getTag();
        if ((CTag == dwarf::DW_TAG_structure_type ||
             CTag == dwarf::DW_TAG_union_type) &&
            !CTy->isForwardDecl()) {
          /// Find a candidate, generate a fixup. Later on the struct/union
          /// pointee type will be replaced with either a real type or
          /// a forward declaration.
          auto TypeEntry = llvm::make_unique<BTFTypeDerived>(DTy, Tag, true);
          auto &Fixup = FixupDerivedTypes[CTy->getName()];
          Fixup.first = CTag == dwarf::DW_TAG_union_type;
          Fixup.second.push_back(TypeEntry.get());
          TypeId = addType(std::move(TypeEntry), DTy);
          return;
        }
      }
    }
  }

  if (Tag == dwarf::DW_TAG_pointer_type || Tag == dwarf::DW_TAG_typedef ||
      Tag == dwarf::DW_TAG_const_type || Tag == dwarf::DW_TAG_volatile_type ||
      Tag == dwarf::DW_TAG_restrict_type) {
    auto TypeEntry = llvm::make_unique<BTFTypeDerived>(DTy, Tag, false);
    TypeId = addType(std::move(TypeEntry), DTy);
  } else if (Tag != dwarf::DW_TAG_member) {
    return;
  }

  // Visit base type of pointer, typedef, const, volatile, restrict or
  // struct/union member.
  uint32_t TempTypeId = 0;
  if (Tag == dwarf::DW_TAG_member)
    visitTypeEntry(DTy->getBaseType(), TempTypeId, true, false);
  else
    visitTypeEntry(DTy->getBaseType(), TempTypeId, CheckPointer, SeenPointer);
}

void BTFDebug::visitTypeEntry(const DIType *Ty, uint32_t &TypeId,
                              bool CheckPointer, bool SeenPointer) {
  if (!Ty || DIToIdMap.find(Ty) != DIToIdMap.end()) {
    TypeId = DIToIdMap[Ty];
    return;
  }

  if (const auto *BTy = dyn_cast<DIBasicType>(Ty))
    visitBasicType(BTy, TypeId);
  else if (const auto *STy = dyn_cast<DISubroutineType>(Ty))
    visitSubroutineType(STy, false, std::unordered_map<uint32_t, StringRef>(),
                        TypeId);
  else if (const auto *CTy = dyn_cast<DICompositeType>(Ty))
    visitCompositeType(CTy, TypeId);
  else if (const auto *DTy = dyn_cast<DIDerivedType>(Ty))
    visitDerivedType(DTy, TypeId, CheckPointer, SeenPointer);
  else
    llvm_unreachable("Unknown DIType");
}

void BTFDebug::visitTypeEntry(const DIType *Ty) {
  uint32_t TypeId;
  visitTypeEntry(Ty, TypeId, false, false);
}

void BTFDebug::visitMapDefType(const DIType *Ty, uint32_t &TypeId) {
  if (!Ty || DIToIdMap.find(Ty) != DIToIdMap.end()) {
    TypeId = DIToIdMap[Ty];
    return;
  }

  // MapDef type is a struct type
  const auto *CTy = dyn_cast<DICompositeType>(Ty);
  if (!CTy)
    return;

  auto Tag = CTy->getTag();
  if (Tag != dwarf::DW_TAG_structure_type || CTy->isForwardDecl())
    return;

  // Record this type
  const DINodeArray Elements = CTy->getElements();
  bool HasBitField = false;
  for (const auto *Element : Elements) {
    auto E = cast<DIDerivedType>(Element);
    if (E->isBitField()) {
      HasBitField = true;
      break;
    }
  }

  auto TypeEntry =
      llvm::make_unique<BTFTypeStruct>(CTy, true, HasBitField, Elements.size());
  StructTypes.push_back(TypeEntry.get());
  TypeId = addType(std::move(TypeEntry), CTy);

  // Visit all struct members
  for (const auto *Element : Elements) {
    const auto *MemberType = cast<DIDerivedType>(Element);
    visitTypeEntry(MemberType->getBaseType());
  }
}

/// Read file contents from the actual file or from the source
std::string BTFDebug::populateFileContent(const DISubprogram *SP) {
  auto File = SP->getFile();
  std::string FileName;

  if (!File->getFilename().startswith("/") && File->getDirectory().size())
    FileName = File->getDirectory().str() + "/" + File->getFilename().str();
  else
    FileName = File->getFilename();

  // No need to populate the contends if it has been populated!
  if (FileContent.find(FileName) != FileContent.end())
    return FileName;

  std::vector<std::string> Content;
  std::string Line;
  Content.push_back(Line); // Line 0 for empty string

  std::unique_ptr<MemoryBuffer> Buf;
  auto Source = File->getSource();
  if (Source)
    Buf = MemoryBuffer::getMemBufferCopy(*Source);
  else if (ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
               MemoryBuffer::getFile(FileName))
    Buf = std::move(*BufOrErr);
  if (Buf)
    for (line_iterator I(*Buf, false), E; I != E; ++I)
      Content.push_back(*I);

  FileContent[FileName] = Content;
  return FileName;
}

void BTFDebug::constructLineInfo(const DISubprogram *SP, MCSymbol *Label,
                                 uint32_t Line, uint32_t Column) {
  std::string FileName = populateFileContent(SP);
  BTFLineInfo LineInfo;

  LineInfo.Label = Label;
  LineInfo.FileNameOff = addString(FileName);
  // If file content is not available, let LineOff = 0.
  if (Line < FileContent[FileName].size())
    LineInfo.LineOff = addString(FileContent[FileName][Line]);
  else
    LineInfo.LineOff = 0;
  LineInfo.LineNum = Line;
  LineInfo.ColumnNum = Column;
  LineInfoTable[SecNameOff].push_back(LineInfo);
}

void BTFDebug::emitCommonHeader() {
  OS.AddComment("0x" + Twine::utohexstr(BTF::MAGIC));
  OS.EmitIntValue(BTF::MAGIC, 2);
  OS.EmitIntValue(BTF::VERSION, 1);
  OS.EmitIntValue(0, 1);
}

void BTFDebug::emitBTFSection() {
  // Do not emit section if no types and only "" string.
  if (!TypeEntries.size() && StringTable.getSize() == 1)
    return;

  MCContext &Ctx = OS.getContext();
  OS.SwitchSection(Ctx.getELFSection(".BTF", ELF::SHT_PROGBITS, 0));

  // Emit header.
  emitCommonHeader();
  OS.EmitIntValue(BTF::HeaderSize, 4);

  uint32_t TypeLen = 0, StrLen;
  for (const auto &TypeEntry : TypeEntries)
    TypeLen += TypeEntry->getSize();
  StrLen = StringTable.getSize();

  OS.EmitIntValue(0, 4);
  OS.EmitIntValue(TypeLen, 4);
  OS.EmitIntValue(TypeLen, 4);
  OS.EmitIntValue(StrLen, 4);

  // Emit type table.
  for (const auto &TypeEntry : TypeEntries)
    TypeEntry->emitType(OS);

  // Emit string table.
  uint32_t StringOffset = 0;
  for (const auto &S : StringTable.getTable()) {
    OS.AddComment("string offset=" + std::to_string(StringOffset));
    OS.EmitBytes(S);
    OS.EmitBytes(StringRef("\0", 1));
    StringOffset += S.size() + 1;
  }
}

void BTFDebug::emitBTFExtSection() {
  // Do not emit section if empty FuncInfoTable and LineInfoTable.
  if (!FuncInfoTable.size() && !LineInfoTable.size() &&
      !OffsetRelocTable.size() && !ExternRelocTable.size())
    return;

  MCContext &Ctx = OS.getContext();
  OS.SwitchSection(Ctx.getELFSection(".BTF.ext", ELF::SHT_PROGBITS, 0));

  // Emit header.
  emitCommonHeader();
  OS.EmitIntValue(BTF::ExtHeaderSize, 4);

  // Account for FuncInfo/LineInfo record size as well.
  uint32_t FuncLen = 4, LineLen = 4;
  // Do not account for optional OffsetReloc/ExternReloc.
  uint32_t OffsetRelocLen = 0, ExternRelocLen = 0;
  for (const auto &FuncSec : FuncInfoTable) {
    FuncLen += BTF::SecFuncInfoSize;
    FuncLen += FuncSec.second.size() * BTF::BPFFuncInfoSize;
  }
  for (const auto &LineSec : LineInfoTable) {
    LineLen += BTF::SecLineInfoSize;
    LineLen += LineSec.second.size() * BTF::BPFLineInfoSize;
  }
  for (const auto &OffsetRelocSec : OffsetRelocTable) {
    OffsetRelocLen += BTF::SecOffsetRelocSize;
    OffsetRelocLen += OffsetRelocSec.second.size() * BTF::BPFOffsetRelocSize;
  }
  for (const auto &ExternRelocSec : ExternRelocTable) {
    ExternRelocLen += BTF::SecExternRelocSize;
    ExternRelocLen += ExternRelocSec.second.size() * BTF::BPFExternRelocSize;
  }

  if (OffsetRelocLen)
    OffsetRelocLen += 4;
  if (ExternRelocLen)
    ExternRelocLen += 4;

  OS.EmitIntValue(0, 4);
  OS.EmitIntValue(FuncLen, 4);
  OS.EmitIntValue(FuncLen, 4);
  OS.EmitIntValue(LineLen, 4);
  OS.EmitIntValue(FuncLen + LineLen, 4);
  OS.EmitIntValue(OffsetRelocLen, 4);
  OS.EmitIntValue(FuncLen + LineLen + OffsetRelocLen, 4);
  OS.EmitIntValue(ExternRelocLen, 4);

  // Emit func_info table.
  OS.AddComment("FuncInfo");
  OS.EmitIntValue(BTF::BPFFuncInfoSize, 4);
  for (const auto &FuncSec : FuncInfoTable) {
    OS.AddComment("FuncInfo section string offset=" +
                  std::to_string(FuncSec.first));
    OS.EmitIntValue(FuncSec.first, 4);
    OS.EmitIntValue(FuncSec.second.size(), 4);
    for (const auto &FuncInfo : FuncSec.second) {
      Asm->EmitLabelReference(FuncInfo.Label, 4);
      OS.EmitIntValue(FuncInfo.TypeId, 4);
    }
  }

  // Emit line_info table.
  OS.AddComment("LineInfo");
  OS.EmitIntValue(BTF::BPFLineInfoSize, 4);
  for (const auto &LineSec : LineInfoTable) {
    OS.AddComment("LineInfo section string offset=" +
                  std::to_string(LineSec.first));
    OS.EmitIntValue(LineSec.first, 4);
    OS.EmitIntValue(LineSec.second.size(), 4);
    for (const auto &LineInfo : LineSec.second) {
      Asm->EmitLabelReference(LineInfo.Label, 4);
      OS.EmitIntValue(LineInfo.FileNameOff, 4);
      OS.EmitIntValue(LineInfo.LineOff, 4);
      OS.AddComment("Line " + std::to_string(LineInfo.LineNum) + " Col " +
                    std::to_string(LineInfo.ColumnNum));
      OS.EmitIntValue(LineInfo.LineNum << 10 | LineInfo.ColumnNum, 4);
    }
  }

  // Emit offset reloc table.
  if (OffsetRelocLen) {
    OS.AddComment("OffsetReloc");
    OS.EmitIntValue(BTF::BPFOffsetRelocSize, 4);
    for (const auto &OffsetRelocSec : OffsetRelocTable) {
      OS.AddComment("Offset reloc section string offset=" +
                    std::to_string(OffsetRelocSec.first));
      OS.EmitIntValue(OffsetRelocSec.first, 4);
      OS.EmitIntValue(OffsetRelocSec.second.size(), 4);
      for (const auto &OffsetRelocInfo : OffsetRelocSec.second) {
        Asm->EmitLabelReference(OffsetRelocInfo.Label, 4);
        OS.EmitIntValue(OffsetRelocInfo.TypeID, 4);
        OS.EmitIntValue(OffsetRelocInfo.OffsetNameOff, 4);
      }
    }
  }

  // Emit extern reloc table.
  if (ExternRelocLen) {
    OS.AddComment("ExternReloc");
    OS.EmitIntValue(BTF::BPFExternRelocSize, 4);
    for (const auto &ExternRelocSec : ExternRelocTable) {
      OS.AddComment("Extern reloc section string offset=" +
                    std::to_string(ExternRelocSec.first));
      OS.EmitIntValue(ExternRelocSec.first, 4);
      OS.EmitIntValue(ExternRelocSec.second.size(), 4);
      for (const auto &ExternRelocInfo : ExternRelocSec.second) {
        Asm->EmitLabelReference(ExternRelocInfo.Label, 4);
        OS.EmitIntValue(ExternRelocInfo.ExternNameOff, 4);
      }
    }
  }
}

void BTFDebug::beginFunctionImpl(const MachineFunction *MF) {
  auto *SP = MF->getFunction().getSubprogram();
  auto *Unit = SP->getUnit();

  if (Unit->getEmissionKind() == DICompileUnit::NoDebug) {
    SkipInstruction = true;
    return;
  }
  SkipInstruction = false;

  // Collect MapDef types. Map definition needs to collect
  // pointee types. Do it first. Otherwise, for the following
  // case:
  //    struct m { ...};
  //    struct t {
  //      struct m *key;
  //    };
  //    foo(struct t *arg);
  //
  //    struct mapdef {
  //      ...
  //      struct m *key;
  //      ...
  //    } __attribute__((section(".maps"))) hash_map;
  //
  // If subroutine foo is traversed first, a type chain
  // "ptr->struct m(fwd)" will be created and later on
  // when traversing mapdef, since "ptr->struct m" exists,
  // the traversal of "struct m" will be omitted.
  if (MapDefNotCollected) {
    processGlobals(true);
    MapDefNotCollected = false;
  }

  // Collect all types locally referenced in this function.
  // Use RetainedNodes so we can collect all argument names
  // even if the argument is not used.
  std::unordered_map<uint32_t, StringRef> FuncArgNames;
  for (const DINode *DN : SP->getRetainedNodes()) {
    if (const auto *DV = dyn_cast<DILocalVariable>(DN)) {
      // Collect function arguments for subprogram func type.
      uint32_t Arg = DV->getArg();
      if (Arg) {
        visitTypeEntry(DV->getType());
        FuncArgNames[Arg] = DV->getName();
      }
    }
  }

  // Construct subprogram func proto type.
  uint32_t ProtoTypeId;
  visitSubroutineType(SP->getType(), true, FuncArgNames, ProtoTypeId);

  // Construct subprogram func type
  auto FuncTypeEntry =
      llvm::make_unique<BTFTypeFunc>(SP->getName(), ProtoTypeId);
  uint32_t FuncTypeId = addType(std::move(FuncTypeEntry));

  for (const auto &TypeEntry : TypeEntries)
    TypeEntry->completeType(*this);

  // Construct funcinfo and the first lineinfo for the function.
  MCSymbol *FuncLabel = Asm->getFunctionBegin();
  BTFFuncInfo FuncInfo;
  FuncInfo.Label = FuncLabel;
  FuncInfo.TypeId = FuncTypeId;
  if (FuncLabel->isInSection()) {
    MCSection &Section = FuncLabel->getSection();
    const MCSectionELF *SectionELF = dyn_cast<MCSectionELF>(&Section);
    assert(SectionELF && "Null section for Function Label");
    SecNameOff = addString(SectionELF->getSectionName());
  } else {
    SecNameOff = addString(".text");
  }
  FuncInfoTable[SecNameOff].push_back(FuncInfo);
}

void BTFDebug::endFunctionImpl(const MachineFunction *MF) {
  SkipInstruction = false;
  LineInfoGenerated = false;
  SecNameOff = 0;
}

/// On-demand populate struct types as requested from abstract member
/// accessing.
unsigned BTFDebug::populateStructType(const DIType *Ty) {
  unsigned Id;
  visitTypeEntry(Ty, Id, false, false);
  for (const auto &TypeEntry : TypeEntries)
    TypeEntry->completeType(*this);
  return Id;
}

// Find struct/array debuginfo types given a type id.
void BTFDebug::setTypeFromId(uint32_t TypeId, BTFTypeStruct **PrevStructType,
                             BTFTypeArray **PrevArrayType) {
  for (const auto &StructType : StructTypes) {
    if (StructType->getId() == TypeId) {
      *PrevStructType = StructType;
      return;
    }
  }
  for (const auto &ArrayType : ArrayTypes) {
    if (ArrayType->getId() == TypeId) {
      *PrevArrayType = ArrayType;
      return;
    }
  }
}

/// Generate a struct member offset relocation.
void BTFDebug::generateOffsetReloc(const MachineInstr *MI,
                                   const MCSymbol *ORSym, DIType *RootTy,
                                   StringRef AccessPattern) {
  BTFTypeStruct *PrevStructType = nullptr;
  BTFTypeArray *PrevArrayType = nullptr;
  unsigned RootId = populateStructType(RootTy);
  setTypeFromId(RootId, &PrevStructType, &PrevArrayType);
  unsigned RootTySize = PrevStructType->getStructSize();
  StringRef IndexPattern = AccessPattern.substr(AccessPattern.find_first_of(':') + 1);

  BTFOffsetReloc OffsetReloc;
  OffsetReloc.Label = ORSym;
  OffsetReloc.OffsetNameOff = addString(IndexPattern.drop_back());
  OffsetReloc.TypeID = RootId;

  uint32_t Start = 0, End = 0, Offset = 0;
  bool FirstAccess = true;
  for (auto C : IndexPattern) {
    if (C != ':') {
      End++;
    } else {
      std::string SubStr = IndexPattern.substr(Start, End - Start);
      int Loc = std::stoi(SubStr);

      if (FirstAccess) {
        Offset = Loc * RootTySize;
        FirstAccess = false;
      } else if (PrevStructType) {
        uint32_t MemberOffset, MemberTypeId;
        PrevStructType->getMemberInfo(Loc, MemberOffset, MemberTypeId);

        Offset += MemberOffset >> 3;
        PrevStructType = nullptr;
        setTypeFromId(MemberTypeId, &PrevStructType, &PrevArrayType);
      } else if (PrevArrayType) {
        uint32_t LocOffset, ElementTypeId;
        PrevArrayType->getLocInfo(Loc, LocOffset, ElementTypeId);

        Offset += LocOffset;
        PrevArrayType = nullptr;
        setTypeFromId(ElementTypeId, &PrevStructType, &PrevArrayType);
      } else {
        llvm_unreachable("Internal Error: BTF offset relocation type traversal error");
      }

      Start = End + 1;
      End = Start;
    }
  }
  AccessOffsets[AccessPattern.str()] = Offset;
  OffsetRelocTable[SecNameOff].push_back(OffsetReloc);
}

void BTFDebug::processLDimm64(const MachineInstr *MI) {
  // If the insn is an LD_imm64, the following two cases
  // will generate an .BTF.ext record.
  //
  // If the insn is "r2 = LD_imm64 @__BTF_...",
  // add this insn into the .BTF.ext OffsetReloc subsection.
  // Relocation looks like:
  //  . SecName:
  //    . InstOffset
  //    . TypeID
  //    . OffSetNameOff
  // Later, the insn is replaced with "r2 = <offset>"
  // where "<offset>" equals to the offset based on current
  // type definitions.
  //
  // If the insn is "r2 = LD_imm64 @VAR" and VAR is
  // a patchable external global, add this insn into the .BTF.ext
  // ExternReloc subsection.
  // Relocation looks like:
  //  . SecName:
  //    . InstOffset
  //    . ExternNameOff
  // Later, the insn is replaced with "r2 = <value>" or
  // "LD_imm64 r2, <value>" where "<value>" = 0.

  // check whether this is a candidate or not
  const MachineOperand &MO = MI->getOperand(1);
  if (MO.isGlobal()) {
    const GlobalValue *GVal = MO.getGlobal();
    auto *GVar = dyn_cast<GlobalVariable>(GVal);
    if (GVar && GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr)) {
      MCSymbol *ORSym = OS.getContext().createTempSymbol();
      OS.EmitLabel(ORSym);

      MDNode *MDN = GVar->getMetadata(LLVMContext::MD_preserve_access_index);
      DIType *Ty = dyn_cast<DIType>(MDN);
      generateOffsetReloc(MI, ORSym, Ty, GVar->getName());
    } else if (GVar && !GVar->hasInitializer() && GVar->hasExternalLinkage() &&
               GVar->getSection() == BPFCoreSharedInfo::PatchableExtSecName) {
      MCSymbol *ORSym = OS.getContext().createTempSymbol();
      OS.EmitLabel(ORSym);

      BTFExternReloc ExternReloc;
      ExternReloc.Label = ORSym;
      ExternReloc.ExternNameOff = addString(GVar->getName());
      ExternRelocTable[SecNameOff].push_back(ExternReloc);
    }
  }
}

void BTFDebug::beginInstruction(const MachineInstr *MI) {
  DebugHandlerBase::beginInstruction(MI);

  if (SkipInstruction || MI->isMetaInstruction() ||
      MI->getFlag(MachineInstr::FrameSetup))
    return;

  if (MI->isInlineAsm()) {
    // Count the number of register definitions to find the asm string.
    unsigned NumDefs = 0;
    for (; MI->getOperand(NumDefs).isReg() && MI->getOperand(NumDefs).isDef();
         ++NumDefs)
      ;

    // Skip this inline asm instruction if the asmstr is empty.
    const char *AsmStr = MI->getOperand(NumDefs).getSymbolName();
    if (AsmStr[0] == 0)
      return;
  }

  if (MI->getOpcode() == BPF::LD_imm64)
    processLDimm64(MI);

  // Skip this instruction if no DebugLoc or the DebugLoc
  // is the same as the previous instruction.
  const DebugLoc &DL = MI->getDebugLoc();
  if (!DL || PrevInstLoc == DL) {
    // This instruction will be skipped, no LineInfo has
    // been generated, construct one based on function signature.
    if (LineInfoGenerated == false) {
      auto *S = MI->getMF()->getFunction().getSubprogram();
      MCSymbol *FuncLabel = Asm->getFunctionBegin();
      constructLineInfo(S, FuncLabel, S->getLine(), 0);
      LineInfoGenerated = true;
    }

    return;
  }

  // Create a temporary label to remember the insn for lineinfo.
  MCSymbol *LineSym = OS.getContext().createTempSymbol();
  OS.EmitLabel(LineSym);

  // Construct the lineinfo.
  auto SP = DL.get()->getScope()->getSubprogram();
  constructLineInfo(SP, LineSym, DL.getLine(), DL.getCol());

  LineInfoGenerated = true;
  PrevInstLoc = DL;
}

void BTFDebug::processGlobals(bool ProcessingMapDef) {
  // Collect all types referenced by globals.
  const Module *M = MMI->getModule();
  for (const GlobalVariable &Global : M->globals()) {
    // Ignore external globals for now.
    if (!Global.hasInitializer() && Global.hasExternalLinkage())
      continue;

    // Decide the section name.
    StringRef SecName;
    if (Global.hasSection()) {
      SecName = Global.getSection();
    } else {
      // data, bss, or readonly sections
      if (Global.isConstant())
        SecName = ".rodata";
      else
        SecName = Global.getInitializer()->isZeroValue() ? ".bss" : ".data";
    }

    if (ProcessingMapDef != SecName.startswith(".maps"))
      continue;

    SmallVector<DIGlobalVariableExpression *, 1> GVs;
    Global.getDebugInfo(GVs);
    uint32_t GVTypeId = 0;
    for (auto *GVE : GVs) {
      if (SecName.startswith(".maps"))
        visitMapDefType(GVE->getVariable()->getType(), GVTypeId);
      else
        visitTypeEntry(GVE->getVariable()->getType(), GVTypeId, false, false);
      break;
    }

    // Only support the following globals:
    //  . static variables
    //  . non-static global variables with section attributes
    // Essentially means:
    //  . .bcc/.data/.rodata DataSec entities only contain static data
    //  . Other DataSec entities contain static or initialized global data.
    //    Initialized global data are mostly used for finding map key/value type
    //    id's. Whether DataSec is readonly or not can be found from
    //    corresponding ELF section flags.
    auto Linkage = Global.getLinkage();
    if (Linkage != GlobalValue::InternalLinkage &&
        (Linkage != GlobalValue::ExternalLinkage || !Global.hasSection()))
      continue;

    uint32_t GVarInfo = Linkage == GlobalValue::ExternalLinkage
                            ? BTF::VAR_GLOBAL_ALLOCATED
                            : BTF::VAR_STATIC;
    auto VarEntry =
        llvm::make_unique<BTFKindVar>(Global.getName(), GVTypeId, GVarInfo);
    uint32_t VarId = addType(std::move(VarEntry));

    // Find or create a DataSec
    if (DataSecEntries.find(SecName) == DataSecEntries.end()) {
      DataSecEntries[SecName] = llvm::make_unique<BTFKindDataSec>(Asm, SecName);
    }

    // Calculate symbol size
    const DataLayout &DL = Global.getParent()->getDataLayout();
    uint32_t Size = DL.getTypeAllocSize(Global.getType()->getElementType());

    DataSecEntries[SecName]->addVar(VarId, Asm->getSymbol(&Global), Size);
  }
}

/// Emit proper patchable instructions.
bool BTFDebug::InstLower(const MachineInstr *MI, MCInst &OutMI) {
  if (MI->getOpcode() == BPF::LD_imm64) {
    const MachineOperand &MO = MI->getOperand(1);
    if (MO.isGlobal()) {
      const GlobalValue *GVal = MO.getGlobal();
      auto *GVar = dyn_cast<GlobalVariable>(GVal);
      if (GVar && GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr)) {
        MDNode *MDN = GVar->getMetadata(LLVMContext::MD_preserve_access_index);
        DIType *Ty = dyn_cast<DIType>(MDN);
        std::string TypeName = Ty->getName();
        int64_t Imm = AccessOffsets[GVar->getName().str()];

        // Emit "mov ri, <imm>" for abstract member accesses.
        OutMI.setOpcode(BPF::MOV_ri);
        OutMI.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
        OutMI.addOperand(MCOperand::createImm(Imm));
        return true;
      } else if (GVar && !GVar->hasInitializer() &&
                 GVar->hasExternalLinkage() &&
                 GVar->getSection() == BPFCoreSharedInfo::PatchableExtSecName) {
        const IntegerType *IntTy = dyn_cast<IntegerType>(GVar->getValueType());
        assert(IntTy);
        // For patchable externals, emit "LD_imm64, ri, 0" if the external
        // variable is 64bit width, emit "mov ri, 0" otherwise.
        if (IntTy->getBitWidth() == 64)
          OutMI.setOpcode(BPF::LD_imm64);
        else
          OutMI.setOpcode(BPF::MOV_ri);
        OutMI.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
        OutMI.addOperand(MCOperand::createImm(0));
        return true;
      }
    }
  }
  return false;
}

void BTFDebug::endModule() {
  // Collect MapDef globals if not collected yet.
  if (MapDefNotCollected) {
    processGlobals(true);
    MapDefNotCollected = false;
  }

  // Collect global types/variables except MapDef globals.
  processGlobals(false);
  for (auto &DataSec : DataSecEntries)
    addType(std::move(DataSec.second));

  // Fixups
  for (auto &Fixup : FixupDerivedTypes) {
    StringRef TypeName = Fixup.first;
    bool IsUnion = Fixup.second.first;

    // Search through struct types
    uint32_t StructTypeId = 0;
    for (const auto &StructType : StructTypes) {
      if (StructType->getName() == TypeName) {
        StructTypeId = StructType->getId();
        break;
      }
    }

    if (StructTypeId == 0) {
      auto FwdTypeEntry = llvm::make_unique<BTFTypeFwd>(TypeName, IsUnion);
      StructTypeId = addType(std::move(FwdTypeEntry));
    }

    for (auto &DType : Fixup.second.second) {
      DType->setPointeeType(StructTypeId);
    }
  }

  // Complete BTF type cross refereences.
  for (const auto &TypeEntry : TypeEntries)
    TypeEntry->completeType(*this);

  // Emit BTF sections.
  emitBTFSection();
  emitBTFExtSection();
}
