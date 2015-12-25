//===- TypeRecord.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPERECORD_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPERECORD_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include <cinttypes>

namespace llvm {
namespace codeview {

class TypeRecord {
protected:
  explicit TypeRecord(TypeRecordKind Kind) : Kind(Kind) {}

public:
  TypeRecordKind getKind() const { return Kind; }

private:
  TypeRecordKind Kind;
};

class ModifierRecord : public TypeRecord {
public:
  ModifierRecord(TypeIndex ModifiedType, ModifierOptions Options)
      : TypeRecord(TypeRecordKind::Modifier), ModifiedType(ModifiedType),
        Options(Options) {}

  TypeIndex getModifiedType() const { return ModifiedType; }
  ModifierOptions getOptions() const { return Options; }

private:
  TypeIndex ModifiedType;
  ModifierOptions Options;
};

class ProcedureRecord : public TypeRecord {
public:
  ProcedureRecord(TypeIndex ReturnType, CallingConvention CallConv,
                  FunctionOptions Options, uint16_t ParameterCount,
                  TypeIndex ArgumentList)
      : TypeRecord(TypeRecordKind::Procedure), ReturnType(ReturnType),
        CallConv(CallConv), Options(Options), ParameterCount(ParameterCount),
        ArgumentList(ArgumentList) {}

  TypeIndex getReturnType() const { return ReturnType; }
  CallingConvention getCallConv() const { return CallConv; }
  FunctionOptions getOptions() const { return Options; }
  uint16_t getParameterCount() const { return ParameterCount; }
  TypeIndex getArgumentList() const { return ArgumentList; }

private:
  TypeIndex ReturnType;
  CallingConvention CallConv;
  FunctionOptions Options;
  uint16_t ParameterCount;
  TypeIndex ArgumentList;
};

class MemberFunctionRecord : public TypeRecord {
public:
  MemberFunctionRecord(TypeIndex ReturnType, TypeIndex ClassType,
                       TypeIndex ThisType, CallingConvention CallConv,
                       FunctionOptions Options, uint16_t ParameterCount,
                       TypeIndex ArgumentList, int32_t ThisPointerAdjustment)
      : TypeRecord(TypeRecordKind::MemberFunction), ReturnType(ReturnType),
        ClassType(ClassType), ThisType(ThisType), CallConv(CallConv),
        Options(Options), ParameterCount(ParameterCount),
        ArgumentList(ArgumentList),
        ThisPointerAdjustment(ThisPointerAdjustment) {}

  TypeIndex getReturnType() const { return ReturnType; }
  TypeIndex getClassType() const { return ClassType; }
  TypeIndex getThisType() const { return ThisType; }
  CallingConvention getCallConv() const { return CallConv; }
  FunctionOptions getOptions() const { return Options; }
  uint16_t getParameterCount() const { return ParameterCount; }
  TypeIndex getArgumentList() const { return ArgumentList; }
  int32_t getThisPointerAdjustment() const { return ThisPointerAdjustment; }

private:
  TypeIndex ReturnType;
  TypeIndex ClassType;
  TypeIndex ThisType;
  CallingConvention CallConv;
  FunctionOptions Options;
  uint16_t ParameterCount;
  TypeIndex ArgumentList;
  int32_t ThisPointerAdjustment;
};

class ArgumentListRecord : public TypeRecord {
public:
  explicit ArgumentListRecord(llvm::ArrayRef<TypeIndex> ArgumentTypes)
      : TypeRecord(TypeRecordKind::ArgumentList), ArgumentTypes(ArgumentTypes) {
  }

  llvm::ArrayRef<TypeIndex> getArgumentTypes() const { return ArgumentTypes; }

private:
  llvm::ArrayRef<TypeIndex> ArgumentTypes;
};

class PointerRecordBase : public TypeRecord {
public:
  PointerRecordBase(TypeIndex ReferentType, PointerKind Kind, PointerMode Mode,
                    PointerOptions Options, uint8_t Size)
      : TypeRecord(TypeRecordKind::Pointer), ReferentType(ReferentType),
        PtrKind(Kind), Mode(Mode), Options(Options), Size(Size) {}

  TypeIndex getReferentType() const { return ReferentType; }
  PointerKind getPointerKind() const { return PtrKind; }
  PointerMode getMode() const { return Mode; }
  PointerOptions getOptions() const { return Options; }
  uint8_t getSize() const { return Size; }

private:
  TypeIndex ReferentType;
  PointerKind PtrKind;
  PointerMode Mode;
  PointerOptions Options;
  uint8_t Size;
};

class PointerRecord : public PointerRecordBase {
public:
  PointerRecord(TypeIndex ReferentType, PointerKind Kind, PointerMode Mode,
                PointerOptions Options, uint8_t Size)
      : PointerRecordBase(ReferentType, Kind, Mode, Options, Size) {}
};

class PointerToMemberRecord : public PointerRecordBase {
public:
  PointerToMemberRecord(TypeIndex ReferentType, PointerKind Kind,
                        PointerMode Mode, PointerOptions Options, uint8_t Size,
                        TypeIndex ContainingType,
                        PointerToMemberRepresentation Representation)
      : PointerRecordBase(ReferentType, Kind, Mode, Options, Size),
        ContainingType(ContainingType), Representation(Representation) {}

  TypeIndex getContainingType() const { return ContainingType; }
  PointerToMemberRepresentation getRepresentation() const {
    return Representation;
  }

private:
  TypeIndex ContainingType;
  PointerToMemberRepresentation Representation;
};

class ArrayRecord : public TypeRecord {
public:
  ArrayRecord(TypeIndex ElementType, TypeIndex IndexType, uint64_t Size,
              llvm::StringRef Name)
      : TypeRecord(TypeRecordKind::Array), ElementType(ElementType),
        IndexType(IndexType), Size(Size), Name(Name) {}

  TypeIndex getElementType() const { return ElementType; }
  TypeIndex getIndexType() const { return IndexType; }
  uint64_t getSize() const { return Size; }
  llvm::StringRef getName() const { return Name; }

private:
  TypeIndex ElementType;
  TypeIndex IndexType;
  uint64_t Size;
  llvm::StringRef Name;
};

class TagRecord : public TypeRecord {
protected:
  TagRecord(TypeRecordKind Kind, uint16_t MemberCount, ClassOptions Options,
            TypeIndex FieldList, StringRef Name, StringRef UniqueName)
      : TypeRecord(Kind), MemberCount(MemberCount), Options(Options),
        FieldList(FieldList), Name(Name), UniqueName(UniqueName) {}

public:
  uint16_t getMemberCount() const { return MemberCount; }
  ClassOptions getOptions() const { return Options; }
  TypeIndex getFieldList() const { return FieldList; }
  StringRef getName() const { return Name; }
  StringRef getUniqueName() const { return UniqueName; }

private:
  uint16_t MemberCount;
  ClassOptions Options;
  TypeIndex FieldList;
  StringRef Name;
  StringRef UniqueName;
};

class AggregateRecord : public TagRecord {
public:
  AggregateRecord(TypeRecordKind Kind, uint16_t MemberCount,
                  ClassOptions Options, HfaKind Hfa,
                  WindowsRTClassKind WinRTKind, TypeIndex FieldList,
                  TypeIndex DerivationList, TypeIndex VTableShape,
                  uint64_t Size, StringRef Name, StringRef UniqueName)
      : TagRecord(Kind, MemberCount, Options, FieldList, Name, UniqueName),
        Hfa(Hfa), WinRTKind(WinRTKind), DerivationList(DerivationList),
        VTableShape(VTableShape), Size(Size) {}

  HfaKind getHfa() const { return Hfa; }
  WindowsRTClassKind getWinRTKind() const { return WinRTKind; }
  TypeIndex getDerivationList() const { return DerivationList; }
  TypeIndex getVTableShape() const { return VTableShape; }
  uint64_t getSize() const { return Size; }

private:
  HfaKind Hfa;
  WindowsRTClassKind WinRTKind;
  TypeIndex DerivationList;
  TypeIndex VTableShape;
  uint64_t Size;
};

class EnumRecord : public TagRecord {
public:
  EnumRecord(uint16_t MemberCount, ClassOptions Options, TypeIndex FieldList,
             StringRef Name, StringRef UniqueName, TypeIndex UnderlyingType)
      : TagRecord(TypeRecordKind::Enum, MemberCount, Options, FieldList, Name,
                  UniqueName),
        UnderlyingType(UnderlyingType) {}

  TypeIndex getUnderlyingType() const { return UnderlyingType; }

private:
  TypeIndex UnderlyingType;
};

class BitFieldRecord : TypeRecord {
public:
  BitFieldRecord(TypeIndex Type, uint8_t BitSize, uint8_t BitOffset)
      : TypeRecord(TypeRecordKind::BitField), Type(Type), BitSize(BitSize),
        BitOffset(BitOffset) {}

  TypeIndex getType() const { return Type; }
  uint8_t getBitOffset() const { return BitOffset; }
  uint8_t getBitSize() const { return BitSize; }

private:
  TypeIndex Type;
  uint8_t BitSize;
  uint8_t BitOffset;
};

class VirtualTableShapeRecord : TypeRecord {
public:
  explicit VirtualTableShapeRecord(ArrayRef<VirtualTableSlotKind> Slots)
      : TypeRecord(TypeRecordKind::VirtualTableShape), Slots(Slots) {}

  ArrayRef<VirtualTableSlotKind> getSlots() const { return Slots; }

private:
  ArrayRef<VirtualTableSlotKind> Slots;
};
}
}

#endif
