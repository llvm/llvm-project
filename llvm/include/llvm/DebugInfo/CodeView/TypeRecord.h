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

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/Support/Endian.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace llvm {

namespace msf {
class StreamReader;
} // end namespace msf

namespace codeview {

using support::little32_t;
using support::ulittle16_t;
using support::ulittle32_t;

typedef CVRecord<TypeLeafKind> CVType;

struct CVMemberRecord {
  TypeLeafKind Kind;
  ArrayRef<uint8_t> Data;
};
typedef msf::VarStreamArray<CVType> CVTypeArray;
typedef iterator_range<CVTypeArray::Iterator> CVTypeRange;

/// Equvalent to CV_fldattr_t in cvinfo.h.
struct MemberAttributes {
  uint16_t Attrs = 0;
  enum {
    MethodKindShift = 2,
  };
  MemberAttributes() = default;

  explicit MemberAttributes(MemberAccess Access)
      : Attrs(static_cast<uint16_t>(Access)) {}

  MemberAttributes(MemberAccess Access, MethodKind Kind, MethodOptions Flags) {
    Attrs = static_cast<uint16_t>(Access);
    Attrs |= (static_cast<uint16_t>(Kind) << MethodKindShift);
    Attrs |= static_cast<uint16_t>(Flags);
  }

  /// Get the access specifier. Valid for any kind of member.
  MemberAccess getAccess() const {
    return MemberAccess(unsigned(Attrs) & unsigned(MethodOptions::AccessMask));
  }

  /// Indicates if a method is defined with friend, virtual, static, etc.
  MethodKind getMethodKind() const {
    return MethodKind(
        (unsigned(Attrs) & unsigned(MethodOptions::MethodKindMask)) >>
        MethodKindShift);
  }

  /// Get the flags that are not included in access control or method
  /// properties.
  MethodOptions getFlags() const {
    return MethodOptions(
        unsigned(Attrs) &
        ~unsigned(MethodOptions::AccessMask | MethodOptions::MethodKindMask));
  }

  /// Is this method virtual.
  bool isVirtual() const {
    auto MP = getMethodKind();
    return MP != MethodKind::Vanilla && MP != MethodKind::Friend &&
           MP != MethodKind::Static;
  }

  /// Does this member introduce a new virtual method.
  bool isIntroducedVirtual() const {
    auto MP = getMethodKind();
    return MP == MethodKind::IntroducingVirtual ||
           MP == MethodKind::PureIntroducingVirtual;
  }
};

// Does not correspond to any tag, this is the tail of an LF_POINTER record
// if it represents a member pointer.
class MemberPointerInfo {
public:
  MemberPointerInfo() = default;

  MemberPointerInfo(TypeIndex ContainingType,
                    PointerToMemberRepresentation Representation)
      : ContainingType(ContainingType), Representation(Representation) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getContainingType() const { return ContainingType; }
  PointerToMemberRepresentation getRepresentation() const {
    return Representation;
  }

  TypeIndex ContainingType;
  PointerToMemberRepresentation Representation;
};

class TypeRecord {
protected:
  TypeRecord() = default;
  explicit TypeRecord(TypeRecordKind Kind) : Kind(Kind) {}

public:
  TypeRecordKind getKind() const { return Kind; }

private:
  TypeRecordKind Kind;
};

// LF_MODIFIER
class ModifierRecord : public TypeRecord {
public:
  explicit ModifierRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  ModifierRecord(TypeIndex ModifiedType, ModifierOptions Modifiers)
      : TypeRecord(TypeRecordKind::Modifier), ModifiedType(ModifiedType),
        Modifiers(Modifiers) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getModifiedType() const { return ModifiedType; }
  ModifierOptions getModifiers() const { return Modifiers; }

  TypeIndex ModifiedType;
  ModifierOptions Modifiers;
};

// LF_PROCEDURE
class ProcedureRecord : public TypeRecord {
public:
  explicit ProcedureRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  ProcedureRecord(TypeIndex ReturnType, CallingConvention CallConv,
                  FunctionOptions Options, uint16_t ParameterCount,
                  TypeIndex ArgumentList)
      : TypeRecord(TypeRecordKind::Procedure), ReturnType(ReturnType),
        CallConv(CallConv), Options(Options), ParameterCount(ParameterCount),
        ArgumentList(ArgumentList) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getReturnType() const { return ReturnType; }
  CallingConvention getCallConv() const { return CallConv; }
  FunctionOptions getOptions() const { return Options; }
  uint16_t getParameterCount() const { return ParameterCount; }
  TypeIndex getArgumentList() const { return ArgumentList; }

  TypeIndex ReturnType;
  CallingConvention CallConv;
  FunctionOptions Options;
  uint16_t ParameterCount;
  TypeIndex ArgumentList;
};

// LF_MFUNCTION
class MemberFunctionRecord : public TypeRecord {
public:
  explicit MemberFunctionRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}

  MemberFunctionRecord(TypeIndex ReturnType, TypeIndex ClassType,
                       TypeIndex ThisType, CallingConvention CallConv,
                       FunctionOptions Options, uint16_t ParameterCount,
                       TypeIndex ArgumentList, int32_t ThisPointerAdjustment)
      : TypeRecord(TypeRecordKind::MemberFunction), ReturnType(ReturnType),
        ClassType(ClassType), ThisType(ThisType), CallConv(CallConv),
        Options(Options), ParameterCount(ParameterCount),
        ArgumentList(ArgumentList),
        ThisPointerAdjustment(ThisPointerAdjustment) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getReturnType() const { return ReturnType; }
  TypeIndex getClassType() const { return ClassType; }
  TypeIndex getThisType() const { return ThisType; }
  CallingConvention getCallConv() const { return CallConv; }
  FunctionOptions getOptions() const { return Options; }
  uint16_t getParameterCount() const { return ParameterCount; }
  TypeIndex getArgumentList() const { return ArgumentList; }
  int32_t getThisPointerAdjustment() const { return ThisPointerAdjustment; }

  TypeIndex ReturnType;
  TypeIndex ClassType;
  TypeIndex ThisType;
  CallingConvention CallConv;
  FunctionOptions Options;
  uint16_t ParameterCount;
  TypeIndex ArgumentList;
  int32_t ThisPointerAdjustment;
};

// LF_MFUNC_ID
class MemberFuncIdRecord : public TypeRecord {
public:
  explicit MemberFuncIdRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  MemberFuncIdRecord(TypeIndex ClassType, TypeIndex FunctionType,
                         StringRef Name)
      : TypeRecord(TypeRecordKind::MemberFuncId), ClassType(ClassType),
        FunctionType(FunctionType), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getClassType() const { return ClassType; }
  TypeIndex getFunctionType() const { return FunctionType; }
  StringRef getName() const { return Name; }
  TypeIndex ClassType;
  TypeIndex FunctionType;
  StringRef Name;
};

// LF_ARGLIST, LF_SUBSTR_LIST
class ArgListRecord : public TypeRecord {
public:
  explicit ArgListRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}

  ArgListRecord(TypeRecordKind Kind, ArrayRef<TypeIndex> Indices)
      : TypeRecord(Kind), StringIndices(Indices) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  ArrayRef<TypeIndex> getIndices() const { return StringIndices; }

  std::vector<TypeIndex> StringIndices;
};

// LF_POINTER
class PointerRecord : public TypeRecord {
public:
  static const uint32_t PointerKindShift = 0;
  static const uint32_t PointerKindMask = 0x1F;

  static const uint32_t PointerModeShift = 5;
  static const uint32_t PointerModeMask = 0x07;

  static const uint32_t PointerOptionMask = 0xFF;

  static const uint32_t PointerSizeShift = 13;
  static const uint32_t PointerSizeMask = 0xFF;

  explicit PointerRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}

  PointerRecord(TypeIndex ReferentType, uint32_t Attrs)
      : TypeRecord(TypeRecordKind::Pointer), ReferentType(ReferentType),
        Attrs(Attrs) {}

  PointerRecord(TypeIndex ReferentType, PointerKind PK, PointerMode PM,
                PointerOptions PO, uint8_t Size)
      : TypeRecord(TypeRecordKind::Pointer), ReferentType(ReferentType),
        Attrs(calcAttrs(PK, PM, PO, Size)) {}

  PointerRecord(TypeIndex ReferentType, PointerKind PK, PointerMode PM,
                PointerOptions PO, uint8_t Size,
                const MemberPointerInfo &Member)
      : TypeRecord(TypeRecordKind::Pointer), ReferentType(ReferentType),
        Attrs(calcAttrs(PK, PM, PO, Size)), MemberInfo(Member) {}

  PointerRecord(TypeIndex ReferentType, uint32_t Attrs,
                const MemberPointerInfo &Member)
      : TypeRecord(TypeRecordKind::Pointer), ReferentType(ReferentType),
        Attrs(Attrs), MemberInfo(Member) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getReferentType() const { return ReferentType; }

  PointerKind getPointerKind() const {
    return static_cast<PointerKind>((Attrs >> PointerKindShift) &
                                    PointerKindMask);
  }

  PointerMode getMode() const {
    return static_cast<PointerMode>((Attrs >> PointerModeShift) &
                                    PointerModeMask);
  }

  PointerOptions getOptions() const {
    return static_cast<PointerOptions>(Attrs);
  }

  uint8_t getSize() const {
    return (Attrs >> PointerSizeShift) & PointerSizeMask;
  }

  MemberPointerInfo getMemberInfo() const { return *MemberInfo; }

  bool isPointerToMember() const {
    return getMode() == PointerMode::PointerToDataMember ||
           getMode() == PointerMode::PointerToMemberFunction;
  }

  bool isFlat() const { return !!(Attrs & uint32_t(PointerOptions::Flat32)); }
  bool isConst() const { return !!(Attrs & uint32_t(PointerOptions::Const)); }

  bool isVolatile() const {
    return !!(Attrs & uint32_t(PointerOptions::Volatile));
  }

  bool isUnaligned() const {
    return !!(Attrs & uint32_t(PointerOptions::Unaligned));
  }

  TypeIndex ReferentType;
  uint32_t Attrs;

  Optional<MemberPointerInfo> MemberInfo;

private:
  static uint32_t calcAttrs(PointerKind PK, PointerMode PM, PointerOptions PO,
                            uint8_t Size) {
    uint32_t A = 0;
    A |= static_cast<uint32_t>(PK);
    A |= static_cast<uint32_t>(PO);
    A |= (static_cast<uint32_t>(PM) << PointerModeShift);
    A |= (static_cast<uint32_t>(Size) << PointerSizeShift);
    return A;
  }
};

// LF_NESTTYPE
class NestedTypeRecord : public TypeRecord {
public:
  explicit NestedTypeRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  NestedTypeRecord(TypeIndex Type, StringRef Name)
      : TypeRecord(TypeRecordKind::NestedType), Type(Type), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getNestedType() const { return Type; }
  StringRef getName() const { return Name; }

  TypeIndex Type;
  StringRef Name;
};

// LF_FIELDLIST
class FieldListRecord : public TypeRecord {
public:
  explicit FieldListRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  explicit FieldListRecord(ArrayRef<uint8_t> Data)
      : TypeRecord(TypeRecordKind::FieldList), Data(Data) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap) { return false; }

  ArrayRef<uint8_t> Data;
};

// LF_ARRAY
class ArrayRecord : public TypeRecord {
public:
  explicit ArrayRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  ArrayRecord(TypeIndex ElementType, TypeIndex IndexType, uint64_t Size,
              StringRef Name)
      : TypeRecord(TypeRecordKind::Array), ElementType(ElementType),
        IndexType(IndexType), Size(Size), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getElementType() const { return ElementType; }
  TypeIndex getIndexType() const { return IndexType; }
  uint64_t getSize() const { return Size; }
  StringRef getName() const { return Name; }

  TypeIndex ElementType;
  TypeIndex IndexType;
  uint64_t Size;
  StringRef Name;
};

class TagRecord : public TypeRecord {
protected:
  explicit TagRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  TagRecord(TypeRecordKind Kind, uint16_t MemberCount, ClassOptions Options,
            TypeIndex FieldList, StringRef Name, StringRef UniqueName)
      : TypeRecord(Kind), MemberCount(MemberCount), Options(Options),
        FieldList(FieldList), Name(Name), UniqueName(UniqueName) {}

public:
  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static const int HfaKindShift = 11;
  static const int HfaKindMask = 0x1800;
  static const int WinRTKindShift = 14;
  static const int WinRTKindMask = 0xC000;

  bool hasUniqueName() const {
    return (Options & ClassOptions::HasUniqueName) != ClassOptions::None;
  }

  uint16_t getMemberCount() const { return MemberCount; }
  ClassOptions getOptions() const { return Options; }
  TypeIndex getFieldList() const { return FieldList; }
  StringRef getName() const { return Name; }
  StringRef getUniqueName() const { return UniqueName; }

  uint16_t MemberCount;
  ClassOptions Options;
  TypeIndex FieldList;
  StringRef Name;
  StringRef UniqueName;
};

// LF_CLASS, LF_STRUCTURE, LF_INTERFACE
class ClassRecord : public TagRecord {
public:
  explicit ClassRecord(TypeRecordKind Kind) : TagRecord(Kind) {}
  ClassRecord(TypeRecordKind Kind, uint16_t MemberCount, ClassOptions Options,
              TypeIndex FieldList, TypeIndex DerivationList,
              TypeIndex VTableShape, uint64_t Size, StringRef Name,
              StringRef UniqueName)
      : TagRecord(Kind, MemberCount, Options, FieldList, Name, UniqueName),
        DerivationList(DerivationList), VTableShape(VTableShape), Size(Size) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  HfaKind getHfa() const {
    uint16_t Value = static_cast<uint16_t>(Options);
    Value = (Value & HfaKindMask) >> HfaKindShift;
    return static_cast<HfaKind>(Value);
  }

  WindowsRTClassKind getWinRTKind() const {
    uint16_t Value = static_cast<uint16_t>(Options);
    Value = (Value & WinRTKindMask) >> WinRTKindShift;
    return static_cast<WindowsRTClassKind>(Value);
  }

  TypeIndex getDerivationList() const { return DerivationList; }
  TypeIndex getVTableShape() const { return VTableShape; }
  uint64_t getSize() const { return Size; }

  TypeIndex DerivationList;
  TypeIndex VTableShape;
  uint64_t Size;
};

// LF_UNION
struct UnionRecord : public TagRecord {
  explicit UnionRecord(TypeRecordKind Kind) : TagRecord(Kind) {}
  UnionRecord(uint16_t MemberCount, ClassOptions Options, TypeIndex FieldList,
              uint64_t Size, StringRef Name, StringRef UniqueName)
      : TagRecord(TypeRecordKind::Union, MemberCount, Options, FieldList, Name,
                  UniqueName),
        Size(Size) {}

  HfaKind getHfa() const {
    uint16_t Value = static_cast<uint16_t>(Options);
    Value = (Value & HfaKindMask) >> HfaKindShift;
    return static_cast<HfaKind>(Value);
  }

  uint64_t getSize() const { return Size; }

  uint64_t Size;
};

// LF_ENUM
class EnumRecord : public TagRecord {
public:
  explicit EnumRecord(TypeRecordKind Kind) : TagRecord(Kind) {}
  EnumRecord(uint16_t MemberCount, ClassOptions Options, TypeIndex FieldList,
             StringRef Name, StringRef UniqueName, TypeIndex UnderlyingType)
      : TagRecord(TypeRecordKind::Enum, MemberCount, Options, FieldList, Name,
                  UniqueName),
        UnderlyingType(UnderlyingType) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getUnderlyingType() const { return UnderlyingType; }
  TypeIndex UnderlyingType;
};

// LF_BITFIELD
class BitFieldRecord : public TypeRecord {
public:
  explicit BitFieldRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  BitFieldRecord(TypeIndex Type, uint8_t BitSize, uint8_t BitOffset)
      : TypeRecord(TypeRecordKind::BitField), Type(Type), BitSize(BitSize),
        BitOffset(BitOffset) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getType() const { return Type; }
  uint8_t getBitOffset() const { return BitOffset; }
  uint8_t getBitSize() const { return BitSize; }
  TypeIndex Type;
  uint8_t BitSize;
  uint8_t BitOffset;
};

// LF_VTSHAPE
class VFTableShapeRecord : public TypeRecord {
public:
  explicit VFTableShapeRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  explicit VFTableShapeRecord(ArrayRef<VFTableSlotKind> Slots)
      : TypeRecord(TypeRecordKind::VFTableShape), SlotsRef(Slots) {}
  explicit VFTableShapeRecord(std::vector<VFTableSlotKind> Slots)
      : TypeRecord(TypeRecordKind::VFTableShape), Slots(std::move(Slots)) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  ArrayRef<VFTableSlotKind> getSlots() const {
    if (!SlotsRef.empty())
      return SlotsRef;
    return Slots;
  }

  uint32_t getEntryCount() const { return getSlots().size(); }
  ArrayRef<VFTableSlotKind> SlotsRef;
  std::vector<VFTableSlotKind> Slots;
};

// LF_TYPESERVER2
class TypeServer2Record : public TypeRecord {
public:
  explicit TypeServer2Record(TypeRecordKind Kind) : TypeRecord(Kind) {}
  TypeServer2Record(StringRef Guid, uint32_t Age, StringRef Name)
      : TypeRecord(TypeRecordKind::TypeServer2), Guid(Guid), Age(Age),
        Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  StringRef getGuid() const { return Guid; }

  uint32_t getAge() const { return Age; }

  StringRef getName() const { return Name; }

  StringRef Guid;
  uint32_t Age;
  StringRef Name;
};

// LF_STRING_ID
class StringIdRecord : public TypeRecord {
public:
  explicit StringIdRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  StringIdRecord(TypeIndex Id, StringRef String)
      : TypeRecord(TypeRecordKind::StringId), Id(Id), String(String) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getId() const { return Id; }

  StringRef getString() const { return String; }
  TypeIndex Id;
  StringRef String;
};

// LF_FUNC_ID
class FuncIdRecord : public TypeRecord {
public:
  explicit FuncIdRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  FuncIdRecord(TypeIndex ParentScope, TypeIndex FunctionType, StringRef Name)
      : TypeRecord(TypeRecordKind::FuncId), ParentScope(ParentScope),
        FunctionType(FunctionType), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getParentScope() const { return ParentScope; }

  TypeIndex getFunctionType() const { return FunctionType; }

  StringRef getName() const { return Name; }

  TypeIndex ParentScope;
  TypeIndex FunctionType;
  StringRef Name;
};

// LF_UDT_SRC_LINE
class UdtSourceLineRecord : public TypeRecord {
public:
  explicit UdtSourceLineRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  UdtSourceLineRecord(TypeIndex UDT, TypeIndex SourceFile, uint32_t LineNumber)
      : TypeRecord(TypeRecordKind::UdtSourceLine), UDT(UDT),
        SourceFile(SourceFile), LineNumber(LineNumber) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getUDT() const { return UDT; }
  TypeIndex getSourceFile() const { return SourceFile; }
  uint32_t getLineNumber() const { return LineNumber; }

  TypeIndex UDT;
  TypeIndex SourceFile;
  uint32_t LineNumber;
};

// LF_UDT_MOD_SRC_LINE
class UdtModSourceLineRecord : public TypeRecord {
public:
  explicit UdtModSourceLineRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  UdtModSourceLineRecord(TypeIndex UDT, TypeIndex SourceFile,
                         uint32_t LineNumber, uint16_t Module)
      : TypeRecord(TypeRecordKind::UdtSourceLine), UDT(UDT),
        SourceFile(SourceFile), LineNumber(LineNumber), Module(Module) {}

  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getUDT() const { return UDT; }
  TypeIndex getSourceFile() const { return SourceFile; }
  uint32_t getLineNumber() const { return LineNumber; }
  uint16_t getModule() const { return Module; }

  TypeIndex UDT;
  TypeIndex SourceFile;
  uint32_t LineNumber;
  uint16_t Module;
};

// LF_BUILDINFO
class BuildInfoRecord : public TypeRecord {
public:
  explicit BuildInfoRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  BuildInfoRecord(ArrayRef<TypeIndex> ArgIndices)
      : TypeRecord(TypeRecordKind::BuildInfo),
        ArgIndices(ArgIndices.begin(), ArgIndices.end()) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  ArrayRef<TypeIndex> getArgs() const { return ArgIndices; }
  SmallVector<TypeIndex, 4> ArgIndices;
};

// LF_VFTABLE
class VFTableRecord : public TypeRecord {
public:
  explicit VFTableRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  VFTableRecord(TypeIndex CompleteClass, TypeIndex OverriddenVFTable,
                uint32_t VFPtrOffset, StringRef Name,
                ArrayRef<StringRef> Methods)
      : TypeRecord(TypeRecordKind::VFTable), CompleteClass(CompleteClass),
        OverriddenVFTable(OverriddenVFTable), VFPtrOffset(VFPtrOffset) {
    MethodNames.push_back(Name);
    MethodNames.insert(MethodNames.end(), Methods.begin(), Methods.end());
  }

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getCompleteClass() const { return CompleteClass; }
  TypeIndex getOverriddenVTable() const { return OverriddenVFTable; }
  uint32_t getVFPtrOffset() const { return VFPtrOffset; }
  StringRef getName() const { return makeArrayRef(MethodNames).front(); }
  ArrayRef<StringRef> getMethodNames() const {
    return makeArrayRef(MethodNames).drop_front();
  }

  TypeIndex CompleteClass;
  TypeIndex OverriddenVFTable;
  uint32_t VFPtrOffset;
  std::vector<StringRef> MethodNames;
};

// LF_ONEMETHOD
class OneMethodRecord : public TypeRecord {
public:
  OneMethodRecord() : TypeRecord(TypeRecordKind::OneMethod) {}
  explicit OneMethodRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  OneMethodRecord(TypeIndex Type, MemberAttributes Attrs, int32_t VFTableOffset,
                  StringRef Name)
      : TypeRecord(TypeRecordKind::OneMethod), Type(Type), Attrs(Attrs),
        VFTableOffset(VFTableOffset), Name(Name) {}
  OneMethodRecord(TypeIndex Type, MemberAccess Access, MethodKind MK,
                  MethodOptions Options, int32_t VFTableOffset, StringRef Name)
      : TypeRecord(TypeRecordKind::OneMethod), Type(Type),
        Attrs(Access, MK, Options), VFTableOffset(VFTableOffset), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getType() const { return Type; }
  MethodKind getMethodKind() const { return Attrs.getMethodKind(); }
  MethodOptions getOptions() const { return Attrs.getFlags(); }
  MemberAccess getAccess() const { return Attrs.getAccess(); }
  int32_t getVFTableOffset() const { return VFTableOffset; }
  StringRef getName() const { return Name; }

  bool isIntroducingVirtual() const {
    return getMethodKind() == MethodKind::IntroducingVirtual ||
           getMethodKind() == MethodKind::PureIntroducingVirtual;
  }

  TypeIndex Type;
  MemberAttributes Attrs;
  int32_t VFTableOffset;
  StringRef Name;
};

// LF_METHODLIST
class MethodOverloadListRecord : public TypeRecord {
public:
  explicit MethodOverloadListRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  MethodOverloadListRecord(ArrayRef<OneMethodRecord> Methods)
      : TypeRecord(TypeRecordKind::MethodOverloadList), Methods(Methods) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  ArrayRef<OneMethodRecord> getMethods() const { return Methods; }
  std::vector<OneMethodRecord> Methods;
};

/// For method overload sets.  LF_METHOD
class OverloadedMethodRecord : public TypeRecord {
public:
  explicit OverloadedMethodRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  OverloadedMethodRecord(uint16_t NumOverloads, TypeIndex MethodList,
                         StringRef Name)
      : TypeRecord(TypeRecordKind::OverloadedMethod),
        NumOverloads(NumOverloads), MethodList(MethodList), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  uint16_t getNumOverloads() const { return NumOverloads; }
  TypeIndex getMethodList() const { return MethodList; }
  StringRef getName() const { return Name; }
  uint16_t NumOverloads;
  TypeIndex MethodList;
  StringRef Name;
};

// LF_MEMBER
class DataMemberRecord : public TypeRecord {
public:
  explicit DataMemberRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  DataMemberRecord(MemberAttributes Attrs, TypeIndex Type, uint64_t Offset,
                   StringRef Name)
      : TypeRecord(TypeRecordKind::DataMember), Attrs(Attrs), Type(Type),
        FieldOffset(Offset), Name(Name) {}
  DataMemberRecord(MemberAccess Access, TypeIndex Type, uint64_t Offset,
                   StringRef Name)
      : TypeRecord(TypeRecordKind::DataMember), Attrs(Access), Type(Type),
        FieldOffset(Offset), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  MemberAccess getAccess() const { return Attrs.getAccess(); }
  TypeIndex getType() const { return Type; }
  uint64_t getFieldOffset() const { return FieldOffset; }
  StringRef getName() const { return Name; }

  MemberAttributes Attrs;
  TypeIndex Type;
  uint64_t FieldOffset;
  StringRef Name;
};

// LF_STMEMBER
class StaticDataMemberRecord : public TypeRecord {
public:
  explicit StaticDataMemberRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  StaticDataMemberRecord(MemberAttributes Attrs, TypeIndex Type, StringRef Name)
      : TypeRecord(TypeRecordKind::StaticDataMember), Attrs(Attrs), Type(Type),
        Name(Name) {}
  StaticDataMemberRecord(MemberAccess Access, TypeIndex Type, StringRef Name)
      : TypeRecord(TypeRecordKind::StaticDataMember), Attrs(Access), Type(Type),
        Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  MemberAccess getAccess() const { return Attrs.getAccess(); }
  TypeIndex getType() const { return Type; }
  StringRef getName() const { return Name; }

  MemberAttributes Attrs;
  TypeIndex Type;
  StringRef Name;
};

// LF_ENUMERATE
class EnumeratorRecord : public TypeRecord {
public:
  explicit EnumeratorRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  EnumeratorRecord(MemberAttributes Attrs, APSInt Value, StringRef Name)
      : TypeRecord(TypeRecordKind::Enumerator), Attrs(Attrs),
        Value(std::move(Value)), Name(Name) {}
  EnumeratorRecord(MemberAccess Access, APSInt Value, StringRef Name)
      : TypeRecord(TypeRecordKind::Enumerator), Attrs(Access),
        Value(std::move(Value)), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  MemberAccess getAccess() const { return Attrs.getAccess(); }
  APSInt getValue() const { return Value; }
  StringRef getName() const { return Name; }

  MemberAttributes Attrs;
  APSInt Value;
  StringRef Name;
};

// LF_VFUNCTAB
class VFPtrRecord : public TypeRecord {
public:
  explicit VFPtrRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  VFPtrRecord(TypeIndex Type)
      : TypeRecord(TypeRecordKind::VFPtr), Type(Type) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex getType() const { return Type; }

  TypeIndex Type;
};

// LF_BCLASS, LF_BINTERFACE
class BaseClassRecord : public TypeRecord {
public:
  explicit BaseClassRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  BaseClassRecord(MemberAttributes Attrs, TypeIndex Type, uint64_t Offset)
      : TypeRecord(TypeRecordKind::BaseClass), Attrs(Attrs), Type(Type),
        Offset(Offset) {}
  BaseClassRecord(MemberAccess Access, TypeIndex Type, uint64_t Offset)
      : TypeRecord(TypeRecordKind::BaseClass), Attrs(Access), Type(Type),
        Offset(Offset) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  MemberAccess getAccess() const { return Attrs.getAccess(); }
  TypeIndex getBaseType() const { return Type; }
  uint64_t getBaseOffset() const { return Offset; }

  MemberAttributes Attrs;
  TypeIndex Type;
  uint64_t Offset;
};

// LF_VBCLASS, LF_IVBCLASS
class VirtualBaseClassRecord : public TypeRecord {
public:
  explicit VirtualBaseClassRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  VirtualBaseClassRecord(TypeRecordKind Kind, MemberAttributes Attrs,
                         TypeIndex BaseType, TypeIndex VBPtrType,
                         uint64_t Offset, uint64_t Index)
      : TypeRecord(Kind), Attrs(Attrs), BaseType(BaseType),
        VBPtrType(VBPtrType), VBPtrOffset(Offset), VTableIndex(Index) {}
  VirtualBaseClassRecord(TypeRecordKind Kind, MemberAccess Access,
                         TypeIndex BaseType, TypeIndex VBPtrType,
                         uint64_t Offset, uint64_t Index)
      : TypeRecord(Kind), Attrs(Access), BaseType(BaseType),
        VBPtrType(VBPtrType), VBPtrOffset(Offset), VTableIndex(Index) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  MemberAccess getAccess() const { return Attrs.getAccess(); }
  TypeIndex getBaseType() const { return BaseType; }
  TypeIndex getVBPtrType() const { return VBPtrType; }
  uint64_t getVBPtrOffset() const { return VBPtrOffset; }
  uint64_t getVTableIndex() const { return VTableIndex; }

  MemberAttributes Attrs;
  TypeIndex BaseType;
  TypeIndex VBPtrType;
  uint64_t VBPtrOffset;
  uint64_t VTableIndex;
};

/// LF_INDEX - Used to chain two large LF_FIELDLIST or LF_METHODLIST records
/// together. The first will end in an LF_INDEX record that points to the next.
class ListContinuationRecord : public TypeRecord {
public:
  explicit ListContinuationRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  ListContinuationRecord(TypeIndex ContinuationIndex)
      : TypeRecord(TypeRecordKind::ListContinuation),
        ContinuationIndex(ContinuationIndex) {}

  TypeIndex getContinuationIndex() const { return ContinuationIndex; }

  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  TypeIndex ContinuationIndex;
};

} // end namespace codeview

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPERECORD_H
