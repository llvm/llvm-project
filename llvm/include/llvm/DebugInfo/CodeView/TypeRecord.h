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
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Error.h"
#include <cinttypes>
#include <utility>

namespace llvm {
namespace codeview {

using llvm::support::little32_t;
using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;

/// Equvalent to CV_fldattr_t in cvinfo.h.
struct MemberAttributes {
  ulittle16_t Attrs;
  enum {
    MethodKindShift = 2,
  };

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
  MemberPointerInfo() {}

  MemberPointerInfo(TypeIndex ContainingType,
                    PointerToMemberRepresentation Representation)
      : ContainingType(ContainingType), Representation(Representation) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<MemberPointerInfo> deserialize(ArrayRef<uint8_t> &Data);

  TypeIndex getContainingType() const { return ContainingType; }
  PointerToMemberRepresentation getRepresentation() const {
    return Representation;
  }

private:
  struct Layout {
    TypeIndex ClassType;
    ulittle16_t Representation; // PointerToMemberRepresentation
  };

  TypeIndex ContainingType;
  PointerToMemberRepresentation Representation;
};

class TypeRecord {
protected:
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

  static Expected<ModifierRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data);

  TypeIndex getModifiedType() const { return ModifiedType; }
  ModifierOptions getModifiers() const { return Modifiers; }

private:
  struct Layout {
    TypeIndex ModifiedType;
    ulittle16_t Modifiers; // ModifierOptions
  };

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

  static Expected<ProcedureRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data);

  static uint32_t getLayoutSize() { return 2 + sizeof(Layout); }

  TypeIndex getReturnType() const { return ReturnType; }
  CallingConvention getCallConv() const { return CallConv; }
  FunctionOptions getOptions() const { return Options; }
  uint16_t getParameterCount() const { return ParameterCount; }
  TypeIndex getArgumentList() const { return ArgumentList; }

private:
  struct Layout {
    TypeIndex ReturnType;
    CallingConvention CallConv;
    FunctionOptions Options;
    ulittle16_t NumParameters;
    TypeIndex ArgListType;
  };

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

  static Expected<MemberFunctionRecord> deserialize(TypeRecordKind Kind,
                                                    ArrayRef<uint8_t> &Data);

  TypeIndex getReturnType() const { return ReturnType; }
  TypeIndex getClassType() const { return ClassType; }
  TypeIndex getThisType() const { return ThisType; }
  CallingConvention getCallConv() const { return CallConv; }
  FunctionOptions getOptions() const { return Options; }
  uint16_t getParameterCount() const { return ParameterCount; }
  TypeIndex getArgumentList() const { return ArgumentList; }
  int32_t getThisPointerAdjustment() const { return ThisPointerAdjustment; }

private:
  struct Layout {
    TypeIndex ReturnType;
    TypeIndex ClassType;
    TypeIndex ThisType;
    CallingConvention CallConv;
    FunctionOptions Options;
    ulittle16_t NumParameters;
    TypeIndex ArgListType;
    little32_t ThisAdjustment;
  };

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

  static Expected<MemberFuncIdRecord> deserialize(TypeRecordKind Kind,
                                                  ArrayRef<uint8_t> &Data);
  TypeIndex getClassType() const { return ClassType; }
  TypeIndex getFunctionType() const { return FunctionType; }
  StringRef getName() const { return Name; }

private:
  struct Layout {
    TypeIndex ClassType;
    TypeIndex FunctionType;
    // Name: The null-terminated name follows.
  };
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

  static Expected<ArgListRecord> deserialize(TypeRecordKind Kind,
                                             ArrayRef<uint8_t> &Data);

  ArrayRef<TypeIndex> getIndices() const { return StringIndices; }

  static uint32_t getLayoutSize() { return 2 + sizeof(Layout); }

private:
  struct Layout {
    ulittle32_t NumArgs; // Number of arguments
                         // ArgTypes[]: Type indicies of arguments
  };

  std::vector<TypeIndex> StringIndices;
};

// LF_POINTER
class PointerRecord : public TypeRecord {
public:
  static const uint32_t PointerKindShift = 0;
  static const uint32_t PointerKindMask = 0x1F;

  static const uint32_t PointerModeShift = 5;
  static const uint32_t PointerModeMask = 0x07;

  static const uint32_t PointerSizeShift = 13;
  static const uint32_t PointerSizeMask = 0xFF;

  explicit PointerRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}

  PointerRecord(TypeIndex ReferentType, PointerKind Kind, PointerMode Mode,
                PointerOptions Options, uint8_t Size)
      : PointerRecord(ReferentType, Kind, Mode, Options, Size,
                      MemberPointerInfo()) {}

  PointerRecord(TypeIndex ReferentType, PointerKind Kind, PointerMode Mode,
                PointerOptions Options, uint8_t Size,
                const MemberPointerInfo &Member)
      : TypeRecord(TypeRecordKind::Pointer), ReferentType(ReferentType),
        PtrKind(Kind), Mode(Mode), Options(Options), Size(Size),
        MemberInfo(Member) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<PointerRecord> deserialize(TypeRecordKind Kind,
                                             ArrayRef<uint8_t> &Data);

  TypeIndex getReferentType() const { return ReferentType; }
  PointerKind getPointerKind() const { return PtrKind; }
  PointerMode getMode() const { return Mode; }
  PointerOptions getOptions() const { return Options; }
  uint8_t getSize() const { return Size; }
  MemberPointerInfo getMemberInfo() const { return MemberInfo; }

  bool isPointerToMember() const {
    return Mode == PointerMode::PointerToDataMember ||
           Mode == PointerMode::PointerToMemberFunction;
  }
  bool isFlat() const {
    return !!(uint32_t(Options) & uint32_t(PointerOptions::Flat32));
  }
  bool isConst() const {
    return !!(uint32_t(Options) & uint32_t(PointerOptions::Const));
  }
  bool isVolatile() const {
    return !!(uint32_t(Options) & uint32_t(PointerOptions::Volatile));
  }
  bool isUnaligned() const {
    return !!(uint32_t(Options) & uint32_t(PointerOptions::Unaligned));
  }

private:
  struct Layout {
    TypeIndex PointeeType;
    ulittle32_t Attrs; // pointer attributes
                       // if pointer to member:
                       //   PointerToMemberTail
    PointerKind getPtrKind() const {
      return PointerKind(Attrs & PointerKindMask);
    }
    PointerMode getPtrMode() const {
      return PointerMode((Attrs >> PointerModeShift) & PointerModeMask);
    }
    uint8_t getPtrSize() const {
      return (Attrs >> PointerSizeShift) & PointerSizeMask;
    }
    bool isFlat() const { return Attrs & (1 << 8); }
    bool isVolatile() const { return Attrs & (1 << 9); }
    bool isConst() const { return Attrs & (1 << 10); }
    bool isUnaligned() const { return Attrs & (1 << 11); }

    bool isPointerToDataMember() const {
      return getPtrMode() == PointerMode::PointerToDataMember;
    }
    bool isPointerToMemberFunction() const {
      return getPtrMode() == PointerMode::PointerToMemberFunction;
    }
    bool isPointerToMember() const {
      return isPointerToMemberFunction() || isPointerToDataMember();
    }
  };

  TypeIndex ReferentType;
  PointerKind PtrKind;
  PointerMode Mode;
  PointerOptions Options;
  uint8_t Size;
  MemberPointerInfo MemberInfo;
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

  static Expected<NestedTypeRecord> deserialize(TypeRecordKind Kind,
                                                ArrayRef<uint8_t> &Data);

  TypeIndex getNestedType() const { return Type; }
  StringRef getName() const { return Name; }

private:
  struct Layout {
    ulittle16_t Pad0; // Should be zero
    TypeIndex Type;   // Type index of nested type
                      // Name: Null-terminated string
  };

  TypeIndex Type;
  StringRef Name;
};

// LF_FIELDLIST
class FieldListRecord : public TypeRecord {
public:
  explicit FieldListRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  FieldListRecord(ArrayRef<uint8_t> ListData)
      : TypeRecord(TypeRecordKind::FieldList), ListData(ListData) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap) { return false; }

  static Expected<FieldListRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data);

  ArrayRef<uint8_t> getFieldListData() const { return ListData; }

private:
  ArrayRef<uint8_t> ListData;
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

  static Expected<ArrayRecord> deserialize(TypeRecordKind Kind,
                                           ArrayRef<uint8_t> &Data);

  TypeIndex getElementType() const { return ElementType; }
  TypeIndex getIndexType() const { return IndexType; }
  uint64_t getSize() const { return Size; }
  llvm::StringRef getName() const { return Name; }

private:
  struct Layout {
    TypeIndex ElementType;
    TypeIndex IndexType;
    // SizeOf: LF_NUMERIC encoded size in bytes. Not element count!
    // Name: The null-terminated name follows.
  };

  TypeIndex ElementType;
  TypeIndex IndexType;
  uint64_t Size;
  llvm::StringRef Name;
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

// LF_CLASS, LF_STRUCTURE, LF_INTERFACE
class ClassRecord : public TagRecord {
public:
  explicit ClassRecord(TypeRecordKind Kind) : TagRecord(Kind) {}
  ClassRecord(TypeRecordKind Kind, uint16_t MemberCount, ClassOptions Options,
              HfaKind Hfa, WindowsRTClassKind WinRTKind, TypeIndex FieldList,
              TypeIndex DerivationList, TypeIndex VTableShape, uint64_t Size,
              StringRef Name, StringRef UniqueName)
      : TagRecord(Kind, MemberCount, Options, FieldList, Name, UniqueName),
        Hfa(Hfa), WinRTKind(WinRTKind), DerivationList(DerivationList),
        VTableShape(VTableShape), Size(Size) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<ClassRecord> deserialize(TypeRecordKind Kind,
                                           ArrayRef<uint8_t> &Data);

  HfaKind getHfa() const { return Hfa; }
  WindowsRTClassKind getWinRTKind() const { return WinRTKind; }
  TypeIndex getDerivationList() const { return DerivationList; }
  TypeIndex getVTableShape() const { return VTableShape; }
  uint64_t getSize() const { return Size; }

private:
  struct Layout {
    ulittle16_t MemberCount; // Number of members in FieldList.
    ulittle16_t Properties;  // ClassOptions bitset
    TypeIndex FieldList;     // LF_FIELDLIST: List of all kinds of members
    TypeIndex DerivedFrom;   // LF_DERIVED: List of known derived classes
    TypeIndex VShape;        // LF_VTSHAPE: Shape of the vftable
    // SizeOf: The 'sizeof' the UDT in bytes is encoded as an LF_NUMERIC
    // integer.
    // Name: The null-terminated name follows.

    bool hasUniqueName() const {
      return Properties & uint16_t(ClassOptions::HasUniqueName);
    }
  };

  HfaKind Hfa;
  WindowsRTClassKind WinRTKind;
  TypeIndex DerivationList;
  TypeIndex VTableShape;
  uint64_t Size;
};

// LF_UNION
struct UnionRecord : public TagRecord {
  explicit UnionRecord(TypeRecordKind Kind) : TagRecord(Kind) {}
  UnionRecord(uint16_t MemberCount, ClassOptions Options, HfaKind Hfa,
              TypeIndex FieldList, uint64_t Size, StringRef Name,
              StringRef UniqueName)
      : TagRecord(TypeRecordKind::Union, MemberCount, Options, FieldList, Name,
                  UniqueName),
        Hfa(Hfa), Size(Size) {}

  static Expected<UnionRecord> deserialize(TypeRecordKind Kind,
                                           ArrayRef<uint8_t> &Data);

  HfaKind getHfa() const { return Hfa; }
  uint64_t getSize() const { return Size; }

private:
  struct Layout {
    ulittle16_t MemberCount; // Number of members in FieldList.
    ulittle16_t Properties;  // ClassOptions bitset
    TypeIndex FieldList;     // LF_FIELDLIST: List of all kinds of members
    // SizeOf: The 'sizeof' the UDT in bytes is encoded as an LF_NUMERIC
    // integer.
    // Name: The null-terminated name follows.

    bool hasUniqueName() const {
      return Properties & uint16_t(ClassOptions::HasUniqueName);
    }
  };

  HfaKind Hfa;
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

  static Expected<EnumRecord> deserialize(TypeRecordKind Kind,
                                          ArrayRef<uint8_t> &Data);

  TypeIndex getUnderlyingType() const { return UnderlyingType; }

private:
  struct Layout {
    ulittle16_t NumEnumerators; // Number of enumerators
    ulittle16_t Properties;
    TypeIndex UnderlyingType;
    TypeIndex FieldListType;
    // Name: The null-terminated name follows.

    bool hasUniqueName() const {
      return Properties & uint16_t(ClassOptions::HasUniqueName);
    }
  };

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

  static Expected<BitFieldRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data);

  TypeIndex getType() const { return Type; }
  uint8_t getBitOffset() const { return BitOffset; }
  uint8_t getBitSize() const { return BitSize; }

private:
  struct Layout {
    TypeIndex Type;
    uint8_t BitSize;
    uint8_t BitOffset;
  };

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

  static Expected<VFTableShapeRecord> deserialize(TypeRecordKind Kind,
                                                  ArrayRef<uint8_t> &Data);

  ArrayRef<VFTableSlotKind> getSlots() const {
    if (!SlotsRef.empty())
      return SlotsRef;
    return Slots;
  }
  uint32_t getEntryCount() const { return getSlots().size(); }

private:
  struct Layout {
    // Number of vftable entries. Each method may have more than one entry due
    // to
    // things like covariant return types.
    ulittle16_t VFEntryCount;
    // Descriptors[]: 4-bit virtual method descriptors of type CV_VTS_desc_e.
  };

private:
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

  static Expected<TypeServer2Record> deserialize(TypeRecordKind Kind,
                                                 ArrayRef<uint8_t> &Data);

  StringRef getGuid() const { return Guid; }

  uint32_t getAge() const { return Age; }

  StringRef getName() const { return Name; }

private:
  struct Layout {
    char Guid[16]; // GUID
    ulittle32_t Age;
    // Name: Name of the PDB as a null-terminated string
  };

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

  static Expected<StringIdRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data);

  TypeIndex getId() const { return Id; }

  StringRef getString() const { return String; }

private:
  struct Layout {
    TypeIndex id;
    // Name: Name of the PDB as a null-terminated string
  };

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

  static Expected<FuncIdRecord> deserialize(TypeRecordKind Kind,
                                            ArrayRef<uint8_t> &Data);

  TypeIndex getParentScope() const { return ParentScope; }

  TypeIndex getFunctionType() const { return FunctionType; }

  StringRef getName() const { return Name; }

private:
  struct Layout {
    TypeIndex ParentScope;
    TypeIndex FunctionType;
    // Name: The null-terminated name follows.
  };

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

  static Expected<UdtSourceLineRecord> deserialize(TypeRecordKind Kind,
                                                   ArrayRef<uint8_t> &Data);

  TypeIndex getUDT() const { return UDT; }
  TypeIndex getSourceFile() const { return SourceFile; }
  uint32_t getLineNumber() const { return LineNumber; }

private:
  struct Layout {
    TypeIndex UDT;        // The user-defined type
    TypeIndex SourceFile; // StringID containing the source filename
    ulittle32_t LineNumber;
  };

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

  static Expected<UdtModSourceLineRecord> deserialize(TypeRecordKind Kind,
                                                      ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    CV_DESERIALIZE(Data, L);

    return UdtModSourceLineRecord(L->UDT, L->SourceFile, L->LineNumber,
                                  L->Module);
  }

  TypeIndex getUDT() const { return UDT; }
  TypeIndex getSourceFile() const { return SourceFile; }
  uint32_t getLineNumber() const { return LineNumber; }
  uint16_t getModule() const { return Module; }

private:
  struct Layout {
    TypeIndex UDT;        // The user-defined type
    TypeIndex SourceFile; // StringID containing the source filename
    ulittle32_t LineNumber;
    ulittle16_t Module; // Module that contributes this UDT definition
  };

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

  static Expected<BuildInfoRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data);

  ArrayRef<TypeIndex> getArgs() const { return ArgIndices; }

private:
  struct Layout {
    ulittle16_t NumArgs; // Number of arguments
                         // ArgTypes[]: Type indicies of arguments
  };
  SmallVector<TypeIndex, 4> ArgIndices;
};

// LF_VFTABLE
class VFTableRecord : public TypeRecord {
public:
  explicit VFTableRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  VFTableRecord(TypeIndex CompleteClass, TypeIndex OverriddenVFTable,
                uint32_t VFPtrOffset, StringRef Name,
                ArrayRef<StringRef> Methods)
      : TypeRecord(TypeRecordKind::VFTable),
        CompleteClass(CompleteClass), OverriddenVFTable(OverriddenVFTable),
        VFPtrOffset(VFPtrOffset), Name(Name), MethodNamesRef(Methods) {}
  VFTableRecord(TypeIndex CompleteClass, TypeIndex OverriddenVFTable,
                uint32_t VFPtrOffset, StringRef Name,
                const std::vector<StringRef> &Methods)
      : TypeRecord(TypeRecordKind::VFTable),
        CompleteClass(CompleteClass), OverriddenVFTable(OverriddenVFTable),
        VFPtrOffset(VFPtrOffset), Name(Name), MethodNames(Methods) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<VFTableRecord> deserialize(TypeRecordKind Kind,
                                             ArrayRef<uint8_t> &Data);

  TypeIndex getCompleteClass() const { return CompleteClass; }
  TypeIndex getOverriddenVTable() const { return OverriddenVFTable; }
  uint32_t getVFPtrOffset() const { return VFPtrOffset; }
  StringRef getName() const { return Name; }
  ArrayRef<StringRef> getMethodNames() const {
    if (!MethodNamesRef.empty())
      return MethodNamesRef;
    return MethodNames;
  }

private:
  struct Layout {
    TypeIndex CompleteClass;     // Class that owns this vftable.
    TypeIndex OverriddenVFTable; // VFTable that this overrides.
    ulittle32_t VFPtrOffset;     // VFPtr offset in CompleteClass
    ulittle32_t NamesLen;        // Length of subsequent names array in bytes.
    // Names: A sequence of null-terminated strings. First string is vftable
    // names.
  };

  TypeIndex CompleteClass;
  TypeIndex OverriddenVFTable;
  ulittle32_t VFPtrOffset;
  StringRef Name;
  ArrayRef<StringRef> MethodNamesRef;
  std::vector<StringRef> MethodNames;
};

// LF_ONEMETHOD
class OneMethodRecord : public TypeRecord {
public:
  explicit OneMethodRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  OneMethodRecord(TypeIndex Type, MethodKind Kind, MethodOptions Options,
                  MemberAccess Access, int32_t VFTableOffset, StringRef Name)
      : TypeRecord(TypeRecordKind::OneMethod), Type(Type), Kind(Kind),
        Options(Options), Access(Access), VFTableOffset(VFTableOffset),
        Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<OneMethodRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data);

  TypeIndex getType() const { return Type; }
  MethodKind getKind() const { return Kind; }
  MethodOptions getOptions() const { return Options; }
  MemberAccess getAccess() const { return Access; }
  int32_t getVFTableOffset() const { return VFTableOffset; }
  StringRef getName() const { return Name; }

  bool isIntroducingVirtual() const {
    return Kind == MethodKind::IntroducingVirtual ||
           Kind == MethodKind::PureIntroducingVirtual;
  }

private:
  struct Layout {
    MemberAttributes Attrs;
    TypeIndex Type;
    // If is introduced virtual method:
    //   VFTableOffset: int32_t offset in vftable
    // Name: Null-terminated string
  };

  TypeIndex Type;
  MethodKind Kind;
  MethodOptions Options;
  MemberAccess Access;
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

  static Expected<MethodOverloadListRecord>
  deserialize(TypeRecordKind Kind, ArrayRef<uint8_t> &Data);

  ArrayRef<OneMethodRecord> getMethods() const { return Methods; }

private:
  struct Layout {
    MemberAttributes Attrs;
    ulittle16_t Padding;

    TypeIndex Type;
    // If is introduced virtual method:
    //   VFTableOffset: int32_t offset in vftable
  };

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

  static Expected<OverloadedMethodRecord> deserialize(TypeRecordKind Kind,
                                                      ArrayRef<uint8_t> &Data);

  uint16_t getNumOverloads() const { return NumOverloads; }
  TypeIndex getMethodList() const { return MethodList; }
  StringRef getName() const { return Name; }

private:
  struct Layout {
    ulittle16_t MethodCount; // Size of overload set
    TypeIndex MethList;      // Type index of methods in overload set
                             // Name: Null-terminated string
  };

  uint16_t NumOverloads;
  TypeIndex MethodList;
  StringRef Name;
};

// LF_MEMBER
class DataMemberRecord : public TypeRecord {
public:
  explicit DataMemberRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  DataMemberRecord(MemberAccess Access, TypeIndex Type, uint64_t Offset,
                   StringRef Name)
      : TypeRecord(TypeRecordKind::DataMember), Access(Access), Type(Type),
        FieldOffset(Offset), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<DataMemberRecord> deserialize(TypeRecordKind Kind,
                                                ArrayRef<uint8_t> &Data);

  MemberAccess getAccess() const { return Access; }
  TypeIndex getType() const { return Type; }
  uint64_t getFieldOffset() const { return FieldOffset; }
  StringRef getName() const { return Name; }

private:
  struct Layout {
    MemberAttributes Attrs; // Access control attributes, etc
    TypeIndex Type;
    // FieldOffset: LF_NUMERIC encoded byte offset
    // Name: Null-terminated string
  };

  MemberAccess Access;
  TypeIndex Type;
  uint64_t FieldOffset;
  StringRef Name;
};

// LF_STMEMBER
class StaticDataMemberRecord : public TypeRecord {
public:
  explicit StaticDataMemberRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  StaticDataMemberRecord(MemberAccess Access, TypeIndex Type, StringRef Name)
      : TypeRecord(TypeRecordKind::StaticDataMember), Access(Access),
        Type(Type), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<StaticDataMemberRecord> deserialize(TypeRecordKind Kind,
                                                      ArrayRef<uint8_t> &Data);

  MemberAccess getAccess() const { return Access; }
  TypeIndex getType() const { return Type; }
  StringRef getName() const { return Name; }

private:
  struct Layout {
    MemberAttributes Attrs; // Access control attributes, etc
    TypeIndex Type;
    // Name: Null-terminated string
  };

  MemberAccess Access;
  TypeIndex Type;
  StringRef Name;
};

// LF_ENUMERATE
class EnumeratorRecord : public TypeRecord {
public:
  explicit EnumeratorRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  EnumeratorRecord(MemberAccess Access, APSInt Value, StringRef Name)
      : TypeRecord(TypeRecordKind::Enumerator), Access(Access),
        Value(std::move(Value)), Name(Name) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<EnumeratorRecord> deserialize(TypeRecordKind Kind,
                                                ArrayRef<uint8_t> &Data);

  MemberAccess getAccess() const { return Access; }
  APSInt getValue() const { return Value; }
  StringRef getName() const { return Name; }

private:
  struct Layout {
    MemberAttributes Attrs; // Access control attributes, etc
                            // EnumValue: LF_NUMERIC encoded enumerator value
                            // Name: Null-terminated string
  };

  MemberAccess Access;
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

  static Expected<VFPtrRecord> deserialize(TypeRecordKind Kind,
                                           ArrayRef<uint8_t> &Data);

  TypeIndex getType() const { return Type; }

private:
  struct Layout {
    ulittle16_t Pad0;
    TypeIndex Type; // Type of vfptr
  };
  TypeIndex Type;
};

// LF_BCLASS, LF_BINTERFACE
class BaseClassRecord : public TypeRecord {
public:
  explicit BaseClassRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  BaseClassRecord(MemberAccess Access, TypeIndex Type, uint64_t Offset)
      : TypeRecord(TypeRecordKind::BaseClass), Access(Access), Type(Type),
        Offset(Offset) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<BaseClassRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data);

  MemberAccess getAccess() const { return Access; }
  TypeIndex getBaseType() const { return Type; }
  uint64_t getBaseOffset() const { return Offset; }

private:
  struct Layout {
    MemberAttributes Attrs; // Access control attributes, etc
    TypeIndex BaseType;     // Base class type
    // BaseOffset: LF_NUMERIC encoded byte offset of base from derived.
  };
  MemberAccess Access;
  TypeIndex Type;
  uint64_t Offset;
};

// LF_VBCLASS, LF_IVBCLASS
class VirtualBaseClassRecord : public TypeRecord {
public:
  explicit VirtualBaseClassRecord(TypeRecordKind Kind) : TypeRecord(Kind) {}
  VirtualBaseClassRecord(MemberAccess Access, TypeIndex BaseType,
                         TypeIndex VBPtrType, uint64_t Offset, uint64_t Index)
      : TypeRecord(TypeRecordKind::VirtualBaseClass), Access(Access),
        BaseType(BaseType), VBPtrType(VBPtrType), VBPtrOffset(Offset),
        VTableIndex(Index) {}

  /// Rewrite member type indices with IndexMap. Returns false if a type index
  /// is not in the map.
  bool remapTypeIndices(ArrayRef<TypeIndex> IndexMap);

  static Expected<VirtualBaseClassRecord> deserialize(TypeRecordKind Kind,
                                                      ArrayRef<uint8_t> &Data);

  MemberAccess getAccess() const { return Access; }
  TypeIndex getBaseType() const { return BaseType; }
  TypeIndex getVBPtrType() const { return VBPtrType; }
  uint64_t getVBPtrOffset() const { return VBPtrOffset; }
  uint64_t getVTableIndex() const { return VTableIndex; }

private:
  struct Layout {
    MemberAttributes Attrs; // Access control attributes, etc.
    TypeIndex BaseType;     // Base class type
    TypeIndex VBPtrType;    // Virtual base pointer type
    // VBPtrOffset: Offset of vbptr from vfptr encoded as LF_NUMERIC.
    // VBTableIndex: Index of vbase within vbtable encoded as LF_NUMERIC.
  };
  MemberAccess Access;
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

  static Expected<ListContinuationRecord> deserialize(TypeRecordKind Kind,
                                                      ArrayRef<uint8_t> &Data);

private:
  struct Layout {
    ulittle16_t Pad0;
    TypeIndex ContinuationIndex;
  };
  TypeIndex ContinuationIndex;
};

typedef CVRecord<TypeLeafKind> CVType;
typedef msf::VarStreamArray<CVType> CVTypeArray;
}
}

#endif
