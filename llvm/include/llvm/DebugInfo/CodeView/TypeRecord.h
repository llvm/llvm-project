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

using llvm::support::little32_t;
using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;

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

//===----------------------------------------------------------------------===//
// On-disk representation of type information

// A CodeView type stream is a sequence of TypeRecords. Records larger than
// 65536 must chain on to a second record. Each TypeRecord is followed by one of
// the leaf types described below.
struct TypeRecordPrefix {
  ulittle16_t Len;  // Type record length, starting from &Leaf.
  ulittle16_t Leaf; // Type record kind (TypeLeafKind)
};

// LF_TYPESERVER2
struct TypeServer2 {
  char Signature[16];  // GUID
  ulittle32_t Age;
  // Name: Name of the PDB as a null-terminated string
};

// LF_STRING_ID
struct StringId {
  TypeIndex id;
};

// LF_FUNC_ID
struct FuncId {
  TypeIndex ParentScope;
  TypeIndex FunctionType;
  // Name: The null-terminated name follows.
};

// LF_CLASS, LF_STRUCT, LF_INTERFACE
struct ClassType {
  ulittle16_t MemberCount; // Number of members in FieldList.
  ulittle16_t Properties;  // ClassOptions bitset
  TypeIndex FieldList;     // LF_FIELDLIST: List of all kinds of members
  TypeIndex DerivedFrom;   // LF_DERIVED: List of known derived classes
  TypeIndex VShape;        // LF_VTSHAPE: Shape of the vftable
  // SizeOf: The 'sizeof' the UDT in bytes is encoded as an LF_NUMERIC integer.
  // Name: The null-terminated name follows.
};

// LF_UNION
struct UnionType {
  ulittle16_t MemberCount; // Number of members in FieldList.
  ulittle16_t Properties;  // ClassOptions bitset
  TypeIndex FieldList;     // LF_FIELDLIST: List of all kinds of members
  // SizeOf: The 'sizeof' the UDT in bytes is encoded as an LF_NUMERIC integer.
  // Name: The null-terminated name follows.
};

// LF_POINTER
struct PointerType {
  TypeIndex PointeeType;
  ulittle32_t Attrs; // pointer attributes
  // if pointer to member:
  //   PointerToMemberTail

  PointerKind getPtrKind() const { return PointerKind(Attrs & 0x1f); }
  PointerMode getPtrMode() const { return PointerMode((Attrs >> 5) & 0x07); }
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

struct PointerToMemberTail {
  TypeIndex ClassType;
  ulittle16_t Representation; // PointerToMemberRepresentation
};

/// In Clang parlance, these are "qualifiers".  LF_MODIFIER
struct TypeModifier {
  TypeIndex ModifiedType;
  ulittle16_t Modifiers; // ModifierOptions
};

// LF_VTSHAPE
struct VTableShape {
  // Number of vftable entries. Each method may have more than one entry due to
  // things like covariant return types.
  ulittle16_t VFEntryCount;
  // Descriptors[]: 4-bit virtual method descriptors of type CV_VTS_desc_e.
};

// LF_UDT_SRC_LINE
struct UDTSrcLine {
  TypeIndex UDT;        // The user-defined type
  TypeIndex SourceFile; // StringID containing the source filename
  ulittle32_t LineNumber;
};

// LF_ARGLIST, LF_SUBSTR_LIST
struct ArgList {
  ulittle32_t NumArgs; // Number of arguments
  // ArgTypes[]: Type indicies of arguments
};

// LF_BUILDINFO
struct BuildInfo {
  ulittle16_t NumArgs; // Number of arguments
  // ArgTypes[]: Type indicies of arguments
};

// LF_ENUM
struct EnumType {
  ulittle16_t NumEnumerators; // Number of enumerators
  ulittle16_t Properties;
  TypeIndex UnderlyingType;
  TypeIndex FieldListType;
  // Name: The null-terminated name follows.
};

// LF_ARRAY
struct ArrayType {
  TypeIndex ElementType;
  TypeIndex IndexType;
  // SizeOf: LF_NUMERIC encoded size in bytes. Not element count!
  // Name: The null-terminated name follows.
};

// LF_VFTABLE
struct VFTableType {
  TypeIndex CompleteClass;     // Class that owns this vftable.
  TypeIndex OverriddenVFTable; // VFTable that this overrides.
  ulittle32_t VFPtrOffset;     // VFPtr offset in CompleteClass
  ulittle32_t NamesLen;        // Length of subsequent names array in bytes.
  // Names: A sequence of null-terminated strings. First string is vftable
  // names.
};

// LF_MFUNC_ID
struct MemberFuncId {
  TypeIndex ClassType;
  TypeIndex FunctionType;
  // Name: The null-terminated name follows.
};

// LF_PROCEDURE
struct ProcedureType {
  TypeIndex ReturnType;
  CallingConvention CallConv;
  FunctionOptions Options;
  ulittle16_t NumParameters;
  TypeIndex ArgListType;
};

// LF_MFUNCTION
struct MemberFunctionType {
  TypeIndex ReturnType;
  TypeIndex ClassType;
  TypeIndex ThisType;
  CallingConvention CallConv;
  FunctionOptions Options;
  ulittle16_t NumParameters;
  TypeIndex ArgListType;
  little32_t ThisAdjustment;
};

//===----------------------------------------------------------------------===//
// Field list records, which do not include leafs or sizes

/// Equvalent to CV_fldattr_t in cvinfo.h.
struct MemberAttributes {
  ulittle16_t Attrs;

  /// Get the access specifier. Valid for any kind of member.
  MemberAccess getAccess() const {
    return MemberAccess(unsigned(Attrs) & unsigned(MethodOptions::AccessMask));
  }

  /// Indicates if a method is defined with friend, virtual, static, etc.
  MethodKind getMethodKind() const {
    return MethodKind(
        (unsigned(Attrs) & unsigned(MethodOptions::MethodKindMask)) >> 2);
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

// LF_NESTTYPE
struct NestedType {
  ulittle16_t Pad0; // Should be zero
  TypeIndex Type;   // Type index of nested type
  // Name: Null-terminated string
};

// LF_ONEMETHOD
struct OneMethod {
  MemberAttributes Attrs;
  TypeIndex Type;
  // If is introduced virtual method:
  //   VFTableOffset: int32_t offset in vftable
  // Name: Null-terminated string

  MethodKind getMethodKind() const {
    return Attrs.getMethodKind();
  }

  bool isVirtual() const { return Attrs.isVirtual(); }
  bool isIntroducedVirtual() const { return Attrs.isIntroducedVirtual(); }
};

struct MethodListEntry {
  MemberAttributes Attrs;
  ulittle16_t Padding;

  TypeIndex Type;
  // If is introduced virtual method:
  //   VFTableOffset: int32_t offset in vftable

  MethodKind getMethodKind() const {
    return Attrs.getMethodKind();
  }

  bool isVirtual() const { return Attrs.isVirtual(); }
  bool isIntroducedVirtual() const { return Attrs.isIntroducedVirtual(); }
};

/// For method overload sets.  LF_METHOD
struct OverloadedMethod {
  ulittle16_t MethodCount; // Size of overload set
  TypeIndex MethList;      // Type index of methods in overload set
  // Name: Null-terminated string
};

// LF_VFUNCTAB
struct VirtualFunctionPointer {
  ulittle16_t Pad0;
  TypeIndex Type;   // Type of vfptr
};

// LF_MEMBER
struct DataMember {
  MemberAttributes Attrs; // Access control attributes, etc
  TypeIndex Type;
  // FieldOffset: LF_NUMERIC encoded byte offset
  // Name: Null-terminated string
};

// LF_STMEMBER
struct StaticDataMember {
  MemberAttributes Attrs; // Access control attributes, etc
  TypeIndex Type;
  // Name: Null-terminated string
};

// LF_ENUMERATE
struct Enumerator {
  MemberAttributes Attrs; // Access control attributes, etc
  // EnumValue: LF_NUMERIC encoded enumerator value
  // Name: Null-terminated string
};

// LF_BCLASS, LF_BINTERFACE
struct BaseClass {
  MemberAttributes Attrs; // Access control attributes, etc
  TypeIndex BaseType;     // Base class type
  // BaseOffset: LF_NUMERIC encoded byte offset of base from derived.
};

// LF_VBCLASS | LV_IVBCLASS
struct VirtualBaseClass {
  MemberAttributes Attrs; // Access control attributes, etc.
  TypeIndex BaseType;     // Base class type
  TypeIndex VBPtrType;    // Virtual base pointer type
  // VBPtrOffset: Offset of vbptr from vfptr encoded as LF_NUMERIC.
  // VBTableIndex: Index of vbase within vbtable encoded as LF_NUMERIC.
};
}
}

#endif
