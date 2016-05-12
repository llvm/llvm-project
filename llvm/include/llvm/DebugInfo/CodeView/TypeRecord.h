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
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/ErrorOr.h"
#include <cinttypes>
#include <tuple>

namespace llvm {
namespace codeview {

using llvm::support::little32_t;
using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;

/// Decodes a numeric "leaf" value. These are integer literals encountered in
/// the type stream. If the value is positive and less than LF_NUMERIC (1 <<
/// 15), it is emitted directly in Data. Otherwise, it has a tag like LF_CHAR
/// that indicates the bitwidth and sign of the numeric data.
bool decodeNumericLeaf(ArrayRef<uint8_t> &Data, APSInt &Num);

inline bool decodeNumericLeaf(StringRef &Data, APSInt &Num) {
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(Data.data()),
                          Data.size());
  bool Success = decodeNumericLeaf(Bytes, Num);
  Data = StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return Success;
}

/// Decode a numeric leaf value that is known to be a uint32_t.
bool decodeUIntLeaf(ArrayRef<uint8_t> &Data, uint64_t &Num);

/// Reinterpret a byte array as an array of characters. Does not interpret as
/// a C string, as StringRef has several helpers (split) that make that easy.
inline StringRef getBytesAsCharacters(ArrayRef<uint8_t> LeafData) {
  return StringRef(reinterpret_cast<const char *>(LeafData.data()),
                   LeafData.size());
}

inline StringRef getBytesAsCString(ArrayRef<uint8_t> LeafData) {
  return getBytesAsCharacters(LeafData).split('\0').first;
}

/// Consumes sizeof(T) bytes from the given byte sequence. Returns an error if
/// there are not enough bytes remaining. Reinterprets the consumed bytes as a
/// T object and points 'Res' at them.
template <typename T, typename U>
inline std::error_code consumeObject(U &Data, const T *&Res) {
  if (Data.size() < sizeof(*Res))
    return std::make_error_code(std::errc::illegal_byte_sequence);
  Res = reinterpret_cast<const T *>(Data.data());
  Data = Data.drop_front(sizeof(*Res));
  return std::error_code();
}

inline std::error_code consumeCString(ArrayRef<uint8_t> &Data, StringRef &Str) {
  if (Data.empty())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  StringRef Rest;
  std::tie(Str, Rest) = getBytesAsCharacters(Data).split('\0');
  // We expect this to be null terminated.  If it was not, it is an error.
  if (Data.size() == Str.size())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  Data = ArrayRef<uint8_t>(Rest.bytes_begin(), Rest.bytes_end());
  return std::error_code();
}

template <typename T>
inline std::error_code consumeArray(ArrayRef<uint8_t> &Data,
                                    ArrayRef<T> &Result, uint32_t N) {
  uint32_t Size = sizeof(T) * N;
  if (Data.size() < Size)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  Result = ArrayRef<T>(reinterpret_cast<const T *>(Data.data()), N);
  Data = Data.drop_front(Size);
  return std::error_code();
}

inline std::error_code consumeUInt32(StringRef &Data, uint32_t &Res) {
  const support::ulittle32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Res = *IntPtr;
  return std::error_code();
}

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

// Does not correspond to any tag, this is the tail of an LF_POINTER record
// if it represents a member pointer.
class MemberPointerInfo {
public:
  MemberPointerInfo() {}

  MemberPointerInfo(TypeIndex ContainingType,
                    PointerToMemberRepresentation Representation)
      : ContainingType(ContainingType), Representation(Representation) {}

  static ErrorOr<MemberPointerInfo> deserialize(ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    TypeIndex T = L->ClassType;
    uint16_t R = L->Representation;
    PointerToMemberRepresentation PMR =
        static_cast<PointerToMemberRepresentation>(R);
    return MemberPointerInfo(T, PMR);
  }

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
  ModifierRecord(TypeIndex ModifiedType, ModifierOptions Modifiers)
      : TypeRecord(TypeRecordKind::Modifier), ModifiedType(ModifiedType),
        Modifiers(Modifiers) {}

  static ErrorOr<ModifierRecord> deserialize(TypeRecordKind Kind,
                                             ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    TypeIndex M = L->ModifiedType;
    uint16_t O = L->Modifiers;
    ModifierOptions MO = static_cast<ModifierOptions>(O);
    return ModifierRecord(M, MO);
  }

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
  ProcedureRecord(TypeIndex ReturnType, CallingConvention CallConv,
                  FunctionOptions Options, uint16_t ParameterCount,
                  TypeIndex ArgumentList)
      : TypeRecord(TypeRecordKind::Procedure), ReturnType(ReturnType),
        CallConv(CallConv), Options(Options), ParameterCount(ParameterCount),
        ArgumentList(ArgumentList) {}

  static ErrorOr<ProcedureRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    return ProcedureRecord(L->ReturnType, L->CallConv, L->Options,
                           L->NumParameters, L->ArgListType);
  }

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
  MemberFunctionRecord(TypeIndex ReturnType, TypeIndex ClassType,
                       TypeIndex ThisType, CallingConvention CallConv,
                       FunctionOptions Options, uint16_t ParameterCount,
                       TypeIndex ArgumentList, int32_t ThisPointerAdjustment)
      : TypeRecord(TypeRecordKind::MemberFunction), ReturnType(ReturnType),
        ClassType(ClassType), ThisType(ThisType), CallConv(CallConv),
        Options(Options), ParameterCount(ParameterCount),
        ArgumentList(ArgumentList),
        ThisPointerAdjustment(ThisPointerAdjustment) {}

  static ErrorOr<MemberFunctionRecord> deserialize(TypeRecordKind Kind,
                                                   ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    return MemberFunctionRecord(L->ReturnType, L->ClassType, L->ThisType,
                                L->CallConv, L->Options, L->NumParameters,
                                L->ArgListType, L->ThisAdjustment);
  }

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
class MemberFunctionIdRecord : public TypeRecord {
public:
  MemberFunctionIdRecord(TypeIndex ClassType, TypeIndex FunctionType,
                         StringRef Name)
      : TypeRecord(TypeRecordKind::MemberFunctionId), ClassType(ClassType),
        FunctionType(FunctionType), Name(Name) {}

  static ErrorOr<MemberFunctionIdRecord> deserialize(TypeRecordKind Kind,
                                                     ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;
    return MemberFunctionIdRecord(L->ClassType, L->FunctionType, Name);
  }

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
class StringListRecord : public TypeRecord {
public:
  StringListRecord(TypeRecordKind Kind, ArrayRef<TypeIndex> Indices)
      : TypeRecord(Kind), StringIndices(Indices) {}

  static ErrorOr<StringListRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data) {
    if (Kind != TypeRecordKind::SubstringList &&
        Kind != TypeRecordKind::ArgumentList)
      return std::make_error_code(std::errc::illegal_byte_sequence);

    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    ArrayRef<TypeIndex> Indices;
    if (auto EC = consumeArray(Data, Indices, L->NumArgs))
      return EC;
    return StringListRecord(Kind, Indices);
  }

  ArrayRef<TypeIndex> getIndices() const { return StringIndices; }

  static uint32_t getLayoutSize() { return 2 + sizeof(Layout); }

private:
  struct Layout {
    ulittle32_t NumArgs; // Number of arguments
                         // ArgTypes[]: Type indicies of arguments
  };

  ArrayRef<TypeIndex> StringIndices;
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

  static ErrorOr<PointerRecord> deserialize(TypeRecordKind Kind,
                                            ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    PointerKind PtrKind = L->getPtrKind();
    PointerMode Mode = L->getPtrMode();
    uint32_t Opts = L->Attrs;
    PointerOptions Options = static_cast<PointerOptions>(Opts);
    uint8_t Size = L->getPtrSize();

    if (L->isPointerToMember()) {
      auto E = MemberPointerInfo::deserialize(Data);
      if (E.getError())
        return std::make_error_code(std::errc::illegal_byte_sequence);
      return PointerRecord(L->PointeeType, PtrKind, Mode, Options, Size, *E);
    }

    return PointerRecord(L->PointeeType, PtrKind, Mode, Options, Size);
  }

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
  NestedTypeRecord(TypeIndex Type, StringRef Name)
      : TypeRecord(TypeRecordKind::NestedType), Type(Type), Name(Name) {}

  static ErrorOr<NestedTypeRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;
    return NestedTypeRecord(L->Type, Name);
  }

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

// LF_ARRAY
class ArrayRecord : public TypeRecord {
public:
  ArrayRecord(TypeIndex ElementType, TypeIndex IndexType, uint64_t Size,
              StringRef Name)
      : TypeRecord(TypeRecordKind::Array), ElementType(ElementType),
        IndexType(IndexType), Size(Size), Name(Name) {}

  static ErrorOr<ArrayRecord> deserialize(TypeRecordKind Kind,
                                          ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    uint64_t Size;
    if (!decodeUIntLeaf(Data, Size))
      return std::make_error_code(std::errc::illegal_byte_sequence);
    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;
    return ArrayRecord(L->ElementType, L->IndexType, Size, Name);
  }

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
  TagRecord(TypeRecordKind Kind, uint16_t MemberCount, ClassOptions Options,
            TypeIndex FieldList, StringRef Name, StringRef UniqueName)
      : TypeRecord(Kind), MemberCount(MemberCount), Options(Options),
        FieldList(FieldList), Name(Name), UniqueName(UniqueName) {}

public:
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
  ClassRecord(TypeRecordKind Kind, uint16_t MemberCount, ClassOptions Options,
              HfaKind Hfa, WindowsRTClassKind WinRTKind, TypeIndex FieldList,
              TypeIndex DerivationList, TypeIndex VTableShape, uint64_t Size,
              StringRef Name, StringRef UniqueName)
      : TagRecord(Kind, MemberCount, Options, FieldList, Name, UniqueName),
        Hfa(Hfa), WinRTKind(WinRTKind), DerivationList(DerivationList),
        VTableShape(VTableShape), Size(Size) {}

  static ErrorOr<ClassRecord> deserialize(TypeRecordKind Kind,
                                          ArrayRef<uint8_t> &Data) {
    uint64_t Size = 0;
    StringRef Name;
    StringRef UniqueName;
    uint16_t Props;

    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    Props = L->Properties;
    uint16_t WrtValue = (Props & WinRTKindMask) >> WinRTKindShift;
    WindowsRTClassKind WRT = static_cast<WindowsRTClassKind>(WrtValue);
    uint16_t HfaMask = (Props & HfaKindMask) >> HfaKindShift;
    HfaKind Hfa = static_cast<HfaKind>(HfaMask);

    if (!decodeUIntLeaf(Data, Size))
      return std::make_error_code(std::errc::illegal_byte_sequence);
    if (auto EC = consumeCString(Data, Name))
      return EC;
    if (Props & uint16_t(ClassOptions::HasUniqueName)) {
      if (auto EC = consumeCString(Data, UniqueName))
        return EC;
    }

    ClassOptions Options = static_cast<ClassOptions>(Props);
    return ClassRecord(Kind, L->MemberCount, Options, Hfa, WRT, L->FieldList,
                       L->DerivedFrom, L->VShape, Size, Name, UniqueName);
  }

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
  };

  HfaKind Hfa;
  WindowsRTClassKind WinRTKind;
  TypeIndex DerivationList;
  TypeIndex VTableShape;
  uint64_t Size;
};

// LF_UNION
struct UnionRecord : public TagRecord {
  UnionRecord(uint16_t MemberCount, ClassOptions Options, HfaKind Hfa,
              TypeIndex FieldList, uint64_t Size, StringRef Name,
              StringRef UniqueName)
      : TagRecord(TypeRecordKind::Union, MemberCount, Options, FieldList, Name,
                  UniqueName),
        Hfa(Hfa), Size(Size) {}

  static ErrorOr<UnionRecord> deserialize(TypeRecordKind Kind,
                                          ArrayRef<uint8_t> &Data) {
    uint64_t Size = 0;
    StringRef Name;
    StringRef UniqueName;
    uint16_t Props;

    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    Props = L->Properties;

    uint16_t HfaMask = (Props & HfaKindMask) >> HfaKindShift;
    HfaKind Hfa = static_cast<HfaKind>(HfaMask);

    if (!decodeUIntLeaf(Data, Size))
      return std::make_error_code(std::errc::illegal_byte_sequence);
    if (auto EC = consumeCString(Data, Name))
      return EC;
    if (Props & uint16_t(ClassOptions::HasUniqueName)) {
      if (auto EC = consumeCString(Data, UniqueName))
        return EC;
    }

    ClassOptions Options = static_cast<ClassOptions>(Props);
    return UnionRecord(L->MemberCount, Options, Hfa, L->FieldList, Size, Name,
                       UniqueName);
  }

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
  };

  HfaKind Hfa;
  uint64_t Size;
};

// LF_ENUM
class EnumRecord : public TagRecord {
public:
  EnumRecord(uint16_t MemberCount, ClassOptions Options, TypeIndex FieldList,
             StringRef Name, StringRef UniqueName, TypeIndex UnderlyingType)
      : TagRecord(TypeRecordKind::Enum, MemberCount, Options, FieldList, Name,
                  UniqueName),
        UnderlyingType(UnderlyingType) {}

  static ErrorOr<EnumRecord> deserialize(TypeRecordKind Kind,
                                         ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    uint16_t P = L->Properties;
    ClassOptions Options = static_cast<ClassOptions>(P);
    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;
    return EnumRecord(L->NumEnumerators, Options, L->FieldListType, Name, Name,
                      L->UnderlyingType);
  }

  TypeIndex getUnderlyingType() const { return UnderlyingType; }

private:
  struct Layout {
    ulittle16_t NumEnumerators; // Number of enumerators
    ulittle16_t Properties;
    TypeIndex UnderlyingType;
    TypeIndex FieldListType;
    // Name: The null-terminated name follows.
  };

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

// LF_VTSHAPE
class VirtualTableShapeRecord : TypeRecord {
public:
  explicit VirtualTableShapeRecord(ArrayRef<VirtualTableSlotKind> Slots)
      : TypeRecord(TypeRecordKind::VirtualTableShape), SlotsRef(Slots) {}
  explicit VirtualTableShapeRecord(std::vector<VirtualTableSlotKind> Slots)
      : TypeRecord(TypeRecordKind::VirtualTableShape), Slots(Slots) {}

  static ErrorOr<VirtualTableShapeRecord> deserialize(TypeRecordKind Kind,
                                                      ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    std::vector<VirtualTableSlotKind> Slots;
    uint16_t Count = L->VFEntryCount;
    while (Count > 0) {
      if (Data.empty())
        return std::make_error_code(std::errc::illegal_byte_sequence);

      // Process up to 2 nibbles at a time (if there are at least 2 remaining)
      uint8_t Value = Data[0] & 0x0F;
      Slots.push_back(static_cast<VirtualTableSlotKind>(Value));
      if (--Count > 0) {
        Value = (Data[0] & 0xF0) >> 4;
        Slots.push_back(static_cast<VirtualTableSlotKind>(Value));
        --Count;
      }
      Data = Data.slice(1);
    }

    return VirtualTableShapeRecord(Slots);
  }

  ArrayRef<VirtualTableSlotKind> getSlots() const {
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
  ArrayRef<VirtualTableSlotKind> SlotsRef;
  std::vector<VirtualTableSlotKind> Slots;
};

// LF_TYPESERVER2
class TypeServer2Record : TypeRecord {
public:
  TypeServer2Record(StringRef Guid, uint32_t Age, StringRef Name)
      : TypeRecord(TypeRecordKind::TypeServer2), Guid(Guid), Age(Age),
        Name(Name) {}

  static ErrorOr<TypeServer2Record> deserialize(TypeRecordKind Kind,
                                                ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;
    return TypeServer2Record(StringRef(L->Guid, 16), L->Age, Name);
  }

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
  StringIdRecord(TypeIndex Id, StringRef String)
      : TypeRecord(TypeRecordKind::StringId), Id(Id), String(String) {}

  static ErrorOr<StringIdRecord> deserialize(TypeRecordKind Kind,
                                             ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;
    return StringIdRecord(L->id, Name);
  }

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
  FuncIdRecord(TypeIndex ParentScope, TypeIndex FunctionType, StringRef Name)
      : TypeRecord(TypeRecordKind::FunctionId), ParentScope(ParentScope),
        FunctionType(FunctionType), Name(Name) {}

  static ErrorOr<FuncIdRecord> deserialize(TypeRecordKind Kind,
                                           ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    return FuncIdRecord(L->ParentScope, L->FunctionType, Name);
  }

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
  UdtSourceLineRecord(TypeIndex UDT, TypeIndex SourceFile, uint32_t LineNumber)
      : TypeRecord(TypeRecordKind::UdtSourceLine), UDT(UDT),
        SourceFile(SourceFile), LineNumber(LineNumber) {}

  static ErrorOr<UdtSourceLineRecord> deserialize(TypeRecordKind Kind,
                                                  ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    return UdtSourceLineRecord(L->UDT, L->SourceFile, L->LineNumber);
  }

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

// LF_BUILDINFO
class BuildInfoRecord : public TypeRecord {
public:
  BuildInfoRecord(ArrayRef<TypeIndex> ArgIndices)
      : TypeRecord(TypeRecordKind::Modifier), ArgIndices(ArgIndices) {}

  static ErrorOr<BuildInfoRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    ArrayRef<TypeIndex> Indices;
    if (auto EC = consumeArray(Data, Indices, L->NumArgs))
      return EC;
    return BuildInfoRecord(Indices);
  }

  ArrayRef<TypeIndex> getArgs() const { return ArgIndices; }

private:
  struct Layout {
    ulittle16_t NumArgs; // Number of arguments
                         // ArgTypes[]: Type indicies of arguments
  };
  ArrayRef<TypeIndex> ArgIndices;
};

// LF_VFTABLE
class VirtualTableRecord : public TypeRecord {
public:
  VirtualTableRecord(TypeIndex CompleteClass, TypeIndex OverriddenVFTable,
                     uint32_t VFPtrOffset, StringRef Name,
                     ArrayRef<StringRef> Methods)
      : TypeRecord(TypeRecordKind::VirtualFunctionTable),
        CompleteClass(CompleteClass), OverriddenVFTable(OverriddenVFTable),
        VFPtrOffset(VFPtrOffset), Name(Name), MethodNamesRef(Methods) {}
  VirtualTableRecord(TypeIndex CompleteClass, TypeIndex OverriddenVFTable,
                     uint32_t VFPtrOffset, StringRef Name,
                     const std::vector<StringRef> &Methods)
      : TypeRecord(TypeRecordKind::VirtualFunctionTable),
        CompleteClass(CompleteClass), OverriddenVFTable(OverriddenVFTable),
        VFPtrOffset(VFPtrOffset), Name(Name), MethodNames(Methods) {}

  static ErrorOr<VirtualTableRecord> deserialize(TypeRecordKind Kind,
                                                 ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    std::vector<StringRef> Names;
    while (!Data.empty()) {
      if (auto EC = consumeCString(Data, Name))
        return EC;
      Names.push_back(Name);
    }
    return VirtualTableRecord(L->CompleteClass, L->OverriddenVFTable,
                              L->VFPtrOffset, Name, Names);
  }

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
  OneMethodRecord(TypeIndex Type, MethodKind Kind, MethodOptions Options,
                  MemberAccess Access, int32_t VFTableOffset, StringRef Name)
      : TypeRecord(TypeRecordKind::OneMethod), Type(Type), Kind(Kind),
        Options(Options), Access(Access), VFTableOffset(VFTableOffset),
        Name(Name) {}

  static ErrorOr<OneMethodRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    MethodOptions Options = L->Attrs.getFlags();
    MethodKind MethKind = L->Attrs.getMethodKind();
    MemberAccess Access = L->Attrs.getAccess();
    int32_t VFTableOffset = 0;
    if (L->Attrs.isIntroducedVirtual()) {
      const little32_t *L;
      if (consumeObject(Data, L))
        return std::make_error_code(std::errc::illegal_byte_sequence);
      VFTableOffset = *L;
    }
    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    return OneMethodRecord(L->Type, MethKind, Options, Access, VFTableOffset,
                           Name);
  }

  TypeIndex getType() const { return Type; }
  MethodKind getKind() const { return Kind; }
  MethodOptions getOptions() const { return Options; }
  MemberAccess getAccess() const { return Access; }
  int32_t getVFTableOffset() const { return VFTableOffset; }
  StringRef getName() const { return Name; }

  bool isIntroducingVirtual() const {
    const uint8_t K = static_cast<uint8_t>(Kind);
    const uint8_t V = static_cast<uint8_t>(MethodKind::IntroducingVirtual);
    const uint8_t PV = static_cast<uint8_t>(MethodKind::PureIntroducingVirtual);
    return (K & V) || (K & PV);
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
class MethodListRecord : public TypeRecord {
public:
  MethodListRecord(TypeIndex Type, MethodKind Kind, MethodOptions Options,
                   MemberAccess Access, int32_t VFTableOffset)
      : TypeRecord(TypeRecordKind::MethodList), Type(Type), Kind(Kind),
        Options(Options), Access(Access), VFTableOffset(VFTableOffset) {}

  static ErrorOr<MethodListRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    MethodOptions Options = L->Attrs.getFlags();
    MethodKind MethKind = L->Attrs.getMethodKind();
    MemberAccess Access = L->Attrs.getAccess();
    int32_t VFTableOffset = 0;
    if (L->Attrs.isIntroducedVirtual()) {
      const little32_t *L;
      if (consumeObject(Data, L))
        return std::make_error_code(std::errc::illegal_byte_sequence);
      VFTableOffset = *L;
    }

    return MethodListRecord(L->Type, MethKind, Options, Access, VFTableOffset);
  }

  TypeIndex getType() const { return Type; }
  MethodKind getMethodKind() const { return Kind; }
  MethodOptions getOptions() const { return Options; }
  MemberAccess getAccess() const { return Access; }
  int32_t getVFTableOffset() const { return VFTableOffset; }

private:
  struct Layout {
    MemberAttributes Attrs;
    ulittle16_t Padding;

    TypeIndex Type;
    // If is introduced virtual method:
    //   VFTableOffset: int32_t offset in vftable
  };

  TypeIndex Type;
  MethodKind Kind;
  MethodOptions Options;
  MemberAccess Access;
  int32_t VFTableOffset;
};

/// For method overload sets.  LF_METHOD
class OverloadedMethodRecord : public TypeRecord {
public:
  OverloadedMethodRecord(uint16_t NumOverloads, TypeIndex MethodList,
                         StringRef Name)
      : TypeRecord(TypeRecordKind::OverloadedMethod),
        NumOverloads(NumOverloads), MethodList(MethodList), Name(Name) {}

  static ErrorOr<OverloadedMethodRecord> deserialize(TypeRecordKind Kind,
                                                     ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;
    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    return OverloadedMethodRecord(L->MethodCount, L->MethList, Name);
  }

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
  DataMemberRecord(MemberAccess Access, TypeIndex Type, uint64_t Offset,
                   StringRef Name)
      : TypeRecord(TypeRecordKind::Member), Access(Access), Type(Type),
        FieldOffset(Offset), Name(Name) {}

  static ErrorOr<DataMemberRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    uint64_t Offset;
    if (!decodeUIntLeaf(Data, Offset))
      return std::make_error_code(std::errc::illegal_byte_sequence);
    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    return DataMemberRecord(L->Attrs.getAccess(), L->Type, Offset, Name);
  }

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
  StaticDataMemberRecord(MemberAccess Access, TypeIndex Type, StringRef Name)
      : TypeRecord(TypeRecordKind::StaticMember), Access(Access), Type(Type),
        Name(Name) {}

  static ErrorOr<StaticDataMemberRecord> deserialize(TypeRecordKind Kind,
                                                     ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    return StaticDataMemberRecord(L->Attrs.getAccess(), L->Type, Name);
  }

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
  EnumeratorRecord(MemberAccess Access, APSInt Value, StringRef Name)
      : TypeRecord(TypeRecordKind::Enumerate), Access(Access), Value(Value),
        Name(Name) {}

  static ErrorOr<EnumeratorRecord> deserialize(TypeRecordKind Kind,
                                               ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    if (Data.empty())
      return std::make_error_code(std::errc::illegal_byte_sequence);
    APSInt Value;
    if (!decodeNumericLeaf(Data, Value))
      return std::make_error_code(std::errc::illegal_byte_sequence);

    StringRef Name;
    if (auto EC = consumeCString(Data, Name))
      return EC;

    return EnumeratorRecord(L->Attrs.getAccess(), Value, Name);
  }

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
class VirtualFunctionPointerRecord : public TypeRecord {
public:
  VirtualFunctionPointerRecord(TypeIndex Type)
      : TypeRecord(TypeRecordKind::VirtualFunctionTablePointer), Type(Type) {}
  static ErrorOr<VirtualFunctionPointerRecord>
  deserialize(TypeRecordKind Kind, ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    return VirtualFunctionPointerRecord(L->Type);
  }

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
  BaseClassRecord(MemberAccess Access, TypeIndex Type, uint64_t Offset)
      : TypeRecord(TypeRecordKind::BaseClass), Access(Access), Type(Type),
        Offset(Offset) {}

  static ErrorOr<BaseClassRecord> deserialize(TypeRecordKind Kind,
                                              ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    uint64_t Offset;
    if (!decodeUIntLeaf(Data, Offset))
      return std::make_error_code(std::errc::illegal_byte_sequence);

    return BaseClassRecord(L->Attrs.getAccess(), L->BaseType, Offset);
  }

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
  VirtualBaseClassRecord(MemberAccess Access, TypeIndex BaseType,
                         TypeIndex VBPtrType, uint64_t Offset, uint64_t Index)
      : TypeRecord(TypeRecordKind::VirtualBaseClass), Access(Access),
        BaseType(BaseType), VBPtrType(VBPtrType), VBPtrOffset(Offset),
        VTableIndex(Index) {}

  static ErrorOr<VirtualBaseClassRecord> deserialize(TypeRecordKind Kind,
                                                     ArrayRef<uint8_t> &Data) {
    const Layout *L = nullptr;
    if (auto EC = consumeObject(Data, L))
      return EC;

    uint64_t Offset;
    uint64_t Index;
    if (!decodeUIntLeaf(Data, Offset))
      return std::make_error_code(std::errc::illegal_byte_sequence);
    if (!decodeUIntLeaf(Data, Index))
      return std::make_error_code(std::errc::illegal_byte_sequence);

    return VirtualBaseClassRecord(L->Attrs.getAccess(), L->BaseType,
                                  L->VBPtrType, Offset, Index);
  }

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
}
}

#endif
