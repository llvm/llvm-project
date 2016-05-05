//===-- TypeDumper.cpp - CodeView type info dumper --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeStream.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::codeview;

/// The names here all end in "*". If the simple type is a pointer type, we
/// return the whole name. Otherwise we lop off the last character in our
/// StringRef.
static const EnumEntry<SimpleTypeKind> SimpleTypeNames[] = {
    {"void*", SimpleTypeKind::Void},
    {"<not translated>*", SimpleTypeKind::NotTranslated},
    {"HRESULT*", SimpleTypeKind::HResult},
    {"signed char*", SimpleTypeKind::SignedCharacter},
    {"unsigned char*", SimpleTypeKind::UnsignedCharacter},
    {"char*", SimpleTypeKind::NarrowCharacter},
    {"wchar_t*", SimpleTypeKind::WideCharacter},
    {"char16_t*", SimpleTypeKind::Character16},
    {"char32_t*", SimpleTypeKind::Character32},
    {"__int8*", SimpleTypeKind::SByte},
    {"unsigned __int8*", SimpleTypeKind::Byte},
    {"short*", SimpleTypeKind::Int16Short},
    {"unsigned short*", SimpleTypeKind::UInt16Short},
    {"__int16*", SimpleTypeKind::Int16},
    {"unsigned __int16*", SimpleTypeKind::UInt16},
    {"long*", SimpleTypeKind::Int32Long},
    {"unsigned long*", SimpleTypeKind::UInt32Long},
    {"int*", SimpleTypeKind::Int32},
    {"unsigned*", SimpleTypeKind::UInt32},
    {"__int64*", SimpleTypeKind::Int64Quad},
    {"unsigned __int64*", SimpleTypeKind::UInt64Quad},
    {"__int64*", SimpleTypeKind::Int64},
    {"unsigned __int64*", SimpleTypeKind::UInt64},
    {"__int128*", SimpleTypeKind::Int128},
    {"unsigned __int128*", SimpleTypeKind::UInt128},
    {"__half*", SimpleTypeKind::Float16},
    {"float*", SimpleTypeKind::Float32},
    {"float*", SimpleTypeKind::Float32PartialPrecision},
    {"__float48*", SimpleTypeKind::Float48},
    {"double*", SimpleTypeKind::Float64},
    {"long double*", SimpleTypeKind::Float80},
    {"__float128*", SimpleTypeKind::Float128},
    {"_Complex float*", SimpleTypeKind::Complex32},
    {"_Complex double*", SimpleTypeKind::Complex64},
    {"_Complex long double*", SimpleTypeKind::Complex80},
    {"_Complex __float128*", SimpleTypeKind::Complex128},
    {"bool*", SimpleTypeKind::Boolean8},
    {"__bool16*", SimpleTypeKind::Boolean16},
    {"__bool32*", SimpleTypeKind::Boolean32},
    {"__bool64*", SimpleTypeKind::Boolean64},
};

static const EnumEntry<TypeLeafKind> LeafTypeNames[] = {
#define LEAF_TYPE(enum, val) { #enum, enum },
#include "llvm/DebugInfo/CodeView/CVLeafTypes.def"
};

#define ENUM_ENTRY(enum_class, enum)                                           \
  { #enum, std::underlying_type < enum_class > ::type(enum_class::enum) }

static const EnumEntry<uint16_t> ClassOptionNames[] = {
    ENUM_ENTRY(ClassOptions, Packed),
    ENUM_ENTRY(ClassOptions, HasConstructorOrDestructor),
    ENUM_ENTRY(ClassOptions, HasOverloadedOperator),
    ENUM_ENTRY(ClassOptions, Nested),
    ENUM_ENTRY(ClassOptions, ContainsNestedClass),
    ENUM_ENTRY(ClassOptions, HasOverloadedAssignmentOperator),
    ENUM_ENTRY(ClassOptions, HasConversionOperator),
    ENUM_ENTRY(ClassOptions, ForwardReference),
    ENUM_ENTRY(ClassOptions, Scoped),
    ENUM_ENTRY(ClassOptions, HasUniqueName),
    ENUM_ENTRY(ClassOptions, Sealed),
    ENUM_ENTRY(ClassOptions, Intrinsic),
};

static const EnumEntry<uint8_t> MemberAccessNames[] = {
    ENUM_ENTRY(MemberAccess, None),
    ENUM_ENTRY(MemberAccess, Private),
    ENUM_ENTRY(MemberAccess, Protected),
    ENUM_ENTRY(MemberAccess, Public),
};

static const EnumEntry<uint16_t> MethodOptionNames[] = {
    ENUM_ENTRY(MethodOptions, Pseudo),
    ENUM_ENTRY(MethodOptions, NoInherit),
    ENUM_ENTRY(MethodOptions, NoConstruct),
    ENUM_ENTRY(MethodOptions, CompilerGenerated),
    ENUM_ENTRY(MethodOptions, Sealed),
};

static const EnumEntry<uint16_t> MemberKindNames[] = {
    ENUM_ENTRY(MethodKind, Vanilla),
    ENUM_ENTRY(MethodKind, Virtual),
    ENUM_ENTRY(MethodKind, Static),
    ENUM_ENTRY(MethodKind, Friend),
    ENUM_ENTRY(MethodKind, IntroducingVirtual),
    ENUM_ENTRY(MethodKind, PureVirtual),
    ENUM_ENTRY(MethodKind, PureIntroducingVirtual),
};

static const EnumEntry<uint8_t> PtrKindNames[] = {
    ENUM_ENTRY(PointerKind, Near16),
    ENUM_ENTRY(PointerKind, Far16),
    ENUM_ENTRY(PointerKind, Huge16),
    ENUM_ENTRY(PointerKind, BasedOnSegment),
    ENUM_ENTRY(PointerKind, BasedOnValue),
    ENUM_ENTRY(PointerKind, BasedOnSegmentValue),
    ENUM_ENTRY(PointerKind, BasedOnAddress),
    ENUM_ENTRY(PointerKind, BasedOnSegmentAddress),
    ENUM_ENTRY(PointerKind, BasedOnType),
    ENUM_ENTRY(PointerKind, BasedOnSelf),
    ENUM_ENTRY(PointerKind, Near32),
    ENUM_ENTRY(PointerKind, Far32),
    ENUM_ENTRY(PointerKind, Near64),
};

static const EnumEntry<uint8_t> PtrModeNames[] = {
    ENUM_ENTRY(PointerMode, Pointer),
    ENUM_ENTRY(PointerMode, LValueReference),
    ENUM_ENTRY(PointerMode, PointerToDataMember),
    ENUM_ENTRY(PointerMode, PointerToMemberFunction),
    ENUM_ENTRY(PointerMode, RValueReference),
};

static const EnumEntry<uint16_t> PtrMemberRepNames[] = {
    ENUM_ENTRY(PointerToMemberRepresentation, Unknown),
    ENUM_ENTRY(PointerToMemberRepresentation, SingleInheritanceData),
    ENUM_ENTRY(PointerToMemberRepresentation, MultipleInheritanceData),
    ENUM_ENTRY(PointerToMemberRepresentation, VirtualInheritanceData),
    ENUM_ENTRY(PointerToMemberRepresentation, GeneralData),
    ENUM_ENTRY(PointerToMemberRepresentation, SingleInheritanceFunction),
    ENUM_ENTRY(PointerToMemberRepresentation, MultipleInheritanceFunction),
    ENUM_ENTRY(PointerToMemberRepresentation, VirtualInheritanceFunction),
    ENUM_ENTRY(PointerToMemberRepresentation, GeneralFunction),
};

static const EnumEntry<uint16_t> TypeModifierNames[] = {
    ENUM_ENTRY(ModifierOptions, Const),
    ENUM_ENTRY(ModifierOptions, Volatile),
    ENUM_ENTRY(ModifierOptions, Unaligned),
};

static const EnumEntry<uint8_t> CallingConventions[] = {
    ENUM_ENTRY(CallingConvention, NearC),
    ENUM_ENTRY(CallingConvention, FarC),
    ENUM_ENTRY(CallingConvention, NearPascal),
    ENUM_ENTRY(CallingConvention, FarPascal),
    ENUM_ENTRY(CallingConvention, NearFast),
    ENUM_ENTRY(CallingConvention, FarFast),
    ENUM_ENTRY(CallingConvention, NearStdCall),
    ENUM_ENTRY(CallingConvention, FarStdCall),
    ENUM_ENTRY(CallingConvention, NearSysCall),
    ENUM_ENTRY(CallingConvention, FarSysCall),
    ENUM_ENTRY(CallingConvention, ThisCall),
    ENUM_ENTRY(CallingConvention, MipsCall),
    ENUM_ENTRY(CallingConvention, Generic),
    ENUM_ENTRY(CallingConvention, AlphaCall),
    ENUM_ENTRY(CallingConvention, PpcCall),
    ENUM_ENTRY(CallingConvention, SHCall),
    ENUM_ENTRY(CallingConvention, ArmCall),
    ENUM_ENTRY(CallingConvention, AM33Call),
    ENUM_ENTRY(CallingConvention, TriCall),
    ENUM_ENTRY(CallingConvention, SH5Call),
    ENUM_ENTRY(CallingConvention, M32RCall),
    ENUM_ENTRY(CallingConvention, ClrCall),
    ENUM_ENTRY(CallingConvention, Inline),
    ENUM_ENTRY(CallingConvention, NearVector),
};

static const EnumEntry<uint8_t> FunctionOptionEnum[] = {
    ENUM_ENTRY(FunctionOptions, CxxReturnUdt),
    ENUM_ENTRY(FunctionOptions, Constructor),
    ENUM_ENTRY(FunctionOptions, ConstructorWithVirtualBases),
};

#undef ENUM_ENTRY


namespace {

/// Use this private dumper implementation to keep implementation details about
/// the visitor out of TypeDumper.h.
class CVTypeDumperImpl : public CVTypeVisitor<CVTypeDumperImpl> {
public:
  CVTypeDumperImpl(CVTypeDumper &CVTD, ScopedPrinter &W, bool PrintRecordBytes)
      : CVTD(CVTD), W(W), PrintRecordBytes(PrintRecordBytes) {}

  /// CVTypeVisitor overrides.
  /// FIXME: Bury these in the .cc file to hide implementation details.
#define TYPE_RECORD(ClassName, LeafEnum)                                       \
  void visit##ClassName(TypeLeafKind LeafType, const ClassName *Record,        \
                        ArrayRef<uint8_t> LeafData);
#define TYPE_RECORD_ALIAS(ClassName, LeafEnum)
#define MEMBER_RECORD(ClassName, LeafEnum)                                     \
  void visit##ClassName(TypeLeafKind LeafType, const ClassName *Record,        \
                        ArrayRef<uint8_t> &FieldData);
#define MEMBER_RECORD_ALIAS(ClassName, LeafEnum)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

  /// Method overload lists are a special case.
  void visitMethodList(TypeLeafKind Leaf, ArrayRef<uint8_t> LeafData);

  void visitUnknownMember(TypeLeafKind Leaf);

  void visitTypeBegin(TypeLeafKind Leaf, ArrayRef<uint8_t> LeafData);
  void visitTypeEnd(TypeLeafKind Leaf, ArrayRef<uint8_t> LeafData);

  void printMemberAttributes(MemberAttributes Attrs);

private:
  /// Forwards to the dumper, which holds the persistent state from visitation.
  StringRef getTypeName(TypeIndex TI) {
    return CVTD.getTypeName(TI);
  }

  void printTypeIndex(StringRef FieldName, TypeIndex TI) {
    CVTD.printTypeIndex(FieldName, TI);
  }

  CVTypeDumper &CVTD;
  ScopedPrinter &W;
  bool PrintRecordBytes = false;

  /// Name of the current type. Only valid before visitTypeEnd.
  StringRef Name;
};

} // end anonymous namespace

/// Reinterpret a byte array as an array of characters. Does not interpret as
/// a C string, as StringRef has several helpers (split) that make that easy.
static StringRef getBytesAsCharacters(ArrayRef<uint8_t> LeafData) {
  return StringRef(reinterpret_cast<const char *>(LeafData.data()),
                   LeafData.size());
}

static StringRef getBytesAsCString(ArrayRef<uint8_t> LeafData) {
  return getBytesAsCharacters(LeafData).split('\0').first;
}

static StringRef getLeafTypeName(TypeLeafKind LT) {
  switch (LT) {
#define KNOWN_TYPE(LeafName, Value, ClassName) \
  case LeafName: return #ClassName;
#include "llvm/DebugInfo/CodeView/CVLeafTypes.def"
  default:
    break;
  }
  return "UnknownLeaf";
}

void CVTypeDumperImpl::visitTypeBegin(TypeLeafKind Leaf,
                                      ArrayRef<uint8_t> LeafData) {
  // Reset Name to the empty string. If the visitor sets it, we know it.
  Name = "";

  W.startLine() << getLeafTypeName(Leaf) << " {\n";
  W.indent();
  W.printEnum("TypeLeafKind", unsigned(Leaf), makeArrayRef(LeafTypeNames));
  W.printHex("TypeIndex", CVTD.getNextTypeIndex());
}

void CVTypeDumperImpl::visitTypeEnd(TypeLeafKind Leaf,
                                    ArrayRef<uint8_t> LeafData) {
  // Always record some name for every type, even if Name is empty. CVUDTNames
  // is indexed by type index, and must have one entry for every type.
  CVTD.recordType(Name);

  if (PrintRecordBytes)
    W.printBinaryBlock("LeafData", getBytesAsCharacters(LeafData));

  W.unindent();
  W.startLine() << "}\n";
}

void CVTypeDumperImpl::visitStringId(TypeLeafKind Leaf, const StringId *String,
                                     ArrayRef<uint8_t> LeafData) {
  W.printHex("Id", String->id.getIndex());
  StringRef StringData = getBytesAsCString(LeafData);
  W.printString("StringData", StringData);
  // Put this in CVUDTNames so it gets printed with LF_UDT_SRC_LINE.
  Name = StringData;
}

void CVTypeDumperImpl::visitArgList(TypeLeafKind Leaf, const ArgList *Args,
                                    ArrayRef<uint8_t> LeafData) {
  W.printNumber("NumArgs", Args->NumArgs);
  ListScope Arguments(W, "Arguments");
  SmallString<256> TypeName("(");
  for (uint32_t ArgI = 0; ArgI != Args->NumArgs; ++ArgI) {
    const TypeIndex *Type;
    if (!consumeObject(LeafData, Type))
      return;
    printTypeIndex("ArgType", *Type);
    StringRef ArgTypeName = getTypeName(*Type);
    TypeName.append(ArgTypeName);
    if (ArgI + 1 != Args->NumArgs)
      TypeName.append(", ");
  }
  TypeName.push_back(')');
  Name = CVTD.saveName(TypeName);
}

void CVTypeDumperImpl::visitClassType(TypeLeafKind Leaf, const ClassType *Class,
                                      ArrayRef<uint8_t> LeafData) {
  W.printNumber("MemberCount", Class->MemberCount);
  uint16_t Props = Class->Properties;
  W.printFlags("Properties", Props, makeArrayRef(ClassOptionNames));
  printTypeIndex("FieldList", Class->FieldList);
  printTypeIndex("DerivedFrom", Class->DerivedFrom);
  printTypeIndex("VShape", Class->VShape);
  uint64_t SizeOf;
  if (!decodeUIntLeaf(LeafData, SizeOf))
    return parseError();
  W.printNumber("SizeOf", SizeOf);
  StringRef LeafChars = getBytesAsCharacters(LeafData);
  StringRef LinkageName;
  std::tie(Name, LinkageName) = LeafChars.split('\0');
  W.printString("Name", Name);
  if (Props & uint16_t(ClassOptions::HasUniqueName)) {
    LinkageName = LinkageName.split('\0').first;
    if (LinkageName.empty())
      return parseError();
    W.printString("LinkageName", LinkageName);
  }
}

void CVTypeDumperImpl::visitUnionType(TypeLeafKind Leaf, const UnionType *Union,
                                      ArrayRef<uint8_t> LeafData) {
  W.printNumber("MemberCount", Union->MemberCount);
  uint16_t Props = Union->Properties;
  W.printFlags("Properties", Props, makeArrayRef(ClassOptionNames));
  printTypeIndex("FieldList", Union->FieldList);
  uint64_t SizeOf;
  if (!decodeUIntLeaf(LeafData, SizeOf))
    return parseError();
  W.printNumber("SizeOf", SizeOf);
  StringRef LeafChars = getBytesAsCharacters(LeafData);
  StringRef LinkageName;
  std::tie(Name, LinkageName) = LeafChars.split('\0');
  W.printString("Name", Name);
  if (Props & uint16_t(ClassOptions::HasUniqueName)) {
    LinkageName = LinkageName.split('\0').first;
    if (LinkageName.empty())
      return parseError();
    W.printString("LinkageName", LinkageName);
  }
}

void CVTypeDumperImpl::visitEnumType(TypeLeafKind Leaf, const EnumType *Enum,
                                 ArrayRef<uint8_t> LeafData) {
  W.printNumber("NumEnumerators", Enum->NumEnumerators);
  W.printFlags("Properties", uint16_t(Enum->Properties),
               makeArrayRef(ClassOptionNames));
  printTypeIndex("UnderlyingType", Enum->UnderlyingType);
  printTypeIndex("FieldListType", Enum->FieldListType);
  Name = getBytesAsCString(LeafData);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitArrayType(TypeLeafKind Leaf, const ArrayType *AT,
                                  ArrayRef<uint8_t> LeafData) {
  printTypeIndex("ElementType", AT->ElementType);
  printTypeIndex("IndexType", AT->IndexType);
  uint64_t SizeOf;
  if (!decodeUIntLeaf(LeafData, SizeOf))
    return parseError();
  W.printNumber("SizeOf", SizeOf);
  Name = getBytesAsCString(LeafData);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitVFTableType(TypeLeafKind Leaf, const VFTableType *VFT,
                                    ArrayRef<uint8_t> LeafData) {
  printTypeIndex("CompleteClass", VFT->CompleteClass);
  printTypeIndex("OverriddenVFTable", VFT->OverriddenVFTable);
  W.printHex("VFPtrOffset", VFT->VFPtrOffset);
  StringRef NamesData = getBytesAsCharacters(LeafData.slice(0, VFT->NamesLen));
  std::tie(Name, NamesData) = NamesData.split('\0');
  W.printString("VFTableName", Name);
  while (!NamesData.empty()) {
    StringRef MethodName;
    std::tie(MethodName, NamesData) = NamesData.split('\0');
    W.printString("MethodName", MethodName);
  }
}

void CVTypeDumperImpl::visitMemberFuncId(TypeLeafKind Leaf, const MemberFuncId *Id,
                                     ArrayRef<uint8_t> LeafData) {
  printTypeIndex("ClassType", Id->ClassType);
  printTypeIndex("FunctionType", Id->FunctionType);
  Name = getBytesAsCString(LeafData);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitProcedureType(TypeLeafKind Leaf,
                                      const ProcedureType *Proc,
                                      ArrayRef<uint8_t> LeafData) {
  printTypeIndex("ReturnType", Proc->ReturnType);
  W.printEnum("CallingConvention", uint8_t(Proc->CallConv),
              makeArrayRef(CallingConventions));
  W.printFlags("FunctionOptions", uint8_t(Proc->Options),
               makeArrayRef(FunctionOptionEnum));
  W.printNumber("NumParameters", Proc->NumParameters);
  printTypeIndex("ArgListType", Proc->ArgListType);

  StringRef ReturnTypeName = getTypeName(Proc->ReturnType);
  StringRef ArgListTypeName = getTypeName(Proc->ArgListType);
  SmallString<256> TypeName(ReturnTypeName);
  TypeName.push_back(' ');
  TypeName.append(ArgListTypeName);
  Name = CVTD.saveName(TypeName);
}

void CVTypeDumperImpl::visitMemberFunctionType(TypeLeafKind Leaf,
                                           const MemberFunctionType *MemberFunc,
                                           ArrayRef<uint8_t> LeafData) {
  printTypeIndex("ReturnType", MemberFunc->ReturnType);
  printTypeIndex("ClassType", MemberFunc->ClassType);
  printTypeIndex("ThisType", MemberFunc->ThisType);
  W.printEnum("CallingConvention", uint8_t(MemberFunc->CallConv),
              makeArrayRef(CallingConventions));
  W.printFlags("FunctionOptions", uint8_t(MemberFunc->Options),
               makeArrayRef(FunctionOptionEnum));
  W.printNumber("NumParameters", MemberFunc->NumParameters);
  printTypeIndex("ArgListType", MemberFunc->ArgListType);
  W.printNumber("ThisAdjustment", MemberFunc->ThisAdjustment);

  StringRef ReturnTypeName = getTypeName(MemberFunc->ReturnType);
  StringRef ClassTypeName = getTypeName(MemberFunc->ClassType);
  StringRef ArgListTypeName = getTypeName(MemberFunc->ArgListType);
  SmallString<256> TypeName(ReturnTypeName);
  TypeName.push_back(' ');
  TypeName.append(ClassTypeName);
  TypeName.append("::");
  TypeName.append(ArgListTypeName);
  Name = CVTD.saveName(TypeName);
}

void CVTypeDumperImpl::visitMethodList(TypeLeafKind Leaf,
                                   ArrayRef<uint8_t> LeafData) {
  while (!LeafData.empty()) {
    const MethodListEntry *Method;
    if (!consumeObject(LeafData, Method))
      return;
    ListScope S(W, "Method");
    printMemberAttributes(Method->Attrs);
    printTypeIndex("Type", Method->Type);
    if (Method->isIntroducedVirtual()) {
      const little32_t *VFTOffsetPtr;
      if (!consumeObject(LeafData, VFTOffsetPtr))
        return;
      W.printHex("VFTableOffset", *VFTOffsetPtr);
    }
  }
}

void CVTypeDumperImpl::visitFuncId(TypeLeafKind Leaf, const FuncId *Func,
                               ArrayRef<uint8_t> LeafData) {
  printTypeIndex("ParentScope", Func->ParentScope);
  printTypeIndex("FunctionType", Func->FunctionType);
  Name = getBytesAsCString(LeafData);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitTypeServer2(TypeLeafKind Leaf,
                                    const TypeServer2 *TypeServer,
                                    ArrayRef<uint8_t> LeafData) {
  W.printBinary("Signature", StringRef(TypeServer->Signature, 16));
  W.printNumber("Age", TypeServer->Age);
  Name = getBytesAsCString(LeafData);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitPointerType(TypeLeafKind Leaf, const PointerType *Ptr,
                                    ArrayRef<uint8_t> LeafData) {
  printTypeIndex("PointeeType", Ptr->PointeeType);
  W.printHex("PointerAttributes", Ptr->Attrs);
  W.printEnum("PtrType", unsigned(Ptr->getPtrKind()),
              makeArrayRef(PtrKindNames));
  W.printEnum("PtrMode", unsigned(Ptr->getPtrMode()),
              makeArrayRef(PtrModeNames));
  W.printNumber("IsFlat", Ptr->isFlat());
  W.printNumber("IsConst", Ptr->isConst());
  W.printNumber("IsVolatile", Ptr->isVolatile());
  W.printNumber("IsUnaligned", Ptr->isUnaligned());

  if (Ptr->isPointerToMember()) {
    const PointerToMemberTail *PMT;
    if (!consumeObject(LeafData, PMT))
      return;
    printTypeIndex("ClassType", PMT->ClassType);
    W.printEnum("Representation", PMT->Representation,
                makeArrayRef(PtrMemberRepNames));

    StringRef PointeeName = getTypeName(Ptr->PointeeType);
    StringRef ClassName = getTypeName(PMT->ClassType);
    SmallString<256> TypeName(PointeeName);
    TypeName.push_back(' ');
    TypeName.append(ClassName);
    TypeName.append("::*");
    Name = CVTD.saveName(TypeName);
  } else {
    W.printBinaryBlock("TailData", getBytesAsCharacters(LeafData));

    SmallString<256> TypeName;
    if (Ptr->isConst())
      TypeName.append("const ");
    if (Ptr->isVolatile())
      TypeName.append("volatile ");
    if (Ptr->isUnaligned())
      TypeName.append("__unaligned ");

    TypeName.append(getTypeName(Ptr->PointeeType));

    if (Ptr->getPtrMode() == PointerMode::LValueReference)
      TypeName.append("&");
    else if (Ptr->getPtrMode() == PointerMode::RValueReference)
      TypeName.append("&&");
    else if (Ptr->getPtrMode() == PointerMode::Pointer)
      TypeName.append("*");

    Name = CVTD.saveName(TypeName);
  }
}

void CVTypeDumperImpl::visitTypeModifier(TypeLeafKind Leaf, const TypeModifier *Mod,
                                     ArrayRef<uint8_t> LeafData) {
  printTypeIndex("ModifiedType", Mod->ModifiedType);
  W.printFlags("Modifiers", Mod->Modifiers, makeArrayRef(TypeModifierNames));

  StringRef ModifiedName = getTypeName(Mod->ModifiedType);
  SmallString<256> TypeName;
  if (Mod->Modifiers & uint16_t(ModifierOptions::Const))
    TypeName.append("const ");
  if (Mod->Modifiers & uint16_t(ModifierOptions::Volatile))
    TypeName.append("volatile ");
  if (Mod->Modifiers & uint16_t(ModifierOptions::Unaligned))
    TypeName.append("__unaligned ");
  TypeName.append(ModifiedName);
  Name = CVTD.saveName(TypeName);
}

void CVTypeDumperImpl::visitVTableShape(TypeLeafKind Leaf, const VTableShape *Shape,
                                    ArrayRef<uint8_t> LeafData) {
  unsigned VFEntryCount = Shape->VFEntryCount;
  W.printNumber("VFEntryCount", VFEntryCount);
  // We could print out whether the methods are near or far, but in practice
  // today everything is CV_VTS_near32, so it's just noise.
}

void CVTypeDumperImpl::visitUDTSrcLine(TypeLeafKind Leaf, const UDTSrcLine *Line,
                                   ArrayRef<uint8_t> LeafData) {
  printTypeIndex("UDT", Line->UDT);
  printTypeIndex("SourceFile", Line->SourceFile);
  W.printNumber("LineNumber", Line->LineNumber);
}

void CVTypeDumperImpl::visitBuildInfo(TypeLeafKind Leaf, const BuildInfo *Args,
                                  ArrayRef<uint8_t> LeafData) {
  W.printNumber("NumArgs", Args->NumArgs);

  ListScope Arguments(W, "Arguments");
  for (uint32_t ArgI = 0; ArgI != Args->NumArgs; ++ArgI) {
    const TypeIndex *Type;
    if (!consumeObject(LeafData, Type))
      return;
    printTypeIndex("ArgType", *Type);
  }
}

void CVTypeDumperImpl::printMemberAttributes(MemberAttributes Attrs) {
  W.printEnum("AccessSpecifier", uint8_t(Attrs.getAccess()),
              makeArrayRef(MemberAccessNames));
  auto MK = Attrs.getMethodKind();
  // Data members will be vanilla. Don't try to print a method kind for them.
  if (MK != MethodKind::Vanilla)
    W.printEnum("MethodKind", unsigned(MK), makeArrayRef(MemberKindNames));
  if (Attrs.getFlags() != MethodOptions::None) {
    W.printFlags("MethodOptions", unsigned(Attrs.getFlags()),
                 makeArrayRef(MethodOptionNames));
  }
}

void CVTypeDumperImpl::visitUnknownMember(TypeLeafKind Leaf) {
  W.printHex("UnknownMember", unsigned(Leaf));
}

void CVTypeDumperImpl::visitNestedType(TypeLeafKind Leaf, const NestedType *Nested,
                                   ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "NestedType");
  printTypeIndex("Type", Nested->Type);
  StringRef Name = getBytesAsCString(FieldData);
  FieldData = FieldData.drop_front(Name.size() + 1);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitOneMethod(TypeLeafKind Leaf, const OneMethod *Method,
                                  ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "OneMethod");
  printMemberAttributes(Method->Attrs);
  printTypeIndex("Type", Method->Type);
  // If virtual, then read the vftable offset.
  if (Method->isIntroducedVirtual()) {
    const little32_t *VFTOffsetPtr;
    if (!consumeObject(FieldData, VFTOffsetPtr))
      return;
    W.printHex("VFTableOffset", *VFTOffsetPtr);
  }
  StringRef Name = getBytesAsCString(FieldData);
  FieldData = FieldData.drop_front(Name.size() + 1);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitOverloadedMethod(TypeLeafKind Leaf,
                                         const OverloadedMethod *Method,
                                         ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "OverloadedMethod");
  W.printHex("MethodCount", Method->MethodCount);
  W.printHex("MethodListIndex", Method->MethList.getIndex());
  StringRef Name = getBytesAsCString(FieldData);
  FieldData = FieldData.drop_front(Name.size() + 1);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitDataMember(TypeLeafKind Leaf, const DataMember *Field,
                                   ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "DataMember");
  printMemberAttributes(Field->Attrs);
  printTypeIndex("Type", Field->Type);
  uint64_t FieldOffset;
  if (!decodeUIntLeaf(FieldData, FieldOffset))
    return parseError();
  W.printHex("FieldOffset", FieldOffset);
  StringRef Name = getBytesAsCString(FieldData);
  FieldData = FieldData.drop_front(Name.size() + 1);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitStaticDataMember(TypeLeafKind Leaf,
                                         const StaticDataMember *Field,
                                         ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "StaticDataMember");
  printMemberAttributes(Field->Attrs);
  printTypeIndex("Type", Field->Type);
  StringRef Name = getBytesAsCString(FieldData);
  FieldData = FieldData.drop_front(Name.size() + 1);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitVirtualFunctionPointer(
    TypeLeafKind Leaf, const VirtualFunctionPointer *VFTable,
    ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "VirtualFunctionPointer");
  printTypeIndex("Type", VFTable->Type);
}

void CVTypeDumperImpl::visitEnumerator(TypeLeafKind Leaf, const Enumerator *Enum,
                                   ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "Enumerator");
  printMemberAttributes(Enum->Attrs);
  APSInt EnumValue;
  if (!decodeNumericLeaf(FieldData, EnumValue))
    return parseError();
  W.printNumber("EnumValue", EnumValue);
  StringRef Name = getBytesAsCString(FieldData);
  FieldData = FieldData.drop_front(Name.size() + 1);
  W.printString("Name", Name);
}

void CVTypeDumperImpl::visitBaseClass(TypeLeafKind Leaf, const BaseClass *Base,
                                  ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "BaseClass");
  printMemberAttributes(Base->Attrs);
  printTypeIndex("BaseType", Base->BaseType);
  uint64_t BaseOffset;
  if (!decodeUIntLeaf(FieldData, BaseOffset))
    return parseError();
  W.printHex("BaseOffset", BaseOffset);
}

void CVTypeDumperImpl::visitVirtualBaseClass(TypeLeafKind Leaf,
                                         const VirtualBaseClass *Base,
                                         ArrayRef<uint8_t> &FieldData) {
  DictScope S(W, "VirtualBaseClass");
  printMemberAttributes(Base->Attrs);
  printTypeIndex("BaseType", Base->BaseType);
  printTypeIndex("VBPtrType", Base->VBPtrType);
  uint64_t VBPtrOffset, VBTableIndex;
  if (!decodeUIntLeaf(FieldData, VBPtrOffset))
    return parseError();
  if (!decodeUIntLeaf(FieldData, VBTableIndex))
    return parseError();
  W.printHex("VBPtrOffset", VBPtrOffset);
  W.printHex("VBTableIndex", VBTableIndex);
}

StringRef CVTypeDumper::getTypeName(TypeIndex TI) {
  if (TI.isNoType())
    return "<no type>";

  if (TI.isSimple()) {
    // This is a simple type.
    for (const auto &SimpleTypeName : SimpleTypeNames) {
      if (SimpleTypeName.Value == TI.getSimpleKind()) {
        if (TI.getSimpleMode() == SimpleTypeMode::Direct)
          return SimpleTypeName.Name.drop_back(1);
        // Otherwise, this is a pointer type. We gloss over the distinction
        // between near, far, 64, 32, etc, and just give a pointer type.
        return SimpleTypeName.Name;
      }
    }
    return "<unknown simple type>";
  }

  // User-defined type.
  StringRef UDTName;
  unsigned UDTIndex = TI.getIndex() - 0x1000;
  if (UDTIndex < CVUDTNames.size())
    return CVUDTNames[UDTIndex];

  return "<unknown UDT>";
}

void CVTypeDumper::printTypeIndex(StringRef FieldName, TypeIndex TI) {
  StringRef TypeName;
  if (!TI.isNoType())
    TypeName = getTypeName(TI);
  if (!TypeName.empty())
    W.printHex(FieldName, TypeName, TI.getIndex());
  else
    W.printHex(FieldName, TI.getIndex());
}

bool CVTypeDumper::dump(ArrayRef<uint8_t> Data) {
  CVTypeDumperImpl Dumper(*this, W, PrintRecordBytes);
  Dumper.visitTypeStream(Data);
  return !Dumper.hadError();
}
