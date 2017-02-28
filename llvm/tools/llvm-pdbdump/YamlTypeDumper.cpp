//===- YamlTypeDumper.cpp ------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YamlTypeDumper.h"
#include "PdbYaml.h"
#include "YamlSerializationContext.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeSerializer.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::yaml;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(TypeIndex)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint64_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(OneMethodRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(VFTableSlotKind)
LLVM_YAML_IS_SEQUENCE_VECTOR(StringRef)
LLVM_YAML_IS_SEQUENCE_VECTOR(CVType)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbTpiFieldListRecord)

namespace {
struct FieldListRecordSplitter : public TypeVisitorCallbacks {
public:
  explicit FieldListRecordSplitter(
      std::vector<llvm::pdb::yaml::PdbTpiFieldListRecord> &Records)
      : Records(Records) {}

#define TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVT, Name##Record &Record) override { \
    visitKnownMemberImpl(CVT);                                                 \
    return Error::success();                                                   \
  }
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  void visitKnownMemberImpl(CVMemberRecord &CVT) {
    llvm::pdb::yaml::PdbTpiFieldListRecord R;
    R.Record = CVT;
    Records.push_back(std::move(R));
  }

  std::vector<llvm::pdb::yaml::PdbTpiFieldListRecord> &Records;
};
}

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<PointerToMemberRepresentation> {
  static void enumeration(IO &IO, PointerToMemberRepresentation &Value) {
    IO.enumCase(Value, "Unknown", PointerToMemberRepresentation::Unknown);
    IO.enumCase(Value, "SingleInheritanceData",
                PointerToMemberRepresentation::SingleInheritanceData);
    IO.enumCase(Value, "MultipleInheritanceData",
                PointerToMemberRepresentation::MultipleInheritanceData);
    IO.enumCase(Value, "VirtualInheritanceData",
                PointerToMemberRepresentation::VirtualInheritanceData);
    IO.enumCase(Value, "GeneralData",
                PointerToMemberRepresentation::GeneralData);
    IO.enumCase(Value, "SingleInheritanceFunction",
                PointerToMemberRepresentation::SingleInheritanceFunction);
    IO.enumCase(Value, "MultipleInheritanceFunction",
                PointerToMemberRepresentation::MultipleInheritanceFunction);
    IO.enumCase(Value, "VirtualInheritanceFunction",
                PointerToMemberRepresentation::VirtualInheritanceFunction);
    IO.enumCase(Value, "GeneralFunction",
                PointerToMemberRepresentation::GeneralFunction);
  }
};

template <> struct ScalarEnumerationTraits<VFTableSlotKind> {
  static void enumeration(IO &IO, VFTableSlotKind &Kind) {
    IO.enumCase(Kind, "Near16", VFTableSlotKind::Near16);
    IO.enumCase(Kind, "Far16", VFTableSlotKind::Far16);
    IO.enumCase(Kind, "This", VFTableSlotKind::This);
    IO.enumCase(Kind, "Outer", VFTableSlotKind::Outer);
    IO.enumCase(Kind, "Meta", VFTableSlotKind::Meta);
    IO.enumCase(Kind, "Near", VFTableSlotKind::Near);
    IO.enumCase(Kind, "Far", VFTableSlotKind::Far);
  }
};

template <> struct ScalarEnumerationTraits<CallingConvention> {
  static void enumeration(IO &IO, CallingConvention &Value) {
    IO.enumCase(Value, "NearC", CallingConvention::NearC);
    IO.enumCase(Value, "FarC", CallingConvention::FarC);
    IO.enumCase(Value, "NearPascal", CallingConvention::NearPascal);
    IO.enumCase(Value, "FarPascal", CallingConvention::FarPascal);
    IO.enumCase(Value, "NearFast", CallingConvention::NearFast);
    IO.enumCase(Value, "FarFast", CallingConvention::FarFast);
    IO.enumCase(Value, "NearStdCall", CallingConvention::NearStdCall);
    IO.enumCase(Value, "FarStdCall", CallingConvention::FarStdCall);
    IO.enumCase(Value, "NearSysCall", CallingConvention::NearSysCall);
    IO.enumCase(Value, "FarSysCall", CallingConvention::FarSysCall);
    IO.enumCase(Value, "ThisCall", CallingConvention::ThisCall);
    IO.enumCase(Value, "MipsCall", CallingConvention::MipsCall);
    IO.enumCase(Value, "Generic", CallingConvention::Generic);
    IO.enumCase(Value, "AlphaCall", CallingConvention::AlphaCall);
    IO.enumCase(Value, "PpcCall", CallingConvention::PpcCall);
    IO.enumCase(Value, "SHCall", CallingConvention::SHCall);
    IO.enumCase(Value, "ArmCall", CallingConvention::ArmCall);
    IO.enumCase(Value, "AM33Call", CallingConvention::AM33Call);
    IO.enumCase(Value, "TriCall", CallingConvention::TriCall);
    IO.enumCase(Value, "SH5Call", CallingConvention::SH5Call);
    IO.enumCase(Value, "M32RCall", CallingConvention::M32RCall);
    IO.enumCase(Value, "ClrCall", CallingConvention::ClrCall);
    IO.enumCase(Value, "Inline", CallingConvention::Inline);
    IO.enumCase(Value, "NearVector", CallingConvention::NearVector);
  }
};

template <> struct ScalarEnumerationTraits<PointerKind> {
  static void enumeration(IO &IO, PointerKind &Kind) {
    IO.enumCase(Kind, "Near16", PointerKind::Near16);
    IO.enumCase(Kind, "Far16", PointerKind::Far16);
    IO.enumCase(Kind, "Huge16", PointerKind::Huge16);
    IO.enumCase(Kind, "BasedOnSegment", PointerKind::BasedOnSegment);
    IO.enumCase(Kind, "BasedOnValue", PointerKind::BasedOnValue);
    IO.enumCase(Kind, "BasedOnSegmentValue", PointerKind::BasedOnSegmentValue);
    IO.enumCase(Kind, "BasedOnAddress", PointerKind::BasedOnAddress);
    IO.enumCase(Kind, "BasedOnSegmentAddress",
                PointerKind::BasedOnSegmentAddress);
    IO.enumCase(Kind, "BasedOnType", PointerKind::BasedOnType);
    IO.enumCase(Kind, "BasedOnSelf", PointerKind::BasedOnSelf);
    IO.enumCase(Kind, "Near32", PointerKind::Near32);
    IO.enumCase(Kind, "Far32", PointerKind::Far32);
    IO.enumCase(Kind, "Near64", PointerKind::Near64);
  }
};

template <> struct ScalarEnumerationTraits<PointerMode> {
  static void enumeration(IO &IO, PointerMode &Mode) {
    IO.enumCase(Mode, "Pointer", PointerMode::Pointer);
    IO.enumCase(Mode, "LValueReference", PointerMode::LValueReference);
    IO.enumCase(Mode, "PointerToDataMember", PointerMode::PointerToDataMember);
    IO.enumCase(Mode, "PointerToMemberFunction",
                PointerMode::PointerToMemberFunction);
    IO.enumCase(Mode, "RValueReference", PointerMode::RValueReference);
  }
};

template <> struct ScalarEnumerationTraits<HfaKind> {
  static void enumeration(IO &IO, HfaKind &Value) {
    IO.enumCase(Value, "None", HfaKind::None);
    IO.enumCase(Value, "Float", HfaKind::Float);
    IO.enumCase(Value, "Double", HfaKind::Double);
    IO.enumCase(Value, "Other", HfaKind::Other);
  }
};

template <> struct ScalarEnumerationTraits<MemberAccess> {
  static void enumeration(IO &IO, MemberAccess &Access) {
    IO.enumCase(Access, "None", MemberAccess::None);
    IO.enumCase(Access, "Private", MemberAccess::Private);
    IO.enumCase(Access, "Protected", MemberAccess::Protected);
    IO.enumCase(Access, "Public", MemberAccess::Public);
  }
};

template <> struct ScalarEnumerationTraits<MethodKind> {
  static void enumeration(IO &IO, MethodKind &Kind) {
    IO.enumCase(Kind, "Vanilla", MethodKind::Vanilla);
    IO.enumCase(Kind, "Virtual", MethodKind::Virtual);
    IO.enumCase(Kind, "Static", MethodKind::Static);
    IO.enumCase(Kind, "Friend", MethodKind::Friend);
    IO.enumCase(Kind, "IntroducingVirtual", MethodKind::IntroducingVirtual);
    IO.enumCase(Kind, "PureVirtual", MethodKind::PureVirtual);
    IO.enumCase(Kind, "PureIntroducingVirtual",
                MethodKind::PureIntroducingVirtual);
  }
};

template <> struct ScalarEnumerationTraits<WindowsRTClassKind> {
  static void enumeration(IO &IO, WindowsRTClassKind &Value) {
    IO.enumCase(Value, "None", WindowsRTClassKind::None);
    IO.enumCase(Value, "Ref", WindowsRTClassKind::RefClass);
    IO.enumCase(Value, "Value", WindowsRTClassKind::ValueClass);
    IO.enumCase(Value, "Interface", WindowsRTClassKind::Interface);
  }
};

template <> struct ScalarBitSetTraits<PointerOptions> {
  static void bitset(IO &IO, PointerOptions &Options) {
    IO.bitSetCase(Options, "None", PointerOptions::None);
    IO.bitSetCase(Options, "Flat32", PointerOptions::Flat32);
    IO.bitSetCase(Options, "Volatile", PointerOptions::Volatile);
    IO.bitSetCase(Options, "Const", PointerOptions::Const);
    IO.bitSetCase(Options, "Unaligned", PointerOptions::Unaligned);
    IO.bitSetCase(Options, "Restrict", PointerOptions::Restrict);
    IO.bitSetCase(Options, "WinRTSmartPointer",
                  PointerOptions::WinRTSmartPointer);
  }
};

template <> struct ScalarBitSetTraits<ModifierOptions> {
  static void bitset(IO &IO, ModifierOptions &Options) {
    IO.bitSetCase(Options, "None", ModifierOptions::None);
    IO.bitSetCase(Options, "Const", ModifierOptions::Const);
    IO.bitSetCase(Options, "Volatile", ModifierOptions::Volatile);
    IO.bitSetCase(Options, "Unaligned", ModifierOptions::Unaligned);
  }
};

template <> struct ScalarBitSetTraits<FunctionOptions> {
  static void bitset(IO &IO, FunctionOptions &Options) {
    IO.bitSetCase(Options, "None", FunctionOptions::None);
    IO.bitSetCase(Options, "CxxReturnUdt", FunctionOptions::CxxReturnUdt);
    IO.bitSetCase(Options, "Constructor", FunctionOptions::Constructor);
    IO.bitSetCase(Options, "ConstructorWithVirtualBases",
                  FunctionOptions::ConstructorWithVirtualBases);
  }
};

template <> struct ScalarBitSetTraits<ClassOptions> {
  static void bitset(IO &IO, ClassOptions &Options) {
    IO.bitSetCase(Options, "None", ClassOptions::None);
    IO.bitSetCase(Options, "HasConstructorOrDestructor",
                  ClassOptions::HasConstructorOrDestructor);
    IO.bitSetCase(Options, "HasOverloadedOperator",
                  ClassOptions::HasOverloadedOperator);
    IO.bitSetCase(Options, "Nested", ClassOptions::Nested);
    IO.bitSetCase(Options, "ContainsNestedClass",
                  ClassOptions::ContainsNestedClass);
    IO.bitSetCase(Options, "HasOverloadedAssignmentOperator",
                  ClassOptions::HasOverloadedAssignmentOperator);
    IO.bitSetCase(Options, "HasConversionOperator",
                  ClassOptions::HasConversionOperator);
    IO.bitSetCase(Options, "ForwardReference", ClassOptions::ForwardReference);
    IO.bitSetCase(Options, "Scoped", ClassOptions::Scoped);
    IO.bitSetCase(Options, "HasUniqueName", ClassOptions::HasUniqueName);
    IO.bitSetCase(Options, "Sealed", ClassOptions::Sealed);
    IO.bitSetCase(Options, "Intrinsic", ClassOptions::Intrinsic);
  }
};

template <> struct ScalarBitSetTraits<MethodOptions> {
  static void bitset(IO &IO, MethodOptions &Options) {
    IO.bitSetCase(Options, "None", MethodOptions::None);
    IO.bitSetCase(Options, "Pseudo", MethodOptions::Pseudo);
    IO.bitSetCase(Options, "NoInherit", MethodOptions::NoInherit);
    IO.bitSetCase(Options, "NoConstruct", MethodOptions::NoConstruct);
    IO.bitSetCase(Options, "CompilerGenerated",
                  MethodOptions::CompilerGenerated);
    IO.bitSetCase(Options, "Sealed", MethodOptions::Sealed);
  }
};

void ScalarTraits<APSInt>::output(const APSInt &S, void *,
                                  llvm::raw_ostream &OS) {
  S.print(OS, true);
}
StringRef ScalarTraits<APSInt>::input(StringRef Scalar, void *Ctx, APSInt &S) {
  S = APSInt(Scalar);
  return "";
}

bool ScalarTraits<APSInt>::mustQuote(StringRef Scalar) { return false; }

void MappingContextTraits<CVType, pdb::yaml::SerializationContext>::mapping(
    IO &IO, CVType &Record, pdb::yaml::SerializationContext &Context) {
  if (IO.outputting()) {
    codeview::TypeDeserializer Deserializer;

    codeview::TypeVisitorCallbackPipeline Pipeline;
    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Context.Dumper);

    codeview::CVTypeVisitor Visitor(Pipeline);
    consumeError(Visitor.visitTypeRecord(Record));
  }
}

void MappingTraits<StringIdRecord>::mapping(IO &IO, StringIdRecord &String) {
  IO.mapRequired("Id", String.Id);
  IO.mapRequired("String", String.String);
}

void MappingTraits<ArgListRecord>::mapping(IO &IO, ArgListRecord &Args) {
  IO.mapRequired("ArgIndices", Args.StringIndices);
}

void MappingTraits<ClassRecord>::mapping(IO &IO, ClassRecord &Class) {
  IO.mapRequired("MemberCount", Class.MemberCount);
  IO.mapRequired("Options", Class.Options);
  IO.mapRequired("FieldList", Class.FieldList);
  IO.mapRequired("Name", Class.Name);
  IO.mapRequired("UniqueName", Class.UniqueName);
  IO.mapRequired("DerivationList", Class.DerivationList);
  IO.mapRequired("VTableShape", Class.VTableShape);
  IO.mapRequired("Size", Class.Size);
}

void MappingTraits<UnionRecord>::mapping(IO &IO, UnionRecord &Union) {
  IO.mapRequired("MemberCount", Union.MemberCount);
  IO.mapRequired("Options", Union.Options);
  IO.mapRequired("FieldList", Union.FieldList);
  IO.mapRequired("Name", Union.Name);
  IO.mapRequired("UniqueName", Union.UniqueName);
  IO.mapRequired("Size", Union.Size);
}

void MappingTraits<EnumRecord>::mapping(IO &IO, EnumRecord &Enum) {
  IO.mapRequired("NumEnumerators", Enum.MemberCount);
  IO.mapRequired("Options", Enum.Options);
  IO.mapRequired("FieldList", Enum.FieldList);
  IO.mapRequired("Name", Enum.Name);
  IO.mapRequired("UniqueName", Enum.UniqueName);
  IO.mapRequired("UnderlyingType", Enum.UnderlyingType);
}

void MappingTraits<ArrayRecord>::mapping(IO &IO, ArrayRecord &AT) {
  IO.mapRequired("ElementType", AT.ElementType);
  IO.mapRequired("IndexType", AT.IndexType);
  IO.mapRequired("Size", AT.Size);
  IO.mapRequired("Name", AT.Name);
}

void MappingTraits<VFTableRecord>::mapping(IO &IO, VFTableRecord &VFT) {
  IO.mapRequired("CompleteClass", VFT.CompleteClass);
  IO.mapRequired("OverriddenVFTable", VFT.OverriddenVFTable);
  IO.mapRequired("VFPtrOffset", VFT.VFPtrOffset);
  IO.mapRequired("MethodNames", VFT.MethodNames);
}

void MappingTraits<MemberFuncIdRecord>::mapping(IO &IO,
                                                MemberFuncIdRecord &Id) {
  IO.mapRequired("ClassType", Id.ClassType);
  IO.mapRequired("FunctionType", Id.FunctionType);
  IO.mapRequired("Name", Id.Name);
}

void MappingTraits<ProcedureRecord>::mapping(IO &IO, ProcedureRecord &Proc) {
  IO.mapRequired("ReturnType", Proc.ReturnType);
  IO.mapRequired("CallConv", Proc.CallConv);
  IO.mapRequired("Options", Proc.Options);
  IO.mapRequired("ParameterCount", Proc.ParameterCount);
  IO.mapRequired("ArgumentList", Proc.ArgumentList);
}

void MappingTraits<MemberFunctionRecord>::mapping(IO &IO,
                                                  MemberFunctionRecord &MF) {
  IO.mapRequired("ReturnType", MF.ReturnType);
  IO.mapRequired("ClassType", MF.ClassType);
  IO.mapRequired("ThisType", MF.ThisType);
  IO.mapRequired("CallConv", MF.CallConv);
  IO.mapRequired("Options", MF.Options);
  IO.mapRequired("ParameterCount", MF.ParameterCount);
  IO.mapRequired("ArgumentList", MF.ArgumentList);
  IO.mapRequired("ThisPointerAdjustment", MF.ThisPointerAdjustment);
}

void MappingTraits<MethodOverloadListRecord>::mapping(
    IO &IO, MethodOverloadListRecord &MethodList) {
  IO.mapRequired("Methods", MethodList.Methods);
}

void MappingTraits<FuncIdRecord>::mapping(IO &IO, FuncIdRecord &Func) {
  IO.mapRequired("ParentScope", Func.ParentScope);
  IO.mapRequired("FunctionType", Func.FunctionType);
  IO.mapRequired("Name", Func.Name);
}

void MappingTraits<TypeServer2Record>::mapping(IO &IO, TypeServer2Record &TS) {
  IO.mapRequired("Guid", TS.Guid);
  IO.mapRequired("Age", TS.Age);
  IO.mapRequired("Name", TS.Name);
}

void MappingTraits<PointerRecord>::mapping(IO &IO, PointerRecord &Ptr) {
  IO.mapRequired("ReferentType", Ptr.ReferentType);
  IO.mapRequired("Attrs", Ptr.Attrs);
  IO.mapOptional("MemberInfo", Ptr.MemberInfo);
}

void MappingTraits<MemberPointerInfo>::mapping(IO &IO, MemberPointerInfo &MPI) {
  IO.mapRequired("ContainingType", MPI.ContainingType);
  IO.mapRequired("Representation", MPI.Representation);
}

void MappingTraits<ModifierRecord>::mapping(IO &IO, ModifierRecord &Mod) {
  IO.mapRequired("ModifiedType", Mod.ModifiedType);
  IO.mapRequired("Modifiers", Mod.Modifiers);
}

void MappingTraits<BitFieldRecord>::mapping(IO &IO, BitFieldRecord &BitField) {
  IO.mapRequired("Type", BitField.Type);
  IO.mapRequired("BitSize", BitField.BitSize);
  IO.mapRequired("BitOffset", BitField.BitOffset);
}

void MappingTraits<VFTableShapeRecord>::mapping(IO &IO,
                                                VFTableShapeRecord &Shape) {
  IO.mapRequired("Slots", Shape.Slots);
}

void MappingTraits<UdtSourceLineRecord>::mapping(IO &IO,
                                                 UdtSourceLineRecord &Line) {
  IO.mapRequired("UDT", Line.UDT);
  IO.mapRequired("SourceFile", Line.SourceFile);
  IO.mapRequired("LineNumber", Line.LineNumber);
}

void MappingTraits<UdtModSourceLineRecord>::mapping(
    IO &IO, UdtModSourceLineRecord &Line) {
  IO.mapRequired("UDT", Line.UDT);
  IO.mapRequired("SourceFile", Line.SourceFile);
  IO.mapRequired("LineNumber", Line.LineNumber);
  IO.mapRequired("Module", Line.Module);
}

void MappingTraits<BuildInfoRecord>::mapping(IO &IO, BuildInfoRecord &Args) {
  IO.mapRequired("ArgIndices", Args.ArgIndices);
}

void MappingTraits<NestedTypeRecord>::mapping(IO &IO,
                                              NestedTypeRecord &Nested) {
  IO.mapRequired("Type", Nested.Type);
  IO.mapRequired("Name", Nested.Name);
}

void MappingTraits<OneMethodRecord>::mapping(IO &IO, OneMethodRecord &Method) {
  IO.mapRequired("Type", Method.Type);
  IO.mapRequired("Attrs", Method.Attrs.Attrs);
  IO.mapRequired("VFTableOffset", Method.VFTableOffset);
  IO.mapRequired("Name", Method.Name);
}

void MappingTraits<OverloadedMethodRecord>::mapping(
    IO &IO, OverloadedMethodRecord &Method) {
  IO.mapRequired("NumOverloads", Method.NumOverloads);
  IO.mapRequired("MethodList", Method.MethodList);
  IO.mapRequired("Name", Method.Name);
}

void MappingTraits<DataMemberRecord>::mapping(IO &IO, DataMemberRecord &Field) {
  IO.mapRequired("Attrs", Field.Attrs.Attrs);
  IO.mapRequired("Type", Field.Type);
  IO.mapRequired("FieldOffset", Field.FieldOffset);
  IO.mapRequired("Name", Field.Name);
}

void MappingTraits<StaticDataMemberRecord>::mapping(
    IO &IO, StaticDataMemberRecord &Field) {
  IO.mapRequired("Attrs", Field.Attrs.Attrs);
  IO.mapRequired("Type", Field.Type);
  IO.mapRequired("Name", Field.Name);
}

void MappingTraits<VFPtrRecord>::mapping(IO &IO, VFPtrRecord &VFTable) {
  IO.mapRequired("Type", VFTable.Type);
}

void MappingTraits<EnumeratorRecord>::mapping(IO &IO, EnumeratorRecord &Enum) {
  IO.mapRequired("Attrs", Enum.Attrs.Attrs);
  IO.mapRequired("Value", Enum.Value);
  IO.mapRequired("Name", Enum.Name);
}

void MappingTraits<BaseClassRecord>::mapping(IO &IO, BaseClassRecord &Base) {
  IO.mapRequired("Attrs", Base.Attrs.Attrs);
  IO.mapRequired("Type", Base.Type);
  IO.mapRequired("Offset", Base.Offset);
}

void MappingTraits<VirtualBaseClassRecord>::mapping(
    IO &IO, VirtualBaseClassRecord &Base) {
  IO.mapRequired("Attrs", Base.Attrs.Attrs);
  IO.mapRequired("BaseType", Base.BaseType);
  IO.mapRequired("VBPtrType", Base.VBPtrType);
  IO.mapRequired("VBPtrOffset", Base.VBPtrOffset);
  IO.mapRequired("VTableIndex", Base.VTableIndex);
}

void MappingTraits<ListContinuationRecord>::mapping(
    IO &IO, ListContinuationRecord &Cont) {
  IO.mapRequired("ContinuationIndex", Cont.ContinuationIndex);
}

void ScalarTraits<codeview::TypeIndex>::output(const codeview::TypeIndex &S,
                                               void *, llvm::raw_ostream &OS) {
  OS << S.getIndex();
}
StringRef ScalarTraits<codeview::TypeIndex>::input(StringRef Scalar, void *Ctx,
                                                   codeview::TypeIndex &S) {
  uint32_t I;
  StringRef Result = ScalarTraits<uint32_t>::input(Scalar, Ctx, I);
  if (!Result.empty())
    return Result;
  S = TypeIndex(I);
  return "";
}
bool ScalarTraits<codeview::TypeIndex>::mustQuote(StringRef Scalar) {
  return false;
}

void ScalarEnumerationTraits<TypeLeafKind>::enumeration(IO &io,
                                                        TypeLeafKind &Value) {
  auto TypeLeafNames = getTypeLeafNames();
  for (const auto &E : TypeLeafNames)
    io.enumCase(Value, E.Name.str().c_str(), E.Value);
}
}
}

Error llvm::codeview::yaml::YamlTypeDumperCallbacks::visitTypeBegin(
    CVType &CVR) {
  YamlIO.mapRequired("Kind", CVR.Type);
  return Error::success();
}

Error llvm::codeview::yaml::YamlTypeDumperCallbacks::visitMemberBegin(
    CVMemberRecord &Record) {
  YamlIO.mapRequired("Kind", Record.Kind);
  return Error::success();
}

void llvm::codeview::yaml::YamlTypeDumperCallbacks::visitKnownRecordImpl(
    const char *Name, CVType &CVR, FieldListRecord &FieldList) {
  std::vector<llvm::pdb::yaml::PdbTpiFieldListRecord> FieldListRecords;
  if (YamlIO.outputting()) {
    // If we are outputting, then `FieldList.Data` contains a huge chunk of data
    // representing the serialized list of members.  We need to split it up into
    // individual CVType records where each record represents an individual
    // member.  This way, we can simply map the entire thing as a Yaml sequence,
    // which will recurse back to the standard handler for top-level fields
    // (top-level and member fields all have the exact same Yaml syntax so use
    // the same parser).
    FieldListRecordSplitter Splitter(FieldListRecords);
    CVTypeVisitor V(Splitter);
    consumeError(V.visitFieldListMemberStream(FieldList.Data));
    YamlIO.mapRequired("FieldList", FieldListRecords, Context);
  } else {
    // If we are not outputting, then the array contains no data starting out,
    // and is instead populated from the sequence represented by the yaml --
    // again, using the same logic that we use for top-level records.
    assert(Context.ActiveSerializer && "There is no active serializer!");
    codeview::TypeVisitorCallbackPipeline Pipeline;
    pdb::TpiHashUpdater Hasher;

    // For Yaml to PDB, dump it (to fill out the record fields from the Yaml)
    // then serialize those fields to bytes, then update their hashes.
    Pipeline.addCallbackToPipeline(Context.Dumper);
    Pipeline.addCallbackToPipeline(*Context.ActiveSerializer);
    Pipeline.addCallbackToPipeline(Hasher);

    codeview::CVTypeVisitor Visitor(Pipeline);
    YamlIO.mapRequired("FieldList", FieldListRecords, Visitor);
  }
}

namespace llvm {
namespace yaml {
template <>
struct MappingContextTraits<pdb::yaml::PdbTpiFieldListRecord,
                            pdb::yaml::SerializationContext> {
  static void mapping(IO &IO, pdb::yaml::PdbTpiFieldListRecord &Obj,
                      pdb::yaml::SerializationContext &Context) {
    assert(IO.outputting());
    codeview::TypeVisitorCallbackPipeline Pipeline;

    BinaryByteStream Data(Obj.Record.Data, llvm::support::little);
    BinaryStreamReader FieldReader(Data);
    codeview::FieldListDeserializer Deserializer(FieldReader);

    // For PDB to Yaml, deserialize into a high level record type, then dump
    // it.
    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Context.Dumper);

    codeview::CVTypeVisitor Visitor(Pipeline);
    consumeError(Visitor.visitMemberRecord(Obj.Record));
  }
};

template <>
struct MappingContextTraits<pdb::yaml::PdbTpiFieldListRecord,
                            codeview::CVTypeVisitor> {
  static void mapping(IO &IO, pdb::yaml::PdbTpiFieldListRecord &Obj,
                      codeview::CVTypeVisitor &Visitor) {
    consumeError(Visitor.visitMemberRecord(Obj.Record));
  }
};
}
}
