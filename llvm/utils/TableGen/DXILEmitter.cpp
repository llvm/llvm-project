//===- DXILEmitter.cpp - DXIL operation Emitter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DXILEmitter uses the descriptions of DXIL operation to construct enum and
// helper functions for DXIL operation.
//
//===----------------------------------------------------------------------===//

#include "SequenceToOffsetTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/DXILABI.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {

struct DXILShaderModel {
  int Major = 0;
  int Minor = 0;
};

struct DXILParam {
  int Pos; // position in parameter list
  ParameterKind Kind;
  StringRef Name; // short, unique name
  StringRef Doc;  // the documentation description of this parameter
  bool IsConst;   // whether this argument requires a constant value in the IR
  StringRef EnumName; // the name of the enum type if applicable
  int MaxValue;       // the maximum value for this parameter if applicable
  DXILParam(const Record *R);
};

struct DXILOperationData {
  StringRef Name; // short, unique name

  StringRef DXILOp;    // name of DXIL operation
  int DXILOpID;        // ID of DXIL operation
  StringRef DXILClass; // name of the opcode class
  StringRef Category;  // classification for this instruction
  StringRef Doc;       // the documentation description of this instruction

  SmallVector<DXILParam> Params; // the operands that this instruction takes
  StringRef OverloadTypes;       // overload types if applicable
  StringRef FnAttr;              // attribute shorthands: rn=does not access
                                 // memory,ro=only reads from memory
  StringRef Intrinsic; // The llvm intrinsic map to DXILOp. Default is "" which
                       // means no map exist
  bool IsDeriv = false;    // whether this is some kind of derivative
  bool IsGradient = false; // whether this requires a gradient calculation
  bool IsFeedback = false; // whether this is a sampler feedback op
  bool IsWave = false;     // whether this requires in-wave, cross-lane functionality
  bool RequiresUniformInputs = false; // whether this operation requires that
                                      // all of its inputs are uniform across
                                      // the wave
  SmallVector<StringRef, 4>
      ShaderStages; // shader stages to which this applies, empty for all.
  DXILShaderModel ShaderModel;           // minimum shader model required
  DXILShaderModel ShaderModelTranslated; // minimum shader model required with
                                         // translation by linker
  int OverloadParamIndex; // parameter index which control the overload.
                          // When < 0, should be only 1 overload type.
  SmallVector<StringRef, 4> counters; // counters for this inst.
  DXILOperationData(const Record *R) {
    Name = R->getValueAsString("name");
    DXILOp = R->getValueAsString("dxil_op");
    DXILOpID = R->getValueAsInt("dxil_opid");
    DXILClass = R->getValueAsDef("op_class")->getValueAsString("name");
    Category = R->getValueAsDef("category")->getValueAsString("name");

    if (R->getValue("llvm_intrinsic")) {
      auto *IntrinsicDef = R->getValueAsDef("llvm_intrinsic");
      auto DefName = IntrinsicDef->getName();
      assert(DefName.starts_with("int_") && "invalid intrinsic name");
      // Remove the int_ from intrinsic name.
      Intrinsic = DefName.substr(4);
    }

    Doc = R->getValueAsString("doc");

    ListInit *ParamList = R->getValueAsListInit("ops");
    OverloadParamIndex = -1;
    for (unsigned I = 0; I < ParamList->size(); ++I) {
      Record *Param = ParamList->getElementAsRecord(I);
      Params.emplace_back(DXILParam(Param));
      auto &CurParam = Params.back();
      if (CurParam.Kind >= ParameterKind::OVERLOAD)
        OverloadParamIndex = I;
    }
    OverloadTypes = R->getValueAsString("oload_types");
    FnAttr = R->getValueAsString("fn_attr");
  }
};
} // end anonymous namespace

static ParameterKind parameterTypeNameToKind(StringRef Name) {
  return StringSwitch<ParameterKind>(Name)
      .Case("void", ParameterKind::VOID)
      .Case("half", ParameterKind::HALF)
      .Case("float", ParameterKind::FLOAT)
      .Case("double", ParameterKind::DOUBLE)
      .Case("i1", ParameterKind::I1)
      .Case("i8", ParameterKind::I8)
      .Case("i16", ParameterKind::I16)
      .Case("i32", ParameterKind::I32)
      .Case("i64", ParameterKind::I64)
      .Case("$o", ParameterKind::OVERLOAD)
      .Case("dx.types.Handle", ParameterKind::DXIL_HANDLE)
      .Case("dx.types.CBufRet", ParameterKind::CBUFFER_RET)
      .Case("dx.types.ResRet", ParameterKind::RESOURCE_RET)
      .Default(ParameterKind::INVALID);
}

DXILParam::DXILParam(const Record *R) {
  Name = R->getValueAsString("name");
  Pos = R->getValueAsInt("pos");
  Kind = parameterTypeNameToKind(R->getValueAsString("llvm_type"));
  if (R->getValue("doc"))
    Doc = R->getValueAsString("doc");
  IsConst = R->getValueAsBit("is_const");
  EnumName = R->getValueAsString("enum_name");
  MaxValue = R->getValueAsInt("max_value");
}

static std::string parameterKindToString(ParameterKind Kind) {
  switch (Kind) {
  case ParameterKind::INVALID:
    return "INVALID";
  case ParameterKind::VOID:
    return "VOID";
  case ParameterKind::HALF:
    return "HALF";
  case ParameterKind::FLOAT:
    return "FLOAT";
  case ParameterKind::DOUBLE:
    return "DOUBLE";
  case ParameterKind::I1:
    return "I1";
  case ParameterKind::I8:
    return "I8";
  case ParameterKind::I16:
    return "I16";
  case ParameterKind::I32:
    return "I32";
  case ParameterKind::I64:
    return "I64";
  case ParameterKind::OVERLOAD:
    return "OVERLOAD";
  case ParameterKind::CBUFFER_RET:
    return "CBUFFER_RET";
  case ParameterKind::RESOURCE_RET:
    return "RESOURCE_RET";
  case ParameterKind::DXIL_HANDLE:
    return "DXIL_HANDLE";
  }
  llvm_unreachable("Unknown llvm::dxil::ParameterKind enum");
}

static void emitDXILOpEnum(DXILOperationData &DXILOp, raw_ostream &OS) {
  // Name = ID, // Doc
  OS << DXILOp.Name << " = " << DXILOp.DXILOpID << ", // " << DXILOp.Doc
     << "\n";
}

static std::string buildCategoryStr(StringSet<> &Cetegorys) {
  std::string Str;
  raw_string_ostream OS(Str);
  for (auto &It : Cetegorys) {
    OS << " " << It.getKey();
  }
  return OS.str();
}

// Emit enum declaration for DXIL.
static void emitDXILEnums(std::vector<DXILOperationData> &DXILOps,
                          raw_ostream &OS) {
  // Sort by Category + OpName.
  llvm::sort(DXILOps, [](DXILOperationData &A, DXILOperationData &B) {
    // Group by Category first.
    if (A.Category == B.Category)
      // Inside same Category, order by OpName.
      return A.DXILOp < B.DXILOp;
    else
      return A.Category < B.Category;
  });

  OS << "// Enumeration for operations specified by DXIL\n";
  OS << "enum class OpCode : unsigned {\n";

  StringMap<StringSet<>> ClassMap;
  StringRef PrevCategory = "";
  for (auto &DXILOp : DXILOps) {
    StringRef Category = DXILOp.Category;
    if (Category != PrevCategory) {
      OS << "\n// " << Category << "\n";
      PrevCategory = Category;
    }
    emitDXILOpEnum(DXILOp, OS);
    auto It = ClassMap.find(DXILOp.DXILClass);
    if (It != ClassMap.end()) {
      It->second.insert(DXILOp.Category);
    } else {
      ClassMap[DXILOp.DXILClass].insert(DXILOp.Category);
    }
  }

  OS << "\n};\n\n";

  std::vector<std::pair<std::string, std::string>> ClassVec;
  for (auto &It : ClassMap) {
    ClassVec.emplace_back(
        std::make_pair(It.getKey().str(), buildCategoryStr(It.second)));
  }
  // Sort by Category + ClassName.
  llvm::sort(ClassVec, [](std::pair<std::string, std::string> &A,
                          std::pair<std::string, std::string> &B) {
    StringRef ClassA = A.first;
    StringRef CategoryA = A.second;
    StringRef ClassB = B.first;
    StringRef CategoryB = B.second;
    // Group by Category first.
    if (CategoryA == CategoryB)
      // Inside same Category, order by ClassName.
      return ClassA < ClassB;
    else
      return CategoryA < CategoryB;
  });

  OS << "// Groups for DXIL operations with equivalent function templates\n";
  OS << "enum class OpCodeClass : unsigned {\n";
  PrevCategory = "";
  for (auto &It : ClassVec) {

    StringRef Category = It.second;
    if (Category != PrevCategory) {
      OS << "\n// " << Category << "\n";
      PrevCategory = Category;
    }
    StringRef Name = It.first;
    OS << Name << ",\n";
  }
  OS << "\n};\n\n";
}

// Emit map from llvm intrinsic to DXIL operation.
static void emitDXILIntrinsicMap(std::vector<DXILOperationData> &DXILOps,
                                 raw_ostream &OS) {
  OS << "\n";
  // FIXME: use array instead of SmallDenseMap.
  OS << "static const SmallDenseMap<Intrinsic::ID, dxil::OpCode> LowerMap = "
        "{\n";
  for (auto &DXILOp : DXILOps) {
    if (DXILOp.Intrinsic.empty())
      continue;
    // {Intrinsic::sin, dxil::OpCode::Sin},
    OS << "  { Intrinsic::" << DXILOp.Intrinsic
       << ", dxil::OpCode::" << DXILOp.DXILOp << "},\n";
  }
  OS << "};\n";
  OS << "\n";
}

static std::string emitDXILOperationFnAttr(StringRef FnAttr) {
  return StringSwitch<std::string>(FnAttr)
      .Case("rn", "Attribute::ReadNone")
      .Case("ro", "Attribute::ReadOnly")
      .Default("Attribute::None");
}

static std::string getOverloadKind(StringRef Overload) {
  return StringSwitch<std::string>(Overload)
      .Case("half", "OverloadKind::HALF")
      .Case("float", "OverloadKind::FLOAT")
      .Case("double", "OverloadKind::DOUBLE")
      .Case("i1", "OverloadKind::I1")
      .Case("i16", "OverloadKind::I16")
      .Case("i32", "OverloadKind::I32")
      .Case("i64", "OverloadKind::I64")
      .Case("udt", "OverloadKind::UserDefineType")
      .Case("obj", "OverloadKind::ObjectType")
      .Default("OverloadKind::VOID");
}

static std::string getDXILOperationOverload(StringRef Overloads) {
  SmallVector<StringRef> OverloadStrs;
  Overloads.split(OverloadStrs, ';', /*MaxSplit*/ -1, /*KeepEmpty*/ false);
  // Format is: OverloadKind::FLOAT | OverloadKind::HALF
  assert(!OverloadStrs.empty() && "Invalid overloads");
  auto It = OverloadStrs.begin();
  std::string Result;
  raw_string_ostream OS(Result);
  OS << getOverloadKind(*It);
  for (++It; It != OverloadStrs.end(); ++It) {
    OS << " | " << getOverloadKind(*It);
  }
  return OS.str();
}

static std::string lowerFirstLetter(StringRef Name) {
  if (Name.empty())
    return "";

  std::string LowerName = Name.str();
  LowerName[0] = llvm::toLower(Name[0]);
  return LowerName;
}

static std::string getDXILOpClassName(StringRef DXILOpClass) {
  // Lower first letter expect for special case.
  return StringSwitch<std::string>(DXILOpClass)
      .Case("CBufferLoad", "cbufferLoad")
      .Case("CBufferLoadLegacy", "cbufferLoadLegacy")
      .Case("GSInstanceID", "gsInstanceID")
      .Default(lowerFirstLetter(DXILOpClass));
}

static void emitDXILOperationTable(std::vector<DXILOperationData> &DXILOps,
                                   raw_ostream &OS) {
  // Sort by DXILOpID.
  llvm::sort(DXILOps, [](DXILOperationData &A, DXILOperationData &B) {
    return A.DXILOpID < B.DXILOpID;
  });

  // Collect Names.
  SequenceToOffsetTable<std::string> OpClassStrings;
  SequenceToOffsetTable<std::string> OpStrings;
  SequenceToOffsetTable<SmallVector<ParameterKind>> Parameters;

  StringMap<SmallVector<ParameterKind>> ParameterMap;
  StringSet<> ClassSet;
  for (auto &DXILOp : DXILOps) {
    OpStrings.add(DXILOp.DXILOp.str());

    if (ClassSet.contains(DXILOp.DXILClass))
      continue;
    ClassSet.insert(DXILOp.DXILClass);
    OpClassStrings.add(getDXILOpClassName(DXILOp.DXILClass));
    SmallVector<ParameterKind> ParamKindVec;
    for (auto &Param : DXILOp.Params) {
      ParamKindVec.emplace_back(Param.Kind);
    }
    ParameterMap[DXILOp.DXILClass] = ParamKindVec;
    Parameters.add(ParamKindVec);
  }

  // Layout names.
  OpStrings.layout();
  OpClassStrings.layout();
  Parameters.layout();

  // Emit the DXIL operation table.
  //{dxil::OpCode::Sin, OpCodeNameIndex, OpCodeClass::Unary,
  // OpCodeClassNameIndex,
  // OverloadKind::FLOAT | OverloadKind::HALF, Attribute::AttrKind::ReadNone, 0,
  // 3, ParameterTableOffset},
  OS << "static const OpCodeProperty *getOpCodeProperty(dxil::OpCode DXILOp) "
        "{\n";

  OS << "  static const OpCodeProperty OpCodeProps[] = {\n";
  for (auto &DXILOp : DXILOps) {
    OS << "  { dxil::OpCode::" << DXILOp.DXILOp << ", "
       << OpStrings.get(DXILOp.DXILOp.str())
       << ", OpCodeClass::" << DXILOp.DXILClass << ", "
       << OpClassStrings.get(getDXILOpClassName(DXILOp.DXILClass)) << ", "
       << getDXILOperationOverload(DXILOp.OverloadTypes) << ", "
       << emitDXILOperationFnAttr(DXILOp.FnAttr) << ", "
       << DXILOp.OverloadParamIndex << ", " << DXILOp.Params.size() << ", "
       << Parameters.get(ParameterMap[DXILOp.DXILClass]) << " },\n";
  }
  OS << "  };\n";

  OS << "  // FIXME: change search to indexing with\n";
  OS << "  // DXILOp once all DXIL op is added.\n";
  OS << "  OpCodeProperty TmpProp;\n";
  OS << "  TmpProp.OpCode = DXILOp;\n";
  OS << "  const OpCodeProperty *Prop =\n";
  OS << "      llvm::lower_bound(OpCodeProps, TmpProp,\n";
  OS << "                        [](const OpCodeProperty &A, const "
        "OpCodeProperty &B) {\n";
  OS << "                          return A.OpCode < B.OpCode;\n";
  OS << "                        });\n";
  OS << "  assert(Prop && \"fail to find OpCodeProperty\");\n";
  OS << "  return Prop;\n";
  OS << "}\n\n";

  // Emit the string tables.
  OS << "static const char *getOpCodeName(dxil::OpCode DXILOp) {\n\n";

  OpStrings.emitStringLiteralDef(OS,
                                 "  static const char DXILOpCodeNameTable[]");

  OS << "  auto *Prop = getOpCodeProperty(DXILOp);\n";
  OS << "  unsigned Index = Prop->OpCodeNameOffset;\n";
  OS << "  return DXILOpCodeNameTable + Index;\n";
  OS << "}\n\n";

  OS << "static const char *getOpCodeClassName(const OpCodeProperty &Prop) "
        "{\n\n";

  OpClassStrings.emitStringLiteralDef(
      OS, "  static const char DXILOpCodeClassNameTable[]");

  OS << "  unsigned Index = Prop.OpCodeClassNameOffset;\n";
  OS << "  return DXILOpCodeClassNameTable + Index;\n";
  OS << "}\n ";

  OS << "static const ParameterKind *getOpCodeParameterKind(const "
        "OpCodeProperty &Prop) "
        "{\n\n";
  OS << "  static const ParameterKind DXILOpParameterKindTable[] = {\n";
  Parameters.emit(
      OS,
      [](raw_ostream &ParamOS, ParameterKind Kind) {
        ParamOS << "ParameterKind::" << parameterKindToString(Kind);
      },
      "ParameterKind::INVALID");
  OS << "  };\n\n";
  OS << "  unsigned Index = Prop.ParameterTableOffset;\n";
  OS << "  return DXILOpParameterKindTable + Index;\n";
  OS << "}\n ";
}

static void EmitDXILOperation(RecordKeeper &Records, raw_ostream &OS) {
  std::vector<Record *> Ops = Records.getAllDerivedDefinitions("dxil_op");
  OS << "// Generated code, do not edit.\n";
  OS << "\n";

  std::vector<DXILOperationData> DXILOps;
  DXILOps.reserve(Ops.size());
  for (auto *Record : Ops) {
    DXILOps.emplace_back(DXILOperationData(Record));
  }

  OS << "#ifdef DXIL_OP_ENUM\n";
  emitDXILEnums(DXILOps, OS);
  OS << "#endif\n\n";

  OS << "#ifdef DXIL_OP_INTRINSIC_MAP\n";
  emitDXILIntrinsicMap(DXILOps, OS);
  OS << "#endif\n\n";

  OS << "#ifdef DXIL_OP_OPERATION_TABLE\n";
  emitDXILOperationTable(DXILOps, OS);
  OS << "#endif\n\n";

  OS << "\n";
}

static TableGen::Emitter::Opt X("gen-dxil-operation", EmitDXILOperation,
                                "Generate DXIL operation information");
