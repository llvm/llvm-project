//===- offload-tblgen/APIGen.cpp - Tablegen backend for Offload header ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that produces the contents of the Offload API
// header. The generated comments are Doxygen compatible.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "GenCommon.hpp"
#include "RecordTypes.hpp"

using namespace llvm;
using namespace offload::tblgen;

// Produce a possibly multi-line comment from the input string
static std::string MakeComment(StringRef in) {
  std::string out = "";
  size_t LineStart = 0;
  size_t LineBreak = 0;
  while (LineBreak < in.size()) {
    LineBreak = in.find_first_of("\n", LineStart);
    if (LineBreak - LineStart <= 1) {
      break;
    }
    out += std::string("/// ") +
           in.substr(LineStart, LineBreak - LineStart).str() + "\n";
    if (LineBreak != std::string::npos)
      LineStart = LineBreak + 1;
  }

  return out;
}

static void ProcessHandle(const HandleRec &H, raw_ostream &OS) {
  if (!H.getName().ends_with("_handle_t")) {
    errs() << "Handle type name (" << H.getName()
           << ") must end with '_handle_t'!\n";
    exit(1);
  }

  auto ImplName = H.getName().substr(0, H.getName().size() - 9) + "_impl_t";
  OS << CommentsHeader;
  OS << formatv("/// @brief {0}\n", H.getDesc());
  OS << formatv("typedef struct {0} *{1};\n", ImplName, H.getName());
}

static void ProcessTypedef(const TypedefRec &T, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("/// @brief {0}\n", T.getDesc());
  OS << formatv("typedef {0} {1};\n", T.getValue(), T.getName());
}

static void ProcessMacro(const MacroRec &M, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("#ifndef {0}\n", M.getName());
  if (auto Condition = M.getCondition()) {
    OS << formatv("#if {0}\n", *Condition);
  }
  OS << "/// @brief " << M.getDesc() << "\n";
  OS << formatv("#define {0} {1}\n", M.getNameWithArgs(), M.getValue());
  if (auto AltValue = M.getAltValue()) {
    OS << "#else\n";
    OS << formatv("#define {0} {1}\n", M.getNameWithArgs(), *AltValue);
  }
  if (auto Condition = M.getCondition()) {
    OS << formatv("#endif // {0}\n", *Condition);
  }
  OS << formatv("#endif // {0}\n", M.getName());
}

static void ProcessFunction(const FunctionRec &F, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("/// @brief {0}\n", F.getDesc());
  OS << CommentsBreak;

  OS << "/// @details\n";
  for (auto &Detail : F.getDetails()) {
    OS << formatv("///    - {0}\n", Detail);
  }
  OS << CommentsBreak;

  // Emit analogue remarks
  auto Analogues = F.getAnalogues();
  if (!Analogues.empty()) {
    OS << "/// @remarks\n///  _Analogues_\n";
    for (auto &Analogue : Analogues) {
      OS << formatv("///    - **{0}**\n", Analogue);
    }
    OS << CommentsBreak;
  }

  OS << "/// @returns\n";
  auto Returns = F.getReturns();
  for (auto &Ret : Returns) {
    OS << formatv("///     - ::{0}\n", Ret.getValue());
    auto RetConditions = Ret.getConditions();
    for (auto &RetCondition : RetConditions) {
      OS << formatv("///         + {0}\n", RetCondition);
    }
  }

  OS << formatv("{0}_APIEXPORT {1}_result_t {0}_APICALL ", PrefixUpper,
                PrefixLower);
  OS << F.getName();
  OS << "(\n";
  auto Params = F.getParams();
  for (auto &Param : Params) {
    OS << MakeParamComment(Param) << "\n";
    OS << "  " << Param.getType() << " " << Param.getName();
    if (Param != Params.back()) {
      OS << ",\n";
    } else {
      OS << "\n";
    }
  }
  OS << ");\n\n";
}

static void ProcessEnum(const EnumRec &Enum, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("/// @brief {0}\n", Enum.getDesc());
  OS << formatv("typedef enum {0} {{\n", Enum.getName());

  uint32_t EtorVal = 0;
  for (const auto &EnumVal : Enum.getValues()) {
    if (Enum.isTyped()) {
      OS << MakeComment(
          formatv("[{0}] {1}", EnumVal.getTaggedType(), EnumVal.getDesc())
              .str());
    } else {
      OS << MakeComment(EnumVal.getDesc());
    }
    OS << formatv(TAB_1 "{0}_{1} = {2},\n", Enum.getEnumValNamePrefix(),
                  EnumVal.getName(), EtorVal++);
  }

  // Add force uint32 val
  OS << formatv(TAB_1 "/// @cond\n" TAB_1
                      "{0}_FORCE_UINT32 = 0x7fffffff\n" TAB_1
                      "/// @endcond\n\n",
                Enum.getEnumValNamePrefix());

  OS << formatv("} {0};\n", Enum.getName());
}

static void ProcessStruct(const StructRec &Struct, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("/// @brief {0}\n", Struct.getDesc());
  OS << formatv("typedef struct {0} {{\n", Struct.getName());

  for (const auto &Member : Struct.getMembers()) {
    OS << formatv(TAB_1 "{0} {1}; {2}", Member.getType(), Member.getName(),
                  MakeComment(Member.getDesc()));
  }

  OS << formatv("} {0};\n\n", Struct.getName());
}

static void ProcessFptrTypedef(const FptrTypedefRec &F, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("/// @brief {0}\n", F.getDesc());
  OS << formatv("typedef {0} (*{1})(", F.getReturn(), F.getName());
  for (const auto &Param : F.getParams()) {
    OS << formatv("\n  // {0}\n  {1} {2}", Param.getDesc(), Param.getType(),
                  Param.getName());
    if (Param != F.getParams().back())
      OS << ",";
  }
  OS << ");\n";
}

static void ProcessFuncParamStruct(const FunctionRec &Func, raw_ostream &OS) {
  if (Func.getParams().size() == 0) {
    return;
  }

  auto FuncParamStructBegin = R"(
///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for {0}
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct {1} {{
)";

  OS << formatv(FuncParamStructBegin, Func.getName(),
                Func.getParamStructName());
  for (const auto &Param : Func.getParams()) {
    OS << TAB_1 << Param.getType() << "* p" << Param.getName() << ";\n";
  }
  OS << formatv("} {0};\n", Func.getParamStructName());
}

static void ProcessFuncWithCodeLocVariant(const FunctionRec &Func,
                                          raw_ostream &OS) {

  auto FuncWithCodeLocBegin = R"(
///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of {0} that also sets source code location information
/// @details See also ::{0}
OL_APIEXPORT ol_result_t OL_APICALL {0}WithCodeLoc(
)";
  OS << formatv(FuncWithCodeLocBegin, Func.getName());
  auto Params = Func.getParams();
  for (auto &Param : Params) {
    OS << "  " << Param.getType() << " " << Param.getName();
    OS << ",\n";
  }
  OS << "ol_code_location_t *CodeLocation);\n\n";
}

void EmitOffloadAPI(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  OS << FileHeader;
  // Generate main API definitions
  for (auto *R : Records.getAllDerivedDefinitions("APIObject")) {
    if (R->isSubClassOf("Macro")) {
      ProcessMacro(MacroRec{R}, OS);
    } else if (R->isSubClassOf("Typedef")) {
      ProcessTypedef(TypedefRec{R}, OS);
    } else if (R->isSubClassOf("Handle")) {
      ProcessHandle(HandleRec{R}, OS);
    } else if (R->isSubClassOf("Function")) {
      ProcessFunction(FunctionRec{R}, OS);
    } else if (R->isSubClassOf("Enum")) {
      ProcessEnum(EnumRec{R}, OS);
    } else if (R->isSubClassOf("Struct")) {
      ProcessStruct(StructRec{R}, OS);
    } else if (R->isSubClassOf("FptrTypedef")) {
      ProcessFptrTypedef(FptrTypedefRec{R}, OS);
    }
  }

  // Generate auxiliary definitions (func param structs etc)
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    ProcessFuncParamStruct(FunctionRec{R}, OS);
  }

  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    ProcessFuncWithCodeLocVariant(FunctionRec{R}, OS);
  }

  OS << FileFooter;
}
